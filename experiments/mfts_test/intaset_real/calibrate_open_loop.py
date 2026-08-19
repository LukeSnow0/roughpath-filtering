"""Open-loop least-squares calibration of InTaSet Johansson parameters.

Fits ``(k_pump, γ1, γ2, a_out)`` on an early segment of ``4T-M2-train`` with
fixed tank areas ``A`` (scale gauge). Estimates sensor bias from the first
quiet samples, then ``meas_std`` / ``sigma_u`` from simulation residuals.

Hold out later minutes for the Pareto experiment (``--holdout-start-min``).

Run:
  uv run --extra experiments python \\
      experiments/mfts_test/intaset_real/calibrate_open_loop.py

Writes:
  ``output/calibrated_params.json``
  ``output/calibration_fit.png``
"""

from __future__ import annotations

import argparse
import os
import sys

os.environ.setdefault(
    "MPLCONFIGDIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), ".mplcache")
)

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
MFTS_DIR = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, MFTS_DIR)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

from intaset_io import load_named, window_trajectory
from intaset_model import (
    IntaSetParams,
    mape_pct,
    save_params,
    simulate_open_loop,
)
from plot_style import (
    DOUBLE_COLUMN_WIDTH,
    PALETTE,
    save_publication_figure,
    style_curve_axis,
)

OUTPUT_DIR = os.path.join(HERE, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

_parser = argparse.ArgumentParser(description="Open-loop InTaSet parameter calibration")
_parser.add_argument("--source", type=str, default="4T-M2-train")
_parser.add_argument(
    "--fit-min",
    type=float,
    default=15.0,
    help="Minutes of data used for calibration (from t=0)",
)
_parser.add_argument(
    "--holdout-start-min",
    type=float,
    default=15.0,
    help="Recorded in JSON meta for Pareto (evaluation starts here)",
)
_parser.add_argument(
    "--dt-fit",
    type=float,
    default=0.5,
    help="Resample step [s] for the LS objective (default 0.5)",
)
_parser.add_argument(
    "--bias-seconds",
    type=float,
    default=5.0,
    help="Initial seconds used to estimate sensor offset",
)
_parser.add_argument(
    "--out",
    type=str,
    default=os.path.join(OUTPUT_DIR, "calibrated_params.json"),
)
_args = _parser.parse_args()


def _logit(p: float, eps: float = 1e-6) -> float:
    p = float(np.clip(p, eps, 1.0 - eps))
    return float(np.log(p / (1.0 - p)))


def _sigmoid(z: float) -> float:
    return float(1.0 / (1.0 + np.exp(-z)))


def _pack(k_pump: float, g1: float, g2: float, a_out: np.ndarray) -> np.ndarray:
    return np.array(
        [
            np.log(max(k_pump, 1e-3)),
            _logit(g1),
            _logit(g2),
            *np.log(np.maximum(a_out, 1e-6)),
        ],
        dtype=np.float64,
    )


def _unpack(z: np.ndarray, A: tuple[float, ...]) -> IntaSetParams:
    k_pump = float(np.exp(z[0]))
    g1 = _sigmoid(z[1])
    g2 = _sigmoid(z[2])
    a_out = tuple(float(x) for x in np.exp(z[3:7]))
    return IntaSetParams(
        gamma1=g1,
        gamma2=g2,
        A=tuple(float(x) for x in A),
        a_out=a_out,
        k_pump=k_pump,
    )


def _resample(t, H, U, dt: float):
    t0, t1 = float(t[0]), float(t[-1])
    t_r = np.arange(t0, t1 + 0.5 * dt, dt, dtype=np.float64)
    if t_r[-1] > t1:
        t_r = t_r[t_r <= t1 + 1e-12]
    H_r = np.column_stack([np.interp(t_r, t, H[:, i]) for i in range(4)])
    U_r = np.column_stack([np.interp(t_r, t, U[:, i]) for i in range(2)])
    return t_r, H_r, U_r


def main():
    traj_full = load_named(_args.source)
    fit_sec = float(_args.fit_min) * 60.0
    traj = window_trajectory(traj_full, start_sec=0.0, duration_sec=fit_sec, window_id=0)

    # Sensor bias: only remove negative empty-tank offsets.
    n_bias = max(2, int(round(_args.bias_seconds / max(traj.dt_sec, 1e-3))))
    bias = np.minimum(0.0, np.median(traj.heights[:n_bias], axis=0))

    H_raw = traj.heights
    H = H_raw - bias
    U = traj.pumps
    t = traj.t_sec - traj.t_sec[0]

    t_r, H_r, U_r = _resample(t, H, U, _args.dt_fit)
    h0 = np.maximum(H_r[0], 1e-3)

    A = IntaSetParams().A
    z0 = _pack(120.0, 0.45, 0.40, np.array(IntaSetParams().a_out))

    def loss(z: np.ndarray) -> float:
        p = _unpack(z, A)
        hat = simulate_open_loop(p, t_r, U_r, h0)
        # Relative + absolute mix keeps low-level tanks from dominating.
        err = hat - H_r
        rel = err / np.maximum(np.abs(H_r), 0.5)
        return float(np.mean(err**2) + 0.25 * np.mean(rel**2))

    print(
        f"Calibrating on {_args.source} [0, {_args.fit_min:g}] min "
        f"(dt_fit={_args.dt_fit:g}s, n={t_r.shape[0]})"
    )
    print(f"  height_bias ≈ {bias}")
    print(f"  initial loss={loss(z0):.4f}")

    res = minimize(
        loss,
        z0,
        method="L-BFGS-B",
        options={"maxiter": 80, "ftol": 1e-10, "disp": False},
    )
    p_fit = _unpack(res.x, A)
    print(f"  final loss={float(res.fun):.4f}  success={res.success}  nit={res.nit}")

    # Residuals on the fit grid → noise scales.
    hat_r = simulate_open_loop(p_fit, t_r, U_r, h0)
    resid = H_r - hat_r
    # Sensor noise ≈ high-frequency increments (model bias is low-frequency).
    dh = np.diff(H_r, axis=0)
    meas_std = float(np.std(dh) / np.sqrt(2.0))
    # Process noise from open-loop residual energy (as pump-flow equivalent).
    rmse = float(np.sqrt(np.mean(resid**2)))
    # Convert ~cm level error over a characteristic time (~30s) into flow noise.
    A_mean = float(np.mean(p_fit.A))
    sigma_u = float(np.clip(A_mean * rmse / 30.0, 5.0, 0.4 * p_fit.k_pump))

    p_fit = IntaSetParams(
        gamma1=p_fit.gamma1,
        gamma2=p_fit.gamma2,
        A=p_fit.A,
        a_out=p_fit.a_out,
        k_pump=p_fit.k_pump,
        meas_std=max(meas_std, 0.05),
        height_bias=tuple(float(x) for x in bias),
        sigma_u=sigma_u,
    )

    # Native-rate diagnostics on fit segment + holdout segment.
    def _eval_segment(start_min: float, dur_min: float, label: str):
        seg = window_trajectory(
            traj_full,
            start_sec=start_min * 60.0,
            duration_sec=dur_min * 60.0,
            window_id=0,
        )
        Hs = seg.heights - np.asarray(p_fit.height_bias)
        ts = seg.t_sec - seg.t_sec[0]
        hat = simulate_open_loop(p_fit, ts, seg.pumps, np.maximum(Hs[0], 1e-3))
        mape = mape_pct(hat, Hs)
        rmse = float(np.sqrt(np.mean((hat - Hs) ** 2)))
        print(f"  {label}: open-loop MAPE={mape:.2f}%  RMSE={rmse:.3f} cm")
        return ts, Hs, hat, mape, rmse

    t_fit, H_fit, hat_fit, mape_fit, rmse_fit = _eval_segment(
        0.0, _args.fit_min, "fit"
    )
    hold_start = float(_args.holdout_start_min)
    hold_dur = min(10.0, (traj_full.duration_sec / 60.0) - hold_start)
    mape_hold = rmse_hold = float("nan")
    H_hold = hat_hold = None
    t_hold = None
    if hold_dur >= 2.0:
        t_hold, H_hold, hat_hold, mape_hold, rmse_hold = _eval_segment(
            hold_start, hold_dur, f"holdout[{hold_start:g},{hold_start + hold_dur:g}]"
        )

    meta = {
        "source": _args.source,
        "fit_min": float(_args.fit_min),
        "holdout_start_min": hold_start,
        "dt_fit": float(_args.dt_fit),
        "mape_fit_pct": mape_fit,
        "rmse_fit_cm": rmse_fit,
        "mape_holdout_pct": mape_hold,
        "rmse_holdout_cm": rmse_hold,
        "optimizer": "L-BFGS-B",
        "success": bool(res.success),
        "nit": int(res.nit),
    }
    save_params(_args.out, p_fit, meta=meta)
    print(f"Wrote {_args.out}")
    print(
        f"  k_pump={p_fit.k_pump:.3g}  γ=({p_fit.gamma1:.3f},{p_fit.gamma2:.3f})  "
        f"a={tuple(round(x, 4) for x in p_fit.a_out)}"
    )
    print(f"  meas_std={p_fit.meas_std:.3f} cm  sigma_u={p_fit.sigma_u:.3g}")

    # Plot fit (+ holdout if available).
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(DOUBLE_COLUMN_WIDTH, 4.15),
        sharex=True,
    )
    for i, ax in enumerate(axes.ravel()):
        ax.plot(
            t_fit / 60.0,
            H_fit[:, i],
            color=PALETTE["neutral_gray"],
            linewidth=0.85,
            alpha=0.75,
            label="Measured",
            zorder=1,
        )
        ax.plot(
            t_fit / 60.0,
            hat_fit[:, i],
            color=PALETTE["teal_green"],
            linewidth=1.55,
            label="Fitted simulation",
            zorder=3,
        )
        if H_hold is not None and t_hold is not None:
            hold_times = hold_start + t_hold / 60.0
            ax.plot(
                hold_times,
                H_hold[:, i],
                color=PALETTE["neutral_gray"],
                linewidth=0.85,
                alpha=0.75,
                zorder=1,
            )
            ax.plot(
                hold_times,
                hat_hold[:, i],
                color=PALETTE["warm_red"],
                linewidth=1.55,
                label="Holdout simulation",
                zorder=3,
            )
            ax.axvline(
                hold_start,
                color=PALETTE["black"],
                linestyle=":",
                linewidth=0.8,
                zorder=0,
            )
        ax.set_title(f"Tank {i + 1}")
        ax.set_ylabel(rf"$h_{i + 1}$ [cm]")
        style_curve_axis(ax, grid_axis="y")
        if i >= 2:
            ax.set_xlabel("Time [min]")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=len(labels),
    )
    fig.subplots_adjust(
        left=0.082,
        right=0.992,
        bottom=0.115,
        top=0.875,
        wspace=0.18,
        hspace=0.2,
    )
    plot_path = os.path.join(OUTPUT_DIR, "calibration_fit.png")
    save_publication_figure(fig, plot_path)
    plt.close(fig)
    print(f"Wrote {plot_path}")


if __name__ == "__main__":
    main()
