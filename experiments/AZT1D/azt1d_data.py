"""Load AZT1D CGM / pump CSVs into a regular continuous–discrete series.

Expects the Mendeley release under::

    experiments/AZT1D/AZT1D 2025/CGM Records/Subject {k}/Subject {k}.csv

Builds a uniform ``dt_min``-minute grid with CGM observations and known
insulin / carbohydrate input rates for the glucose–insulin SDE.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = HERE / "AZT1D 2025" / "CGM Records"

CGM_COL_ALIASES = ("CGM", "Readings (CGM / BGM)")


@dataclass(frozen=True)
class Azt1dSeries:
    subject: int
    t_min: np.ndarray  # (T,) minutes from start
    cgm: np.ndarray  # (T,) mg/dL
    insulin_u_per_min: np.ndarray  # (T,) U/min (basal + spread bolus)
    carb_g_per_min: np.ndarray  # (T,) g/min appearance rate
    dt_min: float
    n_obs: int
    span_days: float


def _data_root(root: Path | str | None) -> Path:
    p = Path(root) if root is not None else DEFAULT_DATA_ROOT
    if not p.is_dir():
        raise FileNotFoundError(
            f"AZT1D CGM Records not found at {p}. "
            "Place the Mendeley 'AZT1D 2025' folder under experiments/AZT1D/."
        )
    return p


def subject_csv_path(subject: int, *, root: Path | str | None = None) -> Path:
    root = _data_root(root)
    path = root / f"Subject {subject}" / f"Subject {subject}.csv"
    if not path.is_file():
        raise FileNotFoundError(f"missing subject CSV: {path}")
    return path


def list_subjects(*, root: Path | str | None = None) -> list[int]:
    root = _data_root(root)
    out: list[int] = []
    for d in sorted(root.glob("Subject *")):
        if not d.is_dir():
            continue
        try:
            out.append(int(d.name.split()[-1]))
        except ValueError:
            continue
    return out


def _parse_float(val: str | None) -> float:
    if val is None:
        return np.nan
    s = str(val).strip()
    if not s:
        return np.nan
    try:
        return float(s)
    except ValueError:
        return np.nan


def _parse_time(val: str) -> datetime:
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%m/%d/%Y %H:%M:%S", "%m/%d/%Y %H:%M"):
        try:
            return datetime.strptime(val.strip(), fmt)
        except ValueError:
            continue
    return datetime.fromisoformat(val.strip())


def _ffill_1d(x: np.ndarray, fill: float = 1.0) -> np.ndarray:
    out = x.astype(np.float64, copy=True)
    last = np.nan
    for i in range(out.size):
        if np.isfinite(out[i]):
            last = out[i]
        elif np.isfinite(last):
            out[i] = last
    # back-fill leading NaNs
    first = fill
    for i in range(out.size):
        if np.isfinite(out[i]):
            first = out[i]
            break
    for i in range(out.size):
        if np.isfinite(out[i]):
            break
        out[i] = first
    out[~np.isfinite(out)] = fill
    return out


def load_subject_arrays(subject: int, *, root: Path | str | None = None) -> dict[str, np.ndarray]:
    path = subject_csv_path(subject, root=root)
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"empty CSV: {path}")
        fields = list(reader.fieldnames)
        cgm_key = next((a for a in CGM_COL_ALIASES if a in fields), None)
        if cgm_key is None:
            raise ValueError(f"no CGM column in {fields}")

        times: list[datetime] = []
        cgm: list[float] = []
        basal: list[float] = []
        bolus: list[float] = []
        carbs: list[float] = []
        for row in reader:
            times.append(_parse_time(row["EventDateTime"]))
            cgm.append(_parse_float(row.get(cgm_key)))
            basal.append(_parse_float(row.get("Basal")))
            bolus.append(_parse_float(row.get("TotalBolusInsulinDelivered")))
            carbs.append(_parse_float(row.get("CarbSize")))

    order = np.argsort(np.asarray(times, dtype=object))
    t0 = times[int(order[0])]
    t_min = np.array([(times[int(i)] - t0).total_seconds() / 60.0 for i in order], dtype=np.float64)
    return {
        "t_min": t_min,
        "cgm": np.asarray(cgm, dtype=np.float64)[order],
        "basal": np.asarray(basal, dtype=np.float64)[order],
        "bolus": np.asarray(bolus, dtype=np.float64)[order],
        "carbs": np.asarray(carbs, dtype=np.float64)[order],
    }


def load_azt1d_series(
    subject: int = 1,
    *,
    max_obs: int | None = 288,
    dt_min: float = 5.0,
    bolus_spread_min: float = 20.0,
    carb_spread_min: float = 60.0,
    basal_max_u_per_h: float = 10.0,
    root: Path | str | None = None,
) -> Azt1dSeries:
    """Resample one subject onto a regular grid for CD filtering.

    Parameters
    ----------
    max_obs:
        Cap on number of CGM samples (``None`` = full record). Default 288 ≈ 1 day.
    bolus_spread_min / carb_spread_min:
        Rectangular windows used to turn impulsive bolus / carb logs into rates.
    basal_max_u_per_h:
        Clip obviously corrupt basal rates (some rows exceed clinical range).
    """
    raw = load_subject_arrays(subject, root=root)
    t_raw = raw["t_min"]
    cgm_raw = raw["cgm"]

    ok = np.isfinite(cgm_raw) & (cgm_raw > 20.0) & (cgm_raw < 500.0)
    t_cgm = t_raw[ok]
    cgm = cgm_raw[ok]
    if t_cgm.size < 20:
        raise ValueError(f"subject {subject}: too few CGM samples ({t_cgm.size})")

    t_start = float(t_cgm[0])
    t_end = float(t_cgm[-1])
    n_full = int(np.floor((t_end - t_start) / dt_min)) + 1
    if max_obs is not None:
        n_full = min(n_full, int(max_obs))
    t_grid = t_start + dt_min * np.arange(n_full, dtype=np.float64)
    t_hi = float(t_grid[-1])

    cgm_grid = np.full(n_full, np.nan, dtype=np.float64)
    j = 0
    for i, tg in enumerate(t_grid):
        while j + 1 < t_cgm.size and abs(t_cgm[j + 1] - tg) <= abs(t_cgm[j] - tg):
            j += 1
        if abs(t_cgm[j] - tg) <= 0.5 * dt_min + 1e-9:
            cgm_grid[i] = cgm[j]
    cgm_grid = _ffill_1d(cgm_grid, fill=float(cgm[0]))

    basal_u_h = np.clip(raw["basal"], 0.0, basal_max_u_per_h)
    basal_ff = _ffill_1d(basal_u_h, fill=1.0)
    basal_on_grid = np.interp(t_grid, t_raw, basal_ff) / 60.0

    insulin = basal_on_grid.copy()
    half = 0.5 * bolus_spread_min
    rate = 1.0 / max(bolus_spread_min, 1e-6)
    for ti, bi in zip(t_raw, raw["bolus"]):
        if not (np.isfinite(bi) and bi > 0.0):
            continue
        if ti < t_start - half or ti > t_hi + half:
            continue
        mask = (t_grid >= ti - half) & (t_grid <= ti + half)
        insulin[mask] += bi * rate

    carb_rate = np.zeros(n_full, dtype=np.float64)
    half_c = 0.5 * carb_spread_min
    rate_c = 1.0 / max(carb_spread_min, 1e-6)
    for ti, ci in zip(t_raw, raw["carbs"]):
        if not (np.isfinite(ci) and ci > 0.0):
            continue
        if ti < t_start - half_c or ti > t_hi + half_c:
            continue
        mask = (t_grid >= ti - half_c) & (t_grid <= ti + half_c)
        carb_rate[mask] += ci * rate_c

    span_days = (t_hi - t_start) / (60.0 * 24.0)
    return Azt1dSeries(
        subject=subject,
        t_min=t_grid - t_start,
        cgm=cgm_grid,
        insulin_u_per_min=insulin,
        carb_g_per_min=carb_rate,
        dt_min=float(dt_min),
        n_obs=n_full,
        span_days=float(span_days),
    )


def inject_cgm_gaussian_noise(
    cgm: np.ndarray,
    *,
    std: float,
    seed: int,
) -> np.ndarray:
    """Add i.i.d. N(0, std²) noise to CGM; ``std<=0`` returns a float64 copy.

    Use the same ``seed`` (and same-length ``cgm``) across scripts so
    recalibration and bakeoff/tracks share one noisy realization.
    """
    x = np.asarray(cgm, dtype=np.float64)
    if float(std) <= 0.0:
        return x.copy()
    rng = np.random.default_rng(int(seed))
    return x + rng.normal(0.0, float(std), size=x.shape)


def cgm_noise_seed_for_subject(base_seed: int, subject: int) -> int:
    """Deterministic per-subject seed for ``inject_cgm_gaussian_noise``."""
    return int(base_seed) + 100_003 * int(subject)
