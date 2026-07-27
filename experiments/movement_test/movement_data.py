"""Loader for the Dryad hourly dabbling-duck GPS dataset.

Osborne, D., Askren, R., Phelps, S., et al. (2024). "Hourly GPS location data
from mallards, green-winged teal, and American wigeon transmittered in Arkansas."
Dryad. https://doi.org/10.5061/dryad.1ns1rn93p  (CC0)

The raw CSV ships projected UTM coordinates (``Final_X``, ``Final_Y`` in metres,
UTM 15N) and an hourly-rounded timestamp (``round_cst``). We pick one individual,
extract its longest run of *consecutive, exactly-hourly* fixes, and return the
positions in kilometres, mean-centred, with time measured in hours. Velocity is
latent (unobserved); only position is measured, with GPS noise.

Download (once):
    See ``experiments/movement_test/data/README.md``; the loader expects
    ``experiments/movement_test/data/hourly_movement.csv``.
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from datetime import datetime

import numpy as np

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
DEFAULT_CSV = os.path.join(DATA_DIR, "hourly_movement.csv")


@dataclass(frozen=True)
class DuckTrack:
    band: str
    species: str
    sex: str
    t_hours: np.ndarray  # (N,) hours since first fix (0, 1, 2, ...)
    xy_km: np.ndarray  # (N, 2) mean-centred position in km (UTM-derived)
    center_km: np.ndarray  # (2,) subtracted centroid (km) in original UTM frame
    dt_hours: float  # nominal sampling interval (1.0 h)


def _parse_dt(s: str) -> datetime | None:
    for fmt in ("%m/%d/%y %H:%M", "%m/%d/%Y %H:%M", "%m/%d/%y %H:%M:%S"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    return None


def _longest_hourly_run(times: list[datetime]) -> tuple[int, int]:
    """Return (start, stop) indices of the longest exactly-1h-spaced contiguous run."""
    best_start, best_len = 0, 1
    cur_start, cur_len = 0, 1
    for i in range(1, len(times)):
        step_h = round((times[i] - times[i - 1]).total_seconds() / 3600.0)
        if step_h == 1:
            cur_len += 1
        else:
            if cur_len > best_len:
                best_start, best_len = cur_start, cur_len
            cur_start, cur_len = i, 1
    if cur_len > best_len:
        best_start, best_len = cur_start, cur_len
    return best_start, best_start + best_len


def load_duck_track(
    band: str = "MALL 825 M",
    *,
    csv_path: str = DEFAULT_CSV,
    max_obs: int | None = None,
) -> DuckTrack:
    """Load one individual's longest regularly-sampled hourly track.

    Args:
        band: USGS band id selecting the individual (default: the mallard with the
            longest contiguous hourly run in the dataset).
        csv_path: Path to ``hourly_movement.csv``.
        max_obs: Optionally truncate to the first ``max_obs`` fixes of the run.

    Returns:
        A :class:`DuckTrack` with mean-centred km positions and hourly times.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"{csv_path} not found. See experiments/movement_test/data/README.md "
            "for the Dryad download (doi:10.5061/dryad.1ns1rn93p)."
        )

    rows: list[list[str]] = []
    with open(csv_path, newline="") as f:
        r = csv.reader(f)
        hdr = next(r)
        idx = {c: i for i, c in enumerate(hdr)}
        for row in r:
            if row[idx["band"]] == band:
                rows.append(row)
    if not rows:
        raise ValueError(f"No rows for band={band!r} in {csv_path}.")

    parsed = []
    for row in rows:
        t = _parse_dt(row[idx["round_cst"]])
        if t is None:
            continue
        try:
            fx = float(row[idx["Final_X"]])
            fy = float(row[idx["Final_Y"]])
        except ValueError:
            continue
        parsed.append((t, fx, fy, row[idx["spp"]], row[idx["sex"]]))
    parsed.sort(key=lambda p: p[0])

    times = [p[0] for p in parsed]
    start, stop = _longest_hourly_run(times)
    run = parsed[start:stop]
    if max_obs is not None:
        run = run[:max_obs]

    t0 = run[0][0]
    t_hours = np.array([(p[0] - t0).total_seconds() / 3600.0 for p in run])
    xy_m = np.array([[p[1], p[2]] for p in run], dtype=np.float64)
    xy_km = xy_m / 1000.0
    center = xy_km.mean(axis=0)
    xy_km = xy_km - center

    return DuckTrack(
        band=band,
        species=run[0][3],
        sex=run[0][4],
        t_hours=t_hours,
        xy_km=xy_km,
        center_km=center,
        dt_hours=1.0,
    )
