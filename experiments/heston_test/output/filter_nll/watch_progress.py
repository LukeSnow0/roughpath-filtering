#!/usr/bin/env python3
"""Daemon: refresh PROGRESS.txt from the filter_nll run log every 15s."""
from __future__ import annotations

import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
LOG = HERE / "filter_nll_diffscale2_20seeds.log"
PROGRESS = HERE / "PROGRESS.txt"
PIDFILE = HERE / "watch_progress.pid"
SEEDS_TOTAL = 20
SEED_RE = re.compile(r"=== seed (\d+)")


def daemonize() -> None:
    if os.fork() > 0:
        raise SystemExit(0)
    os.setsid()
    if os.fork() > 0:
        raise SystemExit(0)
    sys.stdout.flush()
    sys.stderr.flush()
    with open(os.devnull, "rb") as devnull:
        os.dup2(devnull.fileno(), 0)
    logf = open(HERE / "progress_watcher.log", "a", buffering=1)
    os.dup2(logf.fileno(), 1)
    os.dup2(logf.fileno(), 2)


def write_progress() -> bool:
    """Return True if the run appears finished."""
    if not LOG.is_file():
        PROGRESS.write_text(
            f"updated: {datetime.now().strftime('%H:%M:%S')}\nwaiting for log…\n"
        )
        return False
    text = LOG.read_text()
    started = sorted({int(s) for s in SEED_RE.findall(text)})
    lines = text.strip().splitlines()
    last = lines[-1] if lines else ""
    cur = started[-1] if started else None
    n_done = 0
    for s in started:
        block = text.split(f"=== seed {s} ===", 1)[1].split("=== seed", 1)[0]
        if "KLV-PF       KLV-2000" in block:
            n_done += 1
    pct = 100.0 * n_done / SEEDS_TOTAL
    alive = "yes" if _main_alive() else "no"
    msg = (
        f"updated: {datetime.now().strftime('%H:%M:%S')}\n"
        f"run alive: {alive}\n"
        f"run: filter_nll diff_scale=2, 20 seeds, EnKF/PF={{50,100,500,2000}}\n"
        f"seeds completed: {n_done}/{SEEDS_TOTAL} ({pct:.0f}%)\n"
        f"current seed: {cur}\n"
        f"seeds seen: {started}\n"
        f"last log line:\n  {last}\n"
    )
    PROGRESS.write_text(msg)
    print(msg, flush=True)
    return ("filter_pred_nll_pareto.png" in text and "Wrote" in text) or (
        alive == "no" and n_done >= SEEDS_TOTAL
    )


def _main_alive() -> bool:
    try:
        import subprocess

        out = subprocess.check_output(["pgrep", "-f", "filter_nll_pareto.py"], text=True)
        return bool(out.strip())
    except Exception:
        return False


def main() -> None:
    daemonize()
    PIDFILE.write_text(str(os.getpid()) + "\n")
    print(f"watcher start pid={os.getpid()} at {datetime.now()}", flush=True)
    try:
        while True:
            if write_progress():
                with PROGRESS.open("a") as f:
                    f.write("\nDONE\n")
                print("DONE", flush=True)
                break
            time.sleep(10)
    finally:
        if PIDFILE.is_file():
            PIDFILE.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
