#!/usr/bin/env python3
"""
Dispatch diagnostics.py for snapshots in a directory.

By default, only the latest snapshot is processed (pass --all to process all).
A snapshot is considered "done" when:
  1. Its entry exists in diagnostics_cache.json, AND
  2. At least N_EXPECTED_PNGS files matching *_snap{NNNN}*.png exist in output_dir.

To add/remove a check in diagnostics.py that produces a PNG: update N_EXPECTED_PNGS.

Usage:
  python dispatch_diagnostics.py                          # process only latest snap for default directory
  python dispatch_diagnostics.py /disks/emrdata/YujieSnellius/R0.47M0.5BH100000beta1S60n1.5ComptonHiResNewAMR
  python dispatch_diagnostics.py --all                    # process all new/incomplete snaps
  python dispatch_diagnostics.py --overwrite-all          # rerun everything
  python dispatch_diagnostics.py --overwrite-full         # rerun snap_full_* only
  python dispatch_diagnostics.py --overwrite-nonfull      # rerun snap_* (non-full) only
  python dispatch_diagnostics.py --overwrite-snap 67 --overwrite-snap 68
  python dispatch_diagnostics.py --dry-run                # preview without running
"""

import glob
import json
import os
import subprocess
import sys
from typing import List, Optional

import typer
from loguru import logger

from diagnostics import _snap_num, CACHE_FNAME

app = typer.Typer()

# --------------------------------- Defaults --------------------------------- #

_SCRIPT     = os.path.join(os.path.dirname(os.path.abspath(__file__)), "diagnostics.py")

# Update this when you add or remove a per-snapshot PNG-producing check.
N_EXPECTED_PNGS = 6

# --------------------------------- Helpers ---------------------------------- #

def _is_full(snap_path: str) -> bool:
    return "snap_full_" in os.path.basename(snap_path)


def _n_pngs(n: int, output_dir: str) -> int:
    """Count PNG files for this snapshot number in output_dir."""
    return len(glob.glob(os.path.join(output_dir, f"figs/*_snap{n:04d}*.png")))


def _is_done(snap_path: str, output_dir: str, cache: dict) -> bool:
    n = _snap_num(snap_path)
    return str(n) in cache and _n_pngs(n, output_dir) >= N_EXPECTED_PNGS


def _load_cache(output_dir: str) -> dict:
    cache_path = os.path.join(output_dir, CACHE_FNAME)
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            return json.load(f)
    return {}


def _find_snaps(snap_dir: str) -> list:
    """Return all snapshots sorted by snap number."""
    snaps = glob.glob(os.path.join(snap_dir, "snap_[0-9]*.h5"))
    snaps += glob.glob(os.path.join(snap_dir, "snap_full_[0-9]*.h5"))
    return sorted(set(snaps), key=_snap_num)


# ----------------------------------- Main ----------------------------------- #

@app.command()
def main(
    snap_dir: str = typer.Argument(help="Directory containing snapshots."),
    output_dir: str = typer.Argument(help="Output directory for figures and cache."),
    overwrite_all: bool = typer.Option(False, "--overwrite-all", help="Rerun all snapshots."),
    overwrite_full: bool = typer.Option(False, "--overwrite-full", help="Rerun snap_full_*.h5 snapshots."),
    overwrite_nonfull: bool = typer.Option(False, "--overwrite-nonfull", help="Rerun snap_*.h5 (non-full) snapshots."),
    overwrite_snaps: Optional[List[int]] = typer.Option(
        None, "--overwrite-snap",
        help="Rerun a specific snapshot number (repeat for multiple).",
    ),
    all_snaps: bool = typer.Option(False, "--all", help="Process all unprocessed snapshots (default: latest only)."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print what would run without running."),
):
    os.makedirs(output_dir, exist_ok=True)
    log_sink = logger.add(os.path.join(output_dir, "dispatch.log"), mode="a")

    snaps = _find_snaps(snap_dir)
    if not snaps:
        logger.warning("No snapshots found in {}", snap_dir)
        logger.remove(log_sink)
        return

    logger.info("Found {} snapshot(s) in {}", len(snaps), snap_dir)

    if not all_snaps:
        snaps = snaps[-1:]
        logger.info("Latest-only mode: processing snap_{}", _snap_num(snaps[0]))

    cache = _load_cache(output_dir)
    overwrite_nums = set(overwrite_snaps or [])
    n_run = n_skip = 0

    for snap_path in snaps:
        n = _snap_num(snap_path)
        is_full = _is_full(snap_path)

        force = (
            overwrite_all
            or (overwrite_full and is_full)
            or (overwrite_nonfull and not is_full)
            or (n in overwrite_nums)
        )

        if not force and _is_done(snap_path, output_dir, cache):
            logger.debug("Skip snap_{} — already done ({} PNGs, in cache)", n, _n_pngs(n, output_dir))
            n_skip += 1
            continue

        reason = "overwrite" if force else f"incomplete ({_n_pngs(n, output_dir)}/{N_EXPECTED_PNGS} PNGs)"
        if dry_run:
            logger.info("[dry-run] snap_{} ({}) — {}", n, os.path.basename(snap_path), reason)
            n_run += 1
            continue

        logger.info("Running diagnostics for snap_{} ({})", n, reason)
        result = subprocess.run([sys.executable, _SCRIPT, snap_path, output_dir])
        if result.returncode == 0:
            cache = _load_cache(output_dir)  # refresh for next iteration
            n_run += 1
        else:
            logger.error("diagnostics.py failed for snap_{}", n)

    logger.info("Finished. Ran: {}, Skipped: {}", n_run, n_skip)
    logger.remove(log_sink)


if __name__ == "__main__":
    app()
