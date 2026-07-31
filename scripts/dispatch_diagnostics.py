"""
Dispatch diagnostics.py for snapshots in a directory.

By default, only the latest snapshot is processed (pass --all to process all).
A snapshot is considered "done" when:
  1. Its entry exists in diagnostics_cache.json, AND
  2. At least N_EXPECTED_PNGS_PER_SNAP files exist for it in
     output_dir/figs/{series}/snap{NNNN}.png (one series subfolder per check).

N_EXPECTED_PNGS_PER_SNAP is imported from diagnostics.py, derived from its own
check config — no manual sync needed here when checks are added/removed.

This script also integrates measure_speed.py to process Snellius benchmark logs
that match the simulation parameters extracted from the snapshot directory name.

Usage:
  python dispatch_diagnostics.py /path/to/snaps /path/to/output
  python dispatch_diagnostics.py --all                    # process all new/incomplete snaps
  python dispatch_diagnostics.py --overwrite-all          # rerun everything
  python dispatch_diagnostics.py --overwrite-full         # rerun snap_full_* only
  python dispatch_diagnostics.py --overwrite-nonfull      # rerun snap_* (non-full) only
  python dispatch_diagnostics.py --overwrite-snap 67 --overwrite-snap 68
  python dispatch_diagnostics.py --dry-run                # preview without running
  python dispatch_diagnostics.py --no-measure-speed       # skip benchmark log processing
"""

import glob
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

import typer
from diagnostics import CACHE_FNAME, N_EXPECTED_PNGS_PER_SNAP, _snap_num
from loguru import logger
from progress_monitor import _parse_job_name, _parse_run_params

app = typer.Typer()

# --------------------------------- Defaults --------------------------------- #

_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "diagnostics.py")
_MEASURE_SPEED_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "measure_speed.py")
_LOG_DIR = "/data2/yujiehe/rich-monitor/snellius-backup/logs"

# --------------------------------- Helpers ---------------------------------- #


def _is_full(snap_path: str) -> bool:
    return "snap_full_" in os.path.basename(snap_path)


def _n_pngs(n: int, output_dir: str) -> int:
    """Count PNG files for this snapshot number across figs/{series}/ folders."""
    return len(glob.glob(os.path.join(output_dir, f"figs/*/snap{n:04d}.png")))


def _is_done(snap_path: str, output_dir: str, cache: dict) -> bool:
    n = _snap_num(snap_path)
    return str(n) in cache and _n_pngs(n, output_dir) >= N_EXPECTED_PNGS_PER_SNAP


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


# --------------------------------- Speed Measurement --------------------------------- #


def _find_log_files(snap_dir: str, log_dir: str = _LOG_DIR) -> List[Path]:
    """Find log files in log_dir whose SLURM job name matches snap_dir's parameters."""
    try:
        R, Mstar, Mbh, beta, _ = _parse_run_params(snap_dir)
    except Exception as e:
        logger.warning("Could not parse run parameters from {}: {}", snap_dir, e)
        return []

    matched = []
    for path in sorted(Path(log_dir).glob("*.out")):
        m = re.fullmatch(r"\d+_(.+)", path.stem)
        if not m:
            continue
        params = _parse_job_name(m.group(1))
        if params and (
            math.isclose(params["R"], R)
            and math.isclose(params["Mstar"], Mstar)
            and math.isclose(params["Mbh"], Mbh)
            and math.isclose(params["beta"], beta)
        ):
            matched.append(path)
    return matched


def _process_log_files(
    snap_dir: str, output_dir: str, log_dir: str = _LOG_DIR, dry_run: bool = False
) -> int:
    """Find log files matching snap_dir's parameters and delegate to measure_speed.py."""
    speed_output_dir = os.path.join(output_dir, "speed_figs")
    os.makedirs(speed_output_dir, exist_ok=True)

    log_files = _find_log_files(snap_dir, log_dir)
    if not log_files:
        logger.info("No log files found for {} in {}", snap_dir, log_dir)
        return 0

    logger.info("Found {} log file(s) for {}", len(log_files), snap_dir)

    if dry_run:
        for lf in log_files:
            logger.info("[dry-run] Would process {}", lf.name)
        return len(log_files)

    result = subprocess.run(
        [sys.executable, _MEASURE_SPEED_SCRIPT, "--output-dir", speed_output_dir]
        + [str(lf) for lf in log_files]
    )
    if result.returncode != 0:
        logger.error("measure_speed.py failed")
    return len(log_files)


# ----------------------------------- Main ----------------------------------- #


@app.command()
def main(
    snap_dir: str = typer.Argument(help="Directory containing snapshots."),
    output_dir: str = typer.Argument(help="Output directory for figures and cache."),
    overwrite_all: bool = typer.Option(
        False, "--overwrite-all", help="Rerun all snapshots."
    ),
    overwrite_full: bool = typer.Option(
        False, "--overwrite-full", help="Rerun snap_full_*.h5 snapshots."
    ),
    overwrite_nonfull: bool = typer.Option(
        False, "--overwrite-nonfull", help="Rerun snap_*.h5 (non-full) snapshots."
    ),
    overwrite_snaps: Optional[List[int]] = typer.Option(
        None,
        "--overwrite-snap",
        help="Rerun a specific snapshot number (repeat for multiple).",
    ),
    all_snaps: bool = typer.Option(
        False, "--all", help="Process all unprocessed snapshots (default: latest only)."
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Print what would run without running."
    ),
    measure_speed: bool = typer.Option(
        True, "--measure-speed/--no-measure-speed", help="Skip benchmark log processing."
    ),
    log_dir: str = typer.Option(
        _LOG_DIR,
        "--log-dir",
        help="Directory containing benchmark logs.",
    ),
    checks: Optional[List[str]] = typer.Option(
        None,
        "--check",
        "-c",
        help=(
            "Passthrough to diagnostics.py's --check/-c (repeat for multiple). "
            "Default: run all checks. Note: completion detection here always "
            "expects the full N_EXPECTED_PNGS_PER_SNAP count regardless of this "
            "filter — a snapshot missing e.g. resolution_check will never "
            "register as done if only ever run with a restricted subset. "
            "Intended for one-off backfills of snapshots that already have "
            "baseline coverage from a prior full run."
        ),
    ),
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
            logger.debug(
                "Skip snap_{} — already done ({} PNGs, in cache)",
                n,
                _n_pngs(n, output_dir),
            )
            n_skip += 1
            continue

        reason = (
            "overwrite"
            if force
            else f"incomplete ({_n_pngs(n, output_dir)}/{N_EXPECTED_PNGS_PER_SNAP} PNGs)"
        )
        if dry_run:
            logger.info(
                "[dry-run] snap_{} ({}) — {}", n, os.path.basename(snap_path), reason
            )
            continue

        logger.info("Running diagnostics for snap_{} ({})", n, reason)
        check_args = [a for c in (checks or []) for a in ("--check", c)]
        result = subprocess.run(
            [sys.executable, _SCRIPT, snap_path, output_dir] + check_args
        )
        if result.returncode == 0:
            cache = _load_cache(output_dir)  # refresh for next iteration
            n_run += 1
        else:
            logger.error("diagnostics.py failed for snap_{}", n)

    logger.info("Finished snapshot processing. Ran: {}, Skipped: {}", n_run, n_skip)

    # Process benchmark logs if enabled
    if measure_speed:
        logger.info("Processing benchmark logs")
        n_logs = _process_log_files(snap_dir, output_dir, log_dir, dry_run)
        logger.info("Processed {} log file(s)", n_logs)

    logger.remove(log_sink)


if __name__ == "__main__":
    app()
