r"""Dispatch diagnostics for RICH snapshots in one simulation run.

SYNOPSIS
    python scripts/dispatch_diagnostics.py [OPTIONS] SNAP_DIR OUTPUT_DIR

    Uppercase names are values to supply. Brackets mark optional items.
    Both positional arguments are required and must appear in this order.

DESCRIPTION
    Discover snapshot files in SNAP_DIR and run diagnostics.py for files
    that are incomplete or explicitly forced. By default, consider only the
    highest-numbered snapshot, run all diagnostic checks, and process matching
    benchmark logs. Messages go to the terminal and OUTPUT_DIR/dispatch.log.

ARGUMENTS
    SNAP_DIR
        Required directory for ONE simulation run; not a snapshot filename
        or a parent containing multiple runs. Search is not recursive.
        Accepted filenames: snap_<N>.h5 and snap_full_<N>.h5, where N is an
        integer snapshot number. Both forms are included. "Latest" means the
        highest N, not the newest modification time.

        Plot checks need R<Rstar>M<Mstar>BH<Mbh>beta<beta> in the path.
        Benchmark matching additionally needs S<S>n<n>, for example:
        /data/R0.47M0.5BH100000beta1S60n1.5ComptonHiResNewAMR
        Rstar is in solar radii; Mstar and Mbh are in solar masses.

    OUTPUT_DIR
        Required destination directory; created if absent. Reuse the same
        directory to recognize completed snapshots from its cache and plots.

OPTIONS
    --all
        Consider every discovered snapshot. Default: latest snapshot only.
        Completed snapshots are still skipped unless --overwrite is given.
        This option selects snapshots; it does not select diagnostic checks.

    -c, --check NAME
        Run only this diagnostic check. Repeat to select several checks:
        --check slice --check projection. Default: all checks listed below.
        Does not force a rerun or disable benchmark processing.

    --overwrite
        Boolean flag; default: disabled. Rerun selected checks even for completed
        snapshots. Without --all, only the latest snapshot is considered; with
        --all, every discovered snapshot is considered. Omit this flag to skip
        completed snapshots. Incomplete snapshots are processed either way.

    --measure-speed / --no-measure-speed
        Enable or disable matching benchmark-log analysis. Default: enabled.
        This runs independently of --check and snapshot completion.

    --log-dir DIR
        Directory containing benchmark *.out files; searched non-recursively.
        Match Slurm job-name parameters R, Mstar, Mbh, and beta to SNAP_DIR.
        Ignored with --no-measure-speed.
        Default: /data2/yujiehe/rich-monitor/snellius-backup/logs

    --dry-run
        Report eligible snapshots and benchmark logs without running their
        analysis. Default: disabled. Still creates output directories and
        appends to dispatch.log.

    -h, --help
        Show command help and exit without dispatching work.

    --install-completion / --show-completion
        Install shell completion / print its configuration for the current
        shell, then exit. These options are provided by Typer.

CHECKS
    Accepted NAME values for --check (case-sensitive):

    integrity            Check field lengths, unexpected zeros, and NaN/Inf.
    resolution           Plot cell-radius distributions by count and mass.
    conservation         Log total mass and kinetic/thermal/radiation energy.
    compton              Log hot-gas mass fraction; warn above 5% at T > 1e8 K.
    time_evolution       Cache missing scalars for snapshots in the run
                         directory and redraw histories; reuse cached values.
    slice                Nine-panel xy slices: small, middle, and big scales.
    projection           Four-panel xy/xz projections: middle and big scales.
    pericenter_yz        Nine-panel yz slice at x = r_p.
    yz_frac_pericenter   Nine-panel yz slice at the fixed x = 18 code-length
                         probe, labeled by its fraction of r_p.

EXIT STATUS
    0   Dispatch finished, including when no snapshots were found. Logged
        failures from child diagnostics or speed analysis do not currently
        make dispatch exit nonzero; inspect dispatch.log for those failures.
    2   Command-line parsing error, such as a missing required argument.
    Other nonzero values can result from unhandled runtime errors.

FILES
    OUTPUT_DIR/dispatch.log                 Append-only dispatch log.
    OUTPUT_DIR/diagnostics.log              Log written by diagnostics.py.
    OUTPUT_DIR/diagnostics_cache.json       Cached scalars, keyed by number.
    OUTPUT_DIR/figs/<series>/snapNNNN.png    Per-snapshot diagnostic plots.
    OUTPUT_DIR/figs/time_evolution_*.png    Run-wide time histories.
    OUTPUT_DIR/speed_figs/<log-stem>.png    Benchmark plots, when enabled.

NOTES
    A completed snapshot has both a cache entry and the full expected number
    of per-snapshot PNGs. Selecting fewer checks does not reduce this count.
    Use --overwrite to regenerate plots for completed snapshots.

    --overwrite does not expand the latest-only selection or exclude incomplete
    snapshots. It does not invalidate time_evolution's cached scalar values.

    Full and non-full files with the same N share cache keys and figure paths.
    In latest-only mode, neither form has guaranteed priority when N ties.
    Invoke diagnostics.py directly when you need exactly one specific file.

EXAMPLES
    Run these commands from the repository root. Replace /data/RUN with a
    run directory following the naming convention in ARGUMENTS.

    Process the latest snapshot if incomplete:
        python scripts/dispatch_diagnostics.py /data/RUN ./output

    Regenerate only xy nine-panel plots for the latest snapshot:
        python scripts/dispatch_diagnostics.py /data/RUN ./output \
            --check slice --overwrite --no-measure-speed

    Regenerate those plots for every snapshot:
        python scripts/dispatch_diagnostics.py /data/RUN ./output \
            --all --check slice --overwrite --no-measure-speed

    Preview two checks across all incomplete snapshots:
        python scripts/dispatch_diagnostics.py /data/RUN ./output \
            --all -c slice -c projection --no-measure-speed --dry-run

    Run only slice plots for one exact file, bypassing dispatch:
        python scripts/diagnostics.py /data/RUN/snap_full_68.h5 ./output \
            --check slice

SEE ALSO
    diagnostics.py --help; measure_speed.py --help
"""

import glob
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path

import typer
from diagnostics import CACHE_FNAME, CHECK_HELP, N_EXPECTED_PNGS_PER_SNAP, _snap_num
from loguru import logger
from progress_monitor import _parse_job_name, _parse_run_params

app = typer.Typer(
    context_settings={"help_option_names": ["-h", "--help"]},
    rich_markup_mode="markdown",
)

# --------------------------------- Defaults --------------------------------- #

_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "diagnostics.py")
_MEASURE_SPEED_SCRIPT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "measure_speed.py"
)
_LOG_DIR = "/data2/yujiehe/rich-monitor/snellius-backup/logs"

# --------------------------------- Helpers ---------------------------------- #


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


def _find_log_files(snap_dir: str, log_dir: str = _LOG_DIR) -> list[Path]:
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


@app.command(
    epilog=CHECK_HELP
    + r"""

## Examples

Run from the repository root; replace `/data/RUN` with your run directory.

- **Latest incomplete snapshot:**
  `python scripts/dispatch_diagnostics.py /data/RUN ./output`

- **Only xy nine-panel plots, latest snapshot, even if complete:**
  `python scripts/dispatch_diagnostics.py /data/RUN ./output
  --check slice --overwrite --no-measure-speed`

- **The same plots for every snapshot:**
  `python scripts/dispatch_diagnostics.py /data/RUN ./output
  --all --check slice --overwrite --no-measure-speed`

- **Exactly one file, using diagnostics directly:**
  `python scripts/diagnostics.py /data/RUN/snap_full_68.h5 ./output
  --check slice`

## Notes

`--overwrite` does not clear cached time-evolution values. Full and non-full
files with the same number share output names and cache keys; use the direct
command above when the exact filename matters.

The opening docstring in `scripts/dispatch_diagnostics.py` contains the full
reference, including filename conventions, output files, and exit status.
"""
)
def main(
    snap_dir: str = typer.Argument(
        help=(
            "One run directory containing `snap_<N>.h5` and/or `snap_full_<N>.h5`; "
            "searched non-recursively. Latest means highest snapshot number, not modification time."
        ),
    ),
    output_dir: str = typer.Argument(
        help="Destination for figures, cache, and logs; created if absent. Reuse it to recognize completed snapshots.",
    ),
    overwrite: bool = typer.Option(
        False,
        "--overwrite",
        rich_help_panel="Rerun policy",
        help="Rerun completed snapshots (default: disabled). Latest only unless --all is given; incomplete snapshots always run.",
    ),
    all_snaps: bool = typer.Option(
        False,
        "--all",
        rich_help_panel="Snapshot selection",
        help="Consider every snapshot instead of latest only; completed ones still skip unless forced.",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        rich_help_panel="Execution",
        help="Preview without running diagnostics or speed analysis; still creates directories/logs.",
    ),
    measure_speed: bool = typer.Option(
        True,
        "--measure-speed/--no-measure-speed",
        rich_help_panel="Benchmark logs",
        help="Process matching benchmark logs (default: enabled; independent of --check).",
    ),
    log_dir: str = typer.Option(
        _LOG_DIR,
        "--log-dir",
        metavar="DIR",
        rich_help_panel="Benchmark logs",
        help="Search DIR/*.out for Slurm job names matching the run's R, Mstar, Mbh, and beta; save plots in OUTPUT_DIR/speed_figs/.",
    ),
    checks: list[str] | None = typer.Option(
        None,
        "--check",
        "-c",
        metavar="NAME",
        rich_help_panel="Diagnostics",
        help=(
            "Run only this diagnostic (repeatable; default: all checks). "
            "Use 'slice' for nine-panel plots. Add --overwrite to rerun completed "
            "snapshots and --no-measure-speed to skip benchmark logs. "
            "Completion detection still expects the full cache/figure set. "
            "See check descriptions below."
        ),
    ),
):
    """Run diagnostics for new, incomplete, or explicitly forced snapshots.

    By default, consider only the highest-numbered snapshot and run all checks.
    Use --all to consider every snapshot; use --check NAME to select checks.
    These settings are independent. Both full and non-full filenames are discovered.

    Completed snapshots are skipped unless --overwrite is given. Completion
    requires a diagnostics_cache.json entry and the full expected figure count,
    even when --check selects fewer checks. --overwrite does not select files
    or expand the latest-only scope. Use diagnostics.py for exactly one file.

    Matching benchmark logs are processed separately by default; disable this
    with --no-measure-speed. Add --dry-run to preview snapshot selection.
    """
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
    n_run = n_skip = 0

    for snap_path in snaps:
        n = _snap_num(snap_path)
        if not overwrite and _is_done(snap_path, output_dir, cache):
            logger.debug(
                "Skip snap_{} — already done ({} PNGs, in cache)",
                n,
                _n_pngs(n, output_dir),
            )
            n_skip += 1
            continue

        reason = (
            "overwrite"
            if overwrite
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
