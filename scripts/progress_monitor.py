#!/usr/bin/env python3
"""
Progress monitor of the RICH simulation runs.

Queries sacct from Snellius (via SSH), caches results, and writes a CSV report.
"""
import csv
import json
import os
import subprocess
import re
from datetime import date, timedelta
from typing import Optional

import unyt as u
import typer
from loguru import logger

import richio
from diagnostics import _snap_num

app = typer.Typer()

# --------------------------------- Constants -------------------------------- #

LOG_FNAME         = "progress-monitor.log"
OUTPUT_FNAME      = "progress.csv"
SACCT_CACHE_FNAME = "sacct_cache.json"
SNAP_CACHE_FNAME  = "snap_cache.json"

SBU_BUDGET     = 22_300_000   # total core-hour budget for this project

SNELLIUS_HOST  = "snellius"   # SSH alias - configured in ~/.ssh/config
SNELLIUS_USER  = "yhe"
PROJECT_START  = date(2026, 3, 1)   # earliest date to query sacct from
CHUNK_DAYS     = 7                   # sacct window size in days

# Which job fields to query from sacct
SACCT_FIELDS = ["JobID", "JobName", "State", "Start", "End", "Elapsed", "AllocCPUS", "NNodes"]

# Controls which fields appear in the per-job CSV table and in what order.
# To add/remove columns or reorder, edit this list.
# Job fields available (raw sacct + derived):
#   JobID, JobName, State, Start, End, Elapsed, AllocCPUS, NNodes  <- raw sacct
#   WallHours, Cores, CoreHours                                     <- derived
_CSV_FIELDS = ["JobID", "JobName", "State", "Start", "End", "WallHours", "Cores", "CoreHours"]

# ---------------------------------- Utils ----------------------------------- #

def _parse_run_params(path: str) -> tuple:
    """Parse TDE run parameters from a directory path.

    :param path: Path containing the run name, e.g.
        ``/data/R0.47M0.5BH100000beta1S60n1.5ComptonHiResNewAMR/``.
    :returns: ``(R, Mstar, Mbh, beta, n)`` as floats — stellar radius [Rsun],
        stellar mass [Msun], BH mass [Msun], beta, polytropic index.
    :rtype: tuple[float, float, float, float, float]
    :raises Exception: If the path does not match the expected naming convention.

    .. code-block:: python

        R, Mstar, Mbh, beta, n = _parse_run_params("/data/R0.47M0.5BH100000beta1S60n1.5.../")
    """
    float_capture = r"([+-]?[0-9]*[.]?[0-9]+)"
    # https://stackoverflow.com/questions/12643009/regular-expression-for-floating-point-numbers
    m = re.search("R{0}M{0}BH{0}beta{0}S{0}n{0}".format(float_capture), path)
    if m:
        return tuple(float(x) for x in m.group(1, 2, 3, 4, 6))
    else:
        raise Exception(f"No match found in {path} when parsing for TDE parameters.")

def _elapsed_to_hours(elapsed: str) -> float:
    """Convert a sacct Elapsed string to hours.

    :param elapsed: Elapsed time in ``D-HH:MM:SS`` or ``HH:MM:SS`` format.
    :returns: Elapsed time in hours.
    :rtype: float
    """
    if "-" in elapsed:
        days_str, time_part = elapsed.split("-", 1)
        days = int(days_str)
    else:
        time_part, days = elapsed, 0
    h, m, s = (int(x) for x in time_part.split(":"))
    return days * 24 + h + m / 60 + s / 3600


def _fetch_sacct_chunk(start: date, end: date) -> list:
    """Query sacct for one date window and return parsed job records.

    Runs via SSH if :data:`SNELLIUS_HOST` is set, locally otherwise
    (use when running directly on Snellius).

    :param start: First day of the query window (inclusive).
    :param end: Last day of the query window (inclusive).
    :returns: List of job dicts with keys from :data:`SACCT_FIELDS`
        plus ``WallHours``, ``Cores``, ``CoreHours``.
    :rtype: list[dict]
    """
    fmt   = ",".join(SACCT_FIELDS)
    s_str = start.strftime("%Y-%m-%dT00:00:00")
    e_str = end.strftime("%Y-%m-%dT23:59:59")
    sacct_cmd = (
        f"sacct -u {SNELLIUS_USER} "
        f"--starttime={s_str} --endtime={e_str} "
        f"--format={fmt} --parsable2 --noheader"
    )
    cmd = ["ssh", SNELLIUS_HOST, sacct_cmd] if SNELLIUS_HOST \
          else ["bash", "-c", sacct_cmd]

    logger.info("sacct query {} → {}", start, end)
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error("sacct failed: {}", result.stderr.strip())
        return []

    rows = []
    for line in result.stdout.strip().splitlines():
        if not line:
            continue
        parts = line.split("|")
        if len(parts) < len(SACCT_FIELDS):
            continue
        job = dict(zip(SACCT_FIELDS, parts))
        # Skip job steps (12345.batch, 12345.0) - keep only top-level jobs
        if "." in job["JobID"]:
            continue
        # Skip entries with no start time (pending jobs)
        if job["Start"] in ("Unknown", "None", ""):
            continue
        job["WallHours"] = round(_elapsed_to_hours(job["Elapsed"]), 3)
        job["Cores"]     = int(job["AllocCPUS"])
        # SBU = core * hour
        job["CoreHours"] = round(job["WallHours"] * job["Cores"], 1)
        rows.append(job)
    return rows


def _chunk_dates(project_start: date, today: date) -> list:
    """Generate non-overlapping :data:`CHUNK_DAYS`-day windows covering a date range.

    :param project_start: First date of the range.
    :param today: Last date of the range (inclusive).
    :returns: List of ``(start, end)`` date pairs.
    :rtype: list[tuple[date, date]]
    """
    chunks, start = [], project_start
    while start <= today:
        end = min(start + timedelta(days=CHUNK_DAYS - 1), today)
        chunks.append((start, end))
        start = end + timedelta(days=1)
    return chunks


def _get_all_jobs(output_dir: str) -> list:
    """Fetch all sacct jobs since :data:`PROJECT_START` with disk caching.

    Past chunks (end < today) are cached permanently — history does not change.
    The current chunk (includes today) is always re-queried.

    :param output_dir: Directory where :data:`SACCT_CACHE_FNAME` is stored.
    :returns: List of all job dicts across all date chunks.
    :rtype: list[dict]
    """
    cache_path = os.path.join(output_dir, SACCT_CACHE_FNAME)
    cache: dict = {}
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            cache = json.load(f)

    today         = date.today()
    all_jobs      = []
    cache_updated = False

    for start, end in _chunk_dates(PROJECT_START, today):
        key = f"{start}_{end}"
        if key in cache:
            all_jobs.extend(cache[key])
            continue
        rows = _fetch_sacct_chunk(start, end)
        all_jobs.extend(rows)
        if end < today:          # closed chunk - safe to cache permanently
            cache[key]     = rows
            cache_updated  = True

    if cache_updated:
        with open(cache_path, "w") as f:
            json.dump(cache, f, indent=2)
        logger.info("sacct cache updated: {}", cache_path)

    return all_jobs


def _canonical_name(R: float, Mstar: float, Mbh: float, beta: float) -> str:
    """Reconstruct the canonical run name from parameters.

    Matches the directory naming convention so the key is human-readable
    and directly tells collaborators where to find the data.

    :returns: A string like ``R0.47M0.5BH100000beta1``.
    :rtype: str
    """
    return f"R{R:g}M{Mstar:g}BH{int(Mbh)}beta{beta:g}"


def _parse_job_name(name: str) -> dict | None:
    """Parse a SLURM job name like ``M05R047MBH1e5beta1`` into run parameters.

    Naming convention: decimal points are omitted (``0.47`` → ``047``),
    BH mass may use scientific notation (``1e5`` = 10 :sup:`5`).

    :param name: SLURM job name string.
    :returns: Dict with keys ``R``, ``Mstar``, ``Mbh``, ``beta`` as floats,
        or ``None`` if the name does not match.
    :rtype: dict | None
    """
    def _undot(s):
        # Reconstruct float from decimal-less string: "05" -> 0.5, "047" -> 0.47, "1" -> 1.0
        return float(s[0] + "." + s[1:]) if s.startswith("0") and len(s) > 1 else float(s)

    m = re.search(r"M(\d+)R(\d+)MBH([0-9e+]+)beta([0-9.]+)", name, re.IGNORECASE)
    if not m:
        return None
    return dict(
        Mstar = _undot(m.group(1)),
        R     = _undot(m.group(2)),
        Mbh   = float(m.group(3)),
        beta  = float(m.group(4)),
    )


def _get_latest_snap_info(input_dir: str) -> dict:
    """Get the latest snapshot metadata for each run directory under *input_dir*.

    For each subdirectory matching the run naming convention, finds the
    highest-numbered snapshot and loads its metadata.

    :param input_dir: Parent directory containing run subdirectories.
    :returns: Dict mapping a canonical run name (see :func:`_canonical_name`) to a snap info
        dict with keys ``snapnum``, ``time_day``, ``time_in_tfb``, ``cycle``,
        ``point_num``.
    :rtype: dict
    """
    result = {}
    for entry in os.listdir(input_dir):
        run_path = os.path.join(input_dir, entry)
        if not os.path.isdir(run_path):
            continue
        try:
            R, Mstar, Mbh, beta, n = _parse_run_params(entry)
        except Exception:
            continue  # skip dirs that don't follow the naming convention

        tfb   = 4 * (Mbh / 1e4)**0.5 / Mstar * R**1.5 * u.day
        snaps = [f for f in os.listdir(run_path) if f.endswith(".h5")]
        if not snaps:
            continue

        latest_file = max(snaps, key=_snap_num)
        snapnum     = _snap_num(latest_file)
        snap        = richio.load(os.path.join(run_path, latest_file))

        time        = snap.time.to("day")
        time_day    = float(time.v)
        time_in_tfb = float((time / tfb).v)
        cycle       = int(snap.cycle)
        point_num   = len(snap)

        key = _canonical_name(R, Mstar, Mbh, beta)
        result[key] = dict(
            snapnum     = snapnum,
            time_day    = round(time_day, 4),
            time_in_tfb = round(time_in_tfb, 4),
            cycle       = cycle,
            point_num   = point_num,
        )
        logger.info("  {} → snap_{}, t={:.3f} day ({:.3f} tfb), cycle {}, N={}",
                    entry, snapnum, time_day, time_in_tfb, cycle, point_num)

    return result

# ---------------------------------- Output ---------------------------------- #

def _write_report(jobs: list, snaps: dict, output_file: str):
    """Write progress CSV: per-job table + per-run summary.

    Jobs that match a run directory (via :func:`_parse_job_name` +
    :func:`_canonical_name`) get snapshot columns filled in; unmatched jobs
    leave those columns empty.

    :param jobs: List of job dicts from :func:`_get_all_jobs`.
    :param snaps: Dict from :func:`_get_latest_snap_info` mapping params key
        to snap info, or empty dict if unavailable.
    :param output_file: Path to write the CSV report.
    """
    summary: dict = {}
    for job in jobs:
        name = job["JobName"]
        if name not in summary:
            # Try to attach latest snapshot info once per run name
            params = _parse_job_name(name)
            snap = {}
            if params:
                key  = _canonical_name(params["R"], params["Mstar"], params["Mbh"], params["beta"])
                snap = snaps.get(key, {})
            summary[name] = {
                "Jobs":      0,
                "CoreHours": 0.0,
                "Running":   False,
                "SnapNum":   snap.get("snapnum",     ""),
                "TimeDays":  snap.get("time_day",    ""),
                "TimeInTfb": snap.get("time_in_tfb", ""),
                "Cycle":     snap.get("cycle",       ""),
                "PointNum":  snap.get("point_num",   ""),
            }
        summary[name]["Jobs"]      += 1
        summary[name]["CoreHours"] += job["CoreHours"]
        if job["State"] == "RUNNING":
            summary[name]["Running"] = True

    total_core_hours = sum(info["CoreHours"] for info in summary.values())

    with open(output_file, "w", newline="") as f:
        # Summary table (main, parsable)
        _SUMMARY_FIELDS = ["RunName", "Jobs", "TotalCoreHours", "Running",
                           "SnapNum", "TimeDays", "TimeInTfb", "Cycle", "PointNum"]
        writer = csv.DictWriter(f, fieldnames=_SUMMARY_FIELDS)
        writer.writeheader()
        for name, info in sorted(summary.items(), key=lambda x: x[1]["CoreHours"], reverse=True):
            writer.writerow({
                "RunName":       name,
                "Jobs":          info["Jobs"],
                "TotalCoreHours": f"{info['CoreHours']:.1f}",
                "Running":       "yes" if info["Running"] else "no",
                "SnapNum":       info["SnapNum"],
                "TimeDays":      info["TimeDays"],
                "TimeInTfb":     info["TimeInTfb"],
                "Cycle":         info["Cycle"],
                "PointNum":      info["PointNum"],
            })
        writer.writerow({
            "RunName": "TOTAL", "Jobs": "", "TotalCoreHours": f"{total_core_hours:.1f}",
            "Running": "", "SnapNum": "", "TimeDays": "", "TimeInTfb": "", "Cycle": "", "PointNum": "",
        })
        f.write(f"# SBU budget remaining: {SBU_BUDGET - total_core_hours:.0f}"
                f" / {SBU_BUDGET} ({100*(1 - total_core_hours/SBU_BUDGET):.2f}% left)\n")

        # Per-job details (commented, for reference)
        f.write("\n# Per-job details\n")
        f.write("# " + ",".join(_CSV_FIELDS) + "\n")
        for job in sorted(jobs, key=lambda j: j["Start"]):
            f.write("# " + ",".join(str(job.get(k, "")) for k in _CSV_FIELDS) + "\n")

    logger.success("Report written: {}", output_file)


# ----------------------------------- Main ----------------------------------- #

@app.command()
def main(
    input_dir: str = typer.Argument(
        help="Directory containing simulation snapshots.",
    ),
    output_dir: str = typer.Argument(
        help="Output directory for report and cache.",
    ),
):
    """Monitor progress of RICH simulation runs.

    Examples:
    .. code-block:: shell
      python progress_monitor.py /disks/emrdata/YujieSnellius/ /data1/yujiehe/snellius-logs/
    """
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, OUTPUT_FNAME)
    log_file = os.path.join(output_dir, LOG_FNAME)
    log_sink = logger.add(log_file, mode="a")

    logger.info("Monitoring {}, output to {}. Log at {}", input_dir, output_file, log_file)

# ----------------------------------- sacct ---------------------------------- #
    jobs = _get_all_jobs(output_dir)
    logger.info("Retrieved {} job entries from sacct", len(jobs))

# ------------------------------- Snapshot info ------------------------------ #
    snaps = _get_latest_snap_info(input_dir)
    logger.info("Loaded latest snapshot info for {} run(s)", len(snaps))

# ------------------------------- Write report ------------------------------- #
    if jobs:
        _write_report(jobs, snaps, output_file)
    else:
        logger.warning("No jobs retrieved - check SSH access to Snellius")

    logger.success("Done. Progress saved to {}", output_file)
    logger.remove(log_sink)


if __name__ == "__main__":
    app()
