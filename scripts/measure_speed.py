"""Parse Snellius benchmark logs and plot cell-updates per core per second."""

import glob
import os
import re
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import typer

app = typer.Typer()

# --------------------------------- Constants -------------------------------- #

INPUT_FILE = (
    "/data2/yujiehe/rich-monitor/snellius-backup/logs/21732963_M05R047MBH1e5beta1.out"
)
OUTPUT_DIR = "/data2/yujiehe/rich-monitor/snellius-logs/test_figs"
NPLOT = 100

# --------------------------------- Functions -------------------------------- #


def parse_file(path: Path) -> dict:
    text = path.read_text()

    re_ntasks = re.compile(r"--ntasks=(\d+)")
    re_part = re.compile(r"--partition=(\w+)")
    ncores = int(re_ntasks.search(text).group(1))
    partition = re_part.search(text).group(1)

    re_step = re.compile(r"Point num (\d+) dt (\S+) run time (\S+)\nCycle (\d+)")
    steps = re_step.findall(text)  # [(ncells, dt, run_time, cycle), ...]
    perf = []
    ncells = []
    cycle = []
    time = []
    dt = []
    for n, d, t, c in steps:
        if float(t) > 0:
            perf.append(int(n) / (float(t) * ncores))
            ncells.append(int(n))
            cycle.append(int(c))
            time.append(float(t))
            dt.append(float(d))

    return dict(
        partition=partition,
        ncores=ncores,
        perf=np.array(perf),
        ncells=np.array(ncells),
        cycle=np.array(cycle),
        time=np.array(time),
        dt=np.array(dt),
    )


def rolave(data, window):
    """
    Compute the rolling average of a 1D array over a given window size.
    """
    return np.lib.stride_tricks.sliding_window_view(data, window).mean(axis=-1)


# ----------------------------------- Main ----------------------------------- #


@app.command()
def main(
    input_files: List[str] = typer.Argument(
        help="Input RICH output file(s) or glob pattern(s)", default=[INPUT_FILE]
    ),
    output_dir: str = typer.Option(
        OUTPUT_DIR,
        "--output-dir",
        "-o",
        help="Output directory",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="",
    ),
):
    # Expand any glob patterns that the shell didn't expand (e.g. quoted globs)
    paths = []
    for pattern in input_files:
        expanded = glob.glob(pattern)
        if expanded:
            paths.extend(Path(p) for p in sorted(expanded))
        else:
            paths.append(Path(pattern))

    if not paths:
        typer.echo("No files found.", err=True)
        raise typer.Exit(1)

    # ------------------------------- Print summary ------------------------------ #

    print_header = True
    for p in paths:
        if print_header is True:
            print(
                f"{'Partition':<8} {'Ncores':>8} "
                f"{'Median [cell/core/s]':>22} {'Mean':>10}  {'File'}"
            )
            print("-" * 80)

            print_header = False

        fig_fname = os.path.join(output_dir, p.name.replace(".out", ".png"))

        if os.path.exists(fig_fname) and not force:
            if p.stat().st_mtime <= os.path.getmtime(fig_fname):
                # print(f"{fig_fname} is up to date.")
                continue
            # else:
            # print(f"{fig_fname} is outdated. Updating...")

        try:
            rec = parse_file(p)
        except Exception as e:
            # typer.echo(f"Skipping {p.name}: {e}", err=True)
            continue

        if len(rec["cycle"]) == 0:
            # typer.echo(f"Skipping {p.name}: no output found", err=True)
            continue

        print(
            f"{rec['partition']:<8} {rec['ncores']:>8} "
            f"{np.median(rec['perf']):>22.1f} {np.mean(rec['perf']):>10.1f}  {p.name}"
        )

        # ----------------------------------- Plot -----------------------------------

        _n = len(rec["cycle"])
        rolsize = _n // NPLOT if _n // NPLOT >= 2 else None

        fig, ax = plt.subplots(
            4, 1, figsize=(10, 7), sharex=True, constrained_layout=True
        )
        ax[0].plot(rec["cycle"], rec["perf"])
        ax[0].plot(
            rec["cycle"][: (-rolsize + 1)], rolave(rec["perf"], rolsize)
        ) if rolsize else None
        ax[0].axhline(np.mean(rec["perf"]), linestyle="--", color="k")
        ax[0].set_ylabel("Cell/core/s")

        ax[1].plot(rec["cycle"], rec["time"])
        ax[1].plot(
            rec["cycle"][: (-rolsize + 1)], rolave(rec["time"], rolsize)
        ) if rolsize else None
        ax[1].axhline(np.mean(rec["time"]), linestyle="--", color="k")
        ax[1].set_ylabel("Time[s]/step")

        ax[2].plot(rec["cycle"], rec["ncells"])
        ax[2].axhline(np.median(rec["ncells"]), linestyle="--", color="k")
        ax[2].set_ylabel("Number of cells")

        ax[3].plot(rec["cycle"], rec["dt"])
        ax[3].plot(
            rec["cycle"][: (-rolsize + 1)], rolave(rec["dt"], rolsize)
        ) if rolsize else None
        ax[3].axhline(np.median(rec["dt"]), linestyle="--", color="k")
        ax[3].set_ylabel("dt")
        ax[3].set_xlabel("Cycles")

        fig.suptitle(
            f"Partition: {rec['partition']}  Ncores: {rec['ncores']}  {p.name}"
        )
        fig.savefig(fig_fname, dpi=200)
        plt.close(fig)

        print(f"{fig_fname} saved.")


if __name__ == "__main__":
    app()
