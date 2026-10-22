#!/usr/bin/env python3
"""Estimate wall time and Snellius core-hour cost for configured TDE runs.

SYNOPSIS
    python scripts/cost_estimate.py [-h | --help]

DESCRIPTION
    Compute estimates for the RUNS list defined in this file. Print per-run
    step count, fallback time, timestep, cell count, wall time, and SBU,
    followed by the total SBU. No files are read or written.

ARGUMENTS
    None. Simulation parameters are configured in this file.

OPTIONS
    -h, --help
        Show command help and exit without calculating estimates.

    No other command-line options are accepted.

EXIT STATUS
    0   Estimates printed, or help requested.
    2   Unrecognized command-line option or positional argument.
    Other nonzero values can result from unhandled runtime errors.

NOTES
    Edit these Python constants to change assumptions:
        NCORES          Core count (default: 960).
        SPEED           Cell updates/core/second (default: 2500).
        TIMESTEP        Timestep in code units (default: 0.0010).
        NFALLBACKTIME   Simulated fallback times (default: 2).

    Each entry in RUNS supplies:
        label           Display name.
        Mbh, M          Black-hole and stellar masses, in solar masses.
        R               Stellar radius, in solar radii.
        beta            Penetration factor.
        ncells          Estimated cell count including AMR refinement.
        dt              Code-unit timestep; defaults to TIMESTEP.

    Code units use one solar mass, one solar radius, and TSCALE = 1603 s.
    Printed fallback times are in days, wall times in 30-day months, and
    SBU in core hours. Cell counts and sustained speed are input estimates.

EXAMPLES
    Print estimates for all configured runs, from the repository root:
        python scripts/cost_estimate.py
"""

import argparse
import math
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TSCALE = 1603.0  # seconds per code time unit (so G = 1)
NCORES = 960
SPEED = 2500  # cell-updates / core / second (observed sustained, healthy state)
TIMESTEP = 0.0010
NFALLBACKTIME = 2

SEC_PER_DAY = 86400
SEC_PER_MONTH = 30 * SEC_PER_DAY


# ---------------------------------------------------------------------------
# Run definition (all in code units)
# ---------------------------------------------------------------------------


@dataclass
class Run:
    label: str
    Mbh: float  # solar masses
    M: float  # solar masses
    R: float  # solar radii
    beta: float
    ncells: float  # AMR-inflated, hand-set
    dt: float = TIMESTEP  # timestep in code units

    def t_peri(self) -> float:
        Rp = self.R * (self.Mbh / self.M) ** (1 / 3) / self.beta
        return 2 * math.pi * math.sqrt(Rp**3 / self.Mbh)

    def t_apo(self) -> float:
        apo = self.R * (self.Mbh / self.M) ** (2 / 3)
        return 6 * math.sqrt(apo**3 / self.Mbh)

    def t_fb(self) -> float:
        return (
            40
            * (self.Mbh / 1e6) ** 0.5
            * self.M ** (-1)
            * self.R**1.5
            * SEC_PER_DAY
            / TSCALE
        )

    def nsteps(self) -> float:
        return NFALLBACKTIME * self.t_fb() / self.dt


# R from main-sequence M-R relation R ~ M^0.8.
# ncells values are hand-set; the Mbh=1e5/M=0.5 row is anchored to the
# observed ~15M cells in the calibration run (job 21659411 / 21732963).
RUNS = [
    Run("Mbh=1e6, M*=1,   β=2", Mbh=1e6, M=1.0, R=1.00, beta=2, ncells=4e7),
    Run("Mbh=1e6, M*=0.5, β=1", Mbh=1e6, M=0.5, R=0.47, beta=1, ncells=4e7),
    Run("Mbh=1e5, M*=0.5, β=1", Mbh=1e5, M=0.5, R=0.47, beta=1, ncells=3e7),
    Run("Mbh=1e5, M*=1,   β=2", Mbh=1e5, M=1.0, R=1.00, beta=2, ncells=2e7),
    Run("Mbh=1e4, M*=1,   β=2", Mbh=1e4, M=1.0, R=1.00, beta=2, ncells=2e7),
    Run("Mbh=1e4, M*=3,   β=2", Mbh=1e4, M=3.0, R=2.10, beta=2, ncells=2e7),
]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    hdr = (
        f"{'Run':<28} {'Nsteps':>9} {'t_fb[d]':>7} {'dt[code]':>10}"
        f" {'Ncells':>10} {'wall[mo]':>9} {'SBU':>10}"
    )
    print("=" * len(hdr))
    print(hdr)
    print("=" * len(hdr))

    total_sbu = 0.0
    for r in RUNS:
        ns = r.nsteps()
        dt = r.dt
        wall_s = ns * r.ncells / (SPEED * NCORES)
        sbu = wall_s / 3600 * NCORES
        total_sbu += sbu

        print(
            f"{r.label:<28} {ns:>9.2e} {r.t_fb() * TSCALE / SEC_PER_DAY:>7.1f} {dt:>10.2e}"
            f" {r.ncells:>10.1e} {wall_s / SEC_PER_MONTH:>9.2f} {sbu:>10.2e}"
        )

    print("=" * len(hdr))
    print(
        f"{'TOTAL':<28} {''!s:>9} {''!s:>7} {''!s:>10}"
        f" {''!s:>10} {''!s:>9} {total_sbu:>10.2e}"
    )
    print()
    print(
        f"Total {total_sbu:.1e} SBU for {NFALLBACKTIME} fallback times (@ {SPEED} cell/step/core/s, {NCORES} cores)"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Estimate wall time and Snellius core-hour cost for the configured TDE runs.",
        epilog=(
            "ARGUMENTS\n"
            "  None. No simulation-parameter options are accepted.\n\n"
            "CONFIGURATION\n"
            "  Edit RUNS in this script to select simulations. Assumptions:\n"
            f"    NCORES          {NCORES} cores\n"
            f"    SPEED           {SPEED} cell updates/core/second\n"
            f"    TIMESTEP        {TIMESTEP} code time units\n"
            f"    NFALLBACKTIME   {NFALLBACKTIME} fallback times\n\n"
            "OUTPUT\n"
            "  Per-run step count, fallback time (days), timestep (code units),\n"
            "  cell count, wall time (30-day months), and SBU (core hours).\n"
            "  Prints a total SBU; no files are read or written.\n\n"
            "EXAMPLES\n"
            "  python scripts/cost_estimate.py"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.parse_args()
    main()
