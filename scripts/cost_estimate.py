#!/usr/bin/env python3
"""
cost_estimate.py — Cost estimate for the TDE Compton-gray proposal runs.

Everything is computed in code units (1 M_sun, 1 R_sun, TSCALE = 1603 s).
TSCALE is chosen such that G = 1 in code units, so no CGS conversions are
needed for the physics. Only the final wall-time/SBU output uses seconds.

Per-run inputs (see RUNS list):
  Mbh, M, R     -- in solar units
  beta          -- penetration factor
  ncells        -- AMR-inflated cell count; hand-set per run.
                   (Future: derive from Mbh / resolution settings.)
  dt_factor     -- timestep is dt = t_peri / dt_factor  (default 1000).

SBU = core-hours on Snellius.
"""

from dataclasses import dataclass
import math

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
    dt: float = TIMESTEP  # dt = t_peri / dt_factor

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
    main()
