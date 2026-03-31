#!/usr/bin/env python3
"""
Automatic diagnostics for RICH TDE simulation snapshots.
"""
#TODO: write a script that copies this from alice to mac, from mac to kiekpoel

import glob
import json
import os
import re
from typing import List, Optional

import h5py
import matplotlib.pyplot as plt
import numpy as np
import typer
import unyt as u
from loguru import logger

import richio

app = typer.Typer()

# ── Constants ─────────────────────────────────────────────────────────────────
T_COMPTON    = 1e8    # K — Compton cooling threshold
RHO_FLOOR    = 1e-19  # g/cm³ — density floor; cells below this are "fluff"
RHO_VIZ_CUT  = 1e-18  # g/cm³ — cut for auto box-sizing
CACHE_FNAME  = "diagnostics_cache.json"


# ── Utilities ─────────────────────────────────────────────────────────────────

def _snap_num(path: str) -> int:
    m = re.search(r"snap_(\d+)", path)
    if not m:
        m = re.search(r"snap_full_(\d+)", path)
    return int(m.group(1)) if m else -1


def _cell_mass(snap) -> u.unyt_array:
    return (snap.density * snap.volume).to("g")


def _fluff_mask(snap) -> np.ndarray:
    """Background/floor cells: low density, negligible stellar content."""
    return (snap.density.to("g/cm**3").v < RHO_FLOOR) & (np.abs(snap.star - 1) > 1e-3)


def _get_box(snap):
    mask = snap.density.to("g/cm**3").v > RHO_VIZ_CUT
    X, Y, Z = snap.X[mask], snap.Y[mask], snap.Z[mask]
    def pad(lo, hi):
        lo = lo * (1.1 if lo < 0 else 0.9)
        hi = hi * (1.1 if hi > 0 else 0.9)
        return lo, hi
    xlo, xhi = pad(X.min(), X.max())
    ylo, yhi = pad(Y.min(), Y.max())
    zlo, zhi = pad(Z.min(), Z.max())
    return u.unyt_array([xlo, ylo, zlo, xhi, yhi, zhi], X.units)


def _savefig(fig, path: str):
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.success("Saved {}", path)


# ── File integrity ─────────────────────────────────────────────────────────

# Non-particle metadata fields: fixed sizes, not subject to the N-length check.
_METADATA_KEYS = {"Box", "Cycle", "Time"}
# Fields documented as unused in RICH; all-zeros is expected.
_KNOWN_ZERO    = {"Eg_0"}

def integrity_check(snap_path: str) -> list[str]:
    """All particle fields: same length N, no all-zero / NaN / Inf."""
    logger.info("Integrity check: {}", snap_path)
    issues = []
    n_ref = None                             # set once from first particle field

    snap = richio.load(snap_path)
    for key in snap.keys():
        if key in _METADATA_KEYS:
            continue                         # scalars / metadata — skip
        arr = snap[key]
        if arr.ndim == 0:
            continue
        n = arr.shape[0]
        if n_ref is None:
            n_ref = n
        elif n != n_ref:
            issues.append(f"{key}: length {n} ≠ {n_ref}")
        if key not in _KNOWN_ZERO and np.all(arr == 0):
            issues.append(f"{key}: all zeros")
        if np.any(~np.isfinite(np.asarray(arr).astype(float))):
            issues.append(f"{key}: contains NaN/Inf")

    if issues:
        for iss in issues:
            logger.warning("  {}", iss)
    else:
        logger.success("  All fields OK  (N = {})", n_ref)
    return issues


# ── Slices & projections ───────────────────────────────────────────────────

def slice_proj_check(snap, output_dir: str, snap_num: int):
    logger.info("Slice & projection plots...")
    box = _get_box(snap)
    t_day = snap.time.to("day")

    kwfield2slice = {
        "density":     {"label_latex": r"\rho", "cmap": "twilight"},
        "temperature": {"label_latex": "T", "cmap": "inferno"},
        "dissipation": {"label_latex": r"\dot{E}_\mathrm{diss}",
                        "unit_latex":  r"\mathrm{erg\,s^{-1}\,cm^{-3}}",
                        "cmap": "viridis"},
    }
    kwfield2proj = {
        "density":     {"label_latex":r"\Sigma", "cmap": "twilight"},
        "dissipation": {"label_latex":r"\int\dot{E}_\mathrm{diss}\,dz",
                        "cmap":"viridis", "vmin": 14, "vmax": 19}
    }

    for field, kw in kwfield2slice.items():
        fig, ax = plt.subplots()
        snap.plots.slice(
            data=field, res=512, X="X", Y="Y", Z="Z",
            plane="xy", slice_coord=0, box_size=box,
            ax=ax, **kw,
        )
        plt.title(f"{field} slice {t_day:.2f}")
        _savefig(fig, os.path.join(output_dir, f"{field}_slice_snap{snap_num:04d}.png"))

    for field, kw in kwfield2proj.items():
        fig, ax = plt.subplots()
        snap.plots.projection(
            data=field, res=512, X="X", Y="Y", Z="Z",
            box_size=box, ax=ax, **kw
        )
        plt.title(f"{field} projection {t_day:.2f}")
        _savefig(fig, os.path.join(output_dir, f"{field}_proj_snap{snap_num:04d}.png"))


# ── Resolution check ───────────────────────────────────────────────────────

def resolution_check(snap, output_dir: str, snap_num: int):
    logger.info("Resolution check...")
    h    = snap.volume.v ** (1 / 3)   # cell-size proxy [cm]
    mass = _cell_mass(snap).v

    fig, axes = plt.subplots(2, 1, figsize=(7, 7), sharex=True,
                             constrained_layout=True)

    log_h = np.log10(h)
    bins  = np.linspace(log_h.min(), log_h.max(), 101)

    axes[0].hist(log_h, bins=bins, color="steelblue", histtype="step")
    axes[0].set_ylabel("Cell count")
    axes[0].set_yscale("log")
    axes[0].set_title("Cell-size distribution")

    axes[1].hist(log_h, bins=bins, weights=mass / mass.sum(),
                 color="steelblue", histtype="step")
    axes[1].set_xlabel(r"$\log_{10}(h\ [\mathrm{R}_\odot])$")
    axes[1].set_ylabel("Mass fraction")
    axes[1].set_yscale("log")
    axes[1].set_title("Mass-weighted cell-size distribution")

    logger.info(
        "  h: min={:.3e}  median={:.3e}  max={:.3e}  [cm]",
        h.min(), np.median(h), h.max(),
    )
    _savefig(fig, os.path.join(output_dir, f"resolution_check_snap{snap_num:04d}.png"))


# ── Conservation & global energy budget ────────────────────────────────────

def conservation_check(snap) -> dict:
    """Total mass, kinetic / thermal / radiation energy, angular momentum."""
    logger.info("Conservation / global budget...")
    mass = _cell_mass(snap)                              # [g]
    vx   = snap.Vx.to("cm/s");  vy = snap.Vy.to("cm/s");  vz = snap.Vz.to("cm/s")
    ie   = snap.InternalEnergy.to("erg/g")
    erad = snap.Erad.to("erg/g")
    x    = snap.X.to("cm");  y = snap.Y.to("cm");  z = snap.Z.to("cm")

    M   = mass.sum().to("g")
    Ek  = (0.5 * mass * (vx**2 + vy**2 + vz**2)).sum().to("erg")
    Et  = (mass * ie).sum().to("erg")
    Er  = (mass * erad).sum().to("erg")
    Lx  = (mass * (y * vz - z * vy)).sum().to("g*cm**2/s")
    Ly  = (mass * (z * vx - x * vz)).sum().to("g*cm**2/s")
    Lz  = (mass * (x * vy - y * vx)).sum().to("g*cm**2/s")

    logger.info("  M_total       = {:.4e}", M)
    logger.info("  E_kinetic     = {:.4e}", Ek)
    logger.info("  E_thermal     = {:.4e}", Et)
    logger.info("  E_radiation   = {:.4e}", Er)
    logger.info("  E_total       = {:.4e}", Ek + Et + Er)
    logger.info("  L = ({:.3e}, {:.3e}, {:.3e})", Lx, Ly, Lz)

    return dict(
        M_tot_g=float(M.v),
        E_kin_erg=float(Ek.v), E_thm_erg=float(Et.v), E_rad_erg=float(Er.v),
        Lx=float(Lx.v), Ly=float(Ly.v), Lz=float(Lz.v),
    )


# ── Compton cooling: hot-gas fraction ──────────────────────────────────────

def compton_check(snap) -> float:
    """Mass fraction above T_COMPTON; should be small if Compton cooling works."""
    logger.info("Compton cooling check  (T > {:.0e} K)...", T_COMPTON)
    mass = _cell_mass(snap).v
    T    = snap.temperature.to("K").v
    frac = mass[T > T_COMPTON].sum() / mass.sum()
    logger.info("  Hot-gas mass fraction: {}  ({} %)", frac, frac * 100)
    if frac > 0.05:
        logger.warning("  > 5 % — Compton cooling may not be effective!")
    return float(frac)


# ── Smoothing-length approximation check ───────────────────────────────────

def smoothing_check(snap) -> dict:
    """
    Mass and dissipation fraction in cells larger than the median cell size.
    If small, the gradient/smoothing-length approximation is well-justified.
    """
    logger.info("Smoothing-length approximation check...")
    h      = snap.volume.to("cm**3").v ** (1 / 3)
    mass   = _cell_mass(snap).v
    diss_w = (snap.dissipation * snap.volume).to("erg/s").v

    h_med  = np.median(h)
    large  = h > h_med

    f_mass = mass[large].sum() / mass.sum()
    f_diss = diss_w[large].sum() / diss_w.sum()

    logger.info("  Median cell size h = {:.3e} cm", h_med)
    logger.info("  Mass fraction  in h > h_median : {:.4f}", f_mass)
    logger.info("  Diss fraction  in h > h_median : {:.4f}", f_diss)
    return dict(f_mass_large_h=float(f_mass), f_diss_large_h=float(f_diss))


# ── Time-evolution: dissipation & fluff fraction ──────────────────────

def _scalars_for_snap(snap_path: str) -> dict:
    """Compute all time-series scalars for one snapshot (no plots)."""
    snap   = richio.load(snap_path)
    mass   = _cell_mass(snap).v
    vx     = snap.Vx.to("cm/s").v;  vy = snap.Vy.to("cm/s").v;  vz = snap.Vz.to("cm/s").v
    x      = snap.X.to("cm").v;     y  = snap.Y.to("cm").v;      z  = snap.Z.to("cm").v
    diss_w = (snap.dissipation.to("erg/s/cm**3") * snap.volume.to("cm**3")).to("erg/s").v
    fluff  = _fluff_mask(snap)

    return dict(
        time_s              = float(snap.time.to("s").v),
        M_tot_g             = float(mass.sum()),
        E_kin_erg           = float((0.5 * mass * (vx**2 + vy**2 + vz**2)).sum()),
        E_thm_erg           = float((mass * snap.InternalEnergy.to("erg/g").v).sum()),
        E_rad_erg           = float((mass * snap.Erad.to("erg/g").v).sum()),
        Lx_cgs              = float((mass * (y * vz - z * vy)).sum()),
        Ly_cgs              = float((mass * (z * vx - x * vz)).sum()),
        Lz_cgs              = float((mass * (x * vy - y * vx)).sum()),
        diss_total_erg_s    = float(diss_w.sum()),
        diss_fluff_frac     = float(diss_w[fluff].sum() / diss_w.sum()),
        hot_mass_frac       = float(mass[snap.temperature.to("K").v > T_COMPTON].sum() / mass.sum()),
    )


def time_evolution_check(snap_path: str, output_dir: str):
    """
    Build/update a JSON cache of per-snapshot scalars, then plot the time series.
    Only processes snapshots not already in the cache.
    """
    logger.info("Time-evolution checks...")
    cache_path = os.path.join(output_dir, CACHE_FNAME)
    cache: dict = {}
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            cache = json.load(f)

    snap_dir  = os.path.dirname(snap_path)
    all_snaps = sorted(glob.glob(os.path.join(snap_dir, "snap_*.h5"))) #TODO: for all

    for sp in all_snaps:
        key = str(_snap_num(sp))
        if key in cache:
            continue
        logger.info("  Computing scalars for snap_{}", key)
        cache[key] = _scalars_for_snap(sp)

    with open(cache_path, "w") as f:
        json.dump(cache, f, indent=2)
    logger.success("Cache saved: {}", cache_path)

    # ── Plot all cached snapshots ────────────────────────────────────────────
    keys = sorted(cache.keys(), key=int)
    t    = np.array([cache[k]["time_s"]           for k in keys]) / 86400  # → days
    M    = np.array([cache[k]["M_tot_g"]          for k in keys])
    Ek   = np.array([cache[k]["E_kin_erg"]        for k in keys])
    Et   = np.array([cache[k]["E_thm_erg"]        for k in keys])
    Er   = np.array([cache[k]["E_rad_erg"]        for k in keys])
    # Lx   = np.array([cache[k]["Lx_cgs"]           for k in keys])
    # Ly   = np.array([cache[k]["Ly_cgs"]           for k in keys])
    # Lz   = np.array([cache[k]["Lz_cgs"]           for k in keys])
    D    = np.array([cache[k]["diss_total_erg_s"] for k in keys])
    Df   = np.array([cache[k]["diss_fluff_frac"]  for k in keys])
    Th   = np.array([cache[k]["hot_mass_frac"]    for k in keys])

    _tl = "Time [day]"

    # ── Figure 1: physics ────────────────────────────────────────────────────
    fig1, ax1 = plt.subplots(2, 1, figsize=(8, 7), sharex=True,
                              constrained_layout=True)

    ax1[0].plot(t, Ek,       "C1", label="kinetic")
    ax1[0].plot(t, Et,       "C2", label="thermal")
    ax1[0].plot(t, Er,       "C3", label="radiation")
    ax1[0].plot(t, Ek+Et+Er, "k",  lw=0.8, ls="--", label="total")
    ax1[0].legend(fontsize=7)
    ax1[0].set_ylabel("Energy [erg]")
    ax1[0].set_title("Energy budget")
    ax1[0].set_yscale('log')

    ax1[1].plot(t, np.log10(D), "C4")
    ax1[1].set_ylabel(r"$\log_{10}\dot{E}_\mathrm{diss}$  [erg/s]")
    ax1[1].set_title("Total dissipation rate")
    ax1[1].set_xlabel(_tl)

    _savefig(fig1, os.path.join(output_dir, "time_evolution_physics.png"))

    # ── Figure 2: numerical checks ───────────────────────────────────────────
    fig2, ax2 = plt.subplots(3, 1, figsize=(8, 9), sharex=True,
                              constrained_layout=True)

    ax2[0].plot(t, M, "C0")
    ax2[0].set_ylim(1e33 * 0.99, 1e33 * 1.01)
    ax2[0].set_ylabel("Total mass [g]")
    ax2[0].set_title("Mass conservation")

    ax2[1].plot(t, Df, "C5")
    ax2[1].set_ylabel("Fluff dissipation fraction")
    ax2[1].set_title("Dissipation by floor cells")

    ax2[2].plot(t, Th, "C6")
    ax2[2].set_ylabel(r"Hot-gas mass fraction  ($T>10^8$ K)")
    ax2[2].set_title("Compton cooling check")
    ax2[2].set_xlabel(_tl)

    _savefig(fig2, os.path.join(output_dir, "time_evolution_numerics.png"))


# ── Main ──────────────────────────────────────────────────────────────────────

_ALL_CHECKS = ("integrity", "resolution", "conservation", "compton", "time_evolution", "slices")


@app.command()
def main(
    input_file: str = typer.Argument(
        default="/data1/projects/pi-rossiem/TDE_data/R0.47M0.5BH10000beta1S60n1.5ComptonNewAMR/snap_372/snap_372.h5",
        help="Input .h5 snapshot.",
    ),
    output_dir: str = typer.Argument(
        default="./",
        help="Output directory for figures and cache.",
    ),
    checks: Optional[List[str]] = typer.Option(
        None, "--check", "-c",
        help=(
            "Run only the specified check(s). Repeat for multiple. "
            f"Choices: {', '.join(_ALL_CHECKS)}. "
            "Default: run all."
        ),
    ),
):
    """Run RICH TDE snapshot diagnostics.

    Examples:
      python auto_diagnostics.py snap.h5               # run all checks
      python auto_diagnostics.py snap.h5 -c integrity  # integrity only
      python auto_diagnostics.py snap.h5 -c time_evolution -c slices
    """
    if checks:
        unknown = set(checks) - set(_ALL_CHECKS)
        if unknown:
            raise typer.BadParameter(
                f"Unknown check(s): {unknown}. Valid: {_ALL_CHECKS}"
            )
        run = set(checks)
    else:
        run = set(_ALL_CHECKS)

    os.makedirs(output_dir, exist_ok=True)
    snap_num = _snap_num(input_file)

    log_path = os.path.join(output_dir, "diagnostics.log")
    log_sink = logger.add(log_path, mode="a", level="DEBUG",
                          format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}")

    logger.info("=== Diagnostics snap_{} : {} ===", snap_num, input_file)

    if "integrity" in run:
        integrity_check(input_file)

    # Load snapshot only if needed for any remaining check
    _needs_snap = run & {"resolution", "conservation", "compton", "slices"}
    snap = None
    if _needs_snap:
        logger.info("Loading snapshot...")
        snap = richio.load(input_file)
        # snap.info()

    if "resolution" in run:
        resolution_check(snap, output_dir, snap_num)

    if "conservation" in run:
        conservation_check(snap)

    if "compton" in run:
        compton_check(snap)

    if "slices" in run:
        slice_proj_check(snap, output_dir, snap_num)

    if "time_evolution" in run:
        time_evolution_check(input_file, output_dir)

    logger.success("=== Done snap_{} ===", snap_num)
    logger.remove(log_sink)


if __name__ == "__main__":
    app()
