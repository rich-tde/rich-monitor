"""
Automatic diagnostics for RICH TDE simulation snapshots.
"""

import glob
import json
import os
import re
from typing import List, Optional

import h5py
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import typer
import unyt as u
from loguru import logger

import richio

app = typer.Typer()

# --------------------------------- Constants -------------------------------- #

T_COMPTON = u.unyt_quantity(1e8, "K")
RHO_FLOOR = u.unyt_quantity(1e-19, "g/cm**3")
RHO_VIZ_CUT = u.unyt_quantity(1e-18, "g/cm**3")
CACHE_FNAME = "diagnostics_cache.json"
LOG_FNAME = "diagnostics.log"

# --------------------------------- Utilities -------------------------------- #


def _snap_num(path: str) -> int:
    """Extract the snapshot index from a file path.

    Matches patterns ``snap_<N>.h5`` and ``snap_full_<N>.h5``.

    :param path: File path of the snapshot.
    :type path: str
    :returns: Zero-based snapshot index, or ``-1`` if no match is found.
    :rtype: int
    """
    m = re.search(r"snap_(\d+)", path)
    if not m:
        m = re.search(r"snap_full_(\d+)", path)
    return int(m.group(1)) if m else -1


def _cell_mass(snap) -> u.unyt_array:
    """Compute per-cell mass as density * volume.

    :param snap: Loaded RICH snapshot object.
    :returns: Per-cell mass array in grams.
    :rtype: :class:`unyt.unyt_array`
    """
    return snap.density * snap.volume


def _fluff_mask(snap) -> np.ndarray:
    """Boolean mask selecting background / density-floor cells.

    A cell is considered *fluff* when its density is below :data:`RHO_FLOOR`
    **and** its stellar-material tracer differs from unity by more than
    ``1e-3`` (i.e. it carries negligible stellar content).

    :param snap: Loaded RICH snapshot object.
    :returns: Boolean array of shape ``(N,)``; ``True`` for fluff cells.
    :rtype: :class:`numpy.ndarray`
    """
    return (snap.density < RHO_FLOOR) & (np.abs(snap.star - 1) > 1e-3)


def _get_box(snap) -> u.unyt_array:
    """Compute an axis-aligned bounding box for visualisation.

    Cells with density above :data:`RHO_VIZ_CUT` define the extent; each
    bound is then padded by 20 % outward so the domain boundary is never
    clipped in slice / projection plots.

    :param snap: Loaded RICH snapshot object.
    :returns: Array ``[xlo, ylo, zlo, xhi, yhi, zhi]`` in the same length
              units as the snapshot coordinates.
    :rtype: :class:`unyt.unyt_array`
    """
    mask = snap.density > RHO_VIZ_CUT
    X, Y, Z = snap.X[mask], snap.Y[mask], snap.Z[mask]

    def pad(lo, hi):
        lo = lo * (1.2 if lo < 0 else 0.8)
        hi = hi * (1.2 if hi > 0 else 0.8)
        return lo, hi

    xlo, xhi = pad(X.min(), X.max())
    ylo, yhi = pad(Y.min(), Y.max())
    zlo, zhi = pad(Z.min(), Z.max())
    return u.unyt_array([xlo, ylo, zlo, xhi, yhi, zhi], X.units)


def _savefig(fig, path: str, dpi: int = 200):
    """Save a matplotlib figure to *path* and close it.

    :param fig: Figure to save.
    :type fig: :class:`matplotlib.figure.Figure`
    :param path: Destination file path (extension determines format).
    :type path: str
    """
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.success("Saved {}", path)


# ------------------------------ File integrity ------------------------------ #

# Non-particle metadata fields: fixed sizes, not subject to the N-length check.
_METADATA_KEYS = {"Box", "Cycle", "Time"}
# Fields documented as unused in RICH; all-zeros is expected.
_KNOWN_ZERO = {"Eg_0"}


def integrity_check(snap_path: str) -> list[str]:
    """Validate particle fields in an HDF5 snapshot file.

    Checks that:

    * Every particle field has the same length *N*.
    * No field (outside :data:`_KNOWN_ZERO`) is identically zero.
    * No field contains ``NaN`` or ``Inf`` values.

    Metadata keys in :data:`_METADATA_KEYS` (scalars such as ``Box``,
    ``Cycle``, ``Time``) are skipped.

    :param snap_path: Path to the ``.h5`` snapshot file.
    :type snap_path: str
    :returns: List of human-readable issue strings; empty if all fields pass.
    :rtype: list[str]
    """
    logger.info("Integrity check: {}", snap_path)
    issues = []
    n_ref = None  # set once from first particle field

    snap = richio.load(snap_path)
    for key in snap.keys():
        if key in _METADATA_KEYS:
            continue  # scalars / metadata - skip
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


# --------------------------- Slices & projections --------------------------- #


def slice_proj_check(snap, output_dir: str, snap_num: int):
    """Generate mid-plane slice and column-projection plots in one 6-panel figure.

    Produces a 2×3 figure: top row = xy-plane slices of density, temperature,
    and dissipation; bottom row = column-density projection, dissipation
    projection, and an empty panel.  Figure is written to
    ``<output_dir>/figs/slice_proj_snap<NNNN>.png``.

    :param snap: Loaded RICH snapshot object.
    :param output_dir: Root output directory; a ``figs/`` sub-directory must
                       already exist (or be created beforehand).
    :type output_dir: str
    :param snap_num: Snapshot index, used for output filenames.
    :type snap_num: int
    """
    logger.info("Slice & projection plots...")
    box = _get_box(snap)
    t_day = snap.time.to("day")

    slice_panels = [
        ("density", {"label_latex": r"\rho", "cmap": "twilight"}),
        ("temperature", {"label_latex": "T", "cmap": "inferno"}),
        (
            "dissipation",
            {
                "label_latex": r"\dot{E}_\mathrm{diss}",
                "unit_latex": r"\mathrm{erg\,s^{-1}\,cm^{-3}}",
                "cmap": "viridis",
            },
        ),
    ]
    proj_panels = [
        ("density", {"label_latex": r"\Sigma", "cmap": "twilight"}),
        (
            "dissipation",
            {
                "label_latex": r"\int\dot{E}_\mathrm{diss}\,dz",
                "cmap": "viridis",
                "vmin": 14,
                "vmax": 19,
            },
        ),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(23, 10))
    fig.suptitle(f"Slices & Projections  t = {t_day:.2f}", fontsize=14)

    for ax, (field, kw) in zip(axes[0], slice_panels):
        snap.plots.slice(
            data=field,
            res=512,
            X="X",
            Y="Y",
            Z="Z",
            plane="xy",
            slice_coord=0,
            box_size=box,
            ax=ax,
            **kw,
        )
        ax.set_aspect("equal")
        ax.set_title(f"{field} slice")

    for ax, (field, kw) in zip(axes[1], proj_panels):
        snap.plots.projection(
            data=field, res=512, X="X", Y="Y", Z="Z", box_size=box, ax=ax, **kw
        )
        ax.set_aspect("equal")
        ax.set_title(f"{field} projection")

    axes[1, 2].set_visible(False)

    _savefig(fig, os.path.join(output_dir, f"figs/slice_proj_snap{snap_num:04d}.png"))


# ---------------------------- Pericenter zoom-in ---------------------------- #


def _parse_run_params(path: str) -> tuple:
    """Extract (R_star [Rsun], Mstar [Msun], Mbh [Msun], beta) from a path."""
    float_capture = r"([+-]?[0-9]*[.]?[0-9]+)"
    m = re.search(r"R{0}M{0}BH{0}beta{0}".format(float_capture), path)
    if not m:
        raise ValueError(f"Cannot parse TDE params from path: {path}")
    R, Mstar, Mbh, beta = (float(x) for x in m.group(1, 2, 3, 4))
    return R, Mstar, Mbh, beta


def pericenter_check(snap, input_file: str, output_dir: str, snap_num: int):
    """Zoom-in slice plots centred on the pericenter region.

    Produces a 2×2 panel figure (density, temperature, dissipation, sound speed)
    with the box spanning ``x ∈ [-0.5 rp, 2.5 rp]``, ``y ∈ [-1.5 rp, 1.5 rp]``.
    The pericenter distance is derived from run parameters in the path.

    :param snap: Loaded RICH snapshot object.
    :param input_file: Path to the snapshot (used to parse run params).
    :type input_file: str
    :param output_dir: Root output directory; ``figs/`` sub-directory must exist.
    :type output_dir: str
    :param snap_num: Snapshot index used in the output filename.
    :type snap_num: int
    """
    logger.info("Pericenter zoom-in check...")

    # R, Mstar, Mbh are in solar units → rp is in R☉ (= code_length) directly
    R_rsun, Mstar_msun, Mbh_msun, beta = _parse_run_params(input_file)
    rp = R_rsun * (Mbh_msun / Mstar_msun) ** (1.0 / 3.0) / beta * richio.units.lscale
    r0 = 0.6 * rp  # smoothing length

    t = snap.time

    box = [
        -0.5 * rp,
        -1.5 * rp,
        snap.box[5],
        2.5 * rp,
        1.5 * rp,
        snap.box[2],
    ]

    panels = [
        ("density", {"label_latex": r"\rho", "cmap": "twilight"}),
        ("temperature", {"label_latex": "T", "cmap": "inferno"}),
        ("dissipation", {"label_latex": r"\dot{E}_\mathrm{diss}", "cmap": "viridis"}),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(
        rf"Pericenter zoom-in  $r_p={rp.v:.3g}\,R_\odot$  t = {t.to('day').v:.2f}",
        fontsize=13,
    )

    for ax, (field, kw) in zip(axes.flat[:3], panels):
        snap.plots.slice(
            data=field,
            res=512,
            X="X",
            Y="Y",
            Z="Z",
            plane="xy",
            slice_coord=0,
            box_size=box,
            ax=ax,
            **kw,
        )
        ax.set_title(f"{field} slice")

    # Sound speed: compute in code units, convert to km/s once for visualization
    ax_cs = axes.flat[3]
    gamma_eff = snap.pressure / (snap.density * snap.internal_energy) + 1.0
    cs = np.sqrt(np.abs(gamma_eff * snap.pressure / snap.density))
    snap.plots.slice(
        data=cs,
        res=512,
        X="X",
        Y="Y",
        Z="Z",
        plane="xy",
        slice_coord=0,
        box_size=box,
        ax=ax_cs,
        label_latex=r"c_s",
        unit_latex=r"\mathrm{cm\,s^{-1}}",
        cmap="plasma",
    )
    ax_cs.set_title("sound speed slice")

    for ax in axes.flat:
        for radius, label in [(r0.v, r"$r_0$"), (rp.v, r"$r_p$")]:
            ax.add_patch(
                mpatches.Circle(
                    (0, 0),
                    radius,
                    fill=False,
                    linestyle="--",
                    color="white",
                    linewidth=1,
                    zorder=5,
                )
            )
            ax.annotate(
                label,
                xy=(0, radius),
                color="white",
                fontsize=9,
                ha="center",
                va="bottom",
                zorder=6,
            )
        ax.set_xlim(box[0].v, box[3].v)
        ax.set_ylim(box[1].v, box[4].v)

    _savefig(
        fig,
        os.path.join(output_dir, f"figs/pericenter_snap{snap_num:04d}.png"),
        dpi=400,
    )


# ----------------------------- Resolution check ----------------------------- #


def resolution_check(snap, output_dir: str, snap_num: int):
    """Plot cell-size distributions (by count and by mass).

    Uses the cube root of cell volume as a proxy for the smoothing length *h*
    and produces a two-panel histogram (cell count and mass-weighted) saved to
    ``<output_dir>/figs/resolution_check_snap<NNNN>.png``.  Summary statistics
    (min / median / max *h*) are written to the log.

    :param snap: Loaded RICH snapshot object.
    :param output_dir: Root output directory.
    :type output_dir: str
    :param snap_num: Snapshot index, used for output filenames.
    :type snap_num: int
    """
    logger.info("Resolution check...")
    h = snap.volume ** (1 / 3)  # cell-size proxy in code_length (R☉)
    mass = _cell_mass(snap)

    fig, axes = plt.subplots(2, 1, figsize=(7, 7), sharex=True, constrained_layout=True)

    log_h = np.log10(h.v)
    bins = np.linspace(log_h.min(), log_h.max(), 101)

    axes[0].hist(log_h, bins=bins, color="steelblue", histtype="step")
    axes[0].set_ylabel("Cell count")
    axes[0].set_yscale("log")
    axes[0].set_title("Cell-size distribution")

    axes[1].hist(
        log_h,
        bins=bins,
        weights=(mass / mass.sum()).v,
        color="steelblue",
        histtype="step",
    )
    axes[1].set_xlabel(r"$\log_{10}(h\ [\mathrm{R}_\odot])$")
    axes[1].set_ylabel("Mass fraction")
    axes[1].set_yscale("log")
    axes[1].set_title("Mass-weighted cell-size distribution")

    logger.info(
        "  h: min={:.3e}  median={:.3e}  max={:.3e}",
        h.min(),
        np.median(h),
        h.max(),
    )
    _savefig(
        fig, os.path.join(output_dir, f"figs/resolution_check_snap{snap_num:04d}.png")
    )


# -------------------- Conservation & global energy budget ------------------- #


def conservation_check(snap) -> dict:
    """Compute global conserved quantities for a single snapshot.

    Calculates total mass, kinetic / thermal / radiation energy components,
    and the three components of angular momentum.  Results are logged at INFO
    level and returned as a plain dictionary for caching or further analysis.

    :param snap: Loaded RICH snapshot object.
    :returns: Dictionary with keys:

              * ``M_tot_g`` - total mass [g]
              * ``E_kin_erg`` - total kinetic energy [erg]
              * ``E_thm_erg`` - total thermal energy [erg]
              * ``E_rad_erg`` - total radiation energy [erg]
              * ``Lx``, ``Ly``, ``Lz`` - angular-momentum components [g cm² s⁻¹]
    :rtype: dict
    """
    logger.info("Conservation / global budget...")
    mass = _cell_mass(snap)

    M = mass.sum().to("g")
    Ek = (0.5 * mass * (snap.Vx**2 + snap.Vy**2 + snap.Vz**2)).sum().to("erg")
    Et = (mass * snap.InternalEnergy).sum().to("erg")
    Er = (mass * snap.Erad).sum().to("erg")
    Lx = (mass * (snap.Y * snap.Vz - snap.Z * snap.Vy)).sum().to("g*cm**2/s")
    Ly = (mass * (snap.Z * snap.Vx - snap.X * snap.Vz)).sum().to("g*cm**2/s")
    Lz = (mass * (snap.X * snap.Vy - snap.Y * snap.Vx)).sum().to("g*cm**2/s")

    logger.info("  M_total       = {:.4e}", M)
    logger.info("  E_kinetic     = {:.4e}", Ek)
    logger.info("  E_thermal     = {:.4e}", Et)
    logger.info("  E_radiation   = {:.4e}", Er)
    logger.info("  E_total       = {:.4e}", Ek + Et + Er)
    logger.info("  L = ({:.3e}, {:.3e}, {:.3e})", Lx, Ly, Lz)

    return dict(
        M_tot_g=float(M.v),
        E_kin_erg=float(Ek.v),
        E_thm_erg=float(Et.v),
        E_rad_erg=float(Er.v),
        Lx=float(Lx.v),
        Ly=float(Ly.v),
        Lz=float(Lz.v),
    )


# --------------------- Compton cooling: hot-gas fraction -------------------- #


def compton_check(snap) -> float:
    """Compute the hot-gas mass fraction as a Compton-cooling diagnostic.

    Sums the mass of all cells with temperature above :data:`T_COMPTON`
    (10⁸ K) and divides by the total mass.  A fraction exceeding 5 % triggers
    a logged warning indicating that Compton cooling may be ineffective.

    :param snap: Loaded RICH snapshot object.
    :returns: Hot-gas mass fraction ∈ [0, 1].
    :rtype: float
    """
    logger.info("Compton cooling check  (T > {})...", T_COMPTON)
    mass = _cell_mass(snap)
    frac = (mass[snap.temperature > T_COMPTON].sum() / mass.sum()).v
    logger.info("  Hot-gas mass fraction: {}  ({} %)", frac, frac * 100)
    if frac > 0.05:
        logger.warning("  > 5 % - Compton cooling may not be effective!")
    return float(frac)


# ------------------- Smoothing-length approximation check ------------------- #


def smoothing_check(snap) -> dict:
    """Assess the validity of the smoothing-length approximation.

    Computes the fractions of total mass and total dissipation residing in
    cells whose size *h* (cube root of volume) exceeds the median cell size.
    Small fractions indicate that most mass and energy dissipation occur in
    well-resolved regions, justifying the gradient / smoothing-length
    approximation used by RICH.

    :param snap: Loaded RICH snapshot object.
    :returns: Dictionary with keys:

              * ``f_mass_large_h`` - mass fraction in cells with h > median h
              * ``f_diss_large_h`` - dissipation fraction in cells with h > median h
    :rtype: dict
    """
    logger.info("Smoothing-length approximation check...")
    h = snap.volume ** (1 / 3)
    mass = _cell_mass(snap)
    diss_w = snap.dissipation * snap.volume

    h_med = np.median(h)
    large = h > h_med

    f_mass = (mass[large].sum() / mass.sum()).v
    f_diss = (diss_w[large].sum() / diss_w.sum()).v

    logger.info("  Median cell size h = {:.3e}", h_med)
    logger.info("  Mass fraction  in h > h_median : {:.4f}", f_mass)
    logger.info("  Diss fraction  in h > h_median : {:.4f}", f_diss)
    return dict(f_mass_large_h=float(f_mass), f_diss_large_h=float(f_diss))


# --------------- Time-evolution: dissipation & fluff fraction --------------- #


def _scalars_for_snap(snap_path: str) -> dict:
    """Extract all time-series scalar quantities from a single snapshot.

    Loads the snapshot, computes conserved quantities and diagnostic fractions,
    and returns them as plain Python floats suitable for JSON serialisation.
    No figures are produced.

    :param snap_path: Path to the ``.h5`` snapshot file.
    :type snap_path: str
    :returns: Dictionary with keys ``time_s``, ``M_tot_g``, ``E_kin_erg``,
              ``E_thm_erg``, ``E_rad_erg``, ``Lx_cgs``, ``Ly_cgs``,
              ``Lz_cgs``, ``diss_total_erg_s``, ``diss_fluff_frac``,
              ``hot_mass_frac``.
    :rtype: dict
    """
    snap = richio.load(snap_path)
    mass = _cell_mass(snap)
    diss_w = snap.dissipation * snap.volume
    fluff = _fluff_mask(snap)

    return dict(
        time_s=float(snap.time.to("s").v),
        M_tot_g=float(mass.sum().to("g").v),
        E_kin_erg=float(
            (0.5 * mass * (snap.Vx**2 + snap.Vy**2 + snap.Vz**2)).sum().to("erg").v
        ),
        E_thm_erg=float((mass * snap.InternalEnergy).sum().to("erg").v),
        E_rad_erg=float((mass * snap.Erad).sum().to("erg").v),
        diss_total_erg_s=float(diss_w.sum().to("erg/s").v),
        diss_fluff_frac=float((diss_w[fluff].sum() / diss_w.sum()).v),
        hot_mass_frac=float((mass[snap.temperature > T_COMPTON].sum() / mass.sum()).v),
    )


def time_evolution_check(snap_path: str, output_dir: str):
    """Build or update the scalar cache and produce time-evolution plots.

    Scans the directory containing *snap_path* for all ``snap_*.h5`` files,
    computes :func:`_scalars_for_snap` for any snapshot not yet in the JSON
    cache (:data:`CACHE_FNAME`), and saves the updated cache.  Then generates
    two multi-panel figures:

    * ``time_evolution_physics.png`` - energy budget and total dissipation rate.
    * ``time_evolution_numerics.png`` - mass conservation, fluff dissipation
      fraction, and Compton cooling check.

    :param snap_path: Path to any ``.h5`` snapshot in the run directory (used
                      to locate sibling snapshots).
    :type snap_path: str
    :param output_dir: Root output directory; figures are written to the
                       ``figs/`` sub-directory.
    :type output_dir: str
    """
    logger.info("Time-evolution checks...")
    cache_path = os.path.join(output_dir, CACHE_FNAME)
    cache: dict = {}
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            cache = json.load(f)

    snap_dir = os.path.dirname(snap_path)
    all_snaps = sorted(glob.glob(os.path.join(snap_dir, "snap_*.h5")))  # TODO: for all

    for sp in all_snaps:
        key = str(_snap_num(sp))
        if key in cache:
            continue
        logger.info("  Computing scalars for snap_{}", key)
        cache[key] = _scalars_for_snap(sp)

    with open(cache_path, "w") as f:
        json.dump(cache, f, indent=2)
    logger.success("Cache saved: {}", cache_path)

    # ------------------------- Plot all cached snapshots ------------------------ #
    keys = sorted(cache.keys(), key=int)
    t = np.array([cache[k]["time_s"] for k in keys]) / 86400  # → days
    M = np.array([cache[k]["M_tot_g"] for k in keys])
    Ek = np.array([cache[k]["E_kin_erg"] for k in keys])
    Et = np.array([cache[k]["E_thm_erg"] for k in keys])
    Er = np.array([cache[k]["E_rad_erg"] for k in keys])
    # Lx   = np.array([cache[k]["Lx_cgs"]           for k in keys])
    # Ly   = np.array([cache[k]["Ly_cgs"]           for k in keys])
    # Lz   = np.array([cache[k]["Lz_cgs"]           for k in keys])
    D = np.array([cache[k]["diss_total_erg_s"] for k in keys])
    Df = np.array([cache[k]["diss_fluff_frac"] for k in keys])
    Th = np.array([cache[k]["hot_mass_frac"] for k in keys])

    _tl = "Time [day]"

    # ----------------------------- Figure 1: physics ---------------------------- #
    fig1, ax1 = plt.subplots(2, 1, figsize=(8, 7), sharex=True, constrained_layout=True)

    ax1[0].plot(t, Ek, "C1", label="kinetic")
    ax1[0].plot(t, Et, "C2", label="thermal")
    ax1[0].plot(t, Er, "C3", label="radiation")
    ax1[0].plot(t, Ek + Et + Er, "k", lw=0.8, ls="--", label="total")
    ax1[0].legend(fontsize=7)
    ax1[0].set_ylabel("Energy [erg]")
    ax1[0].set_title("Energy budget")
    ax1[0].set_yscale("log")

    ax1[1].plot(t, np.log10(D), "C4")
    ax1[1].set_ylabel(r"$\log_{10}\dot{E}_\mathrm{diss}$  [erg/s]")
    ax1[1].set_title("Total dissipation rate")
    ax1[1].set_xlabel(_tl)

    _savefig(fig1, os.path.join(output_dir, "figs/time_evolution_physics.png"))

    # ------------------------ Figure 2: numerical checks ------------------------ #
    fig2, ax2 = plt.subplots(3, 1, figsize=(8, 9), sharex=True, constrained_layout=True)

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

    _savefig(fig2, os.path.join(output_dir, "figs/time_evolution_numerics.png"))


# ----------------------------------- Main ----------------------------------- #

_ALL_CHECKS = (
    "integrity",
    "resolution",
    "conservation",
    "compton",
    "time_evolution",
    "slices",
    "pericenter",
)


@app.command()
def main(
    input_file: str = typer.Argument(
        help="Input .h5 snapshot.",
    ),
    output_dir: str = typer.Argument(
        default="./",
        help="Output directory for figures and cache.",
    ),
    checks: Optional[List[str]] = typer.Option(
        None,
        "--check",
        "-c",
        help=(
            "Run only the specified check(s). Repeat for multiple. "
            f"Choices: {', '.join(_ALL_CHECKS)}. "
            "Default: run all."
        ),
    ),
):
    """Run RICH TDE snapshot diagnostics.

    Examples:
      python diagnostics.py snap.h5               # run all checks
      python diagnostics.py snap.h5 -c integrity  # integrity only
      python diagnostics.py snap.h5 -c time_evolution -c slices
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
    os.makedirs(os.path.join(output_dir, "figs"), exist_ok=True)
    snap_num = _snap_num(input_file)

    log_path = os.path.join(output_dir, LOG_FNAME)
    log_sink = logger.add(log_path, mode="a")

    logger.info("=== Diagnostics snap_{} : {} ===", snap_num, input_file)

    if "integrity" in run:
        integrity_check(input_file)

    # Load snapshot only if needed for any remaining check
    _needs_snap = run & {
        "resolution",
        "conservation",
        "compton",
        "slices",
        "pericenter",
    }
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

    if "pericenter" in run:
        pericenter_check(snap, input_file, output_dir, snap_num)

    if "time_evolution" in run:
        time_evolution_check(input_file, output_dir)

    logger.success("=== Done snap_{} ===", snap_num)
    logger.remove(log_sink)


if __name__ == "__main__":
    app()
