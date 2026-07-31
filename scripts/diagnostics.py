"""
Automatic diagnostics for RICH TDE simulation snapshots.
"""

import gc
import glob
import json
import os
import re
from typing import List, Optional

import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import typer
import unyt as u
from loguru import logger

import richio
from richio.plots import scalar_map

app = typer.Typer()

# --------------------------------- Constants -------------------------------- #

T_COMPTON = u.unyt_quantity(1e8, "K")
RHO_FLOOR = u.unyt_quantity(1e-19, "g/cm**3")
CACHE_FNAME = "diagnostics_cache.json"
LOG_FNAME = "diagnostics.log"

# --------------------------------- Utilities -------------------------------- #


def _snap_num(path: str) -> int:
    """Snapshot index from a snap_<N>.h5 / snap_full_<N>.h5 path, or -1."""
    m = re.search(r"snap_(\d+)", path)
    if not m:
        m = re.search(r"snap_full_(\d+)", path)
    return int(m.group(1)) if m else -1


def _cell_mass(snap) -> u.unyt_array:
    """Per-cell mass = density * volume."""
    return snap.density * snap.volume


def _fluff_mask(snap) -> np.ndarray:
    """Background / density-floor cells: rho < RHO_FLOOR and non-stellar tracer."""
    return (snap.density < RHO_FLOOR) & (np.abs(snap.star - 1) > 1e-3)


def _savefig(fig, path: str, dpi: int = 200):
    """Save fig to path and close it, creating the parent directory if needed."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.success("Saved {}", path)


# ------------------------------ File integrity ------------------------------ #

# Non-particle metadata fields: fixed sizes, not subject to the N-length check.
_METADATA_KEYS = {"Box", "Cycle", "Time"}
# Fields documented as unused in RICH; all-zeros is expected.
_KNOWN_ZERO = {"Eg_0"}


def integrity_check(snap_path: str) -> list[str]:
    """Check particle fields share one length N, none are all-zero, none have NaN/Inf."""
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


# ------------------------- Shared nine-panel quantity grid ------------------ #


def _parse_tde_params(path: str) -> tuple:
    """Extract (R_star [Rsun], Mstar [Msun], Mbh [Msun], beta) from a path."""
    float_capture = r"([+-]?[0-9]*[.]?[0-9]+)"
    m = re.search(r"R{0}M{0}BH{0}beta{0}".format(float_capture), path)
    if not m:
        raise ValueError(f"Cannot parse TDE params from path: {path}")
    R, Mstar, Mbh, beta = (float(x) for x in m.group(1, 2, 3, 4))
    return R, Mstar, Mbh, beta


def _scaled_box(input_file: str, radius_kind: str, mult: float) -> tuple:
    """Symmetric cubic box of half-width mult * r, r = r_p or r_a. Returns (box, r)."""
    R_rsun, Mstar_msun, Mbh_msun, beta = _parse_tde_params(input_file)
    if radius_kind == "rp":
        r = R_rsun * (Mbh_msun / Mstar_msun) ** (1.0 / 3.0) / beta * richio.units.lscale
    else:
        r = R_rsun * (Mbh_msun / Mstar_msun) ** (2.0 / 3.0) * richio.units.lscale
    half = mult * r
    return [-half, -half, -half, half, half, half], r


def _nine_panel_fields(snap, input_file: str) -> dict:
    """Full-array fields for the shared 9-panel grid.

    Direct fields (density, pressure, temperature, dissipation, Erad) plus
    derived diagnostics: speed, Mach number, fallback time in units of t_min
    (see analysis/amrtimestep/orbital_energy.ipynb), and the Bernoulli
    parameter ``Be = 1/2 v^2 + P/rho + E_rad/3 - Phi`` (>0 => locally
    unbound, BH at origin) shown as ``sgn(Be) log10(|Be / delta_eps|)`` — the
    standard signed-log transform for a quantity that spans many orders of
    magnitude on both sides of zero — where ``delta_eps = G Mbh Rstar / Rt^2``
    is the frozen-in specific-energy spread of tidally disrupted debris.

    The potential is Paczynski-Wiita, ``Phi = G Mbh / (r - r_g)`` with
    ``r_g = 2 G Mbh / c^2``, which mimics the GR innermost-stable-orbit
    behaviour; cells inside ``r_g`` are set to NaN rather than allowed to
    flip sign.
    """
    R_rsun, Mstar_msun, Mbh_msun, _beta = _parse_tde_params(input_file)
    Mbh = Mbh_msun * richio.units.mscale
    Mstar = Mstar_msun * richio.units.mscale
    Rstar = R_rsun * richio.units.lscale
    Rt = Rstar * (Mbh / Mstar) ** (1.0 / 3.0)  # tidal radius (beta-independent)
    delta_eps = u.G * Mbh * Rstar / Rt**2  # frozen-in energy spread
    r_g = 2 * u.G * Mbh / u.c**2  # gravitational radius

    v_mag = np.sqrt(snap.Vx**2 + snap.Vy**2 + snap.Vz**2)
    gamma_eff = snap.pressure / (snap.density * snap.internal_energy) + 1.0
    cs = np.sqrt(np.abs(gamma_eff * snap.pressure / snap.density))

    r = np.sqrt(snap.X**2 + snap.Y**2 + snap.Z**2)
    # Paczynski-Wiita potential; NaN inside r_g so the divergence there can't
    # flip sign and masquerade as (very) bound material.
    dr = r - r_g.to(r.units)
    dr[dr <= 0] = np.nan  # inside the horizon: undefined, not "extremely bound"
    phi = u.G * Mbh / dr
    soe = 0.5 * v_mag**2 - phi  # specific orbital energy (no pressure term)
    bernoulli = 0.5 * v_mag**2 + snap.pressure / snap.density + snap.Erad / 3.0 - phi
    # Signed log of the (dimensionless) Bernoulli / energy-spread ratio. The
    # .to("dimensionless") both validates dimensionlessness and simplifies the
    # mixed code+cgs units; numpy's log10/sign then return a bare ndarray, so
    # re-tag the genuinely-dimensionless result for the uniform plotting path.
    be_ratio = (bernoulli / delta_eps).to("dimensionless")
    bernoulli_symlog = u.unyt_array(
        np.sign(be_ratio) * np.log10(np.abs(be_ratio)), "dimensionless"
    )

    tfb = np.sqrt(-(np.pi**2) / 2 * (u.G * Mbh) ** 2 / soe**3)
    tfb_min = np.pi / np.sqrt(2) * Rstar**1.5 / np.sqrt(u.G * Mstar) * np.sqrt(Mbh / Mstar)

    return {
        "density": snap.density,
        "pressure": snap.pressure,
        "velocity": v_mag,
        "temperature": snap.temperature,
        "dissipation": snap.dissipation,
        "mach": v_mag / cs,  # velocity/velocity -> already dimensionless
        "erad": snap.Erad * snap.density,  # volumetric radiation energy [erg/cm^3]
        "bernoulli": bernoulli_symlog,
        "tfb_ratio": tfb / tfb_min,  # time/time -> already dimensionless
        # raw components, for the velocity-panel streamlines (xy only)
        "Vx": snap.Vx,
        "Vy": snap.Vy,
    }


def _no_white_diverging(name: str = "RdBu_r", cut: float = 0.22, n: int = 256):
    """Diverging colormap with the pale middle band removed.

    A standard diverging map fades to white at zero, which is exactly where
    the bound/unbound distinction has to be readable. Dropping the middle
    ``cut`` fraction of each half leaves saturated blue for <0 and saturated
    red for >0 with a hard break at zero, so sign reads at a glance.
    """
    base = plt.get_cmap(name)
    lo = base(np.linspace(0.0, 0.5 - cut, n // 2))
    hi = base(np.linspace(0.5 + cut, 1.0, n // 2))
    return mcolors.ListedColormap(np.vstack([lo, hi]), name=f"{name}_nowhite")


_BERNOULLI_CMAP = _no_white_diverging()


# (key, panel title, scalar_map kwargs; log_scale defaults True unless given)
_NINE_PANELS = [
    ("density", "Density", dict(label_latex=r"\rho", cmap="twilight")),
    ("pressure", "Pressure", dict(label_latex="P", cmap="rainbow")),
    ("velocity", "Velocity", dict(label_latex=r"|v|", cmap="cividis")),
    ("temperature", "Temperature", dict(label_latex="T", cmap="inferno")),
    (
        "dissipation",
        "Dissipation",
        dict(label_latex=r"\dot{E}_\mathrm{diss}", cmap="viridis"),
    ),
    (
        "mach",
        "Mach number",
        dict(label_latex=r"|v|", unit_latex=r"c_s", cmap="plasma"),
    ),
    (
        "erad",
        "Radiation energy",
        dict(label_latex=r"E_\mathrm{rad}", unit_latex=r"\mathrm{erg\,cm^{-3}}", cmap="magma"),
    ),
    (
        "bernoulli",
        "Bernoulli parameter",
        dict(
            label_latex=r"\mathrm{sgn}(\mathrm{Be})\log_{10}|\mathrm{Be}",
            unit_latex=r"\Delta\epsilon|",
            cmap=_BERNOULLI_CMAP,
            log_scale=False,
        ),
    ),
    (
        "tfb_ratio",
        "Fallback time",
        dict(
            label_latex=r"t_\mathrm{fb}",
            unit_latex=r"t_\mathrm{min}",
            cmap="rainbow",
            # Explicit range: tfb spans ~18 decades, so the automatic top-6
            # clip would land far above the physically interesting band.
            # Focused on the most-bound debris that sets the early fallback
            # rate (0.1 - ~2 t_min); longer-tfb material saturates.
            vmin=-1.0,
            vmax=0.3,
        ),
    ),
]


def _top_orders_range(data, n=4.0):
    """vmin/vmax spanning only the top n orders of magnitude of positive data.

    Dissipation floors span many more orders of magnitude than the
    interesting (shock) region, which drowns out contrast there — clip to
    the top n decades instead of the full range.
    """
    finite = data.v[np.isfinite(data.v) & (data.v > 0)]
    if finite.size == 0:
        return None, None
    vmax = np.ceil(float(np.max(np.log10(finite))) * 2.0) / 2.0
    return vmax - n, vmax


def _plot_nine_panels(fields: dict, si, sxsp, sysp, axes):
    """Render _NINE_PANELS onto 3x3 axes; overlays xy streamlines on the velocity panel."""
    for ax, (key, title, kw) in zip(axes.flat, _NINE_PANELS):
        kw = dict(kw)
        log_scale = kw.pop("log_scale", True)
        data = fields[key][si].in_base("cgs")
        if key == "bernoulli":
            # Robust (percentile) symmetric range: already signed-log
            # transformed, but a handful of cells right at r=0 still diverge
            # (log|Be| -> +inf) and can otherwise wash out the whole panel.
            finite = data.v[np.isfinite(data.v)]
            vmax = float(np.percentile(np.abs(finite), 99.5)) if finite.size else 1.0
            kw.setdefault("vmin", -vmax)
            kw.setdefault("vmax", vmax)
        elif key == "dissipation":
            vmin, vmax = _top_orders_range(data)
            if vmin is not None:
                kw.setdefault("vmin", vmin)
                kw.setdefault("vmax", vmax)
        scalar_map(data, sxsp, sysp, ax=ax, log_scale=log_scale, **kw)
        ax.set_title(title)

        if key == "velocity":
            u_grid = fields["Vx"][si].in_base("cgs").v.T
            v_grid = fields["Vy"][si].in_base("cgs").v.T
            ax.streamplot(
                sxsp.v,
                sysp.v,
                u_grid,
                v_grid,
                color="white",
                linewidth=0.6,
                density=1.3,
                arrowsize=0.7,
            )


def _draw_circles(axes, box, circles):
    """Overlay dashed reference circles (radius, label) and set axis limits from box."""
    for ax in axes.flat:
        for radius, label in circles:
            ax.add_patch(
                mpatches.Circle(
                    (0, 0), radius, fill=False, linestyle="--", color="white",
                    linewidth=1, zorder=5,
                )
            )
            ax.annotate(
                label, xy=(0, radius), color="white", fontsize=9,
                ha="center", va="bottom", zorder=6,
            )
        ax.set_xlim(box[0].v, box[3].v)
        ax.set_ylim(box[1].v, box[4].v)


# --------------------------- Mid-plane slices -------------------------------- #

# size -> (radius_kind, multiplier): box half-width = multiplier * r_{kind}.
# Replaces the old separate pericenter/apocenter zoom-in checks.
_BOX_SCALES = {
    "small": ("rp", 1.75),
    "middle": ("ra", 0.6),
    "big": ("ra", 2.0),
}

# sizes and (plane, integration-axis) pairs projection_check renders — a
# subset of _BOX_SCALES's sizes, all of its planes.
_PROJECTION_SIZES = ("middle", "big")
_PROJECTION_PLANES = [("xy", "z"), ("xz", "y")]

# figs/ series-folder names (each written as figs/{series}/snap{N:04d}.png)
# produced by each PNG-producing check, derived from the scale/plane config
# above rather than hand-typed — so this can't silently drift out of sync as
# that config changes. dispatch_diagnostics.py imports N_EXPECTED_PNGS_PER_SNAP
# instead of hand-maintaining its own copy.
PNG_SERIES_PER_SNAP = {
    "resolution": ["resolution_check"],
    "slice": [f"slice_{size}_xy" for size in _BOX_SCALES],
    "projection": [
        f"proj_{size}_{plane}" for size in _PROJECTION_SIZES for plane, _ax in _PROJECTION_PLANES
    ],
    "pericenter_yz": ["pericenter_yz"],
    # folder name is computed at runtime from r_p (see yz_frac_pericenter_check);
    # placeholder here only contributes to the count below.
    "yz_frac_pericenter": ["<frac>rp_yz"],
}
N_EXPECTED_PNGS_PER_SNAP = sum(len(v) for v in PNG_SERIES_PER_SNAP.values())


def slice_check(snap, fields: dict, input_file: str, output_dir: str, snap_num: int):
    """Mid-plane 9-panel quantity grids (xy only), at each scale in _BOX_SCALES."""
    logger.info("Mid-plane slice plots...")
    t_day = snap.time.to("day")

    for size, (kind, mult) in _BOX_SCALES.items():
        box, r = _scaled_box(input_file, kind, mult)
        label = r"r_p" if kind == "rp" else r"r_a"
        circles = [(r.v, rf"${label}$")]
        if kind == "rp":
            circles.insert(0, (0.6 * r.v, r"$r_0$"))  # smoothing length

        fig, axes = plt.subplots(3, 3, figsize=(18, 15), constrained_layout=True)
        fig.suptitle(
            rf"Mid-plane slices (xy-plane, {size} box)  "
            rf"${label}={r.v:.3g}\,R_\odot$  t = {t_day:.2f}",
            fontsize=16,
        )

        si, sxsp, sysp = snap.to_2dgrid(res=512, plane="xy", slice_coord=0, box_size=box)
        _plot_nine_panels(fields, si, sxsp, sysp, axes)
        _draw_circles(axes, box, circles)

        _savefig(
            fig,
            os.path.join(output_dir, f"figs/slice_{size}_xy/snap{snap_num:04d}.png"),
            dpi=300,
        )


# ----------------------------- Column projections ---------------------------- #


def projection_check(snap, input_file: str, output_dir: str, snap_num: int):
    """Column-projection 4-panel grids (Sigma, Erad, IE, dissipation).

    Middle and big box, xy and xz planes. Sigma = int(rho) dl [g/cm^2]; the
    energy panels are mass-weighted, int(rho * specific_field) dl [erg/cm^2],
    matching conservation_check's convention; dissipation is int(Ediss_dot)
    dl [erg/s/cm^2].
    """
    logger.info("Projection plots...")
    t_day = snap.time.to("day")

    # Pre-multiply on the (much smaller) per-particle arrays so each 3-D
    # kd-tree index below yields the final field directly, instead of
    # materialising density_3d and {Erad,IE}_3d as separate res^3 arrays and
    # multiplying them: doing that at res=512 in 3-D OOM-killed the process.
    rho_erad = snap.density * snap.Erad
    rho_ie = snap.density * snap.internal_energy

    jobs = [
        (size, plane, ax)
        for size in _PROJECTION_SIZES
        for plane, ax in _PROJECTION_PLANES
    ]

    for size, plane, int_axis in jobs:
        box, r = _scaled_box(input_file, *_BOX_SCALES[size])
        pi, pxsp, pysp, pzsp = snap.to_3dgrid(res=256, plane=plane, box_size=box)
        dz = pzsp[1:] - pzsp[:-1]

        # (field, label, cmap, unit_latex, clip_top4_orders)
        panels = [
            (snap.density, r"\Sigma", "twilight", None, False),
            (
                rho_erad,
                rf"\int\rho E_\mathrm{{rad}}\,d{int_axis}",
                "magma",
                r"\mathrm{erg\,cm^{-2}}",
                False,
            ),
            (
                rho_ie,
                rf"\int\rho\,\mathrm{{IE}}\,d{int_axis}",
                "inferno",
                r"\mathrm{erg\,cm^{-2}}",
                False,
            ),
            (
                snap.dissipation,
                rf"\int\dot{{E}}_\mathrm{{diss}}\,d{int_axis}",
                "viridis",
                r"\mathrm{erg\,s^{-1}\,cm^{-2}}",
                True,  # floors span too many decades; clip to the top 4
            ),
        ]

        fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
        fig.suptitle(
            rf"Projections ({plane}-plane, {size} box)  "
            rf"$r={r.v:.3g}\,R_\odot$  t = {t_day:.2f}",
            fontsize=14,
        )

        for ax, (field_1d, label, cmap, unit_latex, clip) in zip(axes.flat, panels):
            field_3d = field_1d[pi]  # one res^3 array alive at a time
            projected = np.sum(field_3d[:-1, :-1, :-1] * dz, axis=-1).in_base("cgs")
            kw = {}
            if clip:
                vmin, vmax = _top_orders_range(projected)
                if vmin is not None:
                    kw["vmin"], kw["vmax"] = vmin, vmax
            scalar_map(
                projected, pxsp, pysp, ax=ax,
                label_latex=label, cmap=cmap, unit_latex=unit_latex, **kw,
            )

        _savefig(
            fig,
            os.path.join(output_dir, f"figs/proj_{size}_{plane}/snap{snap_num:04d}.png"),
            dpi=300,
        )


# ------------------------- Pericenter compression yz-slices ----------------- #

# Fixed slice location [R_sun] for the "fractional" probe; not tied to r_p
# since it's a specific probe point for AMR timestep-limiting analysis (see
# analysis/amrtimestep/amr_timestep.ipynb, the wide yz panel). The other
# yz-slice check (pericenter_yz_check) slices at the actual r_p instead.
_YZ_SLICE_X = 18.0

# (field key, panel title, unit kind "cgs"/"lscale"/"mscale", scalar_map kwargs)
_YZ_PANELS = [
    ("density", "Density", "cgs", dict(label_latex=r"\rho", cmap="twilight")),
    ("pressure", "Pressure", "cgs", dict(label_latex="P", cmap="rainbow")),
    ("velocity", "Velocity", "cgs", dict(label_latex=r"|v|", cmap="cividis")),
    ("temperature", "Temperature", "cgs", dict(label_latex="T", cmap="inferno")),
    (
        "dissipation",
        "Dissipation",
        "cgs",
        dict(label_latex=r"\dot{E}_\mathrm{diss}", cmap="viridis"),
    ),
    (
        "mach",
        "Mach number",
        "cgs",
        dict(label_latex=r"|v|", unit_latex=r"c_s", cmap="plasma"),
    ),
    (
        "erad",
        "Radiation energy",
        "cgs",
        dict(label_latex=r"E_\mathrm{rad}", unit_latex=r"\mathrm{erg\,cm^{-3}}", cmap="magma"),
    ),
    (
        "width",
        "Cell width",
        "lscale",
        dict(label_latex="w", unit_latex=r"R_\odot", cmap="viridis"),
    ),
    (
        "mass",
        "Cell mass",
        "mscale",
        dict(label_latex="m", unit_latex=r"M_\odot", cmap="magma"),
    ),
]


def _yz_panel_fields(snap) -> dict:
    """9-panel fields shared by both yz pericenter-region slice checks."""
    v_mag = np.sqrt(snap.Vx**2 + snap.Vy**2 + snap.Vz**2)
    gamma_eff = snap.pressure / (snap.density * snap.internal_energy) + 1.0
    cs = np.sqrt(np.abs(gamma_eff * snap.pressure / snap.density))
    return {
        "density": snap.density,
        "pressure": snap.pressure,
        "velocity": v_mag,
        "temperature": snap.temperature,
        "dissipation": snap.dissipation,
        "mach": v_mag / cs,  # velocity/velocity -> already dimensionless
        "erad": snap.Erad * snap.density,  # volumetric radiation energy [erg/cm^3]
        "width": (3 * snap.volume / (4 * np.pi)) ** (1 / 3),  # sphere-equiv radius
        "mass": _cell_mass(snap),
    }


def _yz_slice_check(snap, output_dir, snap_num, slice_x, rp, folder, title):
    """Shared 3x3 yz-slice at x=slice_x [R_sun], 1024x512 res (wide in y).

    Box: y=+-r_p, z=+-0.5 r_p (2:1 aspect, matching the resolution).
    """
    t_day = snap.time.to("day")
    half_y, half_z = rp, 0.5 * rp
    box = [-half_y.v, -half_z.v, half_y.v, half_z.v]

    fields = _yz_panel_fields(snap)

    fig, axes = plt.subplots(3, 3, figsize=(21, 11), constrained_layout=True)
    fig.suptitle(rf"{title}  t = {t_day:.2f}", fontsize=16)

    si, sysp, szsp = snap.to_2dgrid(res=(1024, 512), plane="yz", slice_coord=slice_x, box_size=box)

    for ax, (key, ptitle, unit_kind, kw) in zip(axes.flat, _YZ_PANELS):
        kw = dict(kw)
        log_scale = kw.pop("log_scale", True)
        raw = fields[key][si]
        data = raw.in_base("cgs") if unit_kind == "cgs" else raw.to(getattr(richio.units, unit_kind))
        if key == "dissipation":
            vmin, vmax = _top_orders_range(data)
            if vmin is not None:
                kw.setdefault("vmin", vmin)
                kw.setdefault("vmax", vmax)
        scalar_map(data, sysp, szsp, ax=ax, log_scale=log_scale, **kw)
        ax.set_title(ptitle)

        if key == "velocity":
            u_grid = snap.Vy[si].in_base("cgs").v.T
            v_grid = snap.Vz[si].in_base("cgs").v.T
            ax.streamplot(
                sysp.v, szsp.v, u_grid, v_grid,
                color="white", linewidth=0.6, density=1.3, arrowsize=0.7,
            )

    _savefig(fig, os.path.join(output_dir, f"figs/{folder}/snap{snap_num:04d}.png"), dpi=300)


def yz_frac_pericenter_check(snap, input_file: str, output_dir: str, snap_num: int):
    """yz-slice at the fixed x=_YZ_SLICE_X R_sun probe (a fraction of r_p, not r_p itself)."""
    logger.info("Fractional-pericenter yz-slice check...")
    R_rsun, Mstar_msun, Mbh_msun, beta = _parse_tde_params(input_file)
    rp = R_rsun * (Mbh_msun / Mstar_msun) ** (1.0 / 3.0) / beta * richio.units.lscale
    frac = _YZ_SLICE_X / rp.v
    folder = f"{frac:.2f}rp_yz"
    title = rf"${frac:.2f}\,r_p$ compression (yz-plane, $x={_YZ_SLICE_X:g}\,R_\odot$)"
    _yz_slice_check(snap, output_dir, snap_num, _YZ_SLICE_X, rp, folder, title)


def pericenter_yz_check(snap, input_file: str, output_dir: str, snap_num: int):
    """yz-slice exactly at x=r_p (the true pericenter distance)."""
    logger.info("Pericenter yz-slice check...")
    R_rsun, Mstar_msun, Mbh_msun, beta = _parse_tde_params(input_file)
    rp = R_rsun * (Mbh_msun / Mstar_msun) ** (1.0 / 3.0) / beta * richio.units.lscale
    title = rf"Pericenter compression (yz-plane, $x=r_p={rp.v:.3g}\,R_\odot$)"
    _yz_slice_check(snap, output_dir, snap_num, rp.v, rp, "pericenter_yz", title)


# ----------------------------- Resolution check ----------------------------- #


def resolution_check(snap, output_dir: str, snap_num: int):
    """Cell-size (sphere-equivalent radius) histograms, by count and by mass."""
    logger.info("Resolution check...")
    h = (3 * snap.volume / (4 * np.pi)) ** (1 / 3)  # sphere-equiv radius, code_length (R☉)
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
        fig, os.path.join(output_dir, f"figs/resolution_check/snap{snap_num:04d}.png")
    )


# -------------------- Conservation & global energy budget ------------------- #


def conservation_check(snap) -> dict:
    """Total mass and kinetic/thermal/radiation energy; logged and returned as a dict."""
    logger.info("Conservation / global budget...")
    mass = _cell_mass(snap)

    M = mass.sum().to("g")
    Ek = (0.5 * mass * (snap.Vx**2 + snap.Vy**2 + snap.Vz**2)).sum().to("erg")
    Et = (mass * snap.InternalEnergy).sum().to("erg")
    Er = (mass * snap.Erad).sum().to("erg")

    logger.info("  M_total       = {:.4e}", M)
    logger.info("  E_kinetic     = {:.4e}", Ek)
    logger.info("  E_thermal     = {:.4e}", Et)
    logger.info("  E_radiation   = {:.4e}", Er)
    logger.info("  E_total       = {:.4e}", Ek + Et + Er)

    return dict(
        M_tot_g=float(M.v),
        E_kin_erg=float(Ek.v),
        E_thm_erg=float(Et.v),
        E_rad_erg=float(Er.v),
    )


# --------------------- Compton cooling: hot-gas fraction -------------------- #


def compton_check(snap) -> float:
    """Hot-gas (T > T_COMPTON) mass fraction; warns above 5%."""
    logger.info("Compton cooling check  (T > {})...", T_COMPTON)
    mass = _cell_mass(snap)
    frac = (mass[snap.temperature > T_COMPTON].sum() / mass.sum()).v
    logger.info("  Hot-gas mass fraction: {}  ({} %)", frac, frac * 100)
    if frac > 0.05:
        logger.warning("  > 5 % - Compton cooling may not be effective!")
    return float(frac)


# --------------- Time-evolution: dissipation & fluff fraction --------------- #


def _scalars_for_snap(snap_path: str) -> dict:
    """Scalar time-series quantities for one snapshot, as a JSON-safe dict."""
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
    """Cache scalars for every snap_*.h5 in the run dir, then plot the time series."""
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
    "slice",
    "projection",
    "pericenter_yz",
    "yz_frac_pericenter",
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
      python diagnostics.py snap.h5 -c time_evolution -c slice
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
        "slice",
        "projection",
        "pericenter_yz",
        "yz_frac_pericenter",
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

    if "slice" in run:
        fields = _nine_panel_fields(snap, input_file)
        slice_check(snap, fields, input_file, output_dir, snap_num)
        # ~11 per-cell arrays; drop them before the later checks allocate
        # their own grids, or a big snapshot can exhaust memory part-way.
        del fields
        gc.collect()

    if "projection" in run:
        projection_check(snap, input_file, output_dir, snap_num)

    if "pericenter_yz" in run:
        pericenter_yz_check(snap, input_file, output_dir, snap_num)

    if "yz_frac_pericenter" in run:
        yz_frac_pericenter_check(snap, input_file, output_dir, snap_num)

    if "time_evolution" in run:
        time_evolution_check(input_file, output_dir)

    logger.success("=== Done snap_{} ===", snap_num)
    logger.remove(log_sink)


if __name__ == "__main__":
    app()
