# rich-monitor

Repository guidance for Codex and other coding agents. The units and cell-size
rules below are incorporated verbatim from `CLAUDE.md`; keep these shared rules
consistent when updating either file.

## Workspace context

This repository automates monitoring and diagnostics for RICH moving-mesh
radiative hydrodynamic simulations, currently focused on tidal disruption event
(TDE) runs on Snellius.

- `scripts/diagnostics.py`: Typer CLI for one snapshot; integrity, resolution,
  conservation, Compton cooling, time evolution, slices, and projections.
  Writes figures, `diagnostics.log`, and `diagnostics_cache.json`.
- `scripts/dispatch_diagnostics.py`: discovers `snap_<N>.h5` and
  `snap_full_<N>.h5`, processes the latest snapshot by default (`--all` for all),
  and optionally delegates matching benchmark logs to `measure_speed.py`.
- `scripts/progress_monitor.py`: queries Snellius Slurm accounting through SSH,
  caches job and snapshot information, and writes `progress.csv`.
- `scripts/measure_speed.py`: parses benchmark logs and plots cell updates per
  core per second. `scripts/cost_estimate.py` estimates wall time and core hours.
- `richio/`: Git submodule providing snapshot loading, field aliases, units,
  interpolation, plotting, opacity, shock finding, and optional volume rendering.
  Start with `richio/richio/data.py`, `config.py`, `units.py`, and `plots.py`;
  usage documentation is in `richio/docs/docs/`.
- `academic-mplstyle/`: Git submodule containing the plotting style.
- `analysis/`: ignored local research scripts, notebooks, and generated output.
- `scripts/*.sh` and `crontab.bak`: environment-specific launch, backup, upload,
  and scheduling configuration. The cron file is a saved configuration, not
  evidence of the currently installed schedule.
- `YujieSnellius`, `alice`, `snellius-backup`, and `snellius-logs`: ignored
  symlinks to external simulation data and output directories.

## Units (unyt)

Keep quantities as `unyt` arrays end to end — the unit tracking is a correctness
check, so let a wrong operation raise or mismatch instead of silently producing
a bad number.

- Do **not** strip units (`.v`, `.value`, `np.asarray`) and re-attach them with
  `unyt_array(..., "unit")`. That defeats the dimensional check and hides bugs.
  Work with the native `unyt` result; if a `numpy` ufunc drops the unit on a
  genuinely dimensionless value (e.g. `np.log10`/`np.sign` on a dimensionless
  array), re-tagging as `"dimensionless"` is fine — but only there, and comment
  why.
- Do **not** convert (`.to(...)`, `.in_base(...)`) mid-computation. Ratios of
  same-dimension quantities are already dimensionless — no `.to("dimensionless")`
  needed; unyt cancels mixed code/cgs units on its own.
- Convert **only** at a boundary:
  - visualization — right before a plot call (`scalar_map` reads `f.units` for
    the label; `snap.slice` already converts to cgs via its `unit_system` arg);
  - serialization / logging — `float(x.to("erg").v)` for JSON or human-readable
    summaries;
  - matplotlib primitives that reject unyt (`streamplot`, `set_xlim`, `Circle`,
    f-string coords) — `.v` at the call site.

## Cell size

Sphere-equivalent radius is `(3 V / (4 pi))**(1/3)`, **not** `(3/(4 pi)) *
V**(1/3)`. The cube root applies to the whole `3V/4pi`.

## Implementation conventions

- Do not modify `.sh` shell scripts unless the user explicitly authorizes shell
  script changes. They are used by active cron jobs. General requests to update
  scripts or documentation do not authorize edits to shell scripts.
- Use `richio.load(...)` and the existing field registry and unit definitions
  when working with snapshots. Run-directory names encode physical parameters;
  preserve the parsing conventions when changing discovery or dispatch.
- Keep `PNG_SERIES_PER_SNAP` aligned with the figures produced by diagnostics.
  Dispatch imports its derived `N_EXPECTED_PNGS_PER_SNAP`; completion requires
  both a cache entry and enough figures in `figs/{series}/snap{N:04d}.png`.
  A restricted `--check` run still uses the full figure count for completion.
- Consider memory use for large snapshots. Release temporary per-cell arrays
  before allocating subsequent plotting grids, following the existing pattern.
- Treat `richio/` and `academic-mplstyle/` as separate Git repositories when
  making changes inside them.
- Preserve unrelated local edits. Existing code may predate these instructions;
  use the rules above for new work without broadening a task into a cleanup.

## Python command-line documentation

Use GNU command-line documentation conventions for Python scripts:

- Follow the [GNU command-line interface guidance](https://www.gnu.org/prep/standards/html_node/Command_002dLine-Interfaces.html)
  and [GNU `--help` guidance](https://www.gnu.org/prep/standards/html_node/_002d_002dhelp.html).
  Use the conventional reference sections described by
  [Linux man-pages](https://man7.org/linux/man-pages/man7/man-pages.7.html)
  to organize each script's opening docstring.
- Make the opening docstring a self-contained command reference: summary,
  SYNOPSIS, DESCRIPTION, ARGUMENTS, OPTIONS, and EXAMPLES. Include CHECKS or
  other accepted-value lists, EXIT STATUS, FILES, NOTES, and SEE ALSO where
  relevant. Do not substitute a collection of examples for the option reference.
- In the synopsis, use uppercase metavariables for user-supplied values,
  brackets for optional items, and ellipses for repeatable items. State the
  order and required/optional status of positional arguments.
- Document every supported option with its exact spelling, aliases, value
  type, default, accepted values, repeatability, and relevant interactions.
  Distinguish command-line options from constants configured in source code.
  Explain snapshot selection, check selection, and overwrite behavior separately.
- Keep `--help` a concise invocation reference that prints to standard output
  and exits successfully without performing the script's normal work. Include
  arguments, options, defaults, accepted check names, and useful examples.
  Keep it consistent with the opening docstring and actual parser behavior.
- Use executable examples with explicit required arguments. Identify placeholder
  paths and the working directory; use shell continuation backslashes when
  splitting a command across lines.
- Document actual output files, overwrite/cache behavior, and exit-status
  limitations. Do not promise behavior the implementation does not provide.
- Verify documentation against the parser and inspect rendered `--help` output,
  including placeholders and line wrapping. Documentation requests do not
  authorize changes to CLI semantics, scientific calculations, or `.sh` files.

## Running and checking changes

- Shell launchers use `/data2/yujiehe/.conda/envs/richio/bin/python`. Check the
  available interpreter and dependencies before running commands; paths are
  specific to this workspace. The local library can be installed with
  `python -m pip install -e ./richio` when environment setup is needed.
- Inspect CLI options with `python scripts/diagnostics.py --help` or
  `python scripts/dispatch_diagnostics.py --help` in the configured environment.
- For dispatch previews, use explicit input/output directories and `--dry-run`.
  Dry runs still create output directories and a dispatch log.
- Root `ruff.toml` enables import sorting and collection simplification and
  forbids automatic removal of unused imports (`F401`). `richio/pyproject.toml`
  has its own Ruff configuration. Run focused checks on changed Python files.
- No dedicated test suite was found in the inspected workspace. Validate code
  changes with focused checks and, when relevant data is available, a limited
  diagnostic run using `--check` and a separate output directory.
- Backup/upload wrappers perform real `rclone` operations, and progress
  monitoring queries Snellius over SSH; they are operational commands, not
  smoke tests.
