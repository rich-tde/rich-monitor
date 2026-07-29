# rich-monitor

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
