import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import io
    import re
    import json
    import math
    import warnings
    from pathlib import Path

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from jinja2 import Template
    from matplotlib.lines import Line2D

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)
    pd.set_option("display.max_colwidth", None)
    return (
        Line2D,
        Path,
        Template,
        io,
        json,
        math,
        mo,
        np,
        pd,
        plt,
        re,
        warnings,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # Compare force field parameters

    Load the **UFF** and **DREIDING** literature parameters as ground truth,
    then compare any other force-field file (`.def`, `.template`, `.dat`)
    against them.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. Ground truth: UFF & DREIDING literature parameters
    """)
    return


@app.cell
def _():
    # kcal/mol -> K conversion
    # J_per_calorie: exact definition (Rossini, 1935 proposal)
    # N_A, k_B: CODATA 2017 (Mohr, Newell, Taylor & Tiesinga, Metrologia 55, 125 (2018)); same in SI Brochure (2019)
    J_PER_CALORIE = 4.184
    N_A = 6.02214076e23  # Avogadro's number (mol-1)
    K_B = 1.380649e-23  # Boltzmann constant (J/K)
    R_GAS = N_A * K_B  # ideal gas constant (J/mol/K)
    KCAL_PER_MOL_TO_K = J_PER_CALORIE * 1000 / R_GAS
    return (KCAL_PER_MOL_TO_K,)


@app.cell
def _(mo):
    mo.md(r"""
    ### UFF (LLM-transcribed from Rapp\u00e9 et al., 1992, Table I)
    """)
    return


@app.cell
def _(KCAL_PER_MOL_TO_K, pd):
    _uff_raw = pd.read_csv("ff_data/uff_atomic_data_Claude_transcribed.csv")

    # UFF defines one nonbond (x1, D1) pair per element (all atom types of a
    # given element share identical nonbond parameters in this table), so
    # dropping to one row per element is safe here.
    assert (
        _uff_raw.groupby("element")[["nonbond_distance_x1_ang", "nonbond_energy_D1_kcal_mol"]]
        .nunique()
        .le(1)
        .all()
        .all()
    ), "UFF nonbond parameters are not constant within an element; dedup would be lossy"

    UFF_GROUND_TRUTH = (
        _uff_raw.drop_duplicates(subset=["element"])
        .reset_index(drop=True)[["element", "nonbond_distance_x1_ang", "nonbond_energy_D1_kcal_mol"]]
        .rename(columns={
            "nonbond_distance_x1_ang": "x1_ang",
            "nonbond_energy_D1_kcal_mol": "D1_kcal_mol",
        })
    )
    UFF_GROUND_TRUTH["sigma_ang"] = UFF_GROUND_TRUTH["x1_ang"] / 2 ** (1 / 6)
    UFF_GROUND_TRUTH["epsilon_K"] = UFF_GROUND_TRUTH["D1_kcal_mol"] * KCAL_PER_MOL_TO_K
    UFF_GROUND_TRUTH
    return (UFF_GROUND_TRUTH,)


@app.cell
def _(mo):
    mo.md(r"""
    ### DREIDING (LLM-transcribed from Mayo et al., 1990, Table II)
    """)
    return


@app.cell
def _(KCAL_PER_MOL_TO_K, pd):
    dreiding_raw = pd.read_csv("ff_data/dreiding_vdw_parameters_Claude_transcribed.csv")
    dreiding_raw["sigma_ang"] = dreiding_raw["R0_ang"] / 2 ** (1 / 6)
    dreiding_raw["epsilon_K"] = dreiding_raw["D0_kcal_mol"] * KCAL_PER_MOL_TO_K
    dreiding_raw
    return (dreiding_raw,)


@app.cell
def _(mo):
    mo.md(r"""
    Several elements have more than one DREIDING atom type (implicit-hydrogen
    "united atom" carbons `C_3x`/`C_R1`, hydrogen-bonding/bridging hydrogen
    variants, ionic metal variants). GCMC/MOF studies — including
    `ff_data/other/force_field_wei_github_Sept2026.template` — use the
    **explicit-atom** type (`atom_type == element`, e.g. plain `C` and `H`) as
    the framework parameter, falling back to whatever single type exists for
    elements that only have an ionic variant (Na, Ca, Fe, Zn). That's the rule
    applied below; the rows it discards are listed for transparency.
    """)
    return


@app.cell
def _(dreiding_raw):
    _dreiding_ranked = dreiding_raw.copy()
    _dreiding_ranked["_is_explicit"] = _dreiding_ranked["atom_type"] == _dreiding_ranked["element"]
    _dreiding_ranked = _dreiding_ranked.sort_values(
        ["element", "_is_explicit"], ascending=[True, False], kind="stable"
    )

    DREIDING_GROUND_TRUTH = (
        _dreiding_ranked.drop_duplicates(subset=["element"], keep="first")
        .reset_index(drop=True)[["element", "atom_type", "R0_ang", "D0_kcal_mol", "sigma_ang", "epsilon_K"]]
    )
    DREIDING_DISCARDED_ATOM_TYPES = _dreiding_ranked[
        ~_dreiding_ranked.index.isin(
            _dreiding_ranked.drop_duplicates(subset=["element"], keep="first").index
        )
    ][["element", "atom_type", "R0_ang", "D0_kcal_mol", "note"]]

    DREIDING_GROUND_TRUTH
    return DREIDING_DISCARDED_ATOM_TYPES, DREIDING_GROUND_TRUTH


@app.cell
def _(DREIDING_DISCARDED_ATOM_TYPES, mo):
    mo.vstack([
        mo.md("discarded DREIDING atom types (not used as ground truth):"),
        DREIDING_DISCARDED_ATOM_TYPES,
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. Load a force field file to compare
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    Pick a file already in `ff_data/` below, or upload one from anywhere on
    disk (the upload takes priority when both are set). Supported formats:
    RASPA `.def`, RASPA3 `.template` (JSON), and plain `.dat` (whitespace
    columns: `element sigma_ang epsilon_K mass_amu`).
    """)
    return


@app.cell
def _(Path, mo):
    file_browser = mo.ui.file_browser(
        initial_path=Path("ff_data"),
        filetypes=[".def", ".template", ".dat"],
        multiple=False,
        label="browse ff_data/",
    )
    file_upload = mo.ui.file(
        filetypes=[".def", ".template", ".dat"],
        multiple=False,
        label="...or upload a file",
    )
    mo.hstack([file_browser, file_upload], justify="start", gap=2)
    return file_browser, file_upload


@app.cell
def _(pd, re):
    # One interaction entry: name, interaction type, 0..n numeric parameters,
    # optionally followed by a "// source" comment. (RASPA2 force_field_mixing_rules.def)
    _DEF_ENTRY_RE = re.compile(
        r"""
        (?P<element>\S+) \s+
        (?P<interaction>lennard-jones|none) \b
        (?P<params>(?: \s+ [-+]?\d+(?:\.\d*)?(?:[eE][-+]?\d+)? )*)
        (?: \s* // \s* (?P<source>.*?) )?
        \s*$
        """,
        re.VERBOSE | re.IGNORECASE,
    )


    def _resolve_idem(sources):
        """Replace "idem" with the most recent explicit source."""
        resolved = []
        previous = None
        for value in sources:
            if not isinstance(value, str) or not value.strip():
                resolved.append(None)
            elif value.strip().lower() == "idem":
                resolved.append(previous)
            else:
                resolved.append(value)
                previous = value
        return resolved


    def parse_def(text: str) -> pd.DataFrame:
        """Parse a RASPA-style ``force_field_mixing_rules.def`` into a DataFrame
        with columns ``element``, ``epsilon_K``, ``sigma_ang``, ``source``."""
        rows = []
        for line in text.splitlines():
            line = line.split("#", 1)[0].strip()  # "#" starts a header/comment
            if not line:
                continue
            match = _DEF_ENTRY_RE.fullmatch(line)
            if match is None:
                continue
            params = match["params"].split()
            src = (match["source"] or "").strip()
            rows.append({
                "element": match["element"],
                "epsilon_K": float(params[0]) if len(params) > 0 else float("nan"),
                "sigma_ang": float(params[1]) if len(params) > 1 else float("nan"),
                "source": src or None,
            })

        df = pd.DataFrame(rows, columns=["element", "epsilon_K", "sigma_ang", "source"])
        df["element"] = df["element"].str.replace("_", "", regex=False)
        df["source"] = _resolve_idem(df["source"])
        return df

    return (parse_def,)


@app.cell
def _(Template, json, parse_def, pd):
    def parse_template(text: str) -> pd.DataFrame:
        """Parse a RASPA3 ``.template`` (Jinja2-rendered JSON) into a DataFrame
        with columns ``element``, ``epsilon_K``, ``sigma_ang``, ``source``,
        ``mass_amu``. Assumes self-interaction (LJ) parameters are stored as
        ``[epsilon_K, sigma_ang]``, matching the files in ``ff_data``."""
        rendered = Template(text).render(cutoff_radius="NaN")
        data = json.loads(rendered)

        rows = []
        for atom, interaction in zip(data["PseudoAtoms"], data["SelfInteractions"]):
            assert atom["source"] == interaction["source"], (
                f"PseudoAtoms/SelfInteractions source mismatch for {atom['name']!r}"
            )
            epsilon_K, sigma_ang = interaction["parameters"]
            rows.append({
                "element": atom["name"],
                "epsilon_K": epsilon_K,
                "sigma_ang": sigma_ang,
                "mass_amu": atom.get("mass"),
                "source": interaction["source"],
            })
        return pd.DataFrame(rows, columns=["element", "epsilon_K", "sigma_ang", "mass_amu", "source"])


    def parse_dat(text: str) -> pd.DataFrame:
        """Parse a whitespace-columns ``.dat`` file (``element sigma_ang
        epsilon_K mass_amu``, no header). A trailing ``// <source>`` comment on
        a line (as written by this notebook's export) is kept as that row's
        source; lines without one default to ``"UFF"``, matching the plain
        UFF-only .dat files in ``ff_data``. Blank lines and ``#``-prefixed
        comment lines are ignored."""
        rows = []
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            line, _, comment = line.partition("//")
            fields = line.split()
            if not fields:
                continue
            element, sigma_ang, epsilon_K, mass_amu = fields[:4]
            rows.append({
                "element": element,
                "sigma_ang": float(sigma_ang),
                "epsilon_K": float(epsilon_K),
                "mass_amu": float(mass_amu),
                "source": comment.strip() or "UFF",
            })
        return pd.DataFrame(rows, columns=["element", "sigma_ang", "epsilon_K", "mass_amu", "source"])


    PARSERS = {".def": parse_def, ".template": parse_template, ".dat": parse_dat}
    return (PARSERS,)


@app.cell
def _(KCAL_PER_MOL_TO_K, PARSERS, io, pd):
    def parse_csv(text: str) -> pd.DataFrame:
        """Parse one of this repo's ground-truth literature CSVs (UFF Table I or
        DREIDING Table II shape) into the common element/sigma_ang/epsilon_K/
        source shape used elsewhere in this notebook."""
        raw = pd.read_csv(io.StringIO(text))

        if "nonbond_distance_x1_ang" in raw.columns:
            assert (
                raw.groupby("element")[["nonbond_distance_x1_ang", "nonbond_energy_D1_kcal_mol"]]
                .nunique().le(1).all().all()
            ), "UFF nonbond parameters are not constant within an element; dedup would be lossy"
            out = raw.drop_duplicates(subset=["element"]).reset_index(drop=True)
            out["sigma_ang"] = out["nonbond_distance_x1_ang"] / 2 ** (1 / 6)
            out["epsilon_K"] = out["nonbond_energy_D1_kcal_mol"] * KCAL_PER_MOL_TO_K
            out["source"] = "UFF"
            return out[["element", "sigma_ang", "epsilon_K", "source"]]

        if "R0_ang" in raw.columns:
            # Several atom types per element (implicit-H united atoms, ionic
            # variants); prefer the explicit atom type, same rule as section 1.
            ranked = raw.copy()
            ranked["_is_explicit"] = ranked["atom_type"] == ranked["element"]
            ranked = ranked.sort_values(["element", "_is_explicit"], ascending=[True, False], kind="stable")
            out = ranked.drop_duplicates(subset=["element"], keep="first").reset_index(drop=True)
            out["sigma_ang"] = out["R0_ang"] / 2 ** (1 / 6)
            out["epsilon_K"] = out["D0_kcal_mol"] * KCAL_PER_MOL_TO_K
            out["source"] = "DREIDING"
            return out[["element", "sigma_ang", "epsilon_K", "source"]]

        raise ValueError(
            "Unrecognized CSV schema — expected the UFF (`nonbond_distance_x1_ang` "
            "column) or DREIDING (`R0_ang` column) ground-truth tables."
        )


    EXPORT_SOURCE_PARSERS = {**PARSERS, ".csv": parse_csv}
    return (EXPORT_SOURCE_PARSERS,)


@app.cell
def _(json, pd, warnings):
    def to_csv(df: pd.DataFrame) -> str:
        """Serialize the canonical element/sigma_ang/epsilon_K/... dataframe to
        CSV, keeping only the common columns that are actually present."""
        cols = [c for c in ["element", "sigma_ang", "epsilon_K", "mass_amu", "source"] if c in df.columns]
        return df[cols].to_csv(index=False)


    def to_dat(df: pd.DataFrame) -> str:
        """Serialize to the plain whitespace-columns ``.dat`` shape (``element
        sigma_ang epsilon_K mass_amu``, no header, matching ``parse_dat``). Mass
        is written as ``nan`` when the source format didn't carry it."""
        if "mass_amu" not in df.columns:
            warnings.warn("source has no mass_amu column; writing 'nan' for mass in the .dat export.")
        lines = [
            f"{row.element} {row.sigma_ang:.6g} {row.epsilon_K:.6g} {row.mass_amu if 'mass_amu' in df.columns else float('nan'):.6g}"
            for row in df.itertuples()
        ]
        return "\n".join(lines) + "\n"


    def to_def(df: pd.DataFrame) -> str:
        """Serialize to a RASPA2-style ``force_field_mixing_rules.def`` (matching
        ``parse_def``'s expectations: element + trailing ``_``, epsilon before
        sigma, optional ``// source`` comment)."""
        lines = [
            "# general rule for shifted vs truncated",
            "shifted",
            "# general rule tailcorrections",
            "no",
            "# number of defined interactions",
            str(len(df)),
            "# type interaction, parameters.    IMPORTANT: define shortest matches first, so that more specific ones overwrites these",
        ]
        for row in df.itertuples():
            name = f"{row.element}_"
            source = getattr(row, "source", None)
            comment = f"     // {source}" if isinstance(source, str) and source else ""
            lines.append(f"{name:<14} lennard-jones   {row.epsilon_K:<10.6g} {row.sigma_ang:<10.6g}{comment}")
        return "\n".join(lines) + "\n"


    def to_template(df: pd.DataFrame) -> str:
        """Serialize to a RASPA3 ``.template`` (Jinja2 + JSON, matching
        ``parse_template``'s expectations). ``CutOff`` is kept as the
        ``{{ cutoff_radius }}`` placeholder, same as the files in ``ff_data``."""
        pseudo_atoms, self_interactions = [], []
        for row in df.itertuples():
            source = getattr(row, "source", None) or "unknown"
            mass = getattr(row, "mass_amu", None)
            mass = None if mass is None or pd.isna(mass) else float(mass)
            pseudo_atoms.append({
                "name": row.element,
                "framework": True,
                "print_to_output": True,
                "element": row.element,
                "print_as": row.element,
                "mass": mass,
                "charge": 0.0,
                "source": source,
            })
            self_interactions.append({
                "name": row.element,
                "type": "lennard-jones",
                "parameters": [row.epsilon_K, row.sigma_ang],
                "source": source,
            })
        data = {
            "PseudoAtoms": pseudo_atoms,
            "SelfInteractions": self_interactions,
            "MixingRule": "Lorentz-Berthelot",
            "TruncationMethod": "truncated",
            "CutOff": float("nan"),
            "TailCorrections": True,
        }
        rendered = json.dumps(data, indent=2)
        return rendered.replace('"CutOff": NaN', '"CutOff": {{ cutoff_radius }}', 1)


    EXPORTERS = {".csv": to_csv, ".dat": to_dat, ".def": to_def, ".template": to_template}
    return (EXPORTERS,)


@app.cell
def _(Path, mo):
    def load_force_field_file(browser, upload, parsers):
        """Resolve the active (name, suffix, parsed df) from a browser/upload
        pair. Upload takes priority when both are set. Halts the cell (via
        ``mo.stop``) with a helpful message if nothing is selected yet or the
        extension isn't supported."""
        if upload.value:
            name = upload.name()
            text = upload.contents().decode("utf-8")
        elif browser.value:
            name = browser.name()
            text = browser.path().read_text()
        else:
            name = None
            text = None

        mo.stop(name is None, mo.md("**Select a file above (or upload one).**"))

        suffix = Path(name).suffix
        mo.stop(
            suffix not in parsers,
            mo.md(f"**Unsupported file type `{suffix}`** — expected one of {list(parsers)}."),
        )
        return name, suffix, parsers[suffix](text)

    return (load_force_field_file,)


@app.cell
def _(PARSERS, file_browser, file_upload, load_force_field_file, mo):
    active_name, active_suffix, provided_raw = load_force_field_file(file_browser, file_upload, PARSERS)
    mo.md(f"Loaded **{active_name}** as `{active_suffix}` ({len(provided_raw)} entries).")
    return (provided_raw,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 3. Compare against ground truth
    """)
    return


@app.cell
def _(pd, warnings):
    def split_by_force_field(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Split provided parameters into the subset meant to be compared against
        UFF and the subset meant to be compared against DREIDING, based on the
        ``source`` text. Rows whose source names neither (e.g. fluid beads in a
        RASPA .def file) are dropped from both."""
        has_source = df["source"].notna()
        is_uff = has_source & df["source"].str.contains("UFF", case=False, na=False)
        is_dreiding = has_source & df["source"].str.contains("DREIDING", case=False, na=False)
        assert not (is_uff & is_dreiding).any(), "a row's source names both UFF and DREIDING"
        return df[is_uff].copy(), df[is_dreiding].copy()


    def normalize_elements(df: pd.DataFrame, other: pd.DataFrame) -> pd.DataFrame:
        """Rename quirky element labels to match ``other`` (currently: Lawrencium
        is "Lr" in some sources, "Lw" in UFF's own table)."""
        df = df.copy()
        if "Lw" in set(other["element"]) and "Lr" in set(df["element"]) and "Lw" not in set(df["element"]):
            df["element"] = df["element"].replace({"Lr": "Lw"})
        return df


    def dedupe_elements(df: pd.DataFrame, label: str) -> pd.DataFrame:
        """Guard against a file defining more than one atom type per bare
        element (e.g. several oxidation states collapsing to the same symbol).
        An inner join on ``element`` would otherwise silently explode into a
        many-to-many match; keep the first occurrence instead and warn."""
        dup_mask = df["element"].duplicated(keep=False)
        if dup_mask.any():
            dup_elements = sorted(set(df.loc[dup_mask, "element"]))
            warnings.warn(
                f"{label} has multiple entries for the same element {dup_elements}; "
                "keeping the first occurrence of each.",
                stacklevel=2,
            )
        return df.drop_duplicates(subset=["element"], keep="first")


    def compute_errors(provided: pd.DataFrame, ground_truth: pd.DataFrame, suffix: str) -> pd.DataFrame:
        """Inner-join ``provided`` to ``ground_truth`` on element and compute
        signed error, absolute error, signed relative error, and absolute
        relative error for sigma and epsilon. Robust to the two sides using
        different atom-type granularity: elements present on only one side are
        simply dropped (inner join), and duplicate elements on either side are
        deduped with a warning rather than exploding the join."""
        provided = normalize_elements(provided, ground_truth)
        provided = dedupe_elements(provided, "provided force field")
        ground_truth = dedupe_elements(ground_truth, f"ground truth{suffix}")
        merged = pd.merge(
            provided, ground_truth, on="element", how="inner", suffixes=("_provided", suffix)
        )
        for quantity in ["sigma_ang", "epsilon_K"]:
            truth_col = f"{quantity}{suffix}"
            provided_col = f"{quantity}_provided"
            merged[f"{quantity}_error"] = merged[provided_col] - merged[truth_col]
            merged[f"{quantity}_abs_error"] = merged[f"{quantity}_error"].abs()
            merged[f"{quantity}_rel_error"] = merged[f"{quantity}_error"] / merged[truth_col]
            merged[f"{quantity}_abs_rel_error"] = merged[f"{quantity}_abs_error"] / merged[truth_col].abs()
        return merged


    def coverage_summary(provided: pd.DataFrame, ground_truth: pd.DataFrame, ff_name: str) -> str:
        provided = normalize_elements(provided, ground_truth)
        provided_elements = set(provided["element"])
        truth_elements = set(ground_truth["element"])
        missing = truth_elements - provided_elements
        extra = provided_elements - truth_elements
        lines = [
            f"**{ff_name}**: {len(provided_elements & truth_elements)}/{len(truth_elements)} elements matched."
        ]
        if missing:
            lines.append(f"- missing ({len(missing)}): {sorted(missing)}")
        if extra:
            lines.append(f"- not in {ff_name} ground truth ({len(extra)}): {sorted(extra)}")
        return "\n\n".join(lines)

    return compute_errors, coverage_summary, split_by_force_field


@app.cell
def _(
    DREIDING_GROUND_TRUTH,
    UFF_GROUND_TRUTH,
    compute_errors,
    coverage_summary,
    mo,
    provided_raw,
    split_by_force_field,
):
    provided_uff, provided_dreiding = split_by_force_field(provided_raw)

    compare_uff = compute_errors(provided_uff, UFF_GROUND_TRUTH, "_UFF") if len(provided_uff) else None
    compare_dreiding = compute_errors(provided_dreiding, DREIDING_GROUND_TRUTH, "_DREIDING") if len(provided_dreiding) else None

    mo.md("\n\n".join(filter(None, [
        coverage_summary(provided_uff, UFF_GROUND_TRUTH, "UFF") if len(provided_uff) else None,
        coverage_summary(provided_dreiding, DREIDING_GROUND_TRUTH, "DREIDING") if len(provided_dreiding) else None,
        "*(no rows in this file matched a UFF or DREIDING source label)*" if compare_uff is None and compare_dreiding is None else None,
    ])))
    return compare_dreiding, compare_uff


@app.cell
def _(mo):
    mo.md(r"""
    ### Error profile plots
    """)
    return


@app.cell
def _():
    # --- plot design tokens ---------------------------------------------------
    SURFACE = "#fcfcfb"
    INK = "#0b0b0b"
    INK_SECONDARY = "#52514e"
    INK_MUTED = "#8a8880"
    SERIES = "#2a78d6"  # in-band points
    CRITICAL = "#d03b3b"  # out-of-band points

    QUANTITY_UNITS = {"sigma_ang": "\u00c5", "epsilon_K": "K"}
    QUANTITY_LABELS = {"sigma_ang": "sigma", "epsilon_K": "epsilon"}
    return (
        CRITICAL,
        INK,
        INK_MUTED,
        INK_SECONDARY,
        QUANTITY_LABELS,
        QUANTITY_UNITS,
        SERIES,
        SURFACE,
    )


@app.cell
def _(mo):
    n_std_slider = mo.ui.slider(
        0.5, 4.0, 0.25, value=1.0, label="outlier threshold (std devs from mean)"
    )
    n_std_slider
    return (n_std_slider,)


@app.cell
def _(
    INK,
    INK_SECONDARY,
    QUANTITY_LABELS,
    QUANTITY_UNITS,
    SURFACE,
    draw_error_panel,
    np,
    outlier_legend,
    pd,
    plt,
):
    def plot_error_profile(
        df: pd.DataFrame,
        quantity: str,
        *,
        n_std: float = 1.0,
        title_prefix: str = "",
    ) -> tuple[plt.Figure, pd.DataFrame]:
        """Two stacked panels for one quantity: absolute error on top, relative
        error below. Elements are sorted alphabetically. Points further than
        ``n_std`` standard deviations from that panel's mean are drawn in red and
        listed, with their value, in the panel legend.

        Returns ``(figure, outliers)`` where ``outliers`` has columns
        ``element``, ``panel``, ``value``, ``deviation_std``.
        """
        abs_col, rel_col = f"{quantity}_abs_error", f"{quantity}_rel_error"
        unit = QUANTITY_UNITS[quantity]

        data = (
            df[["element", abs_col, rel_col]]
            .sort_values("element", key=lambda s: s.astype(str).str.lower())
            .reset_index(drop=True)
        )
        elements = data["element"].astype(str).to_numpy()
        x = np.arange(len(elements))

        panels = [
            (abs_col, "absolute error", unit, 1.0),
            (rel_col, "relative error", "%", 100.0),
        ]

        plot_width = max(9.0, 0.145 * len(elements))
        fig, axes = plt.subplots(2, 1, figsize=(plot_width, 8.2), sharex=True, facecolor=SURFACE)

        all_outliers, legend_cols = [], [1]
        for ax, (col, panel_name, panel_unit, scale) in zip(axes, panels):
            y = pd.to_numeric(data[col], errors="coerce").to_numpy(dtype=float) * scale
            finite = np.isfinite(y)
            mean = float(np.nanmean(y[finite])) if finite.any() else float("nan")
            std = float(np.nanstd(y[finite], ddof=1)) if finite.sum() > 1 else 0.0

            deviation = np.abs(y - mean) / std if std > 0 else np.zeros_like(y)
            is_out = finite & (deviation > n_std)

            draw_error_panel(ax, x, y, is_out, mean, std, n_std, ylabel=f"{panel_name} [{panel_unit}]")
            legend_cols.append(outlier_legend(ax, elements, y, is_out, panel_unit, n_std))

            order = np.argsort(-deviation[is_out]) if is_out.any() else []
            all_outliers.append(pd.DataFrame({
                "element": elements[is_out][order],
                "panel": panel_name,
                "value": y[is_out][order] / scale,
                "deviation_std": deviation[is_out][order],
            }))

        axes[-1].set_xticks(x)
        axes[-1].set_xticklabels(elements, rotation=90, fontsize=6.5, color=INK_SECONDARY)
        axes[-1].set_xlim(-1, len(elements))
        axes[-1].set_xlabel("element", fontsize=10, color=INK_SECONDARY, labelpad=6)

        axes[0].set_title(
            f"{title_prefix}{QUANTITY_LABELS[quantity]} ({quantity})  \\u00b7  "
            f"red = further than {n_std:g}\\u03c3 from the mean",
            fontsize=13, color=INK, loc="left", pad=12,
        )

        legend_in = 0.3 + 1.6 * max(legend_cols)
        total_w = plot_width + legend_in
        fig.set_size_inches(total_w, 8.2)
        fig.subplots_adjust(
            left=0.85 / total_w, right=1 - legend_in / total_w,
            bottom=0.95 / 8.2, top=1 - 0.55 / 8.2, hspace=0.16,
        )

        return fig, pd.concat(all_outliers, ignore_index=True)

    return (plot_error_profile,)


@app.cell
def _(
    CRITICAL,
    INK,
    INK_MUTED,
    INK_SECONDARY,
    Line2D,
    SERIES,
    SURFACE,
    math,
    np,
):
    def draw_error_panel(ax, x, y, is_out, mean, std, n_std, *, ylabel: str) -> None:
        ax.set_facecolor(SURFACE)
        if np.isfinite(mean) and std > 0:
            ax.axhspan(mean - n_std * std, mean + n_std * std, color=INK_MUTED, alpha=0.10, zorder=0, lw=0)
            ax.axhline(mean, color=INK_MUTED, lw=1, ls="--", dashes=(4, 3), zorder=1)
        if np.nanmin(y) < 0 < np.nanmax(y):
            ax.axhline(0, color=INK_MUTED, lw=1, zorder=1)

        ax.scatter(x[~is_out], y[~is_out], s=34, color=SERIES, lw=0, zorder=3)
        ax.scatter(x[is_out], y[is_out], s=72, color=CRITICAL, edgecolor=SURFACE, linewidth=1.4, zorder=4)

        ax.set_ylabel(ylabel, fontsize=10, color=INK_SECONDARY)
        ax.tick_params(axis="y", labelsize=9, colors=INK_SECONDARY, length=3)
        ax.grid(axis="y", color=INK_MUTED, alpha=0.20, lw=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(INK_MUTED)
            ax.spines[side].set_linewidth(0.8)


    def outlier_legend(ax, elements, y, is_out, unit, n_std) -> int:
        """List each flagged element and its value in a legend beside the panel."""
        suffix = f" {unit}" if unit and unit != "%" else ("%" if unit == "%" else "")
        order = np.argsort(-np.abs(y[is_out] - np.nanmean(y[np.isfinite(y)])))
        labels = [f"{el}  {val:+.3g}{suffix}" for el, val in zip(elements[is_out][order], y[is_out][order])]
        if not labels:
            labels = ["none"]

        handles = [
            Line2D([], [], marker="o", ls="", markersize=6, color=CRITICAL, markeredgecolor=SURFACE, markeredgewidth=1.0)
            for _ in labels
        ]
        ncol = max(1, math.ceil(len(labels) / 12))
        ax.legend(
            handles, labels,
            title=f"outside \u00b1{n_std:g}\u03c3  (n={int(is_out.sum())})",
            loc="upper left", bbox_to_anchor=(1.005, 1.02), ncol=ncol,
            frameon=False, fontsize=7.5, title_fontsize=8.5,
            labelcolor=INK_SECONDARY, handletextpad=0.5,
            columnspacing=1.1, labelspacing=0.35, borderaxespad=0,
        )
        ax.get_legend().get_title().set_color(INK)
        return ncol

    return draw_error_panel, outlier_legend


@app.cell
def _(pd):
    def empty_outliers() -> pd.DataFrame:
        return pd.DataFrame(columns=["element", "panel", "value", "deviation_std"])

    return (empty_outliers,)


@app.cell
def _(mo):
    mo.md(r"""
    #### vs. UFF
    """)
    return


@app.cell
def _(compare_uff, empty_outliers, mo, n_std_slider, plot_error_profile):
    if compare_uff is not None:
        fig_uff_sigma, outliers_uff_sigma = plot_error_profile(compare_uff, "sigma_ang", n_std=n_std_slider.value)
        _output = fig_uff_sigma
    else:
        outliers_uff_sigma = empty_outliers()
        _output = mo.md("*no UFF-labeled entries in this file*")
    _output
    return (outliers_uff_sigma,)


@app.cell
def _(compare_uff, empty_outliers, mo, n_std_slider, plot_error_profile):
    if compare_uff is not None:
        fig_uff_epsilon, outliers_uff_epsilon = plot_error_profile(compare_uff, "epsilon_K", n_std=n_std_slider.value)
        _output = fig_uff_epsilon
    else:
        outliers_uff_epsilon = empty_outliers()
        _output = mo.md("*no UFF-labeled entries in this file*")
    _output
    return (outliers_uff_epsilon,)


@app.cell
def _(mo):
    mo.md(r"""
    #### vs. DREIDING
    """)
    return


@app.cell
def _(compare_dreiding, empty_outliers, mo, n_std_slider, plot_error_profile):
    if compare_dreiding is not None:
        fig_dreiding_sigma, outliers_dreiding_sigma = plot_error_profile(compare_dreiding, "sigma_ang", n_std=n_std_slider.value)
        _output = fig_dreiding_sigma
    else:
        outliers_dreiding_sigma = empty_outliers()
        _output = mo.md("*no DREIDING-labeled entries in this file*")
    _output
    return (outliers_dreiding_sigma,)


@app.cell
def _(compare_dreiding, empty_outliers, mo, n_std_slider, plot_error_profile):
    if compare_dreiding is not None:
        fig_dreiding_epsilon, outliers_dreiding_epsilon = plot_error_profile(compare_dreiding, "epsilon_K", n_std=n_std_slider.value)
        _output = fig_dreiding_epsilon
    else:
        outliers_dreiding_epsilon = empty_outliers()
        _output = mo.md("*no DREIDING-labeled entries in this file*")
    _output
    return (outliers_dreiding_epsilon,)


@app.cell
def _(mo):
    mo.md(r"""
    ### Outlier table (both force fields, current threshold)
    """)
    return


@app.cell
def _(
    outliers_dreiding_epsilon,
    outliers_dreiding_sigma,
    outliers_uff_epsilon,
    outliers_uff_sigma,
    pd,
):
    all_outliers = pd.concat(
        [
            outliers_uff_sigma.assign(force_field="UFF"),
            outliers_uff_epsilon.assign(force_field="UFF"),
            outliers_dreiding_sigma.assign(force_field="DREIDING"),
            outliers_dreiding_epsilon.assign(force_field="DREIDING"),
        ],
        ignore_index=True,
    )
    all_outliers
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Full comparison table
    """)
    return


@app.cell
def _(compare_dreiding, compare_uff, pd):
    combined_comparison = pd.concat(
        [df for df in [compare_uff, compare_dreiding] if df is not None],
        ignore_index=True,
    )
    combined_comparison.filter(
        ["element", "source"]
        + [c for c in combined_comparison.columns if c.endswith(("_provided", "_UFF", "_DREIDING", "_error"))]
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 4. Compare any two force fields directly
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    Compare two arbitrary files element-by-element — e.g.
    `UFF_vincent_2025.dat` against `force_field_mixing_rules.def` — with no
    literature ground truth involved. Entries are matched by element symbol
    and split by their embedded UFF/DREIDING source label, so a DREIDING
    entry is never compared to a UFF entry for the same element; rows whose
    source names neither are excluded (same rule as section 3). If either
    file defines more than one atom type per element, only the first
    occurrence is used (a warning is printed below).
    """)
    return


@app.cell
def _(Path, mo):
    ff_a_browser = mo.ui.file_browser(
        initial_path=Path("ff_data"),
        filetypes=[".def", ".template", ".dat"],
        multiple=False,
        label="browse ff_data/ (force field A)",
    )
    ff_a_upload = mo.ui.file(
        filetypes=[".def", ".template", ".dat"],
        multiple=False,
        label="...or upload (force field A)",
    )
    mo.hstack([ff_a_browser, ff_a_upload], justify="start", gap=2)
    return ff_a_browser, ff_a_upload


@app.cell
def _(Path, mo):
    ff_b_browser = mo.ui.file_browser(
        initial_path=Path("ff_data"),
        filetypes=[".def", ".template", ".dat"],
        multiple=False,
        label="browse ff_data/ (force field B)",
    )
    ff_b_upload = mo.ui.file(
        filetypes=[".def", ".template", ".dat"],
        multiple=False,
        label="...or upload (force field B)",
    )
    mo.hstack([ff_b_browser, ff_b_upload], justify="start", gap=2)
    return ff_b_browser, ff_b_upload


@app.cell
def _(PARSERS, ff_a_browser, ff_a_upload, load_force_field_file, mo):
    ff_a_name, ff_a_suffix, ff_a_raw = load_force_field_file(ff_a_browser, ff_a_upload, PARSERS)
    mo.md(f"Loaded **A = {ff_a_name}** as `{ff_a_suffix}` ({len(ff_a_raw)} entries).")
    return ff_a_name, ff_a_raw


@app.cell
def _(PARSERS, ff_b_browser, ff_b_upload, load_force_field_file, mo):
    ff_b_name, ff_b_suffix, ff_b_raw = load_force_field_file(ff_b_browser, ff_b_upload, PARSERS)
    mo.md(f"Loaded **B = {ff_b_name}** as `{ff_b_suffix}` ({len(ff_b_raw)} entries).")
    return ff_b_name, ff_b_raw


@app.cell
def _(
    compute_errors,
    coverage_summary,
    ff_a_raw,
    ff_b_raw,
    mo,
    split_by_force_field,
):
    a_uff, a_dreiding = split_by_force_field(ff_a_raw)
    b_uff, b_dreiding = split_by_force_field(ff_b_raw)

    pairwise_compare_uff = compute_errors(a_uff, b_uff, "_B") if len(a_uff) and len(b_uff) else None
    pairwise_compare_dreiding = compute_errors(a_dreiding, b_dreiding, "_B") if len(a_dreiding) and len(b_dreiding) else None

    _excluded_a = len(ff_a_raw) - len(a_uff) - len(a_dreiding)
    _excluded_b = len(ff_b_raw) - len(b_uff) - len(b_dreiding)

    mo.md("\n\n".join(filter(None, [
        coverage_summary(a_uff, b_uff, "A vs B, UFF-labeled") if pairwise_compare_uff is not None else None,
        coverage_summary(a_dreiding, b_dreiding, "A vs B, DREIDING-labeled") if pairwise_compare_dreiding is not None else None,
        f"*(A: {_excluded_a} row(s) excluded as neither UFF- nor DREIDING-labeled)*" if _excluded_a else None,
        f"*(B: {_excluded_b} row(s) excluded as neither UFF- nor DREIDING-labeled)*" if _excluded_b else None,
        "**No comparable (same-label) entries found between A and B.**"
        if pairwise_compare_uff is None and pairwise_compare_dreiding is None else None,
    ])))
    return pairwise_compare_dreiding, pairwise_compare_uff


@app.cell
def _(mo):
    n_std_slider_pairwise = mo.ui.slider(
        0.5, 4.0, 0.25, value=1.0, label="outlier threshold (std devs from mean)"
    )
    n_std_slider_pairwise
    return (n_std_slider_pairwise,)


@app.cell
def _(mo):
    mo.md(r"""
    #### UFF-labeled entries: A vs B
    """)
    return


@app.cell
def _(
    empty_outliers,
    ff_a_name,
    ff_b_name,
    mo,
    n_std_slider_pairwise,
    pairwise_compare_uff,
    plot_error_profile,
):
    if pairwise_compare_uff is not None:
        fig_pairwise_uff_sigma, outliers_pairwise_uff_sigma = plot_error_profile(
            pairwise_compare_uff, "sigma_ang", n_std=n_std_slider_pairwise.value,
            title_prefix=f"{ff_a_name} vs {ff_b_name} \u00b7 ",
        )
        _output = fig_pairwise_uff_sigma
    else:
        outliers_pairwise_uff_sigma = empty_outliers()
        _output = mo.md("*no overlapping UFF-labeled entries*")
    _output
    return (outliers_pairwise_uff_sigma,)


@app.cell
def _(
    empty_outliers,
    ff_a_name,
    ff_b_name,
    mo,
    n_std_slider_pairwise,
    pairwise_compare_uff,
    plot_error_profile,
):
    if pairwise_compare_uff is not None:
        fig_pairwise_uff_epsilon, outliers_pairwise_uff_epsilon = plot_error_profile(
            pairwise_compare_uff, "epsilon_K", n_std=n_std_slider_pairwise.value,
            title_prefix=f"{ff_a_name} vs {ff_b_name} \u00b7 ",
        )
        _output = fig_pairwise_uff_epsilon
    else:
        outliers_pairwise_uff_epsilon = empty_outliers()
        _output = mo.md("*no overlapping UFF-labeled entries*")
    _output
    return (outliers_pairwise_uff_epsilon,)


@app.cell
def _(mo):
    mo.md(r"""
    #### DREIDING-labeled entries: A vs B
    """)
    return


@app.cell
def _(
    empty_outliers,
    ff_a_name,
    ff_b_name,
    mo,
    n_std_slider_pairwise,
    pairwise_compare_dreiding,
    plot_error_profile,
):
    if pairwise_compare_dreiding is not None:
        fig_pairwise_dreiding_sigma, outliers_pairwise_dreiding_sigma = plot_error_profile(
            pairwise_compare_dreiding, "sigma_ang", n_std=n_std_slider_pairwise.value,
            title_prefix=f"{ff_a_name} vs {ff_b_name} \u00b7 ",
        )
        _output = fig_pairwise_dreiding_sigma
    else:
        outliers_pairwise_dreiding_sigma = empty_outliers()
        _output = mo.md("*no overlapping DREIDING-labeled entries*")
    _output
    return (outliers_pairwise_dreiding_sigma,)


@app.cell
def _(
    empty_outliers,
    ff_a_name,
    ff_b_name,
    mo,
    n_std_slider_pairwise,
    pairwise_compare_dreiding,
    plot_error_profile,
):
    if pairwise_compare_dreiding is not None:
        fig_pairwise_dreiding_epsilon, outliers_pairwise_dreiding_epsilon = plot_error_profile(
            pairwise_compare_dreiding, "epsilon_K", n_std=n_std_slider_pairwise.value,
            title_prefix=f"{ff_a_name} vs {ff_b_name} \u00b7 ",
        )
        _output = fig_pairwise_dreiding_epsilon
    else:
        outliers_pairwise_dreiding_epsilon = empty_outliers()
        _output = mo.md("*no overlapping DREIDING-labeled entries*")
    _output
    return (outliers_pairwise_dreiding_epsilon,)


@app.cell
def _(mo):
    mo.md(r"""
    ### Outlier table (A vs B, current threshold)
    """)
    return


@app.cell
def _(
    outliers_pairwise_dreiding_epsilon,
    outliers_pairwise_dreiding_sigma,
    outliers_pairwise_uff_epsilon,
    outliers_pairwise_uff_sigma,
    pd,
):
    pairwise_outliers = pd.concat(
        [
            outliers_pairwise_uff_sigma.assign(label="UFF"),
            outliers_pairwise_uff_epsilon.assign(label="UFF"),
            outliers_pairwise_dreiding_sigma.assign(label="DREIDING"),
            outliers_pairwise_dreiding_epsilon.assign(label="DREIDING"),
        ],
        ignore_index=True,
    )
    pairwise_outliers
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Full comparison table (A vs B)
    """)
    return


@app.cell
def _(pairwise_compare_dreiding, pairwise_compare_uff, pd):
    pairwise_comparison = pd.concat(
        [df for df in [pairwise_compare_uff, pairwise_compare_dreiding] if df is not None],
        ignore_index=True,
    )
    pairwise_comparison.filter(
        ["element", "source_provided", "source_B"]
        + [c for c in pairwise_comparison.columns if c.endswith(("_provided", "_B", "_error"))]
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 5. Export
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    Load any file dealt with in this notebook — including the two
    ground-truth CSVs — and re-export it in any of the four formats
    (`.def`, `.template`, `.dat`, `.csv`). Exports always land in
    `ff_data/exports/`; the filename defaults to the source file's name
    with the new extension, and is editable before exporting.
    """)
    return


@app.cell
def _(Path, mo):
    export_browser = mo.ui.file_browser(
        initial_path=Path("ff_data"),
        filetypes=[".def", ".template", ".dat", ".csv"],
        multiple=False,
        label="browse ff_data/ (file to export)",
    )
    export_upload = mo.ui.file(
        filetypes=[".def", ".template", ".dat", ".csv"],
        multiple=False,
        label="...or upload",
    )
    mo.hstack([export_browser, export_upload], justify="start", gap=2)
    return export_browser, export_upload


@app.cell
def _(
    EXPORT_SOURCE_PARSERS,
    export_browser,
    export_upload,
    load_force_field_file,
    mo,
):
    export_source_name, export_source_suffix, export_source_df = load_force_field_file(
        export_browser, export_upload, EXPORT_SOURCE_PARSERS
    )
    mo.md(f"Loaded **{export_source_name}** as `{export_source_suffix}` ({len(export_source_df)} entries).")
    return export_source_df, export_source_name


@app.cell
def _(EXPORTERS, mo):
    export_format_dropdown = mo.ui.dropdown(
        options=list(EXPORTERS.keys()), value=".csv", label="export format"
    )
    export_format_dropdown
    return (export_format_dropdown,)


@app.cell
def _(Path, export_format_dropdown, export_source_name, mo):
    _default_export_name = Path(export_source_name).stem + export_format_dropdown.value
    export_filename_input = mo.ui.text(
        value=_default_export_name, label="export filename (editable)", full_width=True
    )
    export_filename_input
    return (export_filename_input,)


@app.cell
def _(mo):
    export_button = mo.ui.run_button(label="Export to ff_data/exports/")
    export_button
    return (export_button,)


@app.cell
def _(
    EXPORTERS,
    Path,
    export_button,
    export_filename_input,
    export_format_dropdown,
    export_source_df,
    mo,
):
    EXPORTS_DIR = Path("ff_data/exports")

    _target_suffix = export_format_dropdown.value
    _safe_stem = Path(export_filename_input.value).stem
    export_output_path = EXPORTS_DIR / f"{_safe_stem}{_target_suffix}"

    _overwrite_note = " (overwriting existing file)" if export_output_path.exists() else ""

    mo.stop(
        not export_button.value,
        mo.md(
            f"Will write **{export_output_path}**{_overwrite_note} — "
            "click *Export* above to proceed."
        ),
    )

    EXPORTS_DIR.mkdir(parents=True, exist_ok=True)
    _export_text = EXPORTERS[_target_suffix](export_source_df)
    export_output_path.write_text(_export_text)

    mo.vstack([
        mo.md(f"**Wrote {export_output_path}** ({len(_export_text)} bytes, {len(export_source_df)} entries)."),
        mo.plain_text("\n".join(_export_text.splitlines()[:15])),
    ])
    return


if __name__ == "__main__":
    app.run()
