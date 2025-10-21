# university_dual_compare_section.py
# Streamlit: Dual-degree comparison with independent (left/right) University & Faculty filters.
# Enhancements:
#   - Main bar chart: distinct colors + tighter bars.
#   - Extra comparison charts: consistent colors + legend at top-right (not bottom).
#
# Data source priority:
#   1) DATA/university_outcomes_2024_tidy.csv
#   2) /mnt/data/university_outcomes_2024_tidy.csv
#   3) file uploader (CSV)
#
# Scope:
#   - Keep only NTU/NUS/SMU/SUSS
#   - Remove blank rows / rows with all metrics missing
#   - English UI & comments

from functools import lru_cache
from typing import List
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

DEFAULT_CSV_PATHS = [
    "DATA/university_outcomes_2024_tidy.csv",
    "/mnt/data/university_outcomes_2024_tidy.csv",
]
ALLOWED_UNIS = {"NTU", "NUS", "SMU", "SUSS"}

NUMERIC_COLS = [
    "Employed (%)",
    "In Full-Time Permanent Employment (%)",
    "Gross Mean ($)",
    "Gross Median ($)",
    "Gross 25th Percentile ($)",
    "Gross 75th Percentile ($)",
]

# ---------------------------
# Utilities
# ---------------------------

def _ui_segmented(label: str, options: List[str], default: str) -> str:
    """Use segmented control if available; fallback to horizontal radio."""
    try:
        return st.segmented_control(label, options=options, default=default)
    except Exception:
        return st.radio(label, options=options, index=options.index(default), horizontal=True)

def _pill_buttons(label: str, options: List[str], default: str) -> str:
    """Render pill-like buttons (stateful)."""
    key = f"pill_{label}"
    if key not in st.session_state:
        st.session_state[key] = default
    cols = st.columns(len(options))
    for i, opt in enumerate(options):
        with cols[i]:
            if st.button(opt, use_container_width=True, key=f"{key}_{opt}"):
                st.session_state[key] = opt
    st.caption(f"Current metric: **{st.session_state[key]}**")
    return st.session_state[key]

def _coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    for c in NUMERIC_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def _drop_empty_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Remove rows where DegreeStd is blank OR all numeric metrics are NaN."""
    df = df.copy()
    df["DegreeStd"] = df["DegreeStd"].astype(str).str.strip()
    df.loc[df["DegreeStd"].isin(["", "nan", "NaN"]), "DegreeStd"] = np.nan
    df = df.dropna(subset=["DegreeStd"])

    metrics = [c for c in NUMERIC_COLS if c in df.columns]
    if metrics:
        df = df.loc[~df[metrics].isna().all(axis=1)]
    df = df.dropna(how="all")
    return df

@lru_cache(maxsize=1)
def _load_csv(path_hint: str = "") -> pd.DataFrame:
    """Load tidy CSV from known locations or uploader; return numeric-typed frame."""
    tried = []
    if path_hint:
        tried.append(path_hint)
    tried.extend([p for p in DEFAULT_CSV_PATHS if p not in tried])

    last_err = None
    for p in tried:
        try:
            df = pd.read_csv(p)
            return _coerce_numeric(df)
        except Exception as e:
            last_err = e

    upl = st.file_uploader("Upload tidy CSV (university_outcomes_2024_tidy.csv)", type=["csv"])
    if upl is not None:
        df = pd.read_csv(upl)
        return _coerce_numeric(df)

    if last_err:
        st.error(f"Failed to read CSV: {last_err}")
    return pd.DataFrame()

def _prep_base(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to allowed universities, drop blanks, and build display/key columns."""
    if df.empty:
        return df
    df = df.copy()
    df["University"] = df["University"].astype(str).str.upper().str.strip()
    df = df[df["University"].isin(ALLOWED_UNIS)].copy()
    if df.empty:
        return df

    df = _drop_empty_rows(df)

    # Build a unique display label to avoid ambiguity:
    #   "University · Faculty · Degree"
    df["FacultyStd"] = df["FacultyStd"].astype(str).str.strip()
    df["DegreeStd"]  = df["DegreeStd"].astype(str).str.strip()
    df["DisplayLabel"] = df["University"] + " · " + df["FacultyStd"] + " · " + df["DegreeStd"]

    # Machine-friendly key
    df["Key"] = df["University"] + "||" + df["FacultyStd"] + "||" + df["DegreeStd"]
    return df

def _metric_col_salary(choice: str) -> str:
    return {
        "Mean ($)": "Gross Mean ($)",
        "Median ($)": "Gross Median ($)",
        "25th Percentile ($)": "Gross 25th Percentile ($)",
        "75th Percentile ($)": "Gross 75th Percentile ($)",
    }[choice]

def _metric_col_employment(choice: str) -> str:
    return {
        "Employed (%)": "Employed (%)",
        "Full-Time Permanent (%)": "In Full-Time Permanent Employment (%)",
    }[choice]

def _degree_options_for_side(df: pd.DataFrame, uni: str, fac: str) -> pd.DataFrame:
    """Return candidate rows for a side given university and faculty filters."""
    d = df[df["University"] == uni].copy()
    if fac != "(All)":
        d = d[d["FacultyStd"] == fac]
    return d

def _default_index_safe(options: List[str], desired: int) -> int:
    """Clamp default index to options length."""
    if not options:
        return 0
    desired = max(0, desired)
    return min(desired, len(options)-1)

# ---------------------------
# Main section
# ---------------------------

def render_university_dual_compare_section():
    st.markdown("### 🎓 Dual Degree Comparison — NTU · NUS · SMU · SUSS (2024)")
    st.write("Each side has its **own University & Faculty** filters. Select one degree on the left and one on the right, then compare **Gross salary** or **Employment** metrics.")

    # Load data
    df = _load_csv()
    if df.empty:
        st.info("Please provide or confirm `university_outcomes_2024_tidy.csv`.")
        return

    required = {"University","FacultyStd","DegreeStd"}
    if not required.issubset(df.columns):
        st.error(f"CSV missing required columns: {sorted(required - set(df.columns))}")
        with st.expander("Columns present"):
            st.write(list(df.columns))
        return

    df = _prep_base(df)
    if df.empty:
        st.warning("No rows after filtering to NTU/NUS/SMU/SUSS and cleaning blanks.")
        return

    # Optional top-level filter limiting the candidate pool
    uni_pool_opts = sorted(df["University"].unique().tolist())
    sel_pool_unis = st.multiselect("Restrict candidate universities (optional)", uni_pool_opts, default=uni_pool_opts)
    if not sel_pool_unis:
        st.info("Select at least one university.")
        return
    df_pool = df[df["University"].isin(sel_pool_unis)].copy()

    # ---------------------------
    # Left and Right selectors
    # ---------------------------
    left, right = st.columns(2, vertical_alignment="top")

    # LEFT
    with left:
        st.subheader("Left selection")
        uni_left = st.selectbox("University (Left)", sorted(df_pool["University"].unique().tolist()), key="uni_left")
        fac_left_opts = ["(All)"] + sorted(df_pool[df_pool["University"] == uni_left]["FacultyStd"].dropna().unique().tolist())
        fac_left = st.selectbox("Faculty (Left)", fac_left_opts, key="fac_left")

        df_left_cands = _degree_options_for_side(df_pool, uni_left, fac_left)
        if df_left_cands.empty:
            st.warning("No degree options for the current Left filters.")
            return

        left_labels = df_left_cands["DisplayLabel"].tolist()
        left_default_idx = 0
        deg_left_label = st.selectbox("Degree (Left)", left_labels, index=_default_index_safe(left_labels, left_default_idx), key="deg_left_label")
        left_row = df_left_cands[df_left_cands["DisplayLabel"] == deg_left_label].iloc[0]

    # RIGHT
    with right:
        st.subheader("Right selection")
        uni_right = st.selectbox("University (Right)", sorted(df_pool["University"].unique().tolist()), key="uni_right")
        fac_right_opts = ["(All)"] + sorted(df_pool[df_pool["University"] == uni_right]["FacultyStd"].dropna().unique().tolist())
        fac_right = st.selectbox("Faculty (Right)", fac_right_opts, key="fac_right")

        df_right_cands = _degree_options_for_side(df_pool, uni_right, fac_right)
        if df_right_cands.empty:
            st.warning("No degree options for the current Right filters.")
            return

        right_labels = df_right_cands["DisplayLabel"].tolist()
        right_default_idx = 1 if len(right_labels) > 1 else 0
        if right_default_idx < len(right_labels) and right_labels[right_default_idx] == left_labels[0] and len(right_labels) > 1:
            right_default_idx = 1
        deg_right_label = st.selectbox("Degree (Right)", right_labels, index=_default_index_safe(right_labels, right_default_idx), key="deg_right_label")
        right_row = df_right_cands[df_right_cands["DisplayLabel"] == deg_right_label].iloc[0]

    # Guard against identical selection
    if left_row["Key"] == right_row["Key"]:
        st.warning("Left and Right selections are identical. Please choose two different degrees.")
        return

    # Build a two-row comparison frame
    cmp = pd.concat([left_row.to_frame().T, right_row.to_frame().T], ignore_index=True)

    # ---------------------------
    # Metric controls
    # ---------------------------
    metric_type = _ui_segmented("Metric Type", ["Salary (Gross)", "Employment"], "Salary (Gross)")
    if metric_type == "Salary (Gross)":
        choice = _pill_buttons("gross", ["Mean ($)", "Median ($)", "25th Percentile ($)", "75th Percentile ($)"], "Median ($)")
        metric_col = {
            "Mean ($)": "Gross Mean ($)",
            "Median ($)": "Gross Median ($)",
            "25th Percentile ($)": "Gross 25th Percentile ($)",
            "75th Percentile ($)": "Gross 75th Percentile ($)",
        }[choice]
        y_label = f"{choice} — Gross Monthly ($)"
    else:
        choice = _pill_buttons("employment", ["Employed (%)", "Full-Time Permanent (%)"], "Employed (%)")
        metric_col = {
            "Employed (%)": "Employed (%)",
            "Full-Time Permanent (%)": "In Full-Time Permanent Employment (%)",
        }[choice]
        y_label = choice

    if metric_col not in cmp.columns:
        st.error(f"Selected metric column is missing: {metric_col}")
        with st.expander("Columns present in selection"):
            st.write(list(cmp.columns))
        return

    # ---------------------------
    # Color mapping (consistent across all charts)
    # ---------------------------
    # Use a qualitative palette with clear contrast
    palette = px.colors.qualitative.Set2
    left_label  = cmp["DisplayLabel"].iloc[0]
    right_label = cmp["DisplayLabel"].iloc[1]
    color_map = {
        left_label:  palette[0],
        right_label: palette[1],
    }

    # ---------------------------
    # Main comparison chart (2 bars, distinct colors, tight spacing)
    # ---------------------------
    plot_df = cmp[["DisplayLabel", metric_col]].rename(columns={"DisplayLabel": "Degree / Programme", metric_col: "Value"})
    title_unis = sorted({cmp["University"].iloc[0], cmp["University"].iloc[1]})
    title_facets = "Left: " + cmp["FacultyStd"].iloc[0] + " | Right: " + cmp["FacultyStd"].iloc[1]

    fig = px.bar(
        plot_df,
        x="Degree / Programme",
        y="Value",
        color="Degree / Programme",  # ensures different colors for the two bars
        color_discrete_map=color_map,
        text="Value",
        title=f"{' / '.join(title_unis)} — {title_facets} — {y_label}",
        hover_name="Degree / Programme",
        labels={"Value": y_label, "Degree / Programme": "University · Faculty · Degree"},
    )
    # Tight spacing: small gaps; hide legend (redundant for 2 bars with labels)
    fig.update_layout(
        bargap=0.05,         # gap between categories
        bargroupgap=0.02,    # gap within same category groups
        showlegend=False,
        margin=dict(l=10, r=10, t=50, b=10),
        height=420
    )
    fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
    st.plotly_chart(fig, use_container_width=True)

    # ---------------------------
    # Extra comparisons (legend below, no x-axis ticks)
    # ---------------------------
    st.markdown("#### 📊 Extra comparisons")

    # Place legend below the plot
    legend_bottom = dict(
        orientation="h",
        yanchor="top", y=-0.20,  # push below plotting area
        xanchor="center", x=0.50
    )

    def _style_extra(fig):
        # remove x-axis ticks and title; add bottom padding for legend
        fig.update_xaxes(showticklabels=False, title_text=None)
        fig.update_layout(
            legend=legend_bottom,
            barmode="group",
            bargap=0.05, bargroupgap=0.02,
            margin=dict(l=10, r=10, t=40, b=110),
            height=320
        )
        return fig

    sub1, sub2 = st.columns(2)

    # Overall Employment
    if "Employed (%)" in cmp.columns:
        with sub1:
            df_emp = cmp[["DisplayLabel", "Employed (%)"]].rename(
                columns={"DisplayLabel": "Degree / Programme", "Employed (%)": "Rate"}
            )
            fig_emp = px.bar(
                df_emp,
                x="Degree / Programme", y="Rate",
                color="Degree / Programme",
                color_discrete_map=color_map,  # same colors as main chart
                title="Overall Employment Rate (%)",
                text="Rate",
                labels={"Rate": "Employed (%)", "Degree / Programme": "University · Faculty · Degree"},
            )
            # Format as % if data is in 0–1
            if df_emp["Rate"].max() <= 1.0:
                fig_emp.update_traces(texttemplate="%{text:.2%}")
                fig_emp.update_yaxes(tickformat=".0%")
            else:
                fig_emp.update_traces(texttemplate="%{text:.2f}")
                fig_emp.update_yaxes(tickformat=",.0f")

            st.plotly_chart(_style_extra(fig_emp), use_container_width=True)

    # Full-Time Permanent
    if "In Full-Time Permanent Employment (%)" in cmp.columns:
        with sub2:
            df_ftp = cmp[["DisplayLabel", "In Full-Time Permanent Employment (%)"]].rename(
                columns={"DisplayLabel": "Degree / Programme", "In Full-Time Permanent Employment (%)": "Rate"}
            )
            fig_ftp = px.bar(
                df_ftp,
                x="Degree / Programme", y="Rate",
                color="Degree / Programme",
                color_discrete_map=color_map,
                title="Full-Time Permanent Employment Rate (%)",
                text="Rate",
                labels={"Rate": "Full-Time Permanent (%)", "Degree / Programme": "University · Faculty · Degree"},
            )
            if df_ftp["Rate"].max() <= 1.0:
                fig_ftp.update_traces(texttemplate="%{text:.2%}")
                fig_ftp.update_yaxes(tickformat=".0%")
            else:
                fig_ftp.update_traces(texttemplate="%{text:.2f}")
                fig_ftp.update_yaxes(tickformat=",.0f")

            st.plotly_chart(_style_extra(fig_ftp), use_container_width=True)

    # ---------------------------
    # Details table (2 rows)
    # ---------------------------
    show_cols = ["University","FacultyStd","DegreeStd"] + [c for c in NUMERIC_COLS if c in cmp.columns]
    st.markdown("#### Details (selected degrees)")
    st.dataframe(cmp[show_cols].reset_index(drop=True), use_container_width=True)

# Standalone run
if __name__ == "__main__":
    st.set_page_config(page_title="Dual Degree Comparison (NTU/NUS/SMU/SUSS)", layout="wide")
    render_university_dual_compare_section()
