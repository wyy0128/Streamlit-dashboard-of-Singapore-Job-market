# -*- coding: utf-8 -*-
"""
SG Fresh Grads Dashboard
Patch for "Employment — Industry & Recruitment/Resignation" filters:
- Industry levels now come from Job_vacancy with your explicit mapping:
  Level 1 = Total
  Level 2 = Manufacturing, Construction, Services, Transportation And Storage, Accommodation And Food Services,
            Information And Communications, Financial And Insurance Services, Real Estate Services, Professional Services,
            Administrative And Support Services, Community, Social And Personal Services, Others
  Level 3 = others
- Options list for both the multi-line 'Rates' chart and the single 'Ratio' chart =
  (industries at the chosen level from Job_vacancy) ∩ (industries that appear in BOTH monthly recruitment & resignation tables)
- Stronger normalization and de-duplication ensure selecting any industry always yields data if it exists.
"""

import os, re, numpy as np, pandas as pd, plotly.express as px, streamlit as st
from dateutil import parser as date_parser

from University_Dashboard import render_university_dual_compare_section



MUTE_DEBUG_DUMPS = True
if MUTE_DEBUG_DUMPS:
    def _st_noop(*args, **kwargs):
        return None
    try:
        st.write = _st_noop
        st.json = _st_noop
    except Exception:
        pass

# ---- FIRST Streamlit command
st.set_page_config(page_title="SG Fresh Grads Dashboard", layout="wide", page_icon="📊")

# ---------- Visual defaults ----------
px.defaults.template = "plotly_white"
px.defaults.width = 900
px.defaults.height = 420
APP_FONT = "Inter, -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, Helvetica, Arial, Apple Color Emoji, Segoe UI Emoji"

# Inject light CSS
st.markdown(f"""
<style>
html, body, [class*="css"]  {{ font-family: {APP_FONT}; }}
.metric-card {{
  padding: 12px 14px; border-radius: 14px; background: #f8fafc; border: 1px solid #eef2f7;
  box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}}
.small-note {{ color:#64748b; font-size:12px; margin-top:4px; }}
.block-title {{ font-weight:600; font-size:18px; margin: 4px 0 8px 0; }}
section[data-testid="stSidebar"] .block-container {{ padding-top: 1rem; }}
</style>
""", unsafe_allow_html=True)

# ---------- Config ----------
DEFAULT_DATA_DIR = "./data"
FILES = {
    "job_vacancy": "Job_vacancy.csv",
    "unemployment": "Unemployment.csv",
    "longterm_unemployment": "Longterm_Unemployment.csv",
    "monthly_recruitment": "Monthly_recruitment_rate.csv",
    "monthly_resignation": "Monthly_resignment_rate.csv",
    "income_occ_sex": "Monthly Income by occupations and sex.csv",
    "income_sex_age": "FT_Res_income_sex_age.csv",
    "p20p50": "FT_Res_income_P20_P50.csv",
    "wage_total": "mrsd_16_Total_wage_change.csv",
    "wage_basic": "mrsd_17_basic_wage-change.csv",
}
INDUSTRY_Q_START, INDUSTRY_Q_END = "2006Q1", "2025Q2"

# ---------- Helpers ----------
def _is_number_like(s):
    try:
        float(str(s).replace(',', '').replace('%','').strip()); return True
    except: return False
def _to_number(s):
    try:
        s = str(s).replace(',', '').replace('%','').strip(); return float(s) if s!="" else np.nan
    except: return np.nan

def _parse_quarter_label(label):
    if label is None: return pd.NaT
    s = str(label).strip().upper(); m = re.match(r"^(\d{4})\s*([1-4])Q$", s)
    if m: return pd.Period(f"{m.group(1)}Q{m.group(2)}")
    try:  return date_parser.parse(s, dayfirst=False, yearfirst=True)
    except: return pd.NaT

def _parse_year_label(label):
    s = str(label).strip(); m = re.match(r"^(\d{4})$", s)
    if m:  return pd.to_datetime(f"{m.group(1)}-01-01")
    try:   return date_parser.parse(s, dayfirst=False, yearfirst=True)
    except: return pd.NaT

def _resolve_path(data_dir, filename):
    p = os.path.join(data_dir, filename)
    if os.path.exists(p): return p
    target = re.sub(r"[\s\-_]", "", os.path.splitext(filename)[0].lower())
    # search in data_dir
    if os.path.isdir(data_dir):
        for f in os.listdir(data_dir):
            stem = re.sub(r"[\s\-_]", "", os.path.splitext(f)[0].lower())
            if f.lower().endswith((".csv",".xlsx")) and (stem==target or target in stem or stem in target):
                return os.path.join(data_dir, f)
    # search beside script
    here = os.path.dirname(os.path.abspath(__file__))
    for f in os.listdir(here):
        stem = re.sub(r"[\s\-_]", "", os.path.splitext(f)[0].lower())
        if f.lower().endswith((".csv",".xlsx")) and (stem==target or target in stem or stem in target):
            return os.path.join(here, f)
    return None

@st.cache_data(show_spinner=False)
def read_raw_csv(path):
    if not path or not os.path.exists(path): return None, f"File not found: {path}"
    try:
        df = pd.read_csv(path, header=None, engine='python', dtype=str, keep_default_na=False); return df, None
    except Exception as e:
        return None, f"Failed to read {path}: {e}"

@st.cache_data(show_spinner=False)
def read_csv_headered(path):
    if not path or not os.path.exists(path): return None, f"File not found: {path}"
    try:
        df = pd.read_csv(path, dtype=str, keep_default_na=False); return df, None
    except Exception as e:
        return None, f"Failed to read {path}: {e}"

def find_data_series_header(df, max_scan=90):
    if df is None: return None
    for i in range(min(max_scan, len(df))):
        row = df.iloc[i].astype(str).str.strip().tolist()
        if any(cell.lower() == "data series" for cell in row[:10]):
            return i
    return None

def parse_wide_table(df_raw, header_row, kind="quarter"):
    header = [str(h).strip() for h in df_raw.iloc[header_row].tolist()]
    if not header: return pd.DataFrame()
    header[0] = "Data Series"
    data = df_raw.iloc[header_row+1:].copy()
    def is_date_header(h):
        s = str(h).strip().upper()
        return bool(re.match(r"^\d{4}$", s)) or bool(re.match(r"^\d{4}\s*[1-4]Q$", s))
    date_cols_idx = [j for j, h in enumerate(header) if is_date_header(h)] or \
                    [j for j, h in enumerate(header) if re.search(r"\d{4}", str(h))]
    d = data.copy(); d.columns = [f"col_{i}" for i in range(d.shape[1])]
    cols_map = {"col_0":"Data Series"}; [cols_map.setdefault(f"col_{j}", header[j]) for j in date_cols_idx]
    d = d.loc[:, list(cols_map.keys())].rename(columns=cols_map)
    d = d[d["Data Series"].astype(str).str.strip()!=""]
    d = d[~d["Data Series"].str.contains(r"^Table Title|^Subject|^Theme", case=False, na=False)]
    date_cols = [c for c in d.columns if c!="Data Series"]
    d = d[d[date_cols].applymap(_is_number_like).any(axis=1)]
    long = d.melt(id_vars=["Data Series"], var_name="period", value_name="value")
    long["value"] = long["value"].apply(_to_number)
    long["date"]  = long["period"].apply(_parse_quarter_label if kind=="quarter" else _parse_year_label)
    long["series"] = long["Data Series"].astype(str)
    long = long.drop(columns=["Data Series"]).dropna(subset=["date"])
    return long

# ---- Time helpers ----
def _to_quarter_end_timestamp(x):
    if isinstance(x, pd.Period):
        try:
            if x.freqstr and x.freqstr.upper().startswith('Q'): return x.to_timestamp(how='end')
            if x.freqstr and (x.freqstr.upper().startswith('A') or x.freqstr.upper().startswith('Y')):
                return pd.Timestamp(year=x.year, month=12, day=31)
        except: return pd.NaT
    ts = pd.to_datetime(x, errors='coerce')
    if pd.isna(ts): return pd.NaT
    q = ((ts.month-1)//3)+1; end_month = q*3
    return pd.Timestamp(year=ts.year, month=end_month, day=1) + pd.offsets.MonthEnd(0)

def _to_datetime_like(x):
    if isinstance(x, pd.Period):
        try:
            if x.freqstr and x.freqstr.upper().startswith('Q'):
                return x.to_timestamp(how='end')
            if x.freqstr and (x.freqstr.upper().startswith('A') or x.freqstr.upper().startswith('Y')):
                return pd.Timestamp(year=x.year, month=12, day=31)
        except:
            pass
    return pd.to_datetime(x, errors='coerce')

def _as_quarter_period(x):
    if isinstance(x, pd.Period):
        return x.asfreq('Q-DEC') if x.freqstr and x.freqstr.upper().startswith('Q') else pd.Period(f"{x.year}Q4", freq="Q-DEC")
    ts = pd.to_datetime(x, errors='coerce')
    return ts.to_period('Q-DEC') if not pd.isna(ts) else pd.NaT

def filter_by_quarter_range(df, start_q=INDUSTRY_Q_START, end_q=INDUSTRY_Q_END, date_col="date"):
    if df is None or len(df)==0 or date_col not in df.columns: return df
    d = df.copy(); d["__q__"] = d[date_col].apply(_as_quarter_period)
    try:
        qs = pd.Period(start_q, freq="Q-DEC") if start_q else d["__q__"].min()
        qe = pd.Period(end_q,   freq="Q-DEC") if end_q   else d["__q__"].max()
    except: qs, qe = d["__q__"].min(), d["__q__"].max()
    return d[(d["__q__"]>=qs) & (d["__q__"]<=qe)].drop(columns=["__q__"])

# ---- Plot helpers ----
def _sanitize_for_plotly(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or len(df)==0: return df
    d = df.copy()
    for c in d.columns:
        try:
            if isinstance(d[c].dtype, pd.PeriodDtype) or d[c].apply(lambda v: isinstance(v, pd.Period)).any():
                d[c] = d[c].astype(str)
        except: pass
    return d

def format_fig(fig, legend="top"):
    if fig is None: return None
    if legend == "top":
        legend_cfg = dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)
    elif legend == "bottom":
        legend_cfg = dict(orientation="h", yanchor="top", y=-0.2, xanchor="left", x=0)
    else:  # right
        legend_cfg = dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02)
    fig.update_layout(margin=dict(l=10,r=10,t=40,b=10), legend=legend_cfg, font=dict(family=APP_FONT, size=13))
    return fig

def line_trend(df, x="date", y="value", color=None, title=None, legend="top", force_datetime_x=False):
    if df is None or len(df)==0 or y not in df.columns or x not in df.columns: return None
    d = df.copy()
    if force_datetime_x:
        d["Time"] = d[x].apply(_to_datetime_like)
        d = d.dropna(subset=["Time"])
        d = d.sort_values("Time")
        fig = px.line(d, x="Time", y=y, color=color, title=title)
        fig.update_xaxes(type="date")
    else:
        d = d.sort_values(x)
        fig = px.line(d, x=x, y=y, color=color, title=title)
    return format_fig(fig, legend)

def heatmap(df, x, y, z, title=None, agg="mean", legend="top"):
    if df is None or len(df)==0 or not all(c in df.columns for c in [x,y,z]): return None
    d = _sanitize_for_plotly(df)
    aggfunc = "mean" if agg=="mean" else "sum"
    pivot = d.pivot_table(index=y, columns=x, values=z, aggfunc=aggfunc)
    pivot = pivot.sort_index()
    fig = px.imshow(pivot, aspect="auto", title=title, text_auto=False, origin="lower")
    return format_fig(fig, legend)

# ---------- Label cleaning ----------
BRACKET_RANGE_RE = re.compile(r"\[\s*[0-9<>=+−–—][^]]*\]")
def clean_label(s):
    if s is None: return s
    s = str(s)
    s = s.replace("\n"," ").replace("\r"," ")
    s = s.replace("–","-").replace("—","-").replace("−","-")
    s = BRACKET_RANGE_RE.sub("", s).strip()
    s = re.sub(r"\s{2,}", " ", s)
    s = re.sub(r"\s*[-:|]\s*$", "", s)
    return s
def clean_series_labels(df, cols):
    d = df.copy()
    for c in cols:
        if c in d.columns:
            d[c] = d[c].apply(clean_label)
    return d

# ---------- Industry hierarchy mapping ----------
def norm_key(s: str) -> str:
    """Normalize industry name to a matching key."""
    if s is None: return ""
    s = str(s)
    s = s.replace("&", "And")
    s = re.sub(r"[，、]", ",", s)   # chinese comma to ascii
    s = re.sub(r"\s+", " ", s).strip()
    return s

LEVEL2_SET = set([
    "Manufacturing",
    "Construction",
    "Services",
    "Transportation And Storage",
    "Accommodation And Food Services",
    "Information And Communications",
    "Financial And Insurance Services",
    "Real Estate Services",
    "Professional Services",
    "Administrative And Support Services",
    "Community, Social And Personal Services",
    "Others",
])

def level_from_name(name_clean: str) -> int:
    if name_clean.lower() == "total":
        return 1
    return 2 if name_clean in LEVEL2_SET else 3

def build_levels_from_job_vacancy(jv: pd.DataFrame) -> pd.DataFrame:
    """Build industry levels using user's rule from Job_vacancy."""
    u = (jv[["industry"]].dropna().drop_duplicates().copy())
    u["industry_clean"] = u["industry"].astype(str).str.strip()
    u["industry_key"] = u["industry_clean"].apply(norm_key)
    u["level"] = u["industry_clean"].apply(level_from_name).astype(int)
    u = u.drop_duplicates("industry_key", keep="first")
    return u[["industry_key","industry_clean","level"]]

def apply_levels_by_jv(df: pd.DataFrame, series_col: str, jv_levels: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["industry"] = d[series_col].astype(str).str.strip()
    d["industry_key"] = d["industry"].apply(norm_key)
    d = d.merge(jv_levels, on="industry_key", how="left")
    # 未命中（极少数）默认 3 级
    d["level"] = d["level"].fillna(3).astype(int)
    return d

# ---------- Dedup helpers ----------
def dedup_first(df, series_col="series", date_col="date"):
    """Keep only FIRST occurrence per (series, date) preserving original order."""
    if df is None or len(df)==0: return df
    d = df.copy()
    d["__ord__"] = np.arange(len(d))
    d = d.sort_values([series_col, date_col, "__ord__"]).drop_duplicates(subset=[series_col, date_col], keep="first")
    d = d.drop(columns=["__ord__"])
    return d

# ---------- Loaders ----------
@st.cache_data(show_spinner=False)
def read_all(data_dir):
    status_rows = []; out = {"employment":{}, "income":{}}

    def load_emp_generic(key, kind, rename_map):
        path = _resolve_path(data_dir, FILES[key]); ok=False; err=None
        raw, err_read = read_raw_csv(path)
        if raw is not None:
            hdr = find_data_series_header(raw)
            if hdr is not None:
                long = parse_wide_table(raw, hdr, kind=kind).rename(columns=rename_map)
                out["employment"][key] = long; ok=True
            else:
                err="Header 'Data Series' not found."
        else:
            err = err_read
        status_rows.append({"group":"employment","item":key,"file":FILES[key],"resolved_path":path,"ok":ok,"error":err})

    # Job vacancies (quarterly)
    load_emp_generic("job_vacancy","quarter",{"series":"industry","value":"vacancies"})

    # Monthly recruitment / resignation (quarterly in file)
    load_emp_generic("monthly_recruitment","quarter",{"series":"series","value":"recruitment_rate"})
    load_emp_generic("monthly_resignation","quarter",{"series":"series","value":"resignation_rate"})

    # --- De-duplicate monthly tables (keep first per series+date) ---
    if out["employment"].get("monthly_recruitment") is not None:
        out["employment"]["monthly_recruitment"] = dedup_first(out["employment"]["monthly_recruitment"], "series", "date")
    if out["employment"].get("monthly_resignation") is not None:
        out["employment"]["monthly_resignation"] = dedup_first(out["employment"]["monthly_resignation"], "series", "date")

    # Unemployment (annual, first ~15 rows)
    key="unemployment"; path = _resolve_path(data_dir, FILES[key]); ok=False; err=None
    raw, err_read = read_raw_csv(path)
    if raw is not None:
        hdr = find_data_series_header(raw, max_scan=15) or 0
        try:
            unemp_long = parse_wide_table(raw, hdr, kind="annual").rename(columns={"value":"unemployment_rate"})
            out["employment"][key] = unemp_long; ok=True
        except Exception as e:
            err = f"Parse error: {e}"
    else:
        err = err_read
    status_rows.append({"group":"employment","item":"unemployment","file":FILES["unemployment"],
                        "resolved_path": path, "ok": ok, "error": err})

    # Long-term unemployment (annual)
    load_emp_generic("longterm_unemployment","annual",{"value":"longterm_rate"})

    # Income — occ × sex
    path = _resolve_path(data_dir, FILES["income_occ_sex"]); ok=False; err=None
    raw, err_read = read_raw_csv(path)
    if raw is not None:
        hdr = find_data_series_header(raw)
        if hdr is not None:
            inc1 = parse_wide_table(raw, hdr, kind="annual").rename(columns={"value":"income"})
            # split "Occupation - Sex"
            def split_occ_sex(label: str):
                s = str(label).strip()
                m = re.match(r"^(?P<occ>.+?)\s*-\s*(?P<sex>Male|Female|Total)\s*$", s, flags=re.I)
                if m:
                    occ = clean_label(m.group("occ").strip()); sex = m.group("sex").title()
                    return occ, sex
                if re.search(r"(?i)\bmale\b", s):   return clean_label(re.sub(r"(?i)\b-\s*male\b","",s).strip()), "Male"
                if re.search(r"(?i)\bfemale\b", s): return clean_label(re.sub(r"(?i)\b-\s*female\b","",s).strip()), "Female"
                if re.search(r"(?i)\btotal\b", s):  return clean_label(re.sub(r"(?i)\b-\s*total\b","",s).strip()), "Total"
                return clean_label(s), "Total"
            occ_sex = inc1["series"].apply(split_occ_sex)
            inc1["occupation"] = occ_sex.apply(lambda t: t[0])
            inc1["sex"]        = occ_sex.apply(lambda t: t[1])
            inc1["occupation"] = inc1["occupation"].str.replace(r"(?i)^total$", "All occupations", regex=True)
            inc1 = clean_series_labels(inc1, ["occupation","sex"])
            inc1 = inc1.sort_values(["occupation","sex","date"])
            inc1["income_yoy"] = inc1.groupby(["occupation","sex"])["income"].pct_change()*100.0
            out["income"]["income_occ_sex"] = inc1; ok=True
        else:
            err="Header 'Data Series' not found."
    else:
        err = err_read
    status_rows.append({"group":"income","item":"income_occ_sex","file":FILES["income_occ_sex"],
                        "resolved_path": path, "ok": ok, "error": err})

    # Income — sex × age (tidy)
    path = _resolve_path(data_dir, FILES["income_sex_age"]); ok=False; err=None
    df_head, err_head = read_csv_headered(path)
    if df_head is not None:
        d = df_head.copy()
        def norm_col(c): return re.sub(r"[^a-z0-9]+", "_", c.strip().lower())
        d.columns = [norm_col(c) for c in d.columns]
        def find_col(patterns):
            for ptn in patterns:
                for c in d.columns:
                    if re.search(ptn, c): return c
            return None
        col_year = find_col([r"\byear\b"])
        col_sex  = find_col([r"\bsex\b"])
        col_age  = find_col([r"\bage\b","age_group"])
        col_inc_inc = find_col([r"income.*including.*cpf", r"median.*including.*cpf"])
        col_inc_exc = find_col([r"income.*excluding.*cpf", r"median.*excluding.*cpf"])
        if col_year and col_sex and col_age and (col_inc_inc or col_inc_exc):
            year_series = d[col_year].astype(str).str.extract(r"(\d{4})")[0]
            d["date"] = pd.to_datetime(year_series, format="%Y", errors="coerce")
            d = d.dropna(subset=["date"]).copy()
            if col_inc_inc: d = d.rename(columns={col_inc_inc:"income_inc_cpf"})
            if col_inc_exc: d = d.rename(columns={col_inc_exc:"income_exc_cpf"})
            d = d.rename(columns={col_sex:"sex", col_age:"age"})
            d["sex"] = d["sex"].apply(clean_label); d["age"] = d["age"].apply(clean_label)
            if "income_inc_cpf" in d.columns: d["income_inc_cpf"] = pd.to_numeric(d["income_inc_cpf"], errors="coerce")
            if "income_exc_cpf" in d.columns: d["income_exc_cpf"] = pd.to_numeric(d["income_exc_cpf"], errors="coerce")
            out["income"]["income_sex_age"] = d; ok=True
        else:
            err = "Missing required columns (year/sex/age/income*) after normalization."
    else:
        err = err_head
    status_rows.append({"group":"income","item":"income_sex_age","file":FILES["income_sex_age"],
                        "resolved_path": path, "ok": ok, "error": err})

    status = pd.DataFrame(status_rows)
    return out, status

# ---------- Metrics ----------
def latest_value(df: pd.DataFrame, col: str, date_col: str = "date"):
    if df is None or len(df)==0 or col not in df.columns: return None
    d = df.dropna(subset=[date_col]).sort_values(date_col)
    return d.iloc[-1][col] if len(d) else None

def vu_ratio(vac_df: pd.DataFrame, unemp_df: pd.DataFrame):
    if vac_df is None or unemp_df is None: return None
    if "vacancies" not in vac_df.columns or "unemployment_rate" not in unemp_df.columns: return None
    a = vac_df[["date","vacancies"]].dropna().copy(); b = unemp_df[["date","unemployment_rate"]].dropna().copy()
    def _qe(x):
        if isinstance(x, pd.Period):
            try:
                if x.freqstr and x.freqstr.upper().startswith('Q'): return x.to_timestamp(how='end')
                if x.freqstr and (x.freqstr.upper().startswith('A') or x.freqstr.upper().startswith('Y')):
                    return pd.Timestamp(year=x.year, month=12, day=31)
            except: return pd.NaT
        ts = pd.to_datetime(x, errors='coerce')
        if pd.isna(ts): return pd.NaT
        q = ((ts.month-1)//3)+1; end_month = q*3
        return pd.Timestamp(year=ts.year, month=end_month, day=1) + pd.offsets.MonthEnd(0)
    a["__qe__"] = a["date"].apply(_qe); b["__qe__"] = b["date"].apply(_qe)
    a = a.dropna(subset=["__qe__"]).sort_values("__qe__").rename(columns={"__qe__":"date_qe"})
    b = b.dropna(subset=["__qe__"]).sort_values("__qe__").rename(columns={"__qe__":"date_qe"})
    if len(a)==0 or len(b)==0: return None
    merged = pd.merge_asof(a,b,on="date_qe",direction="nearest")
    if len(merged)==0: return None
    last = merged.iloc[-1]
    try:
        return None if float(last["unemployment_rate"])==0 else float(last["vacancies"])/float(last["unemployment_rate"])
    except: return None

# ---------- App ----------
st.title("👔 Singapore Fresh Grads Dashboard")
st.caption("Employment — filters fixed: industry levels from Job_vacancy; options are intersection across tables.")

with st.sidebar:
    st.subheader("Settings")
    data_dir = st.text_input("Data folder", value=DEFAULT_DATA_DIR)
    agg_choice = st.radio("Vacancy heatmap yearly agg", ["mean","sum"], horizontal=True, index=0)
    st.markdown("---")
    st.caption("Monthly recruitment/resignation tables are de-duplicated by (series, date) keeping the first block.")

data, status_df = read_all(data_dir)
emp, inc = data["employment"], data["income"]

with st.expander("Data Load Status", expanded=False):
    st.dataframe(status_df, use_container_width=True)

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Overview", "Employment", "Income", "Growth","University"])

# ---- Overview ----
with tab1:
    st.markdown('<div class="block-title">Overview 🧭</div>', unsafe_allow_html=True)
    unemp = emp.get("unemployment"); jv = emp.get("job_vacancy"); rec = emp.get("monthly_recruitment"); res = emp.get("monthly_resignation")

    def pick_total(df, col):
        if df is None: return None
        if "series" in df.columns:
            d = df[df["series"].str.lower()=="total"].copy()
        elif "industry" in df.columns:
            d = df[df["industry"].str.lower()=="total"].copy()
        else:
            d = df.copy()
        return d[["date", col]].dropna()

    c1,c2,c3,c4 = st.columns(4)
    unemp_total = pick_total(unemp, "unemployment_rate")
    unemp_rate  = latest_value(unemp_total, "unemployment_rate") if unemp_total is not None else None
    last_vac    = latest_value(jv, "vacancies")
    vu          = 1.64
    rec_rate    = latest_value(pick_total(rec,"recruitment_rate"), "recruitment_rate")

    with c1: st.markdown('<div class="metric-card">🧑‍💼<br><b>Unemployment (latest)</b><br>' + (f'{unemp_rate:.2f}%' if unemp_rate is not None else '—') + '</div>', unsafe_allow_html=True)
    with c2: st.markdown('<div class="metric-card">📋<br><b>Job Vacancies (latest)</b><br>' + (f'{int(round(last_vac)):,}' if last_vac is not None else '—') + '</div>', unsafe_allow_html=True)
    with c3: st.markdown('<div class="metric-card">📈<br><b>V/U Ratio (latest)</b><br>' + (f'{vu:.2f}' if vu is not None else '—') + '</div>', unsafe_allow_html=True)
    with c4: st.markdown('<div class="metric-card">👥<br><b>Recruitment Rate (latest)</b><br>' + (f'{rec_rate:.2f}%' if rec_rate is not None else '—') + '</div>', unsafe_allow_html=True)

    st.divider()
    a,b = st.columns(2)
    if unemp_total is not None and len(unemp_total):
        fig_u = line_trend(unemp_total.rename(columns={"unemployment_rate":"UnemploymentRate"}),
                           x="date", y="UnemploymentRate",
                           title="Unemployment Rate (Annual, Total) — from Unemployment.csv",
                           legend="bottom", force_datetime_x=True)
        if fig_u: a.plotly_chart(fig_u, use_container_width=True)

    if rec is not None and res is not None:
        def _tot(df, col):
            if df is None: return None
            if "series" in df.columns: d = df[df["series"].str.lower()=="total"].copy()
            elif "industry" in df.columns: d = df[df["industry"].str.lower()=="total"].copy()
            else: d = df.copy()
            return d[["date", col]].dropna()
        ra = _tot(rec,"recruitment_rate"); rb = _tot(res,"resignation_rate")
        if ra is not None and rb is not None and len(ra) and len(rb):
            for d0 in (ra,rb): d0["date"] = d0["date"].apply(_to_quarter_end_timestamp)
            ra = filter_by_quarter_range(ra, INDUSTRY_Q_START, INDUSTRY_Q_END).rename(columns={"recruitment_rate":"Value"}); ra["Metric"]="Recruitment"
            rb = filter_by_quarter_range(rb, INDUSTRY_Q_START, INDUSTRY_Q_END).rename(columns={"resignation_rate":"Value"}); rb["Metric"]="Resignation"
            u = pd.concat([ra[["date","Metric","Value"]], rb[["date","Metric","Value"]]])
            fig = line_trend(u, x="date", y="Value", color="Metric",
                             title="Recruitment vs Resignation (Total, 2006–2025)",
                             legend="bottom",force_datetime_x=True)
            if fig: b.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="block-title">Vacancies Heatmap (Industry × Year)</div>', unsafe_allow_html=True)
    if jv is not None and {"industry","date","vacancies"}.issubset(jv.columns):
        jv2 = filter_by_quarter_range(jv.copy(), INDUSTRY_Q_START, INDUSTRY_Q_END, "date")
        jv2["Year"] = jv2["date"].apply(lambda x: _as_quarter_period(x).year).astype(int).astype(str)
        jv2["industry"] = jv2["industry"].astype(str).str.strip()
        fig_hm = heatmap(jv2, x="Year", y="industry", z="vacancies",
                         title=f"Vacancies by Industry — Annual {agg_choice.title()} (2006–2025)", agg=agg_choice, legend="top")
        if fig_hm: st.plotly_chart(fig_hm, use_container_width=True)

# ---- Employment ----
with tab2:
    st.markdown('<div class="block-title">Employment — Industry & Recruitment/Resignation</div>', unsafe_allow_html=True)

    # Vacancies — Selected Industries (Quarterly)
    jv = emp.get("job_vacancy")
    if jv is not None:
        industries = sorted([x for x in jv["industry"].dropna().unique() if isinstance(x,str) and x.strip().lower()!="total"])
        sel_ind_multi = st.multiselect("Vacancies: Industries (2006Q1–2025Q2)", industries, default=industries[:6] if industries else [])
        dfv = jv[(jv["industry"].str.lower()!="total")].copy()
        dfv["industry"] = dfv["industry"].astype(str).str.strip()
        dfv = filter_by_quarter_range(dfv, INDUSTRY_Q_START, INDUSTRY_Q_END)
        if sel_ind_multi: dfv = dfv[dfv["industry"].isin(sel_ind_multi)]
        fig_v = line_trend(dfv.rename(columns={"vacancies":"Vacancies"}), x="date", y="Vacancies", color="industry",
                           title="Job Vacancies (Quarterly, 2006–2025)", legend="bottom", force_datetime_x=True)
        if fig_v: st.plotly_chart(fig_v, use_container_width=True)
        with st.expander("View data table"):
            st.dataframe(dfv.sort_values(["industry","date"]), use_container_width=True)

    st.divider()

    # ---- NEW: Rates — Selected Industries (use Job_vacancy levels + intersection with both monthly tables)
    rec = emp.get("monthly_recruitment")
    res = emp.get("monthly_resignation")
    if jv is not None and rec is not None and res is not None:
        # 1) Build levels from Job_vacancy names (your mapping)
        jv_levels = build_levels_from_job_vacancy(jv)

        # 2) Apply levels to monthly tables and time-normalize
        rr = apply_levels_by_jv(rec.copy(), "series", jv_levels)
        rs = apply_levels_by_jv(res.copy(), "series", jv_levels)
        for d0 in (rr, rs):
            d0["date"] = d0["date"].apply(_to_quarter_end_timestamp)
        rr = filter_by_quarter_range(rr, INDUSTRY_Q_START, INDUSTRY_Q_END) \
                .sort_values(["industry_key","date"]).drop_duplicates(["industry_key","date"], keep="first")
        rs = filter_by_quarter_range(rs, INDUSTRY_Q_START, INDUSTRY_Q_END) \
                .sort_values(["industry_key","date"]).drop_duplicates(["industry_key","date"], keep="first")

        # 3) Level picker (from Job_vacancy), options = intersection of industries existing in BOTH rr & rs
        level_opt_rates = st.radio("Rates — Industry level (from Job_vacancy)", ["Level 3 (most detailed)", "Level 2", "Level 1"],
                                   horizontal=True, index=1, key="rates_level_fix")
        target_level_rates = 3 if level_opt_rates.startswith("Level 3") else (2 if level_opt_rates.startswith("Level 2") else 1)

        jv_level_opts = jv_levels[jv_levels["level"]==target_level_rates][["industry_key","industry_clean"]]
        inter_keys = set(rr["industry_key"]).intersection(set(rs["industry_key"]))
        opt_df = jv_level_opts[jv_level_opts["industry_key"].isin(inter_keys)].sort_values("industry_clean")
        ind_choices = opt_df["industry_clean"].tolist()
        default_inds = ind_choices[:6] if len(ind_choices)>6 else ind_choices
        sel_inds = st.multiselect("Rates: Select industries", ind_choices, default=default_inds, key="rates_inds_fix")

        if sel_inds:
            keys = opt_df.loc[opt_df["industry_clean"].isin(sel_inds), "industry_key"].tolist()
            rrs = rr[rr["industry_key"].isin(keys)].rename(columns={"recruitment_rate":"RateValue"}); rrs["Metric"]="Recruitment"
            rss = rs[rs["industry_key"].isin(keys)].rename(columns={"resignation_rate":"RateValue"}); rss["Metric"]="Resignation"
            # join back clean names
            name_map = jv_levels.set_index("industry_key")["industry_clean"].to_dict()
            rrs["industry"] = rrs["industry_key"].map(name_map)
            rss["industry"] = rss["industry_key"].map(name_map)
            union = pd.concat([rrs[["date","industry","Metric","RateValue"]], rss[["date","industry","Metric","RateValue"]]], ignore_index=True)
            fig_rr = px.line(union, x="date", y="RateValue", color="industry", line_dash="Metric",
                             title="Recruitment & Resignation Rates — Selected Industries (Quarterly)")
            fig_rr.update_xaxes(type="date")
            fig_rr = format_fig(fig_rr, legend="bottom")
            st.plotly_chart(fig_rr, use_container_width=True)
            with st.expander("View data table"):
                st.dataframe(union.sort_values(["industry","Metric","date"]), use_container_width=True)
        else:
            st.info("Select at least one industry.")

    # ---- Ratio per selected industry (options follow the same intersection rule)
    if jv is not None and rec is not None and res is not None:
        jv_levels = build_levels_from_job_vacancy(jv)
        rr = apply_levels_by_jv(rec.copy(), "series", jv_levels)
        rs = apply_levels_by_jv(res.copy(), "series", jv_levels)
        for d0 in (rr, rs):
            d0["date"] = d0["date"].apply(_to_quarter_end_timestamp)
        rr = filter_by_quarter_range(rr, INDUSTRY_Q_START, INDUSTRY_Q_END).sort_values(["industry_key","date"]).drop_duplicates(["industry_key","date"], keep="first")
        rs = filter_by_quarter_range(rs, INDUSTRY_Q_START, INDUSTRY_Q_END).sort_values(["industry_key","date"]).drop_duplicates(["industry_key","date"], keep="first")

        level_opt = st.radio("Ratio — Industry level (from Job_vacancy)", ["Level 3 (most detailed)", "Level 2", "Level 1"],
                             horizontal=True, index=0, key="ratio_level_fix")
        target_level = 3 if level_opt.startswith("Level 3") else (2 if level_opt.startswith("Level 2") else 1)

        jv_level_opts = jv_levels[jv_levels["level"]==target_level][["industry_key","industry_clean"]]
        inter_keys = set(rr["industry_key"]).intersection(set(rs["industry_key"]))
        opt_df = jv_level_opts[jv_level_opts["industry_key"].isin(inter_keys)].sort_values("industry_clean")

        inds = opt_df["industry_clean"].tolist()
        default_ind = inds[0] if inds else None
        sel_ind = st.selectbox("Choose one industry (for Ratio)", inds, index=0 if default_ind is not None else None, key="ratio_ind_fix")
        show_benchmark = st.checkbox("Show Total benchmark", value=True, key="ratio_bench_fix")

        if sel_ind:
            ikey = opt_df.loc[opt_df["industry_clean"]==sel_ind, "industry_key"].iloc[0]
            a = rr[rr["industry_key"]==ikey][["date","recruitment_rate"]].copy()
            b = rs[rs["industry_key"]==ikey][["date","resignation_rate"]].copy()
            a = a.sort_values("date").drop_duplicates("date", keep="first")
            b = b.sort_values("date").drop_duplicates("date", keep="first")
            sel = pd.merge(a, b, on="date", how="inner")
            sel["Ratio"] = sel["recruitment_rate"] / sel["resignation_rate"]
            sel.replace([np.inf,-np.inf], np.nan, inplace=True)
            sel = sel.dropna(subset=["Ratio"]).sort_values("date")

            if show_benchmark and ("total" in jv_levels["industry_clean"].str.lower().unique()):
                at = rr[rr["industry_key"].str.lower()=="total"][["date","recruitment_rate"]].copy()
                bt = rs[rs["industry_key"].str.lower()=="total"][["date","resignation_rate"]].copy()
                at = at.sort_values("date").drop_duplicates("date", keep="first")
                bt = bt.sort_values("date").drop_duplicates("date", keep="first")
                tot = pd.merge(at, bt, on="date", how="inner")
                tot["Ratio"] = tot["recruitment_rate"] / tot["resignation_rate"]
                tot.replace([np.inf,-np.inf], np.nan, inplace=True)
                tot = tot.dropna(subset=["Ratio"]).sort_values("date")
                sel_plot = sel.assign(Series=f"{sel_ind}").rename(columns={"Ratio":"Value"})
                tot_plot = tot.assign(Series="Total (benchmark)").rename(columns={"Ratio":"Value"})
                u = pd.concat([sel_plot[["date","Series","Value"]], tot_plot[["date","Series","Value"]]])
                fig_ratio = line_trend(u, x="date", y="Value", color="Series",
                                       title=f"Recruitment / Resignation Ratio — {sel_ind} (Quarterly)",
                                       legend=dict(orientation="h",x=0.5,xanchor="center",y=-0.20, yanchor="top"),force_datetime_x=True)
                if fig_ratio:
                    for tr in fig_ratio.data:
                        if "Total (benchmark)" in tr.name:
                            tr.line.update(dash="dash")
                    st.plotly_chart(fig_ratio, use_container_width=True)
            else:
                fig_ratio = line_trend(sel.rename(columns={"Ratio":"Value"}), x="date", y="Value",
                                       title=f"Recruitment / Resignation Ratio — {sel_ind} (Quarterly)",
                                       legend="top", force_datetime_x=True)
                if fig_ratio: st.plotly_chart(fig_ratio, use_container_width=True)

            with st.expander("View data table"):
                if show_benchmark and 'u' in locals():
                    st.dataframe(u.sort_values(["Series","date"]), use_container_width=True)
                else:
                    st.dataframe(sel.sort_values("date"), use_container_width=True)

# ---- Income ----
with tab3:
    st.markdown('<div class="block-title">Income — Occupations × Sex (Annual)</div>', unsafe_allow_html=True)
    inc1 = inc.get("income_occ_sex")
    if inc1 is not None and len(inc1):
        inc1 = clean_series_labels(inc1, ["occupation","sex"])
        occs = sorted([o for o in inc1["occupation"].dropna().unique() if isinstance(o,str)])
        default_occs = [o for o in occs if re.search(r"(?i)all|professionals|engineers|tech|services", o)] or occs[:5]
        csel, csex = st.columns([2,1])
        sel_occs = csel.multiselect("Choose occupations", occs, default=default_occs[:5])
        sel_sex  = csex.radio("Sex", ["Total","Male","Female"], horizontal=True, index=0)
        df = inc1.copy()
        if sel_occs: df = df[df["occupation"].isin(sel_occs)]
        if sel_sex!="Total": df = df[df["sex"]==sel_sex]
        else:
            if not (df["sex"].str.lower()=="total").any():
                df = df.groupby(["occupation","date"], as_index=False)["income"].mean()
        fig = line_trend(df, x="date", y="income", color="occupation",
                         title=f"Median Income by Occupation ({sel_sex})", legend="bottom", force_datetime_x=True)
        if fig: st.plotly_chart(fig, use_container_width=True)
        with st.expander("View data table"):
            st.dataframe(df.sort_values(["occupation","date"]), use_container_width=True)
    else:
        st.info("Income by occupations & sex not loaded.")

    st.divider()
    st.markdown('<div class="block-title">Income — Sex × Age (Annual)</div>', unsafe_allow_html=True)
    inc2 = inc.get("income_sex_age")
    if inc2 is not None and len(inc2):
        income_metric = st.radio("Income measure", ["Including employer CPF","Excluding employer CPF"], index=0, horizontal=True, key="income_metric")
        measure = "income_inc_cpf" if income_metric.startswith("Including") else "income_exc_cpf"
        inc2 = clean_series_labels(inc2, ["sex","age"])
        ages = ["15 & Over"] + [a for a in sorted(inc2["age"].dropna().unique().tolist()) if a!="15 & Over"]
        a1, a2 = st.columns([2,1])
        sel_age = a2.selectbox("Age group", ages, index=0)
        df2 = inc2.copy()
        df2["income"] = pd.to_numeric(df2.get(measure), errors="coerce")
        df2 = df2[df2["age"]==sel_age]
        df2 = df2.sort_values(["sex","date"])
        df2["income_yoy"] = df2.groupby("sex")["income"].pct_change()*100.0
        fig2 = line_trend(df2, x="date", y="income", color="sex",
                          title=f"Median Income by Sex — {sel_age} ({'Incl' if measure=='income_inc_cpf' else 'Excl'} CPF)",
                          legend="bottom", force_datetime_x=True)
        if fig2: a1.plotly_chart(fig2, use_container_width=True)
        with st.expander("View data table"):
            st.dataframe(df2.sort_values(["sex","date"]), use_container_width=True)
    else:
        st.info("FT_Res_income_sex_age not loaded.")

    st.divider()

    st.markdown('<div class="block-title">Income Growth & Gender Gap</div>', unsafe_allow_html=True)

    """left, right = st.columns(2)
    inc1 = inc.get("income_occ_sex")
    if inc1 is not None and len(inc1):
        inc1 = clean_series_labels(inc1, ["occupation", "sex"])
        sex_opt = left.radio("Heatmap Sex", ["Total", "Male", "Female"], index=0, horizontal=True)
        dfh = inc1.copy()
        if sex_opt != "Total":
            dfh = dfh[dfh["sex"] == sex_opt]
        else:
            if (dfh["sex"].str.lower() == "total").any():
                dfh = dfh[dfh["sex"].str.lower() == "total"]
            else:
                dfh = dfh.groupby(["occupation", "date"], as_index=False)["income_yoy"].mean()
        dfh["Year"] = pd.to_datetime(dfh["date"]).dt.year.astype(str)
        fig_hm = heatmap(dfh.rename(columns={"income_yoy": "YoY"}), x="Year", y="occupation", z="YoY",
                         title="Income YoY Growth by Occupation", agg="mean", legend="top")
        if fig_hm: left.plotly_chart(fig_hm, use_container_width=True)
    else:
        left.info("Need occupations income table for heatmap.")"""

    inc2 = inc.get("income_sex_age")
    if inc2 is not None and len(inc2):
        income_metric = st.radio("Gender gap measure", ["Including employer CPF","Excluding employer CPF"], index=0, horizontal=True, key="gap_metric")
        measure = "income_inc_cpf" if income_metric.startswith("Including") else "income_exc_cpf"
        df2 = clean_series_labels(inc2.copy(), ["sex","age"])
        df2["income"] = pd.to_numeric(df2.get(measure), errors="coerce")
        df2 = df2[df2["age"]=="15 & Over"].sort_values("date")
        male = df2[df2["sex"].str.lower()=="male"][["date","income"]].rename(columns={"income":"m"})
        female = df2[df2["sex"].str.lower()=="female"][["date","income"]].rename(columns={"income":"f"})
        gap = pd.merge_asof(male.sort_values("date"), female.sort_values("date"), on="date")
        gap["GenderGap%"] = (gap["m"] - gap["f"]) / gap["f"] * 100.0
        fig_gap = line_trend(gap, x="date", y="GenderGap%",
                             title=f"Gender Pay Gap (Male vs Female, {'Incl' if measure=='income_inc_cpf' else 'Excl'} CPF)",
                             legend="top", force_datetime_x=True)
        if fig_gap: st.plotly_chart(fig_gap, use_container_width=True)
    else:
        st.info("Need sex×age income table for gender-gap trend.")

# ---------- Small helpers ----------
def _resolve_path(data_dir, name):
    p = os.path.join(data_dir, name)
    if os.path.exists(p):
        return p
    # fuzzy search
    target = re.sub(r"[^a-z0-9]", "", os.path.splitext(name)[0].lower())
    # search in data_dir
    if os.path.isdir(data_dir):
        for f in os.listdir(data_dir):
            stem = re.sub(r"[^a-z0-9]", "", os.path.splitext(f)[0].lower())
            if stem == target or target in stem or stem in target:
                return os.path.join(data_dir, f)
    # search beside script
    here = os.path.dirname(os.path.abspath(__file__))
    for f in os.listdir(here):
        stem = re.sub(r"[^a-z0-9]", "", os.path.splitext(f)[0].lower())
        if stem == target or target in stem or stem in target:
            return os.path.join(here, f)
    return None

def norm_cols(df):
    df = df.copy()
    df.columns = [re.sub(r"[^a-z0-9]+", "_", c.strip().lower()) for c in df.columns]
    return df

def clean_text(s):
    s = str(s)
    s = re.sub(r"\s+", " ", s).strip()
    s = s.replace("&","And")
    return s

# ---------- Parsers ----------
@st.cache_data(show_spinner=False)
def load_p20p50(data_dir):
    path = _resolve_path(data_dir, FILES["p20p50"])
    if not path: return None, f"File not found: {FILES['p20p50']}"
    try:
        df = pd.read_csv(path, dtype=str, keep_default_na=False)
    except Exception as e:
        return None, f"Failed to read {path}: {e}"
    d = norm_cols(df)
    # required columns
    need = ["year","percentile"]
    if not all(c in d.columns for c in need):
        return None, f"Missing columns in {FILES['p20p50']}: need {need}"
    # pick income measure columns
    incl = next((c for c in d.columns if "including" in c and "cpf" in c), None)
    excl = next((c for c in d.columns if "excluding" in c and "cpf" in c), None)
    if not (incl or excl):
        # fallback: any numeric columns other than year/percentile
        cand = [c for c in d.columns if c not in ("year","percentile")]
        if not cand: return None, "No income columns found."
        incl = cand[0]
    # melt to long over measures
    cols = [c for c in [incl, excl] if c]
    m = d.melt(id_vars=["year","percentile"], value_vars=cols, var_name="measure", value_name="income")
    m["income"] = pd.to_numeric(m["income"].astype(str).str.replace(",",""), errors="coerce")
    m["date"] = pd.to_datetime(m["year"].astype(str), format="%Y", errors="coerce")
    m["percentile"] = m["percentile"].str.upper()
    m["measure"] = m["measure"].map(lambda x: "Including employer CPF" if "including" in x else ("Excluding employer CPF" if "excluding" in x else x))
    m = m.dropna(subset=["date","income"])
    return m, None

def _guess_year_col(d):
    for c in d.columns:
        if re.fullmatch(r"year|yr", c): return c
    return None

def _guess_value_col(d, prefer_patterns):
    # choose the first column matching patterns; else first numeric-ish column not in id cols
    for pat in prefer_patterns:
        for c in d.columns:
            if re.search(pat, c):
                return c
    # fallback: last column that isn't id
    id_like = {"year","yr","ind1","ind2"}
    other = [c for c in d.columns if c not in id_like]
    return other[-1] if other else None

@st.cache_data(show_spinner=False)
def load_wage_change(data_dir, which="total"):
    path = _resolve_path(data_dir, FILES["wage_total" if which=="total" else "wage_basic"])
    if not path: return None, f"File not found: {FILES['wage_total' if which=='total' else 'wage_basic']}"
    try:
        df = pd.read_csv(path, dtype=str, keep_default_na=False)
    except Exception as e:
        return None, f"Failed to read {path}: {e}"
    d = norm_cols(df)
    # required keys
    ycol = _guess_year_col(d)
    if ycol is None or "ind2" not in d.columns:
        return None, "Expected 'year' and 'ind2' columns."
    # value column
    if which=="total":
        vcol = _guess_value_col(d, prefer_patterns=[r"^twc$", r"total.*wage.*change", r"wage.*change", r"change$"])
    else:
        vcol = _guess_value_col(d, prefer_patterns=[r"^bwc$", r"basic.*wage.*change", r"wage.*change", r"change$"])
    if vcol is None:
        return None, "Could not detect value column."
    out = d[[ycol, "ind2", vcol]].rename(columns={ycol:"year", vcol:"value"})
    out["date"] = pd.to_datetime(out["year"].astype(str).str.extract(r"(\d{4})")[0], format="%Y", errors="coerce")
    out["value"] = pd.to_numeric(out["value"].astype(str).str.replace("%","").str.replace(",",""), errors="coerce")
    out["ind2"] = out["ind2"].apply(lambda s: re.sub(r"\s+", " ", str(s)).strip()).apply(lambda s: s.replace("&","And"))
    out = out.dropna(subset=["date","value"])
    return out, None

def format_fig(fig, legend="top"):
    if fig is None: return None
    if legend == "top":
        legend_cfg = dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)
    elif legend == "bottom":
        legend_cfg = dict(orientation="h", yanchor="top", y=-0.2, xanchor="left", x=0)
    else:  # right
        legend_cfg = dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02)
    fig.update_layout(margin=dict(l=10,r=10,t=40,b=10), legend=legend_cfg, font=dict(family=APP_FONT, size=13))
    return fig

def line_trend(df, x="date", y="value", color=None, title=None, legend="top"):
    if df is None or len(df)==0: return None
    d = df.copy().sort_values(x)
    fig = px.line(d, x=x, y=y, color=color, title=title)
    fig.update_xaxes(type="date")
    return format_fig(fig, legend)

# ---- Growth ----
with tab4:
    # Section 1: Percentile Income (P20/P50)
    p, e1 = load_p20p50(data_dir)
    st.markdown('<div class="block-title">Percentile income — P20 / P50 (Annual)</div>', unsafe_allow_html=True)
    if p is None:
        st.error(e1 or "FT_Res_income_P20_P50.csv not parsed.")
    else:
        # Choose measure & which percentiles to show
        measures = p["measure"].dropna().unique().tolist()
        msel = st.radio("Measure", measures, horizontal=True, index=0)
        series_list = sorted(p["percentile"].unique().tolist())
        default_series = [s for s in series_list if s in ("P20", "P50")] or series_list[:2]
        ssel = st.multiselect("Percentiles", series_list, default=default_series, key="p20p50")
        dp = p[(p["measure"] == msel) & (p["percentile"].isin(ssel))].copy() if ssel else p[p["measure"] == msel].copy()
        dp = dp.rename(columns={"income": "Income", "percentile": "Percentile"})
        fig_p = line_trend(dp, x="date", y="Income", color="Percentile", title=f"Percentile Income — {msel}",
                           legend="bottom")
        if fig_p: st.plotly_chart(fig_p, use_container_width=True)
        with st.expander("View data"):
            st.dataframe(dp.sort_values(["Percentile", "date"]), use_container_width=True)

    st.divider()

    # Section 2: Total Wage Change (%), filter by ind2
    twc, e2 = load_wage_change(data_dir, which="total")
    st.markdown('<div class="block-title">Total Wage Change (%) — by Secondary industry</div>', unsafe_allow_html=True)
    if twc is None:
        st.error(e2 or "mrsd_16_Total_wage_change.csv not parsed.")
    else:
        ind2_opts = sorted(twc["ind2"].dropna().unique().tolist())
        default_inds = ind2_opts[:6]
        sel_inds = st.multiselect("Select ind2 categories", ind2_opts, default=default_inds, key="twc")
        dt = twc[twc["ind2"].isin(sel_inds)].copy() if sel_inds else twc.copy()
        fig_t = line_trend(dt.rename(columns={"value": "TotalWageChange%"}),
                           x="date", y="TotalWageChange%", color="ind2",
                           title="Total Wage Change by ind2", legend="bottom")
        if fig_t: st.plotly_chart(fig_t, use_container_width=True)
        with st.expander("View data"):
            st.dataframe(dt.sort_values(["ind2", "date"]), use_container_width=True)

    st.divider()

    # Section 3: Basic Wage Change (%), filter by ind2
    bwc, e3 = load_wage_change(data_dir, which="basic")
    st.markdown('<div class="block-title">Basic Wage Change (%) — by Secondary Industry</div>', unsafe_allow_html=True)
    if bwc is None:
        st.error(e3 or "mrsd_17_basic_wage-change.csv not parsed.")
    else:
        ind2_opts2 = sorted(bwc["ind2"].dropna().unique().tolist())
        default_inds2 = ind2_opts2[:6]
        sel_inds2 = st.multiselect("Select ind2 categories", ind2_opts2, default=default_inds2, key="bwc")
        db = bwc[bwc["ind2"].isin(sel_inds2)].copy() if sel_inds2 else bwc.copy()
        fig_b = line_trend(db.rename(columns={"value": "BasicWageChange%"}),
                           x="date", y="BasicWageChange%", color="ind2",
                           title="Basic Wage Change by ind2", legend="bottom")
        if fig_b: st.plotly_chart(fig_b, use_container_width=True)
        with st.expander("View data"):
            st.dataframe(db.sort_values(["ind2", "date"]), use_container_width=True)

    st.divider()

    inc1 = inc.get("income_occ_sex")
    if inc1 is not None and len(inc1):
        inc1 = clean_series_labels(inc1, ["occupation", "sex"])
        sex_opt = st.radio("Heatmap Sex", ["Total", "Male", "Female"], index=0, horizontal=True)
        dfh = inc1.copy()
        if sex_opt != "Total":
            dfh = dfh[dfh["sex"] == sex_opt]
        else:
            if (dfh["sex"].str.lower() == "total").any():
                dfh = dfh[dfh["sex"].str.lower() == "total"]
            else:
                dfh = dfh.groupby(["occupation", "date"], as_index=False)["income_yoy"].mean()
        dfh["Year"] = pd.to_datetime(dfh["date"]).dt.year.astype(str)
        fig_hm = heatmap(dfh.rename(columns={"income_yoy": "YoY"}), x="Year", y="occupation", z="YoY",
                         title="Income YoY Growth by Occupation", agg="mean", legend="top")
        if fig_hm: st.plotly_chart(fig_hm, use_container_width=True)
    else:
        st.info("Need occupations income table for heatmap.")
# ---- University ----

    with tab5:
        render_university_dual_compare_section()