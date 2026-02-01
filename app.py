import re
import io
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# =========================================================
# Page config
# =========================================================
st.set_page_config(page_title="Dawaiyat Project Tracker Dashboard", layout="wide")

# =========================================================
# Styling / Theme
# =========================================================
DARK_CSS = """
<style>
/* App background + typography */
.stApp { background: #0b1220; color: #eef2ff; }
section[data-testid="stSidebar"] { background: #0a1020; }

/* Card look */
.kpi-card {
  background: rgba(255,255,255,0.04);
  border: 1px solid rgba(255,255,255,0.08);
  border-radius: 16px;
  padding: 14px 16px;
}
.kpi-title { font-size: 11px; letter-spacing: .12em; text-transform: uppercase; color: rgba(255,255,255,.65); margin: 0; }
.kpi-value { font-size: 30px; font-weight: 700; margin: 4px 0 0 0; color: #ffffff; }

/* Section title */
.section-title { font-size: 22px; font-weight: 800; margin-top: 4px; margin-bottom: 0; }
.section-sub { font-size: 12px; color: rgba(255,255,255,.6); margin-top: 0; }

/* Buttons */
.stButton>button, .stDownloadButton>button {
  border-radius: 12px !important;
}

/* Tables */
[data-testid="stDataFrame"] { border-radius: 14px; overflow: hidden; }
</style>
"""

LIGHT_CSS = """
<style>
.stApp { background: #ffffff; color: #0f172a; }
section[data-testid="stSidebar"] { background: #f8fafc; }

.kpi-card {
  background: #ffffff;
  border: 1px solid rgba(2,6,23,0.08);
  border-radius: 16px;
  padding: 14px 16px;
  box-shadow: 0 8px 24px rgba(2,6,23,0.05);
}
.kpi-title { font-size: 11px; letter-spacing: .12em; text-transform: uppercase; color: rgba(2,6,23,.55); margin: 0; }
.kpi-value { font-size: 30px; font-weight: 800; margin: 4px 0 0 0; color: #0f172a; }

.section-title { font-size: 22px; font-weight: 900; margin-top: 4px; margin-bottom: 0; }
.section-sub { font-size: 12px; color: rgba(2,6,23,.55); margin-top: 0; }

.stButton>button, .stDownloadButton>button { border-radius: 12px !important; }
[data-testid="stDataFrame"] { border-radius: 14px; overflow: hidden; }
</style>
"""

def apply_theme(dark: bool):
    st.markdown(DARK_CSS if dark else LIGHT_CSS, unsafe_allow_html=True)

# Plotly template (keeps charts consistent with theme)
def plotly_template(dark: bool) -> str:
    return "plotly_dark" if dark else "plotly_white"

# =========================================================
# Helpers
# =========================================================
def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(s).lower()).strip()

def find_col(df: pd.DataFrame, candidates, required=False):
    """Find the first column whose normalized name contains any candidate substring."""
    cols = list(df.columns)
    norm_map = {c: _norm(c) for c in cols}
    cand_norm = [_norm(x) for x in candidates]
    for c in cols:
        for kw in cand_norm:
            if kw and kw in norm_map[c]:
                return c
    if required:
        raise KeyError(f"Missing required column. Tried: {candidates}")
    return None

def to_percent(v):
    """Convert '0.81' or '81%' or '81' into 81.0 (percent)."""
    if pd.isna(v):
        return np.nan
    if isinstance(v, str):
        v = v.replace("%", "").strip()
    try:
        f = float(v)
        return f * 100.0 if f <= 1.0 else f
    except Exception:
        return np.nan

def safe_dt(s):
    return pd.to_datetime(s, errors="coerce")

def ensure_str(s):
    return s.astype(str).str.strip()

def make_week_key(dt_series: pd.Series) -> pd.Series:
    d = safe_dt(dt_series)
    if d.isna().all():
        return pd.Series([np.nan]*len(dt_series), index=dt_series.index)
    iso = d.dt.isocalendar()
    return (iso["year"].astype(str) + "-W" + iso["week"].astype(str).str.zfill(2))

# =========================================================
# Loading: supports (A) single workbook with 2 sheets OR (B) two separate files
# =========================================================
@st.cache_data(show_spinner=False)
def read_any(file, sheet_name=None) -> pd.DataFrame:
    name = getattr(file, "name", "").lower()
    if name.endswith(".csv"):
        return pd.read_csv(file)
    if name.endswith(".xlsx") or name.endswith(".xls"):
        # If sheet_name omitted, read first sheet
        return pd.read_excel(file, sheet_name=sheet_name)
    raise ValueError("Unsupported file type. Upload .xlsx or .csv")

@st.cache_data(show_spinner=False)
def load_from_workbook(wb) -> tuple[pd.DataFrame, pd.DataFrame]:
    xls = pd.ExcelFile(wb)
    sheets = { _norm(s): s for s in xls.sheet_names }
    # Robust sheet matching
    tool_name = None
    met_name = None
    for k, original in sheets.items():
        if "dawaiyat service tool" in k:
            tool_name = original
        if "met actual progress" in k or ("met" in k and "progress" in k):
            met_name = original
    if not tool_name or not met_name:
        raise ValueError("Workbook must contain sheets named 'Dawaiyat Service Tool' and 'MET Actual progress'.")
    tool = pd.read_excel(xls, tool_name)
    met  = pd.read_excel(xls, met_name)
    return tool, met

def build_dashboard_guide():
    st.markdown(
        """
### Dashboard Guide (for your team)

This dashboard compares **Contractor Commitment (Dawaiyat Service Tool)** vs **Site Reality (MET Actual progress)**.

#### 1) KPI Summary (Top 5 Cards)
- **Total Link Codes:** Total unique Link Codes in Dawaiyat Service Tool.
- **Dawaiyat Avg Target:** Average commitment progress (**Percentage of Completion**) from the tool.
- **MET Avg Actual:** Average site-reported progress from MET (**Civil Completion %** or **Fiber Completion %**, depending on Subclass).
- **Missing MET Reports:** Link Codes that exist in the Tool but have **no** records in MET (your blind spots).
- **Critical Lags (>15%):** Items where MET progress is **15%+ behind** the Tool commitment.

#### 2) Strategic 3×3 Chart Grid
- **Regional Coverage (Donut):** Work concentration across regions.
- **Dawaiyat vs MET Trend (Line):** Weekly average trend of Tool vs MET.
- **Operational Status (Donut):** Completed / In Progress / Missing Site Report.
- **Field Supervisor Performance (Bar):** Units managed vs average progress (accountability view).
- **Discipline Distribution (Donut):** Civil vs Fiber mix.
- **System Variance Gap (Area):** Average (MET − Tool). Large negative area = delivery/reporting gap.
- **District Loading (Bar):** Top districts by volume.
- **Execution Lifecycle (Pie):** Stage distribution.
- **Projected vs Realized (Step):** Commitment vs actual trend.

#### 3) Actionable Alert: Missing MET Reports
A list of Link Codes that **must be updated** by site/project teams.

#### 4) Audit Comparison Table
Detailed drill-down.  
- **Variance:** MET% − Tool% (green = ahead, red = behind).  
- **No MET Record:** flagged as **MISSING MET**.
""",
        unsafe_allow_html=True,
    )

# =========================================================
# Header
# =========================================================
with st.sidebar:
    st.header("⚙️ Settings")
    dark_mode = st.toggle("Dark Mode", value=False)  # default light like your screenshot
apply_theme(dark_mode)

# Top header row with "?" guide
top_left, top_right = st.columns([0.8, 0.2], vertical_alignment="center")
with top_left:
    st.markdown("## 📊 Dawaiyat Project Tracker Dashboard")
    st.caption("Contractor Commitment vs Site Actual | Executive Decision View")
with top_right:
    with st.popover("❓ Dashboard Guide", use_container_width=True):
        build_dashboard_guide()
    if st.button("🔄 Update Source Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

st.divider()

# =========================================================
# Upload area
# =========================================================
st.subheader("Upload Source Data")
upload_mode = st.radio(
    "Choose upload mode",
    ["Single Excel (contains both sheets)", "Two files (Tool + MET)"],
    horizontal=True,
)

tool = met = None

if upload_mode == "Single Excel (contains both sheets)":
    wb = st.file_uploader("Upload Excel (.xlsx) with both sheets", type=["xlsx"])
    if wb:
        tool, met = load_from_workbook(wb)
else:
    c1, c2 = st.columns(2)
    with c1:
        tool_file = st.file_uploader("Upload Dawaiyat Service Tool (.xlsx/.csv)", type=["xlsx", "csv"], key="tool_file")
    with c2:
        met_file = st.file_uploader("Upload MET Actual progress (.xlsx/.csv)", type=["xlsx", "csv"], key="met_file")
    if tool_file and met_file:
        # For XLSX here, we read the first sheet (or user can provide a one-sheet xlsx)
        tool = read_any(tool_file)
        met = read_any(met_file)

if tool is None or met is None:
    st.info("⬆️ Upload your source data to start.")
    st.stop()

# =========================================================
# Column mapping (robust to small header changes)
# =========================================================
# Shared identity columns
tool_link = find_col(tool, ["link code"], required=True)
tool_wo   = find_col(tool, ["work order", "wo"], required=False)

met_link  = find_col(met, ["link code"], required=True)
met_wo    = find_col(met, ["work order", "wo"], required=False)

# Subclass / discipline (Civil vs Fiber)
tool_sub  = find_col(tool, ["subclass", "discipline", "work subclass"], required=False)
met_sub   = find_col(met, ["subclass", "discipline", "work subclass"], required=False)

# Progress columns
tool_pct_col = find_col(tool, ["percentage of completion", "overall progress", "progress"], required=True)

met_civil_pct = find_col(met, ["civil completion"], required=False)
met_fiber_pct = find_col(met, ["fiber completion"], required=False)
met_any_pct   = find_col(met, ["overall progress", "actual progress", "completion"], required=False)

# Dates
tool_target_date = find_col(tool, ["targeted completion", "forecasted completion", "target completion", "targeted completion date"], required=False)
met_target_date  = find_col(met,  ["targeted completion", "forecasted completion", "target completion", "mst target"], required=False)

# Filter columns (optional, will show only if found)
f_region     = find_col(tool, ["region"], required=False)
f_project    = find_col(tool, ["project"], required=False)
f_year       = find_col(tool, ["year"], required=False)
f_stage      = find_col(tool, ["stage"], required=False)
f_district   = find_col(tool, ["district"], required=False)
f_supervisor = find_col(tool, ["supervisor"], required=False)
f_category   = find_col(tool, ["category"], required=False)

# =========================================================
# Prepare Tool table
# =========================================================
tool_df = tool.copy()
tool_df["_Link Code"] = ensure_str(tool_df[tool_link])

if tool_wo:
    tool_df["_WO"] = ensure_str(tool_df[tool_wo])
else:
    tool_df["_WO"] = tool_df["_Link Code"]  # fallback

tool_df["_Subclass"] = ensure_str(tool_df[tool_sub]) if tool_sub else ""

tool_df["_Tool_Target_%"] = tool_df[tool_pct_col].apply(to_percent)
tool_df["_Tool_Target_Date"] = safe_dt(tool_df[tool_target_date]) if tool_target_date else pd.NaT

# =========================================================
# Prepare MET table (Subclass-aware progress selection)
# =========================================================
met_df = met.copy()
met_df["_Link Code"] = ensure_str(met_df[met_link])

if met_wo:
    met_df["_WO"] = ensure_str(met_df[met_wo])
else:
    met_df["_WO"] = met_df["_Link Code"]  # fallback

met_df["_Subclass"] = ensure_str(met_df[met_sub]) if met_sub else ""

# Build MET progress by choosing correct column based on subclass
def met_progress_row(row):
    sub = _norm(row.get("_Subclass", ""))
    if "civil" in sub and met_civil_pct:
        return to_percent(row.get(met_civil_pct))
    if ("fiber" in sub or "fibre" in sub) and met_fiber_pct:
        return to_percent(row.get(met_fiber_pct))
    # If subclass empty or columns missing, use any progress
    if met_any_pct:
        return to_percent(row.get(met_any_pct))
    # Fallback: try civil then fiber
    if met_civil_pct:
        v = to_percent(row.get(met_civil_pct))
        if not np.isnan(v):
            return v
    if met_fiber_pct:
        v = to_percent(row.get(met_fiber_pct))
        if not np.isnan(v):
            return v
    return np.nan

met_df["_MET_Actual_%"] = met_df.apply(met_progress_row, axis=1)
met_df["_MET_Target_Date"] = safe_dt(met_df[met_target_date]) if met_target_date else pd.NaT

# =========================================================
# Merge (prefer WO match; if WO missing, Link Code match)
# =========================================================
merge_key = "_WO" if (tool_wo or met_wo) else "_Link Code"

df = pd.merge(
    tool_df,
    met_df[["_Link Code", "_WO", "_Subclass", "_MET_Actual_%", "_MET_Target_Date"]],
    on=merge_key,
    how="left",
    suffixes=("", "_met"),
    indicator=False,
)

# Fill MET Link Code if merge_key was WO and tool has link code
if "_Link Code_met" in df.columns:
    df["_Link Code_MET"] = df["_Link Code_met"].fillna(df["_Link Code"])
else:
    df["_Link Code_MET"] = df["_Link Code"]

# Record status and variance
df["Record Status"] = np.where(df["_MET_Actual_%"].isna(), "NO MET SHEET", "SHEET SYNCED")
df["Variance %"] = df["_MET_Actual_%"] - df["_Tool_Target_%"]
df["Critical Lag"] = df["Variance %"] <= -15

# Missing MET reports counted at Link Code level
tool_link_codes = set(tool_df["_Link Code"].dropna().unique())
met_link_codes  = set(met_df["_Link Code"].dropna().unique())
missing_link_codes = sorted(list(tool_link_codes - met_link_codes))

# =========================================================
# Global Filters (top bar like the screenshot)
# =========================================================
st.markdown('<p class="section-title">Delta Executive BI</p>', unsafe_allow_html=True)
st.markdown('<p class="section-sub">CONTRACTOR COMMITMENT ANALYSIS</p>', unsafe_allow_html=True)

# Build options for filters
def opt_list(series):
    vals = sorted([v for v in series.dropna().unique() if str(v).strip() != ""])
    return ["All"] + vals

# Some filters live on tool_df (authoritative master list)
f_link = st.session_state.get("f_link", "All")

c = st.columns([1.0, 1.0, 1.0, 1.0, 0.9, 1.0, 1.0, 1.0])
with c[0]:
    link_sel = st.selectbox("LINK CODE", opt_list(tool_df["_Link Code"]), index=0)
with c[1]:
    region_sel = st.selectbox("REGION", opt_list(tool_df[f_region]) if f_region else ["All"])
with c[2]:
    project_sel = st.selectbox("PROJECT", opt_list(tool_df[f_project]) if f_project else ["All"])
with c[3]:
    subclass_sel = st.selectbox("SUBCLASS", opt_list(tool_df["_Subclass"]) if tool_sub else ["All"])
with c[4]:
    year_sel = st.selectbox("YEAR", opt_list(tool_df[f_year]) if f_year else ["All"])
with c[5]:
    stage_sel = st.selectbox("STAGE", opt_list(tool_df[f_stage]) if f_stage else ["All"])
with c[6]:
    district_sel = st.selectbox("DISTRICT", opt_list(tool_df[f_district]) if f_district else ["All"])
with c[7]:
    supervisor_sel = st.selectbox("SUPERVISOR", opt_list(tool_df[f_supervisor]) if f_supervisor else ["All"])

# Apply filters to df (merged)
fdf = df.copy()

def apply_filter(df_, col_, val_):
    if col_ is None or val_ == "All":
        return df_
    return df_[df_[col_].astype(str) == str(val_)]

# Note: link filter uses Tool link code (master)
if link_sel != "All":
    fdf = fdf[fdf["_Link Code"] == link_sel]
if f_region and region_sel != "All":
    fdf = fdf[fdf[f_region].astype(str) == str(region_sel)]
if f_project and project_sel != "All":
    fdf = fdf[fdf[f_project].astype(str) == str(project_sel)]
if tool_sub and subclass_sel != "All":
    fdf = fdf[fdf["_Subclass"].astype(str) == str(subclass_sel)]
if f_year and year_sel != "All":
    fdf = fdf[fdf[f_year].astype(str) == str(year_sel)]
if f_stage and stage_sel != "All":
    fdf = fdf[fdf[f_stage].astype(str) == str(stage_sel)]
if f_district and district_sel != "All":
    fdf = fdf[fdf[f_district].astype(str) == str(district_sel)]
if f_supervisor and supervisor_sel != "All":
    fdf = fdf[fdf[f_supervisor].astype(str) == str(supervisor_sel)]

st.divider()

# =========================================================
# KPI summary (Top 5 cards)
# =========================================================
kpi_cols = st.columns(5)
total_link = int(fdf["_Link Code"].nunique())

avg_tool = float(np.nanmean(fdf["_Tool_Target_%"])) if fdf["_Tool_Target_%"].notna().any() else 0.0
avg_met  = float(np.nanmean(fdf["_MET_Actual_%"])) if fdf["_MET_Actual_%"].notna().any() else 0.0

# Missing MET reports: still compute based on current filters except link_code filter:
# If user filters region etc, we compute missing within that filtered scope.
scope_tool = tool_df.copy()
if f_region and region_sel != "All":
    scope_tool = scope_tool[scope_tool[f_region].astype(str) == str(region_sel)]
if f_project and project_sel != "All":
    scope_tool = scope_tool[scope_tool[f_project].astype(str) == str(project_sel)]
if tool_sub and subclass_sel != "All":
    scope_tool = scope_tool[scope_tool["_Subclass"].astype(str) == str(subclass_sel)]
if f_year and year_sel != "All":
    scope_tool = scope_tool[scope_tool[f_year].astype(str) == str(year_sel)]
if f_stage and stage_sel != "All":
    scope_tool = scope_tool[scope_tool[f_stage].astype(str) == str(stage_sel)]
if f_district and district_sel != "All":
    scope_tool = scope_tool[scope_tool[f_district].astype(str) == str(district_sel)]
if f_supervisor and supervisor_sel != "All":
    scope_tool = scope_tool[scope_tool[f_supervisor].astype(str) == str(supervisor_sel)]

scope_link_codes = set(scope_tool["_Link Code"].dropna().unique())
missing_scope = sorted(list(scope_link_codes - met_link_codes))
missing_count = len(missing_scope)

critical_lags = int(fdf["Critical Lag"].sum()) if "Critical Lag" in fdf.columns else 0

def kpi_card(title, value, suffix=""):
    st.markdown(
        f"""
        <div class="kpi-card">
          <p class="kpi-title">{title}</p>
          <p class="kpi-value">{value}{suffix}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with kpi_cols[0]:
    kpi_card("TOTAL LINK CODES", total_link)
with kpi_cols[1]:
    kpi_card("DAWAIYAT AVG TARGET", f"{avg_tool:.1f}", "%")
with kpi_cols[2]:
    kpi_card("MET AVG ACTUAL", f"{avg_met:.1f}", "%")
with kpi_cols[3]:
    kpi_card("MISSING MET REPORTS", missing_count)
with kpi_cols[4]:
    kpi_card("CRITICAL LAGS (>15%)", critical_lags)

st.divider()

# =========================================================
# Strategic 3×3 grid charts
# =========================================================
st.markdown('<p class="section-title">Strategic Performance Grids</p>', unsafe_allow_html=True)
st.markdown('<p class="section-sub">REAL-TIME COMPARATIVE ANALYTICS</p>', unsafe_allow_html=True)

tpl = plotly_template(dark_mode)

row1 = st.columns(3)
row2 = st.columns(3)
row3 = st.columns(3)

# Chart 1: Regional Coverage donut
with row1[0]:
    st.markdown("**GEOGRAPHY DISTRIBUTION**  \nRegional Coverage")
    if f_region:
        g = fdf.groupby(f_region)["_Link Code"].nunique().reset_index(name="count")
        fig = px.pie(g, values="count", names=f_region, hole=0.62)
        fig.update_layout(template=tpl, margin=dict(l=10,r=10,t=20,b=10), height=320, showlegend=True)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Region column not found in the Tool sheet.")

# Chart 2: Dawaiyat vs MET trend (weekly)
with row1[1]:
    st.markdown("**PROGRESS COMPARISON**  \nDawaiyat vs. MET Trend")
    dt_src = fdf["_Tool_Target_Date"]
    if dt_src.notna().any():
        wk = make_week_key(dt_src)
        tmp = fdf.copy()
        tmp["_wk"] = wk
        tt = tmp.groupby("_wk")[["_Tool_Target_%","_MET_Actual_%"]].mean(numeric_only=True).reset_index()
        tt = tt.dropna(subset=["_wk"]).sort_values("_wk")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=tt["_wk"], y=tt["_MET_Actual_%"], mode="lines+markers", name="Site Actual"))
        fig.add_trace(go.Scatter(x=tt["_wk"], y=tt["_Tool_Target_%"], mode="lines+markers", name="Tool Target"))
        fig.update_layout(template=tpl, height=320, margin=dict(l=10,r=10,t=10,b=10), xaxis_title="", yaxis_title="%")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No commitment target dates found for trend chart.")

# Chart 3: Operational Status donut
with row1[2]:
    st.markdown("**WORK ORDER HEALTH**  \nOperational Status")
    status = pd.Series(np.where(fdf["Record Status"]=="NO MET SHEET","Missing Site Report",
                      np.where(fdf["_MET_Actual_%"]>=100,"Completed","In Progress")))
    s = status.value_counts().reset_index()
    s.columns = ["Status","Count"]
    fig = px.pie(s, values="Count", names="Status", hole=0.62)
    fig.update_layout(template=tpl, height=320, margin=dict(l=10,r=10,t=20,b=10))
    st.plotly_chart(fig, use_container_width=True)

# Chart 4: Supervisor performance
with row2[0]:
    st.markdown("**PRODUCTIVITY BY LEAD**  \nField Supervisor Performance")
    if f_supervisor:
        tmp = fdf.copy()
        agg = tmp.groupby(f_supervisor).agg(
            Units_Managed=("_Link Code","nunique"),
            Avg_Progress=("_MET_Actual_%","mean")
        ).reset_index()
        agg["Avg_Progress"] = agg["Avg_Progress"].fillna(0)
        agg = agg.sort_values("Units_Managed", ascending=False).head(10)
        fig = go.Figure()
        fig.add_trace(go.Bar(x=agg[f_supervisor], y=agg["Avg_Progress"], name="Avg Progress (%)"))
        fig.add_trace(go.Bar(x=agg[f_supervisor], y=agg["Units_Managed"], name="Units Managed", opacity=0.45))
        fig.update_layout(barmode="group", template=tpl, height=320, margin=dict(l=10,r=10,t=10,b=10))
        fig.update_xaxes(tickangle=-20)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Supervisor column not found in the Tool sheet.")

# Chart 5: Discipline distribution donut
with row2[1]:
    st.markdown("**WORK SUBCLASS**  \nDiscipline Distribution")
    if tool_sub:
        g = fdf["_Subclass"].replace("", np.nan).fillna("Unknown").value_counts().reset_index()
        g.columns = ["Subclass","Count"]
        fig = px.pie(g, values="Count", names="Subclass", hole=0.62)
        fig.update_layout(template=tpl, height=320, margin=dict(l=10,r=10,t=20,b=10))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Subclass column not found.")

# Chart 6: System variance gap area (weekly)
with row2[2]:
    st.markdown("**MET LAGGING TOOL**  \nSystem Variance Gap")
    dt_src = fdf["_Tool_Target_Date"]
    if dt_src.notna().any():
        wk = make_week_key(dt_src)
        tmp = fdf.copy()
        tmp["_wk"] = wk
        gg = tmp.groupby("_wk")["Variance %"].mean().reset_index()
        gg = gg.dropna(subset=["_wk"]).sort_values("_wk")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=gg["_wk"], y=gg["Variance %"], mode="lines", fill="tozeroy", name="Gap (MET-Tool)"))
        fig.update_layout(template=tpl, height=320, margin=dict(l=10,r=10,t=10,b=10), yaxis_title="%")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No dates to compute variance trend.")

# Chart 7: District loading top 6
with row3[0]:
    st.markdown("**TOP 6 ACTIVE DISTRICTS**  \nDistrict Loading Summary")
    if f_district:
        g = fdf.groupby(f_district)["_Link Code"].nunique().reset_index(name="count")
        g = g.sort_values("count", ascending=False).head(6)
        fig = px.bar(g, x="count", y=f_district, orientation="h")
        fig.update_layout(template=tpl, height=320, margin=dict(l=10,r=10,t=10,b=10), xaxis_title="", yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("District column not found.")

# Chart 8: Stage breakdown pie
with row3[1]:
    st.markdown("**STAGE BREAKDOWN**  \nExecution Lifecycle Stage")
    if f_stage:
        g = fdf[f_stage].fillna("Unknown").value_counts().reset_index()
        g.columns = ["Stage","Count"]
        fig = px.pie(g, values="Count", names="Stage", hole=0.0)
        fig.update_layout(template=tpl, height=320, margin=dict(l=10,r=10,t=20,b=10))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Stage column not found.")

# Chart 9: Projected vs realized (step)
with row3[2]:
    st.markdown("**DELIVERY STEP TREND**  \nProjected vs. Realized")
    dt_src = fdf["_Tool_Target_Date"]
    if dt_src.notna().any():
        wk = make_week_key(dt_src)
        tmp = fdf.copy()
        tmp["_wk"] = wk
        tt = tmp.groupby("_wk")[["_Tool_Target_%","_MET_Actual_%"]].mean(numeric_only=True).reset_index()
        tt = tt.dropna(subset=["_wk"]).sort_values("_wk")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=tt["_wk"], y=tt["_MET_Actual_%"], mode="lines+markers", name="Realized"))
        fig.add_trace(go.Scatter(x=tt["_wk"], y=tt["_Tool_Target_%"], mode="lines", name="Target", line=dict(dash="dot"), shape="hv"))
        fig.update_layout(template=tpl, height=320, margin=dict(l=10,r=10,t=10,b=10), yaxis_title="%")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No dates to build projected vs realized chart.")

st.divider()

# =========================================================
# Actionable alert: Missing MET reports
# =========================================================
st.markdown("### 🚨 Missing MET Actual Reports")
st.caption(f"{missing_count} Link Codes require urgent site status updates (based on your current filter scope).")

if missing_count == 0:
    st.success("✅ No missing MET reports in the selected scope.")
else:
    # Build a compact card grid of missing items
    miss_df = scope_tool[scope_tool["_Link Code"].isin(missing_scope)].copy()
    # Add a representative target progress and district/region
    miss_df["_Target"] = miss_df["_Tool_Target_%"]
    # Keep unique link codes
    miss_u = miss_df.sort_values("_Target", ascending=False).groupby("_Link Code").head(1)

    cards_per_row = 5
    rows = int(np.ceil(len(miss_u) / cards_per_row))
    for r in range(rows):
        cols = st.columns(cards_per_row)
        chunk = miss_u.iloc[r*cards_per_row:(r+1)*cards_per_row]
        for i, (_, row) in enumerate(chunk.iterrows()):
            with cols[i]:
                lc = row["_Link Code"]
                reg = row[f_region] if f_region else ""
                dist = row[f_district] if f_district else ""
                tgt = row["_Target"]
                st.markdown(
                    f"""
                    <div class="kpi-card">
                      <p style="margin:0; font-weight:800;">{lc}</p>
                      <p style="margin:2px 0 10px 0; font-size:12px; opacity:.75;">{reg} • {dist}</p>
                      <p class="kpi-title">TARGET PROGRESS</p>
                      <p style="margin:0; font-size:18px; font-weight:800;">{tgt:.0f}%</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                if st.button("👤 ASSIGN UPDATE", key=f"assign_{lc}", use_container_width=True):
                    st.toast(f"Assigned update request for {lc}", icon="✅")

st.divider()

# =========================================================
# Tables: Audit + Master + Downloads
# =========================================================
tab_audit, tab_master = st.tabs(["🧾 AUDIT", "🗂️ MASTER"])

# Build Audit table
audit_cols = [
    "_Link Code", "_WO",
    f_region, f_district, f_supervisor,
    "_Tool_Target_%", "_MET_Actual_%", "Variance %",
    "_Tool_Target_Date", "Record Status"
]
audit_cols = [c for c in audit_cols if c is not None and c in fdf.columns]

audit = fdf[audit_cols].copy()
audit = audit.rename(columns={
    "_Link Code":"LINK CODE",
    "_WO":"WORK ORDER",
    "_Tool_Target_%":"TARGET %",
    "_MET_Actual_%":"ACTUAL %",
    "_Tool_Target_Date":"TARGET DATE",
})

# Add search
with tab_audit:
    st.subheader("Audit Comparison")
    st.caption("Analyzing variance between commitments and site reality")

    q = st.text_input("Search Link Code, WO, or Supervisor...", "")
    show = audit.copy()
    if q.strip():
        qq = q.strip().lower()
        mask = pd.Series(False, index=show.index)
        for col in show.columns:
            mask = mask | show[col].astype(str).str.lower().str.contains(qq, na=False)
        show = show[mask]

    st.dataframe(show, use_container_width=True, height=420)

    csv = show.to_csv(index=False).encode("utf-8-sig")
    st.download_button("⬇️ EXPORT CSV", data=csv, file_name="audit_comparison.csv", mime="text/csv")

# Master table (keep some key fields if available)
with tab_master:
    st.subheader("Master Data Inventory")
    st.caption("Full technical drill-down of all uploaded project parameters")

    master_cols = [
        "_Link Code","_WO",
        f_supervisor, f_category, f_stage, f_region, f_district,
        "_Tool_Target_%","_MET_Actual_%","Record Status"
    ]
    master_cols = [c for c in master_cols if c is not None and c in fdf.columns]
    master = fdf[master_cols].copy()
    master = master.rename(columns={
        "_Link Code":"LINK CODE",
        "_WO":"WORK ORDER",
        "_Tool_Target_%":"TOOL TARGET %",
        "_MET_Actual_%":"MET ACTUAL %",
        "Record Status":"RECORD STATUS"
    })

    q2 = st.text_input("Search Master...", "", key="master_search")
    show2 = master.copy()
    if q2.strip():
        qq = q2.strip().lower()
        mask = pd.Series(False, index=show2.index)
        for col in show2.columns:
            mask = mask | show2[col].astype(str).str.lower().str.contains(qq, na=False)
        show2 = show2[mask]

    st.dataframe(show2, use_container_width=True, height=420)
    csv2 = show2.to_csv(index=False).encode("utf-8-sig")
    st.download_button("⬇️ EXPORT CSV", data=csv2, file_name="master_data_inventory.csv", mime="text/csv")

# Footer note about progress column logic
st.caption(
    "Note: MET progress is selected by Subclass: **Civil → Civil Completion %**, **Fiber → Fiber Completion %**. "
    "Tool progress is from **Percentage of Completion**."
)
