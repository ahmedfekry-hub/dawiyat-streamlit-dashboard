import io
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

st.set_page_config(page_title="Dawiyat Project Tracker Dashboard", layout="wide")

# ----------------------------
# Styling (light + dark)
# ----------------------------
BASE_CSS = """
<style>
/* Layout */
.block-container { padding-top: 1.2rem; padding-bottom: 2rem; max-width: 1400px; }
header, footer { visibility: hidden; height: 0px; }
[data-testid="stSidebar"] { padding-top: 1rem; }

/* Top bar (pseudo) */
.topbar { display:flex; justify-content:space-between; align-items:center; gap:12px; padding: 6px 2px 14px 2px; }
.brand { display:flex; align-items:center; gap:10px; }
.brand .logo { width:36px; height:36px; border-radius:10px; background: linear-gradient(135deg,#6D28D9,#EC4899,#22C55E); display:flex; align-items:center; justify-content:center; color:white; font-weight:800; }
.brand .title { font-size: 20px; font-weight: 800; }
.brand .subtitle { font-size: 11px; color:#64748B; margin-top:-2px; letter-spacing:.08em; text-transform:uppercase; }
.actions { display:flex; align-items:center; gap:10px; }
.pill { border-radius: 999px; padding: 8px 12px; background:#0F172A0D; border: 1px solid #E2E8F0; font-weight:600; display:flex; align-items:center; gap:8px; }
.pill .dot { width:8px; height:8px; border-radius:50%; background:#22C55E; }
.btn { border-radius: 999px; padding: 10px 14px; background:#2563EB; color:white; font-weight:700; border:none; }
.btn:focus { outline:none; box-shadow:none; }
.help { width:34px; height:34px; border-radius: 999px; border:1px solid #E2E8F0; background:#FFFFFF; display:flex; align-items:center; justify-content:center; font-weight:900; cursor:pointer; }

/* KPI cards */
.kpi-row { display:grid; grid-template-columns: repeat(5, 1fr); gap:14px; }
.kpi { border-radius: 16px; border:1px solid #E2E8F0; background:white; padding: 14px 14px 12px 14px; box-shadow: 0 1px 0 rgba(0,0,0,.02); min-height: 92px; position:relative; }
.kpi .label { font-size:11px; color:#64748B; font-weight:800; letter-spacing:.08em; text-transform:uppercase; }
.kpi .value { font-size: 28px; font-weight: 900; margin-top: 3px; line-height:1.0; }
.kpi .spark { position:absolute; top:14px; right:14px; opacity:.22; }
.kpi .bar { height: 3px; width: 34px; background:#E2E8F0; border-radius: 99px; margin-top: 8px; }
.kpi.blue .bar { background:#2563EB; }
.kpi.purple .bar { background:#7C3AED; }
.kpi.green .bar { background:#22C55E; }
.kpi.red .bar { background:#EF4444; }
.kpi.orange .bar { background:#F59E0B; }

/* Sections */
.h2 { font-size: 18px; font-weight: 900; margin: 18px 0 2px 0; display:flex; align-items:center; gap:10px;}
.h2::before { content:""; width:4px; height:18px; border-radius: 99px; background:#2563EB; display:inline-block; }
.muted { color:#64748B; font-size: 11px; letter-spacing:.08em; text-transform: uppercase; font-weight: 900; margin-bottom: 10px; }

/* Grid cards */
.grid3 { display:grid; grid-template-columns: repeat(3, 1fr); gap:14px; }
.card { border-radius: 16px; border:1px solid #E2E8F0; background:white; padding: 14px; box-shadow: 0 1px 0 rgba(0,0,0,.02); }
.card .ctitle { font-size: 11px; color:#64748B; letter-spacing:.08em; text-transform: uppercase; font-weight: 900; margin-bottom: 6px; }
.card .csubtitle { font-size: 13px; font-weight: 800; margin-bottom: 10px; }

/* Alert section */
.alert-wrap { border: 1px solid #FCA5A5; background:#FEF2F2; border-radius: 16px; padding: 14px; }
.alert-head { display:flex; justify-content:space-between; align-items:center; gap:10px; }
.alert-title { font-size: 16px; font-weight: 900; color:#991B1B; display:flex; align-items:center; gap:10px;}
.badge { background:#FEE2E2; color:#991B1B; border:1px solid #FCA5A5; padding: 6px 10px; border-radius: 999px; font-size: 11px; font-weight: 900; letter-spacing:.08em; text-transform: uppercase;}
.cards-scroll { display:flex; gap: 10px; overflow-x:auto; padding-top: 10px; padding-bottom: 6px; }
.mini { min-width: 260px; max-width: 260px; border-radius: 14px; border:1px solid #E2E8F0; background:white; padding: 12px; }
.mini .idx { font-size:11px; color:#64748B; font-weight:900; letter-spacing:.08em; text-transform: uppercase; }
.mini .lc { font-weight: 900; margin-top: 6px; }
.mini .loc { font-size: 11px; color:#64748B; margin-top: 2px; }
.mini .prog { font-size: 11px; margin-top: 10px; color:#64748B; font-weight: 900; letter-spacing:.08em; text-transform: uppercase;}
.mini .pval { font-weight: 900; float:right; color:#0F172A; }
.mini .btn2 { margin-top: 10px; border-radius: 10px; width:100%; padding:10px 12px; border:1px solid #CBD5E1; background:#FFFFFF; font-weight: 900; color:#2563EB; }

/* Tables */
.table-title { display:flex; align-items:center; justify-content:space-between; gap:10px; margin: 10px 0 8px 0; }
.table-title .left { display:flex; align-items:center; gap:10px; }
.pill2 { background:#F1F5F9; border:1px solid #E2E8F0; border-radius:999px; padding:6px 10px; font-size: 11px; font-weight: 900; letter-spacing:.08em; text-transform: uppercase; color:#0F172A;}
.seg { display:flex; gap:6px; }
.seg button { border-radius: 999px; padding: 8px 12px; border:1px solid #E2E8F0; background:white; font-weight: 900; }
.seg button.active { background:#2563EB; color:white; border-color:#2563EB; }

/* Dark mode overrides */
html[data-theme="dark"] .kpi, html[data-theme="dark"] .card, html[data-theme="dark"] .mini {
  background: #0B1220; border-color: #1E293B; box-shadow:none;
}
html[data-theme="dark"] .kpi .label, html[data-theme="dark"] .muted, html[data-theme="dark"] .card .ctitle {
  color:#94A3B8;
}
html[data-theme="dark"] .brand .subtitle { color:#94A3B8; }
html[data-theme="dark"] .pill { background:#0B1220; border-color:#1E293B; color:#E2E8F0; }
html[data-theme="dark"] .help { background:#0B1220; border-color:#1E293B; color:#E2E8F0; }
html[data-theme="dark"] .alert-wrap { background:#2A0F12; border-color:#7F1D1D; }
html[data-theme="dark"] .badge { background:#3B0A0C; border-color:#7F1D1D; color:#FCA5A5; }
</style>
"""

st.markdown(BASE_CSS, unsafe_allow_html=True)

# ----------------------------
# Sidebar settings
# ----------------------------
with st.sidebar:
    st.subheader("Settings")
    dark_mode = st.toggle("Dark Mode", value=True)

# Streamlit theme switch (simple)
if dark_mode:
    st.markdown(
        """
        <script>
        document.documentElement.setAttribute('data-theme','dark');
        </script>
        """,
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        """
        <script>
        document.documentElement.setAttribute('data-theme','light');
        </script>
        """,
        unsafe_allow_html=True,
    )

# ----------------------------
# Dashboard Guide (Help)
# ----------------------------
if "show_guide" not in st.session_state:
    st.session_state.show_guide = False

GUIDE_MD = """
## Dashboard Guide

### 1) The KPI Summary (Top 5 Cards)
- **Total Link Codes**: total volume of projects currently in your Dawaiyat Service Tool.
- **Dawaiyat Avg Target**: average progress your contractors should have made according to the master tool.
- **MET Avg Actual**: average progress actually reported from the sites.
- **Missing MET Reports**: your **Blind Spot** count (link codes in the tool but missing in MET).
- **Critical Lags (>15%)**: projects where MET progress is behind the tool commitment by more than 15%.

### 2) Strategic 3×3 Chart Grid
- **Regional Coverage**: where your projects are concentrated (Makkah, Jizan, etc.).
- **Dawaiyat vs. MET Trend**: green (MET) should follow blue (Tool). A widening gap = delivery problem.
- **Operational Status**: Completed / In Progress / Missing site reports.
- **Field Supervisor Performance**: accountability (units per supervisor + average progress).
- **Discipline Distribution**: Civil vs Fiber breakdown.
- **System Variance Gap**: red area chart for the “pain” — office vs site reality mismatch.
- **District Loading**: bottleneck districts by volume.
- **Execution Lifecycle**: phase distribution (Implementation vs Handover, etc.).
- **Projected vs Realized**: step chart for the rhythm of delivery.

### 3) Actionable Alert: Missing MET Reports
Each card is a Link Code that has no MET reporting. “Assign Update” simulates assigning a coordinator to fix the gap.

### 4) Audit Comparison Table
Drill-down reconciliation.
- **Variance** = MET − Tool (green ahead, red behind)
- **Missing MET** explicitly flagged when reconciliation fails.
"""

# ----------------------------
# Helpers
# ----------------------------
RENAME_TOOL = {
    "Link Code": "link_code",
    "Work Order": "work_order",
    "District": "district",
    "Subclass": "subclass",
    "WO Cost": "wo_cost",
    "Percentage of Completion": "tool_progress",
    "Region": "region",
    "Project": "project",
    "Year": "year",
    "Stage": "stage",
    "Supervisor": "supervisor",
    "Category": "category",
}

RENAME_MET = {
    "Link Code": "link_code",
    "Work Order": "work_order",
    "District": "district",
    "Subclass": "subclass",
    "Civil Completion  %": "met_civil_progress",
    "Fiber Completion  %": "met_fiber_progress",
    "MST Target": "mst_target_flag",
    "Region": "region",
    "Project": "project",
    "Year": "year",
    "Stage": "stage",
    "Supervisor": "supervisor",
}

REQ_TOOL_SHEET = "Dawaiyat Service Tool"
REQ_MET_SHEET = "MET Actual progress"

def normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df

def safe_float(x):
    try:
        if pd.isna(x):
            return np.nan
        if isinstance(x, str):
            x = x.replace("%", "").replace(",", "").strip()
        return float(x)
    except Exception:
        return np.nan

def ensure_key_cols(df: pd.DataFrame, keys):
    missing = [k for k in keys if k not in df.columns]
    return missing

def read_any_excel_or_csv(file):
    name = getattr(file, "name", "upload")
    if name.lower().endswith(".csv"):
        df = pd.read_csv(file)
        return normalize_cols(df)
    else:
        df = pd.read_excel(file)
        return normalize_cols(df)

def load_from_workbook(wb_file):
    xl = pd.ExcelFile(wb_file)
    sheets = xl.sheet_names
    if REQ_TOOL_SHEET not in sheets or REQ_MET_SHEET not in sheets:
        raise ValueError(f"Workbook must contain sheets: '{REQ_TOOL_SHEET}' and '{REQ_MET_SHEET}'. Found: {sheets}")
    tool = normalize_cols(pd.read_excel(wb_file, sheet_name=REQ_TOOL_SHEET))
    met  = normalize_cols(pd.read_excel(wb_file, sheet_name=REQ_MET_SHEET))
    return tool, met

def rename_and_clean_tool(df):
    df = df.rename(columns={k:v for k,v in RENAME_TOOL.items() if k in df.columns})
    df = df.copy()
    # normalize text
    for c in ["link_code","work_order","district","subclass","region","project","stage","supervisor","category"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()
    if "tool_progress" in df.columns:
        df["tool_progress"] = df["tool_progress"].apply(safe_float)
    if "wo_cost" in df.columns:
        df["wo_cost"] = df["wo_cost"].apply(safe_float)
    # subclass normalize
    if "subclass" in df.columns:
        df["subclass"] = df["subclass"].str.title()
        df.loc[df["subclass"].str.contains("Civil", na=False), "subclass"] = "Civil"
        df.loc[df["subclass"].str.contains("Fiber", na=False), "subclass"] = "Fiber"
    return df

def rename_and_clean_met(df):
    df = df.rename(columns={k:v for k,v in RENAME_MET.items() if k in df.columns})
    df = df.copy()
    for c in ["link_code","work_order","district","subclass","region","project","stage","supervisor"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()
    for c in ["met_civil_progress","met_fiber_progress"]:
        if c in df.columns:
            df[c] = df[c].apply(safe_float)
    if "subclass" in df.columns:
        df["subclass"] = df["subclass"].str.title()
        df.loc[df["subclass"].str.contains("Civil", na=False), "subclass"] = "Civil"
        df.loc[df["subclass"].str.contains("Fiber", na=False), "subclass"] = "Fiber"
    return df

def met_progress_row(row):
    """
    If subclass is Civil -> use met_civil_progress
    If subclass is Fiber -> use met_fiber_progress
    """
    sc = str(row.get("subclass", "")).strip().lower()
    if sc == "civil":
        return row.get("met_civil_progress", np.nan)
    if sc == "fiber":
        return row.get("met_fiber_progress", np.nan)
    # if unknown subclass, fallback: take max available
    return np.nanmax([row.get("met_civil_progress", np.nan), row.get("met_fiber_progress", np.nan)])

def build_master(tool_df, met_df):
    # Required keys for matching
    keys = ["link_code", "work_order", "subclass"]

    # Keep only rows where key columns exist
    for dfname, df in [("Tool", tool_df), ("MET", met_df)]:
        miss = ensure_key_cols(df, keys)
        if miss:
            raise ValueError(f"{dfname} missing columns: {miss}")

    # merge
    merged = tool_df.merge(
        met_df,
        on=keys,
        how="left",
        suffixes=("_tool","_met")
    )

    # decide met progress by subclass (civil/fiber)
    merged["met_progress"] = merged.apply(met_progress_row, axis=1)

    # variance and statuses
    merged["variance"] = merged["met_progress"] - merged["tool_progress"]
    merged["has_met"] = ~merged["met_progress"].isna()

    merged["record_status"] = np.where(
        merged["has_met"], "SHEET SYNCED", "NO MET SHEET"
    )

    # operational status buckets
    def op_bucket(r):
        if not r["has_met"]:
            return "Missing Site Report"
        if pd.isna(r["met_progress"]):
            return "Missing Site Report"
        if r["met_progress"] >= 99:
            return "Completed"
        return "In Progress"

    merged["operational_status"] = merged.apply(op_bucket, axis=1)

    # critical lag (tool - met > 15)
    merged["critical_lag"] = np.where(
        (merged["has_met"]) & ((merged["tool_progress"] - merged["met_progress"]) > 15),
        True, False
    )

    # Minimal display-friendly columns
    show_cols = [
        "link_code","work_order","district","subclass",
        "tool_progress","met_progress","variance",
        "region","project","year","stage","supervisor",
        "operational_status","record_status",
        "wo_cost"
    ]
    for c in show_cols:
        if c not in merged.columns:
            merged[c] = np.nan

    return merged[show_cols]

def pct(x):
    if pd.isna(x):
        return np.nan
    return float(x)

def fmt_pct(x):
    if pd.isna(x):
        return ""
    return f"{x:.1f}%"

def safe_int(x):
    try:
        return int(x)
    except:
        return 0

# ----------------------------
# Header
# ----------------------------
left = """
<div class="topbar">
  <div class="brand">
    <div class="logo">▦</div>
    <div>
      <div class="title">Delta Executive BI</div>
      <div class="subtitle">Contractor Commitment Analysis</div>
    </div>
  </div>
</div>
"""
st.markdown(left, unsafe_allow_html=True)

# ----------------------------
# Upload Section
# ----------------------------
st.caption("Commitment vs Actual | Executive Decision View")

upload_mode = st.radio(
    "Upload Mode",
    ["Single Excel (contains both sheets)", "Two separate files (Tool + MET)"],
    horizontal=True,
    label_visibility="collapsed",
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
        # For XLSX (single-sheet)
        tool = read_any_excel_or_csv(tool_file)
        met  = read_any_excel_or_csv(met_file)

if tool is None or met is None:
    st.info("Please upload the Excel file(s) to start")
    st.stop()

# Clean & normalize
tool_df = rename_and_clean_tool(tool)
met_df  = rename_and_clean_met(met)

# Build master dataset
master = build_master(tool_df, met_df)

# ----------------------------
# Filters row
# ----------------------------
flt_cols = st.columns([0.08, 0.13, 0.13, 0.13, 0.13, 0.12, 0.13, 0.13, 0.15])
with flt_cols[0]:
    st.markdown('<div class="pill"><span class="dot"></span>FILTERS</div>', unsafe_allow_html=True)

def pick(col, label, values):
    with col:
        return st.selectbox(label, values, index=0, label_visibility="collapsed")

link_vals = ["All Link Code"] + sorted(master["link_code"].dropna().unique().tolist())
reg_vals  = ["All Region"] + sorted(master["region"].dropna().unique().tolist())
proj_vals = ["All Project"] + sorted(master["project"].dropna().unique().tolist())
sub_vals  = ["All Subclass"] + sorted(master["subclass"].dropna().unique().tolist())
year_vals = ["All Year"] + sorted([str(y) for y in master["year"].dropna().unique().tolist()])
stage_vals= ["All Stage"] + sorted(master["stage"].dropna().unique().tolist())
dist_vals = ["All District"] + sorted(master["district"].dropna().unique().tolist())
sup_vals  = ["All Supervisor"] + sorted(master["supervisor"].dropna().unique().tolist())

f_link = pick(flt_cols[1], "LINK CODE", link_vals)
f_reg  = pick(flt_cols[2], "REGION", reg_vals)
f_proj = pick(flt_cols[3], "PROJECT", proj_vals)
f_sub  = pick(flt_cols[4], "SUBCLASS", sub_vals)
f_year = pick(flt_cols[5], "YEAR", year_vals)
f_stage= pick(flt_cols[6], "STAGE", stage_vals)
f_dist = pick(flt_cols[7], "DISTRICT", dist_vals)
f_sup  = pick(flt_cols[8], "SUPERVISOR", sup_vals)

filtered = master.copy()
if f_link != "All Link Code": filtered = filtered[filtered["link_code"] == f_link]
if f_reg  != "All Region":    filtered = filtered[filtered["region"] == f_reg]
if f_proj != "All Project":   filtered = filtered[filtered["project"] == f_proj]
if f_sub  != "All Subclass":  filtered = filtered[filtered["subclass"] == f_sub]
if f_year != "All Year":      filtered = filtered[filtered["year"].astype(str) == f_year]
if f_stage!= "All Stage":     filtered = filtered[filtered["stage"] == f_stage]
if f_dist != "All District":  filtered = filtered[filtered["district"] == f_dist]
if f_sup  != "All Supervisor":filtered = filtered[filtered["supervisor"] == f_sup]

# ----------------------------
# KPI Summary
# ----------------------------
total_link_codes = filtered["link_code"].nunique()
avg_tool = filtered["tool_progress"].dropna().mean()
avg_met  = filtered["met_progress"].dropna().mean()
missing_met = filtered[~filtered["has_met"]]["link_code"].nunique()
critical_lags = int(filtered["critical_lag"].sum())

k1, k2, k3, k4, k5 = st.columns(5)

def kpi_html(label, value, cls, icon=""):
    return f"""
    <div class="kpi {cls}">
      <div class="label">{label}</div>
      <div class="value">{value}</div>
      <div class="bar"></div>
      <div class="spark">{icon}</div>
    </div>
    """

k1.markdown(kpi_html("TOTAL LINK CODES", f"{total_link_codes}", "blue", "∿"), unsafe_allow_html=True)
k2.markdown(kpi_html("DAWAIYAT AVG TARGET", fmt_pct(avg_tool), "purple", "◎"), unsafe_allow_html=True)
k3.markdown(kpi_html("MET AVG ACTUAL", fmt_pct(avg_met), "green", "↗"), unsafe_allow_html=True)
k4.markdown(kpi_html("MISSING MET REPORTS", f"{missing_met}", "red", "⦸"), unsafe_allow_html=True)
k5.markdown(kpi_html("CRITICAL LAGS (>15%)", f"{critical_lags}", "orange", "⚠"), unsafe_allow_html=True)

# ----------------------------
# Strategic 3x3 Grid
# ----------------------------
st.markdown('<div class="h2">Strategic Performance Grids</div>', unsafe_allow_html=True)
st.markdown('<div class="muted">REAL-TIME COMPARATIVE ANALYTICS</div>', unsafe_allow_html=True)

g1, g2, g3 = st.columns(3)

# 1) Regional Coverage (donut)
with g1:
    st.markdown('<div class="card"><div class="ctitle">GEOGRAPHY DISTRIBUTION</div><div class="csubtitle">Regional Coverage</div>', unsafe_allow_html=True)
    reg_counts = filtered.groupby("region")["link_code"].nunique().reset_index(name="count")
    if reg_counts.empty:
        st.write("No data")
    else:
        fig = px.pie(reg_counts, names="region", values="count", hole=0.65)
        fig.update_traces(textposition="outside", textinfo="label+value")
        fig.update_layout(margin=dict(l=10,r=10,t=10,b=10), showlegend=True, height=280)
        st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# 2) Dawaiyat vs MET Trend (line)
with g2:
    st.markdown('<div class="card"><div class="ctitle">PROGRESS COMPARISON</div><div class="csubtitle">Dawaiyat vs. MET Trend</div>', unsafe_allow_html=True)
    # build pseudo timeline by sorting link codes (or work orders) for visual rhythm
    trend = filtered.copy()
    trend["x"] = range(1, len(trend)+1)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=trend["x"], y=trend["met_progress"], mode="lines+markers", name="Site Actual"))
    fig.add_trace(go.Scatter(x=trend["x"], y=trend["tool_progress"], mode="lines+markers", name="Tool Target"))
    fig.update_layout(height=280, margin=dict(l=10,r=10,t=10,b=10), legend=dict(orientation="h"))
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# 3) Operational Status (pie)
with g3:
    st.markdown('<div class="card"><div class="ctitle">WORK ORDER HEALTH</div><div class="csubtitle">Operational Status</div>', unsafe_allow_html=True)
    op_counts = filtered["operational_status"].value_counts().reset_index()
    op_counts.columns = ["status","count"]
    if op_counts.empty:
        st.write("No data")
    else:
        fig = px.pie(op_counts, names="status", values="count", hole=0.0)
        fig.update_layout(height=280, margin=dict(l=10,r=10,t=10,b=10), showlegend=True)
        st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

g4, g5, g6 = st.columns(3)

# 4) Field Supervisor Performance (bar + units)
with g4:
    st.markdown('<div class="card"><div class="ctitle">PRODUCTIVITY BY LEAD</div><div class="csubtitle">Field Supervisor Performance</div>', unsafe_allow_html=True)
    sup = filtered.dropna(subset=["supervisor"]).copy()
    if sup.empty:
        st.write("No data")
    else:
        agg = sup.groupby("supervisor").agg(
            avg_progress=("met_progress","mean"),
            units=("work_order","count")
        ).reset_index().sort_values("avg_progress", ascending=False).head(10)
        fig = go.Figure()
        fig.add_trace(go.Bar(x=agg["supervisor"], y=agg["avg_progress"], name="Avg Progress (%)"))
        fig.add_trace(go.Bar(x=agg["supervisor"], y=agg["units"], name="Units Managed", yaxis="y2", opacity=0.6))
        fig.update_layout(
            height=280, margin=dict(l=10,r=10,t=10,b=10),
            yaxis=dict(title="Avg Progress"),
            yaxis2=dict(title="Units", overlaying="y", side="right"),
            legend=dict(orientation="h"),
            barmode="group"
        )
        st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# 5) Discipline Distribution (donut)
with g5:
    st.markdown('<div class="card"><div class="ctitle">WORK SUBCLASS</div><div class="csubtitle">Discipline Distribution</div>', unsafe_allow_html=True)
    subc = filtered["subclass"].value_counts().reset_index()
    subc.columns = ["subclass","count"]
    if subc.empty:
        st.write("No data")
    else:
        fig = px.pie(subc, names="subclass", values="count", hole=0.65)
        fig.update_layout(height=280, margin=dict(l=10,r=10,t=10,b=10), showlegend highlighting=True)
        st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# 6) System Variance Gap (area, met - tool)
with g6:
    st.markdown('<div class="card"><div class="ctitle">MET LAGGING TOOL</div><div class="csubtitle">System Variance Gap</div>', unsafe_allow_html=True)
    gap = filtered.copy()
    gap["x"] = range(1, len(gap)+1)
    gap["gap"] = gap["variance"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=gap["x"], y=gap["gap"], mode="lines", fill="tozeroy", name="Gap (MET-Tool)"))
    fig.update_layout(height=280, margin=dict(l=10,r=10,t=10,b=10), legend=dict(orientation="h"))
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

g7, g8, g9 = st.columns(3)

# 7) District Loading (bar)
with g7:
    st.markdown('<div class="card"><div class="ctitle">TOP ACTIVE DISTRICTS</div><div class="csubtitle">District Loading Summary</div>', unsafe_allow_html=True)
    d = filtered.dropna(subset=["district"]).copy()
    if d.empty:
        st.write("No data")
    else:
        dd = d.groupby("district")["work_order"].count().reset_index(name="count").sort_values("count", ascending=False).head(10)
        fig = px.bar(dd, x="count", y="district", orientation="h")
        fig.update_layout(height=280, margin=dict(l=10,r=10,t=10,b=10), yaxis=dict(categoryorder="total ascending"))
        st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# 8) Execution lifecycle (pie)
with g8:
    st.markdown('<div class="card"><div class="ctitle">STAGE BREAKDOWN</div><div class="csubtitle">Execution Lifecycle Stage</div>', unsafe_allow_html=True)
    s = filtered.dropna(subset=["stage"]).copy()
    if s.empty:
        st.write("No data")
    else:
        ss = s["stage"].value_counts().reset_index()
        ss.columns = ["stage","count"]
        fig = px.pie(ss, names="stage", values="count")
        fig.update_layout(height=280, margin=dict(l=10,r=10,t=10,b=10), showlegend=True)
        st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# 9) Projected vs Realized (step chart)
with g9:
    st.markdown('<div class="card"><div class="ctitle">DELIVERY STEP TREND</div><div class="csubtitle">Projected vs. Realized</div>', unsafe_allow_html=True)
    step = filtered.copy()
    step["x"] = range(1, len(step)+1)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=step["x"], y=step["met_progress"], mode="lines+markers", name="Realized"))
    fig.add_trace(go.Scatter(x=step["x"], y=step["tool_progress"], mode="lines", line=dict(dash="dash"), name="Target"))
    fig.update_layout(height=280, margin=dict(l=10,r=10,t=10,b=10), legend=dict(orientation="h"))
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ----------------------------
# Actionable Alerts: Missing MET
# ----------------------------
missing_df = filtered[~filtered["has_met"]].copy()
missing_links = missing_df[["link_code","district"]].drop_duplicates()

st.markdown(
    f"""
    <div class="alert-wrap">
      <div class="alert-head">
        <div class="alert-title">🧾 Missing MET Actual Reports</div>
        <div class="badge">{missing_links.shape[0]} LINK CODES REQUIRE URGENT SITE STATUS UPDATES</div>
      </div>
      <div class="cards-scroll">
    """,
    unsafe_allow_html=True
)

if missing_links.empty:
    st.markdown('<div class="mini">All good — no missing MET link codes.</div>', unsafe_allow_html=True)
else:
    # show first N
    for i, r in enumerate(missing_links.head(30).itertuples(index=False), start=1):
        lc = r.link_code
        dist = r.district if isinstance(r.district, str) else ""
        st.markdown(
            f"""
            <div class="mini">
              <div class="idx">#{i}</div>
              <div class="lc">{lc}</div>
              <div class="loc">{dist}</div>
              <div class="prog">Target Progress <span class="pval">—</span></div>
              <button class="btn2">ASSIGN UPDATE</button>
            </div>
            """,
            unsafe_allow_html=True
        )

st.markdown("</div></div>", unsafe_allow_html=True)

# ----------------------------
# Audit / Master Table
# ----------------------------
st.markdown('<div class="table-title"><div class="left"><div class="pill2">Audit Comparison</div><div class="pill2">ANALYZING VARIANCE BETWEEN COMMITMENTS AND SITE REALITY</div></div></div>', unsafe_allow_html=True)

# Controls
cA, cB, cC = st.columns([0.55, 0.25, 0.20])
with cA:
    q = st.text_input("Search", placeholder="Search Link Code, WO, or Supervisor...", label_visibility="collapsed")
with cB:
    view = st.radio("View", ["AUDIT", "MASTER"], horizontal=True, label_visibility="collapsed")
with cC:
    csv_bytes = filtered.to_csv(index=False).encode("utf-8")
    st.download_button("EXPORT CSV", data=csv_bytes, file_name="dawiyat_dashboard_export.csv", mime="text/csv")

table_df = filtered.copy()
if q:
    qq = q.lower().strip()
    table_df = table_df[
        table_df["link_code"].astype(str).str.lower().str.contains(qq)
        | table_df["work_order"].astype(str).str.lower().str.contains(qq)
        | table_df["supervisor"].astype(str).str.lower().str.contains(qq)
    ]

if view == "AUDIT":
    # show key reconciliation fields
    audit_cols = ["link_code","work_order","district","subclass","tool_progress","met_progress","variance","stage","supervisor"]
    audit = table_df[audit_cols].copy()

    def variance_label(x):
        if pd.isna(x):
            return "—"
        if x >= 0:
            return f"↑ {abs(x):.1f}%"
        return f"↓ {abs(x):.1f}%"

    audit["variance_label"] = audit["variance"].apply(variance_label)
    audit["met_label"] = audit["met_progress"].apply(lambda x: "MISSING MET" if pd.isna(x) else f"{x:.1f}%")
    audit["tool_label"] = audit["tool_progress"].apply(lambda x: "" if pd.isna(x) else f"{x:.1f}%")

    show = audit.rename(columns={
        "link_code":"IDENTITIES",
        "district":"REGIONAL INFO",
        "tool_label":"TARGET %",
        "met_label":"ACTUAL %",
        "variance_label":"VARIANCE",
    })[["IDENTITIES","work_order","REGIONAL INFO","TARGET %","ACTUAL %","VARIANCE"]]

    st.dataframe(show, use_container_width=True, height=430)
else:
    # Master view
    master_cols = ["link_code","work_order","district","subclass","stage","tool_progress","met_progress","record_status"]
    show = table_df[master_cols].copy()
    show = show.rename(columns={
        "link_code":"IDENTITIES",
        "work_order":"WORK ORDER",
        "district":"DISTRICT",
        "subclass":"CATEGORY",
        "stage":"STAGE",
        "tool_progress":"TARGET %",
        "met_progress":"ACTUAL %",
        "record_status":"RECORD STATUS"
    })
    st.dataframe(show, use_container_width=True, height=430)

# ----------------------------
# Top right actions row (help + refresh)
# ----------------------------
# Put below to not interfere with layout; use columns for clickable actions
a1, a2, a3 = st.columns([0.75, 0.12, 0.13])
with a2:
    if st.button("UPDATE SOURCE DATA"):
        st.rerun()
with a3:
    if st.button(" ? "):
        st.session_state.show_guide = not st.session_state.show_guide

if st.session_state.show_guide:
    st.markdown("---")
    st.markdown(GUIDE_MD)
