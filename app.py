# Dawiyat Project Tracker Dashboard (Streamlit)
# - Supports: Single Excel (both sheets) OR Two separate files (Tool + MET)
# - Robust column detection (handles extra spaces / different names)
# - Civil vs Fiber comparison:
#   Tool: "Percentage of Completion"
#   MET:  Civil -> "Civil Completion  %" ; Fiber -> "Fiber Completion  %"
# - Exports: merged MASTER + AUDIT tables to CSV

import re
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(
    page_title="Dawiyat Project Tracker Dashboard",
    page_icon="📊",
    layout="wide",
)

# -------------------------------
# Helpers
# -------------------------------
def _norm_key(x) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(x).strip().lower())

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df

def find_column(df: pd.DataFrame, candidates, required=True, df_name="sheet"):
    col_map = {_norm_key(c): c for c in df.columns}
    for cand in candidates:
        k = _norm_key(cand)
        if k in col_map:
            return col_map[k]
    if required:
        raise KeyError(
            f"Missing required column in {df_name}. Tried: {candidates}. "
            f"Available: {list(df.columns)}"
        )
    return None

def safe_pct(v):
    """Parse percent-like values (e.g., 100.42%, '25.10', NaN) into float."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return np.nan
    s = str(v).strip()
    if s == "" or s.lower() in {"nan", "none"}:
        return np.nan
    s = s.replace(",", "")
    # remove trailing percent sign
    if s.endswith("%"):
        s = s[:-1].strip()
    # keep only numeric / sign / dot
    s = re.sub(r"[^0-9.+-]", "", s)
    try:
        return float(s)
    except Exception:
        return np.nan

def to_bytes_csv(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8-sig")

# -------------------------------
# Sidebar
# -------------------------------
st.sidebar.title("⚙ Settings")
dark_mode = st.sidebar.toggle("Dark Mode", value=False)

# Light theme / Dark theme quick tweak
if dark_mode:
    st.markdown(
        """
        <style>
        .stApp { background-color: #0e1117; color: #e6edf3; }
        </style>
        """,
        unsafe_allow_html=True,
    )

# -------------------------------
# Header
# -------------------------------
left, right = st.columns([6, 2])
with left:
    st.title("📊 Dawiyat Project Tracker Dashboard")
    st.caption("Contractor Commitment vs Site Actual | Executive Decision View")
with right:
    with st.popover("❓ Dashboard Guide"):
        st.markdown(
            """
### 1) KPI Summary (Top Cards)
- **Total Link Codes**: Total unique link codes in *Dawaiyat Service Tool*  
- **Dawaiyat Avg Target**: Average *Percentage of Completion* from the tool  
- **MET Avg Actual**: Average site-reported progress from MET  
- **Missing MET Reports**: Link codes with no MET record for at least one WO  
- **Critical Lags (>15%)**: WOs where MET is **15% or more** behind the tool

### 2) Civil vs Fiber Logic (Important)
Each **Link Code** may have **two Work Orders**:
- If *Subclass = Civil*: compare Tool **Percentage of Completion** vs MET **Civil Completion %**
- If *Subclass = Fiber*: compare Tool **Percentage of Completion** vs MET **Fiber Completion %**

### 3) Audit Comparison Table
- **Variance %** = Actual % − Tool %  
- Negative variance = behind / lagging  
- **Missing MET** = no MET row found for that Work Order

Tip: Use filters to isolate Region, District, Supervisor, Stage, etc.
            """
        )

# -------------------------------
# Upload Section
# -------------------------------
st.subheader("Upload Source Data")

mode = st.radio(
    "Choose upload mode",
    ["Single Excel (contains both sheets)", "Two files (Tool + MET)"],
    horizontal=True
)

tool_df = met_df = None

if mode.startswith("Single"):
    f = st.file_uploader("Upload Excel (.xlsx) that contains BOTH sheets", type=["xlsx"])
    if f:
        xl = pd.ExcelFile(f)
        # Try by common sheet names first, else fallback to first/second sheet
        sheet_names = xl.sheet_names

        def pick_sheet(name_candidates, default_index):
            for nm in sheet_names:
                if _norm_key(nm) in {_norm_key(c) for c in name_candidates}:
                    return nm
            return sheet_names[default_index]

        tool_sheet = pick_sheet(["Dawaiyat Service Tool", "Service Tool", "Tool"], 0)
        met_sheet  = pick_sheet(["MET Actual progress", "MET Actual Progress", "MET"], 1 if len(sheet_names)>1 else 0)

        tool_df = normalize_columns(pd.read_excel(xl, tool_sheet))
        met_df  = normalize_columns(pd.read_excel(xl, met_sheet))
        st.success(f"Loaded sheets: Tool='{tool_sheet}' | MET='{met_sheet}'")
else:
    f1 = st.file_uploader("Upload Dawaiyat Service Tool (.xlsx)", type=["xlsx"], key="tool")
    f2 = st.file_uploader("Upload MET Actual progress (.xlsx)", type=["xlsx"], key="met")
    if f1 and f2:
        tool_df = normalize_columns(pd.read_excel(f1))
        met_df  = normalize_columns(pd.read_excel(f2))

if tool_df is None or met_df is None:
    st.info("⬆ Upload your source data to start.")
    st.stop()

# -------------------------------
# Standardize required columns
# -------------------------------
# Tool columns
tool_link = find_column(tool_df, ["Link Code", "LinkCode", "LINK CODE"], df_name="Dawaiyat Service Tool")
tool_wo   = find_column(tool_df, ["Work Order", "WorkOrder", "WO"], df_name="Dawaiyat Service Tool")
tool_dist = find_column(tool_df, ["District", "Dist."], required=False, df_name="Dawaiyat Service Tool")
tool_sub  = find_column(tool_df, ["Subclass", "Sub class", "SubClass", "Discipline"], required=False, df_name="Dawaiyat Service Tool")
tool_pct  = find_column(tool_df, ["Percentage of Completion", "Completion %", "% Completion", "Progress %"], df_name="Dawaiyat Service Tool")

# MET columns
met_link = find_column(met_df, ["Link Code", "LinkCode", "LINK CODE"], df_name="MET Actual progress")
met_wo   = find_column(met_df, ["Work Order", "WorkOrder", "WO"], df_name="MET Actual progress")
met_dist = find_column(met_df, ["District", "Dist."], required=False, df_name="MET Actual progress")
met_sub  = find_column(met_df, ["Subclass", "Sub class", "SubClass", "Discipline"], required=False, df_name="MET Actual progress")

met_civil = find_column(met_df, ["Civil Completion %", "Civil Completion  %", "Civil Completion", "Civil %"], df_name="MET Actual progress")
met_fiber = find_column(met_df, ["Fiber Completion %", "Fiber Completion  %", "Fiber Completion", "Fiber %"], df_name="MET Actual progress")

# Rename to standard headers
tool_df = tool_df.rename(columns={
    tool_link: "Link Code",
    tool_wo: "Work Order",
    **({tool_dist: "District"} if tool_dist else {}),
    **({tool_sub: "Subclass"} if tool_sub else {}),
    tool_pct: "Tool Raw %",
})
met_df = met_df.rename(columns={
    met_link: "Link Code",
    met_wo: "Work Order",
    **({met_dist: "District"} if met_dist else {}),
    **({met_sub: "Subclass"} if met_sub else {}),
    met_civil: "MET Civil Raw %",
    met_fiber: "MET Fiber Raw %",
})

# Ensure key columns exist (District/Subclass may be missing in some files)
if "District" not in tool_df.columns:
    tool_df["District"] = ""
if "Subclass" not in tool_df.columns:
    tool_df["Subclass"] = ""

if "District" not in met_df.columns:
    met_df["District"] = ""
if "Subclass" not in met_df.columns:
    met_df["Subclass"] = ""

# Parse numeric %
tool_df["Tool %"] = tool_df["Tool Raw %"].apply(safe_pct)
met_df["MET Civil %"] = met_df["MET Civil Raw %"].apply(safe_pct)
met_df["MET Fiber %"] = met_df["MET Fiber Raw %"].apply(safe_pct)

# -------------------------------
# Merge strategy (robust)
# -------------------------------
# IMPORTANT: many "Link Codes" missing in MET -> do NOT merge on District/Subclass (too strict)
merged = tool_df.merge(
    met_df[["Link Code", "Work Order", "MET Civil %", "MET Fiber %"]],
    on=["Link Code", "Work Order"],
    how="left",
)

def pick_actual(row):
    sub = str(row.get("Subclass", "")).strip().lower()
    if sub == "civil":
        return row.get("MET Civil %", np.nan)
    if sub == "fiber":
        return row.get("MET Fiber %", np.nan)
    # If Subclass empty/unknown: pick any available MET value (max of the two)
    v1 = row.get("MET Civil %", np.nan)
    v2 = row.get("MET Fiber %", np.nan)
    if pd.isna(v1) and pd.isna(v2):
        return np.nan
    return np.nanmax([v1, v2])

merged["Actual %"] = merged.apply(pick_actual, axis=1)
merged["Variance %"] = merged["Actual %"] - merged["Tool %"]
merged["Missing MET"] = merged["Actual %"].isna()

# -------------------------------
# Top Filters (optional columns)
# -------------------------------
st.markdown("---")
fcol = st.columns([1,1,1,1,1,1,1,1])
with fcol[0]:
    flt_link = st.selectbox("LINK CODE", ["All"] + sorted(merged["Link Code"].dropna().astype(str).unique().tolist()))
with fcol[1]:
    flt_dist = st.selectbox("DISTRICT", ["All"] + sorted(merged["District"].dropna().astype(str).unique().tolist()))
with fcol[2]:
    flt_sub  = st.selectbox("SUBCLASS", ["All"] + sorted(merged["Subclass"].dropna().astype(str).unique().tolist()))
with fcol[3]:
    lag_only = st.checkbox("Show only Critical Lags (>15%)", value=False)
with fcol[4]:
    miss_only = st.checkbox("Show only Missing MET", value=False)
with fcol[5]:
    search = st.text_input("Search (Link/WO)", value="")
with fcol[6]:
    st.download_button("⬇ Export AUDIT CSV", data=to_bytes_csv(merged), file_name="audit_comparison.csv", mime="text/csv")
with fcol[7]:
    st.download_button("⬇ Export MASTER CSV", data=to_bytes_csv(merged), file_name="master_data_inventory.csv", mime="text/csv")

view = merged.copy()
if flt_link != "All":
    view = view[view["Link Code"].astype(str) == flt_link]
if flt_dist != "All":
    view = view[view["District"].astype(str) == flt_dist]
if flt_sub != "All":
    view = view[view["Subclass"].astype(str) == flt_sub]
if lag_only:
    view = view[view["Variance %"] < -15]
if miss_only:
    view = view[view["Missing MET"]]
if search.strip():
    s = search.strip().lower()
    view = view[
        view["Link Code"].astype(str).str.lower().str.contains(s, na=False) |
        view["Work Order"].astype(str).str.lower().str.contains(s, na=False)
    ]

# -------------------------------
# KPI Cards
# -------------------------------
k1, k2, k3, k4, k5 = st.columns(5)

total_links = int(view["Link Code"].nunique())
avg_tool = float(view["Tool %"].mean()) if len(view) else 0.0
avg_actual = float(view["Actual %"].mean()) if len(view) else 0.0
missing_count = int(view["Missing MET"].sum())
critical_lags = int((view["Variance %"] < -15).sum())

k1.metric("Total Link Codes", total_links)
k2.metric("Dawaiyat Avg Target", f"{avg_tool:.1f}%")
k3.metric("MET Avg Actual", f"{avg_actual:.1f}%")
k4.metric("Missing MET Reports", missing_count)
k5.metric("Critical Lags (>15%)", critical_lags)

st.divider()

# -------------------------------
# Charts (simple + robust)
# -------------------------------
st.subheader("Strategic Performance Grids")

c1, c2, c3 = st.columns(3)

with c1:
    dist_counts = view["District"].replace("", "UNKNOWN").value_counts().reset_index()
    dist_counts.columns = ["District", "Count"]
    fig = px.pie(dist_counts, names="District", values="Count", title="Regional Coverage (by District)")
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=50, b=10), showlegend=True)
    st.plotly_chart(fig, use_container_width=True)

with c2:
    # Trend by link code: average of tool vs actual
    trend = (
        view.groupby("Link Code", as_index=False)[["Tool %", "Actual %"]]
        .mean(numeric_only=True)
        .sort_values("Link Code")
    )
    fig = px.line(trend, x="Link Code", y=["Tool %", "Actual %"], title="Dawaiyat vs MET Trend")
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=50, b=10), showlegend=True)
    st.plotly_chart(fig, use_container_width=True)

with c3:
    # Operational status: Completed/ In progress not available -> show reported vs missing
    status_counts = view["Missing MET"].map({True: "Missing Site Report", False: "Reported"}).value_counts()
    fig = px.pie(
        names=status_counts.index,
        values=status_counts.values,
        title="Operational Status"
    )
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=50, b=10), showlegend=True)
    st.plotly_chart(fig, use_container_width=True)

# -------------------------------
# Actionable Alert: Missing MET
# -------------------------------
st.subheader("Missing MET Actual Reports")
miss = view[view["Missing MET"]].copy()
st.caption(f"{len(miss)} Work Orders require urgent MET updates")
if len(miss) == 0:
    st.success("No missing MET records in the current filter.")
else:
    st.dataframe(
        miss[["Link Code", "Work Order", "District", "Subclass", "Tool %"]],
        use_container_width=True,
        hide_index=True
    )

# -------------------------------
# Audit & Master Tables
# -------------------------------
st.subheader("Audit Comparison")
audit_cols = ["Link Code", "Work Order", "District", "Subclass", "Tool %", "Actual %", "Variance %", "Missing MET"]
st.dataframe(view[audit_cols], use_container_width=True, hide_index=True)

st.subheader("Master Data Inventory")
st.dataframe(view, use_container_width=True, hide_index=True)
