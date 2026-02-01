import re
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# =============================
# PAGE CONFIG
# =============================
st.set_page_config(
    page_title="Project Management Dashboard",
    layout="wide"
)

# =============================
# THEME TOGGLE
# =============================
def apply_theme(dark=True):
    if dark:
        css = """
        <style>
        .stApp { background-color: #0e1117; color: #fafafa; }
        </style>
        """
    else:
        css = """
        <style>
        .stApp { background-color: #ffffff; color: #111111; }
        </style>
        """
    st.markdown(css, unsafe_allow_html=True)

with st.sidebar:
    st.header("⚙️ Settings")
    dark_mode = st.toggle("Dark Mode", value=True)

apply_theme(dark_mode)

# =============================
# HELPERS
# =============================
def normalize(col):
    return re.sub(r"[^a-z0-9]+", " ", str(col).lower()).strip()

def find_col(df, keywords, required=False):
    for col in df.columns:
        col_n = normalize(col)
        for kw in keywords:
            if normalize(kw) in col_n:
                return col
    if required:
        raise ValueError(f"Missing column with keywords: {keywords}")
    return None

def to_pct(v):
    if pd.isna(v):
        return np.nan
    if isinstance(v, str):
        v = v.replace("%", "").strip()
    try:
        v = float(v)
        return v * 100 if v <= 1 else v
    except:
        return np.nan

# =============================
# LOAD DATA
# =============================
@st.cache_data
def load_excel(file):
    xls = pd.ExcelFile(file)
    tool = pd.read_excel(xls, "Dawaiyat Service Tool")
    met  = pd.read_excel(xls, "MET Actual progress")
    return tool, met

# =============================
# HEADER
# =============================
st.title("📊 Project Management Dashboard")
st.caption("Commitment vs Actual | Executive Decision View")

uploaded = st.file_uploader("Upload Excel File", type=["xlsx"])

if not uploaded:
    st.info("⬆️ Please upload the Excel file to start")
    st.stop()

tool, met = load_excel(uploaded)

# =============================
# COLUMN MAPPING
# =============================
tool_wo = find_col(tool, ["link code", "wo"], True)
met_wo  = find_col(met,  ["link code", "wo"], True)

tool_prog = find_col(tool, ["overall progress", "completion"])
met_prog  = find_col(met,  ["actual progress", "overall progress"])

tool_target = find_col(tool, ["targeted completion", "forecast"])
met_target  = find_col(met,  ["targeted completion", "forecast"])

# =============================
# PREPARE DATA
# =============================
tool["_WO"] = tool[tool_wo].astype(str)
met["_WO"]  = met[met_wo].astype(str)

if tool_prog:
    tool["_Progress_Commitment"] = tool[tool_prog].apply(to_pct)
else:
    tool["_Progress_Commitment"] = np.nan

if met_prog:
    met["_Progress_Actual"] = met[met_prog].apply(to_pct)
else:
    met["_Progress_Actual"] = np.nan

tool["_Target_Commitment"] = pd.to_datetime(tool[tool_target], errors="coerce") if tool_target else pd.NaT
met["_Target_Actual"]      = pd.to_datetime(met[met_target], errors="coerce") if met_target else pd.NaT

df = pd.merge(
    tool, met,
    on="_WO",
    how="outer",
    suffixes=("_tool", "_met"),
    indicator=True
)

df["Progress Gap (%)"] = df["_Progress_Actual"] - df["_Progress_Commitment"]

today = pd.Timestamp.today()
df["Overdue"] = (df["_Target_Commitment"] < today) & (df["_Progress_Actual"] < 100)

# =============================
# FILTERS
# =============================
st.subheader("🔎 Global Filters")

c1, c2, c3, c4 = st.columns(4)

with c1:
    wo_sel = st.selectbox("Link Code (WO)", ["All"] + sorted(df["_WO"].dropna().unique()))
with c2:
    region_col = find_col(df, ["region"])
    region_sel = st.selectbox("Region", ["All"] + sorted(df[region_col].dropna().unique()) if region_col else ["All"])
with c3:
    stage_col = find_col(df, ["stage"])
    stage_sel = st.selectbox("Stage", ["All"] + sorted(df[stage_col].dropna().unique()) if stage_col else ["All"])
with c4:
    dist_col = find_col(df, ["district"])
    dist_sel = st.selectbox("District", ["All"] + sorted(df[dist_col].dropna().unique()) if dist_col else ["All"])

f = df.copy()
if wo_sel != "All":
    f = f[f["_WO"] == wo_sel]
if region_col and region_sel != "All":
    f = f[f[region_col] == region_sel]
if stage_col and stage_sel != "All":
    f = f[f[stage_col] == stage_sel]
if dist_col and dist_sel != "All":
    f = f[f[dist_col] == dist_sel]

# =============================
# KPI CARDS
# =============================
st.subheader("📌 Executive KPIs")

k1, k2, k3, k4 = st.columns(4)
k1.metric("Total WOs", f["_WO"].nunique())
k2.metric("Avg Commitment Progress (%)", float(np.nan_to_num(f["_Progress_Commitment"].mean(), nan=0.0)).__round__(1))
k3.metric("Avg Actual Progress (%)", float(np.nan_to_num(f["_Progress_Actual"].mean(), nan=0.0)).__round__(1))
k4.metric("Avg Progress Gap (%)", float(np.nan_to_num(f["Progress Gap (%)"].mean(), nan=0.0)).__round__(1))

# =============================
# 3x3 DASHBOARD
# =============================
st.subheader("📈 Performance Overview")

r1 = st.columns(3)
r2 = st.columns(3)
r3 = st.columns(3)

with r1[0]:
    avg_c = float(np.nan_to_num(f["_Progress_Commitment"].mean(), nan=0.0))
    fig = px.pie(
        values=[avg_c, max(0, 100 - avg_c)],
        names=["Completed", "Remaining"],
        hole=0.55,
        title="Commitment Progress"
    )
    st.plotly_chart(fig, use_container_width=True)

with r1[1]:
    avg_a = float(np.nan_to_num(f["_Progress_Actual"].mean(), nan=0.0))
    fig = px.pie(
        values=[avg_a, max(0, 100 - avg_a)],
        names=["Completed", "Remaining"],
        hole=0.55,
        title="Actual Progress"
    )
    st.plotly_chart(fig, use_container_width=True)

with r1[2]:
    if f["Progress Gap (%)"].dropna().shape[0] > 0:
        fig = px.histogram(f, x="Progress Gap (%)", nbins=20, title="Progress Gap Distribution")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No gap values available.")

with r2[0]:
    # Trend by commitment target date (if exists)
    if "_Target_Commitment" in f.columns and f["_Target_Commitment"].notna().any():
        ft = f.sort_values("_Target_Commitment")
        fig = px.line(ft, x="_Target_Commitment", y="_Progress_Actual", title="Actual Progress vs Commitment Target Date", markers=True)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No commitment target dates found for trend chart.")

with r2[1]:
    top = f[["_WO", "Progress Gap (%)"]].dropna().head(50)
    if top.shape[0] > 0:
        fig = px.bar(top, x="_WO", y="Progress Gap (%)", title="Gap by WO (first 50)")
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No gap data for WO bar chart.")

with r2[2]:
    overdue = int(f["Overdue"].sum()) if "Overdue" in f.columns else 0
    fig = px.pie(
        values=[overdue, max(0, len(f) - overdue)],
        names=["Overdue", "On Track"],
        hole=0.6,
        title="Schedule Status (Commitment)"
    )
    st.plotly_chart(fig, use_container_width=True)

with r3[0]:
    st.write("📋 Missing in MET (Actual)")
    miss = f[f["_merge"] == "left_only"][["_WO"]].copy() if "_merge" in f.columns else pd.DataFrame()
    st.dataframe(miss, use_container_width=True, height=220)

with r3[1]:
    st.write("📋 Missing in Commitment")
    miss2 = f[f["_merge"] == "right_only"][["_WO"]].copy() if "_merge" in f.columns else pd.DataFrame()
    st.dataframe(miss2, use_container_width=True, height=220)

with r3[2]:
    st.write("📋 Detailed Data (filtered)")
    st.dataframe(f, use_container_width=True, height=220)
