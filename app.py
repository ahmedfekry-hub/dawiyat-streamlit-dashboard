
# ===============================
# Dawaiyat Project Tracker Dashboard
# Full Fixed Script (Python 3.11 compatible)
# ===============================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(
    page_title="Dawiyat Project Tracker Dashboard",
    page_icon="📊",
    layout="wide"
)

# -------------------------------
# Helpers
# -------------------------------
def normalize_columns(df):
    df.columns = [c.strip() for c in df.columns]
    return df

def safe_pct(x):
    try:
        return float(x)
    except:
        return 0.0

# -------------------------------
# Sidebar
# -------------------------------
st.sidebar.title("⚙ Settings")
dark_mode = st.sidebar.toggle("Dark Mode", value=False)

# -------------------------------
# Header
# -------------------------------
c1, c2 = st.columns([6, 2])
with c1:
    st.title("📊 Dawaiyat Project Tracker Dashboard")
    st.caption("Contractor Commitment vs Site Actual | Executive Decision View")
with c2:
    with st.popover("❓ Dashboard Guide"):
        st.markdown("""
### KPI Summary
- **Total Link Codes**: Count of all unique projects
- **Dawaiyat Avg Target**: Planned progress
- **MET Avg Actual**: Site-reported progress
- **Missing MET Reports**: No site data found
- **Critical Lags (>15%)**: High-risk delays

### Comparison Logic
- Civil → *Civil Completion %*
- Fiber → *Fiber Completion %*
""")

# -------------------------------
# Upload Section
# -------------------------------
st.subheader("Upload Source Data")

mode = st.radio(
    "Choose upload mode",
    ["Single Excel (contains both sheets)", "Two files (Tool + MET)"]
)

tool_df = met_df = None

if mode.startswith("Single"):
    f = st.file_uploader("Upload Excel (.xlsx)", type=["xlsx"])
    if f:
        xl = pd.ExcelFile(f)
        tool_df = normalize_columns(pd.read_excel(xl, xl.sheet_names[0]))
        met_df = normalize_columns(pd.read_excel(xl, xl.sheet_names[1]))
else:
    f1 = st.file_uploader("Upload Dawaiyat Service Tool", type=["xlsx"], key="tool")
    f2 = st.file_uploader("Upload MET Actual Progress", type=["xlsx"], key="met")
    if f1 and f2:
        tool_df = normalize_columns(pd.read_excel(f1))
        met_df = normalize_columns(pd.read_excel(f2))

if tool_df is None or met_df is None:
    st.info("⬆ Upload your source data to start.")
    st.stop()

# -------------------------------
# Core Mapping
# -------------------------------
tool_df["Percentage of Completion"] = tool_df["Percentage of Completion"].apply(safe_pct)
met_df["Civil Completion %"] = met_df["Civil Completion %"].apply(safe_pct)
met_df["Fiber Completion %"] = met_df["Fiber Completion %"].apply(safe_pct)

merged = tool_df.merge(
    met_df,
    on=["Link Code", "Work Order", "District", "Subclass"],
    how="left"
)

def pick_actual(row):
    if row["Subclass"].strip().lower() == "civil":
        return row.get("Civil Completion %", 0)
    else:
        return row.get("Fiber Completion %", 0)

merged["Actual %"] = merged.apply(pick_actual, axis=1)
merged["Variance %"] = merged["Actual %"] - merged["Percentage of Completion"]
merged["Missing MET"] = merged["Actual %"].isna() | (merged["Actual %"] == 0)

# -------------------------------
# KPI Cards
# -------------------------------
k1, k2, k3, k4, k5 = st.columns(5)

k1.metric("Total Link Codes", merged["Link Code"].nunique())
k2.metric("Dawaiyat Avg Target", f"{merged['Percentage of Completion'].mean():.1f}%")
k3.metric("MET Avg Actual", f"{merged['Actual %'].mean():.1f}%")
k4.metric("Missing MET Reports", merged["Missing MET"].sum())
k5.metric("Critical Lags (>15%)", (merged["Variance %"] < -15).sum())

st.divider()

# -------------------------------
# Charts
# -------------------------------
st.subheader("Strategic Performance Grids")

c1, c2, c3 = st.columns(3)

with c1:
    fig = px.pie(
        merged,
        names="District",
        title="Regional Coverage"
    )
    fig.update_layout(height=280, margin=dict(l=10, r=10, t=40, b=10), showlegend=True)
    st.plotly_chart(fig, use_container_width=True)

with c2:
    trend = merged.groupby("Link Code")[["Percentage of Completion", "Actual %"]].mean().reset_index()
    fig = px.line(
        trend,
        x="Link Code",
        y=["Percentage of Completion", "Actual %"],
        title="Dawaiyat vs MET Trend"
    )
    fig.update_layout(height=280, margin=dict(l=10, r=10, t=40, b=10), showlegend=True)
    st.plotly_chart(fig, use_container_width=True)

with c3:
    status = merged["Missing MET"].map({True: "Missing Site Report", False: "Reported"})
    fig = px.pie(
        names=status.value_counts().index,
        values=status.value_counts().values,
        title="Operational Status"
    )
    fig.update_layout(height=280, margin=dict(l=10, r=10, t=40, b=10), showlegend=True)
    st.plotly_chart(fig, use_container_width=True)

# -------------------------------
# Tables
# -------------------------------
st.subheader("Audit Comparison")
st.dataframe(
    merged[[
        "Link Code", "Work Order", "District", "Subclass",
        "Percentage of Completion", "Actual %", "Variance %"
    ]],
    use_container_width=True
)

st.subheader("Master Data Inventory")
st.dataframe(merged, use_container_width=True)
