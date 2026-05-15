# ── Imports ───────────────────────────────────────────────────────
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

COLOR = {
    'navy': '#1F3A7A',
    'violet': '#534AB7',
    'green': '#1D9E75',
    'orange': '#D85A30',
    'blue': '#378ADD',
    'amber': '#BA7517',
    'danger': '#E24B4A',
    'warning': '#EF9F27',
    'muted': '#6b7280',
}

# ── Page Configuration ────────────────────────────────────────────
st.set_page_config(
    page_title="UAC Care Pipeline Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────
st.markdown("""
    <style>
        .main { padding: 1rem 2rem; }
        .block-container { padding-top: 1.25rem; }
        .metric-card {
            background: #f8f9fa;
            border-radius: 10px;
            padding: 16px 20px;
            border-left: 4px solid #1F3A7A;
        }
        .section-caption {
            color: #6b7280;
            margin-top: -0.45rem;
            margin-bottom: 0.75rem;
            font-size: 0.92rem;
        }
        .alert-red {
            background: #fff0f0;
            border-left: 4px solid #E24B4A;
            padding: 10px 16px;
            border-radius: 6px;
            margin: 6px 0;
        }
        .alert-green {
            background: #f0fff4;
            border-left: 4px solid #1D9E75;
            padding: 10px 16px;
            border-radius: 6px;
            margin: 6px 0;
        }
        .alert-orange {
            background: #fffbf0;
            border-left: 4px solid #EF9F27;
            padding: 10px 16px;
            border-radius: 6px;
            margin: 6px 0;
        }
        h1 { color: #1F3A7A; }
        h2 { color: #2E4A8E; }
    </style>
""", unsafe_allow_html=True)


def section_header(title: str, caption: str = ""):
    st.markdown(f"### {title}")
    if caption:
        st.markdown(
            f"<div class='section-caption'>{caption}</div>",
            unsafe_allow_html=True
        )


def to_csv_bytes(dataframe: pd.DataFrame) -> bytes:
    return dataframe.to_csv(index=False).encode('utf-8')


def fig_to_png_bytes(fig: go.Figure):
    try:
        return pio.to_image(fig, format='png', scale=2)
    except Exception:
        return None


def fig_to_html_bytes(fig: go.Figure) -> bytes:
    return fig.to_html(include_plotlyjs='cdn').encode('utf-8')

# ── Section 2: Load & Prepare Data ───────────────────────────────

@st.cache_data
def load_data():
    df = pd.read_csv('data/processed/uac_engineered.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)

    # Ensure helper columns exist
    df['Year']    = df['Date'].dt.year
    df['Month']   = df['Date'].dt.month
    df['YearMon'] = df['Date'].dt.to_period('M').astype(str)
    df['Weekday'] = df['Date'].dt.day_name()

    # Re-compute KPIs in case they're missing
    df['Transfer_Efficiency'] = (
        df['CBP_Transferred'] / df['CBP_Custody'].replace(0, np.nan)
    ).round(4)
    df['Discharge_Effectiveness'] = (
        df['HHS_Discharged'] / df['HHS_Care'].replace(0, np.nan)
    ).round(4)
    df['Pipeline_Throughput'] = (
        (df['CBP_Transferred'] + df['HHS_Discharged']) /
        (df['CBP_Apprehended'] + df['HHS_Care']).replace(0, np.nan)
    ).round(4)
    df['Backlog'] = (
        df['CBP_Apprehended'] - df['CBP_Transferred'] +
        df['HHS_Care']        - df['HHS_Discharged']
    )
    df['Outcome_Stability'] = (
        df['Discharge_Effectiveness'].rolling(30).std()
    ).round(4)

    # Bottleneck flags
    df['Bottleneck_Backlog']   = df['HHS_Care']                > 8000
    df['Bottleneck_Transfer']  = df['Transfer_Efficiency']     < 0.40
    df['Bottleneck_Discharge'] = df['Discharge_Effectiveness'] < 0.015
    df['Any_Bottleneck']       = (
        df['Bottleneck_Backlog'] |
        df['Bottleneck_Transfer'] |
        df['Bottleneck_Discharge']
    )

    return df


@st.cache_data(show_spinner=False)
def prepare_filtered_analytics(dataframe, te_threshold, de_threshold, bl_threshold):
    work = dataframe.copy()

    work['Bottleneck_Transfer'] = work['Transfer_Efficiency'] < te_threshold
    work['Bottleneck_Discharge'] = work['Discharge_Effectiveness'] < de_threshold
    work['Bottleneck_Backlog'] = work['HHS_Care'] > bl_threshold
    work['Any_Bottleneck'] = (
        work['Bottleneck_Transfer'] |
        work['Bottleneck_Discharge'] |
        work['Bottleneck_Backlog']
    )

    work['TE_rolling'] = work['Transfer_Efficiency'].rolling(30, center=True).mean()
    work['DE_rolling'] = work['Discharge_Effectiveness'].rolling(30, center=True).mean()
    work['PT_rolling'] = work['Pipeline_Throughput'].rolling(30, center=True).mean()

    bottleneck_data = work[[
        'Date',
        'Bottleneck_Transfer',
        'Bottleneck_Discharge',
        'Bottleneck_Backlog'
    ]].copy()
    bottleneck_data['Transfer'] = bottleneck_data['Bottleneck_Transfer'].astype(int)
    bottleneck_data['Discharge'] = bottleneck_data['Bottleneck_Discharge'].astype(int)
    bottleneck_data['Backlog'] = bottleneck_data['Bottleneck_Backlog'].astype(int)

    bottleneck_monthly = bottleneck_data.set_index('Date').resample('MS').sum()
    monthly_totals = bottleneck_monthly.sum(axis=1).replace(0, np.nan)
    bottleneck_monthly_pct = (
        bottleneck_monthly
        .div(monthly_totals, axis=0)
        .fillna(0)
        * 100
    )

    work['Stagnation_Flag'] = (
        (work['Transfer_Efficiency'] < te_threshold) &
        (work['Discharge_Effectiveness'] < de_threshold) &
        (work['HHS_Care'] > bl_threshold)
    )

    stagnation_runs = []
    current_run_start = None
    current_run_length = 0

    for idx, row in work.iterrows():
        if row['Stagnation_Flag']:
            if current_run_start is None:
                current_run_start = row['Date']
                current_run_length = 1
            else:
                current_run_length += 1
        else:
            if current_run_start is not None and current_run_length >= 3:
                stagnation_runs.append({
                    'Start Date': current_run_start.strftime('%Y-%m-%d'),
                    'End Date': work.iloc[idx - 1]['Date'].strftime('%Y-%m-%d'),
                    'Duration (days)': current_run_length,
                    'Avg HHS Care': work.iloc[idx - current_run_length:idx]['HHS_Care'].mean(),
                    'Avg Transfer Eff (%)': work.iloc[idx - current_run_length:idx]['Transfer_Efficiency'].mean() * 100,
                    'Avg Discharge Eff (%)': work.iloc[idx - current_run_length:idx]['Discharge_Effectiveness'].mean() * 100,
                })
            current_run_start = None
            current_run_length = 0

    if current_run_start is not None and current_run_length >= 3:
        stagnation_runs.append({
            'Start Date': current_run_start.strftime('%Y-%m-%d'),
            'End Date': work.iloc[-1]['Date'].strftime('%Y-%m-%d'),
            'Duration (days)': current_run_length,
            'Avg HHS Care': work.iloc[-current_run_length:]['HHS_Care'].mean(),
            'Avg Transfer Eff (%)': work.iloc[-current_run_length:]['Transfer_Efficiency'].mean() * 100,
            'Avg Discharge Eff (%)': work.iloc[-current_run_length:]['Discharge_Effectiveness'].mean() * 100,
        })

    return work, bottleneck_monthly_pct, pd.DataFrame(stagnation_runs)

# Load data
df = load_data()

# ── Page Title ────────────────────────────────────────────────────
st.title("📊 UAC Care Pipeline Analytics")
st.markdown(
    "**Care Transition Efficiency & Placement Outcome Dashboard** "
    "— HHS Unaccompanied Alien Children Program | Jan 2023 – Dec 2025"
)
st.divider()

# ── Section 3: Sidebar ────────────────────────────────────────────

st.sidebar.image(
    "https://upload.wikimedia.org/wikipedia/commons/thumb/f/f6/"
    "Seal_of_the_United_States_Department_of_Health_%26_Human_"
    "Services.svg/240px-Seal_of_the_United_States_Department_of_"
    "Health_%26_Human_Services.svg.png",
    width=80
)
st.sidebar.title("UAC Pipeline Analytics")
st.sidebar.markdown("**HHS Office of Refugee Resettlement**")
st.sidebar.divider()

# ── Date Range Filter ─────────────────────────────────────────────
st.sidebar.subheader("📅 Date Range")
min_date = df['Date'].min().date()
max_date = df['Date'].max().date()

start_date, end_date = st.sidebar.date_input(
    "Select date range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

# ── Year Filter ───────────────────────────────────────────────────
st.sidebar.subheader("📆 Filter by Year")
years_available = sorted(df['Year'].unique().tolist())
selected_years = st.sidebar.multiselect(
    "Select years",
    options=years_available,
    default=years_available
)

# ── KPI Toggles ───────────────────────────────────────────────────
st.sidebar.subheader("📊 KPI Toggles")
show_te  = st.sidebar.checkbox("Transfer Efficiency Ratio",     value=True)
show_de  = st.sidebar.checkbox("Discharge Effectiveness Index", value=True)
show_pt  = st.sidebar.checkbox("Pipeline Throughput Rate",      value=True)
show_bl  = st.sidebar.checkbox("Backlog Accumulation",          value=True)
show_oss = st.sidebar.checkbox("Outcome Stability Score",       value=False)

# ── Bottleneck Thresholds ─────────────────────────────────────────
st.sidebar.subheader("⚠️ Bottleneck Thresholds")
te_thresh  = st.sidebar.slider(
    "Transfer Efficiency threshold",
    min_value=0.10, max_value=0.80,
    value=0.40, step=0.05
)
de_thresh  = st.sidebar.slider(
    "Discharge Effectiveness threshold",
    min_value=0.005, max_value=0.050,
    value=0.015, step=0.005,
    format="%.3f"
)
bl_thresh  = st.sidebar.slider(
    "Backlog threshold (HHS Care)",
    min_value=2000, max_value=12000,
    value=8000, step=500
)

st.sidebar.divider()
st.sidebar.caption(
    "Data: HHS UAC Daily Reporting Dataset\n"
    "Period: Jan 2023 – Dec 2025\n"
    "Records: 720 valid reporting days"
)

# ── Apply Filters to Dataset ──────────────────────────────────────
filtered_df = df[
    (df['Date'].dt.date >= start_date) &
    (df['Date'].dt.date <= end_date) &
    (df['Year'].isin(selected_years))
].copy()

# Guard against empty filtered data
if filtered_df.empty:
    st.warning("No data found for the selected filters. Please adjust the date range or year selection.")
    st.stop()

filtered_df, bottleneck_monthly_pct, stagnation_df = prepare_filtered_analytics(
    filtered_df,
    te_thresh,
    de_thresh,
    bl_thresh,
)

# ═════════════════════════════════════════════════════════════════════════════
# ──  SECTION 4: KPI METRIC CARDS (Top Row)  ────────────────────────────────
# ═════════════════════════════════════════════════════════════════════════════

section_header(
    "📈 Key Performance Indicators",
    "Quick snapshot of selected KPIs compared to the full baseline period."
)

metrics_to_render = []
if show_te:
    avg_te = filtered_df['Transfer_Efficiency'].mean()
    metrics_to_render.append((
        "Transfer Efficiency Ratio",
        f"{avg_te:.1%}",
        f"{(avg_te - df['Transfer_Efficiency'].mean()):.1%}",
        "% of children transferred from CBP custody to HHS care"
    ))
if show_de:
    avg_de = filtered_df['Discharge_Effectiveness'].mean()
    metrics_to_render.append((
        "Discharge Effectiveness Index",
        f"{avg_de:.2%}",
        f"{(avg_de - df['Discharge_Effectiveness'].mean()):.2%}",
        "% of children discharged from HHS care to safe placement"
    ))
if show_pt:
    avg_pt = filtered_df['Pipeline_Throughput'].mean()
    metrics_to_render.append((
        "Pipeline Throughput Rate",
        f"{avg_pt:.1%}",
        f"{(avg_pt - df['Pipeline_Throughput'].mean()):.1%}",
        "Overall processing rate through care pipeline"
    ))
if show_bl:
    avg_bl = filtered_df['Backlog'].mean()
    metrics_to_render.append((
        "Avg Backlog",
        f"{avg_bl:.0f}",
        f"{(avg_bl - df['Backlog'].mean()):.0f}",
        "Children awaiting transfer or discharge"
    ))

if metrics_to_render:
    metric_cols = st.columns(len(metrics_to_render))
    for col, (label, value, delta, hint) in zip(metric_cols, metrics_to_render):
        with col:
            st.metric(label, value, delta=delta, help=hint)
else:
    st.info("Enable at least one KPI in the sidebar to display metric cards.")

st.divider()

# ═════════════════════════════════════════════════════════════════════════════
# ──  SECTION 5: CARE PIPELINE FLOW CHART  ──────────────────────────────────
# ═════════════════════════════════════════════════════════════════════════════

section_header(
    "🔄 Care Pipeline Flow (Current Period Average)",
    "Average daily volume across key transitions in the care pathway."
)

# Calculate average flow across the period
avg_apprehended = filtered_df['CBP_Apprehended'].mean()
avg_transfer = filtered_df['CBP_Transferred'].mean()
avg_hhs_care = filtered_df['HHS_Care'].mean()
avg_discharge = filtered_df['HHS_Discharged'].mean()

# Create Sankey diagram
fig_sankey = go.Figure(data=[go.Sankey(
    node=dict(
        pad=15,
        thickness=20,
        line=dict(color='black', width=0.5),
        label=[
            "CBP Apprehended",
            "CBP Custody",
            "Transferred to HHS",
            "HHS Care",
            "Discharged"
        ],
        color=[COLOR['violet'], COLOR['green'], COLOR['orange'], COLOR['blue'], COLOR['amber']]
    ),
    link=dict(
        source=[0, 1, 1, 2, 3],
        target=[1, 2, 3, 3, 4],
        value=[
            avg_apprehended,
            avg_transfer,
            avg_hhs_care - avg_transfer,
            avg_hhs_care,
            avg_discharge
        ],
        color=["rgba(83, 74, 183, 0.4)", "rgba(216, 90, 48, 0.4)", "rgba(55, 138, 221, 0.4)", "rgba(55, 138, 221, 0.4)", "rgba(186, 117, 23, 0.4)"]
    )
)])

fig_sankey.update_layout(
    title_text="Daily Average Flow Through Care Pipeline",
    font_size=10,
    height=400,
    showlegend=False
)

st.plotly_chart(fig_sankey, width='stretch')

st.divider()

# ═════════════════════════════════════════════════════════════════════════════
# ──  SECTION 6: TRANSFER & DISCHARGE EFFICIENCY PANELS  ─────────────────────
# ═════════════════════════════════════════════════════════════════════════════

section_header(
    "⚡ Transfer Efficiency vs Discharge Effectiveness",
    "Daily trend with threshold overlays for early warning monitoring."
)

col_left, col_right = st.columns(2)

# Panel 1: Transfer Efficiency Trend
with col_left:
    fig_te = go.Figure()
    
    fig_te.add_trace(go.Scatter(
        x=filtered_df['Date'],
        y=filtered_df['Transfer_Efficiency'] * 100,
        mode='lines',
        name='Transfer Efficiency',
        line=dict(color=COLOR['orange'], width=2),
        fill='tozeroy',
        fillcolor='rgba(216, 90, 48, 0.2)'
    ))
    
    fig_te.add_hline(
        y=te_thresh * 100,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Threshold ({te_thresh:.1%})",
        annotation_position="right"
    )
    
    fig_te.update_layout(
        title="Transfer Efficiency Over Time",
        xaxis_title="Date",
        yaxis_title="Efficiency (%)",
        height=350,
        hovermode='x unified'
    )
    
    st.plotly_chart(fig_te, width='stretch')

# Panel 2: Discharge Effectiveness Trend
with col_right:
    fig_de = go.Figure()
    
    fig_de.add_trace(go.Scatter(
        x=filtered_df['Date'],
        y=filtered_df['Discharge_Effectiveness'] * 100,
        mode='lines',
        name='Discharge Effectiveness',
        line=dict(color=COLOR['green'], width=2),
        fill='tozeroy',
        fillcolor='rgba(29, 158, 117, 0.2)'
    ))
    
    fig_de.add_hline(
        y=de_thresh * 100,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Threshold ({de_thresh:.2%})",
        annotation_position="right"
    )
    
    fig_de.update_layout(
        title="Discharge Effectiveness Over Time",
        xaxis_title="Date",
        yaxis_title="Effectiveness (%)",
        height=350,
        hovermode='x unified'
    )
    
    st.plotly_chart(fig_de, width='stretch')

st.divider()

# ═════════════════════════════════════════════════════════════════════════════
# ──  SECTION 7: BOTTLENECK DETECTION CHART  ────────────────────────────────
# ═════════════════════════════════════════════════════════════════════════════

section_header(
    "⚠️ Bottleneck Risk Heatmap",
    "Monthly mix of bottleneck types based on the selected thresholds."
)

fig_heatmap = go.Figure(data=go.Heatmap(
    z=bottleneck_monthly_pct[['Transfer', 'Discharge', 'Backlog']].T.values,
    x=bottleneck_monthly_pct.index.astype(str),
    y=['Transfer Bottleneck', 'Discharge Bottleneck', 'Backlog Bottleneck'],
    colorscale='RdYlGn_r',
    colorbar=dict(title="% Days<br>with Issue", thickness=15),
    hoverongaps=False
))

fig_heatmap.update_layout(
    title="Monthly Bottleneck Risk Heatmap (% of days with issue)",
    xaxis_title="Month",
    yaxis_title="Bottleneck Type",
    height=300
)

st.plotly_chart(fig_heatmap, width='stretch')

# Summary alert
bottleneck_count = filtered_df['Any_Bottleneck'].sum()
bottleneck_pct = (bottleneck_count / len(filtered_df) * 100) if len(filtered_df) > 0 else 0

if bottleneck_pct > 50:
    st.markdown(
        f'<div class="alert-red"><b>⚠️ CRITICAL:</b> {bottleneck_pct:.1f}% of days have bottlenecks detected.</div>',
        unsafe_allow_html=True
    )
elif bottleneck_pct > 25:
    st.markdown(
        f'<div class="alert-orange"><b>⚠️ WARNING:</b> {bottleneck_pct:.1f}% of days have bottlenecks detected.</div>',
        unsafe_allow_html=True
    )
else:
    st.markdown(
        f'<div class="alert-green"><b>✓ GOOD:</b> {bottleneck_pct:.1f}% of days have bottlenecks detected.</div>',
        unsafe_allow_html=True
    )

st.divider()

# ═════════════════════════════════════════════════════════════════════════════
# ──  SECTION 8: OUTCOME TREND ANALYSIS  ─────────────────────────────────────
# ═════════════════════════════════════════════════════════════════════════════

section_header(
    "📊 Outcome Trend Analysis (30-Day Rolling Average)",
    "Smoothed patterns to evaluate sustained system performance over time."
)

fig_trends = make_subplots(
    rows=1, cols=1,
    specs=[[{"secondary_y": False}]]
)

if show_te:
    fig_trends.add_trace(go.Scatter(
        x=filtered_df['Date'],
        y=filtered_df['TE_rolling'] * 100,
        mode='lines',
        name='Transfer Efficiency',
        line=dict(color=COLOR['orange'], width=2.5)
    ))

if show_de:
    fig_trends.add_trace(go.Scatter(
        x=filtered_df['Date'],
        y=filtered_df['DE_rolling'] * 100,
        mode='lines',
        name='Discharge Effectiveness',
        line=dict(color=COLOR['green'], width=2.5)
    ))

if show_pt:
    fig_trends.add_trace(go.Scatter(
        x=filtered_df['Date'],
        y=filtered_df['PT_rolling'] * 100,
        mode='lines',
        name='Pipeline Throughput',
        line=dict(color=COLOR['blue'], width=2.5)
    ))

if show_oss:
    fig_trends.add_trace(go.Scatter(
        x=filtered_df['Date'],
        y=filtered_df['Outcome_Stability'] * 100,
        mode='lines',
        name='Outcome Stability (σ, lower better)',
        line=dict(color=COLOR['violet'], width=2, dash='dot')
    ))

fig_trends.update_layout(
    title="30-Day Rolling Average of Key Efficiency Metrics",
    xaxis_title="Date",
    yaxis_title="Efficiency (%)",
    height=400,
    hovermode='x unified',
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)

st.plotly_chart(fig_trends, width='stretch')

section_header(
    "⬇️ Export Data & Visuals",
    "Download filtered records and key visuals for reporting."
)

export_col1, export_col2, export_col3 = st.columns(3)
with export_col1:
    st.download_button(
        label="Download filtered data (CSV)",
        data=to_csv_bytes(filtered_df.drop(columns=['Stagnation_Flag'], errors='ignore')),
        file_name="uac_filtered_data.csv",
        mime="text/csv"
    )

with export_col2:
    trends_png = fig_to_png_bytes(fig_trends)
    if trends_png:
        st.download_button(
            label="Download trend chart (PNG)",
            data=trends_png,
            file_name="uac_outcome_trends.png",
            mime="image/png"
        )
    else:
        st.caption("PNG export unavailable. Install `kaleido` to enable image downloads.")

with export_col3:
    st.download_button(
        label="Download flow chart (HTML)",
        data=fig_to_html_bytes(fig_sankey),
        file_name="uac_care_flow_chart.html",
        mime="text/html"
    )

st.divider()

# ═════════════════════════════════════════════════════════════════════════════
# ──  SECTION 9: STAGNATION PERIOD TABLE  ────────────────────────────────────
# ═════════════════════════════════════════════════════════════════════════════

section_header(
    "🔍 Stagnation Period Detection Table",
    "Periods with ≥3 consecutive days of low transfer/discharge and elevated backlog."
)

if len(stagnation_df) > 0:
    # Format for display
    display_df = stagnation_df.copy()
    display_df['Avg HHS Care'] = display_df['Avg HHS Care'].round(0).astype(int)
    display_df['Avg Transfer Eff (%)'] = display_df['Avg Transfer Eff (%)'].round(1)
    display_df['Avg Discharge Eff (%)'] = display_df['Avg Discharge Eff (%)'].round(1)
    
    st.dataframe(
        display_df.sort_values('Duration (days)', ascending=False),
        width='stretch',
        hide_index=True
    )
    
    st.caption(
        f"Showing {len(stagnation_df)} stagnation period(s) with ≥3 consecutive days of "
        f"reduced transfer efficiency, low discharge effectiveness, and elevated backlog."
    )
else:
    st.info("✓ No stagnation periods detected in the selected date range.")