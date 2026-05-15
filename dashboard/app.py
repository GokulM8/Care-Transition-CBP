# ── Imports ───────────────────────────────────────────────────────
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ── Color System ───────────────────────────────────────────────────
COLOR = {
    'navy': '#1F3A7A',
    'navy_dark': '#172a52',
    'violet': '#534AB7',
    'green': '#1D9E75',
    'green_dark': '#155e44',
    'orange': '#D85A30',
    'orange_dark': '#a83d1a',
    'blue': '#378ADD',
    'amber': '#BA7517',
    'danger': '#dc2626',
    'warning': '#f59e0b',
    'success': '#16a34a',
    'muted': '#6b7280',
    'bg_light': '#f5f7fa',
    'bg_white': '#ffffff',
}

# ── Page Configuration ────────────────────────────────────────────
st.set_page_config(
    page_title="UAC Care Pipeline Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS with Font Awesome Icons ────────────────────────
st.markdown("""
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <style>
        :root {
            --primary: #1F3A7A;
            --secondary: #378ADD;
            --primary-color: #1F3A7A;
            --secondary-color: #378ADD;
            --orange: #D85A30;
            --green: #1D9E75;
            --amber: #BA7517;
            --danger: #dc2626;
            --warning: #f59e0b;
            --success: #16a34a;
            --muted: #6b7280;
        }
        
        * { box-sizing: border-box; }
        
        body { 
            background: linear-gradient(135deg, #f5f7fa 0%, #e8eef7 100%) !important;
        }
        
        /* Dashboard Header */
        .dashboard-header {
            background: white !important;
            padding: 2rem !important;
            border-radius: 14px !important;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08) !important;
            margin-bottom: 2rem !important;
            border-top: 4px solid var(--primary-color) !important;
        }
        
        .header-content {
            display: flex !important;
            justify-content: space-between !important;
            align-items: flex-start !important;
            gap: 2rem !important;
        }
        
        .header-left h1 {
            font-size: 28px !important;
            font-weight: 700 !important;
            color: var(--primary-color) !important;
            margin: 0 0 0.5rem 0 !important;
            display: flex !important;
            align-items: center !important;
            gap: 12px !important;
        }
        
        .header-left h1 i {
            font-size: 32px !important;
            color: var(--secondary-color) !important;
        }
        
        .header-left p {
            color: var(--muted) !important;
            font-size: 14px !important;
            margin: 0 !important;
            font-weight: 500 !important;
        }
        
        .header-right {
            background: #f9fafb !important;
            padding: 1rem !important;
            border-radius: 8px !important;
            border-left: 3px solid var(--secondary-color) !important;
            font-size: 13px !important;
            color: #4b5563 !important;
        }
        
        .header-right strong {
            color: var(--primary-color) !important;
        }
        
        /* Section Headers */
        .section-header {
            display: flex !important;
            align-items: center !important;
            gap: 10px !important;
            font-size: 18px !important;
            font-weight: 700 !important;
            color: var(--primary) !important;
            margin: 2rem 0 0.75rem 0 !important;
            padding-bottom: 10px !important;
            border-bottom: 2px solid #e5e7eb !important;
        }
        
        .section-header i {
            font-size: 20px !important;
            color: var(--secondary) !important;
        }
        
        .section-caption {
            color: var(--muted) !important;
            margin: -0.5rem 0 1rem 0 !important;
            font-size: 13px !important;
            font-weight: 400 !important;
        }
        
        /* Chart Containers */
        .chart-container {
            background: white !important;
            border-radius: 12px !important;
            padding: 1.5rem !important;
            box-shadow: 0 1px 3px rgba(0,0,0,0.08) !important;
            margin: 1.5rem 0 !important;
        }
        
        .chart-title-inline {
            font-size: 15px !important;
            font-weight: 600 !important;
            color: var(--primary) !important;
            margin-bottom: 12px !important;
            display: flex !important;
            align-items: center !important;
            gap: 8px !important;
        }
        
        .chart-title-inline i {
            font-size: 16px !important;
            color: var(--secondary) !important;
        }
        
        /* Alert Boxes */
        .alert-critical {
            background: #fef2f2 !important;
            border-left: 4px solid var(--danger) !important;
            padding: 12px 16px !important;
            border-radius: 8px !important;
            margin: 1rem 0 !important;
            color: #7f1d1d !important;
            font-size: 13px !important;
            display: flex !important;
            align-items: flex-start !important;
            gap: 10px !important;
            line-height: 1.5 !important;
        }
        
        .alert-critical i {
            color: var(--danger) !important;
            margin-top: 2px !important;
            flex-shrink: 0 !important;
        }
        
        .alert-warning {
            background: #fffbeb !important;
            border-left: 4px solid var(--warning) !important;
            padding: 12px 16px !important;
            border-radius: 8px !important;
            margin: 1rem 0 !important;
            color: #92400e !important;
            font-size: 13px !important;
            display: flex !important;
            align-items: flex-start !important;
            gap: 10px !important;
            line-height: 1.5 !important;
        }
        
        .alert-warning i {
            color: var(--warning) !important;
            margin-top: 2px !important;
            flex-shrink: 0 !important;
        }
        
        .alert-success {
            background: #f0fdf4 !important;
            border-left: 4px solid var(--success) !important;
            padding: 12px 16px !important;
            border-radius: 8px !important;
            margin: 1rem 0 !important;
            color: #15803d !important;
            font-size: 13px !important;
            display: flex !important;
            align-items: flex-start !important;
            gap: 10px !important;
            line-height: 1.5 !important;
        }
        
        .alert-success i {
            color: var(--success) !important;
            margin-top: 2px !important;
            flex-shrink: 0 !important;
        }
        
        /* Divider */
        .divider {
            border: 0 !important;
            height: 1px !important;
            background: linear-gradient(to right, transparent, #e5e7eb, transparent) !important;
            margin: 2rem 0 !important;
        }
        
        /* Sidebar */
        [data-testid="stSidebar"] {
            background: white !important;
        }
        
        [data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
            gap: 0.5rem !important;
        }

        .sidebar-brand-title {
            text-align: center;
            margin-top: 0.45rem;
            margin-bottom: 0.15rem;
            color: #1F3A7A;
            font-size: 35px;
            font-weight: 700;
            line-height: 1.15;
        }

        .sidebar-brand-title i {
            color: #378ADD;
            margin-right: 6px;
            font-size: 18px;
        }

        .sidebar-brand-subtitle {
            text-align: center;
            margin: 0 0 0.25rem 0;
            color: #6b7280;
            font-size: 12px;
            font-weight: 500;
            line-height: 1.35;
        }

        [data-testid="stSidebar"] .stExpander {
            margin-bottom: 0.35rem;
        }
        
        .sidebar-info {
            background: #f3f4f6;
            padding: 12px;
            border-radius: 8px;
            font-size: 12px;
            color: #4b5563;
            line-height: 1.6;
            border-left: 2px solid #378ADD;
        }
        
        .reset-btn {
            width: 100%;
            padding: 10px;
            background: #f3f4f6;
            border: 1px solid #d1d5db;
            border-radius: 6px;
            color: #1F3A7A;
            font-weight: 600;
            cursor: pointer;
            font-size: 13px;
            margin-top: 1rem;
            transition: all 0.2s;
        }
        
        .reset-btn:hover {
            background: #e5e7eb;
            border-color: #9ca3af;
        }
        
        /* Export Section */
        .export-section {
            background: white;
            border-radius: 12px;
            padding: 2rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.08);
            margin: 2rem 0;
            border-top: 3px solid #16a34a;
        }
        
        .export-buttons {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
        }
        
        /* Metric Row */
        .metric-row {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
            margin: 1.5rem 0;
        }
        
        /* Metrics Container */
        .metrics-container {
            background: white;
            border-radius: 12px;
            padding: 2rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.08);
            margin: 1.5rem 0;
        }
        
        /* Table Styling */
        [data-testid="stDataFrame"] {
            background: white !important;
            border-radius: 8px !important;
        }
        
        /* Typography */
        h1 { color: var(--primary-color) !important; font-weight: 700 !important; }
        h2 { color: var(--primary-color) !important; font-weight: 700 !important; }
        h3 { color: var(--primary-color) !important; font-weight: 700 !important; }
        p { line-height: 1.6 !important; }
    </style>
""", unsafe_allow_html=True)


# ── Helper Functions ──────────────────────────────────────────────

def render_metric_cards(filtered_data, baseline_data, show_te, show_de, show_pt, show_bl):
    """Render KPI metric cards with icons."""
    metrics = []
    
    if show_te:
        avg_te = filtered_data['Transfer_Efficiency'].mean()
        metrics.append({
            'label': 'Transfer Efficiency',
            'value': f'{avg_te:.1%}',
            'delta': f"{(avg_te - baseline_data['Transfer_Efficiency'].mean()):.1%}",
            'help': '% children transferred from CBP to HHS care',
            'icon': 'fa-arrow-right-arrow-left',
        })
    
    if show_de:
        avg_de = filtered_data['Discharge_Effectiveness'].mean()
        metrics.append({
            'label': 'Discharge Effectiveness',
            'value': f'{avg_de:.2%}',
            'delta': f"{(avg_de - baseline_data['Discharge_Effectiveness'].mean()):.2%}",
            'help': '% children discharged to safe placement',
            'icon': 'fa-arrow-up-right-dots',
        })
    
    if show_pt:
        avg_pt = filtered_data['Pipeline_Throughput'].mean()
        metrics.append({
            'label': 'Pipeline Throughput',
            'value': f'{avg_pt:.1%}',
            'delta': f"{(avg_pt - baseline_data['Pipeline_Throughput'].mean()):.1%}",
            'help': 'Overall processing rate through pipeline',
            'icon': 'fa-forward-step',
        })
    
    if show_bl:
        avg_bl = filtered_data['Backlog'].mean()
        metrics.append({
            'label': 'Avg Backlog',
            'value': f'{avg_bl:.0f}',
            'delta': f"{(avg_bl - baseline_data['Backlog'].mean()):.0f}",
            'help': 'Children awaiting transfer/discharge',
            'icon': 'fa-hourglass-end',
        })
    
    if not metrics:
        st.info("Enable at least one KPI in the sidebar to display metrics.")
        return
    
    cols = st.columns(len(metrics))
    for col, metric in zip(cols, metrics):
        with col:
            st.metric(
                label=metric['label'],
                value=metric['value'],
                delta=metric['delta'],
                help=metric['help']
            )


def to_csv_bytes(dataframe: pd.DataFrame) -> bytes:
    """Convert dataframe to CSV bytes."""
    return dataframe.to_csv(index=False).encode('utf-8')


def fig_to_html_bytes(fig: go.Figure) -> bytes:
    """Convert plotly figure to HTML bytes."""
    return fig.to_html(include_plotlyjs='cdn').encode('utf-8')


def render_faqs():
    st.markdown("""
    ## Frequently Asked Questions

    **Q1: What is Transfer Efficiency?**
    - Transfer Efficiency = Children transferred ÷ Children in CBP custody.

    **Q2: How are bottlenecks detected?**
    - Bottlenecks are flagged when transfer/discharge thresholds are breached or HHS care exceeds the backlog threshold.

    **Q3: How do I submit a new project?**
    - Use the **Submit Project** form in the sidebar — submissions are stored locally as CSV.

    **Q4: Can I export charts and data?**
    - Yes — use the Export section near the bottom of the dashboard to download CSV and HTML exports.

    """, unsafe_allow_html=True)


def save_submission(data: dict) -> Path:
    """Append submission dict to data/submissions.csv and return path."""
    out_dir = Path(__file__).resolve().parent.parent / 'data'
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / 'submissions.csv'

    header = ['timestamp', 'name', 'email', 'project_title', 'summary']
    write_header = not out_file.exists()

    with out_file.open('a', encoding='utf-8', newline='') as f:
        import csv
        writer = csv.DictWriter(f, fieldnames=header)
        if write_header:
            writer.writeheader()
        row = {k: data.get(k, '') for k in header}
        writer.writerow(row)

    return out_file


def render_submit_form():
    st.header("Submit a Project")
    st.markdown("Use this form to submit a new project or analysis artifact for review.")

    with st.form("submit_project_form"):
        name = st.text_input("Your name")
        email = st.text_input("Email")
        project_title = st.text_input("Project title")
        summary = st.text_area("Short summary / description", height=150)
        submitted = st.form_submit_button("Submit Project")

        if submitted:
            payload = {
                'timestamp': pd.Timestamp.now().isoformat(),
                'name': name,
                'email': email,
                'project_title': project_title,
                'summary': summary,
            }
            out = save_submission(payload)
            st.success(f"Project submitted — stored at: {out}")
            st.session_state['open_page'] = None
            st.experimental_rerun()


def _csv_path(name: str) -> Path:
    p = Path(__file__).resolve().parent.parent / 'data' / name
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def save_chat_message(name: str, message: str) -> Path:
    out = _csv_path('chat.csv')
    import csv
    write_header = not out.exists()
    with out.open('a', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(['timestamp', 'name', 'message'])
        w.writerow([pd.Timestamp.now().isoformat(), name, message])
    return out


def load_chat_messages(limit: int = 200) -> pd.DataFrame:
    p = _csv_path('chat.csv')
    if not p.exists():
        return pd.DataFrame(columns=['timestamp', 'name', 'message'])
    return pd.read_csv(p).sort_values('timestamp', ascending=False).head(limit)


def render_chat():
    st.header("Chat Support")
    st.markdown("A simple support chat log — messages are stored locally.")
    df_chat = load_chat_messages()
    with st.form('chat_form'):
        cname = st.text_input('Your name')
        cmsg = st.text_area('Message', height=120)
        send = st.form_submit_button('Send')
        if send and cmsg.strip():
            save_chat_message(cname or 'Anonymous', cmsg.strip())
            st.success('Message posted')
            st.experimental_rerun()

    if not df_chat.empty:
        for _, row in df_chat.iterrows():
            st.markdown(f"**{row['name']}** — _{row['timestamp']}_")
            st.write(row['message'])
            st.markdown('---')
    else:
        st.info('No messages yet.')


def save_event(title: str, date: str, notes: str) -> Path:
    out = _csv_path('events.csv')
    import csv
    write_header = not out.exists()
    with out.open('a', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(['date', 'title', 'notes', 'created_at'])
        w.writerow([date, title, notes, pd.Timestamp.now().isoformat()])
    return out


def load_events():
    p = _csv_path('events.csv')
    if not p.exists():
        return pd.DataFrame(columns=['date', 'title', 'notes', 'created_at'])
    df = pd.read_csv(p)
    df['date'] = pd.to_datetime(df['date'])
    return df.sort_values('date')


def render_calendar():
    st.header('Event Calendar')
    st.markdown('Create and view scheduled events.')
    with st.form('event_form'):
        title = st.text_input('Event title')
        date = st.date_input('Event date')
        notes = st.text_area('Notes')
        submit = st.form_submit_button('Add event')
        if submit and title:
            save_event(title, date.isoformat(), notes)
            st.success('Event saved')
            st.experimental_rerun()

    df_events = load_events()
    if not df_events.empty:
        st.table(df_events[['date', 'title', 'notes']].head(50))
    else:
        st.info('No events scheduled.')


def save_job(post: dict) -> Path:
    out = _csv_path('jobs.csv')
    import csv
    write_header = not out.exists()
    with out.open('a', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['id', 'title', 'location', 'description', 'created_at'])
        if write_header:
            writer.writeheader()
        writer.writerow(post)
    return out


def load_jobs():
    p = _csv_path('jobs.csv')
    if not p.exists():
        return pd.DataFrame(columns=['id', 'title', 'location', 'description', 'created_at'])
    return pd.read_csv(p)


def save_application(job_id: str, applicant: dict) -> Path:
    out = _csv_path('applications.csv')
    import csv
    write_header = not out.exists()
    with out.open('a', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['job_id', 'name', 'email', 'cv', 'applied_at'])
        if write_header:
            writer.writeheader()
        row = {'job_id': job_id, 'name': applicant.get('name', ''), 'email': applicant.get('email', ''), 'cv': applicant.get('cv', ''), 'applied_at': pd.Timestamp.now().isoformat()}
        writer.writerow(row)
    return out


def render_jobs():
    st.header('Job Portal')
    st.markdown('View job listings or post a new job (admin).')
    jobs = load_jobs()
    with st.expander('Post a new job'):
        with st.form('post_job'):
            jid = str(int(pd.Timestamp.now().timestamp()))
            title = st.text_input('Job title')
            location = st.text_input('Location')
            desc = st.text_area('Description')
            post = st.form_submit_button('Post job')
            if post and title:
                save_job({'id': jid, 'title': title, 'location': location, 'description': desc, 'created_at': pd.Timestamp.now().isoformat()})
                st.success('Job posted')
                st.experimental_rerun()

    if not jobs.empty:
        for _, job in jobs.sort_values('created_at', ascending=False).iterrows():
            st.subheader(job['title'])
            st.caption(job.get('location', ''))
            st.write(job.get('description', ''))
            with st.form(f'apply_{job.id}'):
                name = st.text_input('Your name', key=f'name_{job.id}')
                email = st.text_input('Email', key=f'email_{job.id}')
                cv = st.text_area('Short CV / summary', key=f'cv_{job.id}')
                apply = st.form_submit_button('Apply')
                if apply and name:
                    save_application(job.id, {'name': name, 'email': email, 'cv': cv})
                    st.success('Application submitted')
    else:
        st.info('No job listings available.')


def render_profile():
    st.header('Profile & Projects')
    st.markdown('Manage your profile and view submitted projects.')
    profile_path = _csv_path('profile.csv')
    if profile_path.exists():
        prof = pd.read_csv(profile_path).iloc[0].to_dict()
    else:
        prof = {'name': '', 'email': '', 'organization': ''}

    with st.form('profile_form'):
        name = st.text_input('Name', value=prof.get('name', ''))
        email = st.text_input('Email', value=prof.get('email', ''))
        org = st.text_input('Organization', value=prof.get('organization', ''))
        save = st.form_submit_button('Save Profile')
        if save:
            profile_path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame([{'name': name, 'email': email, 'organization': org}]).to_csv(profile_path, index=False)
            st.success('Profile saved')

    st.markdown('### Your Submitted Projects')
    subs = _csv_path('submissions.csv')
    if subs.exists():
        df = pd.read_csv(subs)
        st.dataframe(df.sort_values('timestamp', ascending=False).head(20))
    else:
        st.info('No submitted projects yet.')


def render_docs():
    st.header('Technical Documentation')
    readme = Path(__file__).resolve().parent.parent / 'README.md'
    if readme.exists():
        st.markdown(readme.read_text())
    else:
        st.info('No README.md found in project root.')
    st.markdown('### Notebooks')
    nb_dir = Path(__file__).resolve().parent.parent / 'notebooks'
    for nb in nb_dir.glob('*.ipynb'):
        st.write(nb.name)


def render_qa():
    st.header('QA Checklist')
    checks = []
    # check data
    data_file = Path(__file__).resolve().parent.parent / 'data' / 'processed' / 'uac_engineered.csv'
    checks.append(('Processed dataset exists', data_file.exists()))
    checks.append(('Submissions file exists', _csv_path('submissions.csv').exists()))
    checks.append(('Jobs file exists', _csv_path('jobs.csv').exists()))
    checks.append(('Events file exists', _csv_path('events.csv').exists()))
    st.markdown('\n'.join([f"- {'✅' if ok else '❌'} {label}" for label, ok in checks]))


def section_header_html(title: str, icon: str = ""):
    """Render section header with Font Awesome icon."""
    icon_html = f'<i class="fas {icon}"></i>' if icon else ''
    st.markdown(
        f'<div class="section-header">{icon_html} {title}</div>',
        unsafe_allow_html=True
    )


def section_caption_html(caption: str):
    """Render section caption."""
    st.markdown(
        f'<div class="section-caption">{caption}</div>',
        unsafe_allow_html=True
    )


def _is_valid_logo_file(path: Path) -> bool:
    """Lightweight validation for local image files used in sidebar branding."""
    if not path.exists() or not path.is_file() or path.stat().st_size < 100:
        return False

    header = path.read_bytes()[:512].lower()
    if header.startswith(b'\x89png\r\n\x1a\n'):
        return True
    if header.startswith(b'\xff\xd8\xff'):
        return True
    if b'<svg' in header and b'<html' not in header:
        return True
    return False


def resolve_sidebar_logo() -> Path | None:
    """Return the first valid local logo candidate from assets."""
    assets_dir = Path(__file__).resolve().parent.parent / 'assets'
    candidates = [
        'Seal_of_the_United_States_Department_of_Health_and_Human_Services.svg.svg',
        'hhs_seal_official.svg',
        'hhs_seal.svg',
        'hhs_seal.png',
        'hhs_logo.svg',
        'hhs_seal_local.png',
    ]

    for name in candidates:
        candidate = assets_dir / name
        if _is_valid_logo_file(candidate):
            return candidate

    return None


# ── Section 2: Load & Prepare Data ───────────────────────────────

@st.cache_data
def load_data():
    data_path = Path(__file__).resolve().parent.parent / 'data' / 'processed' / 'uac_engineered.csv'
    df = pd.read_csv(data_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)

    df['Year']    = df['Date'].dt.year
    df['Month']   = df['Date'].dt.month
    df['YearMon'] = df['Date'].dt.to_period('M').astype(str)
    df['Weekday'] = df['Date'].dt.day_name()

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
    """Prepare cached analytics for filtered data."""
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

# ═════════════════════════════════════════════════════════════════════════════
# ──  SECTION 3: SIDEBAR (Restructured)  ────────────────────────────────────
# ═════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    sidebar_logo = resolve_sidebar_logo()

    if sidebar_logo is not None:
        c1, c2, c3 = st.columns([1, 2, 1])
        with c2:
            st.image(str(sidebar_logo), width=90)
    else:
        st.markdown(
            """
            <div style="display:flex; justify-content:center; margin: 0.2rem 0 0.35rem 0;">
                <div style="width:76px; height:76px; border-radius:50%; background:linear-gradient(145deg,#f8fbff,#e8f0ff);
                            border:2px solid #cbdaf7; display:flex; align-items:center; justify-content:center;
                            box-shadow: 0 2px 8px rgba(31,58,122,0.12);">
                    <i class="fas fa-shield-halved" style="font-size:33px; color:#1F3A7A;"></i>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown(
        "<div class='sidebar-brand-title'><i class='fas fa-chart-line'></i>UAC Pipeline</div>",
        unsafe_allow_html=True
    )
    st.markdown(
        "<div class='sidebar-brand-subtitle'>HHS Office of Refugee Resettlement</div>",
        unsafe_allow_html=True
    )
    st.divider()
    
    # ── Filters Section ──
    with st.expander("📅 **Filters**", expanded=True):
        st.markdown("##### Date Range")
        min_date = df['Date'].min().date()
        max_date = df['Date'].max().date()
        
        start_date, end_date = st.date_input(
            "Select date range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
            key="date_range",
            label_visibility="collapsed"
        )
        
        st.markdown("##### Year Selection")
        years_available = sorted(df['Year'].unique().tolist())
        selected_years = st.multiselect(
            "Select years",
            options=years_available,
            default=years_available,
            key="selected_years",
            label_visibility="collapsed"
        )
    
    # ── KPI Toggles Section ──
    with st.expander("📊 **Metrics**", expanded=True):
        show_te  = st.checkbox("Transfer Efficiency Ratio",     value=True, key="show_te")
        show_de  = st.checkbox("Discharge Effectiveness Index", value=True, key="show_de")
        show_pt  = st.checkbox("Pipeline Throughput Rate",      value=False, key="show_pt")
        show_bl  = st.checkbox("Backlog Accumulation",          value=True, key="show_bl")
        show_oss = st.checkbox("Outcome Stability Score",       value=False, key="show_oss")
    
    # ── Thresholds Section ──
    with st.expander("⚙️ **Thresholds**", expanded=False):
        st.markdown("##### Transfer Efficiency")
        te_thresh  = st.slider(
            "Threshold",
            min_value=0.10, max_value=0.80,
            value=0.40, step=0.05,
            key="te_thresh",
            label_visibility="collapsed"
        )
        
        st.markdown("##### Discharge Effectiveness")
        de_thresh  = st.slider(
            "Threshold",
            min_value=0.005, max_value=0.050,
            value=0.015, step=0.005,
            format="%.3f",
            key="de_thresh",
            label_visibility="collapsed"
        )
        
        st.markdown("##### Backlog (HHS Care)")
        bl_thresh  = st.slider(
            "Threshold",
            min_value=2000, max_value=12000,
            value=8000, step=500,
            key="bl_thresh",
            label_visibility="collapsed"
        )
    
    # ── Dataset Info Section ──
    with st.expander("ℹ️ **Dataset Info**", expanded=False):
        st.markdown(
            '<div class="sidebar-info">'
            '<strong>Data Source:</strong> HHS UAC Daily Reporting Dataset<br>'
            '<strong>Period:</strong> Jan 2023 – Dec 2025<br>'
            '<strong>Records:</strong> 720 valid reporting days<br>'
            '<strong>Last Updated:</strong> 2025-12-21'
            '</div>',
            unsafe_allow_html=True
        )
    
    st.divider()
    # Quick Links
    if st.button("📋 FAQs", key="open_faqs"):
        st.session_state["open_page"] = "faqs"
        st.experimental_rerun()

    if st.button("📝 Submit Project", key="open_submit"):
        st.session_state["open_page"] = "submit"
        st.experimental_rerun()
    # Extra app pages
    if st.button("💬 Chat Support", key="open_chat"):
        st.session_state["open_page"] = "chat"
        st.experimental_rerun()

    if st.button("📅 Calendar", key="open_calendar"):
        st.session_state["open_page"] = "calendar"
        st.experimental_rerun()

    if st.button("🏢 Job Portal", key="open_jobs"):
        st.session_state["open_page"] = "jobs"
        st.experimental_rerun()

    if st.button("👤 Profile", key="open_profile"):
        st.session_state["open_page"] = "profile"
        st.experimental_rerun()

    if st.button("📚 Documentation", key="open_docs"):
        st.session_state["open_page"] = "docs"
        st.experimental_rerun()

    if st.button("✅ QA Checklist", key="open_qa"):
        st.session_state["open_page"] = "qa"
        st.experimental_rerun()
    
    # Reset Filters Button
    if st.button("🔄 Reset Filters", key="reset_filters", use_container_width=True):
        st.session_state["date_range"] = (min_date, max_date)
        st.session_state["selected_years"] = years_available
        st.session_state["show_te"] = True
        st.session_state["show_de"] = True
        st.session_state["show_pt"] = False
        st.session_state["show_bl"] = True
        st.session_state["show_oss"] = False
        st.session_state["te_thresh"] = 0.40
        st.session_state["de_thresh"] = 0.015
        st.session_state["bl_thresh"] = 8000
        st.rerun()

# ── Apply Filters to Dataset ──────────────────────────────────────
filtered_df = df[
    (df['Date'].dt.date >= start_date) &
    (df['Date'].dt.date <= end_date) &
    (df['Year'].isin(selected_years))
].copy()

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
# ──  DASHBOARD HEADER (Top Summary Bar)  ────────────────────────────────────
# ═════════════════════════════════════════════════════════════════════════════

st.markdown(
    f"""
    <div class="dashboard-header">
        <div class="header-content">
            <div class="header-left">
                <h1><i class="fas fa-chart-line"></i> UAC Care Pipeline Analytics</h1>
                <p>Care Transition Efficiency & Placement Outcome Dashboard — HHS Program</p>
            </div>
            <div class="header-right">
                <strong>Period Selected:</strong> {start_date.strftime('%b %d, %Y')} → {end_date.strftime('%b %d, %Y')}<br>
                <strong>Records:</strong> {len(filtered_df)} days | <strong>Years:</strong> {', '.join(map(str, sorted(selected_years)))}
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# Handle simple in-app pages (FAQs / Submit)
if 'open_page' in st.session_state and st.session_state['open_page'] == 'faqs':
    render_faqs()
    st.stop()

if 'open_page' in st.session_state and st.session_state['open_page'] == 'submit':
    render_submit_form()
    st.stop()

if 'open_page' in st.session_state and st.session_state['open_page'] == 'chat':
    render_chat()
    st.stop()

if 'open_page' in st.session_state and st.session_state['open_page'] == 'calendar':
    render_calendar()
    st.stop()

if 'open_page' in st.session_state and st.session_state['open_page'] == 'jobs':
    render_jobs()
    st.stop()

if 'open_page' in st.session_state and st.session_state['open_page'] == 'profile':
    render_profile()
    st.stop()

if 'open_page' in st.session_state and st.session_state['open_page'] == 'docs':
    render_docs()
    st.stop()

if 'open_page' in st.session_state and st.session_state['open_page'] == 'qa':
    render_qa()
    st.stop()

# ═════════════════════════════════════════════════════════════════════════════
# ──  SECTION 4: KPI METRIC CARDS  ──────────────────────────────────────────
# ═════════════════════════════════════════════════════════════════════════════

st.markdown("<div class='metrics-container'>", unsafe_allow_html=True)
section_header_html("Key Performance Indicators", "fa-gauge")
section_caption_html("Quick snapshot of selected KPIs compared to full baseline period")
render_metric_cards(filtered_df, df, show_te, show_de, show_pt, show_bl)
st.markdown("</div>", unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════════════
# ──  SECTION 5: CARE PIPELINE FLOW CHART (Hero)  ───────────────────────────
# ═════════════════════════════════════════════════════════════════════════════

avg_apprehended = filtered_df['CBP_Apprehended'].mean()
avg_transfer = filtered_df['CBP_Transferred'].mean()
avg_hhs_care = filtered_df['HHS_Care'].mean()
avg_discharge = filtered_df['HHS_Discharged'].mean()

sankey_values = [
    float(np.nan_to_num(avg_apprehended, nan=0.0)),
    float(np.nan_to_num(avg_transfer, nan=0.0)),
    max(0.0, float(np.nan_to_num(avg_hhs_care - avg_transfer, nan=0.0))),
    float(np.nan_to_num(avg_hhs_care, nan=0.0)),
    float(np.nan_to_num(avg_discharge, nan=0.0)),
]

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
            *sankey_values
        ],
        color=["rgba(83, 74, 183, 0.4)", "rgba(216, 90, 48, 0.4)", "rgba(55, 138, 221, 0.4)", "rgba(55, 138, 221, 0.4)", "rgba(186, 117, 23, 0.4)"]
    )
)])

fig_sankey.update_layout(
    title_text="",
    font_size=10,
    height=450,
    showlegend=False,
    margin=dict(l=0, r=0, t=20, b=0)
)

st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
section_header_html("Care Pipeline Flow (Current Period Average)", "fa-diagram-project")
section_caption_html("Average daily volume across key transitions in the care pathway")
st.plotly_chart(fig_sankey, use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════════════
# ──  SECTION 6: TRANSFER & DISCHARGE EFFICIENCY PANELS  ─────────────────────
# ═════════════════════════════════════════════════════════════════════════════

section_header_html("Transfer Efficiency vs Discharge Effectiveness", "fa-arrows-split-up-and-left")
section_caption_html("Daily trends with threshold overlays for early warning monitoring")

col_left, col_right = st.columns(2)

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
        line_color=COLOR['danger'],
        annotation_text=f"Threshold ({te_thresh:.1%})",
        annotation_position="right"
    )
    
    fig_te.update_layout(
        title="",
        xaxis_title="Date",
        yaxis_title="Efficiency (%)",
        height=350,
        hovermode='x unified',
        margin=dict(l=50, r=50, t=20, b=50)
    )
    
    st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
    st.markdown("<div class='chart-title-inline'><i class='fas fa-arrow-right-arrow-left'></i> Transfer Efficiency</div>", unsafe_allow_html=True)
    st.plotly_chart(fig_te, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

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
        line_color=COLOR['danger'],
        annotation_text=f"Threshold ({de_thresh:.2%})",
        annotation_position="right"
    )
    
    fig_de.update_layout(
        title="",
        xaxis_title="Date",
        yaxis_title="Effectiveness (%)",
        height=350,
        hovermode='x unified',
        margin=dict(l=50, r=50, t=20, b=50)
    )
    
    st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
    st.markdown("<div class='chart-title-inline'><i class='fas fa-arrow-up-right-dots'></i> Discharge Effectiveness</div>", unsafe_allow_html=True)
    st.plotly_chart(fig_de, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════════════
# ──  SECTION 7: BOTTLENECK DETECTION CHART  ────────────────────────────────
# ═════════════════════════════════════════════════════════════════════════════

bottleneck_count = filtered_df['Any_Bottleneck'].sum()
bottleneck_pct = (bottleneck_count / len(filtered_df) * 100) if len(filtered_df) > 0 else 0

st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

section_header_html("Bottleneck Risk Analysis", "fa-triangle-exclamation")
section_caption_html("Monthly mix of bottleneck types based on selected thresholds")

# Alert Box
if bottleneck_pct > 50:
    alert_type = "critical"
    icon = "fa-exclamation-circle"
    msg = f"CRITICAL: {bottleneck_pct:.1f}% of days have bottlenecks detected. Immediate action recommended."
elif bottleneck_pct > 25:
    alert_type = "warning"
    icon = "fa-exclamation-triangle"
    msg = f"WARNING: {bottleneck_pct:.1f}% of days have bottlenecks detected. Monitor closely."
else:
    alert_type = "success"
    icon = "fa-check-circle"
    msg = f"GOOD: {bottleneck_pct:.1f}% of days have bottlenecks detected. System performing well."

st.markdown(
    f'<div class="alert-{alert_type}"><i class="fas {icon}"></i><span>{msg}</span></div>',
    unsafe_allow_html=True
)

# Heatmap
fig_heatmap = go.Figure(data=go.Heatmap(
    z=bottleneck_monthly_pct[['Transfer', 'Discharge', 'Backlog']].T.values,
    x=bottleneck_monthly_pct.index.astype(str),
    y=['Transfer Bottleneck', 'Discharge Bottleneck', 'Backlog Bottleneck'],
    colorscale='RdYlGn_r',
    colorbar=dict(title="% Days<br>with Issue", thickness=15),
    hoverongaps=False
))

fig_heatmap.update_layout(
    title="",
    xaxis_title="Month",
    yaxis_title="Bottleneck Type",
    height=300,
    margin=dict(l=150, r=50, t=20, b=50)
)

st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
st.plotly_chart(fig_heatmap, use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════════════
# ──  SECTION 8: OUTCOME TREND ANALYSIS  ─────────────────────────────────────
# ═════════════════════════════════════════════════════════════════════════════

st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

section_header_html("Outcome Trend Analysis (30-Day Rolling Average)", "fa-chart-area")
section_caption_html("Smoothed patterns to evaluate sustained system performance over time")

fig_trends = make_subplots(rows=1, cols=1, specs=[[{"secondary_y": False}]])

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
        name='Outcome Stability (σ)',
        line=dict(color=COLOR['violet'], width=2, dash='dot')
    ))

fig_trends.update_layout(
    title="",
    xaxis_title="Date",
    yaxis_title="Metric Value (%)",
    height=400,
    hovermode='x unified',
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    margin=dict(l=50, r=50, t=20, b=50)
)

st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
st.plotly_chart(fig_trends, use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════════════
# ──  SECTION 9: EXPORT SECTION  ────────────────────────────────────────────
# ═════════════════════════════════════════════════════════════════════════════

st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

st.markdown(
    '<div class="export-section">'
    '<h3 style="margin-top: 0;"><i class="fas fa-download"></i> Export Data & Visuals</h3>'
    '<p style="color: #6b7280; margin-bottom: 1.5rem;">Download filtered records and key visuals for reporting and further analysis.</p>',
    unsafe_allow_html=True
)

export_col1, export_col2, export_col3 = st.columns(3)

with export_col1:
    st.download_button(
        label="📊 Download Data (CSV)",
        data=to_csv_bytes(filtered_df.drop(columns=['Stagnation_Flag'], errors='ignore')),
        file_name="uac_filtered_data.csv",
        mime="text/csv",
        use_container_width=True
    )

with export_col2:
    st.download_button(
        label="📈 Download Trends (HTML)",
        data=fig_to_html_bytes(fig_trends),
        file_name="uac_outcome_trends.html",
        mime="text/html",
        use_container_width=True
    )

with export_col3:
    st.download_button(
        label="🔄 Download Flow Chart (HTML)",
        data=fig_to_html_bytes(fig_sankey),
        file_name="uac_care_flow_chart.html",
        mime="text/html",
        use_container_width=True
    )

st.markdown("</div>", unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════════════
# ──  SECTION 10: STAGNATION PERIOD TABLE  ──────────────────────────────────
# ═════════════════════════════════════════════════════════════════════════════

st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

section_header_html("Stagnation Period Detection", "fa-hourglass-end")
section_caption_html("Periods with ≥3 consecutive days of low transfer/discharge and elevated backlog")

if len(stagnation_df) > 0:
    display_df = stagnation_df.copy()
    display_df['Avg HHS Care'] = display_df['Avg HHS Care'].round(0).astype(int)
    display_df['Avg Transfer Eff (%)'] = display_df['Avg Transfer Eff (%)'].round(1)
    display_df['Avg Discharge Eff (%)'] = display_df['Avg Discharge Eff (%)'].round(1)
    
    st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
    st.dataframe(
        display_df.sort_values('Duration (days)', ascending=False),
        use_container_width=True,
        hide_index=True
    )
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.caption(
        f"Showing {len(stagnation_df)} stagnation period(s). These periods require targeted interventions "
        f"to improve care pipeline efficiency."
    )
else:
    st.markdown(
        '<div style="background: #f0fdf4; border-left: 4px solid #16a34a; padding: 12px 16px; '
        'border-radius: 8px; color: #15803d;"><i class="fas fa-check-circle"></i> '
        '<strong>No stagnation periods detected</strong> in the selected date range.</div>',
        unsafe_allow_html=True
    )

st.markdown("<div style='height: 2rem;'></div>", unsafe_allow_html=True)
