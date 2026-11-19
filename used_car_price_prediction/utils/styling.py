"""
Custom UI/UX styling for the Used Car Price Prediction Application.
Automotive Dark Navy, Slate, and Cyan theme with responsive card layouts.
"""

import streamlit as st

def apply_custom_styles():
    """Injects custom CSS styling for modern automotive dark theme."""
    st.markdown("""
    <style>
    /* Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

    /* Global Typography & Background */
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }

    /* Main Container Padding */
    .block-container {
        padding-top: 1.8rem;
        padding-bottom: 3rem;
        padding-left: 2.2rem;
        padding-right: 2.2rem;
    }

    /* Top Navigation / Header styling */
    .app-header {
        background: linear-gradient(135deg, #0B192C 0%, #1E3E62 60%, #008DDA 100%);
        border: 1px solid rgba(0, 141, 218, 0.3);
        border-radius: 14px;
        padding: 24px 28px;
        margin-bottom: 24px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.35);
    }
    .app-header h1 {
        color: #FFFFFF;
        font-weight: 800;
        letter-spacing: -0.5px;
        margin: 0;
        font-size: 2.1rem;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .app-header p {
        color: #A0C4E2;
        margin: 8px 0 0 0;
        font-size: 0.95rem;
    }

    /* Automotive Metric Cards */
    .metric-card-container {
        background: linear-gradient(145deg, #132743 0%, #1E3E62 100%);
        border: 1px solid rgba(65, 201, 226, 0.2);
        border-radius: 12px;
        padding: 18px 20px;
        box-shadow: 0 6px 18px rgba(0, 0, 0, 0.25);
        transition: transform 0.2s ease, border-color 0.2s ease;
        position: relative;
        overflow: hidden;
    }
    .metric-card-container:hover {
        transform: translateY(-2px);
        border-color: rgba(65, 201, 226, 0.5);
    }
    .metric-card-container::before {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        width: 4px;
        height: 100%;
        background: #008DDA;
    }
    .metric-card-title {
        color: #94A3B8;
        font-size: 0.82rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 6px;
    }
    .metric-card-value {
        color: #FFFFFF;
        font-size: 1.75rem;
        font-weight: 800;
        letter-spacing: -0.5px;
        line-height: 1.2;
    }
    .metric-card-sub {
        color: #41C9E2;
        font-size: 0.8rem;
        margin-top: 5px;
        font-weight: 500;
    }

    /* Section Subheaders */
    .section-title {
        font-size: 1.35rem;
        font-weight: 700;
        color: #F1F6F9;
        margin-top: 1.5rem;
        margin-bottom: 0.8rem;
        display: flex;
        align-items: center;
        gap: 8px;
        border-left: 4px solid #008DDA;
        padding-left: 10px;
    }
    .section-desc {
        color: #94A3B8;
        font-size: 0.9rem;
        margin-bottom: 1.2rem;
    }

    /* Badges */
    .custom-badge {
        display: inline-block;
        padding: 4px 10px;
        font-size: 0.75rem;
        font-weight: 600;
        border-radius: 6px;
        letter-spacing: 0.04em;
    }
    .badge-primary { background: rgba(0, 141, 218, 0.2); color: #41C9E2; border: 1px solid rgba(0, 141, 218, 0.4); }
    .badge-success { background: rgba(16, 185, 129, 0.2); color: #34D399; border: 1px solid rgba(16, 185, 129, 0.4); }
    .badge-warning { background: rgba(245, 158, 11, 0.2); color: #FBBF24; border: 1px solid rgba(245, 158, 11, 0.4); }
    .badge-purple  { background: rgba(168, 85, 247, 0.2); color: #C084FC; border: 1px solid rgba(168, 85, 247, 0.4); }

    /* Prediction Result Box */
    .prediction-result-box {
        background: linear-gradient(135deg, #091e3a 0%, #163864 100%);
        border: 2px solid #008DDA;
        border-radius: 14px;
        padding: 24px;
        text-align: center;
        box-shadow: 0 12px 30px rgba(0, 141, 218, 0.25);
        margin: 20px 0;
    }
    .prediction-title {
        color: #94A3B8;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        font-weight: 600;
    }
    .prediction-amount {
        color: #FFFFFF;
        font-size: 2.8rem;
        font-weight: 800;
        margin: 8px 0;
        text-shadow: 0 2px 10px rgba(0, 141, 218, 0.5);
    }
    .prediction-ci {
        color: #41C9E2;
        font-size: 0.95rem;
        font-weight: 500;
    }

    /* Algorithm Card */
    .algo-box {
        background: #112238;
        border-radius: 10px;
        border: 1px solid #1E3E62;
        padding: 16px;
        margin-bottom: 12px;
    }
    
    /* Code block tweaks */
    pre, code {
        font-family: 'JetBrains Mono', monospace !important;
    }
    </style>
    """, unsafe_allow_html=True)

def render_header(title: str, subtitle: str, badge: str = "B.Tech Final Project"):
    """Renders the top executive header banner."""
    st.markdown(f"""
    <div class="app-header">
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px;">
            <h1>🚗 {title}</h1>
            <span class="custom-badge badge-primary">{badge}</span>
        </div>
        <p>{subtitle}</p>
    </div>
    """, unsafe_allow_html=True)

def render_section_title(title: str, description: str = ""):
    """Renders a styled section title with optional subtitle."""
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)
    if description:
        st.markdown(f'<div class="section-desc">{description}</div>', unsafe_allow_html=True)

def render_metric_card(title: str, value: str, subtext: str = ""):
    """Renders an automotive KPI metric card."""
    st.markdown(f"""
    <div class="metric-card-container">
        <div class="metric-card-title">{title}</div>
        <div class="metric-card-value">{value}</div>
        {f'<div class="metric-card-sub">{subtext}</div>' if subtext else ''}
    </div>
    """, unsafe_allow_html=True)
