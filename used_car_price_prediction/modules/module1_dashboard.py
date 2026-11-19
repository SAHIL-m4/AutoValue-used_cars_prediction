"""
Module 1: Executive Dashboard & Exploratory Overview.
Presents dataset telemetry, price statistics, inventory distribution, and interactive visual analytics.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from utils.styling import render_section_title, render_metric_card

def render_dashboard_page(df: pd.DataFrame, target_col: str, inspection_report: dict):
    """Renders the primary executive dashboard for used car analytics."""
    st.markdown("""
    <div style="background: linear-gradient(135deg, #102A43 0%, #1E3E62 100%); padding: 20px; border-radius: 12px; border-left: 5px solid #008DDA; margin-bottom: 24px;">
        <h2 style="color: #FFFFFF; margin: 0 0 6px 0; font-size: 1.6rem;">🚗 Executive Automotive Overview</h2>
        <p style="color: #A0C4E2; margin: 0; font-size: 0.95rem;">
            Real-time analytics and valuation telemetry derived directly from the verified dataset.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Top KPI Metrics Row
    total_records = len(df)
    total_features = df.shape[1]
    prices = df[target_col].dropna()
    avg_price = prices.mean() if len(prices) > 0 else 0
    min_price = prices.min() if len(prices) > 0 else 0
    max_price = prices.max() if len(prices) > 0 else 0
    median_price = prices.median() if len(prices) > 0 else 0

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        render_metric_card("Total Records", f"{total_records:,}", "Active Vehicle Listings")
    with col2:
        render_metric_card("Total Features", f"{total_features}", f"{len(inspection_report['numeric_features'])} Num / {len(inspection_report['categorical_features'])} Cat")
    with col3:
        render_metric_card("Average Price", f"${avg_price:,.0f}", f"Median: ${median_price:,.0f}")
    with col4:
        render_metric_card("Lowest Price", f"${min_price:,.0f}", "Dataset Floor")
    with col5:
        render_metric_card("Highest Price", f"${max_price:,.0f}", "Dataset Ceiling")

    st.markdown("<br>", unsafe_allow_html=True)

    # Interactive Dashboard Filters
    with st.expander("🔍 Interactive Data Filtering Options", expanded=False):
        f_col1, f_col2, f_col3 = st.columns(3)
        filtered_df = df.copy()

        # Brand filter if available
        brand_col = next((c for c in df.columns if "brand" in c.lower() or "make" in c.lower()), None)
        if brand_col:
            unique_brands = ["All Brands"] + sorted(df[brand_col].astype(str).unique().tolist())
            selected_brand = f_col1.selectbox("Filter by Brand", unique_brands)
            if selected_brand != "All Brands":
                filtered_df = filtered_df[filtered_df[brand_col].astype(str) == selected_brand]

        # Year filter if available
        year_col = next((c for c in df.columns if "year" in c.lower()), None)
        if year_col and pd.api.types.is_numeric_dtype(df[year_col]):
            min_yr, max_yr = int(df[year_col].min()), int(df[year_col].max())
            if min_yr < max_yr:
                selected_years = f_col2.slider("Model Year Range", min_yr, max_yr, (min_yr, max_yr))
                filtered_df = filtered_df[(filtered_df[year_col] >= selected_years[0]) & (filtered_df[year_col] <= selected_years[1])]

        # Price range filter
        p_min, p_max = float(prices.min()), float(prices.max())
        if p_min < p_max:
            selected_prices = f_col3.slider("Price Filter ($)", int(p_min), int(p_max), (int(p_min), int(p_max)))
            filtered_df = filtered_df[(filtered_df[target_col] >= selected_prices[0]) & (filtered_df[target_col] <= selected_prices[1])]

    # Visualizations Row 1: Price Distribution & Brand Market Share
    render_section_title("Market Valuation & Inventory Distributions")
    chart_col1, chart_col2 = st.columns([1, 1])

    with chart_col1:
        # Histogram of prices
        fig_hist = px.histogram(
            filtered_df,
            x=target_col,
            nbins=35,
            marginal="box",
            title="Distribution of Vehicle Prices with Density Boxplot",
            color_discrete_sequence=["#008DDA"],
            template="plotly_dark"
        )
        fig_hist.update_layout(
            paper_bgcolor="#0B192C",
            plot_bgcolor="#112238",
            margin=dict(l=20, r=20, t=40, b=20),
            xaxis_title="Price ($)",
            yaxis_title="Frequency"
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    with chart_col2:
        # Top Brands inventory & median valuation
        if brand_col:
            brand_stats = filtered_df.groupby(brand_col)[target_col].agg(["count", "median"]).reset_index()
            brand_stats = brand_stats.sort_values(by="count", ascending=False).head(10)

            fig_brands = px.bar(
                brand_stats,
                x=brand_col,
                y="median",
                color="count",
                color_continuous_scale="Blues",
                title="Top 10 Brands: Median Price & Listing Count",
                template="plotly_dark",
                labels={"median": "Median Price ($)", "count": "Inventory Count"}
            )
            fig_brands.update_layout(
                paper_bgcolor="#0B192C",
                plot_bgcolor="#112238",
                margin=dict(l=20, r=20, t=40, b=20),
                xaxis_title="Brand",
                yaxis_title="Median Valuation ($)"
            )
            st.plotly_chart(fig_brands, use_container_width=True)
        else:
            st.info("No categorical brand column found for brand bar chart.")

    # Visualizations Row 2: Depreciation Curves (Year vs Price, Mileage vs Price)
    render_section_title("Depreciation Drivers: Mileage & Age Impact")
    chart_col3, chart_col4 = st.columns([1, 1])

    with chart_col3:
        mileage_col = next((c for c in df.columns if any(m in c.lower() for m in ["milage", "mileage", "odometer", "kms"])), None)
        if mileage_col and pd.api.types.is_numeric_dtype(filtered_df[mileage_col]):
            fig_scatter = px.scatter(
                filtered_df,
                x=mileage_col,
                y=target_col,
                color=brand_col if brand_col else None,
                opacity=0.7,
                title=f"Price vs. {mileage_col.title()} (Depreciation Scatter)",
                template="plotly_dark"
            )
            # Add robust OLS line without depending on statsmodels
            valid_pts = filtered_df[[mileage_col, target_col]].dropna()
            if len(valid_pts) > 1:
                x_pts = valid_pts[mileage_col].values.astype(float)
                y_pts = valid_pts[target_col].values.astype(float)
                fit_coef = np.polyfit(x_pts, y_pts, deg=1)
                x_line = np.linspace(x_pts.min(), x_pts.max(), 50)
                y_line = fit_coef[0] * x_line + fit_coef[1]
                fig_scatter.add_trace(go.Scatter(
                    x=x_line, y=y_line, mode='lines',
                    name='OLS Trendline',
                    line=dict(color='#F1F6F9', width=2.5, dash='dash')
                ))
            fig_scatter.update_layout(
                paper_bgcolor="#0B192C",
                plot_bgcolor="#112238",
                margin=dict(l=20, r=20, t=40, b=20)
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
        else:
            st.info("No mileage column detected for scatter analysis.")

    with chart_col4:
        if year_col and pd.api.types.is_numeric_dtype(filtered_df[year_col]):
            year_trend = filtered_df.groupby(year_col)[target_col].agg(["median", "mean"]).reset_index()
            fig_trend = go.Figure()
            fig_trend.add_trace(go.Scatter(
                x=year_trend[year_col], y=year_trend["median"],
                mode='lines+markers', name='Median Price',
                line=dict(color='#008DDA', width=3)
            ))
            fig_trend.add_trace(go.Scatter(
                x=year_trend[year_col], y=year_trend["mean"],
                mode='lines+markers', name='Mean Price',
                line=dict(color='#41C9E2', width=2, dash='dash')
            ))
            fig_trend.update_layout(
                title="Valuation Trajectory by Model Year",
                xaxis_title="Manufacturing Year",
                yaxis_title="Price ($)",
                template="plotly_dark",
                paper_bgcolor="#0B192C",
                plot_bgcolor="#112238",
                margin=dict(l=20, r=20, t=40, b=20)
            )
            st.plotly_chart(fig_trend, use_container_width=True)
        else:
            st.info("No model year column detected for trend analysis.")

    # Visualizations Row 3: Transmission, Fuel Type, and Accident Breakdown
    render_section_title("Specification Breakdowns: Fuel, Transmission & Title")
    pie_col1, pie_col2 = st.columns(2)

    fuel_col = next((c for c in df.columns if "fuel" in c.lower()), None)
    trans_col = next((c for c in df.columns if "trans" in c.lower()), None)

    with pie_col1:
        if fuel_col:
            fuel_counts = filtered_df[fuel_col].value_counts().reset_index()
            fuel_counts.columns = [fuel_col, "count"]
            fig_fuel = px.pie(
                fuel_counts,
                names=fuel_col,
                values="count",
                hole=0.45,
                title="Fuel Type Distribution",
                template="plotly_dark",
                color_discrete_sequence=px.colors.sequential.Blues_r
            )
            fig_fuel.update_layout(paper_bgcolor="#0B192C", margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_fuel, use_container_width=True)

    with pie_col2:
        if trans_col:
            trans_counts = filtered_df[trans_col].value_counts().reset_index()
            trans_counts.columns = [trans_col, "count"]
            fig_trans = px.pie(
                trans_counts,
                names=trans_col,
                values="count",
                hole=0.45,
                title="Transmission Mix",
                template="plotly_dark",
                color_discrete_sequence=px.colors.sequential.Teal_r
            )
            fig_trans.update_layout(paper_bgcolor="#0B192C", margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_trans, use_container_width=True)

    # Dataset Explorer Table
    render_section_title("Dataset Records Explorer")
    st.dataframe(
        filtered_df.head(100),
        use_container_width=True,
        height=320
    )
    st.caption(f"Showing {min(100, len(filtered_df))} of {len(filtered_df):,} matching vehicle records.")
