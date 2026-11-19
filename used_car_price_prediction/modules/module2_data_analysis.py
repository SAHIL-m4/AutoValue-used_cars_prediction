"""
Module 2: Data Analysis & Statistical Foundations (Unit 1).
Includes:
- Descriptive Statistics (mean, median, mode, variance, std dev, skewness, kurtosis)
- Probability Concepts & Bayes' Theorem Demonstrations
- Quantiles, Distributions & Outlier Diagnostics
- Correlation Analysis (Pearson & Spearman)
- Polynomial Curve Fitting (Degrees 1, 2, 3 with SciPy/NumPy)
"""

import streamlit as st
import pandas as pd
import numpy as np
import scipy.stats as stats
import plotly.express as px
import plotly.graph_objects as go
from utils.styling import render_section_title, render_metric_card

def render_data_analysis_page(df: pd.DataFrame, target_col: str, inspection_report: dict):
    """Renders Unit 1 Statistical & Exploratory Analysis."""
    st.markdown("""
    <div style="background: linear-gradient(135deg, #102A43 0%, #1E3E62 100%); padding: 20px; border-radius: 12px; border-left: 5px solid #008DDA; margin-bottom: 24px;">
        <h2 style="color: #FFFFFF; margin: 0 0 6px 0; font-size: 1.6rem;">📊 Unit 1: Statistical Analysis & Probability Modelling</h2>
        <p style="color: #A0C4E2; margin: 0; font-size: 0.95rem;">
            Rigorous descriptive statistics, Bayes' Theorem inference, quantile analysis, correlations, and polynomial curve fitting.
        </p>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Descriptive Statistics",
        "⚖️ Bayes' Theorem & Probability",
        "📉 Quantiles & Distributions",
        "🔥 Correlation Analysis",
        "📐 Polynomial Curve Fitting"
    ])

    numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns]

    # ==========================================
    # TAB 1: Descriptive Statistics
    # ==========================================
    with tab1:
        render_section_title("Parametric & Non-Parametric Summary Statistics")
        st.write("Complete statistical breakdown for all continuous numeric dimensions in the dataset:")

        stats_dict = []
        for col in numeric_cols:
            series = df[col].dropna()
            mode_val = series.mode().iloc[0] if not series.mode().empty else np.nan
            stats_dict.append({
                "Feature": col,
                "Count": int(series.count()),
                "Mean": float(series.mean()),
                "Median": float(series.median()),
                "Mode": float(mode_val),
                "Variance": float(series.var()),
                "Std Dev": float(series.std()),
                "Min": float(series.min()),
                "25% (Q1)": float(series.quantile(0.25)),
                "75% (Q3)": float(series.quantile(0.75)),
                "Max": float(series.max()),
                "Skewness": float(series.skew()),
                "Kurtosis": float(series.kurt())
            })
        stats_df = pd.DataFrame(stats_dict)

        st.dataframe(
            stats_df.style.format({
                "Mean": "{:,.2f}",
                "Median": "{:,.2f}",
                "Mode": "{:,.2f}",
                "Variance": "{:,.2f}",
                "Std Dev": "{:,.2f}",
                "Min": "{:,.2f}",
                "25% (Q1)": "{:,.2f}",
                "75% (Q3)": "{:,.2f}",
                "Max": "{:,.2f}",
                "Skewness": "{:.3f}",
                "Kurtosis": "{:.3f}"
            }),
            use_container_width=True
        )

        st.markdown("<br>", unsafe_allow_html=True)
        sel_col = st.selectbox("Select Feature for In-Depth Statistical Distribution Inspection:", numeric_cols, index=numeric_cols.index(target_col) if target_col in numeric_cols else 0)

        s = df[sel_col].dropna()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Sample Mean", f"{s.mean():,.2f}")
        c2.metric("Median (P50)", f"{s.median():,.2f}")
        c3.metric("Standard Deviation", f"{s.std():,.2f}")
        c4.metric("Skewness (Asymmetry)", f"{s.skew():.3f}", help=">0 indicates right-skewed, <0 indicates left-skewed")

        # Distribution plot with Normal curve overlay
        x_vals = np.linspace(s.min(), s.max(), 200)
        norm_pdf = stats.norm.pdf(x_vals, s.mean(), s.std())

        fig_dist = go.Figure()
        fig_dist.add_trace(go.Histogram(
            x=s, histnorm='probability density', name='Observed Histogram',
            marker_color='#008DDA', opacity=0.75, nbinsx=40
        ))
        fig_dist.add_trace(go.Scatter(
            x=x_vals, y=norm_pdf, mode='lines', name='Theoretical Normal PDF',
            line=dict(color='#F1F6F9', width=2.5, dash='dash')
        ))
        fig_dist.update_layout(
            title=f"Empirical Density vs Theoretical Normal PDF for {sel_col}",
            xaxis_title=sel_col, yaxis_title="Probability Density",
            template="plotly_dark", paper_bgcolor="#0B192C", plot_bgcolor="#112238"
        )
        st.plotly_chart(fig_dist, use_container_width=True)

    # ==========================================
    # TAB 2: Bayes' Theorem Demonstration
    # ==========================================
    with tab2:
        render_section_title("Bayesian Probability & Conditional Reasoning")
        st.markdown(r"""
        Bayes' Theorem updates the prior probability of a hypothesis $H$ given observed evidence $E$:
        $$\mathbf{P(H \mid E) = \frac{P(E \mid H) \cdot P(H)}{P(E)} = \frac{P(E \mid H) \cdot P(H)}{P(E \mid H)P(H) + P(E \mid \neg H)P(\neg H)}}$$
        Here, we formulate an empirical Bayesian hypothesis test on our verified used car dataset:
        - **Hypothesis $H$**: The vehicle is in the **Premium / High Price Tier** ($Price \ge \text{Threshold}$).
        - **Evidence $E$**: An observable attribute (e.g. Brand, Transmission, Fuel Type, or Accident history).
        """)

        b_col1, b_col2 = st.columns(2)
        with b_col1:
            threshold_option = st.radio("Define High Price Hypothesis ($H$):", ["Above Median Price", "Above 75th Percentile (Top Quartile)"], horizontal=True)
            if threshold_option == "Above Median Price":
                threshold_val = float(df[target_col].median())
            else:
                threshold_val = float(df[target_col].quantile(0.75))
            st.caption(f"Price Threshold for $H$: **${threshold_val:,.2f}**")

        with b_col2:
            cat_cols = [c for c in df.select_dtypes(exclude=[np.number]).columns]
            if not cat_cols:
                st.warning("No categorical features found for evidence partitioning.")
                evidence_col = None
            else:
                evidence_col = st.selectbox("Select Evidence Attribute ($E$):", cat_cols)
                evidence_val = st.selectbox(f"Select Specific Condition for {evidence_col}:", df[evidence_col].unique())

        if evidence_col:
            # Empirical Counts
            total_n = len(df)
            is_H = df[target_col] >= threshold_val
            is_E = df[evidence_col] == evidence_val

            count_H = is_H.sum()
            count_not_H = total_n - count_H
            count_E = is_E.sum()
            count_E_given_H = (is_E & is_H).sum()
            count_E_given_not_H = (is_E & (~is_H)).sum()

            # Probabilities
            P_H = count_H / total_n
            P_not_H = 1.0 - P_H
            P_E = count_E / total_n if total_n > 0 else 0
            P_E_given_H = count_E_given_H / count_H if count_H > 0 else 0
            P_E_given_not_H = count_E_given_not_H / count_not_H if count_not_H > 0 else 0

            # Bayes Theorem Calculation
            if P_E > 0:
                P_H_given_E = (P_E_given_H * P_H) / P_E
                bayes_factor = P_H_given_E / P_H if P_H > 0 else 1.0
            else:
                P_H_given_E = 0.0
                bayes_factor = 1.0

            st.markdown("---")
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                render_metric_card("Prior: P(H)", f"{P_H:.1%}", "Base Rate of Premium Cars")
            with m2:
                render_metric_card("Likelihood: P(E|H)", f"{P_E_given_H:.1%}", f"P({evidence_val} | Premium)")
            with m3:
                render_metric_card("Marginal: P(E)", f"{P_E:.1%}", f"Overall Rate of {evidence_val}")
            with m4:
                render_metric_card("Posterior: P(H|E)", f"{P_H_given_E:.1%}", f"Updated Belief ({bayes_factor:.2f}x shift)")

            # Visual Bayes Bar Comparison
            fig_bayes = go.Figure()
            fig_bayes.add_trace(go.Bar(
                name="Probabilities",
                x=["Prior P(H)", "Likelihood P(E|H)", "Marginal Evidence P(E)", "Posterior P(H|E)"],
                y=[P_H, P_E_given_H, P_E, P_H_given_E],
                marker_color=["#41C9E2", "#008DDA", "#94A3B8", "#10B981"],
                text=[f"{P_H:.1%}", f"{P_E_given_H:.1%}", f"{P_E:.1%}", f"{P_H_given_E:.1%}"],
                textposition='auto'
            ))
            fig_bayes.update_layout(
                title=f"Bayesian Probability Shift: Effect of Observing '{evidence_val}'",
                yaxis=dict(title="Probability", range=[0, 1.05]),
                template="plotly_dark", paper_bgcolor="#0B192C", plot_bgcolor="#112238"
            )
            st.plotly_chart(fig_bayes, use_container_width=True)

            st.info(f"""
            💡 **Bayesian Interpretation**: Before checking the car's specifications, any randomly chosen listing has a **{P_H:.1%}** prior probability of being premium. Upon observing that the vehicle has **{evidence_col} = '{evidence_val}'**, our confidence updates to **{P_H_given_E:.1%}** (a **{bayes_factor:.2f}x** change).
            """)

    # ==========================================
    # TAB 3: Quantiles and Distributions
    # ==========================================
    with tab3:
        render_section_title("Quantiles, Empirical CDF & Outlier Diagnostics")
        target_series = df[target_col].dropna()

        # Quantile Table
        quantiles = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
        q_vals = [target_series.quantile(q) for q in quantiles]
        q_df = pd.DataFrame({
            "Percentile": [f"P{int(q*100)}" for q in quantiles],
            "Cumulative Probability": quantiles,
            f"{target_col.title()} ($)": q_vals
        })

        q_col1, q_col2 = st.columns([1, 1])
        with q_col1:
            st.write("**Empirical Quantile Breakdown:**")
            st.dataframe(q_df.style.format({f"{target_col.title()} ($)": "${:,.2f}", "Cumulative Probability": "{:.2f}"}), use_container_width=True)

        with q_col2:
            # Empirical CDF Plot
            sorted_data = np.sort(target_series)
            yvals = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
            fig_ecdf = go.Figure()
            fig_ecdf.add_trace(go.Scatter(x=sorted_data, y=yvals, mode='lines', name='ECDF', line=dict(color='#008DDA', width=3)))
            fig_ecdf.update_layout(
                title=f"Empirical Cumulative Distribution Function (ECDF) of {target_col}",
                xaxis_title="Price ($)", yaxis_title="Cumulative Probability P(X <= x)",
                template="plotly_dark", paper_bgcolor="#0B192C", plot_bgcolor="#112238"
            )
            st.plotly_chart(fig_ecdf, use_container_width=True)

        # Outlier Detection via IQR and Z-Score
        render_section_title("Outlier Identification & Boundary Analysis")
        Q1 = target_series.quantile(0.25)
        Q3 = target_series.quantile(0.75)
        IQR = Q3 - Q1
        lower_iqr = Q1 - 1.5 * IQR
        upper_iqr = Q3 + 1.5 * IQR
        iqr_outliers = target_series[(target_series < lower_iqr) | (target_series > upper_iqr)]

        z_scores = np.abs(stats.zscore(target_series))
        z_outliers = target_series[z_scores > 3.0]

        o1, o2, o3, o4 = st.columns(4)
        o1.metric("IQR Lower Bound", f"${max(0, lower_iqr):,.0f}")
        o2.metric("IQR Upper Bound", f"${upper_iqr:,.0f}")
        o3.metric("IQR Outlier Count", f"{len(iqr_outliers)} ({len(iqr_outliers)/len(target_series):.1%})")
        o4.metric("Z-Score Outliers (|Z|>3)", f"{len(z_outliers)} ({len(z_outliers)/len(target_series):.1%})")

        # Box & Violin Plots
        fig_box = px.violin(
            df, y=target_col, x=inspection_report["categorical_features"][0] if inspection_report["categorical_features"] else None,
            box=True, points="outliers", color_discrete_sequence=["#41C9E2"],
            title=f"Box-Violin Hybrid Distribution of {target_col}",
            template="plotly_dark"
        )
        fig_box.update_layout(paper_bgcolor="#0B192C", plot_bgcolor="#112238")
        st.plotly_chart(fig_box, use_container_width=True)

    # ==========================================
    # TAB 4: Correlation Analysis
    # ==========================================
    with tab4:
        render_section_title("Multivariate Correlation Structures")
        corr_method = st.radio("Correlation Metric:", ["Pearson (Linear Relationship)", "Spearman (Monotonic Rank Correlation)"], horizontal=True)
        method_str = "pearson" if "Pearson" in corr_method else "spearman"

        corr_matrix = df[numeric_cols].corr(method=method_str)

        fig_corr = px.imshow(
            corr_matrix,
            text_auto=".2f",
            color_continuous_scale="RdBu_r",
            zmin=-1, zmax=1,
            title=f"{corr_method.split()[0]} Correlation Matrix Heatmap",
            template="plotly_dark",
            aspect="auto"
        )
        fig_corr.update_layout(paper_bgcolor="#0B192C", plot_bgcolor="#112238")
        st.plotly_chart(fig_corr, use_container_width=True)

        # Significant Correlation ranking with Target
        st.write(f"**Correlation of Numerical Predictors with Target ({target_col}):**")
        target_corr = corr_matrix[target_col].drop(target_col).sort_values(ascending=False).reset_index()
        target_corr.columns = ["Feature", f"{method_str.title()} Correlation"]
        st.dataframe(target_corr.style.format({f"{method_str.title()} Correlation": "{:+.3f}"}), use_container_width=True)

    # ==========================================
    # TAB 5: Polynomial Curve Fitting
    # ==========================================
    with tab5:
        render_section_title("Polynomial Regression Curve Fitting")
        st.markdown(r"""
        Fit parametric polynomial curves to examine non-linear relationships:
        - **Linear (Degree 1)**: $y = w_1 x + w_0$
        - **Quadratic (Degree 2)**: $y = w_2 x^2 + w_1 x + w_0$
        - **Cubic (Degree 3)**: $y = w_3 x^3 + w_2 x^2 + w_1 x + w_0$
        """)

        non_target_num = [c for c in numeric_cols if c != target_col]
        if not non_target_num:
            st.warning("Need at least one numerical predictor for curve fitting.")
        else:
            poly_x_col = st.selectbox("Select Independent Variable (X):", non_target_num)

            x_data = df[poly_x_col].values.astype(float)
            y_data = df[target_col].values.astype(float)

            # Fit degrees 1, 2, 3
            fit_deg1 = np.polyfit(x_data, y_data, deg=1)
            fit_deg2 = np.polyfit(x_data, y_data, deg=2)
            fit_deg3 = np.polyfit(x_data, y_data, deg=3)

            p1 = np.poly1d(fit_deg1)
            p2 = np.poly1d(fit_deg2)
            p3 = np.poly1d(fit_deg3)

            x_plot = np.linspace(x_data.min(), x_data.max(), 300)

            # Compute R2 and RMSE
            def calc_metrics(y_true, y_pred):
                ss_res = np.sum((y_true - y_pred) ** 2)
                ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
                r2 = 1 - (ss_res / ss_tot)
                rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
                return r2, rmse

            r2_1, rmse_1 = calc_metrics(y_data, p1(x_data))
            r2_2, rmse_2 = calc_metrics(y_data, p2(x_data))
            r2_3, rmse_3 = calc_metrics(y_data, p3(x_data))

            p_m1, p_m2, p_m3 = st.columns(3)
            with p_m1:
                st.markdown(f"**Degree 1 (Linear)**<br>R²: `{r2_1:.3f}` | RMSE: `${rmse_1:,.0f}`", unsafe_allow_html=True)
                st.code(f"y = {fit_deg1[0]:.2f}x + {fit_deg1[1]:.2f}")
            with p_m2:
                st.markdown(f"**Degree 2 (Quadratic)**<br>R²: `{r2_2:.3f}` | RMSE: `${rmse_2:,.0f}`", unsafe_allow_html=True)
                st.code(f"y = {fit_deg2[0]:.3e}x² + {fit_deg2[1]:.2f}x + {fit_deg2[2]:.2f}")
            with p_m3:
                st.markdown(f"**Degree 3 (Cubic)**<br>R²: `{r2_3:.3f}` | RMSE: `${rmse_3:,.0f}`", unsafe_allow_html=True)
                st.code(f"y = {fit_deg3[0]:.3e}x³ + {fit_deg3[1]:.3e}x² + {fit_deg3[2]:.2f}x + {fit_deg3[3]:.2f}")

            # Plot curves
            fig_poly = go.Figure()
            fig_poly.add_trace(go.Scatter(x=x_data, y=y_data, mode='markers', name='Actual Data Points', marker=dict(color='#94A3B8', size=5, opacity=0.4)))
            fig_poly.add_trace(go.Scatter(x=x_plot, y=p1(x_plot), mode='lines', name=f'Degree 1 Fit (R²={r2_1:.3f})', line=dict(color='#008DDA', width=3)))
            fig_poly.add_trace(go.Scatter(x=x_plot, y=p2(x_plot), mode='lines', name=f'Degree 2 Fit (R²={r2_2:.3f})', line=dict(color='#F59E0B', width=3)))
            fig_poly.add_trace(go.Scatter(x=x_plot, y=p3(x_plot), mode='lines', name=f'Degree 3 Fit (R²={r2_3:.3f})', line=dict(color='#10B981', width=3, dash='dot')))

            fig_poly.update_layout(
                title=f"Polynomial Curve Fits: {target_col} vs {poly_x_col}",
                xaxis_title=poly_x_col, yaxis_title=target_col,
                template="plotly_dark", paper_bgcolor="#0B192C", plot_bgcolor="#112238"
            )
            st.plotly_chart(fig_poly, use_container_width=True)
