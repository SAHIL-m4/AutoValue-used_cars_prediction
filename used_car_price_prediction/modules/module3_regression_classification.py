"""
Module 3: Regression and Classification (Unit 2).
Implements:
- Linear Regression (OLS)
- Polynomial Regression (Degree 2/3)
- Ridge Regression (L2 Regularized)
- Support Vector Regression (SVR)
- Logistic Regression (Multi-class / Binary Price Tier Classification)
- Evaluation Metrics: MAE, MSE, RMSE, R²
- Model Comparison Benchmarks, Confusion Matrices, and Residual Analysis
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc

from utils.styling import render_section_title, render_metric_card
from modules.database import log_model_benchmark

def render_regression_classification_page(data_bundle: dict):
    """Renders Unit 2 Regression and Classification Suite."""
    st.markdown("""
    <div style="background: linear-gradient(135deg, #102A43 0%, #1E3E62 100%); padding: 20px; border-radius: 12px; border-left: 5px solid #008DDA; margin-bottom: 24px;">
        <h2 style="color: #FFFFFF; margin: 0 0 6px 0; font-size: 1.6rem;">⚡ Unit 2: Supervised Regression & Classification</h2>
        <p style="color: #A0C4E2; margin: 0; font-size: 0.95rem;">
            Comparative benchmarking of classical linear, regularized, kernel, and logistic classifiers on verified vehicle data.
        </p>
    </div>
    """, unsafe_allow_html=True)

    X_train = data_bundle["X_train"]
    X_test = data_bundle["X_test"]
    y_train = data_bundle["y_train"]
    y_test = data_bundle["y_test"]
    feature_names = data_bundle["feature_names"]
    target_col = data_bundle["target_col"]

    tab_reg, tab_clf, tab_comp = st.tabs([
        "📈 Regression Models",
        "🎯 Logistic Classification",
        "⚖️ Benchmark Comparison"
    ])

    # Store models in session_state for reuse and comparison
    if "reg_models" not in st.session_state:
        st.session_state["reg_models"] = {}

    # ==========================================
    # TAB 1: Regression Models
    # ==========================================
    with tab_reg:
        render_section_title("Parametric & Kernel Regression Estimators")
        
        reg_choice = st.selectbox(
            "Select Regression Algorithm to Train & Inspect:",
            ["Linear Regression (OLS)", "Polynomial Regression", "Ridge Regression (L2)", "Support Vector Regression (SVR)"]
        )

        h_col1, h_col2 = st.columns(2)
        model = None
        model_key = ""

        if reg_choice == "Linear Regression (OLS)":
            model_key = "Linear Regression"
            with h_col1:
                fit_intercept = st.checkbox("Fit Intercept Term", value=True)
            model = LinearRegression(fit_intercept=fit_intercept)

        elif reg_choice == "Polynomial Regression":
            model_key = "Polynomial Regression"
            with h_col1:
                poly_degree = st.slider("Polynomial Degree", 2, 3, 2, help="Degree 2 creates quadratic and interaction terms.")
            with h_col2:
                include_bias = st.checkbox("Include Bias Term in Features", value=False)
            model = Pipeline([
                ("poly", PolynomialFeatures(degree=poly_degree, include_bias=include_bias)),
                ("linear", LinearRegression())
            ])

        elif reg_choice == "Ridge Regression (L2)":
            model_key = "Ridge Regression"
            with h_col1:
                alpha = st.slider("L2 Regularization Strength (Alpha)", 0.01, 100.0, 1.0, step=0.1)
            with h_col2:
                solver = st.selectbox("Optimization Solver", ["auto", "svd", "cholesky", "lsqr", "sag"])
            model = Ridge(alpha=alpha, solver=solver, random_state=42)

        elif reg_choice == "Support Vector Regression (SVR)":
            model_key = "SVR (RBF Kernel)"
            with h_col1:
                c_val = st.slider("Regularization Parameter (C)", 1.0, 5000.0, 1000.0, step=100.0)
            with h_col2:
                epsilon = st.slider("Epsilon Margin Tube (ε)", 0.01, 1.0, 0.1, step=0.05)
            model = SVR(kernel="rbf", C=c_val, epsilon=epsilon)

        # Train model
        with st.spinner(f"Training {model_key}..."):
            model.fit(X_train, y_train)
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)

            # Metrics
            train_r2 = r2_score(y_train, y_pred_train)
            test_r2 = r2_score(y_test, y_pred_test)
            mae = mean_absolute_error(y_test, y_pred_test)
            mse = mean_squared_error(y_test, y_pred_test)
            rmse = np.sqrt(mse)

            st.session_state["reg_models"][model_key] = {
                "model": model,
                "train_r2": train_r2,
                "test_r2": test_r2,
                "mae": mae,
                "mse": mse,
                "rmse": rmse,
                "y_pred": y_pred_test
            }
            log_model_benchmark("Unit 2", model_key, test_r2, mae, rmse, mse)

        # Metrics display
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            render_metric_card("Test R² Score", f"{test_r2:.3f}", f"Train R²: {train_r2:.3f}")
        with m2:
            render_metric_card("Mean Abs Error (MAE)", f"${mae:,.0f}", "Average Prediction Error")
        with m3:
            render_metric_card("Root Mean Sq Error", f"${rmse:,.0f}", "Penalty on Large Deviations")
        with m4:
            render_metric_card("Mean Squared Error", f"{mse:,.0f}", "Variance of Residuals")

        # Visualizations: Actual vs Predicted & Residuals
        p_col1, p_col2 = st.columns(2)
        with p_col1:
            fig_act_pred = go.Figure()
            fig_act_pred.add_trace(go.Scatter(
                x=y_test, y=y_pred_test, mode='markers',
                marker=dict(color='#008DDA', opacity=0.6, size=6),
                name='Predictions'
            ))
            # Ideal 45-degree line
            min_val = min(y_test.min(), y_pred_test.min())
            max_val = max(y_test.max(), y_pred_test.max())
            fig_act_pred.add_trace(go.Scatter(
                x=[min_val, max_val], y=[min_val, max_val], mode='lines',
                line=dict(color='#F1F6F9', dash='dash', width=2),
                name='Ideal Match (y = x)'
            ))
            fig_act_pred.update_layout(
                title=f"{model_key}: Actual vs Predicted Valuations",
                xaxis_title="True Vehicle Price ($)", yaxis_title="Predicted Price ($)",
                template="plotly_dark", paper_bgcolor="#0B192C", plot_bgcolor="#112238"
            )
            st.plotly_chart(fig_act_pred, use_container_width=True)

        with p_col2:
            residuals = y_test - y_pred_test
            fig_resid = px.histogram(
                residuals, nbins=35,
                title=f"{model_key}: Residual Error Distribution (y - ŷ)",
                color_discrete_sequence=["#41C9E2"],
                template="plotly_dark"
            )
            fig_resid.update_layout(
                paper_bgcolor="#0B192C", plot_bgcolor="#112238",
                xaxis_title="Residual Error ($)", yaxis_title="Count"
            )
            st.plotly_chart(fig_resid, use_container_width=True)

        # Feature coefficients for Linear / Ridge
        if hasattr(model, "coef_"):
            render_section_title("Learned Feature Coefficients (Weights)")
            coefs = model.coef_
            if len(coefs) == len(feature_names):
                coef_df = pd.DataFrame({"Feature": feature_names, "Coefficient": coefs})
                coef_df["Abs_Weight"] = coef_df["Coefficient"].abs()
                top_coefs = coef_df.sort_values(by="Abs_Weight", ascending=False).head(12)

                fig_coef = px.bar(
                    top_coefs, x="Coefficient", y="Feature", orientation="h",
                    color="Coefficient", color_continuous_scale="RdBu",
                    title="Top Feature Weights Impacting Vehicle Price",
                    template="plotly_dark"
                )
                fig_coef.update_layout(paper_bgcolor="#0B192C", plot_bgcolor="#112238")
                st.plotly_chart(fig_coef, use_container_width=True)

    # ==========================================
    # TAB 2: Logistic Regression (Classification)
    # ==========================================
    with tab_clf:
        render_section_title("Logistic Regression for Vehicle Price-Tier Classification")
        st.markdown(r"""
        Converts the continuous price into categorical valuation tiers to demonstrate Logistic Regression:
        - **Binary Classification**: Budget / Standard vs. Premium ($P \ge \text{Median}$)
        - **Multi-Class Classification**: Budget (P < 33%), Mid-Range (33% ≤ P < 66%), Luxury (P ≥ 66%)
        """)

        clf_mode = st.radio("Classification Problem Formulation:", ["Binary (Budget vs. Premium)", "Multi-Class (Budget vs. Mid-Range vs. Luxury)"], horizontal=True)

        # Prepare categorical labels
        if "Binary" in clf_mode:
            median_p = np.median(y_train)
            y_train_clf = (y_train >= median_p).astype(int)
            y_test_clf = (y_test >= median_p).astype(int)
            class_labels = ["Budget / Standard", "Premium Tier"]
        else:
            q33, q66 = np.quantile(y_train, [0.33, 0.66])
            def discretize(y_arr):
                res = np.zeros(len(y_arr), dtype=int)
                res[(y_arr >= q33) & (y_arr < q66)] = 1
                res[y_arr >= q66] = 2
                return res
            y_train_clf = discretize(y_train)
            y_test_clf = discretize(y_test)
            class_labels = ["Budget", "Mid-Range", "Luxury"]

        c_col1, c_col2 = st.columns(2)
        with c_col1:
            c_inv = st.slider("Inverse Regularization Strength (C)", 0.01, 10.0, 1.0, step=0.1)
        with c_col2:
            max_iter = st.slider("Max Iterations", 100, 1000, 300, step=50)

        # Fit Logistic Regression
        clf_model = LogisticRegression(C=c_inv, max_iter=max_iter, random_state=42)
        clf_model.fit(X_train, y_train_clf)
        y_pred_clf = clf_model.predict(X_test)
        y_proba_clf = clf_model.predict_proba(X_test)

        acc = (y_pred_clf == y_test_clf).mean()
        cm = confusion_matrix(y_test_clf, y_pred_clf)

        cm_col1, cm_col2 = st.columns([1, 1])
        with cm_col1:
            st.metric("Test Classification Accuracy", f"{acc:.1%}")
            # Confusion matrix heatmap
            fig_cm = px.imshow(
                cm, text_auto=True,
                x=class_labels, y=class_labels,
                color_continuous_scale="Blues",
                title="Confusion Matrix Heatmap",
                template="plotly_dark",
                labels=dict(x="Predicted Class", y="Actual Class")
            )
            fig_cm.update_layout(paper_bgcolor="#0B192C", plot_bgcolor="#112238")
            st.plotly_chart(fig_cm, use_container_width=True)

        with cm_col2:
            st.write("**Classification Report (Precision, Recall, F1-Score):**")
            report_dict = classification_report(y_test_clf, y_pred_clf, target_names=class_labels, output_dict=True)
            report_df = pd.DataFrame(report_dict).transpose()
            st.dataframe(report_df.style.format("{:.3f}"), use_container_width=True)

            # ROC Curve for Binary mode
            if "Binary" in clf_mode:
                fpr, tpr, _ = roc_curve(y_test_clf, y_proba_clf[:, 1])
                roc_auc = auc(fpr, tpr)
                fig_roc = go.Figure()
                fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'ROC Curve (AUC = {roc_auc:.3f})', line=dict(color='#008DDA', width=3)))
                fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', line=dict(color='#94A3B8', dash='dash'), name='Random Chance'))
                fig_roc.update_layout(
                    title=f"Receiver Operating Characteristic (ROC): AUC = {roc_auc:.3f}",
                    xaxis_title="False Positive Rate", yaxis_title="True Positive Rate",
                    template="plotly_dark", paper_bgcolor="#0B192C", plot_bgcolor="#112238"
                )
                st.plotly_chart(fig_roc, use_container_width=True)

    # ==========================================
    # TAB 3: Comparison Benchmark
    # ==========================================
    with tab_comp:
        render_section_title("Cross-Model Regression Benchmark Suite")
        st.write("Ensure you have trained algorithms in the first tab to compare them side-by-side:")

        # Automatically train any missing regression models to allow instant full comparison
        algorithms = {
            "Linear Regression": LinearRegression(),
            "Polynomial Reg (Deg 2)": Pipeline([("poly", PolynomialFeatures(degree=2, include_bias=False)), ("linear", LinearRegression())]),
            "Ridge Regression": Ridge(alpha=1.0, random_state=42),
            "SVR (RBF Kernel)": SVR(C=1000.0, epsilon=0.1)
        }

        bench_records = []
        for name, alg in algorithms.items():
            if name not in st.session_state["reg_models"]:
                alg.fit(X_train, y_train)
                y_pred = alg.predict(X_test)
                st.session_state["reg_models"][name] = {
                    "model": alg,
                    "train_r2": r2_score(y_train, alg.predict(X_train)),
                    "test_r2": r2_score(y_test, y_pred),
                    "mae": mean_absolute_error(y_test, y_pred),
                    "mse": mean_squared_error(y_test, y_pred),
                    "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
                    "y_pred": y_pred
                }

            entry = st.session_state["reg_models"][name]
            bench_records.append({
                "Model": name,
                "Test R² Score": entry["test_r2"],
                "Train R² Score": entry["train_r2"],
                "MAE ($)": entry["mae"],
                "RMSE ($)": entry["rmse"]
            })

        bench_df = pd.DataFrame(bench_records).sort_values(by="Test R² Score", ascending=False)
        st.dataframe(bench_df.style.format({
            "Test R² Score": "{:.3f}",
            "Train R² Score": "{:.3f}",
            "MAE ($)": "${:,.0f}",
            "RMSE ($)": "${:,.0f}"
        }), use_container_width=True)

        # Bar chart comparison
        comp_col1, comp_col2 = st.columns(2)
        with comp_col1:
            fig_r2_comp = px.bar(
                bench_df, x="Model", y="Test R² Score",
                color="Test R² Score", color_continuous_scale="Blues",
                title="Model Generalization: Test R² Benchmark",
                template="plotly_dark"
            )
            fig_r2_comp.update_layout(paper_bgcolor="#0B192C", plot_bgcolor="#112238")
            st.plotly_chart(fig_r2_comp, use_container_width=True)

        with comp_col2:
            fig_mae_comp = px.bar(
                bench_df, x="Model", y="MAE ($)",
                color="MAE ($)", color_continuous_scale="Reds_r",
                title="Mean Absolute Error Comparison (Lower is Better)",
                template="plotly_dark"
            )
            fig_mae_comp.update_layout(paper_bgcolor="#0B192C", plot_bgcolor="#112238")
            st.plotly_chart(fig_mae_comp, use_container_width=True)
