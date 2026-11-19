"""
AutoValuate ML: Used Car Price Prediction & Machine Learning Application.
B.Tech Final Year Machine Learning Project.
Main Application Entry Point and Page Router.
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

from utils.styling import apply_custom_styles, render_header
from modules.database import init_db
from modules.data_loader import (
    load_raw_dataset,
    clean_and_inspect_dataframe,
    prepare_modeling_data,
    DEFAULT_DATA_PATH
)
from modules.module1_dashboard import render_dashboard_page
from modules.module2_data_analysis import render_data_analysis_page
from modules.module3_regression_classification import render_regression_classification_page
from modules.module4_clustering_pca import render_clustering_pca_page
from modules.module5_sequential_models import render_sequential_models_page
from modules.module6_ensemble_learning import render_ensemble_learning_page
from modules.module7_price_prediction import render_price_prediction_page

# Set Streamlit Page Configuration
st.set_page_config(
    page_title="AutoValuate ML | Used Car Price Prediction",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply Custom Theme CSS
apply_custom_styles()

# Initialize SQLite database
init_db()

# Sidebar: Project Branding & Dataset Management
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 12px 0 18px 0; border-bottom: 1px solid rgba(0, 141, 218, 0.3);">
        <div style="font-size: 2.5rem; line-height: 1;">🚗</div>
        <div style="color: #FFFFFF; font-size: 1.35rem; font-weight: 800; letter-spacing: -0.5px; margin-top: 6px;">
            AutoValuate ML
        </div>
        <div style="color: #41C9E2; font-size: 0.78rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.08em;">
            Used Car Price Prediction
        </div>
        <div style="display: inline-block; background: rgba(0, 141, 218, 0.2); color: #A0C4E2; font-size: 0.72rem; padding: 3px 8px; border-radius: 4px; margin-top: 6px; border: 1px solid rgba(0, 141, 218, 0.4);">
            B.Tech Final Year Project
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 📂 Dataset Management")
    uploaded_file = st.file_uploader(
        "Upload Custom Car CSV Dataset",
        type=["csv"],
        help="Upload your original used-car dataset CSV here. If omitted, the verified reference dataset will be utilized."
    )

    if uploaded_file is not None:
        st.success(f"Loaded: `{uploaded_file.name}`")
        dataset_source = f"Uploaded ({uploaded_file.name})"
    else:
        st.info("Using verified reference automotive dataset (1,500 records).")
        dataset_source = "Verified Reference (used_cars_sample.csv)"

    st.markdown("---")
    st.markdown("### 🧭 Application Modules")

    nav_selection = st.radio(
        "Navigate to Module:",
        [
            "📊 Module 1: Executive Dashboard",
            "📈 Module 2: Data Analysis (Unit 1)",
            "⚡ Module 3: Regression & Classification (Unit 2)",
            "🔬 Module 4: Clustering & PCA (Unit 3)",
            "⏱️ Module 5: Sequential Models (Unit 4 - HMM)",
            "🌲 Module 6: Advanced & Ensembles (Unit 5)",
            "🎯 Module 7: Used Car Price Predictor",
            "📖 Project Guide & Review Manual"
        ],
        index=0
    )

    st.markdown("---")
    st.markdown("""
    <div style="font-size: 0.75rem; color: #94A3B8; padding: 4px 0;">
        <strong>Frameworks:</strong> Scikit-Learn • Streamlit • SciPy • Plotly<br>
        <strong>Database:</strong> SQLite3 Persistent Storage<br>
        <strong>Compatibility:</strong> Python 3.9+ macOS
    </div>
    """, unsafe_allow_html=True)

# Data Loading & Preprocessing Pipeline
@st.cache_data(show_spinner="Cleaning, encoding and preprocessing dataset...")
def get_processed_data(file_bytes, filename):
    """Loads and caches cleaned data and model partitions."""
    if file_bytes is not None:
        import io
        raw_df = pd.read_csv(io.BytesIO(file_bytes))
    else:
        raw_df = load_raw_dataset()

    cleaned_df, report = clean_and_inspect_dataframe(raw_df)
    target_col = report["detected_target_col"]
    bundle = prepare_modeling_data(cleaned_df, target_col=target_col)
    return cleaned_df, report, bundle

# Execute pipeline
try:
    file_bytes = uploaded_file.getvalue() if uploaded_file is not None else None
    filename = uploaded_file.name if uploaded_file is not None else "default"
    df, inspection_report, data_bundle = get_processed_data(file_bytes, filename)
    target_col = inspection_report["detected_target_col"]
except Exception as e:
    st.error(f"Error processing dataset: {str(e)}")
    st.stop()

# Page Routing
if "Module 1" in nav_selection:
    render_header("Used Car Price Prediction System", "Module 1: High-Level Executive Dashboard & Dataset Summary", badge="Module 1")
    render_dashboard_page(df, target_col, inspection_report)

elif "Module 2" in nav_selection:
    render_header("Descriptive Statistics & Probability Theory", "Module 2 (Unit 1): Parameter Distributions, Bayes' Rule, and Non-Linear Curve Fitting", badge="Unit 1")
    render_data_analysis_page(df, target_col, inspection_report)

elif "Module 3" in nav_selection:
    render_header("Supervised Regression & Classification", "Module 3 (Unit 2): Linear, Ridge, SVR, Logistic Classifiers, and Multi-Metric Evaluations", badge="Unit 2")
    render_regression_classification_page(data_bundle)

elif "Module 4" in nav_selection:
    render_header("Clustering & Dimensionality Reduction", "Module 4 (Unit 3): Market Tier Discovery via K-Means, GMM, Hierarchical Clustering, and PCA", badge="Unit 3")
    render_clustering_pca_page(df, target_col, inspection_report)

elif "Module 5" in nav_selection:
    render_header("Sequential Data Modelling & Markov Processes", "Module 5 (Unit 4): Discrete Markov Chains, Transition Matrices, Forward, Backward, and Viterbi HMM Algorithms", badge="Unit 4")
    render_sequential_models_page()

elif "Module 6" in nav_selection:
    render_header("Advanced Ensembles & Non-Linear Trees", "Module 6 (Unit 5): CART, Bagging, Boosting, Stacking, Feature Importance, and Bayesian Model Averaging (BMA)", badge="Unit 5")
    render_ensemble_learning_page(data_bundle)

elif "Module 7" in nav_selection:
    render_header("Interactive Used Car Valuation Calculator", "Dedicated Used Car Price Prediction Engine with 95% Confidence Intervals & SQLite Persistence", badge="Price Predictor")
    render_price_prediction_page(data_bundle)

elif "Project Guide" in nav_selection:
    render_header("Project Review Manual & Viva Defense Guide", "Comprehensive Guide for Faculty Project Demonstrations & Technical Viva Defense", badge="Documentation")

    st.markdown("""
    ### 🎓 College Project Review & Viva Defense Guide

    This application was architected as an end-to-end, full-stack Machine Learning project covering the complete B.Tech Machine Learning curriculum (Units 1 through 5).

    ---

    #### 📋 Module-by-Module Viva Talking Points

    ##### 1. Executive Dashboard (Module 1)
    - **Demonstrate**: Total vehicle records, average, minimum, and maximum prices, price distribution histogram, and brand market share.
    - **Key Defense Concept**: Data distribution skewness, central tendency metrics (median vs mean for right-skewed prices), and exploratory data analysis (EDA).

    ##### 2. Data Analysis (Unit 1 - Module 2)
    - **Demonstrate**:
      1. Descriptive statistics table (Mean, Variance, Std Dev, Kurtosis, Skewness).
      2. **Bayes' Theorem Simulation**: Prior $P(H)$, Likelihood $P(E|H)$, Marginal $P(E)$, and Posterior $P(H|E)$ calculation.
      3. Quantiles, Empirical CDF, and Outlier detection via IQR and Z-Scores ($|Z| > 3$).
      4. Pearson & Spearman correlation heatmaps.
      5. Polynomial curve fitting ($y = w_1 x + w_0$, $y = w_2 x^2 + w_1 x + w_0$, $y = w_3 x^3 + w_2 x^2 + w_1 x + w_0$) with $R^2$ and RMSE.

    ##### 3. Regression and Classification (Unit 2 - Module 3)
    - **Demonstrate**:
      1. Training Linear Regression (OLS), Polynomial Regression, Ridge Regression (L2 regularization $\alpha$), and Support Vector Regression (SVR with RBF kernel).
      2. Evaluation metrics: MAE, MSE, RMSE, $R^2$.
      3. Actual vs. Predicted plots and Residual error distribution plots.
      4. **Logistic Regression**: Price tier classification into Budget vs. Premium, showing Confusion Matrix, Precision, Recall, F1-Score, and ROC-AUC curve.

    ##### 4. Clustering and Dimensionality Reduction (Unit 3 - Module 4)
    - **Demonstrate**:
      1. **K-Means Clustering**: Unsupervised market tier discovery with centroid profiling.
      2. **Gaussian Mixture Models (GMM)**: Soft probabilistic clustering with BIC/AIC scores.
      3. **Hierarchical Agglomerative Clustering**: Dendrogram visualization.
      4. **Principal Component Analysis (PCA)**: Scree plot, explained variance ratios, 2D and 3D PCA projections.
      5. **Diagnostic Validation**: The Elbow Method (WCSS curve) and Silhouette Score comparison.

    ##### 5. Sequential Data Modelling (Unit 4 - Module 5)
    - **Crucial Theoretical Defense**: Why cross-sectional tabular car datasets are **not** sequential data (violates Markov assumption $P(S_t \mid S_{t-1}, \dots) = P(S_t \mid S_{t-1})$).
    - **Demonstrate**:
      1. Dedicated vehicle degradation telemetry sequence (Normal $\\to$ Minor Wear $\\to$ Critical Fault).
      2. Discrete Markov Chain Transition Matrix $A$, stationary distribution $\pi_{\infty}$, and multi-step transition $A^k$.
      3. **Hidden Markov Model (HMM)** parameters $\lambda = (A, B, \pi)$.
      4. **Forward Algorithm**: Solves the Evaluation Problem ($\alpha_t(i)$ trellis).
      5. **Backward Algorithm**: Backward recursion ($\beta_t(i)$ trellis).
      6. **Viterbi Algorithm**: Solves the Decoding Problem to find the most probable hidden state path $Q^*$.
      7. Interactive sequence tester with smoothed state posteriors $\gamma_t(i)$.

    ##### 6. Advanced Models & Ensemble Learning (Unit 5 - Module 6)
    - **Demonstrate**:
      1. CART Decision Tree, Random Forest (Bagging), Gradient Boosting (GBDT), AdaBoost, and Stacking Regressor.
      2. Ensemble Leaderboard comparing Test $R^2$, MAE, and RMSE.
      3. Feature Importance (MDI / Gini) identifying key price drivers (e.g., Year, Mileage, Brand).
      4. **Bayesian Model Averaging (BMA)**: Weighting hypotheses by posterior probabilities using BIC.

    ##### 7. Used Car Price Predictor (Module 7)
    - **Demonstrate**:
      1. Interactive form with Brand, Model, Year, Mileage, Fuel, Transmission, Accident history.
      2. Clicking "Predict Fair Market Price" returns fair market estimate, 95% confidence interval, comparison with brand/year averages, and automatic logging to SQLite database.
      3. Historical prediction log viewer with CSV download and deletion options.
    """)
