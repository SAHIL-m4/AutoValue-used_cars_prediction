"""
Comprehensive automated unit and integration tests for Used Car Price Prediction ML Application.
Tests data loaders, SQLite persistence, HMM algorithms, regression/classification models, clustering, and ensemble learning.
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
import numpy as np
import pandas as pd

# Add project root to sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from modules.data_loader import (
    load_raw_dataset,
    clean_and_inspect_dataframe,
    prepare_modeling_data,
    load_sequential_dataset
)
from modules.database import (
    init_db,
    log_prediction,
    get_prediction_history,
    clear_prediction_history,
    log_model_benchmark
)
from utils.hmm_algorithms import MarkovModel, HiddenMarkovModel
from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, StackingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA

def test_data_loading_and_cleaning():
    """Verify raw dataset loading, schema detection, and missing value cleaning."""
    df_raw = load_raw_dataset()
    assert isinstance(df_raw, pd.DataFrame)
    assert len(df_raw) > 100
    assert "price" in df_raw.columns

    cleaned_df, report = clean_and_inspect_dataframe(df_raw)
    assert cleaned_df.isnull().sum().sum() == 0
    assert report["detected_target_col"] == "price"
    assert len(report["numeric_features"]) >= 2
    assert len(report["categorical_features"]) >= 2

def test_modeling_data_bundle():
    """Verify train-test splitting and encoding."""
    df_raw = load_raw_dataset()
    cleaned_df, report = clean_and_inspect_dataframe(df_raw)
    bundle = prepare_modeling_data(cleaned_df, target_col="price", test_size=0.2, random_state=42)

    assert "X_train" in bundle
    assert "X_test" in bundle
    assert "y_train" in bundle
    assert "y_test" in bundle
    assert "preprocessor" in bundle
    assert len(bundle["X_train"]) > len(bundle["X_test"])
    assert bundle["X_train"].shape[1] == len(bundle["feature_names"])

def test_database_operations():
    """Verify SQLite initialization, insertion, querying, and deletion."""
    init_db()
    clear_prediction_history()

    sample_log = {
        "brand": "Toyota",
        "model": "Camry",
        "model_year": 2021,
        "milage": 28000,
        "fuel_type": "Gasoline",
        "engine": "2.5L I4",
        "transmission": "Automatic",
        "accident": "None reported",
        "clean_title": "Yes",
        "algorithm_used": "Random Forest",
        "predicted_price": 24500.0,
        "lower_bound": 22000.0,
        "upper_bound": 27000.0
    }
    inserted_id = log_prediction(sample_log)
    assert inserted_id is not None
    assert inserted_id > 0

    history = get_prediction_history(10)
    assert not history.empty
    assert history.iloc[0]["brand"] == "Toyota"
    assert history.iloc[0]["predicted_price"] == 24500.0

    clear_prediction_history()
    empty_hist = get_prediction_history(10)
    assert empty_hist.empty

def test_hmm_algorithms():
    """Verify Markov Model and Hidden Markov Model mathematical accuracy."""
    states = ["Good", "Degraded", "Fault"]
    observations = ["Normal", "Warning", "Alarm"]

    A = np.array([
        [0.8, 0.15, 0.05],
        [0.1, 0.7, 0.2],
        [0.3, 0.2, 0.5]
    ])

    B = np.array([
        [0.8, 0.15, 0.05],
        [0.2, 0.6, 0.2],
        [0.05, 0.25, 0.7]
    ])

    pi = np.array([0.7, 0.2, 0.1])

    # 1. Markov Model test
    mm = MarkovModel(states, A, pi)
    A_3 = mm.get_k_step_transition(3)
    assert np.allclose(A_3.sum(axis=1), 1.0)
    steady_state = mm.compute_steady_state()
    assert np.isclose(steady_state.sum(), 1.0)
    assert np.allclose(steady_state @ A, steady_state, atol=1e-3)

    # 2. HMM Forward, Backward, Viterbi
    hmm = HiddenMarkovModel(states, observations, A, B, pi)
    test_seq = ["Normal", "Warning", "Alarm"]

    f_prob, alpha = hmm.forward_algorithm(test_seq)
    b_prob, beta = hmm.backward_algorithm(test_seq)

    assert f_prob > 0.0
    assert b_prob > 0.0
    # Forward and Backward should compute the identical observation probability
    assert np.isclose(f_prob, b_prob, rtol=1e-5)

    v_path, v_prob, v_trellis, _ = hmm.viterbi_algorithm(test_seq)
    assert len(v_path) == len(test_seq)
    assert v_prob > 0.0
    assert v_prob <= f_prob  # Single best path probability <= sum over all paths

    gamma = hmm.compute_state_posteriors(test_seq)
    assert np.allclose(gamma.sum(axis=1), 1.0)

def test_regression_and_classification_models():
    """Verify all Unit 2 supervised learning algorithms train and predict cleanly."""
    df_raw = load_raw_dataset()
    cleaned_df, report = clean_and_inspect_dataframe(df_raw)
    bundle = prepare_modeling_data(cleaned_df, target_col="price", test_size=0.2, random_state=42)

    X_train, y_train = bundle["X_train"], bundle["y_train"]
    X_test, y_test = bundle["X_test"], bundle["y_test"]

    # Linear Regression
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    preds_lr = lr.predict(X_test)
    assert len(preds_lr) == len(y_test)

    # Ridge Regression
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train, y_train)
    preds_ridge = ridge.predict(X_test)
    assert len(preds_ridge) == len(y_test)

    # Logistic Regression
    y_clf_train = (y_train >= np.median(y_train)).astype(int)
    y_clf_test = (y_test >= np.median(y_train)).astype(int)
    clf = LogisticRegression(max_iter=300)
    clf.fit(X_train, y_clf_train)
    preds_clf = clf.predict(X_test)
    acc = (preds_clf == y_clf_test).mean()
    assert acc > 0.60  # Meaningful learning on vehicle data

def test_clustering_and_pca():
    """Verify Unit 3 unsupervised clustering and dimensionality reduction."""
    df_raw = load_raw_dataset()
    cleaned_df, report = clean_and_inspect_dataframe(df_raw)
    num_cols = report["numeric_features"][:3]
    X_num = cleaned_df[num_cols].values

    # K-Means
    km = KMeans(n_clusters=3, random_state=42, n_init=5)
    labels = km.fit_predict(X_num)
    assert len(labels) == len(X_num)
    assert len(set(labels)) == 3

    # GMM
    gmm = GaussianMixture(n_components=3, random_state=42)
    gmm_labels = gmm.fit_predict(X_num)
    assert len(gmm_labels) == len(X_num)

    # PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_num)
    assert X_pca.shape == (len(X_num), 2)
    assert pca.explained_variance_ratio_.sum() <= 1.0

def test_ensemble_learning_models():
    """Verify Unit 5 tree and ensemble regressors."""
    df_raw = load_raw_dataset()
    cleaned_df, report = clean_and_inspect_dataframe(df_raw)
    bundle = prepare_modeling_data(cleaned_df, target_col="price", test_size=0.2, random_state=42)

    X_train, y_train = bundle["X_train"], bundle["y_train"]
    X_test, y_test = bundle["X_test"], bundle["y_test"]

    # CART Decision Tree
    tree = DecisionTreeRegressor(max_depth=5, random_state=42)
    tree.fit(X_train, y_train)
    assert len(tree.predict(X_test)) == len(y_test)

    # Random Forest
    rf = RandomForestRegressor(n_estimators=25, max_depth=6, random_state=42)
    rf.fit(X_train, y_train)
    assert len(rf.predict(X_test)) == len(y_test)
    assert len(rf.feature_importances_) == X_train.shape[1]

    # Gradient Boosting
    gb = GradientBoostingRegressor(n_estimators=30, max_depth=3, random_state=42)
    gb.fit(X_train, y_train)
    assert len(gb.predict(X_test)) == len(y_test)

    # AdaBoost
    ada = AdaBoostRegressor(n_estimators=20, random_state=42)
    ada.fit(X_train, y_train)
    assert len(ada.predict(X_test)) == len(y_test)

if __name__ == "__main__":
    test_data_loading_and_cleaning()
    print("✓ test_data_loading_and_cleaning passed")
    test_modeling_data_bundle()
    print("✓ test_modeling_data_bundle passed")
    test_database_operations()
    print("✓ test_database_operations passed")
    test_hmm_algorithms()
    print("✓ test_hmm_algorithms passed")
    test_regression_and_classification_models()
    print("✓ test_regression_and_classification_models passed")
    test_clustering_and_pca()
    print("✓ test_clustering_and_pca passed")
    test_ensemble_learning_models()
    print("✓ test_ensemble_learning_models passed")
    print("\n🎉 ALL TESTS PASSED SUCCESSFULLY!")
