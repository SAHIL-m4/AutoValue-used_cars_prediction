"""
Data Loader and Preprocessing Engine.
Performs dynamic dataset inspection, schema detection, cleaning, encoding, and train-test splitting.
Supports both user-uploaded CSVs and the built-in reference dataset.
"""

import os
import re
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, List, Optional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer

DEFAULT_DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "used_cars_sample.csv")
SEQUENTIAL_DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "sequential_maintenance_sample.csv")

def load_raw_dataset(uploaded_file=None, custom_path=None) -> pd.DataFrame:
    """Loads dataset from uploaded file object or local CSV path."""
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            return df
        except Exception as e:
            raise ValueError(f"Error reading uploaded CSV: {str(e)}")
    elif custom_path and os.path.exists(custom_path):
        return pd.read_csv(custom_path)
    elif os.path.exists(DEFAULT_DATA_PATH):
        return pd.read_csv(DEFAULT_DATA_PATH)
    else:
        raise FileNotFoundError("No dataset found. Please upload a CSV file.")

def load_sequential_dataset(uploaded_file=None) -> pd.DataFrame:
    """Loads sequential dataset for HMM demonstrations."""
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file)
    elif os.path.exists(SEQUENTIAL_DATA_PATH):
        return pd.read_csv(SEQUENTIAL_DATA_PATH)
    else:
        raise FileNotFoundError("Sequential sample dataset not found.")

def clean_currency_or_numeric_string(val) -> Optional[float]:
    """Cleans numeric strings with commas, currency symbols, or units like mi, km."""
    if pd.isna(val):
        return np.nan
    if isinstance(val, (int, float)):
        return float(val)
    val_str = str(val).strip()
    # Remove currency symbols and formatting
    cleaned = re.sub(r'[\$,₹,€,£,\s,mi,km,hp,HP,l,L]', '', val_str)
    try:
        return float(cleaned)
    except ValueError:
        return np.nan

def clean_and_inspect_dataframe(df_raw: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Cleans raw DataFrame and returns cleaned DF along with inspection metadata.
    """
    df = df_raw.copy()
    inspection_report = {
        "raw_shape": df_raw.shape,
        "raw_columns": list(df_raw.columns),
        "raw_dtypes": {col: str(dtype) for col, dtype in df_raw.dtypes.items()},
        "missing_counts_before": df_raw.isnull().sum().to_dict(),
        "cleaned_columns": []
    }

    # Standardize column names (lowercase, stripped)
    column_mapping = {col: col.strip() for col in df.columns}
    df.rename(columns=column_mapping, inplace=True)

    # Detect price / target column candidates
    price_candidates = [c for c in df.columns if any(k in c.lower() for k in ["price", "cost", "value", "target", "selling"])]
    target_col = price_candidates[0] if price_candidates else df.columns[-1]
    inspection_report["detected_target_col"] = target_col

    # Clean target if it is object/string
    if df[target_col].dtype == object:
        df[target_col] = df[target_col].apply(clean_currency_or_numeric_string)

    # Clean other numeric columns that might be formatted as strings (e.g. milage, year)
    for col in df.columns:
        if col != target_col and df[col].dtype == object:
            # Check if majority can be converted to float
            sample = df[col].dropna().head(50)
            converted = sample.apply(clean_currency_or_numeric_string)
            if converted.notnull().sum() > len(sample) * 0.7:
                df[col] = df[col].apply(clean_currency_or_numeric_string)

    # Handle missing values
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

    # Impute numeric with median
    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            median_val = df[col].median()
            df[col].fillna(median_val if not np.isnan(median_val) else 0, inplace=True)

    # Impute categorical with mode
    for col in categorical_cols:
        if df[col].isnull().sum() > 0:
            mode_series = df[col].mode()
            mode_val = mode_series[0] if len(mode_series) > 0 else "Unknown"
            df[col].fillna(mode_val, inplace=True)

    # Drop duplicate rows
    df.drop_duplicates(inplace=True)

    inspection_report["cleaned_shape"] = df.shape
    inspection_report["numeric_features"] = [c for c in numeric_cols if c != target_col]
    inspection_report["categorical_features"] = categorical_cols
    inspection_report["missing_counts_after"] = df.isnull().sum().to_dict()

    return df, inspection_report

def prepare_modeling_data(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: Optional[List[str]] = None,
    test_size: float = 0.2,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Encodes features, scales numeric columns, and produces train/test partitions.
    Returns comprehensive bundle containing preprocessed arrays and encoders.
    """
    if feature_cols is None:
        feature_cols = [c for c in df.columns if c != target_col]

    X = df[feature_cols].copy()
    y = df[target_col].values.astype(float)

    numeric_features = [col for col in feature_cols if pd.api.types.is_numeric_dtype(X[col])]
    categorical_features = [col for col in feature_cols if col not in numeric_features]

    # Preprocessor using ColumnTransformer
    transformers = []
    if numeric_features:
        transformers.append(("num", StandardScaler(), numeric_features))
    if categorical_features:
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features))

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    
    # Fit and transform
    X_processed = preprocessor.fit_transform(X)

    # Extract transformed feature names
    feature_names = []
    if numeric_features:
        feature_names.extend(numeric_features)
    if categorical_features:
        cat_encoder = preprocessor.named_transformers_["cat"]
        cat_feature_names = cat_encoder.get_feature_names_out(categorical_features)
        feature_names.extend(list(cat_feature_names))

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y, test_size=test_size, random_state=random_state
    )

    # Keep original unscaled X for tree inspection/explanations
    X_train_df, X_test_df, _, _ = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "X_train_raw": X_train_df,
        "X_test_raw": X_test_df,
        "feature_names": feature_names,
        "feature_cols": feature_cols,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "preprocessor": preprocessor,
        "target_col": target_col,
        "raw_df": df
    }
