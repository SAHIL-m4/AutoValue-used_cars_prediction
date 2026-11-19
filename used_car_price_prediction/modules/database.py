"""
SQLite Database module for persisting prediction logs, evaluation metrics, and user interactions.
"""

import sqlite3
import os
import pandas as pd
from datetime import datetime
from typing import Optional, List, Dict, Any

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "used_cars_app.db")

def get_connection():
    """Returns a connection to the SQLite database."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Initializes database tables if they do not already exist."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS prediction_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT NOT NULL,
        brand TEXT,
        model TEXT,
        model_year INTEGER,
        milage REAL,
        fuel_type TEXT,
        engine TEXT,
        transmission TEXT,
        accident TEXT,
        clean_title TEXT,
        algorithm_used TEXT,
        predicted_price REAL,
        lower_bound REAL,
        upper_bound REAL
    )
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS model_benchmarks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT NOT NULL,
        unit_module TEXT,
        model_name TEXT,
        metric_r2 REAL,
        metric_mae REAL,
        metric_rmse REAL,
        metric_mse REAL,
        details TEXT
    )
    """)

    conn.commit()
    conn.close()

def log_prediction(data: Dict[str, Any]) -> int:
    """Inserts a single car price prediction log."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
    INSERT INTO prediction_history (
        timestamp, brand, model, model_year, milage, fuel_type, engine,
        transmission, accident, clean_title, algorithm_used,
        predicted_price, lower_bound, upper_bound
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        data.get("brand"),
        data.get("model"),
        data.get("model_year"),
        data.get("milage"),
        data.get("fuel_type"),
        data.get("engine"),
        data.get("transmission"),
        data.get("accident"),
        data.get("clean_title"),
        data.get("algorithm_used"),
        data.get("predicted_price"),
        data.get("lower_bound"),
        data.get("upper_bound")
    ))
    conn.commit()
    inserted_id = cursor.lastrowid
    conn.close()
    return inserted_id

def get_prediction_history(limit: int = 50) -> pd.DataFrame:
    """Retrieves prediction history as a pandas DataFrame."""
    conn = get_connection()
    query = f"SELECT * FROM prediction_history ORDER BY id DESC LIMIT {limit}"
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

def clear_prediction_history() -> bool:
    """Clears all records in prediction history."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM prediction_history")
    conn.commit()
    conn.close()
    return True

def log_model_benchmark(unit_module: str, model_name: str, r2: float, mae: float, rmse: float, mse: float, details: str = ""):
    """Records a benchmark result for trained models."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
    INSERT INTO model_benchmarks (
        timestamp, unit_module, model_name, metric_r2, metric_mae, metric_rmse, metric_mse, details
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        unit_module, model_name, r2, mae, rmse, mse, details
    ))
    conn.commit()
    conn.close()
