"""
Anomaly Detection Module using Scikit-Learn IsolationForest.
Detects potential multivariate outliers, scores anomalies, and preserves original row indices.
"""
import time
import numpy as np
import pandas as pd
import plotly.express as px
from typing import Dict, Any, List, Optional
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from ml_engine.preprocessing import clean_dataframe, analyze_features
from ml_engine.model_manager import save_trained_model
from tools.helper_tools import save_chart


def run_anomaly_detection(
    df: pd.DataFrame,
    contamination: float = 0.05,
    feature_columns: Optional[List[str]] = None,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Train an Isolation Forest to flag potential anomalies in the dataset.
    
    Returns structured results including metrics, original indices,
    flagged records, and a Plotly score distribution chart.
    """
    start_time = time.time()

    if df is None or len(df) == 0:
        raise ValueError("Cannot run anomaly detection on an empty or missing dataset.")

    if len(df) < 10:
        raise ValueError(f"Dataset has only {len(df)} records. Minimum 10 required.")

    # Validate contamination bounds
    if not (0.001 <= contamination <= 0.5):
        raise ValueError(f"Contamination must be between 0.001 and 0.5 (received {contamination}).")

    cleaned_df = clean_dataframe(df)

    # Determine numeric feature subset
    if feature_columns:
        # User specified features
        missing_cols = [c for c in feature_columns if c not in cleaned_df.columns]
        if missing_cols:
            raise ValueError(f"Specified feature column(s) not found: {', '.join(missing_cols)}")
        numeric_cols = [c for c in feature_columns if pd.api.types.is_numeric_dtype(cleaned_df[c])]
        if not numeric_cols:
            raise ValueError("None of the specified feature columns are numerical.")
    else:
        # Automatic selection
        feature_info = analyze_features(cleaned_df)
        numeric_cols = feature_info["numeric_features"]

    if not numeric_cols:
        raise ValueError("No usable numerical features found in dataset for anomaly detection.")

    X_numeric = cleaned_df[numeric_cols].copy()

    # Impute missing values with median
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    X_imputed = imputer.fit_transform(X_numeric)
    X_scaled = scaler.fit_transform(X_imputed)

    # Fit Isolation Forest
    model = IsolationForest(
        contamination=contamination,
        random_state=random_state,
        n_estimators=100
    )
    model.fit(X_scaled)

    # Predictions: -1 for anomaly, 1 for normal
    raw_preds = model.predict(X_scaled)
    # Decision function: lower values mean more anomalous
    decision_scores = model.decision_function(X_scaled)

    # Normalize anomaly score to [0, 1] where 1.0 = highest anomaly severity
    min_score = decision_scores.min()
    max_score = decision_scores.max()
    score_range = max_score - min_score if max_score != min_score else 1.0
    normalized_scores = 1.0 - ((decision_scores - min_score) / score_range)

    # Identify anomalies
    is_anomaly = raw_preds == -1
    anomaly_indices = list(df.index[is_anomaly])
    total_records = len(df)
    anomaly_count = int(is_anomaly.sum())
    anomaly_pct = round((anomaly_count / total_records) * 100, 2)

    # Compile flagged records table with anomaly scores
    flagged_df = df.loc[anomaly_indices].copy()
    flagged_df["anomaly_score"] = np.round(normalized_scores[is_anomaly], 4)
    # Sort descending by anomaly score
    flagged_df = flagged_df.sort_values(by="anomaly_score", ascending=False)

    # Generate Plotly visualization: Anomaly Score Distribution
    score_df = pd.DataFrame({
        "Anomaly Score": normalized_scores,
        "Status": ["Potential Anomaly" if a else "Normal" for a in is_anomaly]
    })

    fig = px.histogram(
        score_df,
        x="Anomaly Score",
        color="Status",
        color_discrete_map={"Normal": "#2b5c8f", "Potential Anomaly": "#e63946"},
        barmode="overlay",
        opacity=0.75,
        title=f"Anomaly Score Distribution (Contamination: {contamination * 100:.1f}%)"
    )
    fig.update_layout(xaxis_title="Normalized Anomaly Severity (0 = Normal, 1 = Outlier)", yaxis_title="Record Count")
    chart_path = save_chart(fig, "anomaly_distribution")

    duration = round(time.time() - start_time, 3)

    # Save model artifact
    metadata = save_trained_model(
        model=model,
        preprocessor={"imputer": imputer, "scaler": scaler},
        model_type="anomaly_detection",
        algorithm="IsolationForest",
        feature_names=numeric_cols,
        metrics={
            "total_records": total_records,
            "anomaly_count": anomaly_count,
            "anomaly_percentage": anomaly_pct,
            "training_duration_seconds": duration
        },
        params={"contamination": contamination, "random_state": random_state},
        description=f"Isolation Forest trained on {len(numeric_cols)} numeric features."
    )

    return {
        "success": True,
        "model_id": metadata["model_id"],
        "algorithm": "IsolationForest",
        "total_records": total_records,
        "anomaly_count": anomaly_count,
        "anomaly_percentage": anomaly_pct,
        "features_analyzed": numeric_cols,
        "flagged_indices": anomaly_indices,
        "flagged_records_sample": flagged_df.head(20).to_dict(orient="records"),
        "chart_path": chart_path,
        "training_duration": duration,
        "note": "Records flagged are potential statistical anomalies requiring investigation, not confirmed defects or fraud."
    }
