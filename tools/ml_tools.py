"""
LangChain-compatible Machine Learning Tools for the AI Data Analyst Agent.
Exposes Anomaly Detection, Classification, Regression, Clustering,
Model Evaluation, and Inference to LangGraph.
"""
import json
import pandas as pd
from typing import Dict, Any, List, Optional
from langchain_core.tools import tool

import data_store
from tools.helper_tools import find_best_matching_column
from ml_engine.anomaly_detection import run_anomaly_detection
from ml_engine.classification import run_classification
from ml_engine.regression import run_regression
from ml_engine.clustering import run_clustering
from ml_engine.model_manager import (
    get_model_metadata,
    load_trained_model,
    validate_prediction_input,
    list_trained_models
)


@tool
def detect_anomalies(contamination: float = 0.05) -> str:
    """
    Train an Isolation Forest model to detect potential statistical outliers and anomalies in the loaded dataset.

    Args:
        contamination: Expected proportion of outliers in the data (between 0.001 and 0.5, default is 0.05).

    Returns:
        JSON string containing anomaly counts, percentage, top flagged records, and the chart path.
    """
    if data_store.df is None:
        return json.dumps({"success": False, "message": "No dataset loaded. Please upload or load a dataset first."})

    try:
        results = run_anomaly_detection(data_store.df, contamination=contamination)
        return json.dumps({
            "success": True,
            "algorithm": "IsolationForest",
            "model_id": results["model_id"],
            "total_records": results["total_records"],
            "anomaly_count": results["anomaly_count"],
            "anomaly_percentage": results["anomaly_percentage"],
            "features_analyzed": results["features_analyzed"],
            "chart_path": results["chart_path"],
            "flagged_sample_indices": results["flagged_indices"][:10],
            "note": results["note"]
        })
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)})


@tool
def train_classification_model(target_column: str, test_size: float = 0.2) -> str:
    """
    Train a Random Forest Classifier to predict a categorical or discrete target column.
    Automatically handles missing values, encodes categorical features, and prevents data leakage.

    Args:
        target_column: The name of the target column to predict (e.g. 'Survived', 'Species').
        test_size: Proportion of dataset reserved for testing (default 0.2).

    Returns:
        JSON string containing model metrics (accuracy, precision, recall, f1), top features, and chart paths.
    """
    if data_store.df is None:
        return json.dumps({"success": False, "message": "No dataset loaded. Please upload or load a dataset first."})

    # Resolve target column name using fuzzy matching
    match = find_best_matching_column.invoke({"column_name": target_column})
    if not match.get("success") or match.get("confidence", 0) < 0.7:
        return json.dumps({
            "success": False,
            "error": f"Could not find a matching column for target '{target_column}'. Available columns: {', '.join(data_store.df.columns)}"
        })

    resolved_target = match["matched_column"]

    try:
        results = run_classification(
            df=data_store.df,
            target_column=resolved_target,
            test_size=test_size
        )
        return json.dumps({
            "success": True,
            "algorithm": "RandomForestClassifier",
            "model_id": results["model_id"],
            "target_column": resolved_target,
            "classes": results["classes"],
            "metrics": results["metrics"],
            "top_features": results["top_features"][:8],
            "chart_path": results["charts"]["confusion_matrix_path"],
            "feature_importance_chart_path": results["charts"]["feature_importance_path"],
            "train_samples": results["train_samples"],
            "test_samples": results["test_samples"],
            "warnings": results["warnings"]
        })
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)})


@tool
def train_regression_model(target_column: str, test_size: float = 0.2) -> str:
    """
    Train a Random Forest Regressor to predict a continuous numerical target column.
    Automatically handles missing values, encodes features, and prevents data leakage.

    Args:
        target_column: The numerical target column to predict (e.g. 'Fare', 'Age', 'Sales').
        test_size: Proportion of dataset reserved for testing (default 0.2).

    Returns:
        JSON string containing regression metrics (MAE, RMSE, R² score), top features, and chart paths.
    """
    if data_store.df is None:
        return json.dumps({"success": False, "message": "No dataset loaded. Please upload or load a dataset first."})

    # Resolve target column name using fuzzy matching
    match = find_best_matching_column.invoke({"column_name": target_column})
    if not match.get("success") or match.get("confidence", 0) < 0.7:
        return json.dumps({
            "success": False,
            "error": f"Could not find a matching column for target '{target_column}'. Available columns: {', '.join(data_store.df.columns)}"
        })

    resolved_target = match["matched_column"]

    try:
        results = run_regression(
            df=data_store.df,
            target_column=resolved_target,
            test_size=test_size
        )
        return json.dumps({
            "success": True,
            "algorithm": "RandomForestRegressor",
            "model_id": results["model_id"],
            "target_column": resolved_target,
            "metrics": results["metrics"],
            "top_features": results["top_features"][:8],
            "chart_path": results["charts"]["actual_vs_predicted_path"],
            "residuals_chart_path": results["charts"]["residuals_path"],
            "feature_importance_chart_path": results["charts"]["feature_importance_path"],
            "train_samples": results["train_samples"],
            "test_samples": results["test_samples"],
            "warnings": results["warnings"]
        })
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)})


@tool
def perform_clustering(n_clusters: int = 3, feature_columns: str = "") -> str:
    """
    Perform K-Means clustering on numerical features to discover natural groupings or customer segments.

    Args:
        n_clusters: The number of clusters to form (default 3).
        feature_columns: Optional comma-separated list of numerical column names to use. If blank, automatically uses all usable numerical features.

    Returns:
        JSON string containing cluster distribution, centroid profiles, inertia, and 2D cluster chart path.
    """
    if data_store.df is None:
        return json.dumps({"success": False, "message": "No dataset loaded. Please upload or load a dataset first."})

    selected_cols = None
    if feature_columns and feature_columns.strip():
        raw_cols = [c.strip() for c in feature_columns.split(",") if c.strip()]
        resolved_cols = []
        for col in raw_cols:
            match = find_best_matching_column.invoke({"column_name": col})
            if match.get("success") and match.get("confidence", 0) >= 0.7:
                resolved_cols.append(match["matched_column"])
            else:
                return json.dumps({"success": False, "error": f"Could not resolve feature column '{col}'."})
        selected_cols = resolved_cols

    try:
        results = run_clustering(
            df=data_store.df,
            n_clusters=n_clusters,
            feature_columns=selected_cols,
            run_elbow=False
        )
        return json.dumps({
            "success": True,
            "algorithm": "KMeans",
            "model_id": results["model_id"],
            "n_clusters": results["n_clusters"],
            "features_used": results["features_used"],
            "metrics": results["metrics"],
            "cluster_distribution": results["cluster_distribution"],
            "cluster_profiles": results["cluster_profiles"],
            "chart_path": results["charts"]["cluster_scatter_path"]
        })
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)})


@tool
def evaluate_trained_model(model_id: str) -> str:
    """
    Retrieve evaluation metrics, parameters, and metadata for a previously trained model.

    Args:
        model_id: The unique identifier of the saved model (e.g. 'rf_clf_abc12345').

    Returns:
        JSON string with model metadata and metrics.
    """
    try:
        meta = get_model_metadata(model_id)
        return json.dumps({"success": True, "metadata": meta})
    except Exception as e:
        return json.dumps({"success": False, "error": f"Failed to retrieve model '{model_id}': {e}"})


@tool
def predict_with_model(model_id: str, input_json: str) -> str:
    """
    Generate predictions for new sample(s) using a previously trained and saved model.

    Args:
        model_id: The unique identifier of the model to predict with.
        input_json: JSON string representing a single sample dict or list of sample dicts matching training features.

    Returns:
        JSON string containing the generated predictions (and class probabilities if applicable).
    """
    try:
        data = json.loads(input_json)
        if isinstance(data, dict):
            input_df = pd.DataFrame([data])
        elif isinstance(data, list):
            input_df = pd.DataFrame(data)
        else:
            return json.dumps({"success": False, "error": "input_json must be a JSON object or list of objects."})

        # Validate schema
        valid, err, ordered_df = validate_prediction_input(model_id, input_df)
        if not valid:
            return json.dumps({"success": False, "error": err})

        bundle = load_trained_model(model_id)
        model = bundle["model"]
        preprocessor = bundle["preprocessor"]
        meta = bundle["metadata"]

        # Preprocess input data
        if preprocessor is not None:
            if hasattr(preprocessor, "transform"):
                X_proc = preprocessor.transform(ordered_df)
            elif isinstance(preprocessor, dict) and "imputer" in preprocessor:
                # Anomaly / Clustering preprocessor dict
                X_proc = preprocessor["imputer"].transform(ordered_df)
                if "scaler" in preprocessor:
                    X_proc = preprocessor["scaler"].transform(X_proc)
            else:
                X_proc = ordered_df.values
        else:
            X_proc = ordered_df.values

        preds = model.predict(X_proc)
        result = {
            "success": True,
            "model_id": model_id,
            "model_type": meta.get("model_type"),
            "predictions": preds.tolist()
        }

        # Include probabilities if classification
        if hasattr(model, "predict_proba"):
            try:
                probs = model.predict_proba(X_proc)
                result["probabilities"] = np.round(probs, 4).tolist()
                result["classes"] = [str(c) for c in model.classes_]
            except Exception:
                pass

        return json.dumps(result)
    except Exception as e:
        return json.dumps({"success": False, "error": f"Inference error: {e}"})
