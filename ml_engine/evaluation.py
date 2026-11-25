"""
Standardized model evaluation module.
Generates metrics and reports for Classification, Regression, and Clustering.
Strictly calculates metrics from true and predicted values without hallucination.
"""
import numpy as np
from typing import Dict, Any, List, Optional
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    silhouette_score
)


def evaluate_classification(
    y_true: Any,
    y_pred: Any,
    labels: Optional[List[Any]] = None
) -> Dict[str, Any]:
    """Calculate comprehensive classification metrics."""
    # Ensure standard python types for JSON serialization
    y_t = np.array(y_true)
    y_p = np.array(y_pred)

    acc = float(accuracy_score(y_t, y_p))

    # Detect binary vs multiclass
    unique_classes = np.unique(y_t)
    is_binary = len(unique_classes) == 2

    if is_binary:
        pos_label = unique_classes[1]
        try:
            prec = float(precision_score(y_t, y_p, pos_label=pos_label, zero_division=0))
            rec = float(recall_score(y_t, y_p, pos_label=pos_label, zero_division=0))
            f1 = float(f1_score(y_t, y_p, pos_label=pos_label, zero_division=0))
        except Exception:
            prec = float(precision_score(y_t, y_p, average="weighted", zero_division=0))
            rec = float(recall_score(y_t, y_p, average="weighted", zero_division=0))
            f1 = float(f1_score(y_t, y_p, average="weighted", zero_division=0))
    else:
        prec = float(precision_score(y_t, y_p, average="weighted", zero_division=0))
        rec = float(recall_score(y_t, y_p, average="weighted", zero_division=0))
        f1 = float(f1_score(y_t, y_p, average="weighted", zero_division=0))

    cm = confusion_matrix(y_t, y_p, labels=labels).tolist()
    report = classification_report(y_t, y_p, labels=labels, output_dict=True, zero_division=0)

    return {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "confusion_matrix": cm,
        "classes": [str(c) for c in (labels if labels is not None else unique_classes)],
        "classification_report": report
    }


def evaluate_regression(
    y_true: Any,
    y_pred: Any
) -> Dict[str, Any]:
    """Calculate comprehensive regression metrics."""
    y_t = np.array(y_true, dtype=float)
    y_p = np.array(y_pred, dtype=float)

    mae = float(mean_absolute_error(y_t, y_p))
    mse = float(mean_squared_error(y_t, y_p))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_t, y_p))

    # Mean Percentage Error / Residuals
    residuals = (y_t - y_p).tolist()

    return {
        "mae": round(mae, 4),
        "mse": round(mse, 4),
        "rmse": round(rmse, 4),
        "r2_score": round(r2, 4),
        "residual_mean": round(float(np.mean(residuals)), 4),
        "residual_std": round(float(np.std(residuals)), 4)
    }


def evaluate_clustering(
    X_scaled: np.ndarray,
    labels: np.ndarray,
    inertia: Optional[float] = None
) -> Dict[str, Any]:
    """Evaluate cluster cohesion and separation."""
    metrics: Dict[str, Any] = {}
    if inertia is not None:
        metrics["inertia"] = round(float(inertia), 2)

    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)
    n_samples = len(X_scaled)

    # Silhouette score is only valid when 2 <= n_clusters <= n_samples - 1
    if 2 <= n_clusters < n_samples:
        try:
            # Sample up to 1000 records for performance if dataset is large
            if n_samples > 1000:
                sil = silhouette_score(X_scaled, labels, sample_size=1000, random_state=42)
            else:
                sil = silhouette_score(X_scaled, labels)
            metrics["silhouette_score"] = round(float(sil), 4)
        except Exception as e:
            metrics["silhouette_score"] = None
            metrics["silhouette_note"] = f"Could not calculate silhouette: {e}"
    else:
        metrics["silhouette_score"] = None
        metrics["silhouette_note"] = "Silhouette score requires at least 2 clusters and more samples than clusters."

    return metrics
