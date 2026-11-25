"""
Classification Engine using Scikit-Learn RandomForestClassifier.
Supports binary and multiclass tasks with stratified train-test splits,
class imbalance checks, and Plotly performance visualizations.
"""
import time
import numpy as np
import pandas as pd
import plotly.express as px
from typing import Dict, Any, List, Optional
from sklearn.ensemble import RandomForestClassifier

from ml_engine.preprocessing import prepare_supervised_data
from ml_engine.evaluation import evaluate_classification
from ml_engine.model_manager import save_trained_model
from tools.helper_tools import save_chart


def run_classification(
    df: pd.DataFrame,
    target_column: str,
    test_size: float = 0.2,
    feature_columns: Optional[List[str]] = None,
    n_estimators: int = 100,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Train and evaluate a Random Forest Classifier with strict leakage prevention.
    
    Returns metrics, confusion matrix, feature importances, and chart paths.
    """
    start_time = time.time()

    if df is None or len(df) == 0:
        raise ValueError("Cannot train on an empty dataset.")

    # Prepare data with leakage-free pipeline fitted only on X_train
    prep = prepare_supervised_data(
        df=df,
        target_column=target_column,
        test_size=test_size,
        is_classification=True,
        scale_numeric=False,
        random_state=random_state
    )

    X_train_proc = prep["X_train_proc"]
    X_test_proc = prep["X_test_proc"]
    y_train = prep["y_train"]
    y_test = prep["y_test"]

    # Class balance check
    class_dist = y_train.value_counts(normalize=True).to_dict()
    min_class_prop = min(class_dist.values())
    imbalance_warning = None
    if min_class_prop < 0.1:
        imbalance_warning = f"Severe class imbalance detected: minority class represents only {min_class_prop*100:.1f}% of training data."

    # Fit Random Forest Classifier
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        class_weight="balanced" if min_class_prop < 0.2 else None
    )
    clf.fit(X_train_proc, y_train)

    # Predict on unseen test set
    y_pred = clf.predict(X_test_proc)

    # Evaluate model
    classes = [str(c) for c in clf.classes_]
    metrics = evaluate_classification(y_test, y_pred, labels=clf.classes_)

    # Extract feature importances
    feature_names = prep["transformed_feature_names"]
    importances = clf.feature_importances_
    sorted_idx = np.argsort(importances)[::-1][:15] # Top 15 features
    top_features = [feature_names[i] for i in sorted_idx]
    top_scores = [round(float(importances[i]), 4) for i in sorted_idx]

    # Visualizations:
    # 1. Confusion Matrix
    cm_df = pd.DataFrame(
        metrics["confusion_matrix"],
        index=[f"True: {c}" for c in classes],
        columns=[f"Pred: {c}" for c in classes]
    )
    cm_fig = px.imshow(
        cm_df,
        text_auto=True,
        color_continuous_scale="Blues",
        title=f"Confusion Matrix ({target_column})"
    )
    cm_fig.update_layout(xaxis_title="Predicted Class", yaxis_title="Actual Class")
    cm_chart_path = save_chart(cm_fig, "confusion_matrix")

    # 2. Feature Importance Bar Chart
    fi_df = pd.DataFrame({"Feature": top_features, "Importance": top_scores}).sort_values(by="Importance", ascending=True)
    fi_fig = px.bar(
        fi_df,
        x="Importance",
        y="Feature",
        orientation="h",
        title=f"Top Feature Importances ({target_column})"
    )
    fi_fig.update_layout(xaxis_title="Gini Importance", yaxis_title="Feature")
    fi_chart_path = save_chart(fi_fig, "feature_importance")

    duration = round(time.time() - start_time, 3)

    # Persist model
    metadata = save_trained_model(
        model=clf,
        preprocessor=prep["preprocessor"],
        model_type="classification",
        algorithm="RandomForestClassifier",
        feature_names=prep["usable_features"],
        target_name=target_column,
        target_classes=list(clf.classes_),
        metrics={
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1_score": metrics["f1_score"],
            "test_sample_count": len(y_test),
            "training_duration_seconds": duration
        },
        params={"n_estimators": n_estimators, "test_size": test_size, "random_state": random_state},
        description=f"Random Forest Classifier trained to predict '{target_column}'."
    )

    all_warnings = prep["warnings"]
    if imbalance_warning:
        all_warnings.append(imbalance_warning)

    return {
        "success": True,
        "model_id": metadata["model_id"],
        "algorithm": "RandomForestClassifier",
        "target_column": target_column,
        "classes": classes,
        "metrics": {
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1_score": metrics["f1_score"]
        },
        "confusion_matrix": metrics["confusion_matrix"],
        "top_features": list(zip(top_features, top_scores)),
        "charts": {
            "confusion_matrix_path": cm_chart_path,
            "feature_importance_path": fi_chart_path
        },
        "train_samples": len(y_train),
        "test_samples": len(y_test),
        "training_duration": duration,
        "warnings": all_warnings
    }
