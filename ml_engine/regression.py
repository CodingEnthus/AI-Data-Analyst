"""
Regression Engine using Scikit-Learn RandomForestRegressor.
Trains continuous prediction models, evaluates MAE/MSE/RMSE/R2,
and generates Plotly performance charts (Actual vs Predicted, Residuals).
"""
import time
import numpy as np
import pandas as pd
import plotly.express as px
from typing import Dict, Any, List, Optional
from sklearn.ensemble import RandomForestRegressor

from ml_engine.preprocessing import prepare_supervised_data
from ml_engine.evaluation import evaluate_regression
from ml_engine.model_manager import save_trained_model
from tools.helper_tools import save_chart


def run_regression(
    df: pd.DataFrame,
    target_column: str,
    test_size: float = 0.2,
    feature_columns: Optional[List[str]] = None,
    n_estimators: int = 100,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Train and evaluate a Random Forest Regressor with strict leakage prevention.
    
    Returns metrics, actual vs predicted chart, residual chart, and feature importances.
    """
    start_time = time.time()

    if df is None or len(df) == 0:
        raise ValueError("Cannot train on an empty dataset.")

    # Prepare data with leakage-free pipeline fitted only on X_train
    prep = prepare_supervised_data(
        df=df,
        target_column=target_column,
        test_size=test_size,
        is_classification=False,
        scale_numeric=False,
        random_state=random_state
    )

    X_train_proc = prep["X_train_proc"]
    X_test_proc = prep["X_test_proc"]
    y_train = prep["y_train"]
    y_test = prep["y_test"]

    # Train Random Forest Regressor
    reg = RandomForestRegressor(
        n_estimators=n_estimators,
        random_state=random_state
    )
    reg.fit(X_train_proc, y_train)

    # Predict on test set
    y_pred = reg.predict(X_test_proc)

    # Calculate metrics
    metrics = evaluate_regression(y_test, y_pred)

    # Feature importances
    feature_names = prep["transformed_feature_names"]
    importances = reg.feature_importances_
    sorted_idx = np.argsort(importances)[::-1][:15]
    top_features = [feature_names[i] for i in sorted_idx]
    top_scores = [round(float(importances[i]), 4) for i in sorted_idx]

    # Visualizations:
    # 1. Actual vs Predicted Scatter with y=x diagonal
    actual_pred_df = pd.DataFrame({
        "Actual": y_test.values,
        "Predicted": y_pred
    })
    avp_fig = px.scatter(
        actual_pred_df,
        x="Actual",
        y="Predicted",
        opacity=0.65,
        title=f"Actual vs. Predicted ({target_column}) [R²: {metrics['r2_score']}]"
    )
    # Add ideal reference line y=x
    min_val = min(actual_pred_df["Actual"].min(), actual_pred_df["Predicted"].min())
    max_val = max(actual_pred_df["Actual"].max(), actual_pred_df["Predicted"].max())
    avp_fig.add_shape(
        type="line",
        x0=min_val, y0=min_val,
        x1=max_val, y1=max_val,
        line=dict(color="red", dash="dash", width=2)
    )
    avp_fig.update_layout(xaxis_title=f"Actual {target_column}", yaxis_title=f"Predicted {target_column}")
    avp_chart_path = save_chart(avp_fig, "actual_vs_predicted")

    # 2. Residuals Distribution
    residuals = actual_pred_df["Actual"] - actual_pred_df["Predicted"]
    res_fig = px.histogram(
        residuals,
        nbins=30,
        title=f"Residual Distribution ({target_column})"
    )
    res_fig.update_layout(xaxis_title="Residual (Actual - Predicted)", yaxis_title="Count")
    res_chart_path = save_chart(res_fig, "residuals_distribution")

    # 3. Feature Importance Bar Chart
    fi_df = pd.DataFrame({"Feature": top_features, "Importance": top_scores}).sort_values(by="Importance", ascending=True)
    fi_fig = px.bar(
        fi_df,
        x="Importance",
        y="Feature",
        orientation="h",
        title=f"Top Feature Importances ({target_column})"
    )
    fi_fig.update_layout(xaxis_title="Importance", yaxis_title="Feature")
    fi_chart_path = save_chart(fi_fig, "feature_importance_reg")

    duration = round(time.time() - start_time, 3)

    # Persist model
    metadata = save_trained_model(
        model=reg,
        preprocessor=prep["preprocessor"],
        model_type="regression",
        algorithm="RandomForestRegressor",
        feature_names=prep["usable_features"],
        target_name=target_column,
        metrics={
            "mae": metrics["mae"],
            "mse": metrics["mse"],
            "rmse": metrics["rmse"],
            "r2_score": metrics["r2_score"],
            "test_sample_count": len(y_test),
            "training_duration_seconds": duration
        },
        params={"n_estimators": n_estimators, "test_size": test_size, "random_state": random_state},
        description=f"Random Forest Regressor trained to predict continuous '{target_column}'."
    )

    return {
        "success": True,
        "model_id": metadata["model_id"],
        "algorithm": "RandomForestRegressor",
        "target_column": target_column,
        "metrics": metrics,
        "top_features": list(zip(top_features, top_scores)),
        "charts": {
            "actual_vs_predicted_path": avp_chart_path,
            "residuals_path": res_chart_path,
            "feature_importance_path": fi_chart_path
        },
        "train_samples": len(y_train),
        "test_samples": len(y_test),
        "training_duration": duration,
        "warnings": prep["warnings"]
    }
