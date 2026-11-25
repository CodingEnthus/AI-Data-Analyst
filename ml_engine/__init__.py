"""
Machine Learning Engine Package for AI Data Analyst.
Exports core algorithms, preprocessing pipelines, evaluations, and model management.
"""

from ml_engine.anomaly_detection import run_anomaly_detection
from ml_engine.classification import run_classification
from ml_engine.regression import run_regression
from ml_engine.clustering import run_clustering
from ml_engine.model_manager import (
    save_trained_model,
    load_trained_model,
    get_model_metadata,
    list_trained_models,
    validate_prediction_input
)
from ml_engine.preprocessing import (
    analyze_features,
    clean_dataframe,
    prepare_supervised_data
)
from ml_engine.evaluation import (
    evaluate_classification,
    evaluate_regression,
    evaluate_clustering
)

__all__ = [
    "run_anomaly_detection",
    "run_classification",
    "run_regression",
    "run_clustering",
    "save_trained_model",
    "load_trained_model",
    "get_model_metadata",
    "list_trained_models",
    "validate_prediction_input",
    "analyze_features",
    "clean_dataframe",
    "prepare_supervised_data",
    "evaluate_classification",
    "evaluate_regression",
    "evaluate_clustering"
]
