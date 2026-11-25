"""
Comprehensive Test Suite for Stage 1: Machine Learning Engine.
Validates all 12 core requirements with deterministic random seeds.
"""
import os
import json
import numpy as np
import pandas as pd

from ml_engine.anomaly_detection import run_anomaly_detection
from ml_engine.classification import run_classification
from ml_engine.regression import run_regression
from ml_engine.clustering import run_clustering
from ml_engine.model_manager import (
    load_trained_model,
    get_model_metadata,
    validate_prediction_input
)
from tools.ml_tools import predict_with_model
import data_store
from tools.statistics_tools import calculate_statistics
from tools.duckdb_tools import query_dataset_with_sql


class AssertRaises:
    """Context manager replacing pytest.raises."""
    def __init__(self, expected_exc, match=None):
        self.expected_exc = expected_exc
        self.match = match

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        assert exc_type is not None, f"Expected {self.expected_exc.__name__} was not raised."
        assert issubclass(exc_type, self.expected_exc), f"Expected {self.expected_exc.__name__}, got {exc_type.__name__}"
        if self.match:
            assert self.match in str(exc_val), f"Expected '{self.match}' in '{str(exc_val)}'"
        return True


# 1. Anomaly detection identifies injected anomalous records
def test_anomaly_detection_identifies_injected_outliers():
    np.random.seed(42)
    # 100 normal points centered around 10
    normal_data = np.random.normal(loc=10.0, scale=1.0, size=(100, 3))
    # 5 extreme outliers centered around 100
    outliers = np.array([
        [100.0, 95.0, 105.0],
        [120.0, 110.0, 115.0],
        [90.0, 85.0, 92.0],
        [130.0, 125.0, 140.0],
        [150.0, 145.0, 160.0]
    ])
    combined = np.vstack([normal_data, outliers])
    df = pd.DataFrame(combined, columns=["feature_1", "feature_2", "feature_3"])

    # Injected outlier indices are 100, 101, 102, 103, 104
    result = run_anomaly_detection(df, contamination=0.05, random_state=42)

    assert result["success"] is True
    assert result["anomaly_count"] > 0
    flagged = set(result["flagged_indices"])
    # Verify at least 4 of the 5 extreme outliers were successfully flagged
    detected_injected = flagged.intersection({100, 101, 102, 103, 104})
    assert len(detected_injected) >= 4, f"Expected injected outliers in flagged set, got {detected_injected}"


# 2. Classification trains successfully on a suitable dataset
def test_classification_trains_and_evaluates():
    np.random.seed(42)
    n = 120
    df = pd.DataFrame({
        "age": np.random.randint(18, 65, size=n),
        "income": np.random.normal(50000, 15000, size=n),
        "gender": np.random.choice(["M", "F"], size=n),
        "target": np.random.choice(["Yes", "No"], size=n)
    })

    result = run_classification(df, target_column="target", test_size=0.25, random_state=42)

    assert result["success"] is True
    assert result["algorithm"] == "RandomForestClassifier"
    assert "accuracy" in result["metrics"]
    assert 0.0 <= result["metrics"]["accuracy"] <= 1.0
    assert "precision" in result["metrics"]
    assert "recall" in result["metrics"]
    assert "f1_score" in result["metrics"]
    assert len(result["classes"]) == 2
    assert os.path.exists(result["charts"]["confusion_matrix_path"])
    assert os.path.exists(result["charts"]["feature_importance_path"])


# 3. Regression produces predictions and valid metrics
def test_regression_produces_valid_metrics():
    np.random.seed(42)
    n = 120
    x1 = np.random.uniform(1, 10, size=n)
    x2 = np.random.uniform(5, 20, size=n)
    y = 3.5 * x1 + 1.2 * x2 + np.random.normal(0, 0.5, size=n)

    df = pd.DataFrame({"x1": x1, "x2": x2, "price": y})

    result = run_regression(df, target_column="price", test_size=0.2, random_state=42)

    assert result["success"] is True
    metrics = result["metrics"]
    assert metrics["mae"] >= 0.0
    assert metrics["rmse"] >= 0.0
    assert metrics["r2_score"] > 0.8  # Strong linear signal should yield high R2
    assert os.path.exists(result["charts"]["actual_vs_predicted_path"])
    assert os.path.exists(result["charts"]["residuals_path"])


# 4. K-Means produces valid cluster assignments
def test_kmeans_produces_clusters():
    np.random.seed(42)
    # Generate 3 distinct blobs
    blob1 = np.random.normal(loc=5, scale=0.5, size=(40, 2))
    blob2 = np.random.normal(loc=15, scale=0.5, size=(40, 2))
    blob3 = np.random.normal(loc=25, scale=0.5, size=(40, 2))
    data = np.vstack([blob1, blob2, blob3])
    df = pd.DataFrame(data, columns=["feat_a", "feat_b"])

    result = run_clustering(df, n_clusters=3, run_elbow=True, random_state=42)

    assert result["success"] is True
    assert result["n_clusters"] == 3
    assert len(result["cluster_distribution"]) == 3
    # Check that inertia and silhouette exist
    assert result["metrics"]["inertia"] > 0
    assert result["metrics"]["silhouette_score"] is not None
    assert result["metrics"]["silhouette_score"] > 0.5
    assert os.path.exists(result["charts"]["cluster_scatter_path"])
    assert os.path.exists(result["charts"]["elbow_path"])


# 5. Missing values are handled correctly across pipelines
def test_missing_values_handled_without_crashing():
    np.random.seed(42)
    n = 100
    df = pd.DataFrame({
        "num_1": [np.nan if i % 5 == 0 else float(i) for i in range(n)],
        "num_2": [np.nan if i % 7 == 0 else float(i * 2) for i in range(n)],
        "cat_1": [np.nan if i % 6 == 0 else ("A" if i % 2 == 0 else "B") for i in range(n)],
        "target_clf": ["Pass" if i % 2 == 0 else "Fail" for i in range(n)],
        "target_reg": [float(i * 1.5 + np.random.normal(0, 0.1)) for i in range(n)]
    })

    # Test classification handles missing features
    res_clf = run_classification(df[["num_1", "num_2", "cat_1", "target_clf"]], target_column="target_clf", random_state=42)
    assert res_clf["success"] is True

    # Test regression handles missing features
    res_reg = run_regression(df[["num_1", "num_2", "cat_1", "target_reg"]], target_column="target_reg", random_state=42)
    assert res_reg["success"] is True

    # Test anomaly detection handles missing values
    res_ano = run_anomaly_detection(df[["num_1", "num_2"]], random_state=42)
    assert res_ano["success"] is True


# 6. Constant and empty columns do not cause unexpected crashes
def test_constant_and_empty_columns_dropped_safely():
    n = 80
    df = pd.DataFrame({
        "useful_num": np.random.normal(10, 2, size=n),
        "useful_cat": np.random.choice(["Alpha", "Beta"], size=n),
        "constant_col": [42.0] * n,
        "all_nan_col": [np.nan] * n,
        "target": np.random.choice([0, 1], size=n)
    })

    res = run_classification(df, target_column="target", random_state=42)
    assert res["success"] is True
    # Verify warnings were generated for discarded columns
    warnings_text = " ".join(res["warnings"])
    assert "constant_col" in warnings_text
    assert "all_nan_col" in warnings_text


# 7. Invalid target columns are rejected with informative errors
def test_invalid_target_columns_rejected():
    df = pd.DataFrame({
        "num": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        "non_numeric": ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l"]
    })

    # Non-existent target column
    with AssertRaises(ValueError, match="does not exist"):
        run_classification(df, target_column="ghost_column")

    # Non-numeric target for regression
    with AssertRaises(ValueError, match="must be numeric"):
        run_regression(df, target_column="non_numeric")

    # Single-class target for classification
    single_class_df = pd.DataFrame({
        "feature": range(20),
        "target": ["SameClass"] * 20
    })
    with AssertRaises(ValueError, match="at least 2 distinct classes"):
        run_classification(single_class_df, target_column="target")


# 8. Insufficient dataset sizes are handled gracefully
def test_insufficient_dataset_sizes():
    small_df = pd.DataFrame({
        "a": [1, 2, 3],
        "target": [0, 1, 0]
    })

    with AssertRaises(ValueError, match="Minimum 10"):
        run_classification(small_df, target_column="target")

    with AssertRaises(ValueError, match="Minimum 10"):
        run_anomaly_detection(small_df)

    with AssertRaises(ValueError, match="Minimum 10"):
        run_clustering(small_df)


# 9. Model saving and loading work with full metadata recovery
def test_model_saving_and_loading():
    np.random.seed(42)
    df = pd.DataFrame({
        "x": np.random.normal(0, 1, size=50),
        "y": np.random.choice(["Yes", "No"], size=50)
    })
    res = run_classification(df, target_column="y", random_state=42)
    model_id = res["model_id"]

    # Verify metadata retrieval
    meta = get_model_metadata(model_id)
    assert meta["model_id"] == model_id
    assert meta["algorithm"] == "RandomForestClassifier"
    assert "accuracy" in meta["metrics"]

    # Verify loading model bundle
    bundle = load_trained_model(model_id)
    assert bundle["model"] is not None
    assert bundle["preprocessor"] is not None


# 10. Prediction input schema validation works
def test_prediction_input_schema_validation():
    np.random.seed(42)
    df = pd.DataFrame({
        "age": [20, 30, 40, 50, 60, 25, 35, 45, 55, 65, 22, 32],
        "fare": [10.5, 20.0, 30.0, 40.0, 50.0, 15.0, 25.0, 35.0, 45.0, 55.0, 12.0, 22.0],
        "survived": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
    })
    res = run_classification(df, target_column="survived", random_state=42)
    model_id = res["model_id"]

    # Valid input sample
    valid_sample = {"age": 28, "fare": 18.5}
    is_valid, msg, ordered_df = validate_prediction_input(model_id, pd.DataFrame([valid_sample]))
    assert is_valid is True
    assert list(ordered_df.columns) == ["age", "fare"]

    # Missing required feature
    invalid_sample = {"age": 28} # Missing 'fare'
    is_valid_inv, err_msg, _ = validate_prediction_input(model_id, pd.DataFrame([invalid_sample]))
    assert is_valid_inv is False
    assert "fare" in err_msg

    # Test tool inference
    pred_res_str = predict_with_model.invoke({"model_id": model_id, "input_json": json.dumps(valid_sample)})
    pred_res = json.loads(pred_res_str)
    assert pred_res["success"] is True
    assert len(pred_res["predictions"]) == 1


# 11. Original dataset row indices are preserved
def test_original_dataset_row_indices_preserved():
    # Use non-consecutive or string indices
    custom_indices = [101, 102, 105, 110, 120, 130, 140, 150, 160, 170, 180, 190]
    df = pd.DataFrame({
        "val": [10.0, 10.2, 10.1, 9.9, 10.0, 10.3, 10.1, 10.0, 10.2, 9.8, 10.1, 999.0] # 190 is extreme outlier
    }, index=custom_indices)

    res = run_anomaly_detection(df, contamination=0.1, random_state=42)
    assert res["success"] is True
    assert 190 in res["flagged_indices"]


# 12. Existing analytics and DuckDB SQL functionality remains operational
def test_existing_analytics_and_sql_operational():
    df = pd.DataFrame({
        "Category": ["A", "B", "A", "B", "A"],
        "Sales": [100.0, 200.0, 150.0, 250.0, 300.0]
    })
    data_store.df = df

    # Test statistics tool
    stat_res = calculate_statistics.invoke({"column_name": "Sales", "operation": "mean"})
    assert "200" in str(stat_res)

    # Test DuckDB tool
    sql_res = query_dataset_with_sql.invoke({"query": "SELECT Category, SUM(Sales) as total FROM df GROUP BY Category ORDER BY Category"})
    assert "550" in sql_res # Sum of A: 100+150+300 = 550
    assert "450" in sql_res # Sum of B: 200+250 = 450
