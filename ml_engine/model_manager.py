"""
Model persistence, schema tracking, and metadata management for the ML Engine.
Uses joblib for model artifacts and JSON for metadata.
"""
import os
import json
import uuid
import datetime
import joblib
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple

MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)


def _get_safe_model_path(model_id: str, extension: str = "joblib") -> str:
    """Ensure model filename stays within the models directory (prevent traversal)."""
    clean_id = os.path.basename(model_id)
    return os.path.join(MODELS_DIR, f"{clean_id}.{extension}")


def save_trained_model(
    model: Any,
    preprocessor: Any,
    model_type: str,
    algorithm: str,
    feature_names: List[str],
    target_name: Optional[str] = None,
    metrics: Optional[Dict[str, Any]] = None,
    params: Optional[Dict[str, Any]] = None,
    description: str = "",
    target_classes: Optional[List[Any]] = None
) -> Dict[str, Any]:
    """
    Serialize model, preprocessor, and metadata to disk.
    
    Returns:
        Metadata dictionary including the assigned model_id.
    """
    prefix = {
        "anomaly_detection": "iso_forest",
        "classification": "rf_clf",
        "regression": "rf_reg",
        "clustering": "kmeans"
    }.get(model_type, "model")
    
    model_id = f"{prefix}_{uuid.uuid4().hex[:8]}"
    model_path = _get_safe_model_path(model_id, "joblib")
    meta_path = _get_safe_model_path(model_id, "json")

    metadata = {
        "model_id": model_id,
        "model_type": model_type,
        "algorithm": algorithm,
        "feature_names": feature_names,
        "target_name": target_name,
        "target_classes": [str(c) for c in target_classes] if target_classes is not None else None,
        "metrics": metrics or {},
        "params": {k: str(v) for k, v in (params or {}).items()},
        "created_at": datetime.datetime.now().isoformat(),
        "description": description,
        "model_file": os.path.basename(model_path)
    }

    # Save artifact containing estimator and preprocessor
    bundle = {
        "model": model,
        "preprocessor": preprocessor,
        "metadata": metadata
    }
    joblib.dump(bundle, model_path)

    # Save metadata separately for fast indexing without loading model weights
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return metadata


def load_trained_model(model_id: str) -> Dict[str, Any]:
    """
    Load previously saved model artifact bundle.
    
    Security note: Only loads models from the dedicated local models directory.
    Loading untrusted pickle/joblib files can execute arbitrary code.
    """
    model_path = _get_safe_model_path(model_id, "joblib")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model artifact not found for model ID '{model_id}' at {model_path}")

    bundle = joblib.load(model_path)
    return bundle


def get_model_metadata(model_id: str) -> Dict[str, Any]:
    """Retrieve metadata without deserializing the entire model bundle."""
    meta_path = _get_safe_model_path(model_id, "json")
    if not os.path.exists(meta_path):
        # Fallback to bundle if json missing
        bundle = load_trained_model(model_id)
        return bundle.get("metadata", {})

    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


def list_trained_models() -> List[Dict[str, Any]]:
    """List all available models in the registry."""
    models = []
    if not os.path.exists(MODELS_DIR):
        return models

    for fname in os.listdir(MODELS_DIR):
        if fname.endswith(".json"):
            meta_path = os.path.join(MODELS_DIR, fname)
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    models.append(json.load(f))
            except Exception:
                continue

    # Sort descending by creation date
    models.sort(key=lambda m: m.get("created_at", ""), reverse=True)
    return models


def validate_prediction_input(
    model_id: str,
    input_df: pd.DataFrame
) -> Tuple[bool, str, Optional[pd.DataFrame]]:
    """
    Validate that incoming prediction data matches the training feature schema.
    
    Returns:
        (is_valid, error_message, ordered_dataframe)
    """
    try:
        meta = get_model_metadata(model_id)
    except Exception as e:
        return False, f"Could not retrieve metadata for model '{model_id}': {e}", None

    required_features = meta.get("feature_names", [])
    if not required_features:
        return False, f"Model '{model_id}' has no registered feature schema.", None

    missing = [f for f in required_features if f not in input_df.columns]
    if missing:
        return (
            False,
            f"Input data is missing {len(missing)} required feature(s): {', '.join(missing)}.",
            None
        )

    # Reorder and slice to expected features
    ordered_df = input_df[required_features].copy()
    return True, "", ordered_df
