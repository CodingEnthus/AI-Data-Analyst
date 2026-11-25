"""
Reusable preprocessing engine for machine learning tasks.
Provides automated column identification, missing value imputation,
categorical encoding, scaling, and strict data-leakage prevention.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split


def analyze_features(
    df: pd.DataFrame,
    target_column: Optional[str] = None
) -> Dict[str, Any]:
    """
    Inspect a DataFrame and partition columns into usable feature subsets,
    identifying edge cases such as constant columns and all-NaN columns.
    """
    analysis = {
        "numeric_features": [],
        "categorical_features": [],
        "constant_columns": [],
        "empty_columns": [],
        "high_cardinality_columns": [],
        "warnings": []
    }

    cols_to_check = [c for c in df.columns if c != target_column]

    for col in cols_to_check:
        series = df[col]
        # Check if entirely empty
        if series.isna().all():
            analysis["empty_columns"].append(col)
            analysis["warnings"].append(f"Column '{col}' is entirely empty and was excluded.")
            continue

        # Replace infinities to assess unique values safely
        valid_series = series.replace([np.inf, -np.inf], np.nan).dropna()
        if valid_series.nunique() <= 1:
            analysis["constant_columns"].append(col)
            analysis["warnings"].append(f"Column '{col}' is constant (zero variance) and was excluded.")
            continue

        if pd.api.types.is_numeric_dtype(series):
            analysis["numeric_features"].append(col)
        elif pd.api.types.is_object_dtype(series) or pd.api.types.is_categorical_dtype(series) or pd.api.types.is_bool_dtype(series):
            # Check cardinality: if unique values are over 50 or over 80% of rows, treat as high-cardinality/identifier
            unique_count = series.nunique()
            if unique_count > 50 or (len(df) > 20 and unique_count / len(df) > 0.8):
                analysis["high_cardinality_columns"].append(col)
                analysis["warnings"].append(f"Column '{col}' has high cardinality ({unique_count} unique values) and was excluded.")
            else:
                analysis["categorical_features"].append(col)
        else:
            analysis["warnings"].append(f"Column '{col}' has unsupported dtype '{series.dtype}' and was excluded.")

    return analysis


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Replace infinite values with NaN while preserving original indices."""
    cleaned = df.copy()
    cleaned = cleaned.replace([np.inf, -np.inf], np.nan)
    return cleaned


def build_preprocessor_pipeline(
    numeric_features: List[str],
    categorical_features: List[str],
    scale_numeric: bool = True
) -> ColumnTransformer:
    """
    Construct a Scikit-Learn ColumnTransformer pipeline.
    Imputes median for numeric and most_frequent for categorical.
    """
    transformers = []

    if numeric_features:
        num_steps = [("imputer", SimpleImputer(strategy="median"))]
        if scale_numeric:
            num_steps.append(("scaler", StandardScaler()))
        transformers.append(("num", Pipeline(steps=num_steps), numeric_features))

    if categorical_features:
        cat_steps = [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ]
        transformers.append(("cat", Pipeline(steps=cat_steps), categorical_features))

    return ColumnTransformer(transformers=transformers, remainder="drop")


def prepare_supervised_data(
    df: pd.DataFrame,
    target_column: str,
    test_size: float = 0.2,
    is_classification: bool = True,
    scale_numeric: bool = False,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Split and preprocess dataset for supervised learning with strict leakage prevention.
    Transformers are fitted on X_train ONLY.
    
    Returns a dictionary with train/test splits, transformed matrices,
    fitted preprocessor, feature names, and metadata.
    """
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' does not exist in the dataset.")

    # Clean infinities
    cleaned_df = clean_dataframe(df)

    # Drop rows where target is missing
    valid_mask = cleaned_df[target_column].notna()
    dropped_target_nulls = int((~valid_mask).sum())
    working_df = cleaned_df[valid_mask]

    if len(working_df) < 10:
        raise ValueError(f"Dataset has only {len(working_df)} valid records after target cleaning. Minimum 10 required.")

    analysis = analyze_features(working_df, target_column=target_column)
    usable_features = analysis["numeric_features"] + analysis["categorical_features"]

    if not usable_features:
        raise ValueError("No usable features found after excluding constant, empty, or high-cardinality columns.")

    X = working_df[usable_features]
    y = working_df[target_column]

    # Verify target for task
    if is_classification:
        unique_targets = y.nunique()
        if unique_targets < 2:
            raise ValueError(f"Classification target '{target_column}' must have at least 2 distinct classes (found {unique_targets}).")
        if unique_targets > 20 and pd.api.types.is_numeric_dtype(y):
            # Check if this might actually be a regression task
            analysis["warnings"].append(
                f"Target '{target_column}' has {unique_targets} unique numerical values. "
                "Ensure this is intended as a classification task rather than regression."
            )

        # Check stratification eligibility
        class_counts = y.value_counts()
        can_stratify = (class_counts.min() >= 2) and (test_size * len(working_df) >= len(class_counts))
        stratify_y = y if can_stratify else None
        if not can_stratify and class_counts.min() < 2:
            analysis["warnings"].append("Some classes have fewer than 2 instances; stratifying split was disabled.")
    else:
        # Regression checks
        if not pd.api.types.is_numeric_dtype(y):
            raise ValueError(f"Regression target '{target_column}' must be numeric (found dtype '{y.dtype}').")
        if y.nunique() <= 1:
            raise ValueError(f"Regression target '{target_column}' has zero variance (constant value).")
        stratify_y = None

    # Train-test split (maintaining indices)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_y
    )

    # Build and fit preprocessor ON TRAINING DATA ONLY (Leakage Prevention)
    preprocessor = build_preprocessor_pipeline(
        analysis["numeric_features"],
        analysis["categorical_features"],
        scale_numeric=scale_numeric
    )
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)

    # Retrieve transformed feature names
    transformed_feature_names = []
    if hasattr(preprocessor, "get_feature_names_out"):
        try:
            raw_names = preprocessor.get_feature_names_out()
            # Clean prefixes (num__, cat__)
            transformed_feature_names = [
                n.replace("num__", "").replace("cat__", "") for n in raw_names
            ]
        except Exception:
            transformed_feature_names = [f"feat_{i}" for i in range(X_train_proc.shape[1])]
    else:
        transformed_feature_names = [f"feat_{i}" for i in range(X_train_proc.shape[1])]

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "X_train_proc": X_train_proc,
        "X_test_proc": X_test_proc,
        "preprocessor": preprocessor,
        "usable_features": usable_features,
        "numeric_features": analysis["numeric_features"],
        "categorical_features": analysis["categorical_features"],
        "transformed_feature_names": transformed_feature_names,
        "dropped_target_nulls": dropped_target_nulls,
        "train_indices": list(X_train.index),
        "test_indices": list(X_test.index),
        "warnings": analysis["warnings"]
    }
