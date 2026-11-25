"""
Clustering Engine using Scikit-Learn KMeans.
Performs unsupervised grouping on normalized numerical features,
supports optional elbow-method inertia sweeps, and generates 2D cluster projections.
"""
import time
import numpy as np
import pandas as pd
import plotly.express as px
from typing import Dict, Any, List, Optional
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA

from ml_engine.preprocessing import clean_dataframe, analyze_features
from ml_engine.evaluation import evaluate_clustering
from ml_engine.model_manager import save_trained_model
from tools.helper_tools import save_chart


def run_clustering(
    df: pd.DataFrame,
    n_clusters: int = 3,
    feature_columns: Optional[List[str]] = None,
    run_elbow: bool = False,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Perform K-Means clustering on numerical features.
    
    Returns cluster profiles, distributions, inertia, and 2D Plotly cluster charts.
    """
    start_time = time.time()

    if df is None or len(df) == 0:
        raise ValueError("Cannot perform clustering on an empty dataset.")

    if len(df) < 10:
        raise ValueError(f"Dataset has only {len(df)} records. Minimum 10 required.")

    cleaned_df = clean_dataframe(df)

    # Determine numerical features
    if feature_columns:
        missing = [c for c in feature_columns if c not in cleaned_df.columns]
        if missing:
            raise ValueError(f"Specified feature column(s) not found: {', '.join(missing)}")
        numeric_cols = [c for c in feature_columns if pd.api.types.is_numeric_dtype(cleaned_df[c])]
        if not numeric_cols:
            raise ValueError("None of the specified columns are numerical.")
    else:
        info = analyze_features(cleaned_df)
        numeric_cols = info["numeric_features"]

    if len(numeric_cols) < 2:
        raise ValueError(f"Clustering requires at least 2 numerical features (found {len(numeric_cols)}).")

    if not (2 <= n_clusters <= min(20, len(df) - 1)):
        raise ValueError(f"Number of clusters must be between 2 and {min(20, len(df) - 1)} (received {n_clusters}).")

    X_numeric = cleaned_df[numeric_cols].copy()

    # Preprocess: median imputation + standard scaling
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    X_imputed = imputer.fit_transform(X_numeric)
    X_scaled = scaler.fit_transform(X_imputed)

    # Optional Elbow Method Analysis (k = 2..8)
    elbow_chart_path = None
    if run_elbow:
        max_k = min(8, len(df) - 1)
        k_values = list(range(2, max_k + 1))
        inertias = []
        for k in k_values:
            km_k = KMeans(n_clusters=k, random_state=random_state, n_init="auto")
            km_k.fit(X_scaled)
            inertias.append(km_k.inertia_)

        elbow_df = pd.DataFrame({"Clusters (K)": k_values, "Inertia": inertias})
        elbow_fig = px.line(
            elbow_df,
            x="Clusters (K)",
            y="Inertia",
            markers=True,
            title="Elbow Method Analysis (Inertia vs. Number of Clusters)"
        )
        elbow_fig.update_layout(xaxis_title="Number of Clusters (K)", yaxis_title="Sum of Squared Distances (Inertia)")
        elbow_chart_path = save_chart(elbow_fig, "elbow_analysis")

    # Fit target K-Means model
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init="auto"
    )
    cluster_labels = kmeans.fit_predict(X_scaled)

    # Calculate cluster sizes
    unique_labels, counts = np.unique(cluster_labels, return_counts=True)
    total_records = len(df)
    cluster_distribution = {
        f"Cluster {label}": {
            "count": int(count),
            "percentage": round((count / total_records) * 100, 2)
        }
        for label, count in zip(unique_labels, counts)
    }

    # Calculate cluster profiles (mean of original unscaled features per cluster)
    profile_df = X_numeric.copy()
    profile_df["Cluster"] = [f"Cluster {c}" for c in cluster_labels]
    profiles = profile_df.groupby("Cluster").mean().round(2).to_dict(orient="index")

    # Evaluate clustering
    eval_metrics = evaluate_clustering(X_scaled, cluster_labels, inertia=kmeans.inertia_)

    # 2D Visualization
    cluster_str_labels = [f"Cluster {c}" for c in cluster_labels]
    if len(numeric_cols) == 2:
        # Direct 2D scatter of the two features
        scatter_df = pd.DataFrame({
            numeric_cols[0]: X_numeric[numeric_cols[0]],
            numeric_cols[1]: X_numeric[numeric_cols[1]],
            "Cluster": cluster_str_labels
        })
        scatter_fig = px.scatter(
            scatter_df,
            x=numeric_cols[0],
            y=numeric_cols[1],
            color="Cluster",
            title=f"K-Means Clusters (K={n_clusters})"
        )
    else:
        # Use PCA 2D projection for >= 3 features
        pca = PCA(n_components=2, random_state=random_state)
        pca_coords = pca.fit_transform(X_scaled)
        explained_var = pca.explained_variance_ratio_.sum() * 100
        scatter_df = pd.DataFrame({
            "PCA Component 1": pca_coords[:, 0],
            "PCA Component 2": pca_coords[:, 1],
            "Cluster": cluster_str_labels
        })
        scatter_fig = px.scatter(
            scatter_df,
            x="PCA Component 1",
            y="PCA Component 2",
            color="Cluster",
            title=f"K-Means Clusters (PCA 2D Projection, {explained_var:.1f}% Variance Explained)"
        )

    cluster_chart_path = save_chart(scatter_fig, "kmeans_clusters")

    duration = round(time.time() - start_time, 3)

    # Persist model
    metadata = save_trained_model(
        model=kmeans,
        preprocessor={"imputer": imputer, "scaler": scaler},
        model_type="clustering",
        algorithm="KMeans",
        feature_names=numeric_cols,
        metrics={
            "n_clusters": n_clusters,
            "inertia": eval_metrics.get("inertia"),
            "silhouette_score": eval_metrics.get("silhouette_score"),
            "total_records": total_records,
            "training_duration_seconds": duration
        },
        params={"n_clusters": n_clusters, "random_state": random_state},
        description=f"K-Means clustered on {len(numeric_cols)} numeric features."
    )

    charts = {"cluster_scatter_path": cluster_chart_path}
    if elbow_chart_path:
        charts["elbow_path"] = elbow_chart_path

    return {
        "success": True,
        "model_id": metadata["model_id"],
        "algorithm": "KMeans",
        "n_clusters": n_clusters,
        "features_used": numeric_cols,
        "metrics": eval_metrics,
        "cluster_distribution": cluster_distribution,
        "cluster_profiles": profiles,
        "charts": charts,
        "training_duration": duration
    }
