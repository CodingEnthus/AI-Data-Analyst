# 🔬 Machine Learning Engine Workflows

This document details the exact execution paths, data preprocessing, mathematical training, evaluation, and serialization steps for all 4 Machine Learning algorithms implemented in the repository.

---

## 1. Random Forest Classification Workflow

### Architecture (Mermaid)

```mermaid
flowchart TD
    UserReq["👤 User: 'Train a model to predict Survived'"]
    Agent["🧠 Agent (tools/ml_tools.py: train_classification_model)"]
    
    subgraph PreprocessingStage ["ml_engine/preprocessing.py: prepare_supervised_data()"]
        CleanInf["clean_dataframe(): Replace inf with NaN"]
        DropTargetNull["Drop rows where target is NaN"]
        FeatAnalysis["analyze_features(): Partition numeric vs categorical"]
        DropBadCols["Exclude constant columns & high-cardinality IDs (>50 unique or >80% unique)"]
        Split["train_test_split(X, y, test_size=0.2, random_state=42)"]
        BuildPipeline["build_preprocessor_pipeline():<br/>- SimpleImputer(median) for numeric<br/>- SimpleImputer(most_frequent) + OneHotEncoder for categorical"]
        FitTrain["X_train_proc = preprocessor.fit_transform(X_train)"]
        TransformTest["X_test_proc = preprocessor.transform(X_test) (LEAKAGE FREE)"]
    end

    subgraph TrainingStage ["ml_engine/classification.py: run_classification()"]
        CheckBalance["Class distribution check (class_weight='balanced' if minority < 20%)"]
        FitRF["clf = RandomForestClassifier(n_estimators=100, random_state=42)<br/>clf.fit(X_train_proc, y_train)"]
        PredictTest["y_pred = clf.predict(X_test_proc)"]
    end

    subgraph EvalStage ["ml_engine/evaluation.py & Plotly Charts"]
        MetricsCalc["Calculate Accuracy, Precision, Recall, F1-Score"]
        CMPlot["px.imshow(confusion_matrix) -> outputs/charts/confusion_matrix_xxx.html"]
        FeatImpPlot["Calculate Gini Importances -> px.bar() -> outputs/charts/feat_imp_xxx.html"]
    end

    subgraph PersistenceStage ["ml_engine/model_manager.py: save_trained_model()"]
        SaveJoblib["joblib.dump(pipeline, 'models/clf_{model_id}.joblib')"]
        SaveJSON["json.dump(metadata, 'models/clf_{model_id}.json')"]
    end

    UserReq --> Agent
    Agent --> CleanInf
    CleanInf --> DropTargetNull
    DropTargetNull --> FeatAnalysis
    FeatAnalysis --> DropBadCols
    DropBadCols --> Split
    Split --> BuildPipeline
    BuildPipeline --> FitTrain
    FitTrain --> TransformTest
    
    TransformTest --> CheckBalance
    CheckBalance --> FitRF
    FitRF --> PredictTest
    PredictTest --> MetricsCalc
    
    MetricsCalc --> CMPlot
    MetricsCalc --> FeatImpPlot
    MetricsCalc --> SaveJoblib
    SaveJoblib --> SaveJSON
    
    SaveJSON --> AgentResponse["Return JSON: {accuracy: 0.804, top_features: [...], chart_paths: [...]}"]
    AgentResponse --> UserFinal(["👤 Conversational Summary + Confusion Matrix + Feature Importance Bar Chart"])
```

### Key Technical Details
- **Leakage Prevention**: Transformations are fitted exclusively on `X_train`. `X_test` is transformed using pre-computed statistics.
- **Handling Class Imbalance**: Automatically applies `class_weight="balanced"` if the minority class represents less than 20% of training data ([classification.py#L63](file:///c:/Users/User/Desktop/AI-Data-Analyst/ml_engine/classification.py#L63)).
- **Model Serialization**: Saves both the Scikit-learn estimator and fitted preprocessor into `models/clf_{id}.joblib`.

---

## 2. Random Forest Regression Workflow

### Architecture (Mermaid)

```mermaid
flowchart TD
    UserReq["👤 User: 'Predict Fare based on other features'"]
    Agent["🧠 Agent (tools/ml_tools.py: train_regression_model)"]

    subgraph PreprocessingStage ["ml_engine/preprocessing.py"]
        DataPrep["prepare_supervised_data(is_classification=False)"]
        SplitData["80% X_train, 20% X_test split"]
        PipeTransform["Impute median on numeric, one-hot encode categorical"]
    end

    subgraph RegressionTraining ["ml_engine/regression.py: run_regression()"]
        FitReg["reg = RandomForestRegressor(n_estimators=100, random_state=42)<br/>reg.fit(X_train_proc, y_train)"]
        TestPredict["y_pred = reg.predict(X_test_proc)"]
    end

    subgraph RegressionEval ["ml_engine/evaluation.py & Visualizations"]
        CalcMetrics["Calculate MAE, MSE, RMSE, R-squared (R2)"]
        ActVsPredPlot["px.scatter(x=y_test, y=y_pred) + Reference diagonal line"]
        ResidualPlot["px.histogram(residuals = y_test - y_pred) -> residual_dist_xxx.html"]
        FeatImpBar["Top 15 Feature Importances Bar Chart"]
    end

    subgraph ModelStorage ["ml_engine/model_manager.py"]
        PersistReg["Save models/reg_{model_id}.joblib & models/reg_{model_id}.json"]
    end

    UserReq --> Agent
    Agent --> DataPrep
    DataPrep --> SplitData
    SplitData --> PipeTransform
    PipeTransform --> FitReg
    FitReg --> TestPredict
    TestPredict --> CalcMetrics
    
    CalcMetrics --> ActVsPredPlot
    CalcMetrics --> ResidualPlot
    CalcMetrics --> FeatImpBar
    CalcMetrics --> PersistReg
    
    PersistReg --> SummaryJSON["Return JSON: {mae: 14.2, r2: 0.68, charts: [...]}"]
    SummaryJSON --> FinalOutput(["👤 Conversational Metrics + Residuals Chart + Feature Importance Chart"])
```

### Key Technical Details
- **Evaluation Metrics**: `MAE`, `MSE`, `RMSE`, and $R^2$ score computed via `evaluate_regression()`.
- **Diagnostic Charts**: Generates both an **Actual vs. Predicted** scatter plot with a 45-degree reference line and a **Residuals Distribution Histogram** to verify normal error distribution.

---

## 3. Isolation Forest Anomaly Detection Workflow

### Architecture (Mermaid)

```mermaid
flowchart TD
    UserReq["👤 User: 'Detect unusual records and outliers'"]
    Agent["🧠 Agent (tools/ml_tools.py: detect_anomalies)"]

    subgraph AnomalyEngine ["ml_engine/anomaly_detection.py: run_anomaly_detection()"]
        CleanInfs["clean_dataframe(): Replace infinities with NaN"]
        ValidateContam{"Contamination between 0.001 and 0.5?"}
        SelectNumeric["Extract numeric features only (analyze_features)"]
        ImputeMedian["SimpleImputer(strategy='median') on numeric features"]
        FitIso["iso = IsolationForest(contamination=contamination, n_estimators=100, random_state=42)<br/>iso.fit(X_proc)"]
        PredictOutliers["preds = iso.predict(X_proc) (-1 = Anomaly, 1 = Normal)"]
        ScoreSamples["scores = iso.score_samples(X_proc)"]
    end

    subgraph ArtifactCreation ["Artifact Generation & Visualization"]
        FlagIndices["Identify original DataFrame row indices where pred == -1"]
        AnomalyCount["Count total anomalies & calculate percentage"]
        ScoreHist["px.histogram(scores, color=labels) -> outputs/charts/anomaly_scores_xxx.html"]
        SaveModel["Save models/iso_forest_{id}.joblib & .json"]
    end

    UserReq --> Agent
    Agent --> CleanInfs
    CleanInfs --> ValidateContam
    ValidateContam -->|Valid (e.g. 0.05)| SelectNumeric
    SelectNumeric --> ImputeMedian
    ImputeMedian --> FitIso
    FitIso --> PredictOutliers
    PredictOutliers --> ScoreSamples
    ScoreSamples --> FlagIndices
    FlagIndices --> AnomalyCount
    AnomalyCount --> ScoreHist
    ScoreHist --> SaveModel
    
    SaveModel --> RetJSON["Return JSON: {anomaly_count: 45, percentage: 5.05%, chart_path: '...'}"]
    RetJSON --> UserResult(["👤 Summary Text + Top Flagged Row Indices + Anomaly Score Histogram"])
```

### Key Technical Details
- **Contamination Bounds**: Strict validation ensures `0.001 <= contamination <= 0.5`.
- **Index Preservation**: Original row indices are preserved so users can inspect the exact anomalous records in the raw dataset.

---

## 4. K-Means Clustering Workflow

### Architecture (Mermaid)

```mermaid
flowchart TD
    UserReq["👤 User: 'Group customers into 3 clusters'"]
    Agent["🧠 Agent (tools/ml_tools.py: perform_clustering)"]

    subgraph ClusteringEngine ["ml_engine/clustering.py: run_clustering()"]
        CleanData["clean_dataframe()"]
        CheckK{"2 <= n_clusters <= 20?"}
        FilterNumeric["Extract numerical features only (requires >= 2 features)"]
        ImputeMissing["SimpleImputer(strategy='median')"]
        StandardScale["StandardScaler(): Standardize to zero mean and unit variance"]
        FitKMeans["kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)<br/>labels = kmeans.fit_predict(X_scaled)"]
    end

    subgraph ClusterEvaluation ["Evaluation & Dimensionality Reduction"]
        SilScore["silhouette_score(X_scaled, labels)"]
        CentroidInv["Inverse transform cluster centroids back to original feature scale"]
        PCAReduce["PCA(n_components=2): Project features to PC1 and PC2"]
        Plot2D["px.scatter(x=PC1, y=PC2, color=cluster_labels) -> outputs/charts/kmeans_clusters_xxx.html"]
        SaveKMeans["Save models/kmeans_{id}.joblib & .json"]
    end

    UserReq --> Agent
    Agent --> CleanData
    CleanData --> CheckK
    CheckK -->|Valid| FilterNumeric
    FilterNumeric --> ImputeMissing
    ImputeMissing --> StandardScale
    StandardScale --> FitKMeans
    FitKMeans --> SilScore
    SilScore --> CentroidInv
    CentroidInv --> PCAReduce
    PCAReduce --> Plot2D
    Plot2D --> SaveKMeans
    
    SaveKMeans --> RetClusterJSON["Return JSON: {silhouette_score: 0.58, cluster_counts: [...], chart_path: '...'}"]
    RetClusterJSON --> UserResultCluster(["👤 Cluster Profiles Table + Silhouette Score + 2D PCA Scatter Plot"])
```

### Key Technical Details
- **Feature Standardization**: `StandardScaler()` is mandatory to prevent high-magnitude features from dominating Euclidean distance.
- **2D PCA Projection**: When datasets have $>2$ numerical features, `sklearn.decomposition.PCA(n_components=2)` projects the feature space onto 2 principal components to render a 2D scatter plot with cluster coloring.
- **Centroid Inversion**: Cluster centers are transformed back to original physical units (`scaler.inverse_transform`) so users can interpret real-world averages for each group.
