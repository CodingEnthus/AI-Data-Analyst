# 📋 Repository Component Inventory

This document provides a factual, verified inventory of all software components, frameworks, tools, libraries, data structures, and external services in the **AI Data Analyst** repository.

---

## 1. High-Level Component Inventory

| Component | File / Location | Technology | Responsibility | Communicates With | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **User Interface & Application Shell** | `app.py` | Streamlit 1.59.2 | Renders web UI, handles file upload, displays chat, coordinates session state, mounts Plotly charts, provides ML Studio GUI. | `data_store.py`, `agent.py`, `ml_engine/*`, `outputs/charts/*`, `models/*` | Active |
| **Agent Core & Graph Engine** | `agent.py` | LangGraph 1.2.9, LangChain-Core 1.4.9 | Compiles StateGraph, manages nodes (`chatbot`, `tools`), coordinates memory checkpointer, binds tools to LLM. | `prompts.py`, `tools/*`, Groq API / Ollama, `langgraph.checkpoint.memory.MemorySaver` | Active |
| **System Prompt Definition** | `prompts.py` | Python string constant | Defines instructions, operational boundaries, anti-hallucination rules, and tool descriptions for the LLM. | `agent.py` | Active |
| **Global Data Store** | `data_store.py` | Python module singleton | Holds in-memory global reference to active Pandas DataFrame (`df`). | `app.py`, `tools/*`, `ml_engine/*` | Active |
| **Fuzzy Matching & Chart Utilities** | `tools/helper_tools.py` | `difflib`, `uuid`, `os` | Matches misspelled column names (`difflib.SequenceMatcher`), validates chart axes, persists Plotly figures to HTML. | `data_store.py`, `tools/statistics_tools.py`, `tools/visualization_tools.py`, `tools/ml_tools.py` | Active |
| **Dataset Ingestion Tools** | `tools/dataset_tools.py` | Pandas, `langchain_core.tools` | Implements `load_dataset` and `get_dataset_info` tools. Reads CSV/Excel and inspects shapes, dtypes, null counts. | `data_store.py`, `agent.py` | Active |
| **Descriptive Statistics Tools** | `tools/statistics_tools.py` | Pandas, NumPy | Implements `calculate_statistics` (mean, median, sum, min, max, std, variance, mode, nunique). | `data_store.py`, `tools/helper_tools.py`, `agent.py` | Active |
| **Analytical SQL Engine** | `tools/duckdb_tools.py` | DuckDB 1.5.4, Regex | Implements `query_dataset_with_sql`. Sanitizes SQL queries (read-only enforcement) and executes on in-memory DataFrame. | `data_store.py`, `duckdb`, `agent.py` | Active |
| **Interactive Visualization Tools** | `tools/visualization_tools.py` | Plotly Express 6.9.0 | Implements 8 chart tools (bar, line, scatter, histogram, pie, heatmap, box, violin). Exports standalone HTML. | `data_store.py`, `tools/helper_tools.py`, `outputs/charts/*`, `agent.py` | Active |
| **Machine Learning Tools Adapter** | `tools/ml_tools.py` | LangChain `@tool`, JSON | Adapts `ml_engine` functions for LangGraph tool-calling (anomaly detection, classification, regression, clustering, inference). | `data_store.py`, `ml_engine/*`, `agent.py` | Active |
| **ML Data Preprocessing Pipeline** | `ml_engine/preprocessing.py` | Scikit-Learn (ColumnTransformer, Pipeline, Imputer, Scaler, OneHotEncoder) | Analyzes feature types, drops zero-variance and high-cardinality IDs, splits data, prevents leakage. | `data_store.df`, `ml_engine/*` | Active |
| **Supervised Classification Engine** | `ml_engine/classification.py` | Scikit-Learn (RandomForestClassifier), Plotly | Trains 100-tree classification forest, evaluates accuracy/precision/recall/F1, plots confusion matrix & feature importances. | `ml_engine/preprocessing.py`, `ml_engine/evaluation.py`, `ml_engine/model_manager.py` | Active |
| **Supervised Regression Engine** | `ml_engine/regression.py` | Scikit-Learn (RandomForestRegressor), Plotly | Trains 100-tree regression forest, computes MAE/MSE/RMSE/R2, plots actual vs. predicted and residuals distribution. | `ml_engine/preprocessing.py`, `ml_engine/evaluation.py`, `ml_engine/model_manager.py` | Active |
| **Unsupervised Anomaly Detection** | `ml_engine/anomaly_detection.py` | Scikit-Learn (IsolationForest), Plotly | Fits Isolation Forest with configurable contamination (default 0.05), flags outlier records, plots score distribution. | `ml_engine/preprocessing.py`, `ml_engine/model_manager.py`, `outputs/charts/*` | Active |
| **Unsupervised Clustering Engine** | `ml_engine/clustering.py` | Scikit-Learn (KMeans, PCA, StandardScaler), Plotly | Normalizes features, runs K-Means, calculates Silhouette Score, performs 2D PCA projection for cluster scatter plotting. | `ml_engine/preprocessing.py`, `ml_engine/evaluation.py`, `ml_engine/model_manager.py` | Active |
| **Model Evaluation Engine** | `ml_engine/evaluation.py` | Scikit-Learn Metrics | Centralizes metric computation for classification, regression, and clustering. | `ml_engine/classification.py`, `ml_engine/regression.py`, `ml_engine/clustering.py` | Active |
| **Model Persistence Manager** | `ml_engine/model_manager.py` | `joblib`, `json` | Serializes trained model pipelines to `models/*.joblib` and companion metadata to `models/*.json`. | `models/`, `tools/ml_tools.py`, `app.py` | Active |
| **Legacy / Unused Tools Stub** | `tools.py` (root) | None (7-line stub) | Old placeholder file before modularization into `tools/` package. | None (Unused) | Deprecated / Dormant |
| **Test Suite** | `tests/test_ml_engine.py`, `tests/run_tests.py` | Python `unittest` style assertions | Unit and integration testing for all ML algorithms, data leakage verification, and metric correctness. | `ml_engine/*`, `data_store.py` | Active Test Harness |

---

## 2. Inventory of the 18 Registered Agent Tools

All 18 tools are imported and registered into `agent.py` in the `tools` list:

```python
# Registered in agent.py (Lines 58-77)
tools = [
    load_dataset,                # tools/dataset_tools.py
    get_dataset_info,            # tools/dataset_tools.py
    calculate_statistics,        # tools/statistics_tools.py
    query_dataset_with_sql,      # tools/duckdb_tools.py
    create_bar_chart,            # tools/visualization_tools.py
    create_line_chart,           # tools/visualization_tools.py
    create_scatter_plot,         # tools/visualization_tools.py
    create_histogram,            # tools/visualization_tools.py
    create_pie_chart,            # tools/visualization_tools.py
    create_heatmap,              # tools/visualization_tools.py
    create_box_plot,             # tools/visualization_tools.py
    create_violin_plot,          # tools/visualization_tools.py
    detect_anomalies,            # tools/ml_tools.py
    train_classification_model,  # tools/ml_tools.py
    train_regression_model,      # tools/ml_tools.py
    perform_clustering,          # tools/ml_tools.py
    evaluate_trained_model,      # tools/ml_tools.py
    predict_with_model           # tools/ml_tools.py
]
```

*(Note: `find_best_matching_column` in `tools/helper_tools.py` is decorated with `@tool`, but is used as an internal helper tool by other tools rather than registered directly in the agent's top-level tools list).*

---

## 3. Libraries, Frameworks & External Services

### Local Runtimes (100% On-Premise / In-Memory):
- **Streamlit**: Web rendering and session manager.
- **LangGraph**: StateGraph orchestration, conditional edges, memory saver.
- **Pandas**: In-memory tabular transformations and aggregations.
- **DuckDB**: Columnar SQL engine executing in-process (`database=":memory:"`).
- **Scikit-Learn**: IsolationForest, RandomForestClassifier, RandomForestRegressor, KMeans, PCA, metrics.
- **Plotly Express & Graph Objects**: Vectorized chart compilation to HTML.
- **Joblib**: Disk serialization for trained Scikit-learn pipelines.

### External / Remote Services:
- **Groq API Cloud** (`https://api.groq.com`): Remote LLM endpoint (default: `qwen/qwen3.8-27b`, alternative: `openai/gpt-oss-120b`). Uses `langchain_groq.ChatGroq`.
- **Ollama (Optional Local LLM)**: Local inference server (`http://localhost:11434`, model: `qwen3:4b`). Uses `langchain_ollama.ChatOllama`.

---

## 4. Shared Data Structures

1. `data_store.df`: In-memory `pandas.DataFrame`. Accessible globally across the process.
2. `State(TypedDict)`: LangGraph state schema:
   ```python
   class State(TypedDict):
       messages: Annotated[list, add_messages]
   ```
3. `st.session_state`: Streamlit user state holding `thread_id` (UUID), `messages` (chat log), `dataset_name`, and cached ML results (`ml_anomaly_res`, `ml_classification_res`, etc.).
4. `models/<model_id>.joblib`: Serialized Scikit-learn pipeline dictionary containing the fitted `ColumnTransformer` and the trained estimator.
5. `models/<model_id>.json`: JSON metadata holding feature names, hyperparameters, target column, metrics, and training timestamp.
6. `outputs/charts/*.html`: Standalone Plotly HTML files containing bundled D3/Plotly JavaScript for interactive browser rendering.
