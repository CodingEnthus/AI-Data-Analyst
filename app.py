import os
import uuid
import json
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# App directories
os.makedirs("data", exist_ok=True)
os.makedirs("outputs/charts", exist_ok=True)
os.makedirs("models", exist_ok=True)

import data_store
from agent import graph, tools
from tools.dataset_tools import load_dataset
from langchain_core.messages import HumanMessage, ToolMessage, AIMessage

# ML Engine Imports
from ml_engine.anomaly_detection import run_anomaly_detection
from ml_engine.classification import run_classification
from ml_engine.regression import run_regression
from ml_engine.clustering import run_clustering
from ml_engine.model_manager import list_trained_models, get_model_metadata
from ml_engine.preprocessing import analyze_features

# --- Page Configuration ---
st.set_page_config(
    page_title="AI Data Analyst & ML Studio",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        color: #6c757d;
        font-size: 1.05rem;
        margin-bottom: 1.2rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 12px 16px;
        border: 1px solid #e9ecef;
    }
    .badge-success {
        display: inline-block;
        padding: 4px 8px;
        font-size: 0.85rem;
        font-weight: 600;
        color: #155724;
        background-color: #d4edda;
        border-radius: 4px;
    }
    .badge-warning {
        display: inline-block;
        padding: 4px 8px;
        font-size: 0.85rem;
        font-weight: 600;
        color: #856404;
        background-color: #fff3cd;
        border-radius: 4px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- Session State Initialization ---
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

if "dataset_name" not in st.session_state:
    st.session_state.dataset_name = None

# Store ML Studio intermediate results in session state
if "ml_anomaly_res" not in st.session_state:
    st.session_state.ml_anomaly_res = None
if "ml_classification_res" not in st.session_state:
    st.session_state.ml_classification_res = None
if "ml_regression_res" not in st.session_state:
    st.session_state.ml_regression_res = None
if "ml_clustering_res" not in st.session_state:
    st.session_state.ml_clustering_res = None

# Default Titanic dataset if nothing loaded yet
if data_store.df is None:
    sample_titanic = os.path.join("data", "Titanic-Dataset.csv")
    if os.path.exists(sample_titanic):
        try:
            data_store.df = pd.read_csv(sample_titanic)
            st.session_state.dataset_name = "Titanic-Dataset.csv"
        except Exception:
            pass

# --- Helper Functions ---
def reset_conversation():
    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.messages = []
    st.rerun()

def load_file_to_datastore(file_path: str, display_name: str):
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".csv":
        df = pd.read_csv(file_path)
    elif ext in [".xlsx", ".xls"]:
        df = pd.read_excel(file_path)
    else:
        raise ValueError("Unsupported file format. Please upload CSV or Excel.")

    data_store.df = df
    st.session_state.dataset_name = display_name
    # Clear any previous ML results for fresh dataset
    st.session_state.ml_anomaly_res = None
    st.session_state.ml_classification_res = None
    st.session_state.ml_regression_res = None
    st.session_state.ml_clustering_res = None

    # Notify agent state of new dataset
    config = {"configurable": {"thread_id": st.session_state.thread_id}}
    try:
        graph.invoke({"messages": [HumanMessage(content=f"System Notification: Loaded dataset '{display_name}' ({df.shape[0]} rows, {df.shape[1]} columns).")]}, config=config)
    except Exception:
        pass

def extract_chart_paths_from_response(response_dict):
    """Find any chart HTML files created by visualization or ML tools in this turn."""
    chart_paths = []
    messages = response_dict.get("messages", [])
    for msg in messages:
        if isinstance(msg, ToolMessage):
            content = msg.content
            if isinstance(content, str):
                # Search for chart_path or multiple paths in json
                try:
                    data = json.loads(content)
                    if isinstance(data, dict):
                        # Single chart path
                        if data.get("chart_path"):
                            c_path = os.path.normpath(data["chart_path"])
                            if os.path.exists(c_path) and c_path not in chart_paths:
                                chart_paths.append(c_path)
                        # Nested charts dict
                        if isinstance(data.get("charts"), dict):
                            for p in data["charts"].values():
                                c_path = os.path.normpath(p)
                                if os.path.exists(c_path) and c_path not in chart_paths:
                                    chart_paths.append(c_path)
                        # Feature importance chart
                        if data.get("feature_importance_chart_path"):
                            c_path = os.path.normpath(data["feature_importance_chart_path"])
                            if os.path.exists(c_path) and c_path not in chart_paths:
                                chart_paths.append(c_path)
                except Exception:
                    pass
    return chart_paths

def render_html_chart(chart_path: str, height: int = 500):
    """Embed an interactive Plotly HTML chart with a download button."""
    if os.path.exists(chart_path):
        try:
            with open(chart_path, "r", encoding="utf-8") as f:
                html_content = f.read()
            components.html(html_content, height=height, scrolling=True)
            with open(chart_path, "rb") as f:
                st.download_button(
                    label=f"💾 Download Chart ({os.path.basename(chart_path)})",
                    data=f,
                    file_name=os.path.basename(chart_path),
                    mime="text/html",
                    key=f"dl_chart_{os.path.basename(chart_path)}_{uuid.uuid4().hex[:6]}"
                )
        except Exception as e:
            st.warning(f"Could not render chart: {e}")

# --- Sidebar ---
with st.sidebar:
    st.title("⚙️ Configuration")

    # API Status
    groq_api_key = os.getenv("GROQ_API_KEY")
    if groq_api_key:
        st.markdown('<span class="badge-success">● Groq API Connected</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="badge-warning">⚠ GROQ_API_KEY Not Found</span>', unsafe_allow_html=True)
        user_key = st.text_input("Enter Groq API Key:", type="password")
        if user_key:
            os.environ["GROQ_API_KEY"] = user_key
            st.rerun()

    model_options = ["qwen/qwen3.8-27b", "openai/gpt-oss-120b", "openai/gpt-oss-20b"]
    curr_model = os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")
    default_idx = model_options.index(curr_model) if curr_model in model_options else 0
    selected_model = st.selectbox("Groq Model", model_options, index=default_idx)
    if selected_model != os.getenv("GROQ_MODEL"):
        os.environ["GROQ_MODEL"] = selected_model

    st.divider()

    # Dataset Section
    st.subheader("📂 Dataset")
    
    dataset_source = st.radio(
        "Choose Data Source:",
        ["Sample Dataset (Titanic)", "Upload Custom File"],
        index=0 if st.session_state.dataset_name == "Titanic-Dataset.csv" else 1
    )

    if dataset_source == "Sample Dataset (Titanic)":
        sample_path = os.path.join("data", "Titanic-Dataset.csv")
        if st.button("Load Titanic Dataset", use_container_width=True):
            if os.path.exists(sample_path):
                load_file_to_datastore(sample_path, "Titanic-Dataset.csv")
                st.success("Loaded Titanic-Dataset.csv!")
                st.rerun()
            else:
                st.error("Titanic-Dataset.csv not found in data/ folder.")
    else:
        uploaded_file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx", "xls"])
        if uploaded_file is not None:
            if st.session_state.dataset_name != uploaded_file.name:
                saved_path = os.path.join("data", uploaded_file.name)
                with open(saved_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                load_file_to_datastore(saved_path, uploaded_file.name)
                st.success(f"Uploaded and loaded: {uploaded_file.name}")
                st.rerun()

    # Dataset Summary Metrics
    if data_store.df is not None:
        st.divider()
        st.subheader("📊 Dataset Overview")
        st.write(f"**File:** `{st.session_state.dataset_name}`")
        c1, c2 = st.columns(2)
        c1.metric("Rows", f"{data_store.df.shape[0]:,}")
        c2.metric("Columns", f"{data_store.df.shape[1]}")
        
        missing_count = int(data_store.df.isna().sum().sum())
        st.metric("Missing Values", f"{missing_count:,}")

        with st.expander("🔍 View Data Preview"):
            st.dataframe(data_store.df.head(10), use_container_width=True)
            st.caption("Column Types:")
            st.json({col: str(dtype) for col, dtype in data_store.df.dtypes.items()})

    st.divider()
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        reset_conversation()

# --- Main App Header ---
st.markdown('<div class="main-title">📊 AI Data Analyst & ML Studio</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Conversational data analysis, DuckDB SQL queries, and a dedicated Scikit-Learn Machine Learning Studio.</div>', unsafe_allow_html=True)

if data_store.df is not None:
    st.caption(f"Active Dataset: **{st.session_state.dataset_name}** ({data_store.df.shape[0]:,} rows × {data_store.df.shape[1]} columns)")
else:
    st.info("👋 Welcome! Please select or upload a dataset in the sidebar to start analyzing.")

# --- Top-Level Navigation Tabs ---
main_tab_chat, main_tab_ml = st.tabs(["💬 Conversational Analyst", "🤖 Machine Learning Studio"])

# =====================================================================
# TAB 1: CONVERSATIONAL ANALYST
# =====================================================================
with main_tab_chat:
    # Quick Suggestion Buttons
    st.markdown("##### 💡 Quick Prompts")
    q_col1, q_col2, q_col3, q_col4, q_col5 = st.columns(5)

    suggested_prompt = None
    if q_col1.button("📋 Dataset Info", use_container_width=True):
        suggested_prompt = "Show dataset information"
    if q_col2.button("📈 Age Box Plot", use_container_width=True):
        suggested_prompt = "Create a box plot of Age grouped by Pclass"
    if q_col3.button("🔍 SQL Query", use_container_width=True):
        suggested_prompt = "Using SQL, calculate the survival rate and passenger count for each passenger class."
    if q_col4.button("🚨 Find Anomalies", use_container_width=True):
        suggested_prompt = "Detect unusual records and outliers in this dataset using Isolation Forest."
    if q_col5.button("🎯 Train Classifier", use_container_width=True):
        suggested_prompt = "Train a classification model to predict Survived and show its accuracy and feature importances."

    # Display Chat History
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("charts"):
                for chart_file in msg["charts"]:
                    render_html_chart(chart_file)

    # Chat Input
    user_prompt = st.chat_input("Ask a question, query with SQL, request a chart, or train an ML model...")
    active_input = suggested_prompt if suggested_prompt else user_prompt

    if active_input:
        st.session_state.messages.append({"role": "user", "content": active_input})
        with st.chat_message("user"):
            st.markdown(active_input)

        with st.chat_message("assistant"):
            with st.spinner("Analyzing data and executing tools..."):
                config = {"configurable": {"thread_id": st.session_state.thread_id}}
                try:
                    response = graph.invoke(
                        {"messages": [HumanMessage(content=active_input)]},
                        config=config
                    )

                    ai_content = response["messages"][-1].content
                    chart_paths = extract_chart_paths_from_response(response)

                    st.markdown(ai_content)

                    for chart_file in chart_paths:
                        render_html_chart(chart_file)

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": ai_content,
                        "charts": chart_paths
                    })

                except Exception as e:
                    err_msg = f"❌ An error occurred: {str(e)}"
                    st.error(err_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": err_msg,
                        "charts": []
                    })


# =====================================================================
# TAB 2: MACHINE LEARNING STUDIO
# =====================================================================
with main_tab_ml:
    if data_store.df is None:
        st.warning("⚠️ No dataset loaded. Please upload or load a dataset in the sidebar to use the Machine Learning Studio.")
    else:
        # Inspect columns for dropdowns
        feature_info = analyze_features(data_store.df)
        all_cols = list(data_store.df.columns)
        numeric_cols = feature_info["numeric_features"]

        ml_sub_dashboard, ml_sub_anomaly, ml_sub_clf, ml_sub_reg, ml_sub_cluster, ml_sub_models = st.tabs([
            "📊 Overview",
            "🚨 Anomaly Detection",
            "🎯 Classification",
            "📈 Regression",
            "🔍 Clustering",
            "💾 Model Registry"
        ])

        # --- SUB-TAB 1: ML DASHBOARD ---
        with ml_sub_dashboard:
            st.markdown("### 📊 Machine Learning Studio Dashboard")
            st.markdown("Stage 1 integrates established, battle-tested Scikit-Learn algorithms with zero data leakage.")

            c1, c2, c3, c4 = st.columns(4)
            models_list = list_trained_models()
            c1.metric("Available Algorithms", "4")
            c2.metric("Saved Models", f"{len(models_list)}")
            latest_model = models_list[0]["algorithm"] if models_list else "None"
            c3.metric("Latest Algorithm", latest_model)
            c4.metric("Usable Numeric Features", f"{len(numeric_cols)}")

            st.divider()
            st.markdown("#### 🛠️ Available ML Engines")
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("""
                - **🚨 Anomaly Detection**: `IsolationForest`
                  - Identifies potential multivariate statistical outliers without labels.
                  - Scores outlier severity on a standardized scale.
                - **🎯 Supervised Classification**: `RandomForestClassifier`
                  - Predicts discrete categorical labels (binary or multiclass).
                  - Stratified train-test splits with leakage-free preprocessing.
                """)
            with col_b:
                st.markdown("""
                - **📈 Supervised Regression**: `RandomForestRegressor`
                  - Predicts continuous quantitative targets.
                  - Evaluates MAE, RMSE, and $R^2$ with actual vs. predicted diagnostics.
                - **🔍 Unsupervised Clustering**: `KMeans`
                  - Groups similar records by feature similarity.
                  - Includes inertia, silhouette scores, and optional Elbow analysis.
                """)

        # --- SUB-TAB 2: ANOMALY DETECTION ---
        with ml_sub_anomaly:
            st.markdown("### 🚨 Anomaly Detection (Isolation Forest)")
            st.caption("Detects rare observations and multivariate statistical outliers. Flags records for investigation.")

            col_cfg1, col_cfg2 = st.columns([2, 1])
            with col_cfg1:
                contamination = st.slider(
                    "Expected Contamination (Outlier Proportion)",
                    min_value=0.01,
                    max_value=0.30,
                    value=0.05,
                    step=0.01,
                    help="Approximate expected percentage of anomalies in the dataset."
                )
            with col_cfg2:
                st.write("")
                st.write("")
                run_anomaly_btn = st.button("🚀 Run Anomaly Detection", use_container_width=True)

            if run_anomaly_btn:
                with st.spinner("Training Isolation Forest and scoring anomalies..."):
                    try:
                        res = run_anomaly_detection(data_store.df, contamination=contamination)
                        st.session_state.ml_anomaly_res = res
                        st.success(f"Anomaly Detection completed! Flagged {res['anomaly_count']} potential anomalies ({res['anomaly_percentage']}%).")
                    except Exception as e:
                        st.error(f"Anomaly detection failed: {e}")

            if st.session_state.ml_anomaly_res:
                res = st.session_state.ml_anomaly_res
                st.divider()
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Analyzed Records", f"{res['total_records']:,}")
                m2.metric("Flagged Anomalies", f"{res['anomaly_count']:,}")
                m3.metric("Anomaly Rate", f"{res['anomaly_percentage']}%")
                m4.metric("Model ID", res["model_id"])

                render_html_chart(res["chart_path"])

                st.markdown("#### 📋 Top Flagged Records")
                st.caption(res["note"])
                flagged_df = data_store.df.loc[res["flagged_indices"]].copy()
                st.dataframe(flagged_df.head(50), use_container_width=True)

                csv_data = flagged_df.to_csv(index=True).encode("utf-8")
                st.download_button(
                    label="💾 Download All Flagged Anomalies as CSV",
                    data=csv_data,
                    file_name=f"anomalies_{st.session_state.dataset_name}.csv",
                    mime="text/csv"
                )

        # --- SUB-TAB 3: CLASSIFICATION ---
        with ml_sub_clf:
            st.markdown("### 🎯 Classification (Random Forest Classifier)")
            st.caption("Predicts discrete classes or categories with stratified train-test validation.")

            c_c1, c_c2, c_c3 = st.columns([2, 1, 1])
            with c_c1:
                target_col_clf = st.selectbox("Select Target Column", options=all_cols, index=1 if "Survived" in all_cols else 0)
            with c_c2:
                test_size_clf = st.slider("Test Size Split", 0.1, 0.4, 0.2, 0.05, key="clf_split")
            with c_c3:
                st.write("")
                st.write("")
                train_clf_btn = st.button("🚀 Train Classifier", use_container_width=True)

            if train_clf_btn:
                with st.spinner("Training Random Forest Classifier (zero data leakage)..."):
                    try:
                        res = run_classification(
                            df=data_store.df,
                            target_column=target_col_clf,
                            test_size=test_size_clf
                        )
                        st.session_state.ml_classification_res = res
                        st.success(f"Model trained successfully! Accuracy: {res['metrics']['accuracy'] * 100:.2f}%")
                    except Exception as e:
                        st.error(f"Classification failed: {e}")

            if st.session_state.ml_classification_res:
                res = st.session_state.ml_classification_res
                st.divider()
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Accuracy", f"{res['metrics']['accuracy'] * 100:.2f}%")
                m2.metric("Precision", f"{res['metrics']['precision'] * 100:.2f}%")
                m3.metric("Recall", f"{res['metrics']['recall'] * 100:.2f}%")
                m4.metric("F1-Score", f"{res['metrics']['f1_score'] * 100:.2f}%")

                if res.get("warnings"):
                    for w in res["warnings"]:
                        st.warning(f"⚠️ {w}")

                st.markdown("#### 📊 Model Diagnostics")
                col_cm, col_fi = st.columns(2)
                with col_cm:
                    render_html_chart(res["charts"]["confusion_matrix_path"])
                with col_fi:
                    render_html_chart(res["charts"]["feature_importance_path"])

                st.caption(f"Model ID: `{res['model_id']}` | Train Samples: {res['train_samples']} | Test Samples: {res['test_samples']}")

        # --- SUB-TAB 4: REGRESSION ---
        with ml_sub_reg:
            st.markdown("### 📈 Regression (Random Forest Regressor)")
            st.caption("Predicts continuous numerical values with evaluation metrics and residual diagnostics.")

            r_c1, r_c2, r_c3 = st.columns([2, 1, 1])
            with r_c1:
                target_col_reg = st.selectbox(
                    "Select Continuous Target Column",
                    options=numeric_cols if numeric_cols else all_cols,
                    index=numeric_cols.index("Fare") if "Fare" in numeric_cols else 0
                )
            with r_c2:
                test_size_reg = st.slider("Test Size Split", 0.1, 0.4, 0.2, 0.05, key="reg_split")
            with r_c3:
                st.write("")
                st.write("")
                train_reg_btn = st.button("🚀 Train Regressor", use_container_width=True)

            if train_reg_btn:
                with st.spinner("Training Random Forest Regressor..."):
                    try:
                        res = run_regression(
                            df=data_store.df,
                            target_column=target_col_reg,
                            test_size=test_size_reg
                        )
                        st.session_state.ml_regression_res = res
                        st.success(f"Regression model trained! R²: {res['metrics']['r2_score']:.4f}")
                    except Exception as e:
                        st.error(f"Regression failed: {e}")

            if st.session_state.ml_regression_res:
                res = st.session_state.ml_regression_res
                st.divider()
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("R² Score", f"{res['metrics']['r2_score']:.4f}")
                m2.metric("MAE", f"{res['metrics']['mae']:.4f}")
                m3.metric("RMSE", f"{res['metrics']['rmse']:.4f}")
                m4.metric("MSE", f"{res['metrics']['mse']:.4f}")

                st.markdown("#### 📊 Regression Diagnostics")
                col_avp, col_res = st.columns(2)
                with col_avp:
                    render_html_chart(res["charts"]["actual_vs_predicted_path"])
                with col_res:
                    render_html_chart(res["charts"]["residuals_path"])

                st.caption(f"Model ID: `{res['model_id']}` | Train Samples: {res['train_samples']} | Test Samples: {res['test_samples']}")

        # --- SUB-TAB 5: CLUSTERING ---
        with ml_sub_cluster:
            st.markdown("### 🔍 Unsupervised Clustering (K-Means)")
            st.caption("Discovers natural groupings and segments in your data using normalized numerical features.")

            cl_c1, cl_c2 = st.columns([2, 1])
            with cl_c1:
                cluster_features = st.multiselect(
                    "Select Numerical Features for Clustering",
                    options=numeric_cols,
                    default=numeric_cols[:min(4, len(numeric_cols))]
                )
            with cl_c2:
                n_clusters_val = st.slider("Number of Clusters (K)", min_value=2, max_value=8, value=3)
                run_elbow_check = st.checkbox("Include Elbow Method Analysis", value=False)

            run_cluster_btn = st.button("🚀 Run K-Means Clustering", use_container_width=True)

            if run_cluster_btn:
                if len(cluster_features) < 2:
                    st.error("Please select at least 2 numerical features for clustering.")
                else:
                    with st.spinner("Standardizing features and fitting K-Means..."):
                        try:
                            res = run_clustering(
                                df=data_store.df,
                                n_clusters=n_clusters_val,
                                feature_columns=cluster_features,
                                run_elbow=run_elbow_check
                            )
                            st.session_state.ml_clustering_res = res
                            st.success(f"Formed {res['n_clusters']} clusters successfully!")
                        except Exception as e:
                            st.error(f"Clustering failed: {e}")

            if st.session_state.ml_clustering_res:
                res = st.session_state.ml_clustering_res
                st.divider()
                m1, m2, m3 = st.columns(3)
                m1.metric("Clusters (K)", f"{res['n_clusters']}")
                m1_inertia = res["metrics"].get("inertia", "N/A")
                m2.metric("Inertia", f"{m1_inertia:,}" if isinstance(m1_inertia, (int, float)) else str(m1_inertia))
                sil = res["metrics"].get("silhouette_score")
                m3.metric("Silhouette Score", f"{sil:.4f}" if sil is not None else "N/A")

                render_html_chart(res["charts"]["cluster_scatter_path"])

                if res["charts"].get("elbow_path"):
                    st.markdown("#### 📉 Elbow Analysis")
                    render_html_chart(res["charts"]["elbow_path"])

                st.markdown("#### 👥 Cluster Size Distribution")
                dist_df = pd.DataFrame.from_dict(res["cluster_distribution"], orient="index")
                st.dataframe(dist_df, use_container_width=True)

                st.markdown("#### 🏷️ Cluster Feature Profiles (Mean Values)")
                prof_df = pd.DataFrame.from_dict(res["cluster_profiles"], orient="index")
                st.dataframe(prof_df, use_container_width=True)

        # --- SUB-TAB 6: SAVED MODEL REGISTRY ---
        with ml_sub_models:
            st.markdown("### 💾 Saved Model Registry")
            st.caption("Persistent Scikit-Learn pipelines saved in the local `models/` directory.")

            saved_models = list_trained_models()
            if not saved_models:
                st.info("No models saved yet. Train an anomaly detection, classification, regression, or clustering model to see it here.")
            else:
                table_rows = []
                for m in saved_models:
                    table_rows.append({
                        "Model ID": m.get("model_id"),
                        "Algorithm": m.get("algorithm"),
                        "Task": m.get("model_type"),
                        "Target": m.get("target_name") or "-",
                        "Created At": m.get("created_at", "")[:19].replace("T", " ")
                    })
                st.dataframe(pd.DataFrame(table_rows), use_container_width=True)

                st.markdown("#### 🔎 Model Inspector")
                inspect_id = st.selectbox("Select Model to Inspect Metadata:", options=[m["model_id"] for m in saved_models])
                if inspect_id:
                    meta = get_model_metadata(inspect_id)
                    st.json(meta)
