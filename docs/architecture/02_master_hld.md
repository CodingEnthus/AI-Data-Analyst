# 🏗️ Master High-Level Architecture (HLD)

This document specifies the master High-Level Architecture (HLD) of the **AI Data Analyst** system, organized across 6 distinct architectural layers.

---

## 1. Complete Master Architecture Diagram (Mermaid)

```mermaid
flowchart TD
    %% =========================================================================
    %% LAYER 1: PRESENTATION LAYER
    %% =========================================================================
    subgraph Layer1 ["🖥️ Layer 1: Presentation Layer (Streamlit Frontend - app.py)"]
        User(["👤 User / Analyst"])
        FileUploader["📂 Sidebar: File Uploader (CSV / XLSX)"]
        ChatUI["💬 Conversational Chat Interface"]
        DataPreview["🔍 Dataset Preview & Column Type Inspector"]
        MetricsSummary["📊 Dataset Summary Cards (Rows, Cols, Nulls)"]
        MLStudioGUI["🤖 Dedicated ML Studio GUI (Tabs: Anomaly, Clf, Reg, Cluster)"]
        ChartViewer["📈 Plotly Interactive Chart & HTML Download Viewer"]
    end

    %% =========================================================================
    %% LAYER 2: AGENT ORCHESTRATION LAYER
    %% =========================================================================
    subgraph Layer2 ["🧠 Layer 2: Agent Orchestration Layer (LangGraph Engine - agent.py)"]
        StateGraphInstance["StateGraph(State)"]
        ChatbotNode["Node: 'chatbot' (Agent Prompt & Inference)"]
        ToolsCondition{"Conditional Router: tools_condition"}
        ToolNodeInstance["Node: 'tools' (ToolNode executing 18 tools)"]
        MemorySaverCheckpointer[("💾 MemorySaver (Checkpointer by thread_id)")]
    end

    %% =========================================================================
    %% LAYER 3: LLM & COGNITIVE LAYER
    %% =========================================================================
    subgraph Layer3 ["🤖 Layer 3: Cognitive & LLM Layer"]
        SystemPrompt["prompts.py: SYSTEM_PROMPT (Anti-hallucination & Tool Directives)"]
        ToolBinder["llm.bind_tools(18 tools)"]
        GroqProvider["ChatGroq Cloud (qwen/qwen3.8-27b)"]
        OllamaProvider["ChatOllama Local (qwen3:4b)"]
    end

    %% =========================================================================
    %% LAYER 4: TOOL EXECUTION LAYER
    %% =========================================================================
    subgraph Layer4 ["⚙️ Layer 4: Tool Execution Layer (tools/)"]
        subgraph Group_DataOps ["Dataset & Statistics"]
            Tool_Load["load_dataset"]
            Tool_Info["get_dataset_info"]
            Tool_Stats["calculate_statistics"]
        end

        subgraph Group_SQL ["Analytical SQL"]
            Tool_SQL["query_dataset_with_sql"]
        end

        subgraph Group_Viz ["Plotly Visualizations (8 Tools)"]
            Tool_Bar["create_bar_chart"]
            Tool_Line["create_line_chart"]
            Tool_Scatter["create_scatter_plot"]
            Tool_Hist["create_histogram"]
            Tool_Pie["create_pie_chart"]
            Tool_Heat["create_heatmap"]
            Tool_Box["create_box_plot"]
            Tool_Violin["create_violin_plot"]
        end

        subgraph Group_ML ["Machine Learning Tools"]
            Tool_Anomaly["detect_anomalies"]
            Tool_Clf["train_classification_model"]
            Tool_Reg["train_regression_model"]
            Tool_Cluster["perform_clustering"]
            Tool_Eval["evaluate_trained_model"]
            Tool_Predict["predict_with_model"]
        end

        subgraph Group_Helper ["Helper Utilities"]
            Helper_Fuzzy["find_best_matching_column"]
            Helper_Val["validate_chart_column"]
            Helper_Save["save_chart"]
        end
    end

    %% =========================================================================
    %% LAYER 5: DATA PROCESSING & COMPUTATION LAYER
    %% =========================================================================
    subgraph Layer5 ["⚡ Layer 5: Data Processing & ML Engine"]
        GlobalDataStore[("data_store.df (Global In-Memory DataFrame)")]
        PandasEngine["🐼 Pandas Engine (Vectorized Operations)"]
        DuckDBEngine["🦆 DuckDB Engine (:memory: Columnar SQL Engine)"]
        PlotlyEngine["📊 Plotly Express Core"]
        
        subgraph MLEngineCore ["🔬 ml_engine/ (Scikit-Learn Algorithms)"]
            ML_Prep["preprocessing.py (Imputer, Scaler, OneHotEncoder)"]
            ML_Clf["classification.py (RandomForestClassifier)"]
            ML_Reg["regression.py (RandomForestRegressor)"]
            ML_Iso["anomaly_detection.py (IsolationForest)"]
            ML_KMeans["clustering.py (KMeans + PCA)"]
            ML_Eval["evaluation.py (Metrics & Confusion Matrix)"]
            ML_Mgr["model_manager.py (Joblib Persistence)"]
        end
    end

    %% =========================================================================
    %% LAYER 6: OUTPUT & ARTIFACT LAYER
    %% =========================================================================
    subgraph Layer6 ["📦 Layer 6: Output & Storage Layer"]
        DiskHTML["outputs/charts/*.html (Standalone Interactive Visuals)"]
        DiskModels["models/*.joblib & *.json (Serialized ML Pipelines)"]
        DiskData["data/*.csv & *.xlsx (Raw Datasets)"]
        TextResponse["💬 Conversational Markdown Synthesis"]
    end

    %% =========================================================================
    %% INTER-LAYER DATA FLOW CONNECTIONS
    %% =========================================================================

    %% Ingestion
    User ==>|Uploads File / Selects Sample| FileUploader
    FileUploader -->|pd.read_csv / pd.read_excel| GlobalDataStore
    FileUploader -->|Persists Raw File| DiskData
    GlobalDataStore -.->|Renders Preview & Types| DataPreview
    GlobalDataStore -.->|Calculates Metrics| MetricsSummary

    %% Conversational Dispatch
    User ==>|Asks Question| ChatUI
    ChatUI ==>|graph.invoke(HumanMessage, thread_id)| StateGraphInstance

    %% LangGraph Flow
    StateGraphInstance <-->|Checkpoints & Reads State| MemorySaverCheckpointer
    StateGraphInstance -->|Prepend System Prompt + Messages| ChatbotNode
    ChatbotNode -->|Sends Prompt + Schemas| ToolBinder
    ToolBinder --> GroqProvider
    ToolBinder -.-> OllamaProvider
    GroqProvider -->|Emits AIMessage or Tool Calls| ChatbotNode

    %% Conditional Tool Routing
    ChatbotNode --> ToolsCondition
    ToolsCondition -->|No Tools: Direct Answer| TextResponse
    ToolsCondition -->|Has Tool Calls| ToolNodeInstance

    %% Tool Invocations
    ToolNodeInstance --> Group_DataOps
    ToolNodeInstance --> Group_SQL
    ToolNodeInstance --> Group_Viz
    ToolNodeInstance --> Group_ML

    %% Helpers
    Group_Stats -.-> Helper_Fuzzy
    Group_Viz -.-> Helper_Val
    Group_Viz -.-> Helper_Save
    Group_ML -.-> Helper_Fuzzy

    %% Engine Calls
    Group_DataOps <--> PandasEngine
    Group_SQL <--> DuckDBEngine
    DuckDBEngine <--> GlobalDataStore
    PandasEngine <--> GlobalDataStore

    Group_Viz --> PlotlyEngine
    PlotlyEngine -->|Saves HTML| DiskHTML

    Group_ML --> MLEngineCore
    MLEngineCore <--> GlobalDataStore
    MLEngineCore -->|Saves .joblib & .json| DiskModels
    MLEngineCore -->|Saves Metric Plots| DiskHTML

    %% Return Loops
    Group_DataOps -->|Tool Output| ToolNodeInstance
    Group_SQL -->|Markdown Table| ToolNodeInstance
    Group_Viz -->|chart_path JSON| ToolNodeInstance
    Group_ML -->|Metrics & Artifact JSON| ToolNodeInstance

    ToolNodeInstance ==>|ToolMessage| ChatbotNode
    TextResponse ==> ChatUI
    DiskHTML -.->|components.html()| ChartViewer
    ChartViewer ==> User

    %% ML Studio Direct Access
    User ==>|GUI Configuration| MLStudioGUI
    MLStudioGUI <--> MLEngineCore
```

---

## 2. Layer-by-Layer Architectural Specifications

### Layer 1: Presentation Layer
- **Component**: Built entirely using **Streamlit** ([app.py](file:///c:/Users/User/Desktop/AI-Data-Analyst/app.py)).
- **Sub-components**:
  - `st.sidebar`: Handles API key injection, LLM model selection, data file upload (`st.file_uploader`), dataset preview (`st.dataframe`), and column dtype inspection.
  - `st.tabs`: Organizes the UI into two primary workspaces:
    1. **Conversational Analyst**: Chat history container, prompt suggestions, and real-time inference spinner.
    2. **Machine Learning Studio**: No-code GUI panels for manual parameter tuning across Anomaly Detection, Classification, Regression, and Clustering.
  - `ChartViewer`: Implemented via `streamlit.components.v1.html()`, allowing full pan/zoom interactivity of Plotly charts and immediate HTML export.

### Layer 2: Agent Orchestration Layer
- **Component**: **LangGraph** State Machine ([agent.py](file:///c:/Users/User/Desktop/AI-Data-Analyst/agent.py)).
- **Sub-components**:
  - `StateGraph(State)`: Parameterized by `State(TypedDict)` containing `messages: Annotated[list, add_messages]`.
  - `chatbot` node: Handles prompt assembly, LLM execution, and exception handling for API limits (HTTP 429, 401).
  - `tools` node: An instance of prebuilt `ToolNode(tools)` executing any of the 18 registered tools.
  - `tools_condition`: Dynamic routing edge determining whether to terminate (`END`) or dispatch to `tools`.
  - `MemorySaver`: In-memory checkpointing mechanism keyed by session `thread_id`.

### Layer 3: Cognitive & LLM Layer
- **Components**:
  - **Prompt Management**: [prompts.py](file:///c:/Users/User/Desktop/AI-Data-Analyst/prompts.py) supplies the `SYSTEM_PROMPT` containing anti-hallucination rules and tool guidance.
  - **Tool Binding**: `llm.bind_tools(tools)` serializes function docstrings, argument types, and schemas into standard OpenAI-compatible tool definitions.
  - **Inference Engines**: 
    - Default: **Groq Cloud API** (`ChatGroq`) executing `qwen/qwen3.8-27b` with ultra-low latency.
    - Fallback/Local: **Ollama** (`ChatOllama`) executing `qwen3:4b` locally.

### Layer 4: Tool Execution Layer
Contains 18 registered tools across 4 analytical categories, backed by utility helpers:
1. **Dataset Ingestion & Inspection**: `load_dataset`, `get_dataset_info`.
2. **Descriptive Statistics**: `calculate_statistics` (supports 10 operations).
3. **Columnar SQL**: `query_dataset_with_sql` (DuckDB read-only query engine).
4. **Visual Analytics**: 8 dedicated Plotly tools (`create_bar_chart`, `create_line_chart`, etc.).
5. **Machine Learning**: 6 tools bridging to the Scikit-learn ML engine.
6. **Helper Utilities**: `find_best_matching_column` (fuzzy matching), `validate_chart_column`, `save_chart`.

### Layer 5: Data Processing & ML Engine
- **In-Memory Store**: `data_store.df` maintains the active Pandas DataFrame.
- **Pandas Core**: Executes vectorized mathematical calculations and data frame indexing.
- **DuckDB Core**: Attaches the in-memory DataFrame via zero-copy C pointers (`con.register('df', data_store.df)`).
- **ML Engine (`ml_engine/`)**: Modular Scikit-Learn pipelines featuring automated feature classification, leak-free imputation and encoding, model training, evaluation, and serialization.

### Layer 6: Output & Storage Layer
- **Ephemeral Storage**:
  - `outputs/charts/*.html`: Interactive HTML chart files with embedded D3/Plotly bundles.
  - `models/*.joblib`: Compressed binary model pipelines saved via `joblib.dump()`.
  - `models/*.json`: Model metadata (metrics, features, hyperparameters, timestamps).
  - `data/*`: Uploaded and sample datasets.
