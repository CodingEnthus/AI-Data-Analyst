# 🔗 Component Connection Matrix & Stack Interoperability

This document provides a component-by-component communication matrix and explains how the individual technologies in the stack interact.

---

## 1. Complete Component Connection Matrix

| Source Component | Target Component | Protocol / Interface | Payload Transferred | Purpose |
| :--- | :--- | :--- | :--- | :--- |
| **Streamlit (`app.py`)** | **`data_store.df`** | Direct Python variable assignment | `pd.DataFrame` instance | Ingest uploaded CSV/XLSX into in-memory global storage. |
| **Streamlit (`app.py`)** | **LangGraph (`agent.py`)** | `graph.invoke(input_dict, config)` | `{"messages": [HumanMessage]}` + `thread_id` | Dispatch user prompt to agent state machine. |
| **LangGraph (`agent.py`)** | **MemorySaver** | Internal checkpointer API | State snapshot (`State`) | Retrieve historical messages before turn; persist state after turn. |
| **`chatbot` Node** | **Groq Cloud API (`ChatGroq`)** | HTTPS / REST (`httpx`) | Serialized prompt + tool schemas (JSON) | Request token completion or tool-call generation. |
| **Groq Cloud API** | **`chatbot` Node** | HTTPS Response (JSON) | `AIMessage` with text or `tool_calls` | Return LLM decision to agent. |
| **LangGraph** | **`tools_condition`** | Internal router callable | Latest `AIMessage` in state | Decide whether to route to `"tools"` or `END`. |
| **`tools_condition`** | **`ToolNode`** | LangGraph edge transition | Graph state dictionary | Dispatch execution to registered tools. |
| **`ToolNode`** | **Registered Tools (`tools/*`)** | Dynamic function call (`func(**args)`) | Extracted tool keyword arguments | Execute local Python logic. |
| **`tools/statistics_tools.py`** | **`tools/helper_tools.py`** | `find_best_matching_column.invoke()` | `{"column_name": "..."}` | Fuzzy match requested column name to DataFrame schema. |
| **`tools/statistics_tools.py`** | **`data_store.df`** | In-memory pointer read | `data_store.df[col]` Series | Calculate mathematical aggregation (`mean`, `std`, etc.). |
| **`tools/duckdb_tools.py`** | **DuckDB Engine** | `duckdb.connect(":memory:")` | SQL Query string + DataFrame pointer | Run read-only vectorized SQL query on DataFrame. |
| **`tools/visualization_tools.py`** | **Plotly Express** | `px.bar()`, `px.scatter()`, etc. | DataFrame columns, axis labels | Generate Plotly Figure object. |
| **`tools/visualization_tools.py`** | **`outputs/charts/`** | `fig.write_html(path)` | Self-contained HTML string | Save chart to disk with UUID filename. |
| **`tools/ml_tools.py`** | **`ml_engine/*`** | Direct Python module calls | DataFrame, target column, hyperparameters | Run Scikit-learn preprocessing, training, and evaluation. |
| **`ml_engine/model_manager.py`** | **`models/`** | `joblib.dump()` & `json.dump()` | Serialized pipeline & metadata dict | Persist model artifacts to disk. |
| **Registered Tools** | **`ToolNode`** | Return statement | Python `dict` or JSON string | Provide calculation/chart result. |
| **`ToolNode`** | **`chatbot` Node** | LangGraph edge transition | `ToolMessage` with `tool_call_id` | Feed deterministic facts back to LLM for synthesis. |
| **`chatbot` Node** | **Streamlit (`app.py`)** | Return from `graph.invoke()` | Final state containing `AIMessage` & chart paths | Deliver answer to frontend. |
| **Streamlit (`app.py`)** | **`outputs/charts/`** | Standard file I/O `open().read()` | HTML string | Embed interactive chart into web page via `components.html()`. |

---

## 2. Technology Stack Interoperability: How They Work Together

```
[STREAMLIT (Presentation & Event Loop)]
       │
       ▼ graph.invoke()
[LANGGRAPH (State Machine & Cyclic Dispatch)]
       │
       ├── Reads Prompt & Binds Tools via LANGCHAIN-CORE
       │
       ▼ HTTPS REST Call
[GROQ CLOUD (High-Speed LLM Inference)]
       │
       ▲ Returns Tool Call Decision
       │
[TOOLNODE (Sandbox Execution)]
       │
       ├─────────────────┬─────────────────┬─────────────────┐
       ▼                 ▼                 ▼                 ▼
  [PANDAS/NUMPY]     [DUCKDB]           [PLOTLY]      [SCIKIT-LEARN]
 (Vectorized Math) (Zero-Copy SQL)    (Interactive)    (ML Pipelines)
       │                 │                 │                 │
       └─────────────────┴────────┬────────┴─────────────────┘
                                  │ Writes Artifacts
                                  ▼
                         [DISK: outputs/ & models/]
```

### 1. LangChain
- **Role**: Provides the foundational building blocks: `@tool` decorators, message classes (`HumanMessage`, `AIMessage`, `ToolMessage`, `SystemMessage`), and model abstraction classes (`ChatGroq`, `ChatOllama`).

### 2. LangGraph
- **Role**: Provides the execution graph and state coordination. It prevents unpredictable LLM loops by enforcing a structured state machine with clear entry (`START`), conditional branch (`tools_condition`), and termination (`END`) points.

### 3. Groq LLM Cloud
- **Role**: Acts purely as the **cognitive reasoning engine**. Groq utilizes LPU (Language Processing Unit) hardware to return tool-calling decisions in sub-second latency. It reads user intent, maps it to a tool signature, and synthesizes tool outputs into plain English.

### 4. Pandas & NumPy
- **Role**: In-memory data manipulation. Pandas maintains the DataFrame in RAM; NumPy powers array slicing, zero-variance detection, and feature sorting.

### 5. DuckDB
- **Role**: Embeds a zero-overhead analytical database directly in the Python process. By querying `data_store.df` directly, DuckDB handles complex SQL joins, aggregations, and window functions faster than interpreted Pandas chaining.

### 6. Scikit-Learn
- **Role**: Executes all machine learning pipelines locally on the CPU. It handles missing value imputation (`SimpleImputer`), categorical encoding (`OneHotEncoder`), standardization (`StandardScaler`), model fitting (`RandomForest`, `IsolationForest`, `KMeans`), and metric evaluation.

### 7. Plotly
- **Role**: Visualization compiler. Converts DataFrame subsets into interactive, zoomable D3 JavaScript charts packaged into standalone HTML documents.

### 8. Streamlit
- **Role**: Front-end presentation and session manager. It runs the web server, renders the chat interface, and mounts HTML charts via `components.html()`.

### 9. MemorySaver
- **Role**: In-memory session checkpointer. Indexes state by `thread_id` so subsequent questions remember earlier context without database overhead.

---

## 3. Local vs. Remote Computation Boundaries

| Task | Execution Location | Resource Used | Network Bound? |
| :--- | :--- | :--- | :---: |
| **Natural Language Understanding & Intent Parsing** | Remote (Groq Cloud API) | Groq LPU Cluster | **YES (HTTPS)** |
| **Response Synthesis** | Remote (Groq Cloud API) | Groq LPU Cluster | **YES (HTTPS)** |
| **Data Ingestion & CSV/Excel Parsing** | Local Host | Local CPU & RAM | ❌ NO |
| **Descriptive Statistics (Pandas)** | Local Host | Local CPU & RAM | ❌ NO |
| **SQL Querying (DuckDB)** | Local Host | Local CPU & RAM | ❌ NO |
| **Machine Learning Training & Inference** | Local Host | Local CPU (Scikit-Learn) | ❌ NO |
| **Interactive Chart Rendering (Plotly)** | Local Host & Browser | Local CPU & Client Browser | ❌ NO |
| **State & Checkpointing (MemorySaver)** | Local Host | Local RAM | ❌ NO |

> **Key Architectural Takeaway**: **90% of all analytical and data-processing tasks run 100% locally on-premise.** Sensitive customer data records never leave the local machine; only prompt instructions and column schemas travel to the LLM endpoint for intent classification.
