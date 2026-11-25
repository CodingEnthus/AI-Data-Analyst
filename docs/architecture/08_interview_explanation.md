# 🎙️ Technical Interview Presentation Scripts

Use these presentation scripts to explain your **AI Data Analyst** project during technical interviews. Each script specifies **where to point on the Master HLD diagram**, **what exact phrasing to use**, and **how to preempt interviewer follow-ups**.

---

## 1. The 1-Minute Pitch: Executive Summary

### Where to Point on the HLD Diagram:
1. Point to **Layer 1 (Streamlit)** $\rightarrow$
2. Point to **Layer 2 (LangGraph)** $\rightarrow$
3. Point to **Layer 4 & 5 (Tools & Engines)** $\rightarrow$
4. Point to **Layer 6 (Outputs & Charts)**.

### What to Say:
> *"I built an **Agentic AI Data Analyst** that enables non-technical business users to analyze tabular datasets, execute SQL queries, generate interactive Plotly charts, and train predictive machine learning models using natural language.
>
> Architecturally, it is built on a 6-layer decoupled design:
> 1. At the top is a **Streamlit** presentation layer handling file uploads and conversation state.
> 2. The core brain is a stateful **LangGraph** agent bound to **Groq's LPU-accelerated LLM**.
> 3. Instead of letting the LLM hallucinate numbers or execute arbitrary code strings, our agent operates on the **ReAct pattern**, dynamically delegating tasks across 18 specialized Python tools.
> 4. Mathematical aggregations run in **Pandas**, complex analytical queries execute in an in-memory **DuckDB** SQL engine, and predictive tasks execute in a local **Scikit-Learn ML engine** with strict data leakage prevention.
> 5. Output charts are compiled as standalone **Plotly HTML** files embedded directly into the chat stream with session memory checkpointed via LangGraph's `MemorySaver`."*

### Likely Follow-up Question:
- *"Why did you use LangGraph instead of a simple LangChain chain?"*
- **Quick Defense**: *"LangGraph provides cyclic graph execution and checkpointer state machines. Chains are strictly linear DAGs that cannot easily loop back from tool execution to LLM reflection."*

---

## 2. The 3-Minute Architecture Walkthrough: Technical Interview

### Where to Point on the HLD Diagram:
1. Start at **User Input & Streamlit** (Layer 1).
2. Move to **LangGraph & MemorySaver** (Layer 2).
3. Move to **LLM & Tool Binding** (Layer 3).
4. Point to the **Conditional Edge (`tools_condition`)** and split into **ToolNode** (Layer 4).
5. Point to **Engines** (Layer 5: Pandas, DuckDB, Scikit-Learn).
6. Trace the return loop back to **Chatbot** and out to **Streamlit** (Layer 6).

### What to Say:
> *"Let's trace how the system processes a user request like: 'Create a bar chart of average Fare by Pclass and flag any anomalies.'
>
> 1. **Ingestion & State Dispatch**: The user enters their prompt into the Streamlit interface. `app.py` packages the prompt as a `HumanMessage` and invokes our compiled LangGraph instance, passing a unique session `thread_id`.
> 2. **Checkpointed Memory**: LangGraph's `MemorySaver` retrieves the conversation checkpoint for that `thread_id`, prepends our system prompt, and feeds the full conversation history to the `chatbot` node.
> 3. **Cognitive Reasoning**: The node calls `ChatGroq`. Because our 18 tools are bound to the model using schema contracts, the LLM determines that it cannot answer mathematically on its own. It emits two structured tool calls: `create_bar_chart` and `detect_anomalies`.
> 4. **Conditional Routing**: LangGraph's `tools_condition` intercepts the tool calls and routes execution to the `ToolNode`.
> 5. **Deterministic Local Execution**: 
>    - For the chart: `visualization_tools.py` validates the column names using fuzzy matching (`difflib`), aggregates the data in Pandas, builds a Plotly figure, and persists it to disk as a standalone HTML file in `outputs/charts/`.
>    - For anomalies: `ml_tools.py` invokes our `ml_engine/anomaly_detection.py`, which cleans the data, fits an `IsolationForest`, flags the outlier rows, and saves an anomaly distribution chart.
> 6. **The Cyclic ReAct Loop**: The tool results and chart paths are packaged into `ToolMessage` instances and routed **back** to the `chatbot` node via our cyclic graph edge. The LLM reads the deterministic outputs and synthesizes an intelligent, conversational summary.
> 7. **Multi-Modal Rendering**: Control returns to Streamlit, which displays the markdown text and extracts the HTML chart paths, embedding interactive Plotly widgets directly into the user's browser with download buttons."*

---

## 3. The 5-Minute Deep-Dive: Senior AI Engineer / Architect Level

### Where to Point on the HLD Diagram:
Walk through all 6 layers, highlighting the **design decisions, security boundaries, and data-leakage safeguards**:
- Highlight `data_store.df` (Layer 5) vs. State (Layer 2).
- Highlight the **SQL Security Gate** in `tools/duckdb_tools.py`.
- Highlight the **Data Leakage Pipeline** in `ml_engine/preprocessing.py`.
- Highlight **Joblib & Metadata serialization** in `ml_engine/model_manager.py`.

### What to Say:
> *"As a Senior AI Engineer, I designed this system with three core architectural principles in mind: **Deterministic Safety**, **Data Leakage Prevention**, and **Token/Resource Efficiency**.
>
> ### Principle 1: Deterministic Safety vs. Arbitrary Code Execution
> Many early generative AI data agents used `exec()` or `eval()` to execute arbitrary Python strings written by the LLM. That approach is insecure and prone to syntax crashes.
> In our architecture, the LLM is strictly constrained to **18 strongly-typed tool schemas**. 
> - Furthermore, in `duckdb_tools.py`, we implement a security regex barrier that blocks all mutating commands—`DROP`, `DELETE`, `INSERT`, `UPDATE`, `ALTER`, `ATTACH`, `PRAGMA`. Users can write rich analytical SQL with CTEs and window functions, but cannot mutate the underlying dataset or access the host filesystem.
>
> ### Principle 2: Strict Data Leakage Prevention in the ML Engine
> When building the machine learning studio (`ml_engine/`), we ensured strict enterprise-grade preprocessing standards.
> - In `preprocessing.py`, before any imputation or encoding occurs, we perform an 80/20 `train_test_split`.
> - Our `ColumnTransformer`—which includes `SimpleImputer(strategy='median')`, `StandardScaler()`, and `OneHotEncoder(handle_unknown='ignore')`—is **fitted strictly on `X_train`** via `.fit_transform()`.
> - The test set `X_test` is only transformed using `.transform()`. This guarantees that test distribution metrics never leak into training.
> - Additionally, our preprocessor automatically inspects column cardinality, excluding unique identifier columns like passenger names or ticket numbers that would cause overfitting.
>
> ### Principle 3: In-Memory Optimization & Token Decoupling
> Tabular data can easily contain millions of data points. If you pass raw data tables through the LangGraph state or LLM context, you quickly hit token rate limits and memory bottlenecks.
> - We maintain the active DataFrame in an in-memory singleton `data_store.df`.
> - DuckDB mounts this DataFrame with zero memory copies (`con.register('df', data_store.df)`).
> - Plotly figures write their JavaScript/HTML payloads directly to disk (`outputs/charts/`) and return only lightweight filepath strings through the LangGraph state.
> - This keeps our LangGraph `MemorySaver` snapshots under a few kilobytes, ensuring sub-second graph execution even across extended multi-turn conversations."*
