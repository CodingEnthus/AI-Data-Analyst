# 🛠️ Individual Tool Workflows

This document details the exact execution paths, data transformations, error handling, and guardrails for the primary data operations: **Dataset Ingestion**, **Descriptive Statistics**, **Columnar SQL Querying**, and **Interactive Visualizations**.

---

## Part A: Dataset Ingestion Workflow

### 1. Workflow Architecture (Mermaid)

```mermaid
flowchart TD
    User(["👤 User"])
    
    subgraph UI ["app.py Sidebar"]
        OptionChoice{"Choose Data Source"}
        SampleBtn["Button: 'Load Titanic Dataset'"]
        UploadBox["st.file_uploader(.csv, .xlsx, .xls)"]
    end
    
    subgraph IngestionLogic ["load_file_to_datastore(file_path, display_name)"]
        CheckExt{"Check File Extension"}
        ReadCSV["pd.read_csv(file_path)"]
        ReadExcel["pd.read_excel(file_path)"]
        ThrowError["Raise ValueError('Unsupported format')"]
    end
    
    subgraph StorageState ["State Updates"]
        AssignDS["data_store.df = df"]
        SetSessionName["st.session_state.dataset_name = display_name"]
        ClearMLState["Clear st.session_state ML cache (anomaly, clf, reg, cluster)"]
        NotifyAgent["graph.invoke(System Notification: Loaded dataset...)"]
    end
    
    subgraph UI_Render ["UI Re-render"]
        SummaryCards["Display Rows, Columns, Missing Values"]
        PreviewExpander["Render st.dataframe(df.head(10)) & Column Types"]
    end

    User --> OptionChoice
    OptionChoice -->|Sample Dataset| SampleBtn
    OptionChoice -->|Upload Custom| UploadBox

    SampleBtn -->|Path: data/Titanic-Dataset.csv| CheckExt
    UploadBox -->|Saves to data/filename| CheckExt

    CheckExt -->|.csv| ReadCSV
    CheckExt -->|.xlsx / .xls| ReadExcel
    CheckExt -->|Other| ThrowError

    ReadCSV --> AssignDS
    ReadExcel --> AssignDS

    AssignDS --> SetSessionName
    SetSessionName --> ClearMLState
    ClearMLState --> NotifyAgent
    NotifyAgent --> SummaryCards
    SummaryCards --> PreviewExpander
```

### 2. Implementation Mechanics
- **Files Involved**: [app.py#L118-L141](file:///c:/Users/User/Desktop/AI-Data-Analyst/app.py#L118-L141), [tools/dataset_tools.py#L8-L34](file:///c:/Users/User/Desktop/AI-Data-Analyst/tools/dataset_tools.py#L8-L34), [data_store.py](file:///c:/Users/User/Desktop/AI-Data-Analyst/data_store.py).
- **Format Handling**: Extensions are evaluated via `os.path.splitext(file_path)[1].lower()`. CSV uses C-based `pd.read_csv`, and Excel uses `pd.read_excel` (via `openpyxl`).
- **State Clearing**: When a new dataset loads, previous ML studio states (`st.session_state.ml_anomaly_res`, etc.) are reset to `None` to prevent cross-dataset contamination.
- **Agent Notification**: In [app.py#L136-L140](file:///c:/Users/User/Desktop/AI-Data-Analyst/app.py#L136-L140), a system notification message is dispatched into the active `thread_id` so the agent's memory immediately reflects the new row and column count without the user needing to ask.

---

## Part B: Statistical Analysis Workflow

### 1. Workflow Architecture (Mermaid)

```mermaid
flowchart TD
    UserPrompt["👤 User: 'What is the average age of passengers?'"]
    Chatbot["🤖 Chatbot Node (Groq LLM)"]
    
    subgraph ToolDispatch ["Tool Invocation"]
        ToolCall["Tool: calculate_statistics<br/>Args: {'column_name': 'age', 'operation': 'mean'}"]
    end
    
    subgraph StatsExecution ["tools/statistics_tools.py: calculate_statistics()"]
        CheckDF{"data_store.df is None?"}
        ValidateOp{"operation in operations dict?"}
        FuzzyMatch["helper_tools.find_best_matching_column('age')"]
        ConfidenceCheck{"Confidence >= 0.8?"}
        NumericCheck{"is_numeric_dtype(df[col])?"}
        RunOp["operations['mean'](df['Age'])"]
        FormatSuccess["Return {'success': True, 'column': 'Age', 'result': 29.6991}"]
    end

    subgraph ErrorHandling ["Guardrails & Exceptions"]
        ErrNoDF["Return 'Dataset not loaded'"]
        ErrBadOp["Return 'Unsupported Operation'"]
        ErrNoCol["Return 'No suitable column found'"]
        ErrNotNum["Return 'Operation not supported for non-numeric column'"]
    end

    UserPrompt --> Chatbot
    Chatbot --> ToolCall
    ToolCall --> CheckDF
    
    CheckDF -->|Yes| ErrNoDF
    CheckDF -->|No| ValidateOp
    
    ValidateOp -->|Invalid| ErrBadOp
    ValidateOp -->|Valid| FuzzyMatch
    
    FuzzyMatch --> ConfidenceCheck
    ConfidenceCheck -->|< 0.8| ErrNoCol
    ConfidenceCheck -->|>= 0.8 ('Age')| NumericCheck
    
    NumericCheck -->|False & Numeric Op| ErrNotNum
    NumericCheck -->|True| RunOp
    
    RunOp --> FormatSuccess
    FormatSuccess -->|ToolMessage| Chatbot
    Chatbot -->|Conversational Answer| User(["👤 Final Answer: 'The average age is 29.7 years.'"])
```

### 2. Implementation Mechanics
- **Function**: `calculate_statistics(column_name: str, operation: str)` in [tools/statistics_tools.py#L6-L57](file:///c:/Users/User/Desktop/AI-Data-Analyst/tools/statistics_tools.py#L6-L57).
- **Supported Operations**: `mean`, `median`, `sum`, `min`, `max`, `count`, `std`, `variance`, `mode`, `nunique`.
- **Fuzzy Resolution**: `helper_tools.find_best_matching_column` maps lowercase/misspelled terms (`"age"` $\rightarrow$ `"Age"`).
- **Type Safety**: Enforces `pd.api.types.is_numeric_dtype` before allowing operations like `std` or `mean`.

---

## Part C: Columnar SQL Query Workflow (DuckDB)

### 1. Workflow Architecture (Mermaid)

```mermaid
flowchart TD
    UserPrompt["👤 User: 'Show survival rate and count per class using SQL'"]
    Chatbot["🤖 Chatbot Node (Groq LLM)"]
    
    subgraph ToolDispatch ["Tool Invocation"]
        ToolCall["Tool: query_dataset_with_sql<br/>Args: {'query': 'SELECT Pclass, COUNT(*)... FROM df GROUP BY Pclass'}"]
    end
    
    subgraph SQLSecurity ["tools/duckdb_tools.py Security Gate"]
        CheckDF{"data_store.df is None?"}
        RegexFilter{"Regex Check for Disallowed Keywords:<br/>DROP, DELETE, INSERT, UPDATE, ALTER,<br/>CREATE, ATTACH, DETACH, COPY, PRAGMA"}
        SecurityErr["Return 'Security Error: Modifying statements not allowed'"]
    end

    subgraph DuckDBExec ["DuckDB In-Memory Execution"]
        ConnectDB["duckdb.connect(database=':memory:')"]
        RegisterDF["con.register('df', data_store.df)<br/>con.register('dataset', data_store.df)"]
        RunQuery["result_df = con.execute(cleaned_query).fetchdf()"]
        CheckEmpty{"result_df.empty?"}
    end

    subgraph Formatting ["Table Markdown Formatting"]
        FormatMD["_format_dataframe_to_markdown(result_df, max_rows=50)"]
        RowTruncation{"total_rows > 50?"}
        FullMD["Return '**Query Results (showing first 50...):**\n\n|...|'"]
    end

    UserPrompt --> Chatbot
    Chatbot --> ToolCall
    ToolCall --> CheckDF
    
    CheckDF -->|No Data| ErrNoDF["Return 'Error: No dataset loaded.'"]
    CheckDF -->|Loaded| RegexFilter
    
    RegexFilter -->|Matched Mutating Keyword| SecurityErr
    RegexFilter -->|Passed (Pure SELECT)| ConnectDB
    
    ConnectDB --> RegisterDF
    RegisterDF --> RunQuery
    RunQuery --> CheckEmpty
    
    CheckEmpty -->|Empty| RetZero["Return 'Query executed, returned 0 rows.'"]
    CheckEmpty -->|Has Rows| FormatMD
    
    FormatMD --> RowTruncation
    RowTruncation -->|Yes| FullMD
    RowTruncation -->|No| FullMD
    
    FullMD -->|ToolMessage| Chatbot
    Chatbot -->|Synthesizes Response + Table| User(["👤 Final Markdown Table"])
```

### 2. Implementation Mechanics
- **Function**: `query_dataset_with_sql(query: str)` in [tools/duckdb_tools.py#L31-L95](file:///c:/Users/User/Desktop/AI-Data-Analyst/tools/duckdb_tools.py#L31-L95).
- **Zero-Copy Interoperability**: `con.register("df", data_store.df)` provides zero-copy query execution over Pandas RAM.
- **Security Regex**: Rejects any destructive SQL keyword (`DROP`, `DELETE`, `INSERT`, `UPDATE`, `ALTER`, `ATTACH`, `PRAGMA`).
- **Markdown Serialization**: Implements custom `_format_dataframe_to_markdown()` with float formatting (`.4f`) and null value sanitization (`NULL`), capped at 50 preview rows.

---

## Part D: Interactive Visualization Workflow (Plotly)

### 1. Workflow Architecture (Mermaid)

```mermaid
flowchart TD
    UserPrompt["👤 User: 'Create a bar chart showing average Fare by Pclass'"]
    Chatbot["🤖 Chatbot Node (Groq LLM)"]
    
    subgraph ToolDispatch ["Tool Invocation"]
        ToolCall["Tool: create_bar_chart<br/>Args: {'x_column': 'Pclass', 'y_column': 'Fare'}"]
    end
    
    subgraph VizExecution ["tools/visualization_tools.py: create_bar_chart()"]
        ValidateCols["validate_chart_column('Pclass', 'Fare')"]
        ValCheck{"Both columns matched with confidence >= 0.8?"}
        NumericCheck{"is_numeric_dtype(df[matched_y])?"}
        BuildPlotly["fig = px.bar(df, x=matched_x, y=matched_y, title=...)"]
        SaveFunc["save_chart(fig, 'bar') in helper_tools.py"]
    end

    subgraph ChartPersistence ["Disk Storage"]
        GenerateUUID["Filename: bar_{uuid.hex}.html"]
        WriteHTML["fig.write_html('outputs/charts/bar_xxx.html')"]
        ReturnJSON["Return {'success': True, 'chart_path': 'outputs/charts/bar_xxx.html'}"]
    end

    subgraph StreamlitRender ["Streamlit Multi-Modal Rendering"]
        ParseResponse["app.py: extract_chart_paths_from_response()"]
        ReadDisk["Open outputs/charts/bar_xxx.html"]
        EmbedHTML["components.html(html_content, height=500, scrolling=True)"]
        AttachDownload["st.download_button('💾 Download Chart')"]
    end

    UserPrompt --> Chatbot
    Chatbot --> ToolCall
    ToolCall --> ValidateCols
    
    ValidateCols --> ValCheck
    ValCheck -->|No| ErrCols["Return validation error"]
    ValCheck -->|Yes| NumericCheck
    
    NumericCheck -->|Non-numeric Y| ErrNum["Return 'Y must be numeric'"]
    NumericCheck -->|Valid| BuildPlotly
    
    BuildPlotly --> SaveFunc
    SaveFunc --> GenerateUUID
    GenerateUUID --> WriteHTML
    WriteHTML --> ReturnJSON
    
    ReturnJSON -->|ToolMessage| Chatbot
    Chatbot -->|Summary Text| ParseResponse
    
    ParseResponse --> ReadDisk
    ReadDisk --> EmbedHTML
    EmbedHTML --> AttachDownload
    AttachDownload --> User(["👤 Interactive Chart Displayed in Browser"])
```

### 2. Implementation Mechanics
- **Function**: `create_bar_chart` (and 7 other chart functions) in [tools/visualization_tools.py](file:///c:/Users/User/Desktop/AI-Data-Analyst/tools/visualization_tools.py).
- **Decoupled Architecture**: The LLM never receives raw HTML or SVG code (which would waste thousands of tokens). The tool writes a self-contained HTML file to `outputs/charts/` and returns only the file path string.
- **Frontend Mounting**: In [app.py#L174-L191](file:///c:/Users/User/Desktop/AI-Data-Analyst/app.py#L174-L191), `render_html_chart()` embeds the HTML directly into the DOM using an isolated `iframe` via `components.html()`.
