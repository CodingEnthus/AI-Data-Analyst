# 🔄 Complete Master Execution Workflow

This document illustrates the end-to-end execution workflow of the **AI Data Analyst**, detailing the exact step-by-step path taken from the initial user request to the final response rendering.

---

## 1. Master Runtime Sequence Diagram (Mermaid)

```mermaid
sequenceDiagram
    autonumber
    actor User as 👤 User
    participant Streamlit as 🖥️ app.py (Streamlit UI)
    participant DataStore as 💾 data_store.df
    participant LangGraph as 🧠 agent.py (StateGraph)
    participant Checkpointer as 🗄️ MemorySaver (thread_id)
    participant ChatbotNode as 🤖 'chatbot' Node
    participant LLM as ⚡ Groq Cloud / Ollama
    participant Router as 🔀 tools_condition
    participant ToolNode as ⚙️ 'tools' Node (ToolNode)
    participant PythonTool as 🐍 Registered Tool (tools/*)
    participant CoreEngine as ⚡ Engines (Pandas / DuckDB / ML)
    participant Disk as 📦 Local Disk (outputs/ & models/)

    %% 1. Ingestion / Trigger
    User->>Streamlit: Enters prompt in st.chat_input()
    Streamlit->>Streamlit: Append {"role": "user", "content": prompt} to st.session_state
    
    %% 2. Graph Invocation
    Streamlit->>LangGraph: graph.invoke({"messages": [HumanMessage(content=prompt)]}, config={"thread_id": id})
    LangGraph->>Checkpointer: Retrieve checkpoint history for thread_id
    Checkpointer-->>LangGraph: Returns historical messages [HumanMessage, AIMessage, ...]
    
    %% 3. Node: Chatbot
    LangGraph->>ChatbotNode: Execute chatbot(state)
    ChatbotNode->>ChatbotNode: Prepend SystemMessage(SYSTEM_PROMPT) to state["messages"]
    ChatbotNode->>LLM: llm_with_tools.invoke(messages)
    
    %% 4. Cognitive Decision
    alt Scenario A: Direct Natural Language Response (No Tool Required)
        LLM-->>ChatbotNode: Returns AIMessage(content="Hello! How can I help you analyze your data today?")
        ChatbotNode-->>LangGraph: Returns {"messages": [AIMessage]}
        LangGraph->>Router: tools_condition inspects last message
        Router-->>LangGraph: No tool_calls found -> Route to END
        LangGraph->>Checkpointer: Save final state checkpoint for thread_id
        LangGraph-->>Streamlit: Returns response state dictionary
        Streamlit->>Streamlit: Extract content from response["messages"][-1]
        Streamlit-->>User: Renders text in chat bubble

    else Scenario B: Tool Call Required (Analytics / SQL / Chart / ML)
        LLM-->>ChatbotNode: Returns AIMessage(tool_calls=[{name: "func_name", args: {...}, id: "call_123"}])
        ChatbotNode-->>LangGraph: Returns {"messages": [AIMessage]}
        LangGraph->>Router: tools_condition inspects last message
        Router-->>LangGraph: tool_calls detected -> Route to "tools" Node
        
        %% 5. Node: Tools Execution
        LangGraph->>ToolNode: Execute ToolNode(state)
        ToolNode->>PythonTool: func_name(**args)
        
        %% 6. Engine Execution
        PythonTool->>DataStore: Access data_store.df
        PythonTool->>CoreEngine: Execute calculation / SQL / ML pipeline
        
        opt Produces Artifact (Visualization or ML Model)
            CoreEngine->>Disk: Save HTML chart to outputs/charts/ or .joblib to models/
        end
        
        CoreEngine-->>PythonTool: Raw computation results / chart path
        PythonTool-->>ToolNode: Returns JSON / formatted string result
        ToolNode-->>LangGraph: Wraps in ToolMessage(content=..., tool_call_id="call_123")
        
        %% 7. Cyclic ReAct Loop Back to LLM
        LangGraph->>ChatbotNode: Re-invoke chatbot(state) with ToolMessage appended
        ChatbotNode->>LLM: llm_with_tools.invoke(messages + [ToolMessage])
        Note over LLM: LLM reads deterministic tool output<br/>and synthesizes user-friendly explanation
        LLM-->>ChatbotNode: Returns AIMessage(content="Based on the analysis, ...")
        ChatbotNode-->>LangGraph: Returns {"messages": [AIMessage]}
        
        %% 8. Final Routing & Display
        LangGraph->>Router: tools_condition inspects last message
        Router-->>LangGraph: No further tool_calls -> Route to END
        LangGraph->>Checkpointer: Save state checkpoint for thread_id
        LangGraph-->>Streamlit: Returns final state dictionary
        
        %% 9. Multi-Modal Artifact Parsing in Streamlit
        Streamlit->>Streamlit: extract_chart_paths_from_response(response)
        Streamlit->>Streamlit: Render markdown text
        opt Charts Detected
            Streamlit->>Disk: Read HTML file
            Streamlit-->>User: Embed interactive Plotly widget via components.html() + Download button
        end
        Streamlit-->>User: Renders assistant response in chat UI
    end
```

---

## 2. Step-by-Step Execution Analysis

### Step 1: User Request Ingestion
- The user inputs text via `st.chat_input()` ([app.py#L307](file:///c:/Users/User/Desktop/AI-Data-Analyst/app.py#L307)).
- `app.py` appends the prompt to `st.session_state.messages` to ensure instant UI rendering.

### Step 2: LangGraph Invocation & Checkpoint Retrieval
- `app.py` calls:
  ```python
  config = {"configurable": {"thread_id": st.session_state.thread_id}}
  response = graph.invoke(
      {"messages": [HumanMessage(content=active_input)]},
      config=config
  )
  ```
- LangGraph's `MemorySaver` uses `thread_id` to look up any prior messages from this session.

### Step 3: `chatbot` Node Execution
- Inside `chatbot(state)` ([agent.py#L82-L88](file:///c:/Users/User/Desktop/AI-Data-Analyst/agent.py#L82-L88)), the `SYSTEM_PROMPT` is prepended to the message history.
- The bound model (`llm_with_tools`) is invoked. It checks the prompt against its tool definitions.

### Step 4: Routing Decision via `tools_condition`
- LangGraph evaluates `tools_condition` ([agent.py#L123-L126](file:///c:/Users/User/Desktop/AI-Data-Analyst/agent.py#L123-L126)):
  - **Branch 1 (Direct Answer)**: If the prompt was conversational (*"What tools do you have?"*), the LLM emits plain text without `tool_calls`. Routing proceeds immediately to `END`.
  - **Branch 2 (Tool Invocation)**: If analytical computation is needed, the LLM emits one or more structured `tool_calls` containing the tool name and validated arguments. Routing proceeds to `"tools"`.

### Step 5: `ToolNode` Execution
- `ToolNode` matches the requested tool name against the 18 registered Python functions.
- The Python function executes locally, accessing `data_store.df`, invoking DuckDB, Pandas, or Scikit-Learn.
- If an artifact is produced (e.g., Plotly figure or ML model), it is saved to disk (`outputs/charts/` or `models/`).
- The function returns a structured dictionary or string. `ToolNode` wraps this into a `ToolMessage` carrying the unique `tool_call_id`.

### Step 6: The Cyclic Loop (`tools` $\rightarrow$ `chatbot`)
- The edge `graph_builder.add_edge("tools", "chatbot")` ([agent.py#L127](file:///c:/Users/User/Desktop/AI-Data-Analyst/agent.py#L127)) brings execution back to the `chatbot` node.
- The LLM receives the `ToolMessage` containing the exact mathematical facts.
- The LLM writes a conversational explanation summarizing the findings.

### Step 7: Streamlit Artifact Extraction & Multi-Modal Rendering
- Once the graph reaches `END`, control returns to `app.py`.
- `app.py` invokes `extract_chart_paths_from_response(response)` ([app.py#L142-L172](file:///c:/Users/User/Desktop/AI-Data-Analyst/app.py#L142-L172)).
- It scans all `ToolMessage` payloads for keys named `chart_path`, `charts`, or `feature_importance_chart_path`.
- If an HTML chart file exists on disk, `render_html_chart()` embeds it directly into the chat stream via `streamlit.components.v1.html(html_content, height=500, scrolling=True)` accompanied by an `st.download_button()`.
