from dotenv import load_dotenv
import os
import httpx
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

from prompts import SYSTEM_PROMPT

from tools.dataset_tools import *
from tools.statistics_tools import *
from tools.visualization_tools import *
from tools.duckdb_tools import query_dataset_with_sql
from tools.ml_tools import (
    detect_anomalies,
    train_classification_model,
    train_regression_model,
    perform_clustering,
    evaluate_trained_model,
    predict_with_model
)

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from typing import Annotated
from typing_extensions import TypedDict

load_dotenv()

# Select LLM Provider: "groq" (default) or "ollama"
llm_provider = os.getenv("LLM_PROVIDER", "groq").lower()

if llm_provider == "ollama":
    from langchain_ollama import ChatOllama
    ollama_model = os.getenv("OLLAMA_MODEL", "qwen3:4b")
    ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    llm = ChatOllama(
        model=ollama_model,
        base_url=ollama_base_url,
        temperature=0
    )
else:
    # Default provider: Groq
    from langchain_groq import ChatGroq
    groq_api_key = os.getenv("GROQ_API_KEY")
    groq_model = os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")
    max_tokens_env = int(os.getenv("GROQ_MAX_TOKENS", "700"))
    max_tokens = min(max_tokens_env, 1000)

    llm = ChatGroq(
        model=groq_model,
        temperature=0,
        api_key=groq_api_key or "missing_key",
        max_tokens=max_tokens
    )

tools = [
    load_dataset,
    get_dataset_info,
    calculate_statistics,
    query_dataset_with_sql,
    create_bar_chart,
    create_line_chart,
    create_scatter_plot,
    create_histogram,
    create_pie_chart,
    create_heatmap,
    create_box_plot,
    create_violin_plot,
    detect_anomalies,
    train_classification_model,
    train_regression_model,
    perform_clustering,
    evaluate_trained_model,
    predict_with_model
]
class State(TypedDict):
    messages:Annotated[list,add_messages]
tool_node=ToolNode(tools)
llm_with_tools = llm.bind_tools(tools)
def chatbot(state: State):
    messages = [
        SystemMessage(content=SYSTEM_PROMPT)
    ] + state["messages"]

    try:
        response = llm_with_tools.invoke(messages)
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 429:
            response = AIMessage(
                content="⚠️ [Groq Rate Limit]: Token limit reached (OTPM). Please wait a moment before trying again."
            )
        elif e.response.status_code == 401:
            response = AIMessage(
                content="⚠️ [Groq Authentication Error]: Invalid or missing GROQ_API_KEY. Please verify your Groq API key in the .env file or sidebar."
            )
        else:
            response = AIMessage(content=f"⚠️ [Groq API Error]: HTTP {e.response.status_code} - {e}")
    except (httpx.ConnectError, httpx.ConnectTimeout, ConnectionError) as e:
        if llm_provider == "ollama":
            response = AIMessage(
                content="⚠️ [Ollama Connection Error]: Unable to connect to local Ollama server at http://localhost:11434. Please start it using 'ollama serve' or 'ollama run qwen3:4b'."
            )
        else:
            response = AIMessage(content=f"⚠️ [Network Error]: Could not reach Groq servers: {e}")
    except Exception as e:
        err_msg = str(e)
        if "api_key" in err_msg.lower():
            response = AIMessage(
                content="⚠️ [Groq Configuration Error]: GROQ_API_KEY is missing or invalid. Please check your .env file."
            )
        else:
            response = AIMessage(content=f"⚠️ [Error]: {err_msg}")

    return {
        "messages": [response]
    }
graph_builder=StateGraph(State)
graph_builder.add_node("chatbot",chatbot)
graph_builder.add_node("tools",tool_node)
graph_builder.add_edge(START,"chatbot")
graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition
)
graph_builder.add_edge("tools","chatbot")
memory=MemorySaver()
graph=graph_builder.compile(checkpointer=memory)
if __name__ == "__main__":

    config = {
        "configurable": {
            "thread_id": "1"
        }
    }

    while True:
        try:
            user_input = input("\nYou: ").strip()

            if not user_input:
                continue

            if user_input.lower() in ["exit", "quit"]:
                break

            response = graph.invoke(
                {
                    "messages": [
                        HumanMessage(content=user_input)
                    ]
                },
                config=config
            )

            print("\nAI:", response["messages"][-1].content)
        except Exception as e:
            print(f"\n[Error]: {e}")