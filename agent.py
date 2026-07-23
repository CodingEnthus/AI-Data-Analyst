from dotenv import load_dotenv
import os
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

from prompts import SYSTEM_PROMPT

from tools.dataset_tools import *
from tools.statistics_tools import *
from tools.visualization_tools import *

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph,START,END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode,tools_condition

from typing import Annotated
from typing_extensions import TypedDict

load_dotenv()
groq_api_key=os.getenv("GROQ_API_KEY")
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    api_key=groq_api_key
)
tools = [
    load_dataset,
    get_dataset_info,
    calculate_statistics,
    create_bar_chart,
    create_line_chart,
    create_scatter_plot,
    create_histogram,
    create_pie_chart,
    create_heatmap
]
class State(TypedDict):
    messages:Annotated[list,add_messages]
tool_node=ToolNode(tools)
llm_with_tools = llm.bind_tools(tools)
def chatbot(state:State):
    messages = [
        SystemMessage(content=SYSTEM_PROMPT)
    ] + state["messages"]

    response = llm_with_tools.invoke(messages)

    return{
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
        user_input = input("\nYou: ")

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