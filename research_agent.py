import os
import time
import json
from typing import Annotated, TypedDict, List
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch
from langchain_exa import ExaSearchResults
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()

class HousingResearch(BaseModel):
    city: str = Field(description="Name of the city")
    median_price: str = Field(description="Median housing price")
    price_to_income_ratio: float = Field(description="Price to income ratio")
    key_causes: List[str] = Field(description="Top causes of housing affordability crisis")
    policy_solutions: List[str] = Field(description="Policy responses for housing affordability")

class ResearchReport(BaseModel):
    title: str = Field(description="Title of the research report")
    objective: str = Field(
        description="Purpose of the research and what the analysis aims to investigate"
    )
    methodology: str = Field(
        description="Brief explanation of how the research was conducted using web search tools"
    )
    analysis: List[HousingResearch]
    global_trends: str = Field(
        description="Overall trends across all cities"
    )
    recommendations: List[str] = Field(
        description="Global policy recommendations based on the analysis"
    )
    conclusion: str = Field(
        description="Final summary of the research findings"
    )
    system_latency_ms: float = Field(
        description="Time taken by the agent to generate the report"
    )


tavily_search = TavilySearch(max_results=2)
exa_search = ExaSearchResults(
    exa_api_key=os.getenv("EXA_API_KEY"),
    num_results=1
)
tools = [tavily_search, exa_search]

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    max_tokens=2000
)
llm_with_tools = llm.bind_tools(tools)
llm_structured = llm.with_structured_output(ResearchReport)


class ResearchState(TypedDict):
    messages: Annotated[list, add_messages]
    start_time: float 


def researcher_node(state: ResearchState):
    if "start_time" not in state or state["start_time"] == 0:
        state["start_time"] = time.time()

    sys_msg = SystemMessage(content="""You are a housing policy researcher.
        Use Tavily for quick stats and Exa for deeper research.
        Analyze housing affordability for the requested cities and produce a structured report.
    """)
    last_msg = state["messages"][-1]

    if isinstance(last_msg, ToolMessage):
        last_msg.content = last_msg.content[:800]
        latency = (time.time() - state["start_time"]) * 1000
        messages = [sys_msg] + state["messages"][-2:]
        response = llm_structured.invoke(messages)
        response.system_latency_ms = round(latency, 2)
        return {
            "messages": [
                AIMessage(content=json.dumps(response.dict(), indent=2))
            ]
        }
    response = llm_with_tools.invoke([sys_msg] + state["messages"][-3:])
    return {"messages": [response]}


flow = StateGraph(ResearchState)
flow.add_node("researcher", researcher_node)
flow.add_node("tools", ToolNode(tools))
flow.add_edge(START, "researcher")
flow.add_conditional_edges("researcher", tools_condition)
flow.add_edge("tools", "researcher")
app = flow.compile(checkpointer=MemorySaver())

def run_research():
    print("--- Global Housing Research Agent ---")
    config = {"configurable": {"thread_id": "research_1"}}
    user_query = input("Enter your research query: ")
    print(f"\nProcessing request: {user_query}\n")
    input_data = {
        "messages": [HumanMessage(content=user_query)],
        "start_time": 0
    }

    for event in app.stream(input_data, config, stream_mode="values"):
        msg = event["messages"][-1]
        if isinstance(msg, AIMessage) and "{" in msg.content:
            print("RESEARCH REPORT")
            print(msg.content)

        elif hasattr(msg, "tool_calls") and msg.tool_calls:
            tool_name = msg.tool_calls[0]["name"]
            print(f"Action: Using {tool_name} to search for housing data...")

run_research()