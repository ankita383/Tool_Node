import os
import time
import uvicorn
import json
from typing import Annotated, TypedDict, List
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException
from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()

class HousingResearch(BaseModel):
    city: str
    median_price: str
    price_to_income_ratio: float
    key_causes: List[str]
    policy_solutions: List[str]

class ResearchReport(BaseModel):
    title: str
    objective: str
    methodology: str
    analysis: List[HousingResearch]
    global_trends: str
    recommendations: List[str]
    conclusion: str
    system_latency_ms: float = 0.0

tavily_search = TavilySearch(max_results=3)
tools = [tavily_search]

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
llm_with_tools = llm.bind_tools(tools)
llm_structured = llm.with_structured_output(ResearchReport)

class ResearchState(TypedDict):
    messages: Annotated[list, add_messages]
    start_time: float

def researcher_node(state: ResearchState):
    if state.get("start_time", 0) == 0:
        state["start_time"] = time.time()
        
    sys_msg = SystemMessage(content=(
        "You are a housing policy researcher. Use Tavily to find data. "
        "Search for median prices and income ratios for the requested cities. "
        "When you have found all the data, provide a final summary of your findings "
        "so the formatter can create the final report. Also report the latency"
    ))
    response = llm_with_tools.invoke([sys_msg] + state["messages"])
    return {"messages": [response], "start_time": state["start_time"]}

def formatter_node(state: ResearchState):
    print("--- Generating Structured Report ---")
    report = llm_structured.invoke(state["messages"])
    latency = (time.time() - state["start_time"]) * 1000
    report.system_latency_ms = round(latency, 2)
    
    return {"messages": [AIMessage(content=report.model_dump_json())]}


def should_continue(state: ResearchState):
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"
    return "formatter"

workflow = StateGraph(ResearchState)

workflow.add_node("researcher", researcher_node)
workflow.add_node("tools", ToolNode(tools))
workflow.add_node("formatter", formatter_node)
workflow.add_edge(START, "researcher")
workflow.add_conditional_edges("researcher", should_continue)
workflow.add_edge("tools", "researcher") 
workflow.add_edge("formatter", END)

app_graph = workflow.compile(checkpointer=MemorySaver())

app = FastAPI(title="Housing Agent API")

class QueryRequest(BaseModel):
    query: str
    thread_id: str = "default_research"

@app.post("/research")
async def run_research_endpoint(request: QueryRequest):
    config = {"configurable": {"thread_id": request.thread_id}}
    input_data = {
        "messages": [HumanMessage(content=request.query)],
        "start_time": time.time()
    }
    
    try:
        final_state = app_graph.invoke(input_data, config)
        return json.loads(final_state["messages"][-1].content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

uvicorn.run(app, host="127.0.0.1", port=8000)
