import os
import uvicorn
from typing import Annotated, TypedDict, List
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()

class MathResponse(BaseModel):
    steps: str = Field(description="Reasoning steps used to solve the problem")
    final_answer: float = Field(description="Final numeric result")

@tool
def add(a: float, b: float) -> float:
    """Adds two numbers."""
    return a + b

@tool
def subtract(a: float, b: float) -> float:
    """Subtracts two numbers."""
    return a - b

@tool
def multiply(a: float, b: float) -> float:
    """Multiplies two numbers."""
    return a * b

@tool
def divide(a: float, b: float) -> float:
    """Divides two numbers."""
    if b == 0: return "Error: Division by zero"
    return a / b

tools = [add, subtract, multiply, divide]

class ChatState(TypedDict):
    messages: Annotated[list, add_messages]

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0,
    groq_api_key=os.getenv("GROQ_API_KEY")
).bind_tools(tools)

structured_llm = llm.with_structured_output(MathResponse)

def chatbot_node(state: ChatState):
    sys_msg = SystemMessage(content=(
        "You are a ReAct-style math agent. Break complex expressions into steps. "
        "Solve each step using tools. Once done, stop using tools."
    ))
    response = llm.invoke([sys_msg] + state["messages"])
    return {"messages": [response]}

def format_response(state: ChatState):
    result = structured_llm.invoke(state["messages"])
    return {"messages": [AIMessage(content=result.model_dump_json())]}

def route_after_assistant(state: ChatState):
    last_msg = state["messages"][-1]
    if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
        return "tools"
    return "formatter"

builder = StateGraph(ChatState)
builder.add_node("assistant", chatbot_node)
builder.add_node("tools", ToolNode(tools))
builder.add_node("formatter", format_response)
builder.add_edge(START, "assistant")
builder.add_conditional_edges("assistant", route_after_assistant)
builder.add_edge("tools", "assistant")
builder.add_edge("formatter", END)

memory = MemorySaver()
agent_app = builder.compile(checkpointer=memory)

app = FastAPI(title="ReAct Agent API")

class ChatRequest(BaseModel):
    message: str
    thread_id: str = "default_user"

class ChatResponse(BaseModel):
    response: MathResponse

@app.post("/ask", response_model=ChatResponse)
async def ask_agent(request: ChatRequest):
    try:
        config = {"configurable": {"thread_id": request.thread_id}}
        input_data = {"messages": [HumanMessage(content=request.message)]}
        output = agent_app.invoke(input_data, config)
        final_message_content = output["messages"][-1].content
        math_data = MathResponse.model_validate_json(final_message_content)
        return {"response": math_data}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def read_root():
    return {"status": "Agent is online", "docs": "/docs"}

uvicorn.run(app, host="127.0.0.1", port=8000)
