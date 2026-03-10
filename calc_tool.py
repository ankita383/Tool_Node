import os
import uvicorn
from typing import Annotated, TypedDict
from contextlib import AsyncExitStack, asynccontextmanager
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from fastapi import FastAPI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from fastmcp import Client
from langchain_mcp_adapters.tools import load_mcp_tools

load_dotenv()

class MathResponse(BaseModel):
    steps: str = Field(description="Reasoning steps used to solve the problem")
    final_answer: float = Field(description="Final numeric result")

class ChatState(TypedDict):
    messages: Annotated[list, add_messages]

# Ensure your server.py is running on port 8000/sse
mcp_client = Client("http://localhost:8000/sse") 
agent_app = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global agent_app
    async with AsyncExitStack() as stack:
        await stack.enter_async_context(mcp_client)
        await mcp_client.initialize()
        langchain_tools = await load_mcp_tools(mcp_client.session)
        llm = ChatGroq(
            model="llama-3.1-8b-instant", 
            groq_api_key=os.getenv("GROQ_API_KEY")
        ).bind_tools(langchain_tools)
        structured_llm = llm.with_structured_output(MathResponse)

        async def chatbot_node(state: ChatState):
            sys_msg = SystemMessage(content="You are a ReAct math agent. Solve step-by-step using tools.")
            response = await llm.ainvoke([sys_msg] + state["messages"])
            return {"messages": [response]}

        async def format_response(state: ChatState):
            result = await structured_llm.ainvoke(state["messages"])
            return {"messages": [AIMessage(content=result.model_dump_json())]}

        def route_after_assistant(state: ChatState):
            last_msg = state["messages"][-1]
            if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
                return "tools"
            return "formatter"

        builder = StateGraph(ChatState)
        builder.add_node("assistant", chatbot_node)
        builder.add_node("tools", ToolNode(langchain_tools))
        builder.add_node("formatter", format_response)
        
        builder.add_edge(START, "assistant")
        builder.add_conditional_edges("assistant", route_after_assistant)
        builder.add_edge("tools", "assistant")
        builder.add_edge("formatter", END)

        agent_app = builder.compile(checkpointer=MemorySaver())
        yield

app = FastAPI(lifespan=lifespan)

@app.post("/ask")
async def ask_agent(message: str, thread_id: str = "default"):
    if agent_app is None:
        return {"error": "Agent not initialized. Is the MCP server running?"}
    config = {"configurable": {"thread_id": thread_id}}
    input_data = {"messages": [HumanMessage(content=message)]}
    try:
        output = await agent_app.ainvoke(input_data, config)
        final_content = output["messages"][-1].content
        return {"response": MathResponse.model_validate_json(final_content)}
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8001)
