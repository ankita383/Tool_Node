import os
from typing import Annotated, TypedDict, Union
from dotenv import load_dotenv
from pydantic import BaseModel
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()

class MathResponse(BaseModel):
    operation: str
    a: float
    b: float
    result: Union[float, str]


@tool
def calculator(operation: str, a: float, b: float) -> float:
    """
    Performs basic arithmetic operations.
    Supported: add, subtract, multiply
    """
    if operation == "add":
        return a + b
    if operation == "subtract":
        return a - b
    if operation == "multiply":
        return a * b
    if operation == "divide":
        if b == 0:
            return "Division by zero is not allowed"
        return a / b
    return 0


tools = [calculator]


class ChatState(TypedDict):
    messages: Annotated[list, add_messages]

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    groq_api_key=os.getenv("GROQ_API_KEY")
).bind_tools(tools)

structured_llm = llm.with_structured_output(MathResponse)

def chatbot_node(state: ChatState):
    sys_msg = SystemMessage(content=""" You are a ReAct-style agent.Steps:
                            1. Understand the user's math request. 2. Decide which operation is needed. 
                            3. Use the calculator tool. 4. Return the final answer in structured format."""
    )
    response = llm.invoke([sys_msg] + state["messages"])
    return {"messages": [response]}

def format_response(state: ChatState):
    result = structured_llm.invoke(state["messages"])
    structured_message = AIMessage(content=result.model_dump_json())
    return {"messages": [structured_message]}


flow = StateGraph(ChatState)
flow.add_node("assistant", chatbot_node)
flow.add_node("tools", ToolNode(tools))
flow.add_node("formatter", format_response)
flow.add_edge(START, "assistant")
flow.add_conditional_edges("assistant",tools_condition)
flow.add_edge("tools", "formatter")

memory = MemorySaver()
app = flow.compile(checkpointer=memory)

def run_chatbot():
    print("ReAct Calculator Agent (type 'exit' to quit)\n")
    config = {"configurable": {"thread_id": "1"}}

    while True:
        user_input = input("User: ")
        if user_input.lower() in ["exit", "quit"]:
            break
        input_data = {"messages": [HumanMessage(content=user_input)]}
        for event in app.stream(input_data, config, stream_mode="values"):
            assistant_msg = event["messages"][-1]
            if hasattr(assistant_msg, "tool_calls") and assistant_msg.tool_calls:
                print("Reasoning: Agent calling calculator tool...")
            elif isinstance(assistant_msg, AIMessage):
                print("Structured Output:", assistant_msg.content)

run_chatbot()