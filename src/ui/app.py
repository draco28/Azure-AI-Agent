import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import chainlit as cl
from langchain_core.messages import HumanMessage
from src.agent.setup import setup_agent



@cl.on_chat_start
async def session_start():
    agent = await setup_agent()
    cl.user_session.set("graph", agent)
    cl.user_session.set("messages", [])

@cl.on_message
async def conversation(message: cl.Message):
    graph = cl.user_session.get("graph")
    history = cl.user_session.get("messages")
    
    history.append(HumanMessage(content=message.content))
    
    response = await graph.ainvoke({"messages": history})

    agent_response = response["messages"][-1]

    cl.user_session.set("messages", response["messages"])

    await cl.Message(content=agent_response.content).send()