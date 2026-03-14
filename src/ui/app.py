import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import chainlit as cl
from langchain_core.messages import HumanMessage
from src.agent.setup import setup_agent
from src.db.models import FeedbackRecord



@cl.on_chat_start
async def session_start():
    agent, session_factory = await setup_agent()
    cl.user_session.set("graph", agent)
    cl.user_session.set("session_factory", session_factory)

@cl.on_message
async def conversation(message: cl.Message):
    config = {"configurable": {"thread_id": cl.context.session.id}}
    graph = cl.user_session.get("graph")    
    
    response = await graph.ainvoke({"messages": [HumanMessage(content=message.content)]}, config=config)

    agent_response = response["messages"][-1]

    await cl.Message(content=agent_response.content).send()

@cl.on_feedback
async def on_feedback(feedback: cl.Feedback):
    session_factory = cl.user_session.get("session_factory")
    async with session_factory() as session:
        feedback_record = FeedbackRecord(
            session_id=cl.context.session.id,
            message_id=feedback.forId,
            rating=feedback.value,
            comment=feedback.comment
        )
        session.add(feedback_record)
        await session.commit()
    