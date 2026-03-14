from fastapi import APIRouter, Request
from langchain_core.messages import HumanMessage

from src.api.schema import ChatRequest, ChatResponse

router = APIRouter(prefix="/api")

@router.post("/chat", response_model=ChatResponse)
async def chat(request: Request, body: ChatRequest):
    graph = request.app.state.graph
    config = {"configurable": {"thread_id": body.session_id}}
    response = await graph.ainvoke({
        "messages": [HumanMessage(content=body.message)]
    }, config=config)

    agent_response = response["messages"][-1]

    return ChatResponse(
        response= agent_response.content,
        session_id= body.session_id
    )
    