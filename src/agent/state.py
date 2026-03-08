from typing import TypedDict, Annotated

from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    guardrail_status: str
    guardrail_feedback: str
    cache_hit: bool
    cached_response: str
    user_role: str
    session_id: str

