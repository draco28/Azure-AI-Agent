from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode

from src.agent.state import AgentState
from src.agent.cache import CacheManager
from src.agent.nodes import (
    create_agent_node,
    create_input_guardrail_node,
    create_cache_check_node,
    create_cache_store_node,
    create_output_sanitization_node,
)


def guardrail_condition(state: AgentState) -> str:
    """Route based on guardrail result: blocked → END, safe → continue."""
    if state.get("guardrail_status") == "blocked":
        return "blocked"
    return "safe"


def cache_condition(state: AgentState) -> str:
    """Route based on cache result: hit → END, miss → continue to agent."""
    if state.get("cache_hit"):
        return "hit"
    return "miss"


def agent_condition(state: AgentState) -> str:
    """Route based on agent output: tool calls → tools, final response → cache_store."""
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return "cache_store"


def build_graph(llm_with_tools, tools: list, cache_manager: CacheManager = None, checkpointer=None) -> StateGraph:
    # Create node functions
    input_guardrail_node = create_input_guardrail_node()
    cache_check_node = create_cache_check_node(cache_manager)
    agent_node = create_agent_node(llm_with_tools)
    output_sanitization_node = create_output_sanitization_node()
    cache_store_node = create_cache_store_node(cache_manager)

    graph = StateGraph(AgentState)

    # Add all nodes
    graph.add_node("input_guardrail", input_guardrail_node)
    graph.add_node("cache_check", cache_check_node)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(tools))
    graph.add_node("output_sanitization", output_sanitization_node)
    graph.add_node("cache_store", cache_store_node)

    # Section 1: Entry pipeline (runs once per request)
    graph.add_edge(START, "input_guardrail")

    graph.add_conditional_edges("input_guardrail", guardrail_condition, {
        "blocked": END,
        "safe": "cache_check",
    })

    graph.add_conditional_edges("cache_check", cache_condition, {
        "hit": END,
        "miss": "agent",
    })

    # Section 2: ReAct loop (may repeat)
    graph.add_conditional_edges("agent", agent_condition, {
        "tools": "tools",
        "cache_store": "cache_store",
    })

    graph.add_edge("tools", "output_sanitization")
    graph.add_edge("output_sanitization", "agent")

    # Cache store → END
    graph.add_edge("cache_store", END)

    return graph.compile(checkpointer=checkpointer)
