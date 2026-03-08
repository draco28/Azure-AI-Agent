import logging
import time

from src.agent.state import AgentState
from src.agent.guardrails import Guardrail
from src.agent.sanitizer import Sanitizer
from src.agent.cache import CacheManager

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

logger = logging.getLogger(__name__)


# ─── Input Guardrail Node ────────────────────────────────────

def create_input_guardrail_node():
    guardrail = Guardrail()

    async def input_guardrail_node(state: AgentState) -> dict:
        logger.info("node.input_guardrail.started")
        result = await guardrail.validate(state)
        return result

    return input_guardrail_node


# ─── Cache Check Node ────────────────────────────────────────

def create_cache_check_node(cache_manager: CacheManager):
    async def cache_check_node(state: AgentState) -> dict:
        logger.info("node.cache_check.started")

        user_message = state["messages"][-1].content
        user_role = state.get("user_role", "unknown")

        cached_response = await cache_manager.check(user_message, user_role)

        if cached_response:
            return {
                "cache_hit": True,
                "cached_response": cached_response,
            }

        return {"cache_hit": False, "cached_response": ""}

    return cache_check_node


# ─── Agent Node ───────────────────────────────────────────────

def create_agent_node(llm_with_tools):
    system_message = SystemMessage(content="""You are a helpful AI assistant with access to specialized tools for document-related queries.

Available tools:
- rag_tool: Search document content. Use this for questions about what's INSIDE documents - policies, procedures, facts, numbers, guidelines, or any textual content.
- file_status_tool: Check document workflow status. Use this for questions about a document's STATE - approval status (approved/rejected/pending/processing), who submitted it, which department owns it, or submission dates.

Guidelines:
1. Analyze each question to determine which tool(s) are needed:
   - Content questions (what does it say, what are the rules, explain the policy) → rag_tool
   - Status questions (is it approved, who submitted it, what department) → file_status_tool
   - Combined questions (is it approved AND what does it say about X) → use BOTH tools

2. For multi-part questions, call each relevant tool separately and combine the results in your response.

3. When citing document content, include the source filename and relevant context.

4. If a tool returns no results, inform the user and suggest alternatives.

5. For general conversation unrelated to documents or file status, respond naturally without tools.

6. Be concise but thorough, clearly distinguishing between content information and status information in your responses.
    """)

    def agent_node(state: AgentState) -> dict:
        start_time = time.perf_counter()

        logger.info(
            "agent.invoked",
            extra={
                "turn_count": len(state["messages"]),
            },
        )

        response = llm_with_tools.invoke([system_message] + state['messages'])

        latency_ms = round((time.perf_counter() - start_time) * 1000, 2)

        # Check if the LLM decided to call tools
        if isinstance(response, AIMessage) and response.tool_calls:
            for tc in response.tool_calls:
                logger.info(
                    "agent.tool_selected",
                    extra={
                        "tool_name": tc["name"],
                        "tool_args": tc["args"],
                        "latency_ms": latency_ms,
                    },
                )
        else:
            logger.info(
                "agent.response_generated",
                extra={
                    "response_length": len(response.content) if response.content else 0,
                    "latency_ms": latency_ms,
                },
            )

        return {"messages": [response]}

    return agent_node


# ─── Output Sanitization Node ────────────────────────────────

def create_output_sanitization_node():
    sanitizer = Sanitizer()

    def output_sanitization_node(state: AgentState) -> dict:
        """Sanitize tool outputs before they go back to the agent."""
        logger.info("node.output_sanitization.started")

        # Get the last message — should be a ToolMessage from the tools node
        last_message = state["messages"][-1]

        if not isinstance(last_message, ToolMessage):
            return {}

        sanitized_content, detected = sanitizer.sanitize(last_message.content)

        if detected:
            # Replace the tool message content with sanitized version
            return {
                "messages": [
                    ToolMessage(
                        content=sanitized_content,
                        tool_call_id=last_message.tool_call_id,
                    )
                ]
            }

        return {}

    return output_sanitization_node


# ─── Cache Store Node ────────────────────────────────────────

def create_cache_store_node(cache_manager: CacheManager):
    async def cache_store_node(state: AgentState) -> dict:
        """Store the agent's final response in the semantic cache."""
        logger.info("node.cache_store.started")

        # Find the original user query (first HumanMessage)
        user_query = None
        for msg in reversed(state["messages"]):
            if isinstance(msg, HumanMessage):
                user_query = msg.content
                break

        # Get the agent's final response (last AIMessage without tool calls)
        last_message = state["messages"][-1]

        if user_query and isinstance(last_message, AIMessage) and not last_message.tool_calls:
            user_role = state.get("user_role", "unknown")
            await cache_manager.store(user_query, last_message.content, user_role)

        return {}

    return cache_store_node
