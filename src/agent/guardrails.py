import logging
import re

from nemoguardrails import RailsConfig, LLMRails

from src.agent.state import AgentState

logger = logging.getLogger(__name__)

REFUSAL_PREFIX = "I'm sorry, I cannot process this request"


class Guardrail:
    INJECTION_PATTERNS = [
        re.compile(p, re.IGNORECASE) for p in [
            r"ignore.{0,20}instructions",
            r"ignore.{0,20}prior",
            r"disregard.{0,20}instructions",
            r"forget.{0,20}rules",
            r"you are now",
            r"act as",
            r"pretend.{0,10}you",
            r"reveal.{0,20}(system|prompt)",
            r"show.{0,20}system prompt",
            r"override.{0,20}(safety|restrictions)",
        ]
    ]

    def __init__(self):
        config = RailsConfig.from_path("src/agent/guardrails_config")
        self.rails = LLMRails(config)

    def validate_regex(self, message: str) -> bool:
        """Layer 1: Fast regex check. Returns False if injection detected."""
        for pattern in self.INJECTION_PATTERNS:
            if pattern.search(message):
                logger.warning(
                    "guardrail.regex.blocked",
                    extra={"pattern": pattern.pattern},
                )
                return False
        return True

    async def validate_nemo(self, message: str) -> bool:
        """Layer 2: NeMo LLM-based check. Returns False if blocked."""
        response = await self.rails.generate_async(
            messages=[{"role": "user", "content": message}]
        )
        content = response.get("content", "")
        if content.startswith(REFUSAL_PREFIX):
            logger.warning("guardrail.nemo.blocked")
            return False
        return True

    async def validate(self, state: AgentState) -> dict:
        """Run layered guardrail checks. Returns state update."""
        message = state["messages"][-1].content

        # Layer 1: fast regex
        if not self.validate_regex(message):
            return {
                "guardrail_status": "blocked",
                "guardrail_feedback": "Request blocked: potential prompt injection detected.",
            }

        # Layer 2: NeMo (only if regex passed)
        if not await self.validate_nemo(message):
            return {
                "guardrail_status": "blocked",
                "guardrail_feedback": "Request blocked: content policy violation detected.",
            }

        logger.info("guardrail.passed")
        return {
            "guardrail_status": "safe",
            "guardrail_feedback": "",
        }
