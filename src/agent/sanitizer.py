import logging

from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine

logger = logging.getLogger(__name__)


class Sanitizer:
    """Detects and redacts PII from tool outputs before they reach the LLM."""

    PII_ENTITIES = [
        "PHONE_NUMBER",
        "EMAIL_ADDRESS",
        "CREDIT_CARD",
        "US_SSN",
        "IBAN_CODE",
        "IP_ADDRESS",
    ]

    def __init__(self):
        self.analyzer = AnalyzerEngine()
        self.anonymizer = AnonymizerEngine()

    def sanitize(self, text: str) -> tuple[str, list[str]]:
        """
        Scan text for PII and redact if found.

        Returns:
            Tuple of (sanitized_text, list of detected entity types)
        """
        results = self.analyzer.analyze(
            text=text,
            entities=self.PII_ENTITIES,
            language="en",
        )

        if not results:
            return text, []

        detected = [r.entity_type for r in results]
        logger.warning(
            "sanitizer.pii_detected",
            extra={
                "entity_types": detected,
                "count": len(results),
            },
        )

        anonymized = self.anonymizer.anonymize(text=text, analyzer_results=results)
        return anonymized.text, detected
