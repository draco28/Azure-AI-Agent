import logging
import os

from contextvars import ContextVar
from typing import Optional

from pythonjsonlogger import jsonlogger

request_id_var: ContextVar[str] = ContextVar("request_id", default="unknown")
session_id_var: ContextVar[str] = ContextVar("session_id", default="unknown")

class ContextAwareJsonFormatter(jsonlogger.JsonFormatter):
    def __init__(self, service_name: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.service_name = service_name
        
    def add_fields(self, log_record, record, message_dict):
        super().add_fields(log_record, record, message_dict)
        
        if not log_record.get("timestamp"):
            log_record["timestamp"] = self.formatTime(record, self.datefmt or "%Y-%m-%dT%H:%M:%S.%fZ")

        log_record["level"] = record.levelname
        log_record["service"] = self.service_name
        log_record["request_id"] = request_id_var.get()
        log_record["session_id"] = session_id_var.get()

def setup_logging(service_name: str, level: Optional[str] = None):
    log_level = (
        level or os.getenv("LOG_LEVEL", "INFO")
    ).upper()
    
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    root_logger.handlers.clear()
    
    handler = logging.StreamHandler()
    formatter = ContextAwareJsonFormatter(
        service_name=service_name,
        fmt="%(message)s"
    )
    handler.setFormatter(formatter)
    root_logger.addHandler(handler)

def set_request_context(request_id: str, session_id: str) -> None:
    request_id_var.set(request_id)
    session_id_var.set(session_id)
