"""
HTTP Middleware for request logging and tracing.

This module provides middleware that adds observability features to all HTTP requests,
including request/response logging, latency tracking, and distributed tracing via
request and session IDs.
"""

import uuid
import logging
import time
from typing import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from src.config.logging import set_request_context

logger = logging.getLogger(__name__)


class LoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware that provides comprehensive request logging and tracing.
    
    Features:
        - Generates unique request IDs for distributed tracing
        - Extracts or generates session IDs for user tracking
        - Logs request metadata (method, path, query params, client IP)
        - Measures and logs request latency
        - Propagates trace IDs via response headers
        - Handles and logs exceptions with full context
    
    Response Headers Added:
        - X-Request-ID: Unique identifier for the request
        - X-Session-ID: Session identifier (from header, cookie, or "unknown")
    """
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Process an incoming request with logging and tracing.
        
        Args:
            request: The incoming HTTP request
            call_next: The next middleware or route handler in the chain
            
        Returns:
            Response: The HTTP response with added tracing headers
            
        Raises:
            Exception: Re-raises any exception after logging it
        """
        # Generate unique request ID for distributed tracing
        request_id = str(uuid.uuid4())
        
        # Extract session ID from headers or cookies, fallback to "unknown"
        # Priority: X-Session-Id header > session_id cookie > "unknown"
        session_id = (
            request.headers.get("X-Session-Id") 
            or request.cookies.get("session_id") 
            or "unknown"
        )
        
        # Set context variables for structured logging throughout request lifecycle
        set_request_context(request_id=request_id, session_id=session_id)
        
        # Record start time for latency measurement
        start_time = time.perf_counter()

        # Log incoming request with relevant metadata
        logger.info(
            "api.request_received",
            extra={
                "method": request.method,
                "path": request.url.path,
                "query_params": str(request.query_params),
                "client_host": request.client.host if request.client else None,
            },
        )

        try:
            # Execute the actual request handler
            response = await call_next(request)
        except Exception as exc:
            # Log exception with full context before re-raising
            # Using logger.exception to capture stack trace
            logger.exception(
                "api.request_failed",
                extra={
                    "method": request.method,
                    "path": request.url.path,
                    "error": str(exc),
                },
            )
            raise

        # Calculate request latency in milliseconds
        latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
        
        # Log successful request completion with performance metrics
        logger.info(
            "api.request_completed",
            extra={
                "status_code": response.status_code,
                "latency_ms": latency_ms,
            },
        )
        
        # Add tracing headers to response for client-side correlation
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Session-ID"] = session_id
        
        return response