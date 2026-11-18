from typing import Dict
from app.schemas import FaultInjectionConfig


class ToolExecutionError(Exception):
    """Raised when a tool invocation fails due to timeout, API error, malformed payload, or empty output."""

    def __init__(self, tool_name: str, error_type: str, message: str, is_injected: bool = False):
        super().__init__(message)
        self.tool_name = tool_name
        self.error_type = error_type
        self.message = message
        self.is_injected = is_injected


class FaultInjector:
    """
    Manages deliberate failure injection for testing and demonstrating agent self-correction
    (Requirement 1.d and 3.4 of the take-home assignment).
    """

    def __init__(self, config: FaultInjectionConfig):
        self.config = config
        self._triggered_counts: Dict[str, int] = {}

    def check_and_inject(self, tool_name: str) -> None:
        if not self.config.enabled:
            return

        target = self.config.target_tool
        if target != "any" and target != tool_name:
            return

        count = self._triggered_counts.get(tool_name, 0)
        if count >= self.config.fail_count:
            return

        self._triggered_counts[tool_name] = count + 1
        mode = self.config.failure_mode

        if mode == "timeout":
            raise ToolExecutionError(
                tool_name=tool_name,
                error_type="TimeoutError",
                message=f"[Simulated Fault #{count + 1}] Upstream request to '{tool_name}' timed out after 10000ms (ReadTimeout).",
                is_injected=True,
            )
        elif mode == "http_503_rate_limit":
            raise ToolExecutionError(
                tool_name=tool_name,
                error_type="HTTP503ServiceUnavailable",
                message=f"[Simulated Fault #{count + 1}] Endpoint '{tool_name}' returned HTTP 503 Service Unavailable / Rate Limit Exceeded.",
                is_injected=True,
            )
        elif mode == "malformed_payload":
            raise ToolExecutionError(
                tool_name=tool_name,
                error_type="MalformedPayloadError",
                message=f"[Simulated Fault #{count + 1}] Tool '{tool_name}' returned truncated/corrupted JSON schema missing required 'results' key.",
                is_injected=True,
            )
        elif mode == "empty_results":
            raise ToolExecutionError(
                tool_name=tool_name,
                error_type="EmptyResultSetError",
                message=f"[Simulated Fault #{count + 1}] Query on '{tool_name}' returned 0 matching records due to overly restrictive syntax.",
                is_injected=True,
            )
