"""HumanGate-specific exceptions."""
from __future__ import annotations


class HumanGateUserStop(Exception):
    """Intentional user-requested workflow stop.

    ComfyUI v0.1 integration uses an exception to stop execution. ComfyUI will
    show this as an Error Report even though it is an intentional stop.
    """

    def __init__(self, message: str = "HumanGate stopped by user. This is an intentional v0.1 stop; ComfyUI may display it as an Error Report."):
        super().__init__(message)
