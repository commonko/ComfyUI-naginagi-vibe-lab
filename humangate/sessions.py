"""Thread-safe HumanGate session manager."""
from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class GateSession:
    prompt_id: Optional[str]
    node_id: Optional[str]
    kind: str
    payload: Dict[str, Any]
    gate_id: str = field(init=False)
    created_at: float = field(default_factory=time.time)
    result: Optional[Dict[str, Any]] = None
    event: threading.Event = field(default_factory=threading.Event)

    def __post_init__(self) -> None:
        p = self.prompt_id or "unknown_prompt"
        n = self.node_id or "unknown_node"
        self.gate_id = f"{p}:{n}:{uuid.uuid4().hex}"

    def public_dict(self) -> Dict[str, Any]:
        return {
            "gate_id": self.gate_id,
            "prompt_id": self.prompt_id,
            "node_id": self.node_id,
            "kind": self.kind,
            "payload": self.payload,
            "created_at": self.created_at,
            "age_sec": max(0.0, time.time() - self.created_at),
            "resolved": self.event.is_set(),
        }


class GateSessionManager:
    def __init__(self) -> None:
        self._sessions: Dict[str, GateSession] = {}
        self._lock = threading.Lock()

    def create(self, prompt_id: Optional[str], node_id: Optional[str], kind: str, payload: Dict[str, Any]) -> GateSession:
        session = GateSession(prompt_id=prompt_id, node_id=node_id, kind=kind, payload=payload)
        with self._lock:
            self._sessions[session.gate_id] = session
        return session

    def resolve(self, gate_id: str, result: Dict[str, Any]) -> bool:
        with self._lock:
            session = self._sessions.get(gate_id)
        if session is None:
            return False
        session.result = result
        session.event.set()
        return True

    def get(self, gate_id: str) -> Optional[GateSession]:
        with self._lock:
            return self._sessions.get(gate_id)

    def pop(self, gate_id: str) -> Optional[GateSession]:
        with self._lock:
            return self._sessions.pop(gate_id, None)

    def list_active(self) -> list[Dict[str, Any]]:
        with self._lock:
            sessions = list(self._sessions.values())
        return [session.public_dict() for session in sessions]

    def cleanup(self, max_age_sec: int = 3600) -> int:
        now = time.time()
        removed = 0
        with self._lock:
            for gate_id in list(self._sessions.keys()):
                if now - self._sessions[gate_id].created_at > max_age_sec:
                    self._sessions.pop(gate_id, None)
                    removed += 1
        return removed


manager = GateSessionManager()
