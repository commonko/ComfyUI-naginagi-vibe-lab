"""Standalone tests for HumanGate."""
from __future__ import annotations

import importlib.util
import os
import sys


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPEC = importlib.util.spec_from_file_location(
    "humangate_testpkg",
    os.path.join(ROOT, "__init__.py"),
    submodule_search_locations=[ROOT],
)
pkg = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = pkg
assert SPEC.loader is not None
SPEC.loader.exec_module(pkg)

from humangate_testpkg.humangate.image_utils import normalize_indices  # noqa: E402
from humangate_testpkg.humangate.exceptions import HumanGateUserStop  # noqa: E402
from humangate_testpkg.humangate.nodes import NODE_CLASS_MAPPINGS, _stop_if_requested  # noqa: E402
from humangate_testpkg.humangate.sessions import GateSessionManager  # noqa: E402


def test_manager():
    manager = GateSessionManager()
    session = manager.create("prompt", "node", "pause", {"message": "x"})
    assert session.gate_id in [s["gate_id"] for s in manager.list_active()]
    assert manager.resolve(session.gate_id, {"decision": "resume"})
    assert session.event.is_set()
    assert manager.pop(session.gate_id) is session
    assert manager.pop(session.gate_id) is None
    print("[ok] sessions")


def test_indices():
    assert normalize_indices("1,3,99", 4) == [1, 3]
    assert normalize_indices([], 3) == [0]
    assert normalize_indices([2, 1], 3, allow_multiple=False) == [2]
    print("[ok] indices")


def test_nodes_import_and_text_pick():
    assert "HumanGatePauseImage" in NODE_CLASS_MAPPINGS
    assert "HumanGatePickText" in NODE_CLASS_MAPPINGS
    for cls in NODE_CLASS_MAPPINGS.values():
        cls.INPUT_TYPES()
        assert hasattr(cls, "RETURN_TYPES")
    node = NODE_CLASS_MAPPINGS["HumanGatePickText"]()
    text, idx, label, meta = node.run("a", "b", "c", "d", "pick", "A,B,C,D", "take_last", 0)
    assert text == "d"
    assert idx == 3
    assert label == "D"
    assert "selected_index" in meta
    print("[ok] nodes")


def test_intentional_stop_exception():
    try:
        _stop_if_requested("stop")
    except HumanGateUserStop as exc:
        assert "intentional v0.1 stop" in str(exc)
        assert "Error Report" in str(exc)
    else:
        raise AssertionError("stop did not raise HumanGateUserStop")
    _stop_if_requested("resume")
    print("[ok] intentional stop exception")


def main():
    test_manager()
    test_indices()
    test_nodes_import_and_text_pick()
    test_intentional_stop_exception()
    print("ALL TESTS PASSED")


if __name__ == "__main__":
    main()
