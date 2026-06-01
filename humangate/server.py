"""ComfyUI aiohttp routes for HumanGate."""
from __future__ import annotations

from .sessions import manager


ROUTES_REGISTERED = False
ROUTE_ERROR = ""


def _register_routes() -> None:
    global ROUTES_REGISTERED, ROUTE_ERROR
    try:
        from aiohttp import web  # type: ignore
        from server import PromptServer  # type: ignore
    except Exception as exc:  # ComfyUI is not present during standalone tests.
        ROUTE_ERROR = str(exc)
        return

    routes = PromptServer.instance.routes

    @routes.get("/humangate/sessions")
    async def list_sessions(request):
        return web.json_response({"sessions": manager.list_active()})

    @routes.get("/humangate/session/{gate_id}")
    async def get_session(request):
        gate_id = request.match_info["gate_id"]
        session = manager.get(gate_id)
        if session is None:
            return web.json_response({"error": "not found"}, status=404)
        return web.json_response(session.public_dict())

    @routes.post("/humangate/respond")
    async def respond(request):
        data = await request.json()
        gate_id = data.get("gate_id")
        result = data.get("result", {})
        if not gate_id:
            return web.json_response({"ok": False, "error": "missing gate_id"}, status=400)
        ok = manager.resolve(gate_id, result)
        return web.json_response({"ok": ok})

    @routes.post("/humangate/cancel")
    async def cancel(request):
        data = await request.json()
        gate_id = data.get("gate_id")
        if not gate_id:
            return web.json_response({"ok": False, "error": "missing gate_id"}, status=400)
        ok = manager.resolve(gate_id, {"decision": "stop"})
        return web.json_response({"ok": ok})

    @routes.post("/humangate/cleanup")
    async def cleanup(request):
        data = await request.json()
        max_age_sec = int(data.get("max_age_sec", 3600))
        removed = manager.cleanup(max_age_sec=max_age_sec)
        return web.json_response({"removed": removed})

    ROUTES_REGISTERED = True


_register_routes()
