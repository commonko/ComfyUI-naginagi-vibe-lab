"""ComfyUI HumanGate nodes."""
from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional

from . import server  # noqa: F401 - importing registers routes in ComfyUI.
from .constants import CATEGORY, MAX_PREVIEW_IMAGES
from .exceptions import HumanGateUserStop
from .image_utils import image_batch_to_previews, normalize_indices, slice_images
from .sessions import manager


_LAST_SELECTIONS: Dict[str, List[int]] = {}


def _always_changed(*args, **kwargs):
    return time.time()


def _wait_for_session(session, timeout_sec: int) -> Dict[str, Any]:
    timeout = None if int(timeout_sec) <= 0 else int(timeout_sec)
    ok = session.event.wait(timeout=timeout)
    if not ok:
        return {"decision": "timeout"}
    return session.result or {"decision": "resume"}


def _stop_if_requested(decision: str) -> None:
    if decision == "stop":
        raise HumanGateUserStop()


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def _selection_key(prompt_id: Optional[str], node_id: Optional[str]) -> str:
    return f"{prompt_id or 'unknown_prompt'}:{node_id or 'unknown_node'}"


class HumanGatePauseImage:
    CATEGORY = CATEGORY
    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("image", "decision", "gate_id")
    FUNCTION = "run"
    IS_CHANGED = _always_changed

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "message": ("STRING", {"default": "Paused. Resume or Stop?", "multiline": True}),
                "timeout_sec": ("INT", {"default": 0, "min": 0, "max": 86400, "step": 1}),
                "alert": ("BOOLEAN", {"default": True}),
            },
            "hidden": {
                "prompt_id": "PROMPT_ID",
                "node_id": "UNIQUE_ID",
            },
        }

    def run(self, image, message: str, timeout_sec: int, alert: bool, prompt_id=None, node_id=None):
        session = manager.create(
            prompt_id=prompt_id,
            node_id=node_id,
            kind="pause_image",
            payload={
                "message": message,
                "alert": bool(alert),
                "preview_urls": image_batch_to_previews(image, max_images=1),
            },
        )
        result = _wait_for_session(session, timeout_sec)
        manager.pop(session.gate_id)
        decision = result.get("decision", "resume")
        _stop_if_requested(decision)
        return (image, decision, session.gate_id)


class HumanGateImageChooser:
    CATEGORY = CATEGORY
    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("images", "selected_indices", "selection_json")
    FUNCTION = "run"
    IS_CHANGED = _always_changed

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "message": ("STRING", {"default": "Select image(s), then Resume.", "multiline": True}),
                "selection_mode": (["single", "multiple"], {"default": "single"}),
                "pause_mode": (["always_pause", "pass_through", "take_first", "take_last", "repeat_last"], {"default": "always_pause"}),
                "timeout_sec": ("INT", {"default": 0, "min": 0, "max": 86400, "step": 1}),
            },
            "hidden": {
                "prompt_id": "PROMPT_ID",
                "node_id": "UNIQUE_ID",
            },
        }

    def run(self, images, message: str, selection_mode: str, pause_mode: str, timeout_sec: int, prompt_id=None, node_id=None):
        batch_size = int(images.shape[0])
        allow_multiple = selection_mode == "multiple"
        key = _selection_key(prompt_id, node_id)

        if pause_mode == "pass_through":
            indices = list(range(batch_size))
            meta = {"selected_indices": indices, "mode": pause_mode, "original_batch_size": batch_size}
            return (images, _json(indices), _json(meta))

        if pause_mode == "take_first":
            indices = [0]
            selected = slice_images(images, indices)
            meta = {"selected_indices": indices, "mode": pause_mode, "original_batch_size": batch_size}
            return (selected, _json(indices), _json(meta))

        if pause_mode == "take_last":
            indices = [batch_size - 1]
            selected = slice_images(images, indices)
            meta = {"selected_indices": indices, "mode": pause_mode, "original_batch_size": batch_size}
            return (selected, _json(indices), _json(meta))

        if pause_mode == "repeat_last" and key in _LAST_SELECTIONS:
            indices = normalize_indices(_LAST_SELECTIONS[key], batch_size, allow_multiple=allow_multiple)
            selected = slice_images(images, indices)
            meta = {"selected_indices": indices, "mode": pause_mode, "original_batch_size": batch_size}
            return (selected, _json(indices), _json(meta))

        session = manager.create(
            prompt_id=prompt_id,
            node_id=node_id,
            kind="image_chooser",
            payload={
                "message": message,
                "selection_mode": selection_mode,
                "batch_size": batch_size,
                "preview_urls": image_batch_to_previews(images, max_images=MAX_PREVIEW_IMAGES),
            },
        )
        result = _wait_for_session(session, timeout_sec)
        manager.pop(session.gate_id)

        decision = result.get("decision", "resume")
        _stop_if_requested(decision)
        if decision == "timeout":
            indices = [0]
        else:
            indices = normalize_indices(result.get("selected_indices", [0]), batch_size, allow_multiple=allow_multiple)
        _LAST_SELECTIONS[key] = indices

        selected = slice_images(images, indices)
        meta = {
            "gate_id": session.gate_id,
            "selected_indices": indices,
            "selection_mode": selection_mode,
            "decision": decision,
            "original_batch_size": batch_size,
        }
        return (selected, _json(indices), _json(meta))


class HumanGatePickImage:
    CATEGORY = CATEGORY
    RETURN_TYPES = ("IMAGE", "INT", "STRING", "STRING")
    RETURN_NAMES = ("image", "selected_index", "selected_label", "selection_json")
    FUNCTION = "run"
    IS_CHANGED = _always_changed

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_1": ("IMAGE",),
                "message": ("STRING", {"default": "Pick one input image.", "multiline": True}),
                "labels": ("STRING", {"default": "A,B,C,D", "multiline": False}),
                "pause_mode": (["always_pause", "take_first", "take_last", "repeat_last"], {"default": "always_pause"}),
                "timeout_sec": ("INT", {"default": 0, "min": 0, "max": 86400, "step": 1}),
            },
            "optional": {
                "image_2": ("IMAGE",),
                "image_3": ("IMAGE",),
                "image_4": ("IMAGE",),
            },
            "hidden": {
                "prompt_id": "PROMPT_ID",
                "node_id": "UNIQUE_ID",
            },
        }

    def run(self, image_1, message: str, labels: str, pause_mode: str, timeout_sec: int, image_2=None, image_3=None, image_4=None, prompt_id=None, node_id=None):
        images = [img for img in [image_1, image_2, image_3, image_4] if img is not None]
        label_list = [x.strip() for x in labels.split(",") if x.strip()] or ["A", "B", "C", "D"]
        key = _selection_key(prompt_id, node_id)

        if pause_mode == "take_first":
            selected_index = 0
        elif pause_mode == "take_last":
            selected_index = len(images) - 1
        elif pause_mode == "repeat_last" and key in _LAST_SELECTIONS:
            selected_index = normalize_indices(_LAST_SELECTIONS[key], len(images), allow_multiple=False)[0]
        else:
            previews = []
            for img in images:
                previews.extend(image_batch_to_previews(img, max_images=1))
            session = manager.create(
                prompt_id=prompt_id,
                node_id=node_id,
                kind="pick_image",
                payload={
                    "message": message,
                    "selection_mode": "single",
                    "batch_size": len(images),
                    "labels": label_list[:len(images)],
                    "preview_urls": previews,
                },
            )
            result = _wait_for_session(session, timeout_sec)
            manager.pop(session.gate_id)
            decision = result.get("decision", "resume")
            _stop_if_requested(decision)
            selected_index = 0 if decision == "timeout" else normalize_indices(result.get("selected_indices", [0]), len(images), allow_multiple=False)[0]

        _LAST_SELECTIONS[key] = [selected_index]
        label = label_list[selected_index] if selected_index < len(label_list) else str(selected_index)
        meta = {"selected_index": selected_index, "selected_label": label, "input_count": len(images)}
        return (images[selected_index], int(selected_index), label, _json(meta))


class HumanGatePickText:
    CATEGORY = CATEGORY
    RETURN_TYPES = ("STRING", "INT", "STRING", "STRING")
    RETURN_NAMES = ("text", "selected_index", "selected_label", "selection_json")
    FUNCTION = "run"
    IS_CHANGED = _always_changed

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text_1": ("STRING", {"default": "", "multiline": True}),
                "text_2": ("STRING", {"default": "", "multiline": True}),
                "text_3": ("STRING", {"default": "", "multiline": True}),
                "text_4": ("STRING", {"default": "", "multiline": True}),
                "message": ("STRING", {"default": "Pick one text input.", "multiline": True}),
                "labels": ("STRING", {"default": "A,B,C,D", "multiline": False}),
                "pause_mode": (["always_pause", "take_first", "take_last", "repeat_last"], {"default": "always_pause"}),
                "timeout_sec": ("INT", {"default": 0, "min": 0, "max": 86400, "step": 1}),
            },
            "hidden": {
                "prompt_id": "PROMPT_ID",
                "node_id": "UNIQUE_ID",
            },
        }

    def run(self, text_1, text_2, text_3, text_4, message, labels, pause_mode, timeout_sec, prompt_id=None, node_id=None):
        texts = [text_1, text_2, text_3, text_4]
        label_list = [x.strip() for x in labels.split(",") if x.strip()] or ["A", "B", "C", "D"]
        key = _selection_key(prompt_id, node_id)

        if pause_mode == "take_first":
            selected_index = 0
        elif pause_mode == "take_last":
            selected_index = len(texts) - 1
        elif pause_mode == "repeat_last" and key in _LAST_SELECTIONS:
            selected_index = normalize_indices(_LAST_SELECTIONS[key], len(texts), allow_multiple=False)[0]
        else:
            session = manager.create(
                prompt_id=prompt_id,
                node_id=node_id,
                kind="pick_text",
                payload={
                    "message": message,
                    "selection_mode": "single",
                    "batch_size": len(texts),
                    "labels": label_list,
                    "texts": texts,
                },
            )
            result = _wait_for_session(session, timeout_sec)
            manager.pop(session.gate_id)
            decision = result.get("decision", "resume")
            _stop_if_requested(decision)
            selected_index = 0 if decision == "timeout" else normalize_indices(result.get("selected_indices", [0]), len(texts), allow_multiple=False)[0]

        _LAST_SELECTIONS[key] = [selected_index]
        label = label_list[selected_index] if selected_index < len(label_list) else str(selected_index)
        meta = {"selected_index": selected_index, "selected_label": label, "input_count": len(texts)}
        return (texts[selected_index], int(selected_index), label, _json(meta))


class HumanGateCompareChooser(HumanGateImageChooser):
    @classmethod
    def INPUT_TYPES(cls):
        data = super().INPUT_TYPES()
        data["required"]["message"] = ("STRING", {"default": "Choose the better image(s).", "multiline": True})
        return data


NODE_CLASS_MAPPINGS = {
    "HumanGatePauseImage": HumanGatePauseImage,
    "HumanGateImageChooser": HumanGateImageChooser,
    "HumanGatePickImage": HumanGatePickImage,
    "HumanGatePickText": HumanGatePickText,
    "HumanGateCompareChooser": HumanGateCompareChooser,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "HumanGatePauseImage": "HumanGate: Pause Image",
    "HumanGateImageChooser": "HumanGate: Image Chooser",
    "HumanGatePickImage": "HumanGate: Pick Image",
    "HumanGatePickText": "HumanGate: Pick Text",
    "HumanGateCompareChooser": "HumanGate: Compare Chooser",
}
