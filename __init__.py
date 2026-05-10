"""
ComfyUI-naginagi-vibe-lab
==========================
naginagi が vibe noding する、ちょっぴり実験的で、かなりへんてこで、
ぽんこつな ComfyUI ノードコレクション。

A slightly experimental, quite weird, and lovingly ponkotsu
ComfyUI node collection by naginagi.

Ver. 1 ノード構成:
  ─────────────────────────────────────────────────────────────
   Group       | 技術アプローチ                          | 数
  ─────────────────────────────────────────────────────────────
   Corrector   | LLM テキスト → 埋め込み空間で射影減算    | 4
   Scheduler   | デノイズステップ / 反復ループでの変調    | 6
  ─────────────────────────────────────────────────────────────
                                              合計 10 ノード

CATEGORY 階層:
  conditioning/vibe-lab/Corrector
  conditioning/vibe-lab/Scheduler
"""

from __future__ import annotations

import logging

logger = logging.getLogger("naginagi-vibe-lab")

NODE_CLASS_MAPPINGS: dict = {}
NODE_DISPLAY_NAME_MAPPINGS: dict = {}


def _merge(module_name: str) -> None:
    """サブモジュールから安全にマッピングをマージする。

    1 個の依存欠落で全ノードが消えないようフェイルセーフ動作。
    """
    try:
        mod = __import__(
            f"{__name__}.{module_name}",
            fromlist=["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"],
        )
    except Exception as e:  # noqa: BLE001
        logger.error(
            "[naginagi-vibe-lab] Failed to import %s: %s: %s",
            module_name, type(e).__name__, e,
        )
        return

    cls_map = getattr(mod, "NODE_CLASS_MAPPINGS", None)
    name_map = getattr(mod, "NODE_DISPLAY_NAME_MAPPINGS", None)

    if isinstance(cls_map, dict):
        for k, v in cls_map.items():
            if k in NODE_CLASS_MAPPINGS:
                logger.warning(
                    "[naginagi-vibe-lab] Duplicate node name skipped: %s (from %s)",
                    k, module_name,
                )
                continue
            NODE_CLASS_MAPPINGS[k] = v

    if isinstance(name_map, dict):
        for k, v in name_map.items():
            NODE_DISPLAY_NAME_MAPPINGS.setdefault(k, v)

    logger.info(
        "[naginagi-vibe-lab] Loaded %s: %d nodes",
        module_name, len(cls_map) if isinstance(cls_map, dict) else 0,
    )


_merge("nodes_corrector")
_merge("nodes_scheduler")

logger.info(
    "[naginagi-vibe-lab] Total registered nodes: %d",
    len(NODE_CLASS_MAPPINGS),
)


WEB_DIRECTORY = "./web"

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "WEB_DIRECTORY",
]
