"""Minimal ComfyUI workflow JSON builder."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple


class WF:
    def __init__(self):
        self.nodes: List[Dict[str, Any]] = []
        self.links: List[List[Any]] = []
        self.groups: List[Dict[str, Any]] = []
        self._nid = 0
        self._lid = 0

    def add(self, type_: str, pos: Tuple[int, int], size: Tuple[int, int], *, title: str = "",
            widgets: Optional[List[Any]] = None, inputs: Optional[List[Dict[str, Any]]] = None,
            outputs: Optional[List[Dict[str, Any]]] = None, bgcolor: str = "") -> int:
        self._nid += 1
        node: Dict[str, Any] = {
            "id": self._nid,
            "type": type_,
            "pos": list(pos),
            "size": [size[0], size[1]],
            "flags": {},
            "order": self._nid - 1,
            "mode": 0,
            "inputs": inputs or [],
            "outputs": outputs or [],
            "properties": {"Node name for S&R": type_},
            "widgets_values": widgets or [],
        }
        if title:
            node["title"] = title
        if bgcolor:
            node["bgcolor"] = bgcolor
        self.nodes.append(node)
        return self._nid

    def link(self, src: int, src_slot: int, dst: int, dst_slot: int, type_: str) -> None:
        self._lid += 1
        lid = self._lid
        self.links.append([lid, src, src_slot, dst, dst_slot, type_])
        for node in self.nodes:
            if node["id"] == src and src_slot < len(node["outputs"]):
                node["outputs"][src_slot].setdefault("links", []).append(lid)
            if node["id"] == dst and dst_slot < len(node["inputs"]):
                node["inputs"][dst_slot]["link"] = lid

    def group(self, title: str, bounding: Tuple[int, int, int, int], color: str) -> None:
        self.groups.append({"title": title, "bounding": list(bounding), "color": color, "font_size": 24, "locked": False})

    def to_dict(self) -> Dict[str, Any]:
        return {
            "last_node_id": self._nid,
            "last_link_id": self._lid,
            "nodes": self.nodes,
            "links": self.links,
            "groups": self.groups,
            "config": {},
            "extra": {"ds": {"scale": 0.8, "offset": [0, 0]}},
            "version": 0.4,
        }


def IN(name: str, type_: str, widget: Optional[str] = None) -> Dict[str, Any]:
    data: Dict[str, Any] = {"name": name, "type": type_, "link": None}
    if widget:
        data["widget"] = {"name": widget}
    return data


def OUT(name: str, type_: str, slot: int) -> Dict[str, Any]:
    return {"name": name, "type": type_, "links": [], "slot_index": slot, "shape": 3}
