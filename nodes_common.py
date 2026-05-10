"""
ComfyUI-naginagi-Cond-VibeEdit / nodes_common.py
=================================================
4 グループ (Corrector / TIPO / Scheduler / Genome) で共有する
ユーティリティ関数群。

このファイルは ComfyUI ノードを公開しません — 関数のみ。
グループ固有のヘルパー (RoPE 関連、タグ変換テーブル、特殊な splitter 等) は
それぞれの nodes_*.py 内に閉じ込める方針。
"""

from __future__ import annotations

import math
import re

import torch
from torch import Tensor


# =============================================================================
# Conditioning copy helpers
# =============================================================================

def clone_conditioning(conditioning: list) -> list:
    """
    Conditioning リスト ([(cond_tensor, meta_dict), ...]) のディープコピー。

    meta_dict 内の Tensor 値も clone される。
    Corrector / Scheduler / Genome すべてで使用。
    """
    return [
        (
            c.clone(),
            {k: v.clone() if isinstance(v, Tensor) else v for k, v in meta.items()},
        )
        for c, meta in conditioning
    ]


# =============================================================================
# Tensor shape utilities
# =============================================================================

def pool_or_truncate(tensor: Tensor, target_len: int) -> Tensor:
    """
    seq 次元 (dim=1) を target_len に揃える単純版。
      - 同じ        → そのまま
      - 長い (>tgt) → 先頭から target_len で truncate
      - 短い (<tgt) → ゼロパディング
    """
    if tensor.shape[1] == target_len:
        return tensor
    if tensor.shape[1] > target_len:
        return tensor[:, :target_len, :]
    padding = torch.zeros(
        tensor.shape[0], target_len - tensor.shape[1], tensor.shape[2],
        device=tensor.device, dtype=tensor.dtype,
    )
    return torch.cat([tensor, padding], dim=1)


def mean_pool_to_length(tensor: Tensor, target_len: int) -> Tensor:
    """
    seq 次元 (dim=1) を target_len に **mean pooling** で揃える。

    挙動:
      - 同じ            → そのまま
      - 短い (src<tgt) → pool_or_truncate (ゼロパディング)
      - 長い (src>tgt) → 区間 mean で縮小

    pool_or_truncate との違い: 長い側で「先頭だけ採用」ではなく
    「全区間を平均化して保存」する。情報損失が少ないので Corrector
    の long_text_strategy='mean_pool' 等で使う。
    """
    if tensor.dim() < 2:
        return tensor

    src_len = tensor.shape[1]
    if src_len == target_len:
        return tensor
    if src_len < target_len:
        return pool_or_truncate(tensor, target_len)

    result = torch.zeros(
        tensor.shape[0], target_len, tensor.shape[2],
        device=tensor.device, dtype=tensor.dtype,
    )
    for i in range(target_len):
        start = int(i * src_len / target_len)
        end = max(int((i + 1) * src_len / target_len), start + 1)
        result[:, i, :] = tensor[:, start:end, :].mean(dim=1)
    return result


# =============================================================================
# Text splitting
# =============================================================================

def split_sentences(text: str) -> list[str]:
    """
    テキストを文単位で分割。日本語(。！？) と英語(.!?) 両対応。

    Corrector の sentence_avg 戦略で各文を独立にエンコードする際に使用。
    分割不能なら全文を 1 文として返す。
    """
    lines = text.strip().split("\n")
    sentences = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        parts = re.split(r"(?<=[。！？.!?])\s*", line)
        for part in parts:
            part = part.strip()
            if len(part) > 2:
                sentences.append(part)
    return sentences if sentences else [text]


# =============================================================================
# Interpolation
# =============================================================================

def slerp(v0: Tensor, v1: Tensor, t: float) -> Tensor:
    """
    球面線形補間 (SLERP)。

    高次元埋め込み空間では LERP より方向の保存性が高く、
    Conditioning ブレンドで好まれる。t=0 → v0, t=1 → v1。
    """
    v0_flat = v0.reshape(-1).float()
    v1_flat = v1.reshape(-1).float()

    v0_norm = v0_flat / v0_flat.norm().clamp(min=1e-8)
    v1_norm = v1_flat / v1_flat.norm().clamp(min=1e-8)

    dot = (v0_norm * v1_norm).sum().clamp(-1.0, 1.0)
    omega = torch.acos(dot)

    if omega.abs() < 1e-6:
        result_flat = (1.0 - t) * v0_flat + t * v1_flat
    else:
        sin_omega = torch.sin(omega)
        result_flat = (
            (torch.sin((1.0 - t) * omega) / sin_omega) * v0_flat
            + (torch.sin(t * omega) / sin_omega) * v1_flat
        )

    return result_flat.reshape(v0.shape).to(v0.dtype)


# =============================================================================
# 後方互換エイリアス
# 旧コードでアンダースコア付きの名前を import している箇所からも
# 引き続きアクセスできるようにする。
# =============================================================================

_clone_conditioning = clone_conditioning
_mean_pool_to_length = mean_pool_to_length
_pool_or_truncate = pool_or_truncate
_split_sentences = split_sentences
_slerp = slerp


__all__ = [
    "clone_conditioning",
    "pool_or_truncate",
    "mean_pool_to_length",
    "split_sentences",
    "slerp",
    # underscore aliases (legacy)
    "_clone_conditioning",
    "_pool_or_truncate",
    "_mean_pool_to_length",
    "_split_sentences",
    "_slerp",
]
