"""
ComfyUI-naginagi-Cond-VibeEdit / nodes_corrector.py
=====================================================
Corrector グループ: LLM が出力した「問題テキスト」を CLIP で埋め込み化し、
元の Conditioning から **問題方向ベクトルを射影減算** することで、
再学習や再ノイズ化なしに不具合 (解剖エラー、画風崩れ等) を抑制するノード群。

このグループは TIPO に依存しません。TIPO で前処理したい場合は、
nodes_tipo.py の TIPOCorrectionGenerator を直前に挟みます:

  TIPO なしパイプライン:
    LLM ─→ problem_text ─→ ConditioningCorrector ←─ CONDITIONING

  TIPO ありパイプライン:
    LLM ─→ TIPOCorrectionGenerator ─→ problem_tags
                                          ↓
                              ConditioningCorrector ←─ CONDITIONING

4 ノード構成:
  1. ConditioningProjection         — 素の射影 add/subtract/replace (低レベル)
  2. ConditioningCorrector          — LLM テキスト → CLIP encode → 射影減算 (中核)
  3. ConditioningCorrectorDual      — Pos/Neg 同時補正
  4. ConditioningCorrectorInpaint   — マスク領域だけ補正 (インペイント風)
"""

from __future__ import annotations

import math
import re

import torch
from torch import Tensor

from .nodes_common import (
    clone_conditioning as _clone_conditioning,
    mean_pool_to_length as _mean_pool_to_length,
    pool_or_truncate as _pool_or_truncate,
    split_sentences as _split_sentences,
)


# =============================================================================
# ConditioningProjection
# =============================================================================

class ConditioningProjection:
    """source の方向成分を target に射影し、加算または減算する。"""

    METHODS = ["add", "subtract", "replace"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "target": ("CONDITIONING",),
                "source": ("CONDITIONING",),
                "method": (cls.METHODS, {
                    "tooltip": "add: source 方向を加算, subtract: 減算, replace: 置換"
                }),
                "strength": ("FLOAT", {
                    "default": 0.5, "min": 0.0, "max": 2.0, "step": 0.01,
                }),
                "apply_to_pooled": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "project"
    CATEGORY = "conditioning/vibe-lab/Corrector"
    DESCRIPTION = "source の方向成分を target に射影で加減算します。"

    def project(self, target, source, method, strength, apply_to_pooled):
        out = []
        count = min(len(target), len(source))

        for i in range(count):
            t_cond, t_meta = target[i]
            s_cond, s_meta = source[i]
            result = self._apply_projection(t_cond, s_cond, method, strength)

            new_meta = t_meta.copy()
            if apply_to_pooled:
                tp = t_meta.get("pooled_output")
                sp = s_meta.get("pooled_output")
                if tp is not None and sp is not None:
                    new_meta["pooled_output"] = self._apply_projection(tp, sp, method, strength)

            out.append((result, new_meta))
        return (out,)

    @staticmethod
    def _apply_projection(target: Tensor, source: Tensor, method: str, strength: float) -> Tensor:
        t = target.clone().float()
        s = source.float()
        if t.dim() >= 2 and s.dim() >= 2 and t.shape[1] != s.shape[1]:
            s = _pool_or_truncate(s, t.shape[1])

        direction = s / s.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        dot = (t * direction).sum(dim=-1, keepdim=True)
        projection = dot * direction

        if method == "add":
            result = t + strength * projection
        elif method == "subtract":
            result = t - strength * projection
        elif method == "replace":
            orthogonal = t - projection
            result = orthogonal + strength * projection
        else:
            result = t
        return result.to(target.dtype)

# =============================================================================
# ConditioningCorrector
# =============================================================================

class ConditioningCorrector:
    """
    LLM が分析した「問題テキスト」を CLIP で埋め込み化し、
    元の Conditioning から問題方向を減算して修正する統合ノード。

    long_text_strategy:
      - "sentence_avg": 文ごとに個別エンコード→射影を平均（推奨、最も精度が高い）
      - "mean_pool":    全チャンクを mean-pool で圧縮（高速、情報保持◎）
      - "per_chunk":    チャンクごとに個別射影→合成（方向保持◎）
      - "truncate":     先頭チャンクのみ使用（従来互換）

    TextGenerate ノード（Qwen 3.5 等）の出力を problem_text に接続する想定。
    """

    STRATEGIES = ["sentence_avg", "mean_pool", "per_chunk", "truncate"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning": ("CONDITIONING",),
                "clip": ("CLIP",),
                "problem_text": ("STRING", {
                    "forceInput": True,
                    "tooltip": "LLM が出力した画像の問題記述テキスト。"
                             "TextGenerate ノードの出力を接続。"
                }),
                "subtract_strength": ("FLOAT", {
                    "default": 0.3, "min": 0.0, "max": 2.0, "step": 0.01,
                    "tooltip": "問題方向を減算する強度。大きいほど強く除去。"
                }),
                "long_text_strategy": (cls.STRATEGIES, {
                    "default": "sentence_avg",
                    "tooltip": "長文テキストの処理戦略。"
                             "sentence_avg が最も精度が高い（推奨）。"
                }),
            },
            "optional": {
                "correction_text": ("STRING", {
                    "forceInput": True,
                    "tooltip": "理想状態の記述テキスト（任意）。"
                             "指定すると、この方向を加算して修正を補強。"
                }),
                "add_strength": ("FLOAT", {
                    "default": 0.2, "min": 0.0, "max": 2.0, "step": 0.01,
                    "tooltip": "修正方向を加算する強度。"
                }),
                "normalize_result": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "修正後に L2 正規化を適用するか。"
                }),
                "apply_to_pooled": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "pooled_output にも修正を適用するか (SDXL)。"
                }),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "STRING",)
    RETURN_NAMES = ("conditioning", "analysis_log",)
    FUNCTION = "correct"
    CATEGORY = "conditioning/vibe-lab/Corrector"
    DESCRIPTION = (
        "LLM の画像分析テキストを CLIP で埋め込み、"
        "元の Conditioning から問題方向を減算して修正します。"
        "TextGenerate (Qwen 3.5) の出力を problem_text に接続してください。"
    )

    def correct(self, conditioning, clip, problem_text,
                subtract_strength, long_text_strategy,
                correction_text="", add_strength=0.2,
                normalize_result=False, apply_to_pooled=True):

        log_lines = []
        log_lines.append("=== Conditioning Corrector ===")
        log_lines.append(f"Strategy: {long_text_strategy}")
        log_lines.append(f"Problem text: {len(problem_text)} chars")

        # ターゲットの seq_len を取得
        target_seq_len = conditioning[0][0].shape[1] if conditioning[0][0].dim() >= 2 else 77
        log_lines.append(f"Target seq_len: {target_seq_len}")

        # ─── 1. 戦略に基づいて問題テキストをエンコード ───
        problem_cond_chunks = None  # per_chunk 用
        problem_cond = None

        if long_text_strategy == "sentence_avg":
            problem_cond, problem_pooled, enc_log = self._encode_sentence_avg(
                clip, problem_text, target_seq_len
            )
        elif long_text_strategy == "mean_pool":
            problem_cond, problem_pooled, enc_log = self._encode_mean_pool(
                clip, problem_text, target_seq_len
            )
        elif long_text_strategy == "per_chunk":
            problem_cond_chunks, problem_pooled, enc_log = self._encode_chunked(
                clip, problem_text, target_seq_len
            )
        else:  # truncate
            problem_cond, problem_pooled, enc_log = self._encode_truncate(
                clip, problem_text, target_seq_len
            )

        log_lines.extend(enc_log)

        # ─── 2. correction_text のエンコード ───
        correction_cond = None
        correction_pooled = None
        if correction_text and correction_text.strip():
            correction_tokens = clip.tokenize(correction_text)
            correction_output = clip.encode_from_tokens(
                correction_tokens, return_pooled=True, return_dict=True
            )
            correction_cond = correction_output.pop("cond")
            correction_pooled = correction_output.get("pooled_output")
            log_lines.append(f"Correction text: {len(correction_text)} chars, "
                             f"cond shape: {correction_cond.shape}")

        # ─── 3. Conditioning を修正 ───
        # embed 次元チェック
        target_embed = conditioning[0][0].shape[-1] if conditioning[0][0].dim() >= 2 else 0
        if problem_cond is not None and problem_cond.shape[-1] != target_embed:
            log_lines.append(
                f"⚠ EMBED DIM MISMATCH: target={target_embed}, problem={problem_cond.shape[-1]}. "
                f"Projection will be skipped. This model may need a different encoding approach."
            )

        out = []
        for idx, (cond_tensor, meta) in enumerate(conditioning):
            c = cond_tensor.clone().float()

            # problem 方向の減算
            if long_text_strategy == "per_chunk" and problem_cond_chunks is not None:
                c = self._subtract_per_chunk(c, problem_cond_chunks, subtract_strength)
            elif problem_cond is not None:
                c = self._subtract_direction(c, problem_cond.float(), subtract_strength)

            # correction 方向の加算
            if correction_cond is not None:
                c = self._add_direction(c, correction_cond.float(), add_strength)

            if normalize_result:
                c = c / c.norm(dim=-1, keepdim=True).clamp(min=1e-8)

            c = c.to(cond_tensor.dtype)

            # pooled_output の修正
            new_meta = meta.copy()
            if apply_to_pooled and "pooled_output" in new_meta:
                p = new_meta["pooled_output"]
                if p is not None:
                    p = p.clone().float()
                    if problem_pooled is not None:
                        p = self._subtract_direction(p, problem_pooled.float(), subtract_strength)
                    if correction_pooled is not None:
                        p = self._add_direction(p, correction_pooled.float(), add_strength)
                    if normalize_result:
                        p = p / p.norm(dim=-1, keepdim=True).clamp(min=1e-8)
                    new_meta["pooled_output"] = p.to(meta["pooled_output"].dtype)

            out.append((c, new_meta))

            original_norm = cond_tensor.norm().item()
            modified_norm = c.norm().item()
            log_lines.append(
                f"Cond[{idx}]: norm {original_norm:.4f} → {modified_norm:.4f} "
                f"(delta: {modified_norm - original_norm:+.4f})"
            )

        analysis_log = "\n".join(log_lines)
        return (out, analysis_log)

    # ─── エンコード戦略 ─────────────────────────────────

    def _encode_sentence_avg(self, clip, text: str, target_seq_len: int):
        """
        戦略A: 文ごとにエンコードし、各文の conditioning を平均。

        最も精度が高い。各文が独立にCLIPトークン上限内に収まるため、
        長い LLM 出力でも全文の意味が保持される。
        """
        sentences = _split_sentences(text)
        log = [f"--- sentence_avg ---",
               f"Split into {len(sentences)} sentences"]

        cond_list = []
        pooled_list = []

        for i, sent in enumerate(sentences):
            tokens = clip.tokenize(sent)
            output = clip.encode_from_tokens(tokens, return_pooled=True, return_dict=True)
            cond = output.pop("cond")
            pooled = output.get("pooled_output")

            # target_seq_len に合わせる
            if cond.dim() >= 2 and cond.shape[1] != target_seq_len:
                cond = _mean_pool_to_length(cond, target_seq_len)

            cond_list.append(cond)
            if pooled is not None:
                pooled_list.append(pooled)

            if i < 8:
                log.append(f"  [{i}] \"{sent[:80]}{'...' if len(sent)>80 else ''}\" "
                           f"norm={cond.norm().item():.3f}")
            elif i == 8:
                log.append(f"  ... +{len(sentences) - 8} more")

        # 全文の cond を平均
        avg_cond = torch.stack(cond_list, dim=0).mean(dim=0)
        avg_pooled = torch.stack(pooled_list, dim=0).mean(dim=0) if pooled_list else None

        log.append(f"Averaged cond shape: {avg_cond.shape}, norm: {avg_cond.norm().item():.4f}")
        return avg_cond, avg_pooled, log

    def _encode_mean_pool(self, clip, text: str, target_seq_len: int):
        """
        戦略B: テキスト全体をエンコードし、全チャンクを mean-pool で圧縮。
        """
        tokens = clip.tokenize(text)
        output = clip.encode_from_tokens(tokens, return_pooled=True, return_dict=True)
        cond = output.pop("cond")
        pooled = output.get("pooled_output")

        original_seq_len = cond.shape[1] if cond.dim() >= 2 else 0
        log = [f"--- mean_pool ---",
               f"Full encode: {original_seq_len} tokens ({original_seq_len // max(target_seq_len,1)} chunks)"]

        if cond.dim() >= 2 and cond.shape[1] > target_seq_len:
            cond = _mean_pool_to_length(cond, target_seq_len)
            log.append(f"Pooled: {original_seq_len} → {target_seq_len} tokens (all info retained)")
        elif cond.dim() >= 2 and cond.shape[1] < target_seq_len:
            cond = _pool_or_truncate(cond, target_seq_len)
            log.append(f"Padded: {original_seq_len} → {target_seq_len} tokens")

        log.append(f"Result shape: {cond.shape}, norm: {cond.norm().item():.4f}")
        return cond, pooled, log

    def _encode_chunked(self, clip, text: str, target_seq_len: int):
        """
        戦略D: テキスト全体をエンコードし、チャンクリストで返す。
        """
        tokens = clip.tokenize(text)
        output = clip.encode_from_tokens(tokens, return_pooled=True, return_dict=True)
        full_cond = output.pop("cond")
        pooled = output.get("pooled_output")

        log = [f"--- per_chunk ---",
               f"Full encode: {full_cond.shape}"]

        chunks = []
        if full_cond.dim() >= 2:
            total_tokens = full_cond.shape[1]
            for start in range(0, total_tokens, target_seq_len):
                end = min(start + target_seq_len, total_tokens)
                chunk = full_cond[:, start:end, :]
                if chunk.shape[1] < target_seq_len:
                    chunk = _pool_or_truncate(chunk, target_seq_len)
                chunks.append(chunk)
                log.append(f"  Chunk {len(chunks)-1}: tokens [{start}:{end}], "
                           f"norm={chunk.norm().item():.3f}")
        else:
            chunks.append(full_cond)

        log.append(f"Total chunks: {len(chunks)}, each {target_seq_len} tokens")
        return chunks, pooled, log

    def _encode_truncate(self, clip, text: str, target_seq_len: int):
        """戦略C (legacy): 単純エンコード + 切詰。"""
        tokens = clip.tokenize(text)
        output = clip.encode_from_tokens(tokens, return_pooled=True, return_dict=True)
        cond = output.pop("cond")
        pooled = output.get("pooled_output")

        original_len = cond.shape[1] if cond.dim() >= 2 else 0
        discarded = max(0, original_len - target_seq_len)

        if cond.dim() >= 2 and cond.shape[1] != target_seq_len:
            cond = _pool_or_truncate(cond, target_seq_len)

        log = [f"--- truncate ---",
               f"Encode: {original_len} tokens → truncated to {target_seq_len}"]
        if discarded > 0:
            log.append(f"⚠ WARNING: {discarded} tokens ({discarded*100//original_len}%) DISCARDED")
        log.append(f"Result norm: {cond.norm().item():.4f}")
        return cond, pooled, log

    # ─── 射影操作 ─────────────────────────────────────

    @staticmethod
    def _check_embed_dim(target: Tensor, source: Tensor) -> bool:
        """embed 次元が一致するか確認。不一致の場合 False を返す。"""
        return target.shape[-1] == source.shape[-1]

    @staticmethod
    def _subtract_direction(target: Tensor, direction_source: Tensor, strength: float) -> Tensor:
        """target から direction_source 方向の成分を strength 分だけ減算。"""
        ds = direction_source
        if target.dim() >= 2 and ds.dim() >= 2 and target.shape[1] != ds.shape[1]:
            ds = _mean_pool_to_length(ds, target.shape[1])

        # embed 次元の不一致チェック: 異なる場合は修正をスキップ
        if target.shape[-1] != ds.shape[-1]:
            return target  # 安全にスキップ

        direction = ds / ds.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        dot = (target * direction).sum(dim=-1, keepdim=True)
        projection = dot * direction
        return target - strength * projection

    @staticmethod
    def _add_direction(target: Tensor, direction_source: Tensor, strength: float) -> Tensor:
        """target に direction_source 方向の成分を strength 分だけ加算。"""
        ds = direction_source
        if target.dim() >= 2 and ds.dim() >= 2 and target.shape[1] != ds.shape[1]:
            ds = _mean_pool_to_length(ds, target.shape[1])

        # embed 次元の不一致チェック
        if target.shape[-1] != ds.shape[-1]:
            return target

        direction = ds / ds.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        return target + strength * direction * ds.norm(dim=-1, keepdim=True)

    @staticmethod
    def _subtract_per_chunk(target: Tensor, chunks: list[Tensor], strength: float) -> Tensor:
        """
        各チャンクについて個別に射影を計算し、累積で減算する。
        チャンクごとの強度は 1/N に正規化。
        """
        n = len(chunks)
        if n == 0:
            return target

        per_chunk_strength = strength / n
        result = target.clone()

        for chunk in chunks:
            ds = chunk
            if result.dim() >= 2 and ds.dim() >= 2 and result.shape[1] != ds.shape[1]:
                ds = _mean_pool_to_length(ds, result.shape[1])

            # embed 次元の不一致チェック
            if result.shape[-1] != ds.shape[-1]:
                continue  # このチャンクをスキップ

            direction = ds / ds.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            dot = (result * direction).sum(dim=-1, keepdim=True)
            projection = dot * direction
            result = result - per_chunk_strength * projection

        return result

# =============================================================================
# ConditioningCorrectorDual
# =============================================================================

class ConditioningCorrectorDual:
    """
    LLM 分析テキストで Positive と Negative の Conditioning を同時修正する。

    ConditioningCorrector の拡張版。修正の方向が Positive と Negative で逆になる:
      - Positive: 問題方向を減算（問題の影響を除去）
      - Negative: 問題方向を加算（問題の回避を強化）

    SDXL → Anima Hires.fix ワークフローでの使用を想定:
      1. SDXL のプロンプトテキストを Anima の CLIP で再エンコード
      2. 本ノードで LLM 分析に基づき修正
      3. 修正済み Positive/Negative を Anima KSampler に渡す

    TextGenerate ノード（Qwen 3.5 等）の出力を problem_text に接続。
    """

    STRATEGIES = ConditioningCorrector.STRATEGIES

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "clip": ("CLIP",),
                "problem_text": ("STRING", {
                    "forceInput": True,
                    "tooltip": "LLM が出力した画像の問題記述テキスト。"
                }),
                "positive_subtract_strength": ("FLOAT", {
                    "default": 0.3, "min": 0.0, "max": 2.0, "step": 0.01,
                    "tooltip": "Positive から問題方向を減算する強度。"
                }),
                "negative_add_strength": ("FLOAT", {
                    "default": 0.3, "min": 0.0, "max": 2.0, "step": 0.01,
                    "tooltip": "Negative に問題方向を加算する強度。"
                }),
                "long_text_strategy": (cls.STRATEGIES, {
                    "default": "sentence_avg",
                    "tooltip": "長文処理戦略。sentence_avg 推奨。"
                }),
            },
            "optional": {
                "correction_text": ("STRING", {
                    "forceInput": True,
                    "tooltip": "理想状態の記述（任意）。Positive に加算される。"
                }),
                "correction_add_strength": ("FLOAT", {
                    "default": 0.2, "min": 0.0, "max": 2.0, "step": 0.01,
                    "tooltip": "修正方向を Positive に加算する強度。"
                }),
                "normalize_result": ("BOOLEAN", {
                    "default": False,
                }),
                "apply_to_pooled": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "pooled_output にも修正を適用するか (SDXL)。"
                }),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "STRING",)
    RETURN_NAMES = ("positive", "negative", "analysis_log",)
    FUNCTION = "correct_dual"
    CATEGORY = "conditioning/vibe-lab/Corrector"
    DESCRIPTION = (
        "Positive と Negative の Conditioning を同時修正します。"
        "Positive からは問題方向を減算、Negative には問題方向を加算。"
        "SDXL → Anima の Hires.fix パイプラインに最適。"
    )

    def correct_dual(self, positive, negative, clip, problem_text,
                     positive_subtract_strength, negative_add_strength,
                     long_text_strategy,
                     correction_text="", correction_add_strength=0.2,
                     normalize_result=False, apply_to_pooled=True):

        # ConditioningCorrector のインスタンスを活用
        corrector = ConditioningCorrector()

        log_lines = []
        log_lines.append("=== Conditioning Corrector Dual ===")
        log_lines.append(f"Strategy: {long_text_strategy}")
        log_lines.append(f"Problem text: {len(problem_text)} chars")
        log_lines.append(f"Positive subtract: {positive_subtract_strength}")
        log_lines.append(f"Negative add: {negative_add_strength}")

        # ─── ターゲットの seq_len を取得 ───
        pos_seq_len = positive[0][0].shape[1] if positive[0][0].dim() >= 2 else 77

        # ─── 問題テキストをエンコード ───
        if long_text_strategy == "sentence_avg":
            problem_cond, problem_pooled, enc_log = corrector._encode_sentence_avg(
                clip, problem_text, pos_seq_len)
        elif long_text_strategy == "mean_pool":
            problem_cond, problem_pooled, enc_log = corrector._encode_mean_pool(
                clip, problem_text, pos_seq_len)
        elif long_text_strategy == "per_chunk":
            problem_cond_chunks, problem_pooled, enc_log = corrector._encode_chunked(
                clip, problem_text, pos_seq_len)
            problem_cond = None
        else:
            problem_cond, problem_pooled, enc_log = corrector._encode_truncate(
                clip, problem_text, pos_seq_len)

        log_lines.extend(enc_log)

        # ─── correction_text のエンコード ───
        correction_cond = None
        correction_pooled = None
        if correction_text and correction_text.strip():
            correction_tokens = clip.tokenize(correction_text)
            correction_output = clip.encode_from_tokens(
                correction_tokens, return_pooled=True, return_dict=True)
            correction_cond = correction_output.pop("cond")
            correction_pooled = correction_output.get("pooled_output")
            log_lines.append(f"Correction text: {len(correction_text)} chars")

        # ─── Positive を修正: 問題方向を減算 ───
        log_lines.append("--- Positive (subtract problem) ---")
        pos_out = []
        for idx, (cond_tensor, meta) in enumerate(positive):
            c = cond_tensor.clone().float()

            if long_text_strategy == "per_chunk" and problem_cond is None:
                c = corrector._subtract_per_chunk(c, problem_cond_chunks, positive_subtract_strength)
            elif problem_cond is not None:
                c = corrector._subtract_direction(c, problem_cond.float(), positive_subtract_strength)

            # correction 方向を加算（Positive のみ）
            if correction_cond is not None:
                c = corrector._add_direction(c, correction_cond.float(), correction_add_strength)

            if normalize_result:
                c = c / c.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            c = c.to(cond_tensor.dtype)

            new_meta = meta.copy()
            if apply_to_pooled and "pooled_output" in new_meta and new_meta["pooled_output"] is not None:
                p = new_meta["pooled_output"].clone().float()
                if problem_pooled is not None:
                    p = corrector._subtract_direction(p, problem_pooled.float(), positive_subtract_strength)
                if correction_pooled is not None:
                    p = corrector._add_direction(p, correction_pooled.float(), correction_add_strength)
                if normalize_result:
                    p = p / p.norm(dim=-1, keepdim=True).clamp(min=1e-8)
                new_meta["pooled_output"] = p.to(meta["pooled_output"].dtype)

            pos_out.append((c, new_meta))

            orig = cond_tensor.norm().item()
            mod = c.norm().item()
            log_lines.append(f"  Pos[{idx}]: {orig:.4f} → {mod:.4f} ({mod - orig:+.4f})")

        # ─── Negative を修正: 問題方向を加算 ───
        log_lines.append("--- Negative (add problem) ---")
        neg_out = []
        for idx, (cond_tensor, meta) in enumerate(negative):
            c = cond_tensor.clone().float()

            # Negative には問題方向を加算（Positive とは逆）
            if long_text_strategy == "per_chunk" and problem_cond is None:
                # per_chunk の場合、各チャンクを加算
                n_chunks = len(problem_cond_chunks)
                per_strength = negative_add_strength / n_chunks if n_chunks > 0 else 0
                for chunk in problem_cond_chunks:
                    c = corrector._add_direction(c, chunk.float(), per_strength)
            elif problem_cond is not None:
                c = corrector._add_direction(c, problem_cond.float(), negative_add_strength)

            if normalize_result:
                c = c / c.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            c = c.to(cond_tensor.dtype)

            new_meta = meta.copy()
            if apply_to_pooled and "pooled_output" in new_meta and new_meta["pooled_output"] is not None:
                p = new_meta["pooled_output"].clone().float()
                if problem_pooled is not None:
                    p = corrector._add_direction(p, problem_pooled.float(), negative_add_strength)
                if normalize_result:
                    p = p / p.norm(dim=-1, keepdim=True).clamp(min=1e-8)
                new_meta["pooled_output"] = p.to(meta["pooled_output"].dtype)

            neg_out.append((c, new_meta))

            orig = cond_tensor.norm().item()
            mod = c.norm().item()
            log_lines.append(f"  Neg[{idx}]: {orig:.4f} → {mod:.4f} ({mod - orig:+.4f})")

        analysis_log = "\n".join(log_lines)
        return (pos_out, neg_out, analysis_log)

# =============================================================================
# ConditioningCorrectorInpaint
# =============================================================================

class ConditioningCorrectorInpaint:
    """
    Inpainting 用の Conditioning 修正ノード。

    マスク領域のみに修正済み Conditioning を適用し、
    マスク外は元の Conditioning をそのまま保持する。

    内部処理:
      1. LLM 分析テキストで Conditioning を修正（DualCorrector と同じロジック）
      2. 修正済み Conditioning にマスクを適用（ConditioningSetMask 相当）
      3. 元の Conditioning と修正済み Conditioning を結合
         → マスク内 = 修正版、マスク外 = オリジナル

    出力は KSampler にそのまま接続可能。
    SetLatentNoiseMask で latent にもマスクを適用すること。

    ワークフロー:
      LoadImage (マスク付き)
        ├→ VAEEncodeForInpaint → SetLatentNoiseMask → KSampler latent
        └→ image → TextGenerate (マスク領域の分析)
      CLIPTextEncode (pos/neg)
        └→ ConditioningCorrectorInpaint → KSampler positive/negative
    """

    STRATEGIES = ConditioningCorrector.STRATEGIES

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "clip": ("CLIP",),
                "mask": ("MASK",),
                "problem_text": ("STRING", {
                    "forceInput": True,
                    "tooltip": "LLM が出力したマスク領域の問題記述。"
                }),
                "subtract_strength": ("FLOAT", {
                    "default": 0.4, "min": 0.0, "max": 2.0, "step": 0.01,
                    "tooltip": "Positive から問題方向を減算する強度。"
                }),
                "negative_add_strength": ("FLOAT", {
                    "default": 0.4, "min": 0.0, "max": 2.0, "step": 0.01,
                    "tooltip": "Negative に問題方向を加算する強度。"
                }),
                "mask_strength": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "マスク領域での修正適用強度。"
                }),
                "long_text_strategy": (cls.STRATEGIES, {
                    "default": "sentence_avg",
                }),
                "set_area": (["default", "bounds"],{
                    "default": "bounds",
                    "tooltip": "bounds: マスクのバウンディングボックスに限定（推奨）。"
                             "default: 画像全体に対してマスク重みを適用。"
                }),
            },
            "optional": {
                "correction_text": ("STRING", {
                    "forceInput": True,
                    "tooltip": "マスク領域の理想状態の記述（任意）。"
                }),
                "correction_strength": ("FLOAT", {
                    "default": 0.3, "min": 0.0, "max": 2.0, "step": 0.01,
                }),
                "normalize_result": ("BOOLEAN", {"default": False}),
                "apply_to_pooled": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "STRING",)
    RETURN_NAMES = ("positive", "negative", "analysis_log",)
    FUNCTION = "correct_inpaint"
    CATEGORY = "conditioning/vibe-lab/Corrector"
    DESCRIPTION = (
        "Inpainting 用 Conditioning 修正。マスク領域のみに修正を適用し、"
        "マスク外は元の Conditioning を保持します。"
        "SetLatentNoiseMask と併用してください。"
    )

    def correct_inpaint(self, positive, negative, clip, mask, problem_text,
                        subtract_strength, negative_add_strength,
                        mask_strength, long_text_strategy, set_area,
                        correction_text="", correction_strength=0.3,
                        normalize_result=False, apply_to_pooled=True):

        corrector = ConditioningCorrector()
        log_lines = []
        log_lines.append("=== Conditioning Corrector Inpaint ===")
        log_lines.append(f"Strategy: {long_text_strategy}")
        log_lines.append(f"Mask shape: {mask.shape}")
        log_lines.append(f"Mask coverage: {mask.mean().item()*100:.1f}%")
        log_lines.append(f"Set area: {set_area}")
        log_lines.append(f"Subtract strength: {subtract_strength}")
        log_lines.append(f"Negative add: {negative_add_strength}")
        log_lines.append(f"Mask strength: {mask_strength}")

        target_seq_len = positive[0][0].shape[1] if positive[0][0].dim() >= 2 else 77

        # ─── 問題テキストをエンコード ───
        problem_cond_chunks = None
        problem_cond = None

        if long_text_strategy == "sentence_avg":
            problem_cond, problem_pooled, enc_log = corrector._encode_sentence_avg(
                clip, problem_text, target_seq_len)
        elif long_text_strategy == "mean_pool":
            problem_cond, problem_pooled, enc_log = corrector._encode_mean_pool(
                clip, problem_text, target_seq_len)
        elif long_text_strategy == "per_chunk":
            problem_cond_chunks, problem_pooled, enc_log = corrector._encode_chunked(
                clip, problem_text, target_seq_len)
        else:
            problem_cond, problem_pooled, enc_log = corrector._encode_truncate(
                clip, problem_text, target_seq_len)

        log_lines.extend(enc_log)

        # ─── correction_text ───
        correction_cond = None
        correction_pooled = None
        if correction_text and correction_text.strip():
            correction_tokens = clip.tokenize(correction_text)
            correction_output = clip.encode_from_tokens(
                correction_tokens, return_pooled=True, return_dict=True)
            correction_cond = correction_output.pop("cond")
            correction_pooled = correction_output.get("pooled_output")
            log_lines.append(f"Correction text: {len(correction_text)} chars")

        set_area_to_bounds = (set_area == "bounds")

        # ─── Positive 修正: マスク領域のみ ───
        log_lines.append("--- Positive (masked correction) ---")
        pos_out = []

        for idx, (cond_tensor, meta) in enumerate(positive):
            # 元の Conditioning はそのまま保持（マスク外に適用）
            pos_out.append((cond_tensor, meta.copy()))

            # 修正版を作成
            c = cond_tensor.clone().float()

            if long_text_strategy == "per_chunk" and problem_cond is None:
                c = corrector._subtract_per_chunk(c, problem_cond_chunks, subtract_strength)
            elif problem_cond is not None:
                c = corrector._subtract_direction(c, problem_cond.float(), subtract_strength)

            if correction_cond is not None:
                c = corrector._add_direction(c, correction_cond.float(), correction_strength)

            if normalize_result:
                c = c / c.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            c = c.to(cond_tensor.dtype)

            # pooled_output
            new_meta = meta.copy()
            if apply_to_pooled and "pooled_output" in new_meta and new_meta["pooled_output"] is not None:
                p = new_meta["pooled_output"].clone().float()
                if problem_pooled is not None:
                    p = corrector._subtract_direction(p, problem_pooled.float(), subtract_strength)
                if correction_pooled is not None:
                    p = corrector._add_direction(p, correction_pooled.float(), correction_strength)
                if normalize_result:
                    p = p / p.norm(dim=-1, keepdim=True).clamp(min=1e-8)
                new_meta["pooled_output"] = p.to(meta["pooled_output"].dtype)

            # マスクを修正版に適用
            new_meta["mask"] = mask
            new_meta["set_area_to_bounds"] = set_area_to_bounds
            new_meta["mask_strength"] = mask_strength

            # 修正版を追加（ComfyUI は複数 Conditioning をリストで受け取り、
            # マスク付きのものはマスク領域にのみ適用する）
            pos_out.append((c, new_meta))

            orig = cond_tensor.norm().item()
            mod = c.norm().item()
            log_lines.append(f"  Pos[{idx}]: orig={orig:.4f}, corrected={mod:.4f} (masked)")

        # ─── Negative 修正: マスク領域のみ ───
        log_lines.append("--- Negative (masked correction) ---")
        neg_out = []

        for idx, (cond_tensor, meta) in enumerate(negative):
            # 元の Negative をそのまま保持
            neg_out.append((cond_tensor, meta.copy()))

            # 修正版: 問題方向を加算
            c = cond_tensor.clone().float()

            if long_text_strategy == "per_chunk" and problem_cond is None:
                n_chunks = len(problem_cond_chunks)
                per_s = negative_add_strength / n_chunks if n_chunks > 0 else 0
                for chunk in problem_cond_chunks:
                    c = corrector._add_direction(c, chunk.float(), per_s)
            elif problem_cond is not None:
                c = corrector._add_direction(c, problem_cond.float(), negative_add_strength)

            if normalize_result:
                c = c / c.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            c = c.to(cond_tensor.dtype)

            new_meta = meta.copy()
            if apply_to_pooled and "pooled_output" in new_meta and new_meta["pooled_output"] is not None:
                p = new_meta["pooled_output"].clone().float()
                if problem_pooled is not None:
                    p = corrector._add_direction(p, problem_pooled.float(), negative_add_strength)
                if normalize_result:
                    p = p / p.norm(dim=-1, keepdim=True).clamp(min=1e-8)
                new_meta["pooled_output"] = p.to(meta["pooled_output"].dtype)

            new_meta["mask"] = mask
            new_meta["set_area_to_bounds"] = set_area_to_bounds
            new_meta["mask_strength"] = mask_strength

            neg_out.append((c, new_meta))

            orig = cond_tensor.norm().item()
            mod = c.norm().item()
            log_lines.append(f"  Neg[{idx}]: orig={orig:.4f}, corrected={mod:.4f} (masked)")

        log_lines.append(f"Output: {len(pos_out)} positive entries, {len(neg_out)} negative entries")
        log_lines.append("  (each = original unmasked + corrected masked)")

        analysis_log = "\n".join(log_lines)
        return (pos_out, neg_out, analysis_log)


# =============================================================================
# Node mappings
# =============================================================================

NODE_CLASS_MAPPINGS = {
    "ConditioningProjection":        ConditioningProjection,
    "ConditioningCorrector":         ConditioningCorrector,
    "ConditioningCorrectorDual":     ConditioningCorrectorDual,
    "ConditioningCorrectorInpaint":  ConditioningCorrectorInpaint,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ConditioningProjection":        "naginagi · Conditioning Projection 🎯",
    "ConditioningCorrector":         "naginagi · Conditioning Corrector (LLM, no TIPO) 🎯🤖",
    "ConditioningCorrectorDual":     "naginagi · Conditioning Corrector Dual (Pos+Neg) 🎯➕➖",
    "ConditioningCorrectorInpaint":  "naginagi · Conditioning Corrector Inpaint (LLM+Mask) 🎯🩹",
}
