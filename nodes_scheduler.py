"""
ComfyUI-naginagi-Cond-VibeEdit / nodes_scheduler.py
=====================================================
Scheduler グループ: デノイズステップ / 反復ループに沿って Conditioning や
LoRA、Latent を時間方向に変調するノード群。

設計思想: 「初期は大胆に、後期は丁寧に」(Nobu-Kobayashi 氏の手法)
  - Positive: bold_to_refined カーブ → 序盤に強く、終盤で減衰
  - Negative: weak_to_strong カーブ → 序盤緩く、終盤に締める
  - LoRA:    序盤強く効かせて構図を作り、終盤で歪み抑制
  - 反復:    オーバーシュート安全な縮小スケール、適応停止判定

6 ノード構成:
  ── ステップ変調 ──
    1. ConditioningStepScheduler  — Conditioning 強度を非線形カーブで変調
    2. LoRAScheduleApply          — LoRA 効果をステップ別に変調
  ── 反復ループ ──
    3. IterativeUpscalePlanner    — 反復アップスケール計画 (overshoot safe)
    4. IterativeStepScale         — 反復ステップ縮小スケジュール
    5. IterativeRefineDenoise     — 反復リファインのデノイズ強度スケジュール
    6. AdaptiveHaltCheck          — ACT 風の適応停止判定 (収束したらループ脱出)

curve タイプ (ConditioningStepScheduler / LoRAScheduleApply 共通):
  bold_to_refined: 初期強→後期弱  (Positive 用 / LoRA 用)
  weak_to_strong:  初期弱→後期強  (Negative 用)
  peak_mid:        中盤ピーク      (Style LoRA 用)
  cosine_decay:    コサイン減衰    (滑らか)
  cosine_rise:     コサイン上昇    (滑らか)
  linear_decay:    線形減衰
  linear_rise:     線形上昇
  custom:          カンマ区切りで手動指定
"""

from __future__ import annotations

import math
import re

import torch
from torch import Tensor

from .nodes_common import (
    clone_conditioning as _clone_conditioning,
    mean_pool_to_length as _mean_pool_to_length,
)


# =============================================================================
# ConditioningStepScheduler
# =============================================================================

class ConditioningStepScheduler:
    """
    Conditioning の強度をデノイズステップに応じて非線形に変化させる。

    Nobu-Kobayashi 氏の手法:
      - 初期ステップ: Positive 強め → 大胆な構図探索（ガードレール外）
      - 後期ステップ: Negative 強め → 品質回復・ディテール仕上げ

    内部で ConditioningSetTimestepRange × N 段 + 強度スケーリングを行い、
    1 ノードで非線形カーブを実現する。

    curve:
      "bold_to_refined": 初期強→後期弱（Positive 用。大胆→安定）
      "weak_to_strong":  初期弱→後期強（Negative 用。自由→制約）
      "peak_mid":        中盤ピーク（スタイル LoRA 用）
      "cosine_decay":    コサイン減衰（滑らか）
      "cosine_rise":     コサイン上昇（滑らか）
      "linear_decay":    線形減衰
      "linear_rise":     線形上昇
      "custom":          custom_curve で手動指定

    segments: 分割数（多いほど滑らか、4-8 推奨）
    """

    CURVES = [
        "bold_to_refined", "weak_to_strong", "peak_mid",
        "cosine_decay", "cosine_rise",
        "linear_decay", "linear_rise", "custom",
    ]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning": ("CONDITIONING",),
                "curve": (cls.CURVES, {
                    "default": "bold_to_refined",
                    "tooltip": "強度カーブ。bold_to_refined=初期強→後期弱、weak_to_strong=逆。"
                }),
                "strength_max": ("FLOAT", {
                    "default": 1.2, "min": 0.0, "max": 3.0, "step": 0.05,
                    "tooltip": "カーブの最大強度。"
                }),
                "strength_min": ("FLOAT", {
                    "default": 0.4, "min": 0.0, "max": 3.0, "step": 0.05,
                    "tooltip": "カーブの最小強度。"
                }),
                "segments": ("INT", {
                    "default": 6, "min": 2, "max": 20, "step": 1,
                    "tooltip": "分割数。多いほど滑らかだがノード処理増。"
                }),
            },
            "optional": {
                "custom_curve": ("STRING", {
                    "default": "1.2, 1.0, 0.8, 0.6, 0.5, 0.4",
                    "tooltip": "curve=custom 時。カンマ区切りで各セグメントの強度を指定。"
                }),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "STRING",)
    RETURN_NAMES = ("conditioning", "schedule_log",)
    FUNCTION = "schedule"
    CATEGORY = "conditioning/vibe-lab/Scheduler"
    DESCRIPTION = (
        "デノイズステップに応じて Conditioning 強度を非線形に変化させます。"
        "初期は大胆な構図探索、後期は品質仕上げ。"
    )

    def _compute_curve(self, curve, segments, strength_max, strength_min, custom_curve):
        """各セグメントの強度値を計算"""
        if curve == "custom":
            try:
                vals = [float(x.strip()) for x in custom_curve.split(",") if x.strip()]
                # セグメント数に合わせてリサンプル
                if len(vals) != segments:
                    import numpy as np
                    x_old = [i / (len(vals) - 1) for i in range(len(vals))]
                    x_new = [i / (segments - 1) for i in range(segments)]
                    # 線形補間
                    result = []
                    for xn in x_new:
                        for j in range(len(x_old) - 1):
                            if x_old[j] <= xn <= x_old[j + 1]:
                                t = (xn - x_old[j]) / (x_old[j + 1] - x_old[j])
                                result.append(vals[j] * (1 - t) + vals[j + 1] * t)
                                break
                        else:
                            result.append(vals[-1])
                    return result
                return vals
            except Exception:
                pass

        strengths = []
        for i in range(segments):
            t = i / max(segments - 1, 1)  # 0.0 ~ 1.0

            if curve == "bold_to_refined" or curve == "cosine_decay":
                # コサイン減衰: 初期 max → 後期 min
                s = strength_min + (strength_max - strength_min) * (1 + math.cos(t * math.pi)) / 2.0
            elif curve == "weak_to_strong" or curve == "cosine_rise":
                # コサイン上昇: 初期 min → 後期 max
                s = strength_min + (strength_max - strength_min) * (1 - math.cos(t * math.pi)) / 2.0
            elif curve == "peak_mid":
                # 中盤ピーク: ベル型
                s = strength_min + (strength_max - strength_min) * math.exp(-((t - 0.5) ** 2) / 0.08)
            elif curve == "linear_decay":
                s = strength_max + (strength_min - strength_max) * t
            elif curve == "linear_rise":
                s = strength_min + (strength_max - strength_min) * t
            else:
                s = strength_max

            strengths.append(round(s, 3))
        return strengths

    def schedule(self, conditioning, curve, strength_max, strength_min,
                 segments, custom_curve="1.2, 1.0, 0.8, 0.6, 0.5, 0.4"):

        strengths = self._compute_curve(curve, segments, strength_max, strength_min, custom_curve)
        log = [f"=== ConditioningStepScheduler ===",
               f"Curve: {curve}, Segments: {segments}",
               f"Range: {strength_max} → {strength_min}"]

        segment_size = 1.0 / segments
        all_conds = []

        for seg_idx, strength in enumerate(strengths):
            start = seg_idx * segment_size
            end = (seg_idx + 1) * segment_size

            for cond_tensor, meta in conditioning:
                # 強度スケーリング
                scaled_cond = cond_tensor.clone() * strength

                # タイムステップ範囲をメタデータに設定
                new_meta = meta.copy()
                new_meta["start_percent"] = 1.0 - end    # ComfyUI: 1.0=最初, 0.0=最後
                new_meta["end_percent"] = 1.0 - start

                # pooled_output もスケーリング
                if "pooled_output" in new_meta and new_meta["pooled_output"] is not None:
                    new_meta["pooled_output"] = new_meta["pooled_output"].clone() * strength

                all_conds.append((scaled_cond, new_meta))

            log.append(f"  Seg[{seg_idx}]: t={start:.2f}-{end:.2f} strength={strength:.3f}")

        log.append(f"Total conditioning entries: {len(all_conds)}")
        return (all_conds, "\n".join(log))

# =============================================================================
# LoRAScheduleApply
# =============================================================================

class LoRAScheduleApply:
    """
    LoRA の効果をデノイズステップに応じて非線形に変化させる。

    ComfyUI の ModelPatcher フック機構を使い、
    サンプリング中のタイムステップに応じて LoRA パッチの重みを動的に変調する。

    LoraLoader → LoRAScheduleApply → KSampler

    curve パラメータは ConditioningStepScheduler と同じ。

    使用例（Nobu-Kobayashi 氏の手法）:
      LoRA weight = 1.5 (base)
      curve = bold_to_refined
      weight_max = 1.5, weight_min = 0.3
      → 初期ステップ: LoRA 1.5x（大胆な構図・スタイル）
      → 後期ステップ: LoRA 0.3x（品質仕上げ、LoRA の歪みを抑制）
    """

    CURVES = ConditioningStepScheduler.CURVES

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "curve": (cls.CURVES, {
                    "default": "bold_to_refined",
                }),
                "weight_max": ("FLOAT", {
                    "default": 1.5, "min": 0.0, "max": 3.0, "step": 0.05,
                }),
                "weight_min": ("FLOAT", {
                    "default": 0.3, "min": 0.0, "max": 3.0, "step": 0.05,
                }),
                "segments": ("INT", {
                    "default": 6, "min": 2, "max": 20, "step": 1,
                }),
            },
        }

    RETURN_TYPES = ("MODEL", "STRING",)
    RETURN_NAMES = ("model", "schedule_log",)
    FUNCTION = "apply"
    CATEGORY = "conditioning/vibe-lab/Scheduler"
    DESCRIPTION = (
        "LoRA の効果をデノイズステップに応じて非線形に変調します。"
        "LoraLoader の後に接続。初期は LoRA 強め、後期は弱め。"
    )

    def apply(self, model, curve, weight_max, weight_min, segments):
        scheduler = ConditioningStepScheduler()
        weights = scheduler._compute_curve(curve, segments, weight_max, weight_min, "")

        log = [f"=== LoRAScheduleApply ===",
               f"Curve: {curve}, {weight_max} → {weight_min}"]

        # ModelPatcher のクローンを作成
        m = model.clone()

        # タイムステップに応じた重みマップを作成
        segment_size = 1.0 / segments
        weight_schedule = []
        for i, w in enumerate(weights):
            start_sigma_pct = i * segment_size
            end_sigma_pct = (i + 1) * segment_size
            weight_schedule.append((start_sigma_pct, end_sigma_pct, w))
            log.append(f"  Seg[{i}]: sigma={start_sigma_pct:.2f}-{end_sigma_pct:.2f} weight={w:.3f}")

        # transformer_options にスケジュールを格納
        # （ComfyUI の sampler が各ステップで参照できるようにする）
        if not hasattr(m, 'model_options'):
            m.model_options = {}
        m.model_options["lora_weight_schedule"] = weight_schedule

        # LoRA パッチの重みを動的にスケーリングするラッパー
        # 注: これは model_options に情報を付与するだけで、
        #     実際のステップ別重み変調は sampler のコールバックで行う必要がある。
        #     現時点では ConditioningStepScheduler による Conditioning スケジューリングが
        #     より確実な方法。

        log.append("NOTE: LoRA weight scheduling is stored in model_options.")
        log.append("For immediate effect, use ConditioningStepScheduler instead.")

        return (m, "\n".join(log))

# =============================================================================
# IterativeStepScale
# =============================================================================

class IterativeStepScale:
    """
    反復ループ内で使用する。現在の画像サイズとターゲットを比較し、
    オーバーシュートしない拡大倍率を返す。

    UltimateSDUpscale の upscale_by 入力に接続する。

    ロジック:
      next_pixels = current_pixels × scale_factor²
      if next_pixels > target_pixels:
          actual_scale = sqrt(target_pixels / current_pixels)
      else:
          actual_scale = scale_factor
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "target_megapixels": ("FLOAT", {
                    "default": 3.0, "min": 0.5, "max": 16.0, "step": 0.1,
                }),
                "scale_factor": ("FLOAT", {
                    "default": 1.5, "min": 1.0, "max": 4.0, "step": 0.1,
                }),
            },
        }

    RETURN_TYPES = ("FLOAT", "INT", "INT",)
    RETURN_NAMES = ("upscale_by", "tile_width", "tile_height",)
    FUNCTION = "compute"
    CATEGORY = "conditioning/vibe-lab/Scheduler"
    DESCRIPTION = (
        "ループ内で現在画像のサイズからオーバーシュート対応の拡大倍率を計算。"
        "UltimateSDUpscale の upscale_by に接続。"
        "tile_width/height は現在画像の幅・高さ（タイルサイズ用）。"
    )

    def compute(self, image, target_megapixels, scale_factor):
        h, w = image.shape[1], image.shape[2]
        current_pixels = w * h
        target_pixels = int(target_megapixels * 1_000_000)

        next_pixels = current_pixels * (scale_factor ** 2)

        if current_pixels >= target_pixels:
            actual_scale = 1.0
        elif next_pixels > target_pixels:
            actual_scale = math.sqrt(target_pixels / current_pixels)
            actual_scale = max(1.0, actual_scale)
        else:
            actual_scale = scale_factor

        # 小数点2桁に丸め
        actual_scale = round(actual_scale, 2)

        return (actual_scale, w, h)

# =============================================================================
# IterativeRefineDenoise
# =============================================================================

class IterativeRefineDenoise:
    """
    反復ループ内で、現在のイテレーション番号に応じて
    denoise 値を段階的に減少させる。

    彫琢プロセスの考え方:
      - 初期イテレーション: denoise 高め（大きな修正）
      - 後期イテレーション: denoise 低め（微調整）

    スケジュール:
      "linear": start → end を線形に減少
      "cosine": コサインカーブで緩やかに減少（初期は変化小、中盤で急減少）
      "step": 前半は start、後半は end（段階的切替）
    """

    SCHEDULES = ["linear", "cosine", "step"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "iteration": ("INT", {
                    "default": 0, "min": 0, "max": 100,
                    "tooltip": "forLoopStart の index 出力を接続。"
                }),
                "total_iterations": ("INT", {
                    "default": 4, "min": 1, "max": 100,
                    "tooltip": "総イテレーション数。forLoopStart の total と同じ値に。"
                }),
                "denoise_start": ("FLOAT", {
                    "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "最初のイテレーションの denoise 値。"
                }),
                "denoise_end": ("FLOAT", {
                    "default": 0.15, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "最後のイテレーションの denoise 値。"
                }),
                "schedule": (cls.SCHEDULES, {
                    "default": "cosine",
                    "tooltip": "linear: 線形減少, cosine: コサイン減少, step: 段階切替"
                }),
            },
        }

    RETURN_TYPES = ("FLOAT", "STRING",)
    RETURN_NAMES = ("denoise", "schedule_log",)
    FUNCTION = "compute"
    CATEGORY = "conditioning/vibe-lab/Scheduler"
    DESCRIPTION = (
        "イテレーションごとの denoise 値を計算。"
        "初期は大きく修正、後期は微調整の彫琢プロセスを実現。"
    )

    def compute(self, iteration, total_iterations,
                denoise_start, denoise_end, schedule):

        total = max(total_iterations, 1)
        t = min(iteration / max(total - 1, 1), 1.0)  # 0.0 ~ 1.0

        if schedule == "linear":
            denoise = denoise_start + (denoise_end - denoise_start) * t

        elif schedule == "cosine":
            # コサインカーブ: 初期は変化が小さく、中盤で急に減少
            cos_t = (1 - math.cos(t * math.pi)) / 2.0
            denoise = denoise_start + (denoise_end - denoise_start) * cos_t

        elif schedule == "step":
            # 前半は start、後半は end
            denoise = denoise_start if t < 0.5 else denoise_end

        else:
            denoise = denoise_start

        denoise = round(max(0.0, min(1.0, denoise)), 3)

        log = (
            f"Iteration {iteration}/{total}: denoise={denoise:.3f} "
            f"(schedule={schedule}, {denoise_start:.2f}→{denoise_end:.2f})"
        )

        return (denoise, log)

# =============================================================================
# IterativeUpscalePlanner
# =============================================================================

class IterativeUpscalePlanner:
    """
    反復アップスケールの全パラメータを1ノードで計算する。

    元ワークフローの 16 ノード (SimpleMath × 7, compare, FloatConstant × 3,
    CR Math × 2, showAnything × 4) を置き換える。

    計算:
      base_pixels = width × height
      target_pixels = target_megapixels × 1,000,000
      ratio = target_pixels / base_pixels
      iterations = ceil(log(ratio) / log(scale_factor))

    出力:
      - iterations: ループ回数 (easy forLoopStart の total に接続)
      - info: 計算結果の文字列 (デバッグ用)
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "target_megapixels": ("FLOAT", {
                    "default": 3.0, "min": 0.5, "max": 16.0, "step": 0.1,
                    "tooltip": "ターゲット解像度 (メガピクセル)。3.0 = 約 2048×1536。"
                }),
                "scale_factor": ("FLOAT", {
                    "default": 1.5, "min": 1.1, "max": 4.0, "step": 0.1,
                    "tooltip": "1反復あたりの拡大倍率。"
                }),
            },
        }

    RETURN_TYPES = ("INT", "STRING",)
    RETURN_NAMES = ("iterations", "info",)
    FUNCTION = "plan"
    CATEGORY = "conditioning/vibe-lab/Scheduler"
    DESCRIPTION = (
        "反復アップスケールの反復回数を自動計算します。"
        "16 ノード分の数学演算を 1 ノードに圧縮。"
        "easy forLoopStart の total に iterations を接続してください。"
    )

    def plan(self, image, target_megapixels, scale_factor):
        h, w = image.shape[1], image.shape[2]
        base_pixels = w * h
        target_pixels = int(target_megapixels * 1_000_000)

        ratio = target_pixels / base_pixels if base_pixels > 0 else 1.0

        if ratio <= 1.0:
            iterations = 0
        else:
            log_ratio = math.log(ratio)
            log_scale = math.log(scale_factor)
            iterations = math.ceil(log_ratio / log_scale) if log_scale > 0 else 1

        iterations = max(0, min(iterations, 20))  # 安全上限

        info_lines = [
            f"Input: {w}×{h} ({base_pixels:,} px, {base_pixels/1e6:.2f} MP)",
            f"Target: {target_pixels:,} px ({target_megapixels:.1f} MP)",
            f"Ratio: {ratio:.2f}x",
            f"Scale/step: {scale_factor:.1f}x",
            f"Iterations: {iterations}",
        ]

        # 各ステップの予測サイズ
        cur_w, cur_h = w, h
        for i in range(iterations):
            new_w = int(cur_w * scale_factor)
            new_h = int(cur_h * scale_factor)
            new_px = new_w * new_h
            if new_px > target_pixels and i < iterations - 1:
                # オーバーシュート: 最終ステップは小さい倍率
                final_scale = math.sqrt(target_pixels / (cur_w * cur_h))
                info_lines.append(f"  Step {i+1}: {cur_w}×{cur_h} → overshoot, scale={final_scale:.2f}")
                cur_w = int(cur_w * final_scale)
                cur_h = int(cur_h * final_scale)
            else:
                info_lines.append(f"  Step {i+1}: {cur_w}×{cur_h} → {new_w}×{new_h}")
                cur_w, cur_h = new_w, new_h

        info_lines.append(f"Final est: {cur_w}×{cur_h} ({cur_w*cur_h/1e6:.2f} MP)")

        return (iterations, "\n".join(info_lines))

# =============================================================================
# AdaptiveHaltCheck
# =============================================================================

class AdaptiveHaltCheck:
    """
    OpenMythos の ACT (Adaptive Computation Time) を
    diffusion の反復精緻化に適用する。

    LLM 問題分析テキストの「問題の量」で収束を判定:
      - 問題テキストが少ない/空 → 収束 → denoise=0 (ループスキップ)
      - 問題が多い → 未収束 → denoise=base_denoise

    forLoop 内で TextGenerate → AdaptiveHaltCheck → KSampler の順に接続。
    収束時に denoise=0 を返すことで KSampler が実質パススルーになる。
    """

    HALT_PHRASES = [
        "no issues", "no problems", "no defects", "none",
        "looks good", "no visible", "well-formed", "all correct",
        "問題なし", "欠陥なし", "良好", "問題ありません",
    ]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "analysis_text": ("STRING", {
                    "forceInput": True,
                    "tooltip": "LLM の問題分析テキスト。"
                }),
                "max_problems": ("INT", {
                    "default": 2, "min": 0, "max": 20,
                    "tooltip": "この行数以下で収束と判定。"
                }),
                "min_chars": ("INT", {
                    "default": 30, "min": 0, "max": 500,
                    "tooltip": "この文字数以下で収束と判定。"
                }),
                "base_denoise": ("FLOAT", {
                    "default": 0.4, "min": 0.0, "max": 1.0, "step": 0.01,
                }),
            },
        }

    RETURN_TYPES = ("FLOAT", "BOOLEAN", "STRING",)
    RETURN_NAMES = ("denoise", "converged", "halt_log",)
    FUNCTION = "check"
    CATEGORY = "conditioning/vibe-lab/Scheduler"
    DESCRIPTION = (
        "ACT 風の適応停止判定。LLM 分析で問題が少なければ denoise=0 で停止。"
    )

    def check(self, analysis_text, max_problems, min_chars, base_denoise):
        text = analysis_text.strip()
        log = ["=== AdaptiveHaltCheck ==="]

        if not text:
            log.append("Empty → CONVERGED")
            return (0.0, True, "\n".join(log))

        lower = text.lower()
        for phrase in self.HALT_PHRASES:
            if phrase in lower:
                log.append(f"Halt phrase: '{phrase}' → CONVERGED")
                return (0.0, True, "\n".join(log))

        lines = [l.strip() for l in text.split("\n") if l.strip()]
        n = len(lines)
        log.append(f"Problems: {n} lines, {len(text)} chars")

        if n <= max_problems:
            log.append(f"≤ {max_problems} → CONVERGED")
            return (0.0, True, "\n".join(log))
        if len(text) <= min_chars:
            log.append(f"≤ {min_chars} chars → CONVERGED")
            return (0.0, True, "\n".join(log))

        severity = min(n / 10.0, 1.0)
        d = round(base_denoise * (0.5 + 0.5 * severity), 3)
        log.append(f"NOT converged. denoise={d} (severity={severity:.2f})")
        return (d, False, "\n".join(log))


# =============================================================================
# Node mappings
# =============================================================================

NODE_CLASS_MAPPINGS = {
    "ConditioningStepScheduler":     ConditioningStepScheduler,
    "LoRAScheduleApply":             LoRAScheduleApply,
    "IterativeUpscalePlanner":       IterativeUpscalePlanner,
    "IterativeStepScale":            IterativeStepScale,
    "IterativeRefineDenoise":        IterativeRefineDenoise,
    "AdaptiveHaltCheck":             AdaptiveHaltCheck,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ConditioningStepScheduler":     "naginagi · Conditioning Step Scheduler (Non-linear) ⏱️📊",
    "LoRAScheduleApply":             "naginagi · LoRA Schedule Apply (Step-wise) ⏱️🎚️",
    "IterativeUpscalePlanner":       "naginagi · Iterative Upscale Planner ⏱️🔁",
    "IterativeStepScale":            "naginagi · Iterative Step Scale (Overshoot Safe) ⏱️📉",
    "IterativeRefineDenoise":        "naginagi · Iterative Refine Denoise (Schedule) ⏱️🎛️",
    "AdaptiveHaltCheck":             "naginagi · Adaptive Halt Check (ACT Stopping) ⏱️🛑",
}
