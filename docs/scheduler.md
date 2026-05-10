# ⏱️ Scheduler グループ

デノイズステップや反復ループに沿って、Conditioning の強度・LoRA の効き・デノイズ強度などを **時間方向に変化させる** ノード群。

## 設計思想 ― 「初期は大胆に、後期は丁寧に」

| 対象 | 初期ステップ | 後期ステップ | 推奨カーブ |
|---|---|---|---|
| Positive cond | **強め** (構図探索) | 弱め (品質仕上げ) | `bold_to_refined` |
| Negative cond | 弱め (自由度重視) | **強め** (制約締め) | `weak_to_strong` |
| LoRA | **強め** (スタイル確立) | 弱め (歪み抑制) | `bold_to_refined` |

## 8 種類のカーブ

| Curve | 形状 | 用途 |
|---|---|---|
| `bold_to_refined` | 単調減少 | Positive cond / LoRA |
| `weak_to_strong` | 単調増加 | Negative cond |
| `peak_mid` | 山型 | Style LoRA (中盤で個性最大化) |
| `cosine_decay` | コサイン下降 | 滑らかな減衰 |
| `cosine_rise` | コサイン上昇 | 滑らかな増加 |
| `linear_decay` | 直線下降 | シンプル |
| `linear_rise` | 直線上昇 | シンプル |
| `custom` | 任意 | カンマ区切り (`"1.2, 1.0, 0.8, 0.6, 0.5, 0.4"`) |

## ノード詳細

### `ConditioningStepScheduler` ⏱️📊
内部で `ConditioningSetTimestepRange × N段` + 強度スケーリングを実行し、1 ノードで非線形カーブを実現。

| パラメータ | 型 | デフォルト | 説明 |
|---|---|---|---|
| `conditioning` | CONDITIONING | — | 対象 |
| `curve` | enum | `bold_to_refined` | 8 種類 (上記) |
| `strength_max` | FLOAT | 1.2 | 0.0–3.0 |
| `strength_min` | FLOAT | 0.4 | 0.0–3.0 |
| `segments` | INT | 6 | 2–20 (多いほど滑らか) |
| `custom_curve` | STRING | — | `curve=custom` 時のみ |

### `LoRAScheduleApply` ⏱️🎚️
LoRA の効きをデノイズステップに応じて変調。`LoraLoader` の後に挟んで KSampler に渡す。

### `IterativeUpscalePlanner` ⏱️🔁
反復アップスケールの反復回数を自動計算 (16 ノード分の数学演算を 1 ノードに圧縮)。`easy forLoopStart` の `total` に接続。

### `IterativeStepScale` ⏱️📉 (Overshoot Safe)
ループ内で現在画像サイズから **オーバーシュート対応** の拡大倍率を計算。`UltimateSDUpscale` の `upscale_by` に接続。

### `IterativeRefineDenoise` ⏱️🎛️
反復ごとの `denoise` 値を非線形に計算。「初期は大きく修正、後期は微調整」の彫琢プロセス。

### `AdaptiveHaltCheck` ⏱️🛑
ACT (Adaptive Computation Time) 風の適応停止判定。LLM 分析結果から「もう問題なし」と判断したら `denoise=0` を返してループ脱出。

## 推奨ワークフロー

### A) 非線形ステップ
```
CLIPTextEncode (Pos) → ConditioningStepScheduler [bold_to_refined, max=1.2, min=0.4]
CLIPTextEncode (Neg) → ConditioningStepScheduler [weak_to_strong,  max=1.2, min=0.3]
                                                              ↓
                                                         KSampler
```

### B) LoRA 変調 + Conditioning 変調
```
LoraLoader → LoRAScheduleApply [bold_to_refined, 1.5→0.3]
                  ↓ MODEL
       (Pos cond は ConditioningStepScheduler 経由)
                  ↓
              KSampler
```

### C) 反復アップスケール
```
IterativeUpscalePlanner → iterations
       ↓
       │ (forLoopStart に渡す)
┌──────┴──────────────────────────────────┐
│  current_image                          │
│       ↓                                 │
│  IterativeStepScale → upscale_by        │
│       ↓                                 │
│  UltimateSDUpscale                      │
│       ↓                                 │
│  IterativeRefineDenoise → denoise       │
│       ↓                                 │
│  KSampler                               │
│       ↓                                 │
│  AdaptiveHaltCheck (収束で break)       │
└─────────────────────────────────────────┘
```

## トラブルシューティング

| 症状 | 対処 |
|---|---|
| カーブが効いてない感じ | `segments` を増やす (6→10)、`strength_max - strength_min` を広げる |
| LoRAScheduleApply が無効 | サンプラーが `model_options.lora_weight_schedule` を参照していない可能性。`ConditioningStepScheduler` + 弱め LoRA で代替 |
| AdaptiveHaltCheck が早すぎ/遅すぎ | 閾値 (`tau`) を調整、`min_iters` で最低反復回数を保証 |
| 反復で解像度が指数的に膨張 | `IterativeStepScale` を必ず挟む (Overshoot Safe) |
