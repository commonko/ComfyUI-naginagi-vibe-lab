# 🎯 Corrector グループ

LLM が出力した「問題テキスト」を CLIP で埋め込み化し、元の Conditioning から **問題方向ベクトルを射影減算** することで、再生成や再学習なしで欠陥を抑制します。

## 動作原理

target conditioning $c$ から source の方向 $s$ を強度 $\alpha$ で減算する射影:

$$
\hat{s} = \frac{s}{\|s\|}, \quad
c' = c - \alpha \, (c \cdot \hat{s}) \, \hat{s}
$$

$c$ を $s$ の張る部分空間に直交する超平面へ射影する操作で、$\alpha=1$ のとき「$s$ 方向の成分が完全消去された $c$」になります。

## ノード詳細

### `ConditioningProjection` 🎯
低レベルの素のベクトル射影。LLM 経由ではなく自分で別途 CLIP encode した方向 cond を直接渡す上級者向けノード。

| パラメータ | 型 | デフォルト | 説明 |
|---|---|---|---|
| `target` | CONDITIONING | — | 補正対象 |
| `source` | CONDITIONING | — | 射影方向 |
| `method` | enum | `subtract` | `add` / `subtract` / `replace` |
| `strength` | FLOAT | 0.3 | 0.0–2.0 |

### `ConditioningCorrector` 🎯🤖 (中核)
LLM 分析テキストを内部で CLIP encode して射影減算する統合ノード。

| パラメータ | 型 | デフォルト | 説明 |
|---|---|---|---|
| `conditioning` | CONDITIONING | — | 元の Conditioning |
| `clip` | CLIP | — | エンコーダ |
| `problem_text` | STRING | — | LLM 出力テキスト (forceInput) |
| `subtract_strength` | FLOAT | **0.3** | 0.0–2.0 |
| `long_text_strategy` | enum | `sentence_avg` | `sentence_avg` / `mean_pool` / `per_chunk` / `truncate` |
| `correction_text` | STRING | — | 理想状態の記述 (任意) |
| `add_strength` | FLOAT | 0.2 | 0.0–2.0 |
| `apply_to_pooled` | BOOL | True | SDXL 用 |

**Returns**: `(conditioning, analysis_log)`

### `ConditioningCorrectorDual` 🎯➕➖
Positive/Negative 両方を同時補正。Pos からは減算、Neg には加算。SDXL → Anima Hi-fix 等の v-pred 設定で有効。

### `ConditioningCorrectorInpaint` 🎯🩹
LLM 分析 + マスクで「特定領域だけ修正」。Inpaint ワークフローで顔だけ・手だけ等の用途。`SetLatentNoiseMask` と併用。

## 推奨ワークフロー

### 基本形
```
LoadImage → TextGenerate(Qwen3.5)
                  ↓ STRING (LLM 自然言語出力)
CLIPTextEncode ─→ CONDITIONING ─→ ConditioningCorrector ─→ KSampler
                                       ↑       ↑
                                  problem_text  CLIP
```

### Pos/Neg 同時補正
```
TextGenerate ─→ problem_text
       ┌──────────┴──────────┐
       ↓                     ↓
   Positive               Negative
       ↓                     ↓
       ConditioningCorrectorDual
            ↓        ↓
        positive  negative → KSampler
```

### Inpaint (顔だけ修正)
```
LoadImage → MaskFromCLIPSeg("face") → mask
                                        ↓
TextGenerate(face_problem) → problem_text
                                ↓
CONDITIONING → ConditioningCorrectorInpaint ←─ mask
                       ↓
                   KSampler
```

## パラメータ調整の目安

| 状況 | `subtract_strength` | `add_strength` | `long_text_strategy` |
|---|---|---|---|
| 軽微なエラー | 0.2 | 0.1 | `truncate` |
| 標準 (推奨初期値) | **0.3** | **0.2** | **`sentence_avg`** |
| 強めの修正 | 0.5 | 0.3 | `per_chunk` |
| 高速処理優先 | 0.3 | 0.2 | `mean_pool` |

⚠️ `subtract_strength > 0.7` は cond の意味的崩壊を引き起こすことがあります。

## long_text_strategy の比較

| Strategy | 挙動 | 適用場面 |
|---|---|---|
| `sentence_avg` | 文ごとに個別エンコード → 射影方向を平均 | **推奨**。LLM が複数の問題を列挙した時最も精度が高い |
| `mean_pool` | 全チャンクを mean-pool で 77 トークンに圧縮 | 高速・情報保持 ◎ |
| `per_chunk` | チャンクごとに個別射影 → 結果を合成 | 方向保持 ◎ |
| `truncate` | 先頭チャンクのみ使用 | 従来互換 |
