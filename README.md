<div align="center">

<img src="assets/title.jpg" alt="ComfyUI-naginagi-vibe-lab" width="100%">

[English](README_en.md) | **日本語**

</div>

- [Top](#comfyui-naginagi-vibe-lab)
- [インストール](#インストール)
- [Nodes](#-nodes)
  - [Corrector](#corrector)
  - [Scheduler](#scheduler)
- [動作確認環境](#動作確認環境)

# ComfyUI-naginagi-vibe-lab

**naginagi が vibe noding する、ちょっぴり実験的で、かなりへんてこで、ぽんこつなノードコレクションです。**

- LLM の画像分析テキストから Conditioning を直接補正する `Corrector` 系
- ステップ / 反復ループに沿って Conditioning や LoRA を変調する `Scheduler` 系
- ComfyUI 標準の `TextGenerate` (Qwen 3.5 等) との組み合わせを想定

## インストール

### A) ComfyUI Manager (推奨)

ComfyUI Manager の **"Install via Git URL"** に下記を入力:

```
https://github.com/<your-account>/ComfyUI-naginagi-vibe-lab
```

### B) 手動インストール

```bash
cd <ComfyUI>/custom_nodes/
git clone https://github.com/<your-account>/ComfyUI-naginagi-vibe-lab.git
```

依存ライブラリは ComfyUI 本体に含まれているものだけです。追加 `pip` 不要。

## 🍓 Nodes

すべてのノードは ComfyUI ノードメニュー `conditioning/vibe-lab/...` 配下に表示されます。表示名は `naginagi · ` プレフィックスで検索できます。

---

### Corrector

LLM の画像分析テキストを CLIP で埋め込み化し、元の Conditioning から **問題方向ベクトルを射影減算** することで、再生成や再学習なしで欠陥 (解剖エラー、画風崩れ、色被り等) を抑制するノード群です。

- `ConditioningProjection` 🎯
  - 2 つの CONDITIONING 間で素のベクトル射影 (add / subtract / replace) を行う低レベルノード
- `ConditioningCorrector` 🎯🤖
  - LLM テキスト → CLIP encode → 射影減算を 1 ノードで実行 (中核)
- `ConditioningCorrectorDual` 🎯➕➖
  - Positive と Negative を同時補正。Pos からは減算、Neg には加算
- `ConditioningCorrectorInpaint` 🎯🩹
  - マスク領域だけ補正。`SetLatentNoiseMask` と併用

詳細は [docs/corrector.md](docs/corrector.md) を参照。

---

### Scheduler

デノイズステップや反復ループに沿って Conditioning・LoRA・反復処理を時間方向に変調するノード群です。「初期は大胆に、後期は丁寧に」という非線形カーブ手法を実現します。

- `ConditioningStepScheduler` ⏱️📊
  - Conditioning 強度を 8 種類のカーブで非線形変調 (`bold_to_refined` / `weak_to_strong` 他)
- `LoRAScheduleApply` ⏱️🎚️
  - LoRA 効果をステップ別に変調
- `IterativeUpscalePlanner` ⏱️🔁
  - 反復アップスケールの反復回数を自動計算
- `IterativeStepScale` ⏱️📉
  - オーバーシュート対応の拡大倍率を計算 (`UltimateSDUpscale` 連携)
- `IterativeRefineDenoise` ⏱️🎛️
  - 反復ごとの `denoise` 値を非線形に計算
- `AdaptiveHaltCheck` ⏱️🛑
  - LLM 分析で「もう問題なし」と判定したら自動停止

詳細は [docs/scheduler.md](docs/scheduler.md) を参照。

---

---

## 📂 Workflows

`workflows/` フォルダにサンプルワークフロー JSON が入っています。ComfyUI の Load ボタンまたはドラッグ&ドロップで読み込めます。各ワークフロー内の `📝 Workflow Guide` ノートに使い方が書いてあります。

| ファイル | 内容 | 使用する vibe-lab ノード |
|---|---|---|
| `01_llm_conditioning_corrector.json` | LLM 分析 → DualCorrector → img2img (基本形) | `ConditioningCorrectorDual` |
| `02_method_a_t5gemma_conditioning_correction.json` | T5Gemma で Pos/Problem/Neg をエンコード → ConditioningProjection で手動射影 (実験的) | `ConditioningProjection` |
| `03_sdxl_iterative_refinement.json` | Base t2i → (LLM 分析 → Corrector → i2i) × N回 の反復改善ループ | `ConditioningCorrectorDual`, `IterativeRefineDenoise` |

---

## 動作確認環境

- ComfyUI 0.18.1 / Python 3.13 / PyTorch CUDA 13.0
- RTX 4060 Ti 16GB / Windows
- SDXL (NoobAI-XL v-pred) / Anima preview3 / Flux.2 Klein

## ライセンス

MIT License — see [LICENSE](LICENSE).
