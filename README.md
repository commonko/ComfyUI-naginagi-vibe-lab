<div align="center">

<img src="assets/title.jpg" alt="ComfyUI-naginagi-vibe-lab" width="100%">

[English](README_en.md) | **日本語**

</div>

- [Top](#comfyui-naginagi-vibe-lab)
- [インストール](#インストール)
- [Nodes](#-nodes)
  - [Corrector](#corrector)
  - [Scheduler](#scheduler)
  - [HumanGate](#humangate)
- [Workflows](#-workflows)
- [動作確認環境](#動作確認環境)

# ComfyUI-naginagi-vibe-lab

**naginagi が vibe noding する、ちょっぴり実験的で、かなりへんてこで、ぽんこつなノードコレクションです。**

- LLM の画像分析テキストから Conditioning を直接補正する `Corrector` 系
- ステップ / 反復ループに沿って Conditioning や LoRA を変調する `Scheduler` 系
- ワークフロー実行中にユーザーが介入できる `HumanGate` 系 *(v1.1 NEW)*
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

### HumanGate

> *v1.1 NEW* — Nodes 2.0 対応

ワークフロー実行中に **ユーザーが介入** できる human-in-the-loop ノード群。フルスクリーンオーバーレイで画像/テキストを表示し、選択・承認・中止の判断を挟めます。

- `HumanGatePauseImage` ⏸️
  - IMAGE をパススルーする前にユーザーの Resume/Stop を待つ
- `HumanGateImageChooser` 🖼️
  - IMAGE バッチから選択 (`single` / `multiple`)。`pause_mode` で `always_pause` / `pass_through` / `take_first` / `take_last` / `repeat_last` を切替可
- `HumanGatePickImage` 👆
  - 最大 4 つの IMAGE 入力から 1 つを選択 (A/B/C/D ラベル付き)
- `HumanGatePickText` 📝
  - 最大 4 つの STRING 入力から 1 つを選択
- `HumanGateCompareChooser` ⚖️
  - ImageChooser の A/B 比較バリアント

**キーボードショートカット**:
`1`-`9` = 選択 / `Enter` = Resume / `Esc` = Stop / `A` = 全選択 / `C` = 選択解除

**バックエンド API** (`/humangate/*`):
| Method | Path | 用途 |
|---|---|---|
| GET | `/humangate/sessions` | 待機中セッション一覧 |
| GET | `/humangate/session/{gate_id}` | 1 セッション詳細 |
| POST | `/humangate/respond` | Resume/Stop + 選択インデックス |
| POST | `/humangate/cancel` | 強制 Stop |
| POST | `/humangate/cleanup` | 完了済みセッション掃除 |

> ⚠️ **Stop について**: v0.1 では `HumanGateUserStop` 例外を raise します。ComfyUI 上では Error Report として表示されますが、意図的な停止であり異常ではありません。v0.2+ で非エラーキャンセル API への移行を予定しています。

詳細は [docs/humangate.md](docs/humangate.md) を参照。

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
