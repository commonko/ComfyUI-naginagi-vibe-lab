# ⏸️ HumanGate — Human-in-the-loop ノード

ワークフロー実行中にユーザーが介入 (一時停止、選択、中止) できるノード群。

Nodes 2.0 対応: ノード body のリサイズは行わず、フルスクリーンオーバーレイ (`web/humangate.js` + `web/humangate.css`) で操作する。

## ノード一覧

### `HumanGatePauseImage` ⏸️

IMAGE をパススルーする前にユーザーの Resume / Stop を待つ。

| 入力 | 型 | デフォルト | 説明 |
|---|---|---|---|
| `image` | IMAGE | — | パススルーする画像 |
| `message` | STRING | `"Paused. Resume or Stop?"` | オーバーレイに表示するメッセージ |
| `timeout_sec` | INT | 0 | 0=無制限。秒数指定で自動 Resume |
| `alert` | BOOLEAN | True | オーバーレイ表示時にアラートを鳴らすか |

| 出力 | 型 |
|---|---|
| `image` | IMAGE |
| `decision` | STRING (`"resume"` or `"stop"`) |
| `gate_id` | STRING |

### `HumanGateImageChooser` 🖼️

IMAGE バッチから選択した画像だけを返す。

| 入力 | 型 | デフォルト | 説明 |
|---|---|---|---|
| `images` | IMAGE | — | バッチ画像 |
| `message` | STRING | `"Select image(s), then Resume."` | メッセージ |
| `selection_mode` | enum | `single` | `single` / `multiple` |
| `pause_mode` | enum | `always_pause` | 下記参照 |
| `timeout_sec` | INT | 0 | 0=無制限 |

**pause_mode**:
| 値 | 挙動 |
|---|---|
| `always_pause` | 毎回一時停止してユーザーに選択させる |
| `pass_through` | 一時停止せずバッチ全体をパススルー |
| `take_first` | 一時停止せず先頭画像を返す |
| `take_last` | 一時停止せず末尾画像を返す |
| `repeat_last` | 同じ prompt_id:node_id の前回選択を再利用。初回は一時停止 |

| 出力 | 型 |
|---|---|
| `images` | IMAGE (選択分) |
| `selected_indices` | STRING (JSON) |
| `selection_json` | STRING (メタデータ JSON) |

### `HumanGatePickImage` 👆

最大 4 つの IMAGE 入力から 1 つを選択。

| 入力 | 型 | 必須 |
|---|---|---|
| `image_1` | IMAGE | ✓ |
| `image_2` ~ `image_4` | IMAGE | |
| `message` | STRING | ✓ |
| `labels` | STRING (`"A,B,C,D"`) | ✓ |
| `pause_mode` | enum | ✓ |
| `timeout_sec` | INT | ✓ |

| 出力 | 型 |
|---|---|
| `image` | IMAGE |
| `selected_index` | INT |
| `selected_label` | STRING |
| `selection_json` | STRING |

### `HumanGatePickText` 📝

4 つの STRING 入力から 1 つを選択。

| 入力 | 型 |
|---|---|
| `text_1` ~ `text_4` | STRING (multiline) |
| `message` | STRING |
| `labels` | STRING |
| `pause_mode` | enum |
| `timeout_sec` | INT |

| 出力 | 型 |
|---|---|
| `text` | STRING |
| `selected_index` | INT |
| `selected_label` | STRING |
| `selection_json` | STRING |

### `HumanGateCompareChooser` ⚖️

`HumanGateImageChooser` の A/B 比較バリアント (サブクラス)。デフォルトメッセージが `"Choose the better image(s)."` に変更されている。

## バックエンド API

| Method | Path | Body | 説明 |
|---|---|---|---|
| GET | `/humangate/sessions` | — | 待機中セッション一覧 |
| GET | `/humangate/session/{gate_id}` | — | 1 セッション詳細 |
| POST | `/humangate/respond` | `{gate_id, result: {decision, selected_indices}}` | Resume/Stop |
| POST | `/humangate/cancel` | `{gate_id}` | 強制 Stop |
| POST | `/humangate/cleanup` | `{max_age_sec}` | 古いセッション削除 |

`server.py` は `humangate/nodes.py` が import された時点で自動的にルートを登録する (`from . import server` → `_register_routes()`)。

## フロントエンド

`web/humangate.js` + `web/humangate.css` が ComfyUI 拡張としてロードされる。

- `api.fetchApi` を使用 (ComfyUI の認証を自動通過)
- 750ms 間隔でポーリング
- 待機セッション検出時にフルスクリーンオーバーレイを表示

### キーボードショートカット

| Key | Action |
|---|---|
| `1`-`9` | 項目を選択/トグル |
| `Enter` | Resume |
| `Esc` | Stop |
| `A` | 全選択 (multiple モード) |
| `C` | 選択解除 |

## Stop について (v0.1)

Stop は `HumanGateUserStop` 例外を raise して実装。ComfyUI は例外を Error Report として表示するが、これは意図的な停止。

v0.2+ で ComfyUI が安定した非エラーキャンセル API を提供した場合、例外ベースの実装を置き換える予定。`HumanGateUserStop` はフォールバックとして残す。

詳細は [ROADMAP.md](../ROADMAP.md) を参照。

## サンプルワークフロー

`humangate/examples/` に 3 つの JSON ワークフローが含まれる:

| ファイル | 内容 |
|---|---|
| `01_pause_resume.json` | LoadImage → Pause → Preview |
| `02_image_batch_chooser.json` | バッチ生成 → Image Chooser → Preview |
| `03_pick_image_input.json` | 2 つの画像から 1 つを Pick |

モデル・画像ファイル名はプレースホルダー。自分の環境に合わせて変更してください。
