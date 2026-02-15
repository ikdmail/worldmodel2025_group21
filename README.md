# Manual-Driven Adaptation for Web Agents
### ～成功指向アクションテンプレート（SOP）によるデータ効率の向上～

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

## 📌 プロジェクト概要
本リポジトリは、未知のドメインにおけるWebエージェントの適応能力を飛躍的に高めるフレームワーク
**「マニュアル駆動型適応（Manual-Driven Adaptation）」**の実装を公開するものです。

従来のWebエージェントは、UIの微細な変更（IDの更新など）に弱く、また自然言語のヒントだけでは複雑な業務フローを完遂できない課題がありました。本手法は、成功ログを抽象化した
**SOP（Standard Operating Procedures）**を外付けのマニュアルとして導入し、実行時に動的な記号接地（Just-In-Time Grounding）を行うことで、この問題を解決します。

## 🚀 主な特徴
- **IDレスな抽象化テンプレート**: 要素ID（BID）に依存せず、論理的な操作手順をSOPとして保持。UI変更への高い堅牢性を実現。
- **動的記号接地（Just-In-Time Grounding）**: 実行直前の画面状態（AXTree/DOM）に基づき、SOPの論理操作を具体的な要素IDへリアルタイムに紐付け。
- **プランニング時の形式検証**: 生成されたプランに対し、実行前に属性の正当性や網羅性を自動検証し、迷走を抑制。

## 📂 システム構成
本システムは `BrowserGym` をベースとし、LLM（Gemini-2.5-flash等）を中枢に据えたエージェント構成をとっています。

### 主要モジュール
- `Autonomous_agent_main.py`: 全体のエージェント・ループと意思決定の制御。
- `ConcretePlanner.py`: SOPと現在の画面情報を照らし合わせ、実行可能なプランを生成。
- `ObservationExtractor.py`: 多大なDOM情報から、タスクに関連する要素のみを抽出・解析。
- `GoalAnalyzer.py`: ユーザーの最終目的と現状のギャップを分析し、マニフェストを生成。

## 📊 実験結果 (Benchmark: WorkArena L1)
ServiceNow等の複雑なUIを対象としたベンチマークテストにおいて、以下の成果を収めました。

- **成功率 (Success Rate)**: **+12.2% 向上** (33.3% → 45.5%)
- **実行効率**: 平均ステップ数を削減したタスク完遂に寄与。

## 🛠 セットアップ
```bash
# リポジトリのクローン
git clone [https://github.com/your-username/manual-driven-agent.git](https://github.com/your-username/manual-driven-agent.git)
cd manual-driven-agent

# 依存関係のインストール
pip install -r requirements.txt

# BrowserGymのセットアップ
playwright install chromium

#config.yamlのセットアップ
WorkArenaにもどづく、HUGGING_FACE_HUB_TOKENなど、以下の設定を行う。

gemini_api_key: "AIXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXx"
gemini_api_url: "https://generativelanguage.googleapis.com/v1beta/openai/"
language: "ja"  # "ja" または "en"
sop_enabled: True
HUGGING_FACE_HUB_TOKEN: "hf_XXXXXXXXXXXXXXXXXXXXXXX"
