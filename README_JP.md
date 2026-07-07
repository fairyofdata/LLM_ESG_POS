**🌐 Available Versions:** [🇺🇸 English](/README.md) | [🇰🇷 한국어 (Korean)](/README_KR.md)

---

# LEPOS: LLMベース ESG重視ポートフォリオ最適化サービス 📊🌱

> 🏆🥈 光云大学 第8回 産学連携SWプロジェクト展示会 優秀賞(2位)
> 🏆🥈 光云大学 SW融合大学 2024年卒業展示会 優秀賞(2位)

![Python](https://img.shields.io/badge/python-3.11+-blue)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B)
![License](https://img.shields.io/badge/license-MIT-green)
[![tests](https://github.com/fairyofdata/LLM_ESG_POS/actions/workflows/tests.yml/badge.svg)](https://github.com/fairyofdata/LLM_ESG_POS/actions/workflows/tests.yml)

LEPOSは、**LLMとファインチューニングした韓国語言語モデルを用いてニュース記事のテキストから直接**
韓国上場企業のESG(環境・社会・ガバナンス)を評価し、ユーザーのESG選好を投資家ビューとして
組み込んだ**ブラック・リターマン最適化**によりパーソナライズされたポートフォリオを構築する
サービスです。

従来のESG評価は算出根拠が不透明で、評価機関ごとに結果が異なり、大企業しかカバーしません。
LEPOSはスコアリングのパイプライン全体が透明に公開されており、5つの評価機関(MSCI、S&P、
Sustainalytics、ISS、KCGS)の評価傾向を学習し、テキストさえあれば評価機関がカバーしない
企業でもスコアリングできます。

## 🎬 デモ

[![UI/UXデモ動画](https://img.youtube.com/vi/kHAtgLC4PJY/0.jpg)](https://www.youtube.com/watch?v=kHAtgLC4PJY)

*(サムネイルをクリックするとYouTubeでUI/UX紹介動画をご覧いただけます。)*

**ポートフォリオダッシュボード** — E/S/G選好スライダー、最適化されたウェイト、パフォーマンス指標:

![Dashboard](docs/screenshot_dashboard.png)

**企業別詳細画面** — 5年ESGスコア推移、ローソク足チャート、ポートフォリオ加重ニュースワードクラウド:

![Company detail](docs/screenshot_detail.png)

## 🔬 研究ハイライト

卒業展示会の後、最適化モデルを改良し、フォローアップのバックテスト研究(2020–2024年、
t−1年のスコアでt年をリバランス — 先読みバイアスを除去)で検証しました。LLM-ESG最適化
ポートフォリオは**すべての指標で**ベンチマークを上回りました:

| ポートフォリオ | 5年累積リターン | CAGR | ボラティリティ | シャープ | 最大ドローダウン | カルマー |
|---|---|---|---|---|---|---|
| KOSPI | 1.092 | 0.018 | 0.202 | 0.190 | 0.357 | 0.051 |
| ESG ETF | 1.159 | 0.031 | 0.203 | 0.251 | 0.352 | 0.087 |
| 等ウェイト | 1.247 | 0.046 | 0.204 | 0.323 | 0.362 | 0.128 |
| **LEPOS (τ = 1.3)** | **1.377** | **0.068** | **0.154** | **0.503** | **0.187** | **0.362** |

この結果は韓国市場の現実的な取引コストを反映しても維持され(累積−1%pt)、同一ユニバースで
ESGビューなしに最適化した対照群とのアブレーションにより、ESGシグナルの寄与は**リスク低減**
であることが確認されました: 最大ドローダウン2.5分の1、シャープレシオ優位。

➡️ 方法論・頑健性検証・再現データの全体: [docs/research/RESEARCH.md](docs/research/RESEARCH.md)

## 🏗️ 仕組み

![Architecture](docs/architecture.png)

### 1. テキストデータパイプライン (研究 — `notebooks/`)

KOSPI 68社のニュース記事(2019–2023年、**138万件**)をNaverニュースから収集し、
前処理(企業名の標準化・匿名化、メタ情報の除去)を経て、OpenAI API(GPT-3.5-turbo)で
ラベルをブートストラップした後、**KoELECTRA**分類器のカスケードをファインチューニング
しました:

| モデル | タスク | 出力 | Accuracy / R² |
|---|---|---|---|
| A0 | 無関係記事のフィルタリング | 保持/除去 | 0.69 |
| A1 | 企業関連度 | 関連/無関係 | — |
| A2 | ESG関連度 | 関連/無関係 | 0.76 |
| A3 | ESGセンチメント | −1 / 0 / +1 | 0.73 |
| B1 | 評価機関別ESGスコア回帰 | 機関別 0.0–7.0 | R² 0.69 |
| C | E/S/G領域分類 | 領域別フラグ | 0.83 |

B1は記事テキストに定量ドメインデータ(温室効果ガス排出量、取締役会構成、財務指標)を
組み合わせて評価機関別に学習(5モデル)するため、各機関の評価傾向を再現でき、機関が
カバーしていない企業のスコアも算出できます。

ファインチューニング済みチェックポイントがあれば [`scripts/score_text.py`](scripts/score_text.py)
で分類器の推論を再現できます。

### 2. パーソナライズと最適化 (サービング — `src/`)

1. 15問のアンケートがユーザーの価値観を5つの評価機関の評価項目にマッピングし、
   領域×機関の選好行列を作ります (`src/scoring/survey.py`)。
2. 企業ごとのE/S/Gコンポーネントスコアとユーザー選好が**ブラック・リターマン**モデルの
   ビュー(P、Q)となり、投資スタイル(財務中心 ↔ ESG中心)がτを決定します。
   共分散には**Ledoit-Wolf縮小推定**を使用します (`src/optimization/black_litterman.py`)。
3. シャープレシオを最大化するウェイト(ロングオンリー、全額投資)を算出し、Streamlit
   ダッシュボードで企業別詳細(5年ESG推移、ローソク足チャート、ニュースワードクラウド)と
   PDF/HTMLレポートのエクスポートを提供します。

## 📁 プロジェクト構成

```text
LLM_ESG_POS/
├── app/                     # Streamlit UI (st.navigationマルチページ)
│   ├── main.py              #   エントリーポイント
│   └── pages/               #   ホーム、アンケート、ダッシュボード、ニュース、ESG紹介
├── src/                     # ビジネスロジック (型ヒント・docstring付き、UI非依存)
│   ├── config.py            #   pathlibベースのパス・定数
│   ├── data/                #   ESGテーブル読込、市場データ (FinanceDataReader)
│   ├── scoring/             #   アンケートスコアリング行列
│   ├── optimization/        #   ブラック・リターマン + シャープ最大化
│   ├── collection/          #   Naverニュースクローラー
│   ├── visualization/       #   ポートフォリオ加重ワードクラウド
│   └── reporting/           #   HTML/PDFレポートエクスポート
├── notebooks/               # 研究パイプライン (01 収集 → 06 最適化)
├── data/
│   ├── processed/           # ESGスコアテーブル(2019–2023)、企業プロフィール
│   ├── dummy/               # 実験用サンプルデータ
│   └── user/                # ランタイム状態 (gitignore)
├── docs/                    # アーキテクチャ、最終報告書、発表資料
│   └── research/            # バックテスト研究 (RESEARCH.md + データ + チャート)
├── scripts/                 # スタンドアロンツール (KoELECTRA推論デモなど)
├── tests/                   # pytestテスト (スコアリング/読込/最適化)
└── requirements.txt
```

## 🚀 実行方法

```bash
git clone https://github.com/fairyofdata/LLM_ESG_POS.git
cd LLM_ESG_POS
pip install -r requirements.txt
streamlit run app/main.py
```

アンケートを完了すると、ダッシュボードが初回のみKRXの5年分の株価をダウンロードし
(約1分)、パーソナライズされたポートフォリオを表示します。

テストの実行:

```bash
pytest
```

**オプションのシステム依存関係** (なくても該当機能のみ無効化されます):

| 依存関係 | 機能 |
|---|---|
| [wkhtmltopdf](https://wkhtmltopdf.org/) | PDFレポートエクスポート (HTMLエクスポートは常に可能) |
| Chrome / Chromium | 「最新ニュース」ライブクローリングページ |
| 韓国語フォント | ワードクラウドのレンダリング |

## 📚 ドキュメント

- [最終報告書 (韓国語, PDF)](docs/final_report_kr.pdf) — 全28ページ
- [卒業展示会発表資料 (韓国語, PPTX)](docs/presentation_kr.pptx)
- [バックテスト研究](docs/research/RESEARCH.md)
- システム構成図: [docs/system_diagram_kr.png](docs/system_diagram_kr.png)

## 🔭 今後の拡張

1. **カバレッジ拡大**: スタートアップ・非上場企業を含む約1,000社へのスコアリング拡張(B2モデル)。
2. **リアルタイムスコアリング**: リアルタイムのニュースストリームに基づくESGスコア更新。
3. **最適化制約の強化**: セクター別ウェイト上限、回転率制限。
4. **機関別信頼ウェイト**: 研究段階のP/Q設計(機関単位のビュー + ユーザー信頼ベクトル)の
   アプリへの反映。

## 👥 About

**光云大学 第8回産学連携SWプロジェクト & SW融合大学卒業制作**

- **Team KWargs**: ペク・ジホン (PM · 前処理、NLPモデルA0/C、最適化アルゴリズム)、
  キム・ナヨン (FE · 収集パイプライン、Streamlit UI)、チャン・ハンジェ (BE ·
  ラベリングパイプライン、NLPモデルA2/A3/B1)
- **指導教授**: チョ・ミンス教授 (光云大学 情報融合学部)
- **連携企業**: Billions Lab (ピョ・スジン博士)
- **SW登録番号**: C-2024-042035

## 📄 ライセンス

MIT — [LICENSE](LICENSE) を参照。
