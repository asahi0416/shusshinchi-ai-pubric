# AI顔画像 出身地＆ルーツ判定システム (Japan Face Roots AI)

AIが顔写真の骨格・目鼻立ちの特徴から、日本列島の地域遺伝的ルーツ（47都道府県・6地方ブロック・縄文系/弥生系）を立体解析して判定するWebアプリケーションです。単独の写真はもちろん、卒業アルバムや部活動などの集合写真を使うことで個人差が相殺され、より高精度に判定できます。

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 主な機能

### 1. グループ・集合写真診断（高精度・推奨）
- **ベイズ対数尤度比統合 (LLR)**:
  単一写真では「表情・メイク・角度・個体差ノイズ」が地域シグナルの約5〜12倍存在します。同郷（同じ地元・学校・部活など）の仲間（3〜8人以上）の集合写真を入力すると、個人差ノイズが統計的に相殺（$1/\sqrt{N}$）され、**真の地域遺伝ルーツだけが増幅されて都道府県Top-1精度50%以上・地方80%以上の高精度**を達成します。
- **全顔自動検出・ナンバリング**:
  画像内の全員の顔を左から順に自動検出（`#1`〜`#N`）し、個別のサムネイル・診断結果とグループ統合結果を同時に可視化します。

### 2. 個人診断 (1名)
- 正面を向いた単独の顔写真（ファイルアップロードまたはWebカメラ）から、瞬時にルーツを解析。
- **推定地域（6地方ブロック）**: 確率グラフで視覚的に表示。
- **47都道府県ランキング**: 最有力県から全47都道府県の推定確率を算出（Top 5 ランキング）。

### 3. 縄文系 vs 弥生系 顔タイプ分析
- 3D68点解剖骨格マーカー（彫りの深さ、鼻幅比率、目眉間距離、輪郭縦横比）のZ-score合成により、
- 「縄文系（濃い／彫深）スコア %」vs「弥生系（涼やか／面長）スコア %」を客観的正規分布で診断・解説します。

### 4. SNSシェア機能
- 診断結果をワンクリックでポストできる「X (旧Twitter) でシェア」ボタンおよび、そのままコピーして使える共有用テキスト生成機能を備えています。

### 5. プライバシー保護
- アップロードされた画像および解析データは、セキュアな暗号化通信により厳重に管理されます。

---

## 🔬 技術アーキテクチャ

```text
[入力画像]
   │
   ├── (1) 顔検出 & アライメント (InsightFace / MTCNN)
   │
   ├── (2) マルチモーダル特徴抽出
   │       ├── 3D解剖学的骨格マーカー (68点立体座標 / Z軸彫深さ)
   │       ├── 空間幾何特徴量 (目鼻口の比率・距離行列)
   │       ├── FaceNet 512d 埋め込み
   │       └── ArcFace 空間特徴量 (性別不変直交補空間射影)
   │
   ├── (3) 事前確率補正ベイズ対数尤度比統合 (LLR)
   │       log P(県 | 全員) ∝ log P_prior(県) + Σ [log P_i(県) - log P_mean(県)]
   │
   └── (4) 出力
           ├── 6地方ブロック推定確率 (同期Max-Pooling)
           ├── 47都道府県確率ランキング (温度キャリブレーション T=0.50)
           └── 縄文系・弥生系 形質比率スコアリング
```

---

## 🚀 インストールと起動手順

### 前提条件
- Python 3.10 以上
- （推奨）CUDA 対応 NVIDIA GPU（CPU環境でも動作します）

### 1. リポジトリのクローン
```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
```

### 2. 依存パッケージのインストール
```bash
pip install -r requirements.txt
```

> **Note (GPU利用時)**:
> PyTorch および ONNX Runtime を CUDA 対応版でインストールすると、RTX系GPUで高速に推論可能です。

### 3. アプリケーションの起動
#### Windowsの場合:
`run_app.bat` をダブルクリックするだけで起動します。

#### コマンドラインの場合:
```bash
streamlit run app.py
```
起動後、ブラウザで `http://localhost:8501` を開いてください。

---

## 📁 フォルダ構成

```text
├── app.py                      # Streamlit WebアプリケーションUI
├── predictor.py                # 3D骨格・マルチモーダル・ベイズLLR推論エンジン
├── face_processor.py           # 顔検出・アライメント・FaceNet特徴抽出
├── regions.py                  # 47都道府県および6地方区分の定義辞書
├── requirements.txt            # 必要Pythonパッケージ一覧
├── run_app.bat                 # Windowsワンクリック起動スクリプト
├── README.md                   # 本ドキュメント
├── .gitignore                  # Git除外設定
├── .streamlit/
│   └── config.toml             # Streamlit設定（Deployボタン等非表示）
├── data/
│   └── dataset.csv             # 統計メタデータ
└── models/
    ├── multimodal_arcface_lm.pkl       # マルチモーダル主モデル (7.0 MB)
    ├── face_origin_pure_morph_svm.pkl  # 純粋骨格モデル (312 KB)
    ├── face_origin_model.pkl           # 総合判定モデル (13.7 MB)
    └── model_approach4_composite.pkl   # 複合モデル (13.7 MB)
```

---

## 📄 ライセンス

本プロジェクトは [MIT License](LICENSE) のもとで公開されています。
学習済みモデルおよびコードは研究・教育・エンターテインメント目的で自由にご利用いただけます。
