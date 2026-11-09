"""
出身地判定AI - Webアプリケーション (Streamlit / 一般公開版)
顔画像から「地方ブロック」「47都道府県ランキング」「縄文/弥生顔タイプ」を判定・可視化
"""

import sys
import urllib.parse
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

import hashlib
# モジュールのインポート
from predictor import FaceOriginPredictor
from regions import PREFECTURES, REGIONS
from uploader import upload_diagnosis_async

# ページ設定
st.set_page_config(
    page_title="AI顔貌 出身地＆ルーツ判定 | 日本列島の地域遺伝形質をAIで診断",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ----------------- 管理者モード（シークレットURL対応） -----------------
# 例: https://shusshinchi-ai.streamlit.app/?page=admin または ?admin=1
query_params = st.query_params
if query_params.get("page") == "admin" or query_params.get("admin") in ["1", "true"]:
    admin_html_path = Path(__file__).parent / "admin.html"
    if admin_html_path.exists():
        with open(admin_html_path, "r", encoding="utf-8") as f:
            admin_code = f.read()
        import streamlit.components.v1 as components
        st.markdown("""
        <style>
            header[data-testid="stHeader"], footer, #MainMenu { display: none !important; }
            .block-container { padding: 0 !important; max-width: 100% !important; }
            iframe { width: 100% !important; min-height: 98vh !important; height: 100% !important; border: none !important; }
        </style>
        """, unsafe_allow_html=True)
        components.html(admin_code, height=980, scrolling=True)
        st.stop()

# カスタムCSS（洗練されたモダン＆和風テイスト）
st.markdown("""
<style>
    /* Streamlit標準UI要素の完全消去 */
    header[data-testid="stHeader"], footer, #MainMenu, .stDeployButton, [data-testid="stDeployButton"], [data-testid="stToolbar"], [data-testid="stToolbarActions"], [data-testid="stStatusWidget"], [data-testid="stDecoration"] {
        display: none !important;
        visibility: hidden !important;
        height: 0px !important;
    }

    /* Google Fonts: Zen Maru Gothic (丸みのある可愛いフォント) */
    @import url('https://fonts.googleapis.com/css2?family=Zen+Maru+Gothic:wght@400;500;700;900&family=Noto+Sans+JP:wght@400;700&display=swap');

    /* 全体背景: ふんわり桜色＆ミルキーパステル */
    html, body, [class*="css"], .stApp {
        font-family: 'Zen Maru Gothic', sans-serif !important;
        background: linear-gradient(135deg, #FFF7F9 0%, #F8FAFC 50%, #FFF9F5 100%) !important;
        color: #1E293B;
    }

    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 3.5rem !important;
        max-width: 1060px !important;
    }

    /* ── 可愛いトップヘッダー ── */
    .hero-container {
        background: linear-gradient(135deg, #FFFFFF 0%, #FFF5F7 60%, #FFF8F0 100%);
        border: 2px solid #FFE4E9;
        border-radius: 24px;
        padding: 32px 30px;
        text-align: center;
        margin-bottom: 20px;
        box-shadow: 0 10px 30px rgba(255, 117, 140, 0.12);
        position: relative;
        overflow: hidden;
    }
    .hero-badge {
        display: inline-block;
        background: linear-gradient(135deg, #FF7EB3 0%, #FF758C 100%);
        color: #FFFFFF;
        padding: 5px 16px;
        border-radius: 999px;
        font-size: 0.82rem;
        font-weight: 700;
        letter-spacing: 0.05em;
        margin-bottom: 10px;
        box-shadow: 0 4px 12px rgba(255, 117, 140, 0.3);
    }
    .hero-title {
        font-size: clamp(1.8rem, 4.5vw, 2.4rem);
        font-weight: 900;
        margin: 0 0 10px 0;
        background: linear-gradient(90deg, #FF6584, #FF7EB3, #FF9A76);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        letter-spacing: -0.01em;
    }
    .hero-subtitle {
        font-size: 0.98rem;
        color: #475569;
        line-height: 1.7;
        font-weight: 500;
        margin: 0 auto;
        max-width: 520px;
    }

    /* ── 卒アル・おすすめバナー ── */
    .recommend-banner {
        background: linear-gradient(135deg, #FFF0F5 0%, #FFF9EB 100%);
        border: 2px dashed #FFCCD5;
        border-radius: 18px;
        padding: 14px 20px;
        margin: 8px 0 22px 0;
        display: flex;
        align-items: center;
        gap: 12px;
        box-shadow: 0 4px 14px rgba(255, 154, 118, 0.08);
    }
    .recommend-banner-icon { font-size: 1.8rem; flex-shrink: 0; }
    .recommend-banner-text {
        font-size: 0.92rem;
        color: #9D174D;
        line-height: 1.55;
        font-weight: 700;
    }
    .recommend-banner-sub {
        font-size: 0.82rem;
        color: #B45309;
        font-weight: 500;
    }

    /* ── 結果発表カード (超可愛いピンクゴールド) ── */
    .result-card-main {
        background: linear-gradient(135deg, #FFFFFF 0%, #FFF5F7 60%, #FFF9F3 100%);
        border: 2.5px solid #FFB6C1;
        border-radius: 22px;
        padding: 24px;
        margin-bottom: 20px;
        box-shadow: 0 10px 30px rgba(255, 117, 140, 0.16);
        position: relative;
        text-align: center;
    }
    .result-card-badge {
        display: inline-block;
        background: #FFF0F3;
        color: #E11D48;
        border: 1px solid #FECDD3;
        padding: 4px 14px;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 700;
        margin-bottom: 8px;
    }
    .result-card-pref {
        font-size: clamp(1.4rem, 3.5vw, 1.85rem);
        font-weight: 900;
        color: #1E293B;
        margin: 6px 0 10px 0;
    }
    .pref-highlight {
        color: #E11D48;
        background: linear-gradient(transparent 65%, #FFE4E6 65%);
        padding: 0 4px;
    }
    .result-card-region {
        font-size: 1.05rem;
        font-weight: 700;
        color: #475569;
        margin-bottom: 12px;
    }
    .result-card-desc {
        font-size: 0.88rem;
        color: #64748B;
        line-height: 1.6;
        background: rgba(255, 255, 255, 0.85);
        border: 1px solid #FFE4E9;
        padding: 12px 16px;
        border-radius: 14px;
        text-align: left;
    }

    /* ── 可愛いハッシュタグ ── */
    .tag-container {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin-top: 10px;
        justify-content: center;
    }
    .tag-pill {
        background: rgba(255, 255, 255, 0.9);
        border: 1px solid #FFB6C1;
        border-radius: 999px;
        padding: 4px 12px;
        font-size: 0.78rem;
        font-weight: 700;
        color: #E11D48;
        box-shadow: 0 2px 6px rgba(255, 117, 140, 0.1);
    }

    /* ── スタイリッシュな順位サークルバッジ (環境依存絵文字メダルの完全代替) ── */
    .rank-num {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 26px;
        height: 26px;
        border-radius: 50%;
        font-size: 0.82rem;
        font-weight: 800;
        margin-right: 10px;
        background: #F1F5F9;
        color: #64748B;
        flex-shrink: 0;
    }
    .rank-num-1 { background: #FFE4E6; color: #E11D48; border: 1.5px solid #FDA4AF; }
    .rank-num-2 { background: #F1F5F9; color: #334155; border: 1.5px solid #CBD5E1; }
    .rank-num-3 { background: #FFEDD5; color: #C2410C; border: 1.5px solid #FDBA74; }

    .recommend-badge {
        display: inline-block;
        background: #FF758C;
        color: #FFFFFF;
        font-size: 0.72rem;
        font-weight: 800;
        padding: 3px 8px;
        border-radius: 6px;
        letter-spacing: 0.04em;
        flex-shrink: 0;
    }

    /* ── ランキングカード ── */
    .rank-item {
        background: #FFFFFF;
        border: 1.5px solid #F1F5F9;
        padding: 12px 18px;
        border-radius: 14px;
        margin-bottom: 10px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        transition: transform 0.2s, box-shadow 0.2s;
        box-shadow: 0 2px 8px rgba(0,0,0,0.02);
    }
    .rank-item:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(255, 117, 140, 0.12);
        border-color: #FFCCD5;
    }
    .rank-item-1 {
        background: linear-gradient(135deg, #FFF1F2 0%, #FFFFFF 100%);
        border: 2px solid #FDA4AF;
    }
    .rank-item-2 {
        background: linear-gradient(135deg, #F8FAFC 0%, #FFFFFF 100%);
        border: 2px solid #E2E8F0;
    }
    .rank-item-3 {
        background: linear-gradient(135deg, #FFF7ED 0%, #FFFFFF 100%);
        border: 2px solid #FDBA74;
    }

    /* ── プログレスバー (ピンク〜オレンジのグラデーション) ── */
    .stProgress > div > div > div > div {
        background-image: linear-gradient(to right, #FF7EB3, #FFA07A) !important;
        border-radius: 999px !important;
    }

    /* ── ボタン類のぷっくり角丸化 ── */
    .stButton > button {
        border-radius: 999px !important;
        font-family: 'Zen Maru Gothic', sans-serif !important;
        font-weight: 700 !important;
        border: none !important;
        transition: all 0.2s ease !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 18px rgba(255, 117, 140, 0.25) !important;
    }

    /* ── スマートフォン・モバイル向け完全レスポンシブ最適化 ── */
    @media (max-width: 768px) {
        .block-container {
            padding-top: 1rem !important;
            padding-bottom: 2rem !important;
            padding-left: 0.75rem !important;
            padding-right: 0.75rem !important;
        }
        .hero-container {
            padding: 18px 14px !important;
            border-radius: 18px !important;
            margin-bottom: 14px !important;
        }
        .hero-title {
            font-size: 1.55rem !important;
        }
        .hero-subtitle {
            font-size: 0.85rem !important;
            line-height: 1.55 !important;
        }
        /* ラジオボタンをスマホで縦並び＆100%幅にして画面外はみ出し・横スクロールを完全防止 */
        div[role="radiogroup"] {
            display: flex !important;
            flex-direction: column !important;
            width: 100% !important;
            gap: 8px !important;
            padding: 10px 12px !important;
            box-sizing: border-box !important;
        }
        div[role="radiogroup"] > label {
            margin-bottom: 0px !important;
            width: 100% !important;
        }
        /* おすすめバナー */
        .recommend-banner {
            padding: 12px 14px !important;
            gap: 10px !important;
            border-radius: 14px !important;
            margin-bottom: 16px !important;
        }
        .recommend-banner-text {
            font-size: 0.85rem !important;
            line-height: 1.45 !important;
        }
        .recommend-banner-sub {
            font-size: 0.76rem !important;
            line-height: 1.4 !important;
        }
        /* 結果発表カード */
        .result-card-main {
            padding: 18px 12px !important;
            border-radius: 18px !important;
            margin-bottom: 16px !important;
        }
        .result-card-pref {
            font-size: 1.45rem !important;
        }
        .result-card-desc {
            font-size: 0.82rem !important;
            padding: 10px 12px !important;
        }
        /* ランキングカード */
        .rank-item {
            padding: 10px 12px !important;
        }
        .rank-item > div:first-child span {
            font-size: 0.95rem !important;
        }
    }

    /* ── ラジオボタン ── */
    div[role="radiogroup"] {
        background: rgba(255, 255, 255, 0.7);
        border: 1px solid #FFE4E9;
        border-radius: 16px;
        padding: 8px 12px;
        display: inline-flex;
    }

</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_predictor():
    """推論エンジンのロード"""
    try:
        return FaceOriginPredictor()
    except Exception as e:
        st.error(f"モデルのロードに失敗しました: {e}")
        return None

predictor = load_predictor()

# ----------------- サイドバー（一般公開用） -----------------
with st.sidebar:
    st.header("アプリについて")
    st.markdown("""
    本アプリは、最先端の**3D顔面解剖骨格解析**と**マルチモーダル深層学習**を用い、顔写真から日本列島の地域遺伝的ルーツ（地方・都道府県・縄文/弥生）を推定するWebアプリケーションです。

    ---
    ### 科学的アプローチ
    - **3D68点解剖骨格解析**:
      目・鼻・顎・頬骨の彫りの深さや比率を立体座標化。
    - **縄文 vs 弥生 形質スコアリング**:
      濃顔・彫深（縄文系）と涼やか・面長（弥生系）のバランスを客観的正規分布で算出。
    - **集合写真ベイズ尤度比統合 (LLR) 【高精度の鍵】**:
      単一の顔写真では「表情・角度・個体差ノイズ」が地域差シグナルの約5〜12倍存在します。
      同郷の複数人（3〜8人以上）の骨格をベイズ対数尤度比で統合することで、
      個人差ノイズが統計的に相殺（1/√Nに消去）され、地方・都道府県の的中率が飛躍的に高まります。

    ---
    ### データ取り扱いについて
    アップロードされた顔画像および診断結果は、AIの精度向上・学術分析目的でのみ安全に暗号化通信により管理・蓄積されます。第三者への開示・商用目的での利用は行われません。
    """)


# ----------------- 全画面ドラッグ＆ドロップ (Global Drag & Drop) -----------------
import streamlit.components.v1 as components
components.html("""
<script>
(function() {
  const pDoc = window.parent.document;
  const pWin = window.parent;
  if (!pDoc || pDoc.__global_drop_ready) return;
  pDoc.__global_drop_ready = true;

  // 画面全体のドロップ案内オーバーレイ
  let overlay = pDoc.getElementById('global-drag-overlay');
  if (!overlay) {
    overlay = pDoc.createElement('div');
    overlay.id = 'global-drag-overlay';
    overlay.style.cssText = `
      position: fixed; top: 0; left: 0; width: 100vw; height: 100vh; z-index: 9999999;
      background: linear-gradient(135deg, rgba(255, 101, 132, 0.92), rgba(255, 126, 179, 0.92)); backdrop-filter: blur(6px);
      display: none; align-items: center; justify-content: center;
      color: #ffffff; font-family: 'Zen Maru Gothic', sans-serif;
      pointer-events: none; transition: opacity 0.2s ease; opacity: 0;
    `;
    overlay.innerHTML = `
      <div style="text-align: center; border: 3px dashed rgba(255,255,255,0.95); border-radius: 24px; padding: 36px 50px; background: rgba(255,255,255,0.18); box-shadow: 0 20px 50px rgba(255, 101, 132, 0.35);">
        <div style="display: inline-flex; align-items: center; justify-content: center; width: 64px; height: 64px; border-radius: 50%; background: rgba(255, 255, 255, 0.9); color: #FF6584; font-size: 1.2rem; font-weight: 900; margin-bottom: 14px; box-shadow: 0 4px 12px rgba(255, 101, 132, 0.2);">DROP</div>
        <div style="font-size: 1.6rem; font-weight: 900; margin-bottom: 8px; letter-spacing: 0.04em; color: #FFFFFF;">画面のどこでもドロップOK</div>
        <div style="font-size: 0.95rem; opacity: 0.95; font-weight: 600; color: #FFFFFF;">ファイルを離すと自動で診断を開始します</div>
      </div>
    `;
    pDoc.body.appendChild(overlay);
  }

  let dragCounter = 0;

  pWin.addEventListener('dragenter', (e) => {
    e.preventDefault();
    dragCounter++;
    if (e.dataTransfer && e.dataTransfer.types && Array.from(e.dataTransfer.types).includes('Files')) {
      overlay.style.display = 'flex';
      requestAnimationFrame(() => { overlay.style.opacity = '1'; });
    }
  });

  pWin.addEventListener('dragover', (e) => {
    e.preventDefault();
    if (e.dataTransfer) {
      e.dataTransfer.dropEffect = 'copy';
    }
  });

  pWin.addEventListener('dragleave', (e) => {
    e.preventDefault();
    dragCounter--;
    if (dragCounter <= 0) {
      dragCounter = 0;
      overlay.style.opacity = '0';
      setTimeout(() => { if (dragCounter === 0) overlay.style.display = 'none'; }, 200);
    }
  });

  pWin.addEventListener('drop', (e) => {
    e.preventDefault();
    dragCounter = 0;
    overlay.style.opacity = '0';
    setTimeout(() => { overlay.style.display = 'none'; }, 200);

    const files = e.dataTransfer ? e.dataTransfer.files : null;
    if (!files || files.length === 0) return;

    // Streamlit 内の file input を取得
    const fileInputs = pDoc.querySelectorAll('input[type="file"]');
    if (!fileInputs || fileInputs.length === 0) return;

    // 表示中の input[type="file"] を優先取得
    let targetInput = null;
    for (const inp of fileInputs) {
      if (inp.offsetParent !== null) {
        targetInput = inp;
        break;
      }
    }
    if (!targetInput) targetInput = fileInputs[0];

    try {
      const dt = new DataTransfer();
      for (let i = 0; i < files.length; i++) {
        dt.items.add(files[i]);
      }
      
      const descriptor = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'files');
      if (descriptor && descriptor.set) {
        descriptor.set.call(targetInput, dt.files);
      } else {
        targetInput.files = dt.files;
      }

      targetInput.dispatchEvent(new Event('change', { bubbles: true }));
      targetInput.dispatchEvent(new Event('input', { bubbles: true }));
    } catch (err) {
      console.error('Global drag & drop error:', err);
    }
  });
})();
</script>
""", height=0, width=0)

# ----------------- メイン画面ヘッダー -----------------
st.markdown("""
<div class="hero-container">
    <div class="hero-badge">AI FACE ROOTS</div>
    <div class="hero-title">AI出身地・ルーツ診断</div>
    <div class="hero-subtitle">
        あなたの顔、どこの都道府県の系統に近い？<br>
        全国47都道府県でいちばん顔の骨格特徴が似ている地域をAIが判定します。
    </div>
</div>
""", unsafe_allow_html=True)

# 診断モード選択
diag_mode = st.radio(
    "診断モードを選択",
    [
        "ひとり写真で診断 (1名)",
        "集合写真で診断 (複数名 / おすすめ)"
    ],
    horizontal=True
)

# 集合写真おすすめバナー
st.markdown("""
<div class="recommend-banner">
    <div class="recommend-badge">RECOMMEND</div>
    <div>
        <div class="recommend-banner-text">卒業アルバムやクラス写真、部活動などの集合写真で診断すると精度UP</div>
        <div class="recommend-banner-sub">同じ学校や地元の仲間（3名以上）の写真を使うと、個人差が相殺されてより確かな地域傾向がわかります。</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ----------------------------------------------------
# A. 個人診断モード (1名)
# ----------------------------------------------------
if diag_mode == "ひとり写真で診断 (1名)":
    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.subheader("写真の選択")
        input_method = st.radio("入力方法を選択", ["ファイルからアップロード", "Webカメラで撮影"], horizontal=True)
        
        image_input = None
        raw_bytes = None
        filename = None
        if input_method == "ファイルからアップロード":
            uploaded_file = st.file_uploader("正面を向いた顔写真を選択してください (JPG, PNG, WebP)", type=["jpg", "jpeg", "png", "webp"])
            if uploaded_file is not None:
                raw_bytes = uploaded_file.getvalue()
                filename = getattr(uploaded_file, "name", "uploaded.jpg")
                image_input = Image.open(uploaded_file).convert("RGB")
        else:
            camera_file = st.camera_input("カメラで撮影")
            if camera_file is not None:
                raw_bytes = camera_file.getvalue()
                filename = "camera.jpg"
                image_input = Image.open(camera_file).convert("RGB")

        if image_input:
            st.image(image_input, caption="入力された写真", use_container_width=True)
            
        st.markdown("""
        <div style="background: linear-gradient(135deg, #FFF0F5 0%, #FFF8F0 100%); border: 1.5px solid #FFCCD5; border-radius: 14px; padding: 12px 16px; margin-top: 14px;">
            <div style="font-size: 0.85rem; color: #BE185D; line-height: 1.5; font-weight: 700;">
                <b>より高精度に診断したい場合</b><br>
                <span style="font-weight: 500; color: #831843;">卒業アルバムや部活動などの複数人で診断する「集合写真モード」がおすすめです。</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.subheader("診断結果")
        if image_input is None:
            st.info("左側のエリアから写真をアップロードまたは撮影してください。")
        else:
            if predictor is None:
                st.error("推論エンジンのロードに失敗しました。")
            else:
                with st.spinner("AIが顔の骨格・目鼻立ちを立体解析中..."):
                    result = predictor.predict(image_input)

                if not result["success"]:
                    st.warning(result["error"])
                else:
                    # 検出オーバーレイ画像作成
                    box = result["face_box"]
                    landmarks = result["landmarks"]
                    draw_img = image_input.copy()
                    draw = ImageDraw.Draw(draw_img)
                    draw.rectangle(box, outline="#2563EB", width=4)
                    for pt in landmarks:
                        draw.ellipse([pt[0]-4, pt[1]-4, pt[0]+4, pt[1]+4], fill="#EF4444", outline="#FFFFFF")

                    top_region = result["top_regions"][0]
                    top_pref = result["top_prefectures"][0]
                    jy = result["jomon_yayoi"]

                    # メイン判定カード
                    st.markdown(f"""
                    <div class="result-card-main">
                        <div class="result-card-badge">DIAGNOSIS RESULT</div>
                        <div class="result-card-pref">
                            あなたの顔は <span class="pref-highlight">【{top_pref['prefecture']}】</span> の系統
                        </div>
                        <div class="result-card-region">
                            推定エリア: <b>{top_region['region']}</b> <span style="font-size: 0.95rem; color: #FF6584;">({top_region['percentage']}%)</span>
                        </div>
                        <div class="result-card-desc">{top_region.get('description', '')}</div>
                    </div>
                    """, unsafe_allow_html=True)

                    # 縄文系 vs 弥生系 分析
                    st.markdown("#### 顔タイプ分析（縄文系 / 弥生系）")
                    c_jy1, c_jy2 = st.columns([1, 1])
                    with c_jy1:
                        st.metric("縄文系スコア (濃いめ/彫深)", f"{jy['jomon_percent']}%")
                    with c_jy2:
                        st.metric("弥生系スコア (すっきり/涼やか)", f"{jy['yayoi_percent']}%")

                    st.progress(jy["jomon_percent"] / 100.0)
                    st.caption(f"**判定**: **{jy['face_type']}** | {jy['description']}")

                    # 可愛い特徴タグ
                    tags_html = ""
                    if jy["jomon_percent"] >= 50:
                        tags_html = '<div class="tag-container"><span class="tag-pill">#ぱっちり目</span><span class="tag-pill">#立体骨格</span><span class="tag-pill">#濃いめエモ顔</span><span class="tag-pill">#縄文ルーツ</span></div>'
                    else:
                        tags_html = '<div class="tag-container"><span class="tag-pill">#透明感</span><span class="tag-pill">#涼やか美人</span><span class="tag-pill">#すっきり小顔</span><span class="tag-pill">#弥生ルーツ</span></div>'
                    st.markdown(tags_html, unsafe_allow_html=True)

                    # Supabase Storage への非同期保存
                    if raw_bytes:
                        f_hash = hashlib.md5(raw_bytes).hexdigest()
                        if st.session_state.get("last_uploaded_single") != f_hash:
                            summary = {
                                "region": top_region['region'],
                                "region_percentage": top_region['percentage'],
                                "top_pref": top_pref['prefecture'],
                                "top_pref_percentage": round(top_pref['percentage'], 2),
                                "top_prefectures": result["top_prefectures"][:5],
                                "jomon_percent": jy['jomon_percent'],
                                "yayoi_percent": jy['yayoi_percent'],
                                "face_type": jy['face_type']
                            }
                            upload_diagnosis_async(raw_bytes, filename, summary, mode="single")
                            st.session_state["last_uploaded_single"] = f_hash

    # 下部セクション: 地方グラフ & 都道府県ランキング & SNSシェア
    if image_input and predictor and 'result' in locals() and result["success"]:
        st.divider()
        res_col1, res_col2 = st.columns([1, 1], gap="large")

        with res_col1:
            st.markdown("### 地方ブロック判定（6区分）")
            df_region = pd.DataFrame(result["top_regions"])
            fig_region = px.bar(
                df_region,
                x="percentage",
                y="region",
                orientation="h",
                text="percentage",
                color="percentage",
                color_continuous_scale="Sunset",
                labels={"percentage": "確率 (%)", "region": "地方ブロック"},
                title="地方ブロックごとの推定確率"
            )
            fig_region.update_layout(yaxis=dict(autorange="reversed"), showlegend=False, height=360)
            fig_region.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
            st.plotly_chart(fig_region, use_container_width=True)

        with res_col2:
            st.markdown("### 47都道府県 推定確率ランキング (Top 5)")
            top5 = result["top_prefectures"][:5]

            for idx, p in enumerate(top5):
                rank_num = idx + 1
                badge_class = f"rank-num rank-num-{rank_num}" if rank_num <= 3 else "rank-num"
                r_class = f"rank-item rank-item-{rank_num}" if rank_num <= 3 else "rank-item"
                st.markdown(f"""
                <div class="{r_class}">
                    <div style="display: flex; align-items: center;">
                        <span class="{badge_class}">{rank_num}</span>
                        <span style="font-weight:700; color:#1E293B; font-size:1.05rem;">{p['prefecture']}</span>
                        <span style="color:#64748B; font-size:0.82rem; margin-left:8px;">({p['region']})</span>
                    </div>
                    <div style="font-weight:800; color:#E11D48; font-size:1.05rem;">
                        {p['percentage']:.2f}%
                    </div>
                </div>
                """, unsafe_allow_html=True)

            with st.expander("全47都道府県の推定確率をすべて見る"):
                df_all = pd.DataFrame(result["top_prefectures"])[["rank", "prefecture", "region", "percentage"]]
                df_all.columns = ["順位", "都道府県", "地方", "推定確率 (%)"]
                st.dataframe(df_all, use_container_width=True, hide_index=True)

        # SNSシェアセクション
        st.divider()
        st.markdown("### 診断結果をシェアする")
        share_text = f"【AI顔画像 出身地＆ルーツ判定結果】\n推定エリア: 【{top_region['region']}】 ({top_region['percentage']}%)\n最有力都道府県: 【{top_pref['prefecture']}】 ({top_pref['percentage']:.1f}%)\n顔タイプ: {jy['face_type']} (縄文 {jy['jomon_percent']}% / 弥生 {jy['yayoi_percent']}%)\n#出身地判定AI #AI顔診断 #ルーツ診断 #縄文顔 #弥生顔"
        encoded_text = urllib.parse.quote(share_text)
        x_url = f"https://twitter.com/intent/tweet?text={encoded_text}"

        s_col1, s_col2 = st.columns([1, 2])
        with s_col1:
            st.link_button("X (旧Twitter) で結果をポスト", url=x_url, use_container_width=True, type="primary")
        with s_col2:
            st.text_area("シェア用テキスト（コピーして使えます）", share_text, height=105)

# ----------------------------------------------------
# B. グループ・集合写真診断モード (複数名)
# ----------------------------------------------------
else:
    st.markdown("""
    <div class="recommend-banner">
        <div class="recommend-badge">GROUP</div>
        <div>
            <div class="recommend-banner-text">集合写真・グループ診断モード</div>
            <div class="recommend-banner-sub">卒業アルバムやクラス写真から全員の顔を自動解析。人数が多いほど的中精度が向上します。</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    g_col1, g_col2 = st.columns([1, 1], gap="large")

    with g_col1:
        st.subheader("集合写真の選択")
        g_input_method = st.radio("入力方法を選択", ["ファイルからアップロード", "Webカメラで撮影"], horizontal=True, key="group_input_method")
        
        g_image_input = None
        g_raw_bytes = None
        g_filename = None
        if g_input_method == "ファイルからアップロード":
            g_uploaded_file = st.file_uploader("集合写真・複数人の顔写真を選択 (JPG, PNG, WebP)", type=["jpg", "jpeg", "png", "webp"], key="group_file_uploader")
            if g_uploaded_file is not None:
                g_raw_bytes = g_uploaded_file.getvalue()
                g_filename = getattr(g_uploaded_file, "name", "group_upload.jpg")
                g_image_input = Image.open(g_uploaded_file).convert("RGB")
        else:
            g_camera_file = st.camera_input("カメラで撮影", key="group_camera_input")
            if g_camera_file is not None:
                g_raw_bytes = g_camera_file.getvalue()
                g_filename = "group_camera.jpg"
                g_image_input = Image.open(g_camera_file).convert("RGB")

        if g_image_input:
            st.image(g_image_input, caption="入力された集合写真", use_container_width=True)

    with g_col2:
        st.subheader("グループ統合 判定結果")
        if g_image_input is None:
            st.info("左側のエリアから集合写真をアップロードまたは撮影してください。")
        else:
            if predictor is None:
                st.error("推論エンジンのロードに失敗しました。")
            else:
                with st.spinner("写真から全員の顔を検出中... ベイズ尤度比を統合解析しています..."):
                    g_result = predictor.predict_group(g_image_input)

                if not g_result["success"]:
                    st.warning(g_result["error"])
                else:
                    g_top_region = g_result["group_top_regions"][0]
                    g_top_pref = g_result["group_top_prefectures"][0]

                    st.markdown(f"""
                    <div class="result-card-main">
                        <div class="result-card-badge">GROUP RESULT ({g_result['num_faces']}名統合)</div>
                        <div class="result-card-pref">
                            グループ全体の傾向は <span class="pref-highlight">【{g_top_pref['prefecture']}】</span> タイプ
                        </div>
                        <div class="result-card-region">
                            推定エリア: <b>{g_top_region['region']}</b> <span style="font-size: 0.95rem; color: #FF6584;">({g_top_region['percentage']}%)</span>
                        </div>
                        <div class="result-card-desc">{g_top_region.get('description', '')}</div>
                    </div>
                    """, unsafe_allow_html=True)

                    # グループ平均顔タイプ分析
                    st.markdown("#### グループ平均 顔タイプ分析（縄文系 / 弥生系）")
                    jomon_vals = [ind["jomon_yayoi"]["jomon_percent"] for ind in g_result["individual_results"]]
                    avg_jomon = round(float(np.mean(jomon_vals)), 1)
                    avg_yayoi = round(100.0 - avg_jomon, 1)

                    c_g1, c_g2 = st.columns([1, 1])
                    with c_g1:
                        st.metric("縄文系平均スコア (濃い/彫深)", f"{avg_jomon}%")
                    with c_g2:
                        st.metric("弥生系平均スコア (涼やか/面長)", f"{avg_yayoi}%")

                    st.progress(avg_jomon / 100.0)
                    grp_type = "縄文系傾向" if avg_jomon >= 50 else "弥生系傾向"
                    st.caption(f"**グループ全体傾向**: **{grp_type}**（メンバー{g_result['num_faces']}名の平均値）")

                    # Supabase Storage への非同期保存
                    if g_raw_bytes:
                        g_f_hash = hashlib.md5(g_raw_bytes).hexdigest()
                        if st.session_state.get("last_uploaded_group") != g_f_hash:
                            g_summary = {
                                "num_faces": g_result['num_faces'],
                                "region": g_top_region['region'],
                                "region_percentage": g_top_region['percentage'],
                                "top_pref": g_top_pref['prefecture'],
                                "top_pref_percentage": round(g_top_pref['percentage'], 2),
                                "top_prefectures": g_result["group_top_prefectures"][:5],
                                "avg_jomon_percent": avg_jomon,
                                "avg_yayoi_percent": avg_yayoi,
                                "group_type": grp_type
                            }
                            upload_diagnosis_async(g_raw_bytes, g_filename, g_summary, mode="group")
                            st.session_state["last_uploaded_group"] = g_f_hash

    # グループ診断: 下部セクション (ランキング ＆ 個別メンバー診断カード & SNSシェア)
    if g_image_input and predictor and 'g_result' in locals() and g_result["success"]:
        st.divider()
        g_res_col1, g_res_col2 = st.columns([1, 1], gap="large")

        with g_res_col1:
            st.markdown("### グループ統合 地方ブロック判定（6区分）")
            df_g_region = pd.DataFrame(g_result["group_top_regions"])
            fig_g_region = px.bar(
                df_g_region,
                x="percentage",
                y="region",
                orientation="h",
                text="percentage",
                color="percentage",
                color_continuous_scale="Sunset",
                labels={"percentage": "確率 (%)", "region": "地方ブロック"},
                title=f"グループ統合 ({g_result['num_faces']}名) 地方ブロックごとの推定確率"
            )
            fig_g_region.update_layout(yaxis=dict(autorange="reversed"), showlegend=False, height=360)
            fig_g_region.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
            st.plotly_chart(fig_g_region, use_container_width=True)

        with g_res_col2:
            st.markdown("### グループ統合 47都道府県 推定確率ランキング (Top 5)")
            g_top5 = g_result["group_top_prefectures"][:5]

            for idx, p in enumerate(g_top5):
                rank_num = idx + 1
                badge_class = f"rank-num rank-num-{rank_num}" if rank_num <= 3 else "rank-num"
                r_class = f"rank-item rank-item-{rank_num}" if rank_num <= 3 else "rank-item"
                st.markdown(f"""
                <div class="{r_class}">
                    <div style="display: flex; align-items: center;">
                        <span class="{badge_class}">{rank_num}</span>
                        <span style="font-weight:700; color:#1E293B; font-size:1.05rem;">{p['prefecture']}</span>
                        <span style="color:#64748B; font-size:0.82rem; margin-left:8px;">({p['region']})</span>
                    </div>
                    <div style="font-weight:800; color:#E11D48; font-size:1.05rem;">
                        {p['percentage']:.2f}%
                    </div>
                </div>
                """, unsafe_allow_html=True)

            with st.expander("グループ統合 全47都道府県の推定確率をすべて見る"):
                df_g_all = pd.DataFrame(g_result["group_top_prefectures"])[["rank", "prefecture", "region", "percentage"]]
                df_g_all.columns = ["順位", "都道府県", "地方", "推定確率 (%)"]
                st.dataframe(df_g_all, use_container_width=True, hide_index=True)

        # メンバー個別診断カード
        st.divider()
        st.markdown(f"### 検出されたメンバー一覧（全{g_result['num_faces']}名）")
        st.caption("※個人単体には表情や角度などの個人差ノイズが含まれますが、上記のグループベイズ統合により個人差が相殺され、高精度な共通出身地が導き出されます。")
        
        st.image(g_result["annotated_image"], caption="検出された各メンバーの顔枠 (左から順に #1〜)", use_container_width=True)

        n_members = len(g_result["individual_results"])
        cols_per_row = min(4, max(2, n_members))
        m_cols = st.columns(cols_per_row)
        for i, ind in enumerate(g_result["individual_results"]):
            c = m_cols[i % cols_per_row]
            with c:
                st.markdown(f"""
                <div style="background:#FFFFFF; border:1px solid #E2E8F0; border-radius:12px; padding:12px; margin-bottom:12px; box-shadow:0 2px 4px rgba(0,0,0,0.03);">
                    <div style="font-weight:800; color:#2563EB; font-size:1.05rem; margin-bottom:6px;">顔 #{ind['face_id']}</div>
                </div>
                """, unsafe_allow_html=True)
                st.image(ind["crop_image"], caption=f"#{ind['face_id']}", use_container_width=True)
                st.markdown(f"- 個別推定県: **{ind['top_pref']}** ({ind['top_pref_pct']}%)")
                st.markdown(f"- 個別地方: **{ind['top_region']}** ({ind['top_region_pct']}%)")
                st.markdown(f"- 顔タイプ: **{ind['jomon_yayoi']['face_type']}**")
                st.markdown(f"- 縄文: {ind['jomon_yayoi']['jomon_percent']}% / 弥生: {ind['jomon_yayoi']['yayoi_percent']}%")

        # SNSシェアセクション (グループ用)
        st.divider()
        st.markdown("### グループ診断結果をシェアする")
        grp_share_text = f"【AI集合写真 出身地＆ルーツ判定結果】\nメンバー{g_result['num_faces']}名の骨格統合判定\nグループ推定地域: 【{g_top_region['region']}】 ({g_top_region['percentage']}%)\n最有力都道府県: 【{g_top_pref['prefecture']}】 ({g_top_pref['percentage']:.1f}%)\nグループ傾向: {grp_type}\n#出身地判定AI #集合写真診断 #同郷判定 #AI顔診断"
        grp_encoded_text = urllib.parse.quote(grp_share_text)
        grp_x_url = f"https://twitter.com/intent/tweet?text={grp_encoded_text}"

        gs_col1, gs_col2 = st.columns([1, 2])
        with gs_col1:
            st.link_button("𝕏 (Twitter) でグループ診断結果をポスト", url=grp_x_url, use_container_width=True, type="primary")
        with gs_col2:
            st.text_area("シェア用テキスト（コピーして使えます）", grp_share_text, height=105)
