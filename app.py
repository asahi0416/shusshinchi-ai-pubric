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
    page_icon="🗾",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# カスタムCSS（洗練されたモダン＆和風テイスト）
st.markdown("""
<style>
    /* Streamlit標準UI要素（Deployボタン・ハンバーガーメニュー・フッター・ヘッダーバー）の完全消去 */
    header[data-testid="stHeader"] {
        display: none !important;
        visibility: hidden !important;
        height: 0px !important;
    }
    #MainMenu {
        display: none !important;
        visibility: hidden !important;
    }
    footer {
        display: none !important;
        visibility: hidden !important;
    }
    .stDeployButton, [data-testid="stDeployButton"] {
        display: none !important;
        visibility: hidden !important;
    }
    [data-testid="stToolbar"] {
        display: none !important;
        visibility: hidden !important;
    }
    [data-testid="stToolbarActions"] {
        display: none !important;
        visibility: hidden !important;
    }
    [data-testid="stStatusWidget"] {
        display: none !important;
        visibility: hidden !important;
    }
    [data-testid="stDecoration"] {
        display: none !important;
    }
    .block-container {
        padding-top: 1.8rem !important;
        padding-bottom: 3rem !important;
    }
    /* 全体フォント・レイアウト */
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+JP:wght@400;500;700;800&display=swap');
    html, body, [class*="css"] {
        font-family: 'Noto Sans JP', sans-serif;
    }

    /* ヘッダーバナー */
    .hero-container {
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 50%, #1E3A8A 100%);
        border-radius: 16px;
        padding: 32px 36px;
        color: #FFFFFF;
        margin-bottom: 24px;
        box-shadow: 0 10px 25px -5px rgba(15, 23, 42, 0.25);
    }
    .hero-badge {
        display: inline-block;
        background: rgba(59, 130, 246, 0.25);
        border: 1px solid rgba(147, 197, 253, 0.4);
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.85rem;
        font-weight: 600;
        letter-spacing: 0.05em;
        color: #93C5FD;
        margin-bottom: 12px;
    }
    .hero-title {
        font-size: 2.2rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        margin: 0 0 10px 0;
        background: linear-gradient(to right, #FFFFFF, #E2E8F0);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .hero-subtitle {
        font-size: 1.05rem;
        color: #CBD5E1;
        line-height: 1.6;
        margin: 0;
    }

    /* 結果カード */
    .result-card-main {
        background: linear-gradient(135deg, #EFF6FF 0%, #DBEAFE 100%);
        padding: 22px 24px;
        border-radius: 14px;
        border-left: 6px solid #2563EB;
        margin-bottom: 18px;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.08);
    }
    .result-card-title {
        font-size: 0.85rem;
        font-weight: 700;
        color: #1E40AF;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }
    .result-card-region {
        font-size: 1.85rem;
        font-weight: 800;
        color: #1E3A8A;
        margin: 6px 0 10px 0;
    }
    .result-card-pref {
        font-size: 1.15rem;
        font-weight: 700;
        color: #1D4ED8;
        margin-bottom: 10px;
    }
    .result-card-desc {
        font-size: 0.94rem;
        color: #334155;
        line-height: 1.6;
        background: rgba(255, 255, 255, 0.6);
        padding: 10px 14px;
        border-radius: 8px;
    }

    /* ランキングアイテム */
    .rank-item {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        padding: 10px 16px;
        border-radius: 10px;
        margin-bottom: 8px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        transition: all 0.2s ease;
    }
    .rank-item:hover {
        border-color: #93C5FD;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    }
    .rank-item-1 {
        background: linear-gradient(90deg, #FEF3C7 0%, #FFFFFF 100%);
        border: 1px solid #FCD34D;
    }
    .rank-item-2 {
        background: linear-gradient(90deg, #F1F5F9 0%, #FFFFFF 100%);
        border: 1px solid #CBD5E1;
    }
    .rank-item-3 {
        background: linear-gradient(90deg, #FFEDD5 0%, #FFFFFF 100%);
        border: 1px solid #FDBA74;
    }

    /* SNSシェアボックス */
    .share-container {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 18px 20px;
        margin-top: 20px;
    }

    /* プログレスバーグラデーション */
    .stProgress > div > div > div > div {
        background-image: linear-gradient(to right, #3B82F6, #8B5CF6);
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
    st.header("🗾 アプリについて")
    st.markdown("""
    本アプリは、最先端の**3D顔面解剖骨格解析**と**マルチモーダル深層学習**を用い、顔写真から日本列島の地域遺伝的ルーツ（地方・都道府県・縄文/弥生）を推定するWebアプリケーションです。

    ---
    ### 🔬 科学的アプローチ
    - **3D68点解剖骨格解析**:
      目・鼻・顎・頬骨の彫りの深さや比率を立体座標化。
    - **縄文 vs 弥生 形質スコアリング**:
      濃顔・彫深（縄文系）と涼やか・面長（弥生系）のバランスを客観的正規分布で算出。
    - **👥 集合写真ベイズ尤度比統合 (LLR) 【高精度の鍵】**:
      単一の顔写真では「表情・角度・個体差ノイズ」が地域差シグナルの約5〜12倍存在します。
      同郷の複数人（3〜8人以上）の骨格をベイズ対数尤度比で統合することで、
      個人差ノイズが統計的に相殺（1/√Nに消去）され、地方・都道府県の的中率が飛躍的に高まります。

    ---
    ### 🔒 データ取り扱いについて
    アップロードされた顔画像および診断結果は、AIの精度向上・学術分析目的でのみ安全に暗号化通信により管理・蓄積されます。第三者への開示・商用目的での利用は行われません。
    """)

# ----------------- メイン画面ヘッダー -----------------
st.markdown("""
<div class="hero-container">
    <div class="hero-badge">🧬 3D骨格 ＆ マルチモーダルAI</div>
    <div class="hero-title">🗾 AI顔画像 出身地＆ルーツ判定</div>
    <div class="hero-subtitle">
        あなたの顔貌骨格に眠る、日本列島の地域的特徴をAIが解明。<br>
        「6地方ブロック」「47都道府県ランキング」「縄文系／弥生系タイプ」を瞬時に診断します。
    </div>
</div>
""", unsafe_allow_html=True)

# 診断モード選択
diag_mode = st.radio(
    "診断モードを選択してください",
    [
        "👥 グループ・集合写真診断 (複数名) ★精度大幅UP・推奨",
        "👤 個人診断 (1名)"
    ],
    horizontal=True,
    help="同じ地元・学校・部活などのメンバーが写った集合写真（複数名）または単独写真（1名）を選択できます。"
)

# なぜ集合写真の方が精度が高いのか？常時解説バナー
st.markdown("""
<div style="background: linear-gradient(135deg, #F0FDF4 0%, #DCFCE7 100%); border: 1.5px solid #86EFAC; border-radius: 14px; padding: 18px 22px; margin: 12px 0 24px 0; box-shadow: 0 3px 10px rgba(22, 101, 52, 0.06);">
    <div style="font-weight: 800; color: #166534; font-size: 1.08rem; display: flex; align-items: center; gap: 8px; margin-bottom: 6px;">
        <span>🎯</span> なぜ「集合写真・グループ診断」のほうが圧倒的に精度が高いのか？
    </div>
    <div style="color: #15803D; font-size: 0.94rem; line-height: 1.65;">
        <b>1人の写真</b>には、表情・メイク・角度・個人の骨格差といった「個人差ノイズ」が大きく含まれるため、AIでも特定が難しくなります。<br>
        しかし<b>同郷の仲間（同じ地元・学校・部活・サークルなど）が3〜8人以上写った集合写真</b>では、個人差が統計的に綺麗に打ち消し合い（相殺）、
        骨格に刻まれた<b>純粋な地域遺伝ルーツ（地域シグナル）だけが共鳴・増幅</b>されるため、<b>的中精度が劇的に向上（都道府県Top-1が50%以上へ急上昇）</b>します！
    </div>
</div>
""", unsafe_allow_html=True)

# ----------------------------------------------------
# A. 個人診断モード (1名)
# ----------------------------------------------------
if diag_mode == "👤 個人診断 (1名)":
    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.subheader("📷 写真の入力")
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
        <div style="background: #EFF6FF; border: 1px solid #BFDBFE; border-radius: 10px; padding: 12px 14px; margin-top: 14px;">
            <div style="font-size: 0.88rem; color: #1E40AF; line-height: 1.5;">
                💡 <b>さらに高い精度で診断したい場合</b><br>
                同じ地元・出身校の友人と写った写真があれば、上の「👥 グループ・集合写真診断」をお試しください。個人差のブレが相殺されて的中精度が大幅に高まります。
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.subheader("🎯 診断結果")
        if image_input is None:
            st.info("👈 左側のエリアから写真をアップロードまたは撮影してください。")
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
                        <div class="result-card-title">AI骨格ルーツ診断結果</div>
                        <div class="result-card-region">
                            🗾 推定地域: <span style="color: #2563EB;">【{top_region['region']}】</span> <span style="font-size: 1.15rem; color: #3B82F6;">({top_region['percentage']}%)</span>
                        </div>
                        <div class="result-card-pref">
                            🏆 最有力都道府県: 【{top_pref['prefecture']}】 ({top_pref['percentage']:.2f}%)
                        </div>
                        <div class="result-card-desc">{top_region.get('description', '')}</div>
                    </div>
                    """, unsafe_allow_html=True)

                    # 縄文系 vs 弥生系 分析
                    st.markdown("#### 🧬 顔タイプ分析（縄文系 vs 弥生系）")
                    c_jy1, c_jy2 = st.columns([1, 1])
                    with c_jy1:
                        st.metric("縄文系スコア (濃い/彫深)", f"{jy['jomon_percent']}%")
                    with c_jy2:
                        st.metric("弥生系スコア (涼やか/面長)", f"{jy['yayoi_percent']}%")

                    st.progress(jy["jomon_percent"] / 100.0)
                    st.caption(f"**判定タイプ**: **{jy['face_type']}** | {jy['description']}")

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
            st.markdown("### 🗺️ 全地方ブロック判定（6区分）")
            df_region = pd.DataFrame(result["top_regions"])
            fig_region = px.bar(
                df_region,
                x="percentage",
                y="region",
                orientation="h",
                text="percentage",
                color="percentage",
                color_continuous_scale="Viridis",
                labels={"percentage": "確率 (%)", "region": "地方ブロック"},
                title="地方ブロックごとの推定確率"
            )
            fig_region.update_layout(yaxis=dict(autorange="reversed"), showlegend=False, height=360)
            fig_region.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
            st.plotly_chart(fig_region, use_container_width=True)

        with res_col2:
            st.markdown("### 🏆 47都道府県 推定確率ランキング (Top 5)")
            medals = ["🥇", "🥈", "🥉", "4位", "5位"]
            top5 = result["top_prefectures"][:5]

            for idx, p in enumerate(top5):
                m_icon = medals[idx]
                r_class = f"rank-item rank-item-{idx+1}" if idx < 3 else "rank-item"
                st.markdown(f"""
                <div class="{r_class}">
                    <div>
                        <span style="font-weight:800; font-size:1.15rem; margin-right:8px;">{m_icon}</span>
                        <span style="font-weight:700; color:#1E3A8A; font-size:1.1rem;">{p['prefecture']}</span>
                        <span style="color:#64748B; font-size:0.85rem; margin-left:8px;">({p['region']})</span>
                    </div>
                    <div style="font-weight:800; font-size:1.15rem; color:#2563EB;">{p['percentage']:.2f}%</div>
                </div>
                """, unsafe_allow_html=True)

            with st.expander("全47都道府県の推定確率をすべて見る"):
                df_all = pd.DataFrame(result["top_prefectures"])[["rank", "prefecture", "region", "percentage"]]
                df_all.columns = ["順位", "都道府県", "地方", "推定確率 (%)"]
                st.dataframe(df_all, use_container_width=True, hide_index=True)

        # SNSシェアセクション
        st.divider()
        st.markdown("### 📱 診断結果をシェアする")
        share_text = f"【AI顔画像 出身地＆ルーツ診断結果】\n🗾 推定地域: 【{top_region['region']}】 ({top_region['percentage']}%)\n🏆 最有力県: 【{top_pref['prefecture']}】 ({top_pref['percentage']:.1f}%)\n🧬 顔タイプ: {jy['face_type']} (縄文 {jy['jomon_percent']}% / 弥生 {jy['yayoi_percent']}%)\n#出身地判定AI #AI顔診断 #ルーツ診断 #縄文顔 #弥生顔"
        encoded_text = urllib.parse.quote(share_text)
        x_url = f"https://twitter.com/intent/tweet?text={encoded_text}"

        s_col1, s_col2 = st.columns([1, 2])
        with s_col1:
            st.link_button("𝕏 (Twitter) で診断結果をポスト", url=x_url, use_container_width=True, type="primary")
        with s_col2:
            st.text_area("シェア用テキスト（コピーして使えます）", share_text, height=105)

# ----------------------------------------------------
# B. グループ・集合写真診断モード (複数名)
# ----------------------------------------------------
else:
    st.markdown("""
    <div style="background: linear-gradient(135deg, #FEF3C7 0%, #FFFBEB 100%); border: 1.5px solid #FCD34D; border-radius: 12px; padding: 14px 18px; margin-bottom: 18px;">
        <div style="font-weight: 800; color: #92400E; font-size: 1.0rem; margin-bottom: 4px;">
            👥 【最高精度モード】同郷メンバーの集合推論（ベイズ対数尤度比統合）
        </div>
        <div style="color: #B45309; font-size: 0.91rem; line-height: 1.6;">
            写真に写っている全員の3D顔貌骨格を個別検出し、ベイズ統計統合で個人差ノイズを完全消去。<br>
            同郷グループ（3〜8名）全体の共通ルーツを最高精度（都道府県Top-1精度50%以上）で診断します。
        </div>
    </div>
    """, unsafe_allow_html=True)

    g_col1, g_col2 = st.columns([1, 1], gap="large")

    with g_col1:
        st.subheader("📷 集合写真の入力")
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
        st.subheader("🎯 グループ統合判定結果")
        if g_image_input is None:
            st.info("👈 左側のエリアから集合写真をアップロードまたは撮影してください。")
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
                        <div class="result-card-title">👥 グループ統合 骨格ルーツ判定 ({g_result['num_faces']}名統合)</div>
                        <div class="result-card-region">
                            🗾 推定地域: <span style="color: #2563EB;">【{g_top_region['region']}】</span> <span style="font-size: 1.15rem; color: #3B82F6;">({g_top_region['percentage']}%)</span>
                        </div>
                        <div class="result-card-pref">
                            🏆 最有力都道府県: 【{g_top_pref['prefecture']}】 ({g_top_pref['percentage']:.2f}%)
                        </div>
                        <div class="result-card-desc">{g_top_region.get('description', '')}</div>
                    </div>
                    """, unsafe_allow_html=True)

                    # グループ平均顔タイプ分析
                    st.markdown("#### 🧬 グループ平均 顔タイプ分析（縄文系 vs 弥生系）")
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
            st.markdown("### 🗺️ グループ統合 地方ブロック判定（6区分）")
            df_g_region = pd.DataFrame(g_result["group_top_regions"])
            fig_g_region = px.bar(
                df_g_region,
                x="percentage",
                y="region",
                orientation="h",
                text="percentage",
                color="percentage",
                color_continuous_scale="Viridis",
                labels={"percentage": "確率 (%)", "region": "地方ブロック"},
                title=f"グループ統合 ({g_result['num_faces']}名) 地方ブロックごとの推定確率"
            )
            fig_g_region.update_layout(yaxis=dict(autorange="reversed"), showlegend=False, height=360)
            fig_g_region.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
            st.plotly_chart(fig_g_region, use_container_width=True)

        with g_res_col2:
            st.markdown("### 🏆 グループ統合 47都道府県 推定確率ランキング (Top 5)")
            medals = ["🥇", "🥈", "🥉", "4位", "5位"]
            g_top5 = g_result["group_top_prefectures"][:5]

            for idx, p in enumerate(g_top5):
                m_icon = medals[idx]
                r_class = f"rank-item rank-item-{idx+1}" if idx < 3 else "rank-item"
                st.markdown(f"""
                <div class="{r_class}">
                    <div>
                        <span style="font-weight:800; font-size:1.15rem; margin-right:8px;">{m_icon}</span>
                        <span style="font-weight:700; color:#1E3A8A; font-size:1.1rem;">{p['prefecture']}</span>
                        <span style="color:#64748B; font-size:0.85rem; margin-left:8px;">({p['region']})</span>
                    </div>
                    <div style="font-weight:800; font-size:1.15rem; color:#2563EB;">{p['percentage']:.2f}%</div>
                </div>
                """, unsafe_allow_html=True)

            with st.expander("グループ統合 全47都道府県の推定確率をすべて見る"):
                df_g_all = pd.DataFrame(g_result["group_top_prefectures"])[["rank", "prefecture", "region", "percentage"]]
                df_g_all.columns = ["順位", "都道府県", "地方", "推定確率 (%)"]
                st.dataframe(df_g_all, use_container_width=True, hide_index=True)

        # メンバー個別診断カード
        st.divider()
        st.markdown(f"### 👥 検出されたメンバー一覧（全{g_result['num_faces']}名）")
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
        st.markdown("### 📱 グループ診断結果をシェアする")
        grp_share_text = f"【AI集合写真 出身地＆ルーツ診断結果】\n👥 メンバー{g_result['num_faces']}名の骨格統合判定！\n🗾 グループ推定地域: 【{g_top_region['region']}】 ({g_top_region['percentage']}%)\n🏆 最有力都道府県: 【{g_top_pref['prefecture']}】 ({g_top_pref['percentage']:.1f}%)\n🧬 グループ傾向: {grp_type}\n#出身地判定AI #集合写真診断 #同郷判定 #AI顔診断"
        grp_encoded_text = urllib.parse.quote(grp_share_text)
        grp_x_url = f"https://twitter.com/intent/tweet?text={grp_encoded_text}"

        gs_col1, gs_col2 = st.columns([1, 2])
        with gs_col1:
            st.link_button("𝕏 (Twitter) でグループ診断結果をポスト", url=grp_x_url, use_container_width=True, type="primary")
        with gs_col2:
            st.text_area("シェア用テキスト（コピーして使えます）", grp_share_text, height=105)
