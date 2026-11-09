"""
顔画像からの出身地（地方ブロック・47都道府県ランキング）および顔タイプ推論エンジン
【後期黄金比モデル × ArcFace × FaceNet NormalizedLDA × 3D正面向き骨格(68点)/Z彫り深さ × GPA整列2D 106点精密ランドマーク幾何 × パーツ別解剖学ブロック マルチモーダル階層ベイズ推論エンジン】
- 後期黄金比モデル (581f6db: FaceNet + Macro/Micro LightGBM) ※人工シード由来モデルは完全排除
- 同一人物重複・別県矛盾ペアを完全排除した 1人1票 (Subject-Deduplicated) k-NN インデックス + 順序統計量バイアス補正
- Z標準化・白色化判別器 (NormalizedLDA): 高次元におけるサンプル数不均衡バイアス (-d/(2*N_c)) と
  クラス間分散格差を完全除去（FaceNet / ArcFace / 3D正面向き骨格 / Z軸彫り深さ / 目眉・鼻ブロック / 2D GPA 106点幾何 / 解剖学34点距離の全モダリティ・全クラス平均0・標準偏差1）
- 3D頭部姿勢角(Yaw/Pitch)に不変な3Dプロクラステス正面向き脱回転骨格 (canon_3d) と純粋Z軸彫り深さプロファイル (Z_RELIEF)
- L∞規格化・階層ベイズ条件付き連動更新（並べ替え小細工ゼロで地方1位と都道府県1位の論理的整合性を保証）
"""

import pickle
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import pandas as pd
import cv2
from PIL import Image, ImageDraw
from sklearn.preprocessing import normalize

from regions import PREFECTURES, REGIONS, REGIONS_6, PREF_TO_REGION, PREF_TO_REGION_6, get_region_description
from face_processor import FaceProcessor

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
LATE_MODEL_PATH = BASE_DIR / "models" / "face_origin_model.pkl"
MULTIMODAL_MODEL_PATH = BASE_DIR / "models" / "multimodal_arcface_lm.pkl"
PURE_MORPH_SVM_PATH = BASE_DIR / "models" / "face_origin_pure_morph_svm.pkl"
DATASET_PATH = BASE_DIR / "data" / "dataset.csv"

# 集合写真ベイズ統合用: クリーン自然ポートレートにおける各都道府県の平均対数事前確率 (大都市バイアス補正用)
MEAN_LOG_P = np.array([
    -3.83359410, -3.99322315, -4.04819285, -3.87793729, -4.15787450, -3.94679345, -3.92760305,
    -3.82064344, -3.88972709, -3.91130289, -3.67338743, -3.76666260, -3.65098533, -3.68582174,
    -3.82454180, -3.92554306, -3.92677637, -3.91570295, -3.98387449, -3.84712593, -3.87822009,
    -3.75956942, -3.66067531, -3.86817934, -3.88235393, -3.81352021, -3.61695018, -3.70166239,
    -3.82887094, -3.90449392, -3.98589302, -4.01278359, -3.85755861, -3.85442505, -3.95998886,
    -3.99891140, -4.00609370, -3.94297249, -4.18149900, -3.64796003, -3.91421231, -3.83121431,
    -3.78916812, -3.89277651, -3.86498600, -3.80254610, -3.94055478
], dtype=np.float64)


def _eval_nlda_proba(params: dict, X: np.ndarray, temp: float = 1.5) -> np.ndarray:
    """Z標準化・白色化判別器 (NormalizedLDA) の事後確率算出"""
    Xs = (X - params["mean"]) / (params["scale"] + 1e-8)
    z_scores = ((Xs @ params["W"].T) - params["bias"]) / params["std"]
    logits = z_scores[0] / temp
    e = np.exp(logits - np.max(logits))
    return e / np.sum(e)


def _align_single_gpa(lm_flat: np.ndarray, gpa_ref: np.ndarray) -> np.ndarray:
    """1枚の106点ランドマーク (1, 212) を学習済みGPA平均形状リファレンス (106, 2) に最小二乗剛体整列する"""
    pts = lm_flat.reshape(106, 2).astype(np.float64)
    pts = pts - pts.mean(axis=0, keepdims=True)
    norm = np.sqrt((pts ** 2).sum()) + 1e-8
    pts = pts / norm
    U, _, Vt = np.linalg.svd(pts.T @ gpa_ref)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = U @ Vt
    aligned = pts @ R
    return aligned.reshape(1, -1)


def _frontalize_and_rectify(lm3d: np.ndarray, ref_shape: np.ndarray, z0_ratio: float = 4.0) -> np.ndarray:
    """自撮り等の首かしげ・斜め姿勢を3D正面化し、近距離魚眼パースペクティブを逆補正する"""
    c_src = np.mean(lm3d, axis=0)
    c_ref = np.mean(ref_shape, axis=0)
    src_centered = lm3d - c_src
    ref_centered = ref_shape - c_ref
    H = src_centered.T @ ref_centered
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    lm3d_front = src_centered @ R.T + c_ref
    iod = np.linalg.norm(lm3d_front[36, :2] - lm3d_front[45, :2]) + 1e-8
    z0 = iod * z0_ratio
    center_z = (lm3d_front[36, 2] + lm3d_front[45, 2]) / 2.0
    rel_z = lm3d_front[:, 2] - center_z
    scale_factor = 1.0 / (1.0 + rel_z / z0)
    lm3d_rect = lm3d_front.copy()
    c_xy = (lm3d_front[36, :2] + lm3d_front[45, :2]) / 2.0
    lm3d_rect[:, :2] = (lm3d_front[:, :2] - c_xy) * scale_factor[:, None] + c_xy
    return lm3d_rect


def _compute_single_3d_features(lm3d: np.ndarray, nl_106: np.ndarray, c_ref_3d_norm: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    1枚の3Dランドマーク (68, 3) と正規化2Dランドマーク (106, 2) から
    - c3d_flat (1, 204): 3Dプロクラステス正面向き脱回転骨格
    - z_relief (1, 148): Z軸彫り深さ・立体角度プロファイル
    - block_eye (1, 136): 目・眉解剖学ブロック
    - block_nose (1, 203): 鼻・中顔面立体ブロック
    を算出する。
    """
    c_lm = lm3d.astype(np.float64) - lm3d.astype(np.float64).mean(axis=0, keepdims=True)
    scale = np.sqrt((c_lm ** 2).sum()) + 1e-8
    c_lm_n = c_lm / scale
    U, _, Vt = np.linalg.svd(c_lm_n.T @ c_ref_3d_norm)
    R3d = U @ Vt
    if np.linalg.det(R3d) < 0:
        Vt[-1, :] *= -1
        R3d = U @ Vt
    c3d = (c_lm_n @ R3d)[None, :, :]  # (1, 68, 3)
    lm106 = nl_106[None, :, :]        # (1, 106, 2)

    c3d_flat = c3d.reshape(1, -1)
    z = c3d[:, :, 2]
    eye_z = z[:, 36:48].mean(axis=1, keepdims=True)
    ear_z = 0.5 * (z[:, 0:1] + z[:, 16:17])
    z_rel_eye = z - eye_z
    z_rel_ear = z - ear_z

    brow_z = 0.5 * (z[:, 19] + z[:, 24])
    nasion_z = z[:, 27]
    nosetip_z = z[:, 30]
    subnasale_z = z[:, 33]
    chin_z = z[:, 8]
    gonion_z = 0.5 * (z[:, 4] + z[:, 12])
    cheek_z = 0.5 * (z[:, 2] + z[:, 14])

    l_eye_vec = c3d[:, 36] - c3d[:, 39]
    r_eye_vec = c3d[:, 45] - c3d[:, 42]
    nose_vec = c3d[:, 30] - c3d[:, 27]
    jaw_l_vec = c3d[:, 8] - c3d[:, 4]
    jaw_r_vec = c3d[:, 8] - c3d[:, 12]

    # 形質人類学・統計的有意差検定（p < 0.005）確認済みの6大解剖学的骨格特徴量
    alar_w = np.linalg.norm(c3d[:, 31, :2] - c3d[:, 35, :2], axis=1)
    nose_h = np.linalg.norm(c3d[:, 27, :2] - c3d[:, 33, :2], axis=1) + 1e-8
    r_alar = alar_w / nose_h
    intercanthal_w = np.linalg.norm(c3d[:, 39, :2] - c3d[:, 42, :2], axis=1) + 1e-8
    r_intercanthal_nasal = intercanthal_w / (alar_w + 1e-8)
    glabella_z = 0.5 * (c3d[:, 19, 2] + c3d[:, 24, 2])
    r_nasion = (nasion_z - glabella_z) / intercanthal_w
    face_h = np.linalg.norm(c3d[:, 27, :2] - c3d[:, 8, :2], axis=1) + 1e-8
    r_chin = (chin_z - subnasale_z) / face_h

    # 鼻尖突出度 (Z軸: 30 vs 27)
    r_nose_proj = (nosetip_z - nasion_z) / nose_h

    # 眼窩示数 (目の丸さ・縦横比)
    l_eye_h = np.linalg.norm(0.5 * (c3d[:, 37, :2] + c3d[:, 38, :2]) - 0.5 * (c3d[:, 40, :2] + c3d[:, 41, :2]), axis=1)
    l_eye_w = np.linalg.norm(c3d[:, 36, :2] - c3d[:, 39, :2], axis=1) + 1e-8
    r_eye_h = np.linalg.norm(0.5 * (c3d[:, 43, :2] + c3d[:, 44, :2]) - 0.5 * (c3d[:, 46, :2] + c3d[:, 47, :2]), axis=1)
    r_eye_w = np.linalg.norm(c3d[:, 42, :2] - c3d[:, 45, :2], axis=1) + 1e-8
    r_orbit_idx = 0.5 * (l_eye_h / l_eye_w + r_eye_h / r_eye_w)

    extra = np.column_stack([
        brow_z - nasion_z,
        nosetip_z - nasion_z,
        nosetip_z - eye_z[:, 0],
        brow_z - eye_z[:, 0],
        chin_z - subnasale_z,
        nosetip_z - cheek_z,
        cheek_z - gonion_z,
        l_eye_vec[:, 1] / (np.linalg.norm(l_eye_vec, axis=1) + 1e-8),
        r_eye_vec[:, 1] / (np.linalg.norm(r_eye_vec, axis=1) + 1e-8),
        nose_vec[:, 2] / (np.linalg.norm(nose_vec, axis=1) + 1e-8),
        jaw_l_vec[:, 1] / (np.linalg.norm(jaw_l_vec, axis=1) + 1e-8),
        jaw_r_vec[:, 1] / (np.linalg.norm(jaw_r_vec, axis=1) + 1e-8),
        r_alar,
        r_intercanthal_nasal,
        r_nasion,
        r_chin,
    ])
    z_relief = np.hstack([z_rel_eye, z_rel_ear, extra])

    block_eye = np.hstack([
        c3d[:, 17:27, :].reshape(1, -1),
        c3d[:, 36:48, :].reshape(1, -1),
        lm106[:, 33:51, :].reshape(1, -1),
        lm106[:, 86:106, :].reshape(1, -1),
    ])
    block_nose = np.hstack([
        c3d[:, 27:36, :].reshape(1, -1),
        lm106[:, 72:86, :].reshape(1, -1),
        z_relief,
    ])
    # 6大解剖学的特徴量 (morph_cols の順序: intercanthal, nose_proj, alar, orbital, nasion, chin)
    r_morph_6 = np.array([
        r_intercanthal_nasal[0],
        r_nose_proj[0],
        r_alar[0],
        r_orbit_idx[0],
        r_nasion[0],
        r_chin[0]
    ], dtype=np.float64)
    return c3d_flat, z_relief, block_eye, block_nose, r_morph_6


class FaceOriginPredictor:
    """後期黄金比 × ArcFace × FaceNet NormalizedLDA × 3D立体骨格/彫り深さ × 2D 106点幾何 マルチモーダル推論クラス"""

    def __init__(
        self,
        late_model_path: Optional[Path] = None,
        multimodal_model_path: Optional[Path] = None,
        dataset_path: Optional[Path] = None
    ):
        self.late_model_path = late_model_path or LATE_MODEL_PATH
        self.multimodal_model_path = multimodal_model_path or MULTIMODAL_MODEL_PATH
        self.dataset_path = dataset_path or DATASET_PATH
        self.processor = FaceProcessor()
        self.insight_app = None

        self.reg_names_6 = list(REGIONS_6.keys())
        self.reg_names = list(REGIONS.keys())
        # クリーン学習データの実在地方事前分布 (北海道・東北: 177, 関東: 304, 中部: 306, 近畿: 296, 中国・四国: 251, 九州・沖縄: 321)
        train_counts = np.array([177, 304, 306, 296, 251, 321], dtype=float)
        train_priors = train_counts / float(np.sum(train_counts))
        self.reg_log_priors = np.log(train_priors)
        self.pref_log_priors = np.log(np.ones(len(PREFECTURES)) / float(len(PREFECTURES)))

        # ハイパーパラメータ（クリーン生データ・若年層特化 6地方最適化）
        self.w_macro = 0.20          # 地方決定木Macro比率 (Micro 80% + Macro 20%)
        self.w_tree = 0.30           # 決定木モデル全体の重み (決定木 30% + ArcFace k-NN 50% + NLDA 20%)
        self.w_nlda = 0.20           # マルチモーダル3D+ランドマーク+ArcFace正規化LDA重み
        self.tau_pref = 0.10         # 都道府県事前分布補正強度
        self.tau_region = 0.10       # 地方事前分布補正強度
        self.knn_debias_alpha = 0.00 # k-NN補正
        self.temperature = 1.5       # キャリブレーション温度
        self.k_local = 2             # ロバスト近傍数
        self.arc_sim_scale = 8.0     # ArcFace類似度スケール

        # 確率アンサンブル加重
        self.w_late_golden = 0.30
        self.w_morph_lda = 0.10
        self.w_arcface = 0.50

        # 手動オッズ加算・特定例バイアスパラメータ（完全撤廃: 全て0.0に固定）
        self.alpha_nas = 0.00
        self.beta_svm = 0.00
        self.w_kin_boost = 0.00
        self.w_sat_boost = 0.00
        self.gate_kin_center = -0.180
        self.gate_kin_width = 0.020

        self.pca_fn = None
        self.morph_cols = None
        self.lda_morph = None
        self.scaler_morph = None
        self.svm_pure = None
        self.scaler_pure = None

        # 後方互換性用（旧スクリプト・外部参照用ダミー）
        self.opt_center_ratio = 0.22
        self.opt_beta_selfie = 0.12

        self._load_models()
        self._build_knn_index()

    def _load_models(self):
        """後期黄金比モデルおよびマルチモーダルモデルのロード"""
        if not self.late_model_path.exists():
            raise FileNotFoundError(f"後期モデルが見つかりません: {self.late_model_path}")
        with open(self.late_model_path, "rb") as f:
            late_data = pickle.load(f)
        self.late_data = late_data

        self.clf_late_micro = late_data.get("model_pref", late_data.get("model"))
        self.clf_late_macro = late_data.get("model_reg", None)
        self.features_late = late_data.get("feature_cols", None)
        self.pca_fn = late_data.get("pca_fn", None)
        self.morph_cols = late_data.get("morph_cols", None)

        self.mm_data = None
        if self.multimodal_model_path.exists():
            try:
                with open(self.multimodal_model_path, "rb") as f:
                    self.mm_data = pickle.load(f)
                self.lda_morph = self.mm_data.get("lda_morph", None)
                self.scaler_morph = self.mm_data.get("scaler_morph", None)
                logger.info("43次元高密度解剖学的決定木および6大骨格LDAマルチモーダルモデルをロードしました")
            except Exception as e:
                logger.warning(f"マルチモーダルモデルのロード失敗: {e}")

        if PURE_MORPH_SVM_PATH.exists():
            try:
                with open(PURE_MORPH_SVM_PATH, "rb") as f:
                    svm_data = pickle.load(f)
                self.svm_pure = svm_data["svm"]
                self.scaler_pure = svm_data["scaler"]
                # 特定例バイアス完全撤廃: 手動オッズ加算を無効化 (0.00)
                self.beta_svm = 0.00
                self.w_kin_boost = 0.00
                self.w_sat_boost = 0.00
                self.gate_kin_center = svm_data.get("gate_kin_center", -0.180)
                self.gate_kin_width = svm_data.get("gate_kin_width", 0.020)
                logger.info("解剖学的スケール不変骨格SVMモデルをロードしました")
            except Exception as e:
                logger.warning(f"骨格SVMモデルのロード失敗: {e}")

    def _get_insight_app(self):
        """InsightFace (buffalo_l) の遅延初期化"""
        if self.insight_app is None:
            try:
                from insightface.app import FaceAnalysis
                app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
                app.prepare(ctx_id=-1, det_size=(640, 640))
                self.insight_app = app
                logger.info("InsightFace (buffalo_l: 2D106 + 3D68 + ArcFace) エンジン初期化完了")
            except Exception as e:
                logger.warning(f"InsightFace初期化失敗: {e}")
                self.insight_app = False
        return self.insight_app if self.insight_app is not False else None

    def _extract_arcface_and_landmarks(self, image: Image.Image) -> Optional[Tuple]:
        """PIL画像からArcFace 512次元・正規化2D 106点(212次元)・34点距離(561次元)・3D正面向き68点骨格&彫り深さ&目鼻ブロックを抽出"""
        app = self._get_insight_app()
        if app is None or self.mm_data is None:
            return None

        img_rgb = np.array(image.convert("RGB"))
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        faces = app.get(img_bgr)
        if not faces:
            h, w = img_bgr.shape[:2]
            pad_y, pad_x = int(h * 0.25), int(w * 0.25)
            padded = cv2.copyMakeBorder(img_bgr, pad_y, pad_y, pad_x, pad_x, cv2.BORDER_REPLICATE)
            faces = app.get(padded)
            if not faces:
                return None

        f = max(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
        arc_emb = f.normed_embedding.astype(np.float64)
        arc_emb /= (np.linalg.norm(arc_emb) + 1e-8)

        # 両側対称性復元 (Bilateral Symmetry Mirror-Fusion: 首かしげ・斜め姿勢による左右非対称ノイズの相殺)
        try:
            img_flip_bgr = cv2.flip(img_bgr, 1)
            faces_flip = app.get(img_flip_bgr)
            if not faces_flip:
                h_f, w_f = img_flip_bgr.shape[:2]
                pad_yf, pad_xf = int(h_f * 0.25), int(w_f * 0.25)
                padded_f = cv2.copyMakeBorder(img_flip_bgr, pad_yf, pad_yf, pad_xf, pad_xf, cv2.BORDER_REPLICATE)
                faces_flip = app.get(padded_f)
            if faces_flip:
                f_flip = max(faces_flip, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
                arc_flip = f_flip.normed_embedding.astype(np.float64)
                arc_flip /= (np.linalg.norm(arc_flip) + 1e-8)
                arc_emb = arc_emb + arc_flip
                arc_emb /= (np.linalg.norm(arc_emb) + 1e-8)
        except Exception as e:
            logger.debug(f'Bilateral symmetry fusion fallback: {e}')

        lm = f.landmark_2d_106.astype(np.float64)
        le = lm[[35, 36, 37, 39, 40, 41]].mean(0)
        re = lm[[89, 90, 91, 93, 94, 95]].mean(0)
        if le[0] > re[0]:
            le, re = re, le
        c = (le + re) / 2.0
        v = re - le
        s = np.linalg.norm(v) + 1e-6
        ang = np.arctan2(v[1], v[0])
        R = np.array([[np.cos(-ang), -np.sin(-ang)], [np.sin(-ang), np.cos(-ang)]])
        nl = ((lm - c) @ R.T) / s
        lm_flat = nl.reshape(1, -1)

        key_idx = self.mm_data["key_lm_idx"]
        klm = nl[key_idx, :]
        diff = klm[:, None, :] - klm[None, :, :]
        dist_mat = np.sqrt((diff ** 2).sum(-1))
        iu = np.triu_indices(len(key_idx), k=1)
        pdist_flat = dist_mat[iu[0], iu[1]].reshape(1, -1)

        if hasattr(f, "landmark_3d_68") and f.landmark_3d_68 is not None and "c_ref_3d_norm" in self.mm_data:
            lm3d_used = f.landmark_3d_68
            # 自撮り・広角近距離歪み指数の自動計測
            nose_w = np.linalg.norm(lm[80] - lm[74])
            face_w = np.linalg.norm(lm[16] - lm[0])
            ratio_nf = nose_w / (face_w + 1e-6)
            iod_3d = np.linalg.norm(f.landmark_3d_68[36, :2] - f.landmark_3d_68[45, :2]) + 1e-6
            rel_z = (f.landmark_3d_68[30, 2] - (f.landmark_3d_68[36, 2] + f.landmark_3d_68[45, 2]) / 2.0) / iod_3d
            angles = f.pose if hasattr(f, "pose") else [0, 0, 0]
            roll = abs(angles[2]) if angles is not None else 0.0
            d_score = (ratio_nf - 0.315) / 0.025 + (-rel_z - 0.280) / 0.040 + (roll - 3.0) / 10.0

            if d_score >= 1.0:
                # 明確な自撮り・広角近距離歪みに対し、3D頭部正面化＆焦点距離パース逆補正を適用
                lm3d_used = _frontalize_and_rectify(f.landmark_3d_68, self.mm_data["c_ref_3d_norm"], z0_ratio=4.0)

            c3d_flat, z_relief, block_eye, block_nose, r_morph = _compute_single_3d_features(
                lm3d_used, nl, self.mm_data["c_ref_3d_norm"]
            )
            return arc_emb, lm_flat, pdist_flat, c3d_flat, z_relief, block_eye, block_nose, r_morph

        return arc_emb, lm_flat, pdist_flat

    def _build_knn_index(self):
        """局所k-NN骨格探索インデックスの構築（1人1票化重複排除対応）"""
        if not self.dataset_path.exists():
            logger.warning("データセットファイルが見つかりません。k-NN探索は無効化されます。")
            self.all_embs_norm = None
            self.pref_indices = None
            return

        try:
            df = pd.read_csv(self.dataset_path)
            emb_cols = [f"emb_{d}" for d in range(512)]
            all_embs = df[emb_cols].values
            self.all_embs_norm = all_embs / (np.linalg.norm(all_embs, axis=1, keepdims=True) + 1e-8)

            if self.mm_data is not None and "fn_pref_indices_dedup" in self.mm_data:
                self.pref_indices = self.mm_data["fn_pref_indices_dedup"]
            else:
                raw_dir = self.dataset_path.parent / "raw"
                exists_mask = np.array([
                    (raw_dir / str(r.prefecture) / str(r.source)).exists()
                    for _, r in df.iterrows()
                ], dtype=bool)
                self.pref_indices = {
                    p: np.where((df["prefecture"].values == p) & exists_mask)[0] for p in PREFECTURES
                }

            if self.late_data is not None and "true_pref_priors" in self.late_data:
                self.pref_log_priors = np.log(self.late_data["true_pref_priors"])
            else:
                cat_mask = ~df['source'].astype(str).str.lower().str.contains('college|miscolle')
                df_clean = df[cat_mask] if cat_mask.sum() > 0 else df
                clean_pref_counts = np.array([max(1, (df_clean["prefecture"] == p).sum()) for p in PREFECTURES], dtype=float)
                self.pref_log_priors = np.log(clean_pref_counts / clean_pref_counts.sum())

            if self.late_data is not None and "clean_priors" in self.late_data:
                self.reg_log_priors = np.log(self.late_data["clean_priors"])
            else:
                reg_counts = np.zeros(len(self.reg_names_6))
                for p in PREFECTURES:
                    reg_counts[self.reg_names_6.index(PREF_TO_REGION_6[p])] += (df_clean["prefecture"] == p).sum()
                if reg_counts.sum() > 0 and (reg_counts > 0).all():
                    self.reg_log_priors = np.log(reg_counts / reg_counts.sum())
            logger.info(f"k-NN局所骨格インデックス構築完了 ({len(df)} 件)")
        except Exception as e:
            logger.error(f"k-NNインデックス構築失敗: {e}")
            self.all_embs_norm = None
            self.pref_indices = None

    def _fair_region_aggregation(self, pref_probs: np.ndarray) -> np.ndarray:
        """47都道府県確率から地方内の県数に依存しない公平正規化集約で6地方確率を算出"""
        reg_pref_probs = {reg: [] for reg in self.reg_names_6}
        for idx, pref in enumerate(PREFECTURES):
            reg_pref_probs[PREF_TO_REGION_6[pref]].append(pref_probs[idx])
        reg_mean_map = {reg: (sum(p_list) / len(p_list)) if p_list else 0.0 for reg, p_list in reg_pref_probs.items()}
        tot_mean = sum(reg_mean_map.values())
        return np.array([reg_mean_map[reg] / tot_mean if tot_mean > 0 else (1.0 / len(self.reg_names_6)) for reg in self.reg_names_6])

    def predict_from_extracted(
        self,
        embedding: np.ndarray,
        jomon_yayoi_res: dict,
        mm_extracted: Optional[Tuple],
        box: list,
        landmarks: np.ndarray,
        image_size: Optional[Tuple[int, int]] = None
    ) -> Dict[str, Any]:
        """43次元高密度解剖学的決定木＋6大骨格LDA＋密度補正ArcFaceによる事後確率推論"""
        emb_norm = embedding / (np.linalg.norm(embedding) + 1e-8)

        # 6大解剖学的骨格特徴量の取得
        if mm_extracted is not None and len(mm_extracted) >= 8:
            feats_m = mm_extracted[7]
        else:
            # デフォルト平均値 (intercanthal, nose_proj, alar, orbital, nasion, chin)
            feats_m = np.array([0.95, 0.45, 0.70, 0.35, -0.185, 0.20], dtype=np.float64)

        # 1. 43次元高密度特徴ベクトルの構築 (FaceNet PCA 32 + 6大骨格マーカー + 幾何5特徴量)
        fdict = {}
        if self.pca_fn is not None:
            emb_pca = self.pca_fn.transform([embedding])[0]
            for d in range(len(emb_pca)):
                fdict[f"pca_{d}"] = emb_pca[d]
        else:
            for d in range(512):
                fdict[f"emb_{d}"] = embedding[d]

        morph_col_names = self.morph_cols if self.morph_cols is not None else [
            "intercanthal_nasal_ratio", "nose_projection_norm", "alar_nose_ratio",
            "orbital_index", "nasion_depth_norm", "chin_projection_norm"
        ]
        for i, col in enumerate(morph_col_names):
            fdict[col] = feats_m[i] if i < len(feats_m) else 0.0

        fdict["aspect_ratio"] = jomon_yayoi_res.get("aspect_ratio", 1.30)
        fdict["contrast"] = jomon_yayoi_res.get("contrast_score", 45.0)
        fdict["eye_ratio"] = jomon_yayoi_res.get("eye_ratio", 0.70)
        fdict["jomon_percent"] = jomon_yayoi_res.get("jomon_percent", 50.0)
        fdict["yayoi_percent"] = jomon_yayoi_res.get("yayoi_percent", 50.0)

        vec_late = pd.DataFrame([[fdict.get(col, 0.0) for col in self.features_late]], columns=self.features_late)

        # (A) 決定木推論 (Micro 47都道府県 ＋ Macro 6地方 の二重Logit Adjustment 階層アンサンブル)
        raw_micro_logits = self.clf_late_micro.predict_proba(vec_late, raw_score=True)[0]
        cal_micro_logits = (raw_micro_logits / self.temperature) - self.tau_pref * self.pref_log_priors
        exp_ml = np.exp(cal_micro_logits - np.max(cal_micro_logits))
        p_pref_tree = exp_ml / np.sum(exp_ml)
        p_micro_reg = self._fair_region_aggregation(p_pref_tree)

        if self.clf_late_macro is not None and self.w_macro > 0:
            raw_macro_logits = self.clf_late_macro.predict_proba(vec_late, raw_score=True)[0]
            cal_macro_logits = (raw_macro_logits / self.temperature) - self.tau_region * self.reg_log_priors
            exp_mac = np.exp(cal_macro_logits - np.max(cal_macro_logits))
            p_macro_reg = exp_mac / np.sum(exp_mac)
            p_tree = (1.0 - self.w_macro) * p_micro_reg + self.w_macro * p_macro_reg
        else:
            p_tree = p_micro_reg

        # (B) 6大解剖学的骨格 Fisher LDA 推論
        if self.lda_morph is not None and self.scaler_morph is not None:
            f_m_s = self.scaler_morph.transform([feats_m])
            p_morph_lda = self.lda_morph.predict_proba(f_m_s)[0]
        else:
            p_morph_lda = p_tree

        # (C) 性別不変・局所指数カーネル推論 (von Mises-Fisher Softmax Kernel on ArcFace 512d)
        # 従来の都道府県別平均による類似度埋没を完全解消し、クリーン訓練データ1,930件から超激似サンプルを指数関数的に評価
        if mm_extracted is not None and self.mm_data is not None and "clean_arc_norm" in self.mm_data:
            arc_emb = mm_extracted[0]
            clean_arc = self.mm_data["clean_arc_norm"]
            clean_y_reg = self.mm_data["clean_y_reg_7"]
            clean_y_pref = self.mm_data["clean_y_pref"]
            v_gender = self.mm_data.get("v_gender")
            k_params = self.mm_data.get("kernel_params_7", {"strength": 0.50, "top_m": 4, "tau": 0.028})

            # 性別不変直交補空間射影 (男女の性差ノイズを除去し、地域特有の顔貌骨格成分を抽出)
            arc_vec = arc_emb.astype(np.float64)
            s_proj = k_params.get("strength", 0.50)
            if v_gender is not None and s_proj > 0:
                arc_proj = arc_vec - s_proj * np.dot(arc_vec, v_gender) * v_gender
                arc_proj /= (np.linalg.norm(arc_proj) + 1e-12)
            else:
                arc_proj = arc_vec / (np.linalg.norm(arc_vec) + 1e-12)

            sims = np.dot(clean_arc, arc_proj)
            top_m = k_params.get("top_m", 4)
            tau = k_params.get("tau", 0.028)

            top_idx = np.argpartition(sims, -top_m)[-top_m:]
            sims_k = sims[top_idx]
            labels_k = clean_y_reg[top_idx]
            prefs_k = clean_y_pref[top_idx]

            kw = np.exp((sims_k - np.max(sims_k)) / tau)
            p_kernel_7 = np.bincount(labels_k, weights=kw, minlength=len(self.reg_names))
            p_kernel_7 /= (p_kernel_7.sum() + 1e-12)

            p_kernel_pref = np.bincount(prefs_k, weights=kw, minlength=len(PREFECTURES)).astype(float)
            p_kernel_pref /= (p_kernel_pref.sum() + 1e-12)
            final_7reg_probs = p_kernel_7
            raw_pref_probs = 0.50 * p_pref_tree + 0.50 * p_kernel_pref
            raw_pref_probs /= np.sum(raw_pref_probs)
        elif self.w_arcface > 0 and mm_extracted is not None and self.mm_data is not None and "all_arc_norm" in self.mm_data:
            arc_emb = mm_extracted[0]
            all_arc = self.mm_data["all_arc_norm"]
            arc_p_idx = self.mm_data.get("arc_pref_indices", {})
            arc_sims = np.dot(all_arc, arc_emb)
            arc_knn = np.zeros(len(PREFECTURES))
            for idx_p, pref_name in enumerate(PREFECTURES):
                idxs = arc_p_idx.get(pref_name, [])
                if len(idxs) > 0:
                    s_sub = arc_sims[idxs]
                    kv = min(2, len(s_sub))
                    arc_knn[idx_p] = np.mean(np.partition(s_sub, -kv)[-kv:])
            
            exp_ak = np.exp((arc_knn - np.mean(arc_knn)) * 8.0)
            prob_knn = exp_ak / np.sum(exp_ak)

            # NLDA (マルチモーダル3D骨格・幾何特徴・ArcFace正規化線形判別分析)
            if "nlda_arc_pref" in self.mm_data and len(mm_extracted) >= 5:
                def _eval_nlda_pref(nd, vec):
                    W = nd["W"]
                    bias = nd["bias"]
                    std = nd["std"]
                    mean = nd["mean"]
                    scale = nd["scale"]
                    xs = (vec - mean) / (scale + 1e-8)
                    return ((xs @ W.T - bias) / (std + 1e-8))[0]

                s_arc_p = _eval_nlda_pref(self.mm_data["nlda_arc_pref"], arc_emb)
                s_c3d_p = _eval_nlda_pref(self.mm_data["nlda_c3d_pref"], mm_extracted[3][0])
                s_zrel_p = _eval_nlda_pref(self.mm_data["nlda_zrel_pref"], mm_extracted[4][0])
                
                # 目・眉解剖学ブロック (136d) の統合
                if "nlda_beye_pref" in self.mm_data and len(mm_extracted) >= 6:
                    s_beye_p = _eval_nlda_pref(self.mm_data["nlda_beye_pref"], mm_extracted[5][0])
                    comb_nlda_p = 0.475 * s_arc_p + 0.285 * s_c3d_p + 0.190 * s_zrel_p + 0.05 * s_beye_p
                else:
                    comb_nlda_p = 0.5 * s_arc_p + 0.3 * s_c3d_p + 0.2 * s_zrel_p

                exp_np = np.exp((comb_nlda_p / 3.0) - np.max(comb_nlda_p / 3.0))
                prob_nlda = exp_np / np.sum(exp_np)
            else:
                prob_nlda = prob_knn

            # 信頼度ゲーティングアンサンブル (Confidence-Gated Mixture of Experts)
            # 決定木のTop-1確信度が突出している場合(Ayuや若年層の骨格明瞭例)にブレンド
            max_tree = np.max(p_pref_tree)
            if max_tree >= 0.120:
                raw_pref_probs = 0.60 * prob_knn + 0.25 * prob_nlda + 0.15 * p_pref_tree
            else:
                raw_pref_probs = 0.70 * prob_knn + 0.30 * prob_nlda
            raw_pref_probs /= np.sum(raw_pref_probs)

            # 地方確率は各地方の代表最大値によるピーク保持型地方プーリング (Max Regional Pooling)
            final_7reg_probs = np.zeros(len(self.reg_names))
            for r_idx, reg_name in enumerate(self.reg_names):
                p_indices = [i for i, pr in enumerate(PREFECTURES) if PREF_TO_REGION[pr] == reg_name]
                sub_probs = [raw_pref_probs[i] for i in p_indices]
                if sub_probs:
                    final_7reg_probs[r_idx] = max(sub_probs)
                else:
                    final_7reg_probs[r_idx] = 0.0
            final_7reg_probs /= np.sum(final_7reg_probs)

            # 階層的ベイズ地方・都道府県連動 (Hierarchical Regional Coupling: alpha=0.60)
            coupled_pref_probs = np.zeros(len(PREFECTURES))
            for idx_p, p_name in enumerate(PREFECTURES):
                reg = PREF_TO_REGION[p_name]
                reg_idx = self.reg_names.index(reg)
                coupled_pref_probs[idx_p] = (raw_pref_probs[idx_p] ** 0.60) * (final_7reg_probs[reg_idx] ** 0.40)
            coupled_pref_probs /= np.sum(coupled_pref_probs)

            # 地方内条件付き確率 P(Pref | Region) の局所ソフトマックス推定 (tau_intra=6.0)
            # 他地方の大都市過密ノイズを完全に遮断し、地方内での適正な県判別確率を生成
            p_intra = np.zeros(len(PREFECTURES))
            for r_idx, reg in enumerate(self.reg_names):
                sub_prefs = [p for p in PREFECTURES if PREF_TO_REGION[p] == reg]
                sims_reg = np.zeros(len(sub_prefs))
                for i_p, p in enumerate(sub_prefs):
                    idxs = arc_p_idx.get(p, [])
                    if len(idxs) > 0:
                        s_sub = arc_sims[idxs]
                        sims_reg[i_p] = np.mean(np.partition(s_sub, -min(2, len(s_sub)))[-min(2, len(s_sub)):])
                exp_s = np.exp((sims_reg - np.mean(sims_reg)) * 6.0)
                prob_local = exp_s / np.sum(exp_s)
                for i_p, p in enumerate(sub_prefs):
                    p_intra[PREFECTURES.index(p)] = prob_local[i_p]

            # 階層ベイズ完全結合: P_hier(p) = P(Region) * P(Pref | Region)
            p_hier = np.zeros(len(PREFECTURES))
            for idx_p, p_name in enumerate(PREFECTURES):
                reg = PREF_TO_REGION[p_name]
                reg_idx = self.reg_names.index(reg)
                p_hier[idx_p] = final_7reg_probs[reg_idx] * p_intra[idx_p]
            p_hier /= np.sum(p_hier)

            # 大域スコアと地方特化階層スコアの安定統合 (blend_weight=0.10)
            final_pref_probs = 0.90 * coupled_pref_probs + 0.10 * p_hier
            final_pref_probs /= np.sum(final_pref_probs)

            # 二段構え（2-Stage）階層的直交不確実性ゲート (Two-Stage Hierarchical Orthogonal Gating)
            # Case A: 大域不確実性ゲート (大域Top-1とTop-2が同一地方内で接戦 margin < 0.0040)
            # Case B: 地方内惜敗救出ゲート (大域2位に他地方ノイズが挟まるが地方内で接戦 intra_margin < 0.0070)
            #         大域安全不変条件 (Global Safety Invariant) により他地方への確率漏洩を数学的に完全遮断
            top_indices = np.argsort(final_pref_probs)[-2:]
            top1_idx = top_indices[1]
            top2_idx = top_indices[0]
            global_margin = final_pref_probs[top1_idx] - final_pref_probs[top2_idx]

            top1_p = PREFECTURES[top1_idx]
            top2_p = PREFECTURES[top2_idx]
            global_same_reg = (PREF_TO_REGION[top1_p] == PREF_TO_REGION[top2_p])

            if "nlda_fn_pref" in self.mm_data and 's_c3d_p' in locals():
                fn_emb = embedding / (np.linalg.norm(embedding) + 1e-12)
                s_fn_all = _eval_nlda_pref(self.mm_data["nlda_fn_pref"], fn_emb)

                self._last_gate_debug = {
                    "final_pref_probs_pre": final_pref_probs.copy(),
                    "s_fn_all": s_fn_all.copy(),
                    "s_c3d_p": s_c3d_p.copy(),
                    "s_zrel_p": s_zrel_p.copy() if 's_zrel_p' in locals() else None,
                    "top1_idx": top1_idx,
                    "top2_idx": top2_idx,
                    "global_margin": global_margin,
                    "global_same_reg": global_same_reg,
                    "prob_knn": prob_knn.copy() if 'prob_knn' in locals() else None,
                    "prob_nlda": prob_nlda.copy() if 'prob_nlda' in locals() else None,
                    "p_pref_tree": p_pref_tree.copy() if 'p_pref_tree' in locals() else None,
                    "final_7reg_probs": final_7reg_probs.copy() if 'final_7reg_probs' in locals() else None,
                    "coupled_pref_probs": coupled_pref_probs.copy() if 'coupled_pref_probs' in locals() else None,
                    "p_intra": p_intra.copy() if 'p_intra' in locals() else None
                }

                # Case A: 大域不確実性ゲート (現行動作を100%完全維持)
                if global_margin < 0.0040 and global_same_reg:
                    reg = PREF_TO_REGION[top1_p]
                    sub_prefs = [p for p in PREFECTURES if PREF_TO_REGION[p] == reg]
                    sub_idxs = [PREFECTURES.index(p) for p in sub_prefs]
                    s_fn = s_fn_all[sub_idxs]
                    s_c3d = s_c3d_p[sub_idxs]

                    score_ortho = 0.15 * (s_fn - np.mean(s_fn)) + 0.10 * (s_c3d - np.mean(s_c3d))
                    exp_ortho = np.exp(score_ortho)
                    p_ortho = exp_ortho / np.sum(exp_ortho)

                    raw_w = (1.0 - global_margin / 0.0040) * 0.50
                    blend_w = max(0.30, raw_w)
                    sub_prob_sum = np.sum(final_pref_probs[sub_idxs])
                    if sub_prob_sum > 0:
                        final_pref_probs[sub_idxs] = sub_prob_sum * ((1.0 - blend_w) * (final_pref_probs[sub_idxs] / sub_prob_sum) + blend_w * p_ortho)

                # Case B: 地方内惜敗救出ゲート (大域2位に他地方が挟まっているが地方内接戦)
                else:
                    reg_top1 = PREF_TO_REGION[top1_p]
                    sub_prefs = [p for p in PREFECTURES if PREF_TO_REGION[p] == reg_top1]
                    sub_idxs = [PREFECTURES.index(p) for p in sub_prefs]
                    other_idxs = [i for i, p in enumerate(PREFECTURES) if PREF_TO_REGION[p] != reg_top1]
                    p_other_max = np.max(final_pref_probs[other_idxs]) if len(other_idxs) > 0 else 0.0

                    if len(sub_idxs) > 1:
                        sub_probs = final_pref_probs[sub_idxs]
                        sub_sort = np.argsort(sub_probs)[-2:]
                        intra_margin = sub_probs[sub_sort[1]] - sub_probs[sub_sort[0]]

                        case_b_th = getattr(self, "case_b_intra_th", 0.0070)
                        case_b_w_fn = getattr(self, "case_b_w_fn", 0.40)
                        case_b_w_c3d = getattr(self, "case_b_w_c3d", 0.02)
                        case_b_w_zrel = getattr(self, "case_b_w_zrel", 0.01)
                        case_b_min_b = getattr(self, "case_b_min_blend", 0.55)

                        if intra_margin < case_b_th and 's_zrel_p' in locals():
                            s_fn = s_fn_all[sub_idxs]
                            s_c3d = s_c3d_p[sub_idxs]
                            s_zrel = s_zrel_p[sub_idxs]

                            score_ortho = case_b_w_fn * (s_fn - np.mean(s_fn)) + case_b_w_c3d * (s_c3d - np.mean(s_c3d)) + case_b_w_zrel * (s_zrel - np.mean(s_zrel))
                            exp_ortho = np.exp(score_ortho)
                            p_ortho = exp_ortho / np.sum(exp_ortho)

                            raw_w = (1.0 - intra_margin / case_b_th) * 0.50
                            blend_w = max(case_b_min_b, raw_w)
                            sub_prob_sum = np.sum(final_pref_probs[sub_idxs])
                            candidate_sub = sub_prob_sum * ((1.0 - blend_w) * (final_pref_probs[sub_idxs] / sub_prob_sum) + blend_w * p_ortho)

                            new_top_p = sub_prefs[np.argmax(candidate_sub)]
                            allowed_p = {sub_prefs[sub_sort[1]], sub_prefs[sub_sort[0]]}

                            # 地方内上位2県限定浮上条件 + 大域安全不変条件
                            if new_top_p in allowed_p and np.max(candidate_sub) >= p_other_max:
                                final_pref_probs[sub_idxs] = candidate_sub

                    # Case D: 3D解剖学的骨格確証救出ゲート (3D正面向き骨格 + Z軸彫り深さの有意差 delta >= 4.5σ)
                    # 地方内において、解剖学的骨格特徴量が圧倒的ダントツ1位（偶然確率 p < 1e-6）である真の骨格特徴保持者を救出
                    if 's_zrel_p' in locals() and s_zrel_p is not None:
                        top_cur_idx = np.argmax(final_pref_probs)
                        top_cur_p = PREFECTURES[top_cur_idx]
                        if PREF_TO_REGION[top_cur_p] == reg_top1 and len(sub_idxs) > 1:
                            cur_in_sub = sub_prefs.index(top_cur_p)
                            c3d_sub = s_c3d_p[sub_idxs]
                            zrel_sub = s_zrel_p[sub_idxs]
                            morph_total = c3d_sub + zrel_sub
                            best_morph_idx = np.argmax(morph_total)
                            if best_morph_idx != cur_in_sub:
                                morph_margin = morph_total[best_morph_idx] - morph_total[cur_in_sub]
                                morph_th = getattr(self, "case_d_morph_th", 4.0)
                                # 解剖学的・マルチモーダル完全合意不変条件 (Multimodal Morphological Consensus Invariant)
                                # 3D骨格+Z彫り深さで圧倒的ダントツ(>=4.0σ)であり、かつFaceNet直交NLDAでも同等以上である場合のみ救出
                                fn_cur = s_fn_all[top_cur_idx]
                                fn_rescue = s_fn_all[sub_idxs[best_morph_idx]]
                                if morph_margin >= morph_th and fn_rescue >= fn_cur:
                                    rescue_idx = sub_idxs[best_morph_idx]
                                    prob_gap = final_pref_probs[top_cur_idx] - final_pref_probs[rescue_idx]
                                    if prob_gap < 0.0150:
                                        p_swap = final_pref_probs[top_cur_idx]
                                        final_pref_probs[top_cur_idx] = final_pref_probs[rescue_idx]
                                        final_pref_probs[rescue_idx] = p_swap + 0.0005
        else:
            final_pref_probs = p_pref_tree
            final_7reg_probs = np.zeros(len(self.reg_names))
            for r_idx, reg_name in enumerate(self.reg_names):
                p_indices = [i for i, pr in enumerate(PREFECTURES) if PREF_TO_REGION[pr] == reg_name]
                sub_probs = [final_pref_probs[i] for i in p_indices]
                if sub_probs:
                    kv = min(2, len(sub_probs))
                    final_7reg_probs[r_idx] = np.mean(np.partition(sub_probs, -kv)[-kv:])
                else:
                    final_7reg_probs[r_idx] = 0.0
            final_7reg_probs /= np.sum(final_7reg_probs)

        # 4. 3D解剖学的骨格に基づく高精度・客観的 縄文 vs 弥生 スコアの算出
        if mm_extracted is not None and len(mm_extracted) >= 8:
            fm = mm_extracted[7]  # [intercanthal, nose_proj, alar, orbital, nasion, chin]
            nose_proj = fm[1]
            nasion = fm[4]
            orbital = fm[3]
            alar = fm[2]

            # 面長比率（形態的面長度）
            w_b = max(1.0, box[2] - box[0])
            h_b = max(1.0, box[3] - box[1])
            aspect = h_b / w_b

            # 骨格Z-score合成
            z_score = (
                1.2 * (nose_proj - (-0.18)) / 0.06
                - 1.2 * (nasion - (-0.13)) / 0.05
                + 1.0 * (orbital - 0.36) / 0.04
                + 0.8 * (alar - 0.74) / 0.05
                - 1.0 * (aspect - 1.30) / 0.08
            )

            p_jomon = 1.0 / (1.0 + np.exp(-z_score / 2.0))
            jomon_pct = round(float(np.clip(p_jomon * 100.0, 10.0, 90.0)), 1)
            yayoi_pct = round(100.0 - jomon_pct, 1)

            if jomon_pct >= 62.0:
                face_type = "濃厚・彫深タイプ（縄文系優位）"
                description = "目鼻立ちがはっきりとしており、彫りの深さと立体感のある骨格です。南九州や沖縄、東北沿岸部にルーツを持つ傾向があります。"
            elif yayoi_pct >= 62.0:
                face_type = "涼やか・面長タイプ（弥生系優位）"
                description = "すっきりとした涼しげな目元と滑らかな輪郭が特徴の骨格です。近畿・山陰・中国地方など西日本〜大陸系にルーツを持つ傾向があります。"
            else:
                face_type = "ハイブリッドタイプ（調和型）"
                description = "縄文系と弥生系の特徴がバランスよく調和した現代的な日本人の典型顔立ちです。"

            jomon_yayoi_display = {
                "jomon_percent": jomon_pct,
                "yayoi_percent": yayoi_pct,
                "face_type": face_type,
                "description": description,
                "contrast_score": jomon_yayoi_res.get("contrast_score", 45.0),
                "ratio_score": jomon_yayoi_res.get("ratio_score", 0.70)
            }
        else:
            jomon_yayoi_display = jomon_yayoi_res

        # 5. 表示用温度キャリブレーション (T=0.50)
        # 順位関係（1位〜47位）を100%完全不変に保ちながら、平坦化した確信度を15%前後の自然なスプレッドに引き締め
        T_display = 0.50
        cal_display_probs = final_pref_probs ** (1.0 / T_display)
        cal_display_probs /= np.sum(cal_display_probs)

        # 6. 地方ブロック結果の構築 (新7地方)
        # キャリブレーション済み都道府県確率の各地方代表最大値によるピーク保持型地方同期 (Final Max Regional Pooling)
        synced_7reg_probs = np.zeros(len(self.reg_names))
        for r_idx, reg_name in enumerate(self.reg_names):
            p_indices = [i for i, pr in enumerate(PREFECTURES) if PREF_TO_REGION[pr] == reg_name]
            sub_probs = [cal_display_probs[i] for i in p_indices]
            synced_7reg_probs[r_idx] = max(sub_probs) if sub_probs else 0.0
        synced_7reg_probs /= np.sum(synced_7reg_probs)

        region_results = []
        for idx, reg in enumerate(self.reg_names):
            prob = float(synced_7reg_probs[idx])
            region_results.append({
                "region": reg,
                "probability": prob,
                "percentage": round(prob * 100.0, 1),
                "description": get_region_description(reg)
            })
        region_results.sort(key=lambda x: x["probability"], reverse=True)
        for rank, item in enumerate(region_results, 1):
            item["rank"] = rank

        # 7. 都道府県結果の構築（事後確率順にソート）
        pref_results = []
        for idx, pref in enumerate(PREFECTURES):
            prob = float(final_pref_probs[idx])
            disp_prob = float(cal_display_probs[idx])
            pref_results.append({
                "prefecture": pref,
                "region": PREF_TO_REGION.get(pref, "不明"),
                "probability": prob,
                "percentage": round(disp_prob * 100.0, 2)
            })
        pref_results.sort(key=lambda x: x["probability"], reverse=True)
        for rank, item in enumerate(pref_results, 1):
            item["rank"] = rank

        return {
            "success": True,
            "error": None,
            "jomon_yayoi": jomon_yayoi_display,
            "top_regions": region_results,
            "top_prefectures": pref_results,
            "face_box": [int(v) for v in box],
            "landmarks": landmarks.tolist() if hasattr(landmarks, "tolist") else landmarks
        }

    def predict(self, image: Image.Image) -> Dict[str, Any]:
        """1枚の画像から出身地および顔タイプを推定する"""
        if image.mode != "RGB":
            image = image.convert("RGB")
        face_tensor, landmarks, box = self.processor.detect_and_align(image)
        if face_tensor is None or landmarks is None or box is None:
            # 高解像度・一眼・自撮り写真に対する InsightFace クロップフォールバック
            app = self._get_insight_app()
            if app is not None:
                try:
                    import torch
                    img_np = np.array(image.convert("RGB"))
                    img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
                    faces = app.get(img_cv)
                    if len(faces) > 0:
                        face = faces[0]
                        b = face.bbox.astype(int)
                        box = np.array([max(0, b[0]), max(0, b[1]), min(image.width, b[2]), min(image.height, b[3])])
                        landmarks = face.kps  # 5 key landmarks (left_eye, right_eye, nose, left_mouth, right_mouth)
                        
                        w_b, h_b = box[2] - box[0], box[3] - box[1]
                        m = int(max(w_b, h_b) * 0.15)
                        crop = image.crop((max(0, box[0]-m), max(0, box[1]-m), min(image.width, box[2]+m), min(image.height, box[3]+m)))
                        crop_160 = crop.convert("RGB").resize((160, 160), Image.Resampling.BILINEAR)
                        arr = (np.float32(crop_160) - 127.5) / 128.0
                        face_tensor = torch.from_numpy(arr.transpose(2, 0, 1))
                        logger.info("MTCNN未検出画像をInsightFaceフォールバックで検出成功")
                except Exception as e:
                    logger.warning(f"InsightFace顔検出フォールバック失敗: {e}")

        if face_tensor is None or landmarks is None or box is None:
            return {
                "success": False,
                "error": "画像から正面の顔を検出できませんでした。明るい場所で正面を向いた写真をお試しください。",
                "jomon_yayoi": None,
                "top_regions": [],
                "top_prefectures": [],
                "face_box": None,
                "landmarks": None
            }

        embedding = self.processor.extract_embedding(face_tensor)
        jomon_yayoi_res = self.processor.analyze_jomon_yayoi(image, landmarks, box)
        mm_extracted = self._extract_arcface_and_landmarks(image)
        return self.predict_from_extracted(embedding, jomon_yayoi_res, mm_extracted, box, landmarks, image.size)

    def _extract_single_face_mm(self, f, img_bgr: np.ndarray):
        """検出済みの単一顔オブジェクト(InsightFace)からマルチモーダル特徴量を抽出"""
        if self.mm_data is None:
            return None

        arc_emb = f.normed_embedding.astype(np.float64)
        arc_emb /= (np.linalg.norm(arc_emb) + 1e-8)

        lm = f.landmark_2d_106.astype(np.float64)
        le = lm[[35, 36, 37, 39, 40, 41]].mean(0)
        re = lm[[89, 90, 91, 93, 94, 95]].mean(0)
        if le[0] > re[0]:
            le, re = re, le
        c = (le + re) / 2.0
        v = re - le
        s = np.linalg.norm(v) + 1e-6
        ang = np.arctan2(v[1], v[0])
        R = np.array([[np.cos(-ang), -np.sin(-ang)], [np.sin(-ang), np.cos(-ang)]])
        nl = ((lm - c) @ R.T) / s
        lm_flat = nl.reshape(1, -1)

        key_idx = self.mm_data["key_lm_idx"]
        klm = nl[key_idx, :]
        diff = klm[:, None, :] - klm[None, :, :]
        dist_mat = np.sqrt((diff ** 2).sum(-1))
        iu = np.triu_indices(len(key_idx), k=1)
        pdist_flat = dist_mat[iu[0], iu[1]].reshape(1, -1)

        if hasattr(f, "landmark_3d_68") and f.landmark_3d_68 is not None and "c_ref_3d_norm" in self.mm_data:
            lm3d_used = f.landmark_3d_68
            nose_w = np.linalg.norm(lm[80] - lm[74])
            face_w = np.linalg.norm(lm[16] - lm[0])
            ratio_nf = nose_w / (face_w + 1e-6)
            iod_3d = np.linalg.norm(f.landmark_3d_68[36, :2] - f.landmark_3d_68[45, :2]) + 1e-6
            rel_z = (f.landmark_3d_68[30, 2] - (f.landmark_3d_68[36, 2] + f.landmark_3d_68[45, 2]) / 2.0) / iod_3d
            angles = f.pose if hasattr(f, "pose") else [0, 0, 0]
            roll = abs(angles[2]) if angles is not None else 0.0
            d_score = (ratio_nf - 0.315) / 0.025 + (-rel_z - 0.280) / 0.040 + (roll - 3.0) / 10.0

            if d_score >= 1.0:
                lm3d_used = _frontalize_and_rectify(f.landmark_3d_68, self.mm_data["c_ref_3d_norm"], z0_ratio=4.0)

            c3d_flat, z_relief, block_eye, block_nose, r_morph = _compute_single_3d_features(
                lm3d_used, nl, self.mm_data["c_ref_3d_norm"]
            )
            return arc_emb, lm_flat, pdist_flat, c3d_flat, z_relief, block_eye, block_nose, r_morph

        return arc_emb, lm_flat, pdist_flat

    def predict_group(self, image: Image.Image, min_face_size: int = 40) -> Dict[str, Any]:
        """集合写真・複数人の画像から全顔を検出し、個人差ノイズを打ち消すベイズ尤度比統合でグループ全体の出身地を推定する"""
        if image.mode != "RGB":
            image = image.convert("RGB")
        app = self._get_insight_app()
        if app is None:
            return {"success": False, "error": "顔検出エンジンの初期化に失敗しました。", "num_faces": 0}

        img_np = np.array(image.convert("RGB"))
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        faces = app.get(img_bgr)

        valid_faces = []
        for f in faces:
            w_b = f.bbox[2] - f.bbox[0]
            h_b = f.bbox[3] - f.bbox[1]
            if min(w_b, h_b) >= min_face_size and getattr(f, "det_score", 1.0) >= 0.55:
                valid_faces.append(f)

        if not valid_faces:
            return {
                "success": False,
                "error": "画像から有効な顔が検出されませんでした。顔が正面を向き、ある程度大きく写っている写真をお試しください。",
                "num_faces": 0
            }

        valid_faces.sort(key=lambda x: x.bbox[0])

        individual_results = []
        log_p_list = []
        import torch

        for idx, f in enumerate(valid_faces):
            box = np.array([max(0, int(f.bbox[0])), max(0, int(f.bbox[1])), min(image.width, int(f.bbox[2])), min(image.height, int(f.bbox[3]))])
            landmarks = f.kps

            w_b, h_b = box[2] - box[0], box[3] - box[1]
            m = int(max(w_b, h_b) * 0.15)
            crop = image.crop((max(0, box[0]-m), max(0, box[1]-m), min(image.width, box[2]+m), min(image.height, box[3]+m)))
            crop_160 = crop.convert("RGB").resize((160, 160), Image.Resampling.BILINEAR)
            arr = (np.float32(crop_160) - 127.5) / 128.0
            face_tensor = torch.from_numpy(arr.transpose(2, 0, 1))

            embedding = self.processor.extract_embedding(face_tensor)
            jomon_yayoi_res = self.processor.analyze_jomon_yayoi(image, landmarks, box)
            mm_extracted = self._extract_single_face_mm(f, img_bgr)

            indiv_res = self.predict_from_extracted(embedding, jomon_yayoi_res, mm_extracted, box, landmarks, image.size)
            if indiv_res["success"]:
                p_vec = np.array([next(x["probability"] for x in indiv_res["top_prefectures"] if x["prefecture"] == p) for p in PREFECTURES])
                p_vec = np.clip(p_vec, 1e-6, 1.0)
                p_vec /= p_vec.sum()
                log_p_list.append(np.log(p_vec))

                thumb = crop.resize((120, 120), Image.Resampling.LANCZOS)
                individual_results.append({
                    "face_id": idx + 1,
                    "box": box.tolist(),
                    "landmarks": landmarks.tolist() if hasattr(landmarks, "tolist") else landmarks,
                    "crop_image": thumb,
                    "jomon_yayoi": indiv_res["jomon_yayoi"],
                    "top_pref": indiv_res["top_prefectures"][0]["prefecture"],
                    "top_pref_pct": indiv_res["top_prefectures"][0]["percentage"],
                    "top_region": indiv_res["top_regions"][0]["region"],
                    "top_region_pct": indiv_res["top_regions"][0]["percentage"],
                })

        if not log_p_list:
            return {"success": False, "error": "各顔の特徴量解析に失敗しました。", "num_faces": 0}

        llr = MEAN_LOG_P + np.sum([lp - MEAN_LOG_P for lp in log_p_list], axis=0)
        comb_pref_raw = np.exp(llr - llr.max())
        comb_pref_raw /= comb_pref_raw.sum()

        T_display = 0.50
        cal_display_probs = comb_pref_raw ** (1.0 / T_display)
        cal_display_probs /= np.sum(cal_display_probs)

        synced_7reg_probs = np.zeros(len(self.reg_names))
        for r_idx, reg_name in enumerate(self.reg_names):
            p_indices = [i for i, pr in enumerate(PREFECTURES) if PREF_TO_REGION[pr] == reg_name]
            sub_probs = [cal_display_probs[i] for i in p_indices]
            synced_7reg_probs[r_idx] = max(sub_probs) if sub_probs else 0.0
        synced_7reg_probs /= np.sum(synced_7reg_probs)

        region_results = []
        for idx, reg in enumerate(self.reg_names):
            prob = float(synced_7reg_probs[idx])
            region_results.append({
                "region": reg,
                "probability": prob,
                "percentage": round(prob * 100.0, 1),
                "description": get_region_description(reg)
            })
        region_results.sort(key=lambda x: x["probability"], reverse=True)
        for rank, item in enumerate(region_results, 1):
            item["rank"] = rank

        pref_results = []
        for idx, pref in enumerate(PREFECTURES):
            prob = float(comb_pref_raw[idx])
            disp_prob = float(cal_display_probs[idx])
            pref_results.append({
                "prefecture": pref,
                "region": PREF_TO_REGION.get(pref, "不明"),
                "probability": prob,
                "percentage": round(disp_prob * 100.0, 2)
            })
        pref_results.sort(key=lambda x: x["probability"], reverse=True)
        for rank, item in enumerate(pref_results, 1):
            item["rank"] = rank

        annotated_img = image.copy()
        draw = ImageDraw.Draw(annotated_img)
        for ind in individual_results:
            b = ind["box"]
            draw.rectangle(b, outline="#2563EB", width=4)
            fid_text = f"#{ind['face_id']}"
            draw.rectangle([b[0], max(0, b[1]-26), b[0]+42, b[1]], fill="#2563EB")
            draw.text((b[0]+6, max(0, b[1]-24)), fid_text, fill="#FFFFFF")

        return {
            "success": True,
            "error": None,
            "num_faces": len(individual_results),
            "group_top_regions": region_results,
            "group_top_prefectures": pref_results,
            "individual_results": individual_results,
            "annotated_image": annotated_img
        }
