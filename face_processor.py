"""
顔画像処理・特徴量抽出・顔タイプ（縄文/弥生）分析モジュール
"""

from typing import Dict, Optional, Tuple, Any, List
import numpy as np
import cv2
import torch
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1

class FaceProcessor:
    """顔検出、アライメント、特徴量抽出、縄文/弥生顔分析を行うクラス"""

    def __init__(self, device: Optional[str] = None):
        if device is None:
            self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
            
        # 顔検出・ランドマーク検出器（MTCNN）
        self.mtcnn = MTCNN(
            image_size=160,
            margin=20,
            min_face_size=40,
            thresholds=[0.6, 0.7, 0.7],
            factor=0.709,
            post_process=True,
            device=self.device,
            keep_all=False  # 最もスコアの高い顔を1つ抽出
        )
        
        # 512次元顔埋め込みモデル（FaceNet）
        self.resnet = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)

    def detect_single_face(self, image: Image.Image) -> Tuple[Optional[torch.Tensor], Optional[np.ndarray], Optional[np.ndarray]]:
        """
        画像中に『正面の顔がちょうど1人だけ』写っている場合のみ検出し、
        テンソル・ランドマーク・バウンディングボックスを返す。
        複数人（ツーショット、集合写真、別人の混入）が写っている場合は None を返す。
        """
        if image.mode != 'RGB':
            image = image.convert('RGB')

        boxes, probs, landmarks = self.mtcnn.detect(image, landmarks=True)

        if boxes is None or len(boxes) == 0:
            return None, None, None

        # 確信度85%以上の有効な顔の数をカウント
        valid_faces = []
        for i, (b, p) in enumerate(zip(boxes, probs)):
            if p is not None and p >= 0.85:
                w = b[2] - b[0]
                h = b[3] - b[1]
                # 小さすぎる顔（背景の通行人など）でなければ有効
                if w >= 50 and h >= 50:
                    valid_faces.append(i)

        # ちょうど1人だけ写っている単独写真でなければ除外
        if len(valid_faces) != 1:
            return None, None, None

        idx = valid_faces[0]
        box = boxes[idx]
        pts = landmarks[idx]

        # クロップ顔テンソル取得
        face_tensor = self.mtcnn(image)
        if face_tensor is None:
            return None, pts, box

        return face_tensor, pts, box

    def crop_face_pil(self, image: Image.Image, box: np.ndarray, target_size: int = 256, margin_ratio: float = 0.25) -> Image.Image:
        """
        バウンディングボックス周辺（目・鼻・口・顎ライン＋マージン）を正方形にクロップして返す。
        余計な背景や別の人物の写り込みを完全にカットする。
        """
        if image.mode != 'RGB':
            image = image.convert('RGB')

        img_w, img_h = image.size
        x1, y1, x2, y2 = box
        bw = x2 - x1
        bh = y2 - y1

        # 正方形になるように中心から拡張
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        side = max(bw, bh) * (1.0 + margin_ratio)

        nx1 = max(0, int(cx - side / 2.0))
        ny1 = max(0, int(cy - side / 2.0))
        nx2 = min(img_w, int(cx + side / 2.0))
        ny2 = min(img_h, int(cy + side / 2.0))

        cropped = image.crop((nx1, ny1, nx2, ny2))
        return cropped.resize((target_size, target_size), Image.Resampling.LANCZOS)

    def check_face_distortion(
        self,
        landmarks: np.ndarray,
        image_size: Tuple[int, int] = (256, 256)
    ) -> Tuple[bool, List[str]]:
        """
        顔ランドマーク（両目、鼻、両口角の5点）を用いて、レンズ歪み・パース歪み・極端な傾き・煽り・見下ろしを判定。
        戻り値: (is_distorted: bool, reasons: List[str])
        """
        if landmarks is None or len(landmarks) < 5:
            return True, ["ランドマーク検出不可"]

        le, re, nose, lm, rm = landmarks[0], landmarks[1], landmarks[2], landmarks[3], landmarks[4]
        reasons = []

        # 1. ロール角（首の傾き）
        dx = re[0] - le[0]
        dy = re[1] - le[1]
        roll_deg = abs(np.degrees(np.arctan2(dy, dx)))
        if roll_deg > 15.0:
            reasons.append(f"首の傾き歪み({roll_deg:.1f}°)")

        # 2. 目の間隔
        eye_dist = np.hypot(dx, dy)
        if eye_dist <= 0:
            return True, ["両目間隔異常"]

        # 3. 左右対称性（ヨー角・横向きパース歪み）
        d_le_nose = abs(nose[0] - le[0])
        d_re_nose = abs(re[0] - nose[0])
        max_d = max(d_le_nose, d_re_nose)
        if max_d > 0:
            sym_ratio = min(d_le_nose, d_re_nose) / max_d
            if sym_ratio < 0.45:
                reasons.append(f"横向きパース歪み(対称度{sym_ratio:.2f})")

        # 4. 上下ピッチ比（見下ろし・上目遣い・煽りパース）
        eye_mid = ((le[0] + re[0]) / 2.0, (le[1] + re[1]) / 2.0)
        mouth_mid = ((lm[0] + rm[0]) / 2.0, (lm[1] + rm[1]) / 2.0)
        d_eye_nose = abs(nose[1] - eye_mid[1])
        d_nose_mouth = abs(mouth_mid[1] - nose[1])
        if d_nose_mouth > 0:
            pitch_ratio = d_eye_nose / d_nose_mouth
            if pitch_ratio > 3.4:
                reasons.append(f"上目遣い・見下ろしパース歪み(比{pitch_ratio:.2f})")
            elif pitch_ratio < 1.1:
                reasons.append(f"下からの煽りパース歪み(比{pitch_ratio:.2f})")

        # 5. 顔パーツ縦横比（広角・魚眼・アスペクト比潰れ）
        vert_dist = np.hypot(mouth_mid[0] - eye_mid[0], mouth_mid[1] - eye_mid[1])
        aspect_ratio = vert_dist / eye_dist
        if aspect_ratio < 0.88 or aspect_ratio > 1.18:
            reasons.append(f"顔パーツ縦横比歪み(比{aspect_ratio:.2f})")

        # 6. クロップ境界の端切れチェック
        w, h = image_size
        min_x = min(le[0], re[0], nose[0], lm[0], rm[0])
        max_x = max(le[0], re[0], nose[0], lm[0], rm[0])
        min_y = min(le[1], re[1])
        max_y = max(lm[1], rm[1])
        if min_x < 25 or max_x > (w - 25) or min_y < 25 or max_y > (h - 25):
            reasons.append("顔パーツ端切れ歪み")

        return (len(reasons) > 0), reasons

    def detect_and_align(self, image: Image.Image) -> Tuple[Optional[torch.Tensor], Optional[np.ndarray], Optional[np.ndarray]]:
        """推論用: 画面に入力された顔を検出"""
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
        boxes, probs, landmarks = self.mtcnn.detect(image, landmarks=True)
        
        if boxes is None or len(boxes) == 0 or probs[0] is None or probs[0] < 0.75:
            return None, None, None
            
        box = boxes[0]
        pts = landmarks[0]
        
        face_tensor = self.mtcnn(image)
        if face_tensor is None:
            return None, pts, box
            
        return face_tensor, pts, box

    def extract_embedding(self, face_tensor: torch.Tensor) -> np.ndarray:
        """
        クロップされた顔テンソルから512次元の顔特徴ベクトルを抽出
        """
        with torch.no_grad():
            tensor = face_tensor.unsqueeze(0).to(self.device)
            embedding = self.resnet(tensor).cpu().numpy()[0]
            # L2正規化
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
        return embedding

    def analyze_jomon_yayoi(self, image: Image.Image, landmarks: np.ndarray, box: np.ndarray) -> Dict[str, Any]:
        """
        顔の幾何学的比率と陰影（ホリの深さ）から縄文系 vs 弥生系スコアを算出する。
        
        人類学的知見に基づく特徴：
        - 縄文系：目力・大きな目、目と眉が近い（ホリ深）、鼻幅広、顔の陰影・凹凸が強い
        - 弥生系：切れ長の一重〜涼しげな目、目と眉が離れている（平坦）、直線的な鼻、あっさりした凹凸
        """
        # landmarks: 0:左目, 1:右目, 2:鼻先, 3:左口角, 4:右口角
        left_eye, right_eye, nose, left_mouth, right_mouth = landmarks
        
        # 1. 幾何比率の計算
        # 目の間隔
        eye_dist = np.linalg.norm(right_eye - left_eye)
        # 鼻幅（口幅との比率や目の間隔との比率）
        mouth_dist = np.linalg.norm(right_mouth - left_mouth)
        # 目の中点
        eye_center = (left_eye + right_eye) / 2.0
        # 口の中点
        mouth_center = (left_mouth + right_mouth) / 2.0
        
        # 目〜鼻の距離、鼻〜口の距離
        eye_nose_dist = np.linalg.norm(nose - eye_center)
        nose_mouth_dist = np.linalg.norm(mouth_center - nose)
        face_height = np.linalg.norm(mouth_center - eye_center)
        
        # 顔のバウンディングボックスの幅と高さ
        box_w = max(1.0, box[2] - box[0])
        box_h = max(1.0, box[3] - box[1])
        aspect_ratio = box_h / box_w  # 面長度（弥生系は高い）

        # 比率特徴
        # 縄文系は目と眉が近く、顔が引き締まっている（face_heightに対するeye_dist比率が高い）
        eye_to_height_ratio = eye_dist / max(1.0, face_height)
        
        # 2. 陰影・ホリの深さ（コントラスト）分析
        np_img = np.array(image.convert('L'))
        x1, y1, x2, y2 = [max(0, int(v)) for v in box]
        x2 = min(np_img.shape[1], x2)
        y2 = min(np_img.shape[0], y2)
        
        face_roi = np_img[y1:y2, x1:x2]
        if face_roi.size > 0:
            # ラプラシアン分散で顔の凹凸・エッジの強さ（ホリの深さ）を測定
            laplacian_var = cv2.Laplacian(face_roi, cv2.CV_64F).var()
            # 明暗コントラスト標準偏差
            contrast_std = np.std(face_roi)
        else:
            laplacian_var = 100.0
            contrast_std = 30.0

        # スコアリング（ベースラインからの偏差）
        # 縄文系指標：eye_to_height_ratio大、contrast_std大、aspect_ratio小
        # 標準的な日本人平均値を基準(0.0)とするz-score的モデリング
        score_component = 0.0
        
        # ホリの深さ（陰影コントラスト）
        score_component += (contrast_std - 38.0) * 1.5
        # 輪郭比率（面長なら弥生系、角・丸なら縄文系）
        score_component += (1.30 - aspect_ratio) * 40.0
        # 目の間隔比率
        score_component += (eye_to_height_ratio - 0.70) * 50.0

        # シグモイド関数で 0〜100% にマッピング
        jomon_prob = 1.0 / (1.0 + np.exp(-score_component / 20.0))
        jomon_percent = round(float(np.clip(jomon_prob * 100.0, 5.0, 95.0)), 1)
        yayoi_percent = round(100.0 - jomon_percent, 1)

        # 特徴解説コメント
        if jomon_percent >= 65:
            face_type = "濃厚・彫深タイプ（縄文系優位）"
            description = "目鼻立ちがはっきりとしており、立体感とメリハリのある顔立ちです。南九州や沖縄、東北沿岸部にルーツを持つ傾向があります。"
        elif yayoi_percent >= 65:
            face_type = "涼やか・面長タイプ（弥生系優位）"
            description = "すっきりとした涼しげな目元と滑らかな輪郭が特徴です。近畿・山陰・中国地方など西日本〜大陸系にルーツを持つ傾向があります。"
        else:
            face_type = "ハイブリッドタイプ（調和型）"
            description = "縄文系と弥生系の特徴がバランスよく調和した現代的な日本人の典型顔立ちです。"

        return {
            "jomon_percent": jomon_percent,
            "yayoi_percent": yayoi_percent,
            "face_type": face_type,
            "description": description,
            "aspect_ratio": round(aspect_ratio, 2),
            "eye_ratio": round(float(eye_to_height_ratio), 3),
            "contrast_score": round(contrast_std, 1)
        }
