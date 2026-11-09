"""
Supabase Storage 非同期画像・EXIF・診断結果アップローダー
- 動物顔診断と同一のSupabase Storage環境を使用し、専用フォルダ 'shusshinchi/' に完全分離保存
- ユーザーの生画像バイナリを1ビットも劣化・加工させずEXIF付きで完全保存
- 診断結果（推定地方・都道府県Top1〜5・縄文弥生スコア）およびGPSメタデータをJSON保存
- バックグラウンドスレッドで非同期実行し、Streamlit UIの推論速度・操作感を1ミリ秒も阻害しない
"""

import json
import logging
import threading
import time
import uuid
import urllib.request
import io
from typing import Dict, Any, Optional, Tuple
from PIL import Image, ExifTags

logger = logging.getLogger(__name__)

# Supabase Storage 設定（動物顔診断と共通のプロジェクト・バケット、専用フォルダで完全分離）
SUPABASE_URL = "https://cweuchydgpyumvnunmyn.supabase.co"
SUPABASE_ANON_KEY = (
    "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9."
    "eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImN3ZXVjaHlkZ3B5dW12bnVubXluIiwicm9sZSI6ImFub24iLCJpYXQiOjE3Nzc4NTM5NzgsImV4cCI6MjA5MzQyOTk3OH0."
    "qJ5kmfL6CD6OhYS3aC3aqEPQXZGwdiZ1pWK-Kd-4egE"
)
BUCKET_NAME = "images"
STORAGE_PREFIX = "shusshinchi"


def _dms_to_decimal(dms: Any, ref: str) -> Optional[float]:
    """度分秒（DMS）タプルを十進角（Decimal Degrees）に変換"""
    try:
        degrees = float(dms[0])
        minutes = float(dms[1]) / 60.0
        seconds = float(dms[2]) / 3600.0
        val = degrees + minutes + seconds
        if ref in ["S", "W"]:
            val = -val
        return round(val, 6)
    except Exception:
        return None


def extract_exif_metadata(image_bytes: bytes) -> Dict[str, Any]:
    """
    画像生バイナリからEXIFメタデータ（GPS緯度経度、撮影日時、カメラ機種等）を抽出する
    """
    meta: Dict[str, Any] = {
        "has_exif": False,
        "make": None,
        "model": None,
        "datetime": None,
        "gps": None
    }
    try:
        with Image.open(io.BytesIO(image_bytes)) as img:
            exif = img.getexif()
            if not exif:
                return meta

            meta["has_exif"] = True

            # 基本タグの抽出
            # 0x010F: Make, 0x0110: Model, 0x0132: DateTime
            meta["make"] = str(exif.get(0x010F)) if 0x010F in exif else None
            meta["model"] = str(exif.get(0x0110)) if 0x0110 in exif else None
            meta["datetime"] = str(exif.get(0x0132)) if 0x0132 in exif else None

            # Exif IFD の探索（DateTimeOriginal など）
            if hasattr(ExifTags, "IFD") and hasattr(ExifTags.IFD, "Exif"):
                exif_ifd = exif.get_ifd(ExifTags.IFD.Exif)
                if exif_ifd and 0x9003 in exif_ifd:  # 0x9003: DateTimeOriginal
                    meta["datetime"] = str(exif_ifd[0x9003])

            # GPS IFD の探索
            if hasattr(ExifTags, "IFD") and hasattr(ExifTags.IFD, "GPSInfo"):
                gps_ifd = exif.get_ifd(ExifTags.IFD.GPSInfo)
                if gps_ifd:
                    # 1: GPSLatitudeRef, 2: GPSLatitude, 3: GPSLongitudeRef, 4: GPSLongitude
                    lat_ref = gps_ifd.get(1)
                    lat = gps_ifd.get(2)
                    lng_ref = gps_ifd.get(3)
                    lng = gps_ifd.get(4)
                    alt = gps_ifd.get(6)  # GPSAltitude

                    if lat and lat_ref and lng and lng_ref:
                        dec_lat = _dms_to_decimal(lat, lat_ref)
                        dec_lng = _dms_to_decimal(lng, lng_ref)
                        if dec_lat is not None and dec_lng is not None:
                            meta["gps"] = {
                                "latitude": dec_lat,
                                "longitude": dec_lng,
                                "altitude": float(alt) if alt is not None else None
                            }
    except Exception as e:
        logger.warning(f"EXIF抽出中に例外が発生しました（無視して続行）: {e}")

    return meta


def _upload_bytes_to_supabase(path: str, data: bytes, content_type: str = "image/jpeg") -> bool:
    """Supabase Storage REST API を使ってバイナリデータをアップロード"""
    url = f"{SUPABASE_URL}/storage/v1/object/{BUCKET_NAME}/{path}"
    headers = {
        "Authorization": f"Bearer {SUPABASE_ANON_KEY}",
        "apikey": SUPABASE_ANON_KEY,
        "Content-Type": content_type,
        "x-upsert": "true",
    }
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return resp.status in (200, 201)
    except Exception as e:
        logger.error(f"Supabaseアップロード失敗 [{path}]: {e}")
        return False


def _worker_upload(image_bytes: bytes, original_filename: str, result_summary: dict, mode: str):
    """バックグラウンドスレッドで実行されるアップロードワーカー"""
    try:
        timestamp_str = time.strftime("%Y%m%d_%H%M%S")
        rand_id = uuid.uuid4().hex[:8]
        base_name = f"{timestamp_str}_{rand_id}"

        # 1. EXIF メタデータを抽出
        exif_info = extract_exif_metadata(image_bytes)

        # 拡張子の判定
        ext = "jpg"
        content_type = "image/jpeg"
        if original_filename:
            fn_lower = original_filename.lower()
            if fn_lower.endswith(".png"):
                ext = "png"
                content_type = "image/png"
            elif fn_lower.endswith(".webp"):
                ext = "webp"
                content_type = "image/webp"

        image_path = f"{STORAGE_PREFIX}/{base_name}_original.{ext}"
        meta_path = f"{STORAGE_PREFIX}/{base_name}_meta.json"

        # 2. 原本画像（EXIF完全保持）のアップロード
        img_success = _upload_bytes_to_supabase(image_path, image_bytes, content_type=content_type)

        # 3. 診断結果＋EXIF統合メタデータの作成とアップロード
        meta_payload = {
            "id": base_name,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "mode": mode,
            "original_filename": original_filename,
            "image_storage_path": image_path,
            "exif": exif_info,
            "diagnosis": result_summary
        }
        meta_bytes = json.dumps(meta_payload, ensure_ascii=False, indent=2).encode("utf-8")
        meta_success = _upload_bytes_to_supabase(meta_path, meta_bytes, content_type="application/json")

        if img_success and meta_success:
            logger.info(f"Supabaseへの診断ログ保存完了: {base_name} (GPS: {exif_info.get('gps')})")
    except Exception as e:
        logger.error(f"非同期保存ワーカーで例外が発生しました: {e}")


def upload_diagnosis_async(image_bytes: bytes, original_filename: str, result_summary: dict, mode: str = "single"):
    """
    診断結果と画像（EXIF付き）を非同期バックグラウンドで Supabase Storage に保存する。
    メインスレッドを一切待たせず、推論処理とUI表示を最速で完了させる。
    """
    if not image_bytes:
        return

    # スレッドを起動して非同期実行（デーモンスレッド）
    t = threading.Thread(
        target=_worker_upload,
        args=(image_bytes, original_filename, result_summary, mode),
        daemon=True
    )
    t.start()
