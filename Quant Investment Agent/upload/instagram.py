"""Module 7: Instagram Graph API 자동 업로드.

캐러셀 포스트 형식으로 7장 이미지 업로드.
업로드 순서:
  1. 각 이미지를 Instagram 미디어 컨테이너로 업로드
  2. 캐러셀 컨테이너 생성
  3. 캐러셀 게시

필요 조건:
  - Facebook Business Manager 계정
  - Instagram Professional 계정 연결
  - Facebook App (Graph API 권한: instagram_basic, instagram_content_publish)
  - 장기 액세스 토큰 (60일 유효)

환경 변수 (.env):
  INSTAGRAM_ACCESS_TOKEN  — 장기 액세스 토큰
  INSTAGRAM_USER_ID       — Instagram Business 계정 ID
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()

GRAPH_API_BASE = "https://graph.facebook.com/v19.0"
ACCESS_TOKEN = os.environ.get("INSTAGRAM_ACCESS_TOKEN", "")
USER_ID = os.environ.get("INSTAGRAM_USER_ID", "")

# Retry settings for media publish
MAX_RETRIES = 5
RETRY_DELAY_SECONDS = 5


class InstagramUploader:
    """Uploads a carousel post to Instagram via Graph API."""

    def __init__(self, access_token: str = ACCESS_TOKEN,
                 user_id: str = USER_ID) -> None:
        if not access_token or not user_id:
            raise EnvironmentError(
                "INSTAGRAM_ACCESS_TOKEN 및 INSTAGRAM_USER_ID를 .env에 설정하세요.")
        self._token = access_token
        self._user_id = user_id

    # ── Public interface ───────────────────────────────────────────────────────

    def upload_carousel(self, image_paths: list[Path],
                        caption: str) -> dict:
        """Upload a carousel post.

        Args:
            image_paths: 순서대로 업로드할 PNG 파일 경로 리스트 (최대 10장)
            caption:     포스트 캡션 (해시태그 포함)

        Returns:
            {"success": bool, "post_id": str | None, "error": str | None}
        """
        if not image_paths:
            return {"success": False, "post_id": None,
                    "error": "업로드할 이미지가 없습니다."}

        if len(image_paths) > 10:
            image_paths = image_paths[:10]

        # Step 1: Upload each image as a media item (is_carousel_item=true)
        container_ids: list[str] = []
        for path in image_paths:
            cid = self._create_image_container(path)
            if cid is None:
                return {"success": False, "post_id": None,
                        "error": f"이미지 컨테이너 생성 실패: {path.name}"}
            container_ids.append(cid)

        # Step 2: Create carousel container
        carousel_id = self._create_carousel_container(container_ids, caption)
        if carousel_id is None:
            return {"success": False, "post_id": None,
                    "error": "캐러셀 컨테이너 생성 실패"}

        # Step 3: Publish (with retry — Instagram needs processing time)
        post_id = self._publish(carousel_id)
        if post_id is None:
            return {"success": False, "post_id": None,
                    "error": "게시 실패 (최대 재시도 초과)"}

        return {"success": True, "post_id": post_id, "error": None}

    # ── Step 1: Create image media container ──────────────────────────────────

    def _create_image_container(self, path: Path) -> str | None:
        """Creates a single image container for carousel use."""
        url = f"{GRAPH_API_BASE}/{self._user_id}/media"

        # Note: Instagram Graph API requires a publicly accessible image URL.
        # For local files, you must host them first (e.g., S3, CDN).
        # Here we demonstrate the API call structure; replace image_url with
        # your hosted URL in production.
        image_url = _get_public_url(path)
        if not image_url:
            return None

        params = {
            "image_url": image_url,
            "is_carousel_item": "true",
            "access_token": self._token,
        }

        try:
            resp = requests.post(url, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            return data.get("id")
        except requests.RequestException as e:
            print(f"[ERROR] 이미지 컨테이너 생성 실패 ({path.name}): {e}")
            return None

    # ── Step 2: Create carousel container ────────────────────────────────────

    def _create_carousel_container(self, children: list[str],
                                   caption: str) -> str | None:
        url = f"{GRAPH_API_BASE}/{self._user_id}/media"
        params = {
            "media_type": "CAROUSEL",
            "children": ",".join(children),
            "caption": caption,
            "access_token": self._token,
        }
        try:
            resp = requests.post(url, params=params, timeout=30)
            resp.raise_for_status()
            return resp.json().get("id")
        except requests.RequestException as e:
            print(f"[ERROR] 캐러셀 컨테이너 생성 실패: {e}")
            return None

    # ── Step 3: Publish ───────────────────────────────────────────────────────

    def _publish(self, creation_id: str) -> str | None:
        url = f"{GRAPH_API_BASE}/{self._user_id}/media_publish"
        params = {
            "creation_id": creation_id,
            "access_token": self._token,
        }
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                resp = requests.post(url, params=params, timeout=30)
                resp.raise_for_status()
                data = resp.json()
                if "id" in data:
                    return data["id"]
                # Instagram may return error during processing; wait and retry
                print(f"[WARN] 게시 응답에 id 없음 (시도 {attempt}/{MAX_RETRIES}): {data}")
            except requests.RequestException as e:
                print(f"[ERROR] 게시 오류 (시도 {attempt}/{MAX_RETRIES}): {e}")

            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY_SECONDS)

        return None


# ── Caption generator ─────────────────────────────────────────────────────────

def generate_caption(analysis: dict) -> str:
    """Generate a Korean Instagram caption with hashtags."""
    qqq = analysis.get("qqq_signal", {})
    fg = analysis.get("fear_greed", {})
    macro = analysis.get("macro", {})
    date_str = analysis.get("date", "")

    signal_label = qqq.get("signal_label", "QQQ (보유)")
    fg_score = fg.get("score", 50)
    fg_label = fg.get("label", "중립")
    macro_score = macro.get("score", 50)
    macro_label = macro.get("label", "중립")

    caption = f"""📊 {date_str} 퀀트 시그널

🎯 오늘의 신호: {signal_label}
😰 공포탐욕지수: {fg_score} ({fg_label})
🌐 매크로 환경: {macro_score}점 ({macro_label})

AI 앙상블 모델(XGBoost, LightGBM, LSTM, Transformer)이 분석한 QQQ 레버리지 타이밍 시그널입니다.

⚠️ 본 콘텐츠는 투자 권유가 아닙니다. 모든 투자 결정은 본인 책임입니다.

#퀀트투자 #QQQ #ETF투자 #기술적분석 #매크로분석 #공포탐욕지수 #섹터로테이션 #AI투자 #TQQQ #QLD #주식투자 #미국주식
"""
    return caption.strip()


# ── Public URL helper ─────────────────────────────────────────────────────────

def _get_public_url(path: Path) -> str | None:
    """Returns a publicly accessible URL for the image.

    In production: upload to S3/CDN and return the URL.
    For local testing, set MEDIA_BASE_URL in environment to a tunneled URL
    (e.g., ngrok http 8080) and serve with `python -m http.server 8080`.
    """
    base_url = os.environ.get("MEDIA_BASE_URL", "")
    if not base_url:
        print(f"[WARN] MEDIA_BASE_URL 미설정 — 로컬 파일 업로드 불가. "
              f"S3나 ngrok URL을 .env에 설정하세요.")
        return None
    return f"{base_url.rstrip('/')}/{path.name}"
