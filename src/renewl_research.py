#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import re
import sys
import time
import logging
import argparse
from typing import Dict, Optional, List, Tuple
from pathlib import Path
from urllib.parse import unquote
import requests
from dotenv import load_dotenv

# -----------------------------------------------------------------------------
# 프로젝트 루트 경로 추가 (energy_scraper 모듈 임포트용)
# -----------------------------------------------------------------------------
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from energy_scraper.nas import upload_to_nas
from energy_scraper.slack import send_slack_message
from energy_scraper.logger import setup_logger  # 반환 None일 수 있어 방어

# -----------------------------------------------------------------------------
# 상수
# -----------------------------------------------------------------------------
BASE = "https://www.knrec.or.kr"
LIST_URL = "https://www.knrec.or.kr/biz/korea/briefing/list.do"
DEFAULT_ENV = "/mnt/data_sdd/energy_gpt/.env_renewaldata"

# file_down('1290','1','briefing') 패턴 (공백 허용 + 따옴표 종류 일치)
onclick_re = re.compile(
    r"""file_down\(
        \s* (['"])(\d+)\1 \s* , \s*
        (['"])(\d+)\3 \s* , \s*
        (['"])([^'"]+)\5 \s*
    \)""",
    re.VERBOSE
)

# 페이징: onclick="fn_move(5)" 형태
fnmove_re = re.compile(r'onclick\s*=\s*["\']\s*fn_move\((\d+)\)\s*["\']', re.I)

# 리스트 페이지의 hidden input 파싱
hidden_input_re = re.compile(
    r'<input[^>]*type=["\']hidden["\'][^>]*name=["\']([^"\']+)["\'][^>]*value=["\']([^"\']*)["\']',
    re.I
)

# 현재 페이지(li class="on") 탐지 (로깅용)
current_on_re = re.compile(r'<li[^>]*class=["\']on["\'][^>]*>.*?fn_move\((\d+)\)', re.I | re.S)
fallback_on_num = re.compile(r'<li[^>]*class=["\']on["\'][^>]*>\s*(\d+)\s*<', re.I)

# -----------------------------------------------------------------------------
# .env 선로드
# -----------------------------------------------------------------------------
load_dotenv(dotenv_path=DEFAULT_ENV, override=True)

# -----------------------------------------------------------------------------
# 로거 1회 설정
# -----------------------------------------------------------------------------
def init_logger() -> logging.Logger:
    try:
        setup_logger('renewal_research')  # 반환이 None이어도 아래에서 보강
    except Exception:
        pass
    lg = logging.getLogger('renewal_research')
    lg.setLevel(logging.INFO)
    if not lg.handlers:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s - %(name)s - %(message)s"))
        lg.addHandler(h)
    return lg

logger = init_logger()

# -----------------------------------------------------------------------------
# 유틸
# -----------------------------------------------------------------------------
def parse_items(html: str) -> List[Tuple[str, str, str]]:
    """HTML에서 file_down(...) 호출 인자(no, gubun, kinds) 추출"""
    seen = set()
    out: List[Tuple[str, str, str]] = []
    for m in onclick_re.finditer(html):
        key = (m.group(2), m.group(4), m.group(6))
        if key not in seen:
            seen.add(key)
            out.append(key)
    return out

def find_max_page_via_fnmove(html: str) -> int:
    """fn_move(n)에서 최대 n을 반환. 없으면 1."""
    nums = [int(x) for x in fnmove_re.findall(html)]
    return max(nums) if nums else 1

def extract_form_params(html: str) -> Dict[str, str]:
    """hidden input name/value를 dict로 추출"""
    params: Dict[str, str] = {}
    for name, value in hidden_input_re.findall(html):
        params[name] = value
    return params

def detect_current_page(html: str) -> Optional[int]:
    """현재 활성 페이지 번호 추정(로깅용)"""
    m = current_on_re.search(html)
    if m:
        return int(m.group(1))
    m = fallback_on_num.search(html)
    return int(m.group(1)) if m else None

def _sanitize_filename(name: str) -> str:
    name = (name or "").strip().strip('"').strip("'")
    name = name.replace("/", "_").replace("\\", "_").replace("\0", "")
    return (name[:255] or "download.bin")

def filename_from_cd(cd: str, fallback: str) -> str:
    """Content-Disposition에서 파일명 파싱 (RFC5987 우선)"""
    cd = cd or ""
    m = re.search(r'filename\*\s*=\s*UTF-8\'\'([^;]+)', cd, re.IGNORECASE)
    if m:
        return _sanitize_filename(unquote(m.group(1)))
    m = re.search(r'filename\s*=\s*"([^"]+)"', cd, re.IGNORECASE)
    if m:
        return _sanitize_filename(m.group(1))
    m = re.search(r'filename\s*=\s*([^";]+)', cd, re.IGNORECASE)
    if m:
        return _sanitize_filename(m.group(1))
    return _sanitize_filename(fallback)

def download_one(session: requests.Session, no: str, gubun: str, kinds: str, save_dir: Path):
    url = f"{BASE}/biz/file/File_down.do?no={no}&gubun={gubun}&kinds={kinds}"
    r = session.get(url, timeout=60)
    r.raise_for_status()

    cd = r.headers.get("Content-Disposition", "")
    name = filename_from_cd(cd, f"{no}_{kinds}.bin")
    path = save_dir / name
    with open(path, "wb") as f:
        f.write(r.content)
    logger.info(f"로컬 저장 완료: {path}")

    try:
        nas_path = upload_to_nas(str(path))
        logger.info(f"NAS 업로드 완료: {nas_path}")
    except Exception as e:
        logger.warning(f"NAS 업로드 실패: {name} | {e}")
        try:
            send_slack_message(f"리뉴얼 데이터 NAS 업로드 실패: {name}")
        except Exception:
            logger.debug("슬랙 알림 실패 무시")

def fetch_list_page(session: requests.Session, page: int, base_params: Optional[Dict[str, str]]) -> str:
    """
    1페이지: GET
    2페이지~: POST로 base_params + pageIndex 전송 (무시/404 시 GET 쿼리로 폴백)
    """
    if page == 1 or not base_params:
        r = session.get(LIST_URL, timeout=20)
    else:
        data = dict(base_params)
        data["pageIndex"] = str(page)
        r = session.post(LIST_URL, data=data, timeout=20)
        # 서버가 POST 파라미터를 무시하거나 리다이렉트로 1페이지 돌려줄 경우 대비
        if r.status_code == 404 or (r.history and r.url.endswith("list.do")):
            r = session.get(f"{LIST_URL}?pageIndex={page}", timeout=20)
    if r.status_code == 404:
        raise requests.HTTPError("404 Not Found", response=r)
    r.raise_for_status()
    return r.text

# -----------------------------------------------------------------------------
# 메인
# -----------------------------------------------------------------------------
def renewl_main(pages: Optional[int] = None):
    """
    knrec.or.kr 브리핑 자료 전체 수집.
    :param pages: 최대 페이지 수 (None이면 fn_move()로 자동 감지)
    """
    # CLI 옵션(.env 덮어쓰기 허용)
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile")
    parser.add_argument("--env-file")
    args, _ = parser.parse_known_args()

    if args.profile:
        p = f".env_{args.profile}"
        if not os.path.exists(p):
            p = f".env.{args.profile}"
        if os.path.exists(p):
            load_dotenv(dotenv_path=p, override=True)
            logger.info(f".env 로드(프로필): {p}")
    elif args.env_file and os.path.exists(args.env_file):
        load_dotenv(dotenv_path=args.env_file, override=True)
        logger.info(f".env 로드(지정): {args.env_file}")

    # 저장 경로
    save_dir_str = os.getenv("DOWNLOAD_DIR")
    if not save_dir_str:
        raise ValueError("DOWNLOAD_DIR 환경변수가 설정되지 않았습니다.")
    save_dir = Path(save_dir_str)
    save_dir.mkdir(parents=True, exist_ok=True)

    headers = {
        "User-Agent": "Mozilla/5.0",
        "Referer": LIST_URL,
    }

    logger.info("리뉴얼 데이터 수집 시작...")
    total = 0
    seen = set()

    with requests.Session() as s:
        s.headers.update(headers)

        # 첫 페이지
        first_html = fetch_list_page(s, 1, base_params=None)
        base_params = extract_form_params(first_html)      # ★ hidden 값 확보
        max_page = find_max_page_via_fnmove(first_html) if (pages is None) else int(pages)
        cur = detect_current_page(first_html)
        logger.info(f"감지된 최대 페이지: {max_page} (표시 페이지: {cur})")

        # 1..max_page 순회
        for page in range(1, max_page + 1):
            html = first_html if page == 1 else fetch_list_page(s, page, base_params)
            cur = detect_current_page(html)
            logger.info(f"{page}페이지 요청 → 표시: {cur}")

            items = parse_items(html)
            if not items:
                logger.info(f"{page}페이지: 항목 없음")
                continue

            new_items = [k for k in items if k not in seen]
            for k in new_items:
                seen.add(k)

            for no, gubun, kinds in new_items:
                try:
                    download_one(s, no, gubun, kinds, save_dir)
                    total += 1
                    time.sleep(0.4)  # 매너 딜레이
                except requests.HTTPError as e:
                    logger.error(f"다운로드 실패 (HTTPError): {no}, {gubun}, {kinds} | {e}")
                except Exception as e:
                    logger.error(f"다운로드 실패: {no}, {gubun}, {kinds} | {e}")

    logger.info(f"수집 완료. 총 {total}개 파일.")
    try:
        send_slack_message(f"리뉴얼 데이터 수집 완료: 총 {total}개 파일 다운로드 및 NAS 업로드 성공.")
    except Exception:
        logger.debug("슬랙 알림 실패 무시")


if __name__ == "__main__":
    # None → 사이트에서 최대 페이지 자동 감지
    renewl_main(pages=None)
