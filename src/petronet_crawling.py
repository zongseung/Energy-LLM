# src/petronet_crawling.py
import os
import re
import json
import time
import random
import hashlib
import logging
from datetime import datetime
from typing import Dict, List
import requests
from urllib.parse import urlparse, parse_qs, urljoin
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter, Retry

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from energy_scraper.logger import setup_logger
from energy_scraper.slack import send_slack_message
from energy_scraper.metadata import save_metadata_to_mongo
from energy_scraper.nas import upload_to_nas

# ========= 로깅/환경 로더 =========
setup_logger()

def load_env(profile: str | None, env_file: str | None = None):
    """
    로드 우선순위:
    1) 기본 .env (override=False)
    2) --env-file 로 전달된 파일 (override=True)
    3) profile이 있으면 .env_{profile} -> .env.{profile} (override=True)
    4) 루트의 .env1 자동 로드 (override=True)
    """
    from dotenv import load_dotenv
    load_dotenv(override=False)  # 1) 기본 .env

    # 2) --env-file 명시 로드
    if env_file and os.path.exists(env_file):
        load_dotenv(dotenv_path=env_file, override=True)
        print(f"[INFO] Loaded {env_file}")
        return

    # 3) profile 기반 로드
    if profile:
        for candidate in (f".env_{profile}", f".env.{profile}"):
            if os.path.exists(candidate):
                load_dotenv(dotenv_path=candidate, override=True)
                print(f"[INFO] Loaded {candidate}")
                return
        print(f"[WARN] .env_{profile} / .env.{profile} not found. 기본 .env 유지")

    # 4) .env1 자동 감지
    if os.path.exists(".env1"):
        load_dotenv(dotenv_path=".env1", override=True)
        print("[INFO] Loaded .env1")

# --profile / --env-file 먼저 읽어서 환경 세팅
import argparse
_profile_parser = argparse.ArgumentParser(add_help=False)
_profile_parser.add_argument("--profile", help="환경 프로필(.env_{profile} 또는 .env.{profile})")
_profile_parser.add_argument("--env-file", help="임의 경로의 env 파일(.env1 등)")
_profile_args, _unknown = _profile_parser.parse_known_args()
load_env(_profile_args.profile, _profile_args.env_file)

# ========= 기본 설정 =========
LIST_URL = "https://www.petronet.co.kr/v4/sub.jsp"
POST_URL = LIST_URL
DOWNLOAD_HREF_PREFIX = "/servlet/dvboard.FileDownloadV4"

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)
COOKIE_STRING = os.environ.get("PETRONET_COOKIES", "")

# ---- 로컬 저장 디렉터리 (다운로드 위치) ----
LOCAL_DOWNLOAD_DIR = os.path.abspath(
    os.environ.get("DOWNLOAD_DIR", "./downloads/petronet")
)
os.makedirs(LOCAL_DOWNLOAD_DIR, exist_ok=True)

# ---- 상태/로그 디렉터리 ----
STATE_ROOT = os.path.abspath(os.environ.get("STATE_ROOT", "./state"))
STATE_DIR = os.path.join(STATE_ROOT, "petronet")
os.makedirs(STATE_DIR, exist_ok=True)

SEEN_FILE    = os.path.join(STATE_DIR, "seen.json")
ID_LOG_PATH  = os.path.join(STATE_DIR, "downloaded_ids.json")
HASH_LOG     = os.path.join(STATE_DIR, "downloaded_hashes.json")

SLACK_WEBHOOK_URL = os.environ.get("SLACK_WEBHOOK_URL")

PAGE_UNIT   = int(os.environ.get("PETRONET_PAGE_UNIT", "10"))
MAX_PAGES   = int(os.environ.get("PETRONET_MAX_PAGES", "9999"))
SLEEP_MIN   = float(os.environ.get("PETRONET_SLEEP_MIN", "0.8"))
SLEEP_MAX   = float(os.environ.get("PETRONET_SLEEP_MAX", "1.8"))
RETRY       = int(os.environ.get("PETRONET_RETRY", "2"))
REQ_TIMEOUT = int(os.environ.get("REQ_TIMEOUT", "30"))

HEADERS = {
    "Origin": "https://www.petronet.co.kr",
    "Referer": LIST_URL,
    "User-Agent": USER_AGENT,
    "Accept": "application/pdf,application/octet-stream,text/html;q=0.9,*/*;q=0.8",
}

# NAS 업로드용 원격 목적지 (FTP에서 사용)
NAS_FOLDER = os.environ.get("NAS_FOLDER", "/db/petro")  # .env1 값 사용
os.environ["NAS_FOLDER"] = NAS_FOLDER  # nas 모듈이 import 시 읽어가도록 보장
from energy_scraper.nas import upload_to_nas  # 이제 import
logging.info(f"[STATE] NAS upload dest={NAS_FOLDER}")
# 시작 상태 로깅
logging.info(f"[STATE] CWD={os.getcwd()}")
logging.info(f"[STATE] LOCAL_DOWNLOAD_DIR={LOCAL_DOWNLOAD_DIR}")
logging.info(f"[STATE] STATE_DIR={os.path.abspath(STATE_DIR)}")
logging.info(f"[STATE] NAS_FOLDER(remote)={NAS_FOLDER}")

BASE_FORM = {
    "pageType": "list",
    "tbName": "",
    "bbsSeq": "",
    "fmuId": "REPORTAND",
    "smuId": "REPORT",
    "tmuId": "PSBODB..TBREPORTFOG",
    "fmuOrd": "02",
    "smuOrd": "02_01",
    "tmuOrd": "02_01_02",
    "totalPages": "",
}
PAGE_KEY  = "thisPage"
BLOCK_KEY = "thisBlock"

def _new_session() -> requests.Session:
    s = requests.Session()
    retries = Retry(
        total=3,
        backoff_factor=0.6,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET", "POST", "HEAD", "OPTIONS"),
    )
    adapter = HTTPAdapter(max_retries=retries, pool_connections=10, pool_maxsize=20)
    s.mount("http://", adapter)
    s.mount("https://", adapter)
    s.headers.update(HEADERS)
    if COOKIE_STRING:
        for kv in COOKIE_STRING.split(";"):
            if "=" in kv and kv.strip():
                k, v = kv.strip().split("=", 1)
                s.cookies.set(k.strip(), v.strip(), domain=".petronet.co.kr")
    return s

SESSION = _new_session()

# ---- 유틸 ----
def ensure_dir(path: str): os.makedirs(path, exist_ok=True)

def load_json(path: str, default):
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f: return json.load(f)
        except Exception: return default
    return default

def save_json(path: str, data):
    with open(path, "w", encoding="utf-8") as f: json.dump(data, f, ensure_ascii=False, indent=2)

def load_seen(): return load_json(SEEN_FILE, {"urls": [], "files": []})
def save_seen(seen): save_json(SEEN_FILE, seen)
def load_ids() -> set: return set(load_json(ID_LOG_PATH, []))
def save_ids(ids: set): save_json(ID_LOG_PATH, sorted(list(ids)))
def load_hashes() -> set: return set(load_json(HASH_LOG, []))
def save_hashes(hs: set): save_json(HASH_LOG, sorted(list(hs)))

def safe_filename(name: str) -> str:
    name = re.sub(r'[\\/*?:"<>|]', "_", name)
    name = " ".join(name.split())
    return name[:180].rstrip("._ ")

def file_md5(path: str, chunk: int = 65536) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""): h.update(b)
    return h.hexdigest()

def guess_filename_from_response(resp: requests.Response, fallback: str) -> str:
    cd = resp.headers.get("Content-Disposition", "") or ""
    m = re.search(r'filename\*?=(?:UTF-8\'\')?"?([^";]+)"?', cd, re.I)
    if m:
        try: return requests.utils.unquote(m.group(1)).strip()
        except Exception: return m.group(1).strip()
    qs = parse_qs(urlparse(resp.url).query)
    if "filename" in qs and qs["filename"]:
        try: return requests.utils.unquote(qs["filename"][0]).strip()
        except Exception: return qs["filename"][0].strip()
    return fallback

def expected_pdf_path(fname: str) -> str:
    """로컬 저장 경로 (다운로드 위치)"""
    return os.path.join(LOCAL_DOWNLOAD_DIR, safe_filename(fname))

# ---- 페이징/파싱/다운로드 ----
def build_post_data(base_form: Dict, page_no: int, page_unit: int) -> Dict:
    d = dict(base_form)
    d[PAGE_KEY]       = str(page_no)
    d[BLOCK_KEY]      = str((page_no - 1) // page_unit)
    d["SELVOLUMNM"]   = str((page_no - 1) * page_unit)
    d["PAGE_TOTAL"]   = str(page_no * page_unit)
    d["DELETE_TOTAL"] = d["SELVOLUMNM"]
    d["pageUnit"]     = str(page_unit)
    d.setdefault("recordCountPerPage", str(page_unit))
    return d

def fetch_page(page_no: int, page_unit: int) -> str:
    logging.info(f"[PETRONET] fetch page {page_no}")
    payload = build_post_data(BASE_FORM, page_no, page_unit)
    r = SESSION.post(POST_URL, data=payload, timeout=REQ_TIMEOUT)
    r.raise_for_status()
    return r.text

def find_total_pages(html: str) -> int:
    soup = BeautifulSoup(html, "html.parser")
    last = soup.select_one("ul.pagination a.last")
    if last and last.get("onclick"):
        m = re.search(r"goPage\('(\d+)'", last["onclick"])
        if m: return int(m.group(1))
    mx = 1
    for a in soup.select("ul.pagination a[onclick]"):
        oc = a.get("onclick") or ""
        m = re.search(r"goPage\('(\d+)'", oc)
        if m: mx = max(mx, int(m.group(1)))
    if mx > 1: return mx
    nums = []
    for a in soup.select("ul.pagination a"):
        t = (a.get_text() or "").strip()
        if t.isdigit(): nums.append(int(t))
    return max(nums) if nums else 1

def extract_download_links(html: str, base: str) -> List[str]:
    soup = BeautifulSoup(html, "html.parser")
    links: List[str] = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if href.startswith(DOWNLOAD_HREF_PREFIX):
            links.append(urljoin(base, href))
    for a in soup.select("a[onclick]"):
        oc = a.get("onclick") or ""
        m = re.search(r"['\"](/servlet/dvboard\.FileDownloadV4[^'\"]+)", oc)
        if m: links.append(urljoin(base, m.group(1)))
    return list(dict.fromkeys(links))

def _save_bytes_to(path: str, first_chunk: bytes, rest_bytes: bytes):
    with open(path, "wb") as f:
        if first_chunk: f.write(first_chunk)
        if rest_bytes: f.write(rest_bytes)

def download_file(url: str, idx: int, seen: Dict, use_hash: bool, known_hashes: set):
    ensure_dir(LOCAL_DOWNLOAD_DIR)
    debug_dir = os.path.join(LOCAL_DOWNLOAD_DIR, "debug")
    ensure_dir(debug_dir)

    if url in seen["urls"]:
        logging.info(f"[SKIP] url done: {url}")
        return None

    req_headers = dict(HEADERS)
    req_headers["Connection"] = "close"
    req_headers["Accept-Encoding"] = "identity"

    for attempt in range(RETRY + 1):
        try:
            with SESSION.get(url, stream=True, timeout=60, headers=req_headers) as r:
                r.raise_for_status()
                ctype = (r.headers.get("Content-Type") or "").lower()
                first = b""
                for chunk in r.iter_content(65536):
                    if chunk:
                        first = chunk
                        break
                if not first:
                    raise IOError("empty first chunk")

                is_pdf = (b"%PDF" in first) or ("pdf" in ctype)
                if not is_pdf:
                    html_name = os.path.join(debug_dir, f"notpdf_{idx}.html")
                    _save_bytes_to(html_name, first, r.content if hasattr(r, "content") else b"")
                    logging.warning(f"[SKIP-NOTPDF] saved debug: {html_name}")
                    return None

                fname = guess_filename_from_response(r, f"file_{idx}.pdf")
                path = expected_pdf_path(fname)

                if os.path.exists(path):
                    if use_hash:
                        try:
                            h = file_md5(path)
                            if h in known_hashes:
                                logging.info(f"[SKIP] have file(hash): {fname}")
                                seen["urls"].append(url); save_seen(seen)
                                return None
                        except Exception:
                            pass
                    logging.info(f"[SKIP] have file: {fname}")
                    seen["urls"].append(url); save_seen(seen)
                    return None

                with open(path, "wb") as f:
                    f.write(first)
                    for chunk in r.iter_content(65536):
                        if chunk:
                            f.write(chunk)

                logging.info(f"[OK] {path}")
                seen["urls"].append(url); seen["files"].append(fname); save_seen(seen)
                return path

        except Exception as e:
            logging.warning(f"[WARN] download failed try {attempt+1}/{RETRY+1}: {repr(e)}")
            if attempt < RETRY:
                time.sleep(1.2)
            else:
                # non-stream fallback
                try:
                    r2 = SESSION.get(url, stream=False, timeout=60, headers=req_headers)
                    r2.raise_for_status()
                    body = r2.content or b""
                    ctype2 = (r2.headers.get("Content-Type") or "").lower()
                    if not (body.startswith(b"%PDF") or "pdf" in ctype2):
                        html_name = os.path.join(debug_dir, f"notpdf_{idx}.html")
                        _save_bytes_to(html_name, body[:0], body)
                        logging.warning(f"[SKIP-NOTPDF] saved debug: {html_name}")
                        return None

                    fname = guess_filename_from_response(r2, f"file_{idx}.pdf")
                    path = expected_pdf_path(fname)

                    if os.path.exists(path):
                        logging.info(f"[SKIP] have file: {fname}")
                        seen["urls"].append(url); save_seen(seen)
                        return None

                    with open(path, "wb") as f:
                        f.write(body)

                    logging.info(f"[OK-FALLBACK] {path}")
                    seen["urls"].append(url); seen["files"].append(fname); save_seen(seen)
                    return path

                except Exception as ee:
                    logging.error(f"[FAIL] giving up: {url} | {repr(ee)}")
                    return None

def iterate_pages(full: bool, pages: int, start: int | None, end: int | None):
    first_html = fetch_page(1, page_unit=PAGE_UNIT)
    total_pages = find_total_pages(first_html)
    logging.info(f"[PETRONET] detected total pages: {total_pages} (unit={PAGE_UNIT})")

    if start is not None:
        from_p = max(1, start)
        to_p = min(total_pages, end if end else total_pages)
    elif full:
        from_p, to_p = 1, total_pages
    else:
        from_p, to_p = 1, min(pages, total_pages)

    yield 1, first_html
    for p in range(2, to_p + 1):
        if p < from_p:
            continue
        html = fetch_page(p, page_unit=PAGE_UNIT)
        yield p, html

def petronet_main(full: bool = False, pages: int = 1, start: int | None = None, end: int | None = None, use_hash: bool = False):
    ensure_dir(LOCAL_DOWNLOAD_DIR)
    ensure_dir(STATE_DIR)

    seen = load_seen()
    downloaded_ids = load_ids()
    known_hashes = load_hashes() if use_hash else set()

    new_downloads = 0
    fail_dl = 0
    skipped = 0
    fail_nas = 0

    prev_first_link = None
    same_first_count = 0
    idx = 1

    try:
        for page_no, html in iterate_pages(full, pages, start, end):
            links = extract_download_links(html, LIST_URL)
            logging.info(f"[PETRONET|PAGE {page_no}] {len(links)} links")

            if links:
                if prev_first_link == links[0]:
                    same_first_count += 1
                else:
                    same_first_count = 0
                prev_first_link = links[0]

                if same_first_count >= 1:
                    logging.info(f"[PETRONET] page {page_no}: first link repeated -> stop")
                    break
            else:
                logging.info(f"[PETRONET] page {page_no}: no links found")
                continue

            for u in links:
                if u in downloaded_ids:
                    skipped += 1
                    logging.info(f"[SKIP-id] {u}")
                    continue

                saved_path = download_file(u, idx, seen, use_hash, known_hashes)
                idx += 1
                time.sleep(random.uniform(SLEEP_MIN, SLEEP_MAX))

                if not saved_path:
                    fail_dl += 1
                    continue

                if use_hash:
                    try:
                        h = file_md5(saved_path)
                        if h not in known_hashes:
                            known_hashes.add(h)
                            save_hashes(known_hashes)
                    except Exception:
                        pass

                try:
                    # upload_to_nas가 .env의 NAS_* 정보를 사용하여 원격 업로드 수행
                    # (시그니처가 dest_dir를 받는다면: upload_to_nas(saved_path, dest_dir=NAS_FOLDER))
                    upload_to_nas(saved_path, dest_dir=os.getenv("NAS_FOLDER", "/db/petro"))
                    
                except Exception as e:
                    fail_nas += 1
                    logging.error(f"[NAS 업로드 실패] {os.path.basename(saved_path)} | {e}")

                meta = {
                    "source": "petronet",
                    "title": os.path.splitext(os.path.basename(saved_path))[0],
                    "date": None,
                    "pdf_url": u,
                    "downloaded_path": saved_path,
                    "created_at": datetime.utcnow().isoformat() + "Z",
                }
                save_metadata_to_mongo(meta)

                downloaded_ids.add(u)
                save_ids(downloaded_ids)
                new_downloads += 1

    finally:
        msg = f"[PETRONET 요약] 신규 {new_downloads}건 / 스킵 {skipped} / DL실패 {fail_dl} / NAS실패 {fail_nas}"
        logging.info(msg)
        if SLACK_WEBHOOK_URL:
            try:
                send_slack_message(SLACK_WEBHOOK_URL, msg, username="리포트 수집기")
            except Exception:
                pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", help="환경 프로필(.env_{profile} 또는 .env.{profile})")
    parser.add_argument("--env-file", help="임의 경로의 env 파일(.env1 등)")
    parser.add_argument("--full", action="store_true", help="끝 페이지까지 전부 수집")
    parser.add_argument("--pages", type=int, default=1, help="최신 N페이지만 수집 (기본=5)")
    parser.add_argument("--start", type=int, default=None, help="시작 페이지")
    parser.add_argument("--end", type=int, default=None, help="끝 페이지")
    parser.add_argument("--use-hash", action="store_true", help="파일 MD5로 중복 추가검사")
    args = parser.parse_args()

    # 인자로 들어온 env를 한 번 더 반영 (재호출 안전)
    load_env(args.profile, args.env_file)

    petronet_main(
        full=args.full,
        pages=args.pages,
        start=args.start,
        end=args.end,
        use_hash=args.use_hash,
    )
