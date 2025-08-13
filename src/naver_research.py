# src/naver_crawling.py
import os
import requests
import json
import logging
import argparse
import hashlib
import time
from datetime import datetime
from bs4 import BeautifulSoup
from urllib.parse import urlparse, parse_qs, urljoin
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter, Retry

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))  # energy_scraper 코드 임포트

from energy_scraper.logger import setup_logger
from energy_scraper.slack import send_slack_message
from energy_scraper.metadata import save_metadata_to_mongo
from energy_scraper.nas import upload_to_nas

# ------------------ 초기 설정 ------------------
setup_logger()
from dotenv import load_dotenv
import argparse

def load_env(profile: str | None):
    # 1) 공통 .env 로드
    load_dotenv(override=False)

    # 2) profile이 지정되면 .env_{profile} 먼저, 없으면 .env.{profile} 시도
    if profile:
        env_file = f".env_{profile}"  # 언더바 우선
        if os.path.exists(env_file):
            load_dotenv(dotenv_path=env_file, override=True)
            print(f"[INFO] Loaded {env_file}")
            return
        # fallback to dot version
        env_file = f".env.{profile}"
        if os.path.exists(env_file):
            load_dotenv(dotenv_path=env_file, override=True)
            print(f"[INFO] Loaded {env_file}")
        else:
            print(f"[WARN] {env_file} not found. 기본 .env 값 사용")


# ------------------ 초기 설정 ------------------
setup_logger()

# ★ argparse로 profile 옵션 추가
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--profile", choices=["naver", "petro"], help="환경 프로필(.env_{profile})")
# 기존 인자랑 충돌 안 나게 parse_known_args 사용
profile_args, _ = parser.parse_known_args()

# ★ 환경변수 로드
load_env(profile_args.profile)
BASE_URL = "https://finance.naver.com/research/industry_list.naver"
BASE_ORIGIN = "https://finance.naver.com"

SLACK_WEBHOOK_URL = os.environ.get("SLACK_WEBHOOK_URL")
DEFAULT_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; EnergyScraper/1.0)"}
REQ_TIMEOUT = int(os.getenv("REQ_TIMEOUT", "30"))

def _new_session() -> requests.Session:
    s = requests.Session()
    retries = Retry(
        total=3,
        backoff_factor=0.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods={"GET", "HEAD", "OPTIONS"},
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retries, pool_connections=10, pool_maxsize=20)
    s.mount("http://", adapter)
    s.mount("https://", adapter)
    s.headers.update(DEFAULT_HEADERS)
    return s

SESSION = _new_session()

# ------------------ 유틸 ------------------
def load_downloaded_ids() -> set:
    if not os.path.exists(LOG_PATH):
        return set()
    try:
        with open(LOG_PATH, "r") as f:
            return set(json.load(f))
    except Exception:
        return set()

def save_downloaded_id(nid: str):
    ids = load_downloaded_ids()
    ids.add(nid)
    with open(LOG_PATH, "w") as f:
        json.dump(sorted(list(ids)), f, indent=2, ensure_ascii=False)

def load_hashes() -> set:
    if not os.path.exists(HASH_LOG_PATH):
        return set()
    try:
        with open(HASH_LOG_PATH, "r") as f:
            return set(json.load(f))
    except Exception:
        return set()

def save_hash(h: str):
    hs = load_hashes()
    hs.add(h)
    with open(HASH_LOG_PATH, "w") as f:
        json.dump(sorted(list(hs)), f, indent=2, ensure_ascii=False)

def file_md5(path: str, chunk: int = 65536) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()

def sanitize_filename(name: str, max_len: int = 160) -> str:
    bad = '/\\:*?"<>|\n\r\t'
    for ch in bad:
        name = name.replace(ch, "_")
    name = " ".join(name.split())
    return (name[:max_len]).rstrip("._ ")

def get_nid_from_url(url: str) -> str:
    # 네이버 상세 링크에 nid 파라미터가 붙는 경우가 많음
    parsed = urlparse(url)
    query = parse_qs(parsed.query)
    return query.get("nid", [""])[0] or url

def build_filename(title: str, date: str | None, company: str | None) -> str:
    d = parse_date(date) or (date or "")
    c = (company or "").strip()
    t = title.strip()
    # (YYYY-MM-DD) 제목 - 회사.pdf
    parts = []
    if d:
        parts.append(f"({d})")
    parts.append(t)
    if c:
        parts.append(f"- {c}")
    return sanitize_filename(" ".join(parts)) + ".pdf"

# 로컬 저장 디렉터리 (반드시 존재하게)
LOCAL_DOWNLOAD_DIR = os.path.abspath(
    os.environ.get("DOWNLOAD_DIR", "./downloads/naver")
)
os.makedirs(LOCAL_DOWNLOAD_DIR, exist_ok=True)

# 상태 저장(로그/해시) 디렉터리
STATE_ROOT = os.path.abspath(os.environ.get("STATE_ROOT", "./state"))
STATE_DIR = os.path.join(STATE_ROOT, "naver")
os.makedirs(STATE_DIR, exist_ok=True)

LOG_PATH = os.path.abspath(os.environ.get("LOG_PATH", os.path.join(STATE_DIR, "downloaded_ids.json")))
HASH_LOG_PATH = os.path.abspath(os.environ.get("HASH_LOG_PATH", os.path.join(STATE_DIR, "downloaded_hashes.json")))

# NAS(원격) 관련 정보는 upload_to_nas()에서 .env를 통해 읽어 사용
NAS_FOLDER = os.environ.get("NAS_FOLDER", "/naverResearch")  # 원격 목적지 폴더(FTP)

# 시작 로그 (비밀값 제외)
logging.info(f"[STATE] CWD={os.getcwd()}")
logging.info(f"[STATE] ENV_PROFILE={profile_args.profile}")
logging.info(f"[STATE] LOCAL_DOWNLOAD_DIR={LOCAL_DOWNLOAD_DIR}")
logging.info(f"[STATE] STATE_DIR={os.path.abspath(STATE_DIR)}")
logging.info(f"[STATE] LOG_PATH={LOG_PATH}")
logging.info(f"[STATE] HASH_LOG_PATH={HASH_LOG_PATH}")
logging.info(f"[STATE] NAS_FOLDER(remote)={NAS_FOLDER}")

def expected_pdf_path(title: str, date: str | None = None, company: str | None = None) -> str:
    """
    로컬에 저장될 파일의 절대 경로를 반환.
    (업로드는 별도로 upload_to_nas()가 처리)
    """
    return os.path.join(LOCAL_DOWNLOAD_DIR, build_filename(title, date, company))

def make_key(paper: dict) -> str:
    # 기본은 nid 또는 pdf_url을 키로 사용
    return (paper.get("nid") or paper.get("pdf_url") or "").strip()

def dedupe_batch(papers: list[dict]) -> list[dict]:
    seen, out = set(), []
    for p in papers:
        k = make_key(p) or f"{p.get('title','').strip()}|{p.get('date','').strip()}"
        if k in seen:
            continue
        seen.add(k)
        out.append(p)
    return out

# ★ 핵심 수정: 파일 존재를 먼저 보고, 파일이 없는데 seen_id만 있으면 재다운로드
def should_skip(paper: dict, downloaded_ids: set, use_hash: bool, known_hashes: set) -> tuple[bool, str]:
    k = make_key(paper)
    path = expected_pdf_path(paper["title"], paper.get("date"), paper.get("company"))

    # 1) 파일이 '있을 때만' 스킵 판단
    if os.path.exists(path) and os.path.getsize(path) > 0:
        if use_hash:
            try:
                h = file_md5(path)
                if h in known_hashes:
                    return True, "file_hash_seen"
            except Exception:
                pass
        return True, "file_exists"

    # 2) 파일이 없지만 seen_id가 기록되어 있으면, 복구 다운로드를 위해 스킵하지 않음
    if k and k in downloaded_ids:
        return False, "missing_file_but_seen"

    # 3) 그 외에는 다운로드 진행
    return False, ""

def parse_date(raw: str) -> str | None:
    raw = (raw or "").strip()
    for fmt in ("%y.%m.%d", "%Y-%m-%d", "%Y.%m.%d", "%Y/%m/%d"):
        try:
            return datetime.strptime(raw, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return None

# ------------------ 크롤러 (네이버) ------------------
def build_list_url(upjong: str, page: int) -> str:
    # upjong은 percent-encoded 값 기대 (기본: 에너지)
    return f"{BASE_URL}?searchType=upjong&upjong={upjong}&page={page}"

def get_research_papers(page: int = 1, upjong: str = "%BF%A1%B3%CA%C1%F6"):
    url = build_list_url(upjong, page)
    resp = SESSION.get(url, timeout=REQ_TIMEOUT)
    resp.raise_for_status()
    logging.info(f"[NAVER] 요청 성공: {url}")

    soup = BeautifulSoup(resp.text, "html.parser")
    rows = soup.select("table.type_1 tr")

    papers = []
    # 첫 tr은 헤더 가능성 높음
    for row in rows[1:]:
        cols = row.find_all("td")
        # 빈줄/공지 스킵
        if len(cols) < 5:
            continue

        title_cell = cols[1]
        company_cell = cols[2]
        date_cell = cols[4]

        link_tag = title_cell.find("a")
        pdf_tag = cols[3].find("a")

        if not link_tag or not pdf_tag:
            continue

        title = title_cell.get_text(strip=True)
        company = company_cell.get_text(strip=True)
        date = date_cell.get_text(strip=True)

        view_url = urljoin(BASE_ORIGIN, link_tag.get("href", ""))
        nid = get_nid_from_url(view_url)

        pdf_url = urljoin(BASE_ORIGIN, pdf_tag.get("href", ""))
        if not pdf_url:
            continue

        papers.append({
            "nid": nid,
            "title": title,
            "company": company,
            "date": date,
            "pdf_url": pdf_url,
            "source": "naver_research",
        })
    # 예의상 딜레이 (차단 예방)
    time.sleep(0.2)
    return papers

def has_next_page(current_page: int, upjong: str = "%BF%A1%B3%CA%C1%F6") -> bool:
    url = build_list_url(upjong, current_page)
    resp = SESSION.get(url, timeout=REQ_TIMEOUT)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    return soup.select_one("td.pgR a") is not None

def get_all_papers(upjong: str = "%BF%A1%B3%CA%C1%F6"):
    all_papers, page = [], 1
    while True:
        logging.info(f"[NAVER] {page}페이지 처리 중")
        items = get_research_papers(page, upjong=upjong)
        if not items:
            break
        all_papers.extend(items)
        if not has_next_page(page, upjong=upjong):
            break
        page += 1
    return all_papers

def get_first_n_pages(n: int, upjong: str = "%BF%A1%B3%CA%C1%F6"):
    all_papers, page = [], 1
    n = max(1, n)
    while page <= n:
        logging.info(f"[NAVER] {page}페이지 처리 중")
        items = get_research_papers(page, upjong=upjong)
        if not items:
            break
        all_papers.extend(items)
        if not has_next_page(page, upjong=upjong):
            break
        page += 1
    return all_papers

def get_pages(start: int, end: int | None = None, upjong: str = "%BF%A1%B3%CA%C1%F6"):
    assert start >= 1, "start는 1 이상이어야 합니다."
    all_papers, page = [], start
    while True:
        logging.info(f"[NAVER] {page}페이지 처리 중")
        items = get_research_papers(page, upjong=upjong)
        if not items:
            break
        all_papers.extend(items)

        if end is not None and page >= end:
            break
        if not has_next_page(page, upjong=upjong):
            break
        page += 1
    return all_papers

# ------------------ 다운로드/메타/NAS ------------------
def download_pdf(title: str, pdf_url: str, raw_date: str, source: str = "naver_research", company: str | None = None):
    logging.info(f"[DL] {title} → {pdf_url}")

    headers = dict(DEFAULT_HEADERS)
    headers["Referer"] = BASE_ORIGIN

    with SESSION.get(pdf_url, timeout=REQ_TIMEOUT, headers=headers, stream=True) as resp:
        resp.raise_for_status()

        # content-type 힌트로 PDF 여부 점검 (일부는 text/html로 리다이렉트되기도 함)
        ctype = (resp.headers.get("Content-Type") or "").lower()
        if "pdf" not in ctype:
            logging.warning(f"[WARN] Content-Type='{ctype}' (PDF가 아닐 수 있음): {pdf_url}")

        os.makedirs(LOCAL_DOWNLOAD_DIR, exist_ok=True)
        pdf_path = expected_pdf_path(title, raw_date, company)

        with open(pdf_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1024 * 64):
                if not chunk:
                    continue
                f.write(chunk)

    logging.info(f"[SAVE] {pdf_path}")

    metadata = {
        "source": source,
        "title": title,
        "date": parse_date(raw_date),
        "company": company,
        "pdf_url": pdf_url,
        "downloaded_path": pdf_path,
        "created_at": datetime.utcnow().isoformat() + "Z",
    }

    print("\n===== 추출된 메타데이터 =====")
    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    print("===========================\n")
    return metadata

# ------------------ 실행 진입점 (통합 메인에서 호출) ------------------
def naver_main(full: bool = False, pages: int = 1, start: int | None = None, end: int | None = None,
               use_hash: bool = False, upjong: str = "%BF%A1%B3%CA%C1%F6"):
    if not os.path.exists(LOG_PATH):
        logging.info("처음 실행입니다. 전체 리포트를 다운로드합니다.")
        full = True
        
    os.makedirs(LOCAL_DOWNLOAD_DIR, exist_ok=True)
    os.makedirs(STATE_DIR, exist_ok=True)


    downloaded_ids = load_downloaded_ids()
    known_hashes = load_hashes() if use_hash else set()

    # 시작 상태 가시화
    logging.info(f"[STATE] LOG_PATH={LOG_PATH}")
    logging.info(f"[STATE] HASH_LOG_PATH={HASH_LOG_PATH}")
    logging.info(f"[STATE] LOCAL_DOWNLOAD_DIR={LOCAL_DOWNLOAD_DIR}")

    logging.info(f"[STATE] downloaded_ids={len(downloaded_ids)} known_hashes={len(known_hashes)}")

    # 수집
    if start is not None:
        papers = get_pages(start=start, end=end, upjong=upjong)
    elif full:
        papers = get_all_papers(upjong=upjong)
    else:
        papers = get_first_n_pages(n=pages, upjong=upjong)

    # 배치 내 중복 제거
    papers = dedupe_batch(papers)
    logging.info(f"수집된 리포트 개수(배치 중복 제거 후): {len(papers)}")

    new_downloads, fail_dl, fail_nas, skipped = 0, 0, 0, 0
    for paper in papers:
        skip, reason = should_skip(paper, downloaded_ids, use_hash, known_hashes)
        if skip:
            skipped += 1
            logging.info(f"[SKIP-{reason}] {paper.get('title')}")
            continue
        elif reason == "missing_file_but_seen":
            logging.warning(f"[REPAIR] 파일이 없지만 상태에는 있음 → 재다운로드: {paper.get('title')}")

        metadata = None
        try:
            metadata = download_pdf(
                title=paper["title"],
                pdf_url=paper["pdf_url"],
                raw_date=paper["date"],
                source=paper.get("source", "naver_research"),
                company=paper.get("company"),
            )
        except Exception as e:
            fail_dl += 1
            logging.error(f"PDF 다운로드 실패: {paper['title']} | {e}")
            if SLACK_WEBHOOK_URL:
                send_slack_message(
                    webhook_url=SLACK_WEBHOOK_URL,
                    message=f"[PDF 다운로드 실패] {paper['title']}\n{str(e)}",
                    username="리포트 수집기"
                )
            continue

        try:
            upload_to_nas(metadata["downloaded_path"])
        except Exception as e:
            fail_nas += 1
            logging.error(f"NAS 업로드 실패: {paper['title']} | {e}")
            if SLACK_WEBHOOK_URL:
                send_slack_message(
                    webhook_url=SLACK_WEBHOOK_URL,
                    message=f"[NAS 업로드 실패] {paper['title']}\n{str(e)}",
                    username="리포트 수집기"
                )
            continue

        save_downloaded_id(make_key(paper))
        save_metadata_to_mongo(metadata)
        new_downloads += 1

        if use_hash:
            try:
                h = file_md5(metadata["downloaded_path"])
                save_hash(h)
                known_hashes.add(h)
            except Exception:
                pass

    msg = f"[요약] 신규 {new_downloads}건 / 스킵 {skipped} / DL실패 {fail_dl} / NAS실패 {fail_nas}"
    logging.info(msg)
    if SLACK_WEBHOOK_URL:
        send_slack_message(
            webhook_url=SLACK_WEBHOOK_URL,
            message=msg,
            username="리포트 수집기"
        )

# 단독 실행도 가능하게 (개발/테스트 용)
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true", help="끝 페이지까지 전부 수집")
    parser.add_argument("--pages", type=int, default=10, help="최신 N페이지만 수집 (기본=10)")
    parser.add_argument("--start", type=int, default=None, help="시작 페이지 (예: 1)")
    parser.add_argument("--end", type=int, default=None, help="끝 페이지 (옵션)")
    parser.add_argument("--use-hash", action="store_true", help="파일 MD5로 중복 추가검사")
    parser.add_argument("--upjong", type=str, default="%BF%A1%B3%CA%C1%F6", help="업종 코드(Percent-encoded). 기본=에너지")
    # parser.add_argument("--resync-nas", action="store_true",
    #                 help="파일이 이미 있어도 NAS로 업로드만 다시 시도")
    args = parser.parse_args()

    naver_main(
        full=args.full,
        pages=args.pages,
        start=args.start,
        end=args.end,
        use_hash=args.use_hash,
        upjong=args.upjong,
    )
