# energy_scraper/nas.py
import os
from ftplib import FTP, error_perm
from typing import Optional

# ❗ 모듈 import 시 .env를 강제 로드하지 않습니다.
#    (상위 애플리케이션이 이미 load_dotenv를 호출했다고 가정)
# from dotenv import load_dotenv
# load_dotenv()

def _resolve_config(dest_dir: Optional[str] = None):
    """
    호출 시점에 환경변수를 읽어 FTP 설정을 구성합니다.
    dest_dir가 주어지면 NAS_FOLDER를 덮어씁니다.
    """
    NAS_IP = os.getenv("NAS_IP")
    FTP_PORT = int(os.getenv("FTP_PORT", "21"))
    USERNAME = os.getenv("NAS_USERNAME")
    PASSWORD = os.getenv("NAS_PASSWORD")
    NAS_FOLDER = dest_dir if dest_dir is not None else os.getenv("NAS_FOLDER", "/")

    if not NAS_IP or not USERNAME or not PASSWORD:
        raise ValueError("NAS 접속 정보가 누락되었습니다. NAS_IP / NAS_USERNAME / NAS_PASSWORD 를 확인하세요.")
    return NAS_IP, FTP_PORT, USERNAME, PASSWORD, NAS_FOLDER

def _ftp_makedirs(ftp: FTP, path: str):
    """
    FTP 상에 중첩 디렉터리 생성. 존재하면 건너뜀.
    """
    # 절대경로 기준 처리
    parts = [p for p in path.split("/") if p != ""]
    if path.startswith("/"):
        ftp.cwd("/")
    for part in parts:
        try:
            ftp.cwd(part)
        except error_perm:
            ftp.mkd(part)
            ftp.cwd(part)

def upload_to_nas(file_path: str, *, dest_dir: Optional[str] = None,
                  timeout: int = 60, retries: int = 2, passive: bool = True) -> str:
    """
    지정한 파일을 NAS의 FTP 경로에 업로드합니다.
    - dest_dir: 업로드 목적지 디렉터리(예: '/db/petro'). 미지정 시 환경변수 NAS_FOLDER 사용.
    - timeout: FTP 연결 타임아웃(초)
    - retries: 실패 시 재시도 횟수
    - passive: PASV 모드 사용 여부
    반환값: 서버 상의 최종 경로(예: '/db/petro/파일.pdf')
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"로컬 파일이 존재하지 않습니다: {file_path}")

    NAS_IP, FTP_PORT, USERNAME, PASSWORD, NAS_FOLDER = _resolve_config(dest_dir)
    file_name = os.path.basename(file_path)

    last_err = None
    for attempt in range(retries + 1):
        try:
            ftp = FTP(timeout=timeout)
            # Python 3.9+: 명령 인코딩(한글 파일명 대응)
            try:
                ftp.encoding = "utf-8"
            except Exception:
                pass

            ftp.connect(NAS_IP, FTP_PORT)
            ftp.login(USERNAME, PASSWORD)
            ftp.set_pasv(passive)

            # 목적지 디렉터리 보장
            _ftp_makedirs(ftp, NAS_FOLDER)

            # 업로드
            with open(file_path, "rb") as f:
                ftp.storbinary(f"STOR {file_name}", f)

            ftp.quit()
            server_path = (NAS_FOLDER.rstrip("/") or "/") + "/" + file_name
            print(f"NAS 업로드 완료: {server_path}")
            return server_path

        except Exception as e:
            last_err = e
            try:
                ftp.quit()
            except Exception:
                pass
            if attempt < retries:
                continue
            break

    raise RuntimeError(f"NAS 업로드 실패: {file_path} -> {NAS_FOLDER} | {last_err}")
