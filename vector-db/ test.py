import os
import sys
import re
import hashlib
from pathlib import Path
from typing import List, Tuple
from tqdm import tqdm

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
from pdf2image import convert_from_path
import pytesseract

import chromadb
from chromadb.config import Settings

load_dotenv(dotenv_path="/mnt/data_sdd/energy_gpt/vector-db/.env", override=True)

# =========================
# 1) 설정 (환경변수로 덮어쓰기 가능)
# =========================
PDF_DIR       = os.getenv("PDF_DIR", "./data/pdfs")       # PDF 폴더
CHROMA_DIR    = os.getenv("CHROMA_DIR", "./chroma_db")    # 벡터DB 저장 폴더
COLLECTION    = os.getenv("COLLECTION", "ko_pdfs")
CHUNK_SIZE    = int(os.getenv("CHUNK_SIZE", "800"))       # 한국어: 800~1200 권장
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
USE_OCR       = os.getenv("USE_OCR_FALLBACK", "true").lower() != "false"
# pdf2image용 poppler 경로(윈도/비표준 경로일 때만 사용)
POPPLER_PATH  = os.getenv("POPPLER_PATH", None)
# Tesseract 한국어 + 영어
TESS_LANG     = os.getenv("TESS_LANG", "kor+eng")
# 임베딩 모델 (원하면 env로 교체 가능)
EMB_MODEL     = os.getenv("EMB_MODEL", "BAAI/bge-m3")

# =========================
# 2) 유틸
# =========================
def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8", "ignore")).hexdigest()

def file_sha1(path: Path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

_ws_collapse = re.compile(r"[ \t\f\v]+")
def normalize_text(s: str) -> str:
    s = s.replace("\x00", "").replace("\r", "\n")
    s = re.sub(r"\n{3,}", "\n\n", s)       # 과도한 개행 축약
    s = _ws_collapse.sub(" ", s)           # 공백 정리
    return s.strip()

def chunk_text(text: str, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP) -> List[str]:
    text = normalize_text(text)
    if not text:
        return []
    n, start, out = len(text), 0, []
    while start < n:
        end = min(start + size, n)
        seg = text[start:end].strip()
        if seg:
            out.append(seg)
        # 한글 단어 중간 토막 방지(느슨한 매너): 다음 시작점을 공백 기준으로 미세 조정
        next_start = max(0, end - overlap)
        # 공백/개행이 나올 때까지 살짝 전진 (최대 20자)
        steps = 0
        while next_start < n and text[next_start] not in (" ", "\n") and steps < 20:
            next_start += 1
            steps += 1
        start = next_start if next_start < n else n
    return out

def extract_pages(path: Path) -> List[Tuple[int, str]]:
    """
    각 페이지에서 텍스트를 추출하여 (page_no, text) 리스트 반환.
    - 기본: pypdf
    - 비텍스트/스캔본: Tesseract OCR (선택)
    """
    out: List[Tuple[int, str]] = []
    try:
        reader = PdfReader(str(path))
        for i, page in enumerate(reader.pages):
            t = page.extract_text() or ""
            t = normalize_text(t)
            if not t and USE_OCR:
                try:
                    kwargs = dict(first_page=i + 1, last_page=i + 1, fmt="png")
                    if POPPLER_PATH:
                        kwargs["poppler_path"] = POPPLER_PATH
                    img = convert_from_path(str(path), **kwargs)[0]
                    t = pytesseract.image_to_string(img, lang=TESS_LANG)
                    t = normalize_text(t)
                except Exception:
                    t = ""
            out.append((i + 1, t))
    except Exception as e:
        print(f"[WARN] {path} 파싱 실패: {e}")
    return out

# =========================
# 3) 메인
# =========================
def main():
    pdf_root = Path(PDF_DIR)
    if not pdf_root.exists():
        print(f"[ERR] PDF_DIR 경로 없음: {pdf_root}")
        sys.exit(1)

    print(f"임베딩 모델 로딩: {EMB_MODEL}")
    model = SentenceTransformer(EMB_MODEL)  # 한국어/영어 모두 우수(bge-m3)

    client = chromadb.PersistentClient(path=CHROMA_DIR, settings=Settings(allow_reset=False))
    coll = client.get_or_create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})

    pdfs = [p for p in pdf_root.rglob("*.pdf")]
    print(f"발견된 PDF: {len(pdfs)}개")
    total_chunks = 0

    for path in tqdm(pdfs, desc="Indexing"):
        try:
            file_id = file_sha1(path)[:16]
            mtime = int(path.stat().st_mtime)

            ids, docs, metas = [], [], []
            for page_no, text in extract_pages(path):
                if not text:
                    continue
                chunks = chunk_text(text)
                for cidx, chunk in enumerate(chunks):
                    uid = sha1(f"{file_id}|{page_no}|{cidx}|{mtime}")[:24]
                    ids.append(uid)
                    docs.append(chunk)
                    metas.append(
                        {
                            "file": str(path),
                            "page": page_no,
                            "chunk_idx": cidx,
                            "file_hash": file_id,
                            "mtime": mtime,
                        }
                    )

            if not docs:
                continue

            # Chroma는 동일 id add 시 에러 → 선삭제 후 add로 업서트 효과
            try:
                coll.delete(ids=ids)
            except Exception:
                pass

            embs = model.encode(docs, normalize_embeddings=True, show_progress_bar=False)
            coll.add(ids=ids, documents=docs, metadatas=metas, embeddings=embs)
            total_chunks += len(ids)
        except Exception as e:
            print(f"[WARN] 인덱싱 실패: {path} | {e}")

    print(f"완료! 추가된 청크: {total_chunks}")

    # 샘플 검색
    if total_chunks:
        q = os.getenv("SAMPLE_QUERY", "문서에서 핵심 정책 요약")
        q_emb = model.encode([q], normalize_embeddings=True)[0]
        res = coll.query(
            query_embeddings=[q_emb],
            n_results=int(os.getenv("SAMPLE_TOPK", "5")),
            include=["documents", "metadatas", "distances"],
        )
        print("\n=== 샘플 검색 결과 ===")
        for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
            score = 1 - dist
            print(f"- {Path(meta['file']).name} p.{meta['page']} (score={score:.3f})")
            snippet = doc[:200].replace("\n", " ")
            print(snippet + ("..." if len(doc) > 200 else ""))

if __name__ == "__main__":
    main()
