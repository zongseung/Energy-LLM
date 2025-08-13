# 1. 베이스
FROM python:3.13-slim

# 2. 필수 패키지 + uv 설치
RUN apt-get update \
 && apt-get install -y --no-install-recommends curl gcc \
 && rm -rf /var/lib/apt/lists/* \
 && curl -LsSf https://astral.sh/uv/install.sh | sh

# 3. 환경
ENV PATH="/root/.local/bin:$PATH" \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1 \
    TZ=Asia/Seoul \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

# 4. 작업 디렉토리
WORKDIR /app

# 5. 의존성 해석/설치 (lock이 있으면 재현성, 없으면 lock 생성)
COPY pyproject.toml uv.lock* ./
RUN uv sync --frozen --no-dev || (uv lock && uv sync --no-dev)

# 6. uv가 만든 가상환경(.venv)을 PATH에 추가
ENV PATH="/app/.venv/bin:$PATH"

# 7. 소스 복사 (프로젝트 구조에 맞춰 유지)
COPY energy_scraper ./energy_scraper
COPY src ./src               
COPY main.py ./

# 8. 파이썬 모듈 경로
ENV PYTHONPATH=/app

# 9. 실행
ENTRYPOINT ["python", "main.py"]
