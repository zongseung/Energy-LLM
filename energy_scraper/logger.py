# src/energy_scraper/logger.py

import logging
from logging.handlers import RotatingFileHandler
import os

def setup_logger(log_dir="logs", filename="run.log"):
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, filename)

    handler = RotatingFileHandler(
        log_path, maxBytes=1024 * 1024, backupCount=5, encoding="utf-8"
    )

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            handler,
            logging.StreamHandler()
        ]
    )

    logging.info(f"로그 파일 경로: {log_path}")