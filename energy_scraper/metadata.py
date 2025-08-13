from pymongo import MongoClient
from dotenv import load_dotenv
import os

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
DB_NAME = os.getenv("MONGO_DB", "energyData")
COLLECTION_NAME = os.getenv("MONGO_COLLECTION", "naverReport")

client = MongoClient(MONGO_URI)
db = client[DB_NAME]
collection = db[COLLECTION_NAME]

def save_metadata_to_mongo(metadata: dict):
    """pdf_url 기준으로 중복 없이 저장"""
    if collection.find_one({"pdf_url": metadata["pdf_url"]}):
        print(f"이미 저장됨: {metadata['title']}")
        return
    collection.insert_one(metadata)
    print(f"저장 완료: {metadata['title']}")
