import schedule
import time
import pandas as pd
from pymongo import MongoClient
import os
from dotenv import load_dotenv
from get_text_embedding import get_text_embedding

# 부서별 크롤링 모듈 가져오기
from liberal_arts_crawling import crawl_liberal_arts
from economics_crawling import crawl_economics
from business_crawling import crawl_business
from engineering_crawling import crawl_engineering
from science_crawling import crawl_science

# OpenAI API 설정
openai_api_key = os.getenv("OPENAI_API_KEY")

# MongoDB 설정
mongo_path = os.getenv("MONGO_PATH")
mongo_client = MongoClient(mongo_path, serverSelectionTimeoutMS=60000, socketTimeoutMS=60000)
db = mongo_client["notice-db"]
collection = db["test_test"]

def save_to_mongo(notices):
    """MongoDB에 공지사항 저장"""
    for notice in notices:
        # OpenAI 임베딩 생성
        # embedding = get_text_embedding(notice["context"])
        # notice["embedding"] = embedding

        # MongoDB에 삽입
        collection.update_one(
            {"department": notice["department"], "title": notice["title"], "date": notice["date"]},  # 중복 방지
            {"$set": notice},
            upsert=True
        )
        print(f"[{notice['department']}] Saved to MongoDB: {notice['title']}")


def crawl_and_store(department_name, crawl_function):
    """크롤링 → MongoDB 저장 전체 파이프라인"""
    start_time = time.time()
    print(f"[{department_name}] 공지사항 크롤링 시작")
    notices = crawl_function()
    unique_notices = pd.DataFrame(notices).drop_duplicates(subset=['title', 'date'])
    save_to_mongo(unique_notices.to_dict('records'))

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"[{department_name}] 모든 공지사항이 저장되었습니다. (소요 시간: {elapsed_time:.2f}초)")

def main():
    load_dotenv()
    # 스케줄링을 위한 래퍼 함수 정의
    schedule.every().day.at("10:00").do(crawl_and_store, "문과대학", crawl_liberal_arts)
    schedule.every().day.at("10:00").do(crawl_and_store, "상경대학", crawl_economics)
    schedule.every().day.at("10:00").do(crawl_and_store, "경영대학", crawl_business)
    schedule.every().day.at("10:00").do(crawl_and_store, "공과대학", crawl_engineering)
    schedule.every().day.at("10:00").do(crawl_and_store, "이과대학", crawl_science)
    print("스케줄러가 시작되었습니다. Ctrl+C로 중지할 수 있습니다.")

    while True:
        schedule.run_pending()
        time.sleep(1)

if __name__ == "__main__":
    main()