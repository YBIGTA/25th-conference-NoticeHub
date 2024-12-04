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

# 스케줄 시간 리스트
SCHEDULED_TIMES = ["10:00, 18:00"]

load_dotenv()

# OpenAI API 설정
openai_api_key = os.getenv("OPENAI_API_KEY")

# MongoDB 설정
MONGO_PATH = os.getenv("MONGO_PATH")
client = MongoClient(MONGO_PATH)
db = client["notice-db"]
collection = db["test_embedded_ec2"]


def save_to_mongo(notices):
    """MongoDB에 공지사항 저장 및 임베딩 생성"""
    for notice in notices:
        try:
            # 기존 데이터 확인 (임베딩이 이미 존재하는 경우 재생성하지 않음)
            existing_notice = collection.find_one(
                {"department": notice["department"], "title": notice["title"], "date": notice["date"]}
            )

            if existing_notice and "embedding" in existing_notice:
                # 기존 임베딩 재사용
                notice["embedding"] = existing_notice["embedding"]
            else:
                # 새로운 임베딩 생성
                embedding = get_text_embedding(notice["context"])
                notice["embedding"] = embedding
                print(f"[{notice['department']}] Generated new embedding for: {notice['title']}")

            # MongoDB에 삽입 또는 업데이트
            collection.update_one(
                {"department": notice["department"], "title": notice["title"], "date": notice["date"]},
                {"$set": notice},
                upsert=True
            )
            print(f"[{notice['department']}] Saved to MongoDB: {notice['title']}")

        except Exception as e:
            print(f"Error saving notice to MongoDB: {e}")


def crawl_and_store(department_name, crawl_function):
    """크롤링 → MongoDB 저장 전체 파이프라인"""
    start_time = time.time()
    print(f"[{department_name}] 공지사항 크롤링 시작")

    try:
        notices = crawl_function()
        unique_notices = pd.DataFrame(notices).drop_duplicates(subset=['title', 'date'])
        save_to_mongo(unique_notices.to_dict('records'))
    except Exception as e:
        print(f"Error in crawl_and_store for {department_name}: {e}")

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"[{department_name}] 모든 공지사항이 저장되었습니다. (소요 시간: {elapsed_time:.2f}초)")

def main():
    # # 스케줄링을 위한 래퍼 함수 정의
    # for scheduled_time in SCHEDULED_TIMES:
    #     schedule.every().day.at(scheduled_time).do(crawl_and_store, "문과대학", crawl_liberal_arts)
    #     schedule.every().day.at(scheduled_time).do(crawl_and_store, "상경대학", crawl_economics)
    #     schedule.every().day.at(scheduled_time).do(crawl_and_store, "경영대학", crawl_business)
    #     schedule.every().day.at(scheduled_time).do(crawl_and_store, "공과대학", crawl_engineering)
    #     schedule.every().day.at(scheduled_time).do(crawl_and_store, "이과대학", crawl_science)
    # print("스케줄러가 시작되었습니다. Ctrl+C로 중지할 수 있습니다.")

    # while True:
    #     schedule.run_pending()
    #     time.sleep(1)

    crawl_and_store("문과대학", crawl_liberal_arts)

if __name__ == "__main__":
    main()