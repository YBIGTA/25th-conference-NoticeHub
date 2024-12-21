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
from bio_crawling import crawl_bio
from ai_crawling import crawl_ai
from theology_crawling import crawl_theology
from social_sciences_crawling import crawl_social_sciences
from music_college_crawling import crawl_music
from che_college_crawling import crawl_che
from edu_college_crawling import crawl_edu
from university_college_crawling import crawl_university_college
from glc_crawling import crawl_glc
from architecture_notice_scraper import crawl_architecture
from astro_notice_scraper import crawl_astro
from chemeng_notice_scraper import crawl_chemeng
from chemistry_notice_scraper import crawl_chemistry
from chinese_language_notice_scraper import crawl_chinese
from economics_notice_scraper import crawl_economics_notice
from ee_notice_scraper import crawl_ee
from english_language_notice_scraper import crawl_english
from french_language_notice_scraper import crawl_french_language_literature
from geo_notice_scraper import crawl_geo
from german_language_literature_notice_scraper import crawl_german_language_literature
from history_notice_scraper import crawl_history
from korean_literature_notice_scraper import crawl_korean
from library_information_notice_scraper import crawl_library_information_science
from math_notice_scraper import crawl_math
from philosophy_notice_scraper import crawl_philosophy
from physics_notice_scraper import crawl_physics
from russian_language_notice_scraper import crawl_russian_language_literature
from statistics_notice_scraper import crawl_statistics
from urban_notice_scraper import crawl_urban


# 스케줄 시간 리스트
SCHEDULED_TIMES = ["10:00", "18:00"]

load_dotenv()

# OpenAI API 설정
openai_api_key = os.getenv("OPENAI_API_KEY")

# MongoDB 설정
MONGO_PATH = os.getenv("MONGO_PATH")
client = MongoClient(MONGO_PATH)
db = client["notice-db"]
collection = db["final_notices"]


def save_to_mongo(notices):
    """MongoDB에 공지사항 저장 및 임베딩 생성"""
    for notice in notices:
        try:
            # 기존 데이터 확인 (임베딩이 이미 존재하는 경우 재생성하지 않음)
            existing_notice = collection.find_one(
                {"department": notice["department"], "title": notice["title"], "date": notice["date"]}
            )

            if existing_notice:
                # 기존 데이터가 있으면 메시지만 출력하고 업데이트를 건너뜀
                print(f"[{notice['department']}] Already existing: {notice['title']}")
                continue

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
    # 스케줄링을 위한 래퍼 함수 정의
    for scheduled_time in SCHEDULED_TIMES:
        schedule.every().day.at(scheduled_time).do(crawl_and_store, "문과대학", crawl_liberal_arts)
        schedule.every().day.at(scheduled_time).do(crawl_and_store, "상경대학", crawl_economics)
        schedule.every().day.at(scheduled_time).do(crawl_and_store, "경영대학", crawl_business)
        schedule.every().day.at(scheduled_time).do(crawl_and_store, "공과대학", crawl_engineering)
        schedule.every().day.at(scheduled_time).do(crawl_and_store, "이과대학", crawl_science)
        schedule.every().day.at(scheduled_time).do(crawl_and_store, "생명시스템대학", crawl_bio)
        schedule.every().day.at(scheduled_time).do(crawl_and_store, "인공지능융합대학", crawl_ai)
        schedule.every().day.at(scheduled_time).do(crawl_and_store, "신과대학", crawl_theology)
        schedule.every().day.at(scheduled_time).do(crawl_and_store, "사회과학대학", crawl_social_sciences)
        schedule.every().day.at(scheduled_time).do(crawl_and_store, "음악대학", crawl_music)
        schedule.every().day.at(scheduled_time).do(crawl_and_store, "생활과학대학", crawl_che)
        schedule.every().day.at(scheduled_time).do(crawl_and_store, "교육과학대학", crawl_edu)
        schedule.every().day.at(scheduled_time).do(crawl_and_store, "학부대학", crawl_university_college)
        schedule.every().day.at(scheduled_time).do(crawl_and_store, "글로벌인재대학", crawl_glc)
        schedule.every().day.at(scheduled_time).do(crawl_and_store, "건축학과", crawl_architecture)
        schedule.every().day.at(scheduled_time).do(crawl_and_store, "천문학과", crawl_astro)
        schedule.every().day.at(scheduled_time).do(crawl_and_store, "화학공학과", crawl_chemeng)
        schedule.every().day.at(scheduled_time).do(crawl_and_store, "화학과", crawl_chemistry)
        schedule.every().day.at(scheduled_time).do(crawl_and_store, "중어중문학과", crawl_chinese)
        schedule.every().day.at(scheduled_time).do(crawl_and_store, "경제학과", crawl_economics_notice)
        schedule.every().day.at(scheduled_time).do(crawl_and_store, "전기전자공학과", crawl_ee)
        schedule.every().day.at(scheduled_time).do(crawl_and_store, "영어영문학과", crawl_english)
        schedule.every().day.at(scheduled_time).do(crawl_and_store, "불어불문학과", crawl_french_language_literature)
        schedule.every().day.at(scheduled_time).do(crawl_and_store, "지리학과", crawl_geo)
        schedule.every().day.at(scheduled_time).do(crawl_and_store, "독어독문학과", crawl_german_language_literature)
        schedule.every().day.at(scheduled_time).do(crawl_and_store, "사학과", crawl_history)
        schedule.every().day.at(scheduled_time).do(crawl_and_store, "국어국문학과", crawl_korean)
        schedule.every().day.at(scheduled_time).do(crawl_and_store, "문헌정보학과", crawl_library_information_science)
        schedule.every().day.at(scheduled_time).do(crawl_and_store, "수학과", crawl_math)
        schedule.every().day.at(scheduled_time).do(crawl_and_store, "철학과", crawl_philosophy)
        schedule.every().day.at(scheduled_time).do(crawl_and_store, "물리학과", crawl_physics)
        schedule.every().day.at(scheduled_time).do(crawl_and_store, "노어노문학과", crawl_russian_language_literature)
        schedule.every().day.at(scheduled_time).do(crawl_and_store, "통계학과", crawl_statistics)
        schedule.every().day.at(scheduled_time).do(crawl_and_store, "도시공학과", crawl_urban)
    print("스케줄러가 시작되었습니다. Ctrl+C로 중지할 수 있습니다.")

    while True:
        schedule.run_pending()
        time.sleep(1)

if __name__ == "__main__":
    main()