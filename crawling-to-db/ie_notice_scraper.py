import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import urllib3
import re
import boto3
from dotenv import load_dotenv

# SSL 경고 비활성화
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# .env 파일에서 환경 변수 로드
load_dotenv()

# 환경 변수에서 S3 설정 불러오기
AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_KEY")
AWS_REGION = os.getenv("AWS_REGION")
BUCKET_NAME = os.getenv("BUCKET_NAME")

# AWS S3 클라이언트 생성
s3 = boto3.client(
    's3',
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY,
    region_name=AWS_REGION
)

def scrape_page_industrial_engineering(offset):
    """산업공학과 공지사항 특정 페이지 크롤링"""
    base_url = "https://ie.yonsei.ac.kr/ie/community/under_notice.do"
    params = {
        'mode': 'list',
        'articleLimit': 10,
        'article.offset': offset
    }

    response = requests.get(base_url, params=params, verify=False)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, 'html.parser')

    all_notices = []

    for row in soup.select('#jwxe_main_content > div > div > div > ul > li'):
        title_tag = row.select_one('dl > dt > a')
        date_tag = row.select_one('dl > dd.board-list-content-info > span.board-list-content-date')

        if title_tag and date_tag:
            title = title_tag.get_text(strip=True).replace("[학부][안내]", "").strip()
            link = urljoin(base_url, title_tag['href'])
            date = date_tag.get_text(strip=True)

            # 상세 내용 크롤링
            context = scrape_notice_context(link)

            all_notices.append({
                'department': '산업공학과',
                'title': title,
                'date': date,
                'link': link,
                'context': context
            })

    return all_notices

def scrape_notice_context(link):
    """개별 공지사항의 상세 내용을 크롤링"""
    response = requests.get(link, verify=False)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, 'html.parser')

    # 상세 내용 크롤링
    context_tag = soup.select_one('#jwxe_main_content > div > div.board-wrap')
    context = context_tag.get_text(strip=True) if context_tag else "내용 없음"
    
    return context

def crawl_industrial_engineering():
    """산업공학과의 모든 공지사항 데이터를 반환"""
    all_notices = []
    for page in range(5):  # 1페이지부터 5페이지까지 크롤링
        offset = page * 10
        print(f"산업공학과 {page + 1}페이지 크롤링 중...")
        all_notices.extend(scrape_page_industrial_engineering(offset))
    return all_notices