import os
import requests
from bs4 import BeautifulSoup
from datetime import datetime
from urllib.parse import urljoin
import boto3
from dotenv import load_dotenv

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

def scrape_page_atmos():
    """대기과학과 공지사항 전체 데이터를 크롤링"""
    base_url = "https://atmos.yonsei.ac.kr/categories/%EA%B3%B5%EC%A7%80%EC%82%AC%ED%95%AD/"
    response = requests.get(base_url)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, 'html.parser')

    all_notices = []

    rows = soup.select('div.col-xs-11')  # 제목과 날짜를 포함하는 부모 태그 선택
    for row in rows:
        # 제목과 링크 추출
        title_tag = row.select_one('h3 a')
        date_tag = row.select_one('p.date-comments i.fa.fa-calendar-o')

        if title_tag and date_tag:
            title = title_tag.get_text(strip=True)  # 제목 추출
            link = urljoin(base_url, title_tag['href'])  # 절대 경로로 변환
            raw_date = date_tag.next_sibling.strip()  # 날짜 텍스트 추출
            date = convert_date(raw_date)

            # 상세 내용 크롤링
            context = scrape_notice_context(link)

            all_notices.append({
                'department': '대기과학과',
                'title': title,
                'date': date,
                'link': link,
                'context': context
            })
    
    return all_notices

def convert_date(raw_date):
    """날짜를 'YYYY.MM.DD' 형식으로 변환"""
    try:
        return datetime.strptime(raw_date, '%B %d, %Y').strftime('%Y.%m.%d')
    except ValueError:
        return raw_date

def scrape_notice_context(link):
    """개별 공지사항의 상세 내용을 크롤링"""
    response = requests.get(link)
    if response.status_code == 404:
        return "링크를 찾을 수 없습니다."
    response.raise_for_status()
    soup = BeautifulSoup(response.text, 'html.parser')

    # 상세 내용 크롤링
    content_tag = soup.select_one('div#post-content')
    context = content_tag.get_text(strip=True) if content_tag else "내용 없음"

    return context

def crawl_atmos():
    """대기과학과의 모든 공지사항 데이터를 반환"""
    all_notices = scrape_page_atmos()
    return all_notices
