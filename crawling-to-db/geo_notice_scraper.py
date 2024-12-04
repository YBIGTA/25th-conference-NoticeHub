import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import urllib3
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

def scrape_page_geo(page):
    """지구시스템과학과 공지사항 특정 페이지 크롤링"""
    base_url = "http://geo.yonsei.ac.kr/index.php"
    params = {
        'hCode': 'BOARD',
        'bo_idx': 2,
        'page': 'list',
        'pg': page
    }

    response = requests.get(base_url, params=params, verify=False)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, 'html.parser')

    all_notices = []

    for row in soup.select('tr'):
        title_tag = row.select_one('a[href^="?page=view"]')
        date_tag = row.select_one('td:nth-last-child(1)')

        if title_tag and date_tag:
            title = title_tag.get_text(strip=True)
            link = urljoin(base_url, title_tag['href'])
            date = date_tag.get_text(strip=True)

            # 상세 내용 크롤링
            context = scrape_notice_context(link)

            all_notices.append({
                'department': '지구시스템과학과',
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
    context_tag = soup.select_one('div.board_content')
    context = context_tag.get_text(strip=True) if context_tag else "내용 없음"
    
    return context

def crawl_geo():
    """지구시스템과학과의 모든 공지사항 데이터를 반환"""
    all_notices = []
    for page in range(1, 3):  # 1페이지부터 2페이지까지 크롤링
        print(f"지구시스템과학과 {page}페이지 크롤링 중...")
        all_notices.extend(scrape_page_geo(page))
    return all_notices
