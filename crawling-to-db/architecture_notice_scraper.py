import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import re
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

def scrape_page_architecture(offset):
    """건축공학과 공지사항 특정 페이지 크롤링"""
    url = f"https://arch.yonsei.ac.kr/notice/page/{offset}"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/113.0.0.0 Safari/537.36'
    }
    
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        print(f"페이지 {offset}에 접속할 수 없습니다. 상태 코드: {response.status_code}")
        return []
    
    soup = BeautifulSoup(response.text, 'html.parser')
    rows = soup.select('#body > div.dcore.dcore-list.dcore-notice > div.overflow-x-outer > div > table > tbody > tr')

    all_notices = []

    for row in rows:
        title_tag = row.select_one('td.title > a')
        date_tag = row.select_one('td.packed.hide-on-small-only')

        if title_tag and date_tag:
            title = title_tag.get_text(strip=True)
            link = "https://arch.yonsei.ac.kr" + title_tag['href']
            raw_date = date_tag.get_text(strip=True)

            # 날짜 형식 변환
            date = re.sub(r'(\d{4})\.(\d{2})\.(\d{2})', r'\1.\2.\3', raw_date)

            # 중복 제거
            if any(notice for notice in all_notices if notice['title'] == title and notice['date'] == date):
                continue

            # 상세 내용 크롤링
            context = scrape_notice_context(link)

            all_notices.append({
                'department': '건축공학과',
                'title': title,
                'date': date,
                'link': link,
                'context': context
            })
    
    return all_notices

def scrape_notice_context(link):
    """개별 공지사항의 상세 내용을 크롤링"""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/113.0.0.0 Safari/537.36'
    }
    response = requests.get(link, headers=headers)
    if response.status_code != 200:
        return "내용 없음"
    
    soup = BeautifulSoup(response.text, 'html.parser')
    content_tag = soup.select_one('#body > div.dcore.dcore-view.dcore-notice')
    return content_tag.get_text(strip=True) if content_tag else "내용 없음"

def upload_to_s3(data, filename):
    """공지사항 데이터를 S3에 업로드"""
    try:
        # CSV 파일 생성
        csv_data = data.to_csv(index=False, encoding='utf-8-sig')

        # S3 업로드
        s3.put_object(
            Bucket=BUCKET_NAME,
            Key=filename,
            Body=csv_data,
            ContentType='text/csv'
        )
        print(f"S3에 업로드 성공: {filename}")
    except Exception as e:
        print(f"S3 업로드 실패: {e}")

def crawl_architecture():
    """건축공학과의 모든 공지사항 데이터를 반환"""
    all_notices = []
    for page in range(1, 6):  # 1페이지부터 5페이지까지 크롤링
        offset = page * 10
        print(f"건축공학과 {page}페이지 크롤링 중...")
        all_notices.extend(scrape_page_architecture(offset))

    return all_notices
