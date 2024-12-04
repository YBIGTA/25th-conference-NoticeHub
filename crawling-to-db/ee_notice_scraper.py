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

def format_date(raw_date):
    """날짜 형식을 변환"""
    match = re.match(r'(\d{2})\.(\d{2})\.(\d{2})', raw_date)
    if match:
        return f"20{match.group(1)}.{match.group(2)}.{match.group(3)}"
    return "날짜 없음"

def scrape_page_ee(offset):
    """전기전자공학과 공지사항에서 특정 페이지의 데이터를 크롤링"""
    base_url = "https://ee.yonsei.ac.kr/ee/community/academic_notice.do"
    params = {
        'mode': 'list',
        'articleLimit': 10,
        'article.offset': offset
    }

    response = requests.get(base_url, params=params)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, 'html.parser')

    all_notices = []

    for row in soup.select('tr'):
        # 제목과 링크 추출
        title_tag = row.select_one('a.c-board-title')
        if not title_tag:
            continue
        title = title_tag.get_text(strip=True).replace("[공지]", "").strip()
        link = urljoin(base_url, title_tag['href'])

        # 날짜 추출
        date_tag = row.select_one('td:nth-child(5)')
        raw_date = date_tag.get_text(strip=True) if date_tag else "날짜 없음"
        formatted_date = format_date(raw_date)

        # 상세 내용 크롤링
        context = scrape_notice_context(link)

        # 데이터 저장
        all_notices.append({
            'department': '전기전자공학과',
            'title': title,
            'date': formatted_date,
            'link': link,
            'context': context
        })
    
    return all_notices

def scrape_notice_context(link):
    """공지사항 상세 페이지에서 내용을 크롤링"""
    try:
        response = requests.get(link)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        # 공지사항 내용 추출
        context_tag = soup.select_one('div.fr-view')
        context = context_tag.get_text(strip=True) if context_tag else "내용 없음"
        return context
    except Exception as e:
        print(f"Error fetching context for link {link}: {e}")
        return "링크 오류"

def upload_to_s3(data, filename):
    """공지사항 데이터를 S3에 업로드"""
    try:
        csv_data = data.to_csv(index=False, encoding='utf-8-sig')

        s3.put_object(
            Bucket=BUCKET_NAME,
            Key=filename,
            Body=csv_data,
            ContentType='text/csv'
        )
        print(f"S3에 업로드 성공: {filename}")
    except Exception as e:
        print(f"S3 업로드 실패: {e}")

def crawl_ee():
    """전기전자공학과의 모든 공지사항 데이터를 반환"""
    all_notices = []
    for page in range(5):  # 1페이지부터 5페이지까지 크롤링
        offset = page * 10
        print(f"전기전자공학과 {page + 1}페이지 크롤링 중...")
        all_notices.extend(scrape_page_ee(offset))

    return all_notices
