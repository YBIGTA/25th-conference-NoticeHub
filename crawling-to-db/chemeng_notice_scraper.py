import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import pandas as pd
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

def scrape_page_chemeng(page):
    """화공생명공학과 공지사항 특정 페이지 크롤링"""
    base_url = f"https://chemeng.yonsei.ac.kr/?c=209&s=209&gbn=list&gp={page}"
    response = requests.get(base_url)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, 'html.parser')

    all_notices = []

    rows = soup.select('#bbsStandardWrap > table > tbody > tr')
    for row in rows:
        # 제목과 링크 추출
        title_tag = row.select_one('td.col-tit > a')
        if title_tag:
            title = title_tag.get_text(strip=True)
            link = urljoin("https://chemeng.yonsei.ac.kr", title_tag['href'])  # 상대 경로를 절대 경로로 변환
        else:
            title = "제목 없음"
            link = None

        # 날짜 추출
        date_tag = row.select_one('td.col-date')
        date = date_tag.get_text(strip=True) if date_tag else "날짜 없음"

        # 중복 확인: 동일한 title과 date가 이미 추가된 경우 스킵
        if any(notice for notice in all_notices if notice['title'] == title and notice['date'] == date):
            continue

        # 상세 내용 크롤링
        context = scrape_notice_context(link) if link else "내용 없음"

        all_notices.append({
            'department': '화공생명공학과',
            'title': title,
            'date': date,
            'link': link,
            'context': context
        })
    
    return all_notices

def scrape_notice_context(link):
    """개별 공지사항의 상세 내용을 크롤링"""
    response = requests.get(link)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, 'html.parser')

    # 상세 내용 크롤링
    content_tag = soup.select_one('#bbsContents')
    context = content_tag.get_text(strip=True) if content_tag else "내용 없음"
    return context

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

def crawl_chemeng():
    """화공생명공학과의 모든 공지사항 데이터를 반환"""
    all_notices = []
    for page in range(1, 6):  # 1페이지부터 5페이지까지 크롤링
        print(f"화공생명공학과 {page}페이지 크롤링 중...")
        all_notices.extend(scrape_page_chemeng(page))

    return all_notices
