import os
import re
import requests
from bs4 import BeautifulSoup
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

def scrape_page_urban(offset):
    """도시공학과 공지사항에서 특정 페이지의 데이터를 크롤링"""
    base_url = "https://urban.yonsei.ac.kr/urban/notice/notice.do"
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
        title_tag = row.select_one('td.text-left > div > a')
        date_tag = row.select_one('td:nth-child(5)')

        if title_tag and date_tag:
            title = title_tag.get_text(strip=True).replace("[공지]", "").strip()
            link = urljoin(base_url, title_tag['href'])
            raw_date = date_tag.get_text(strip=True)

            # 날짜 변환
            date = re.sub(r'(\d{4})\.(\d{1,2})\.(\d{1,2})', lambda x: f"{x.group(1)}.{x.group(2).zfill(2)}.{x.group(3).zfill(2)}", raw_date)

            # 상세 내용 크롤링
            context = scrape_notice_context(link)

            all_notices.append({
                'department': '도시공학과',
                'title': title,
                'date': date,
                'link': link,
                'context': context
            })

    return all_notices

def scrape_notice_context(link):
    """도시공학과 공지사항 상세 페이지에서 내용을 크롤링"""
    response = requests.get(link)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, 'html.parser')

    content_tag = soup.select_one('#jwxe_main_content > div > div > div > dl.board-write-box.board-write-box-v03 > dd')
    context = content_tag.get_text(strip=True) if content_tag else "내용 없음"
    return context

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

def crawl_urban():
    """도시공학과의 모든 공지사항 데이터를 반환"""
    all_notices = []
    for page in range(5):
        offset = page * 10
        print(f"도시공학과 {page + 1}페이지 크롤링 중...")
        all_notices.extend(scrape_page_urban(offset))

    return all_notices
