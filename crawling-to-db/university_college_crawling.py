import os
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

def scrape_page_university_college(offset):
    """학부대학 공지사항에서 특정 페이지의 데이터를 크롤링"""
    base_url = "https://universitycollege.yonsei.ac.kr/fresh/infomation/student.do"
    params = {
        'mode': 'list',
        'articleLimit': 10,
        'article.offset': offset
    }

    response = requests.get(base_url, params=params)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, 'html.parser')

    all_notices = []

    for row in soup.select('td.text-left'):
        title_tag = row.select_one('a.c-board-title')
        date_tag = row.select_one('div.c-board-info-m > span:nth-child(2)')

        if title_tag and date_tag:
            title = title_tag.get_text(strip=True).replace("[공지]", "").strip()
            link = urljoin(base_url, title_tag['href'])
            date = date_tag.get_text(strip=True)
            context, images = scrape_notice_context(link, title)

            all_notices.append({
                'department': '학부대학',
                'title': title,
                'date': date,
                'link': link,
                'context': context,
                'images': ", ".join(images)  # S3 이미지 URL을 쉼표로 구분하여 저장
            })

    return all_notices

def scrape_notice_context(link, title):
    """학부대학 공지사항 상세 페이지에서 내용을 크롤링하고 이미지를 S3에 업로드"""
    response = requests.get(link)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, 'html.parser')

    context_tag = soup.select_one('dd > div.fr-view')
    context = context_tag.get_text(strip=True) if context_tag else "내용 없음"

    # 이미지 업로드
    image_tags = soup.select('dd > div.fr-view img')
    image_urls = []
    for idx, img_tag in enumerate(image_tags, start=1):
        img_src = img_tag.get('src')
        if img_src:
            img_url = urljoin(link, img_src)
            s3_image_url = upload_image_to_s3(img_url, title, idx)
            if s3_image_url:
                image_urls.append(s3_image_url)
    
    return context, image_urls

def upload_image_to_s3(img_url, title, idx):
    """이미지를 다운로드하고 S3에 업로드 (중복 업로드 방지)"""
    try:
        response = requests.get(img_url, stream=True)
        response.raise_for_status()

        # 이미지 파일명 생성: title + 번호 + 확장자
        ext = os.path.splitext(img_url)[-1]
        safe_title = "".join(c if c.isalnum() or c in " _-" else "_" for c in title)
        s3_image_key = f"images/{safe_title}_{idx}{ext}"

        # S3에 파일 존재 여부 확인
        try:
            s3.head_object(Bucket=BUCKET_NAME, Key=s3_image_key)
            print(f"이미 존재하는 파일: {s3_image_key}")
            return f"https://{BUCKET_NAME}.s3.{AWS_REGION}.amazonaws.com/{s3_image_key}"
        except s3.exceptions.ClientError as e:
            if e.response['Error']['Code'] != '404':
                print(f"S3 객체 확인 중 에러 발생: {e}")
                return None

        # 파일이 존재하지 않으면 업로드
        s3.upload_fileobj(response.raw, BUCKET_NAME, s3_image_key)
        s3_image_url = f"https://{BUCKET_NAME}.s3.{AWS_REGION}.amazonaws.com/{s3_image_key}"
        print(f"이미지 업로드 성공: {s3_image_url}")
        return s3_image_url

    except Exception as e:
        print(f"이미지 업로드 실패: {img_url}, 에러: {e}")
        return None

def crawl_university_college():
    """학부대학의 모든 공지사항 데이터를 반환"""
    all_notices = []
    for page in range(5):
        offset = page * 10
        print(f"학부대학 {page + 1}페이지 크롤링 중...")
        all_notices.extend(scrape_page_university_college(offset))
    return all_notices