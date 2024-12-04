import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import pandas as pd
import re
from dotenv import load_dotenv

# .env 파일에서 환경 변수 로드
load_dotenv()

def scrape_page_english(offset):
    """영어영문학과 공지사항 특정 페이지 크롤링"""
    base_url = "https://ysenglish.yonsei.ac.kr/ysenglish/process/process.do"
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
        title_tag = row.select_one('a.c-board-title')
        tds = row.select('td')

        if title_tag and tds:
            title = title_tag.get_text(strip=True).replace("[공지]", "").strip()
            link = urljoin(base_url, title_tag['href'])
            raw_date = tds[-1].get_text(strip=True)

            if re.match(r'^\d{2}\.\d{2}\.\d{2}$', raw_date):
                date = f"20{raw_date}"
            else:
                continue

            context = scrape_notice_context(link)

            all_notices.append({
                'department': '영어영문학과',
                'title': title,
                'date': date,
                'link': link,
                'context': context
            })

    return all_notices

def scrape_notice_context(link):
    """영어영문학과 공지사항 상세 페이지 크롤링"""
    response = requests.get(link)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, 'html.parser')

    context_tag = soup.select_one('div.fr-view')
    context = context_tag.get_text(strip=True) if context_tag else "내용 없음"
    return context

def crawl_english():
    """영어영문학과의 모든 공지사항 데이터를 반환"""
    all_notices = []
    for page in range(5):
        offset = page * 10
        print(f"영어영문학과 {page + 1}페이지 크롤링 중...")
        all_notices.extend(scrape_page_english(offset))
    return all_notices
