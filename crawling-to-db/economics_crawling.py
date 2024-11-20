import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

def scrape_page_economics(offset):
    """상경대학 공지사항에서 특정 페이지의 데이터를 크롤링"""
    base_url = "https://yce.yonsei.ac.kr/ybe/notice/notice.do"
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
        td_tags = row.select('td')

        if title_tag and len(td_tags) > 0:
            title = title_tag.get_text(strip=True).replace("[공지]", "").strip()
            link = urljoin(base_url, title_tag['href'])
            date = "20" + td_tags[-1].get_text(strip=True)
            context = scrape_notice_context(link)

            all_notices.append({
                'department': '상경대학',
                'title': title,
                'date': date,
                'link': link,
                'context': context
            })

    return all_notices

def scrape_notice_context(link):
    """상경대학 공지사항 상세 페이지에서 내용을 크롤링"""
    response = requests.get(link)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, 'html.parser')

    context_tag = soup.select_one('dd .fr-view')
    return context_tag.get_text(strip=True) if context_tag else "내용 없음"

def crawl_economics():
    """상경대학의 모든 공지사항 데이터를 반환"""
    all_notices = []
    for page in range(5):
        offset = page * 10
        all_notices.extend(scrape_page_economics(offset))
    return all_notices