import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

def scrape_page_business(page):
    """경영대학 공지사항에서 특정 페이지의 데이터를 크롤링"""
    base_url = "https://ysb.yonsei.ac.kr/board.asp"
    params = {
        'mid': 'm06_01',
        'page': page
    }

    response = requests.get(base_url, params=params)
    response.encoding = 'euc-kr'  # 한글 인코딩 설정 (utf-8 사용 시 깨짐)
    soup = BeautifulSoup(response.text, 'html.parser')

    all_notices = []

    for row in soup.select('tr'):
        title_tag = row.select_one('a[href*="board.asp?act=view"]')
        date_tag = row.select_one('td.board_date')

        if title_tag and date_tag:
            title = title_tag.get_text(strip=True).replace("[공지]", "").strip()
            link = urljoin(base_url, title_tag['href'])
            date = date_tag.get_text(strip=True)
            context = scrape_notice_context(link)

            all_notices.append({
                'department': '경영대학',
                'title': title,
                'date': date,
                'link': link,
                'context': context
            })

    return all_notices

def scrape_notice_context(link):
    """경영대학 공지사항 상세 페이지에서 내용을 크롤링"""
    response = requests.get(link)
    response.encoding = 'euc-kr'  # 한글 인코딩 설정
    response.raise_for_status()
    soup = BeautifulSoup(response.text, 'html.parser')

    context_tag = soup.select_one('#BoardContent')
    return context_tag.get_text(strip=True) if context_tag else "내용 없음"

def crawl_business():
    """경영대학의 모든 공지사항 데이터를 반환"""
    all_notices = []
    for page in range(1, 6):
        all_notices.extend(scrape_page_business(page))
    return all_notices