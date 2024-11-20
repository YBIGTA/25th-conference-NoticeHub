import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

def scrape_page_engineering(offset):
    """공과대학 공지사항에서 특정 페이지의 데이터를 크롤링"""
    base_url = "https://engineering.yonsei.ac.kr/engineering/board/notice.do"
    params = {
        'mode': 'list',
        'articleLimit': 10,
        'article.offset': offset
    }

    response = requests.get(base_url, params=params)
    response.raise_for_status()  # 요청 성공 확인
    soup = BeautifulSoup(response.text, 'html.parser')

    all_notices = []

    for row in soup.select('tr'):  # 각 공지사항이 있는 행 선택
        title_tag = row.select_one('a.c-board-title')
        td_tags = row.select('td')  # <td> 태그 모두 선택

        if title_tag and len(td_tags) > 0:  # <td> 태그가 존재하는지 확인
            title = title_tag.get_text(strip=True).replace("[공지]", "").strip()  # 공지 태그 제거
            link = urljoin(base_url, title_tag['href'])  # 절대 경로로 변환

            # 작성일은 마지막 <td> 태그에서 추출
            date = "20" + td_tags[-1].get_text(strip=True)  # 연도를 맞춘 작성일

            # 공지사항의 상세 페이지로 이동하여 내용 크롤링
            context = scrape_notice_context(link)

            all_notices.append({
                'department': '공과대학',
                'title': title,
                'date': date,
                'link': link,
                'context': context
            })

    return all_notices

def scrape_notice_context(link):
    """공과대학 공지사항 상세 페이지에서 내용을 크롤링"""
    response = requests.get(link)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, 'html.parser')

    # 상세 내용이 담긴 <div class="fr-view">에서 텍스트 추출
    context_tag = soup.select_one('div.fr-view')
    return context_tag.get_text(strip=True) if context_tag else "내용 없음"

def crawl_engineering():
    """공과대학의 모든 공지사항 데이터를 반환"""
    all_notices = []
    for page in range(5):
        offset = page * 10
        all_notices.extend(scrape_page_engineering(offset))
    return all_notices