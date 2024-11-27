import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

def scrape_page_science(page_number):
    """이과대학 공지사항에서 특정 페이지의 데이터를 크롤링"""
    base_url = "http://science.yonsei.ac.kr"
    notice_url = f"{base_url}/community/notice"
    url = f"{notice_url}?p={page_number}"

    response = requests.get(url)
    response.raise_for_status()  # 요청 성공 확인
    soup = BeautifulSoup(response.text, 'html.parser')

    all_notices = []

    for row in soup.select('tr'):  # 각 공지사항이 있는 행 선택
        title_tag = row.select_one('td.nxb-list-table__title a')
        date_tag = row.select_one('td.nxb-list-table__date')

        if title_tag and date_tag:  # 제목과 날짜가 있는지 확인
            title = title_tag.get_text(strip=True)
            link = urljoin(base_url, title_tag['href'])  # 절대 경로로 변환
            date = date_tag.get_text(strip=True)

            # 공지사항의 상세 페이지로 이동하여 내용 크롤링
            context = scrape_notice_context(link)

            all_notices.append({
                'department': '이과대학',
                'title': title,
                'date': date,
                'link': link,
                'context': context
            })

    return all_notices

def scrape_notice_context(link):
    """이과대학 공지사항 상세 페이지에서 내용을 크롤링"""
    response = requests.get(link)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, 'html.parser')

    # 상세 내용이 담긴 <div class="editor-contents">에서 텍스트 추출
    context_tag = soup.select_one('div.editor-contents')
    return context_tag.get_text(strip=True) if context_tag else "내용 없음"

def crawl_science():
    """이과대학의 모든 공지사항 데이터를 반환"""
    all_notices = []
    for page in range(1, 6):  # 1페이지부터 5페이지까지 크롤링
        all_notices.extend(scrape_page_science(page))
    return all_notices