import json
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup

# 셀레니움 드라이버 실행
options = webdriver.ChromeOptions()
options.add_argument('--headless')  # 창 없이 실행
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

# 금융상품 페이지 열기
url = "https://www.fsb.or.kr/rateinst_0100.act"
driver.get(url)
time.sleep(3)

documents = []

# 상품 테이블 전체 로딩 후 탐색
product_rows = driver.find_elements(By.CSS_SELECTOR, "table.tbl_ty1 tbody tr")

for idx, row in enumerate(product_rows):
    try:
        # ‘자세히’ 버튼 클릭
        detail_button = row.find_element(By.CSS_SELECTOR, "a[title='자세히 보기']")
        driver.execute_script("arguments[0].click();", detail_button)
        time.sleep(2)

        # 팝업 또는 하단 레이어에서 정보 추출
        detail_html = driver.page_source
        soup = BeautifulSoup(detail_html, "html.parser")
        detail_section = soup.select_one("div.layer_pop")  # 실제 팝업 div 선택자 확인 필요

        # 예시 데이터 추출 (실제 HTML 구조에 따라 수정 필요)
        bank = row.find_elements(By.TAG_NAME, "td")[0].text.strip()
        product_name = row.find_elements(By.TAG_NAME, "td")[1].text.strip()
        product_type = "적금"  # 예시로 지정
        content = detail_section.get_text(strip=True)
        key_summary = content[:100]  # 요약 예시

        documents.append({
            "id": f"적금_{idx:03d}",
            "bank": bank,
            "product_name": product_name,
            "type": product_type,
            "content": content,
            "key_summary": key_summary,
            "metadata": {}  # 필요시 더 채우기
        })

        # 닫기 버튼 클릭
        close_btn = driver.find_element(By.CSS_SELECTOR, "div.layer_pop .btn_close")
        driver.execute_script("arguments[0].click();", close_btn)
        time.sleep(1)

    except Exception as e:
        print(f"[{idx}] 오류 발생: {e}")
        continue

driver.quit()

# JSON 저장
with open("fsb_products.json", "w", encoding="utf-8") as f:
    json.dump({"documents": documents}, f, ensure_ascii=False, indent=2)
