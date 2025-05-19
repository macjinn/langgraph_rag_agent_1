from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait, Select
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import time
import csv

options = Options()
options.add_argument("--headless")
driver = webdriver.Chrome(options=options)

url = "https://finlife.fss.or.kr/finlife/ldng/houseMrtg/list.do?menuNo=700007"

driver.get(url)

wait = WebDriverWait(driver, 15)

# 🔽 이 부분에서 'financeType' 셀렉터가 나타날 때까지 기다립니다.
wait.until(EC.presence_of_element_located((By.ID, "financeType")))

# 이제 안정적으로 드롭다운 선택
select_elem = Select(driver.find_element(By.ID, "financeType"))
select_elem.select_by_visible_text("전체")
wait = WebDriverWait(driver, 10)

# 1. '금융권역' 드롭다운에서 '전체' 선택
select_elem = Select(driver.find_element(By.ID, "financeType"))
select_elem.select_by_visible_text("전체")

# 2. '금융상품 검색' 버튼 클릭
driver.find_element(By.ID, "btnSearch").click()

# 3. 결과 테이블 로딩 대기
wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "tbody tr.basicList")))

# 전체 상품 목록 추출
rows = driver.find_elements(By.CSS_SELECTOR, "tbody tr.basicList")
print(f"총 상품 수: {len(rows)}")

results = []

for idx in range(len(rows)):
    try:
        # 목록 및 상세 리로딩
        driver.execute_script("window.scrollTo(0, 0);")
        rows = driver.find_elements(By.CSS_SELECTOR, "tbody tr.basicList")
        detail_btn = rows[idx].find_element(By.CSS_SELECTOR, "button")
        driver.execute_script("arguments[0].click();", detail_btn)
        time.sleep(1)

        # 페이지 전체 HTML 파싱
        soup = BeautifulSoup(driver.page_source, "html.parser")
        base_row = soup.select("tbody tr.basicList")[idx]
        base_cols = base_row.select("td")

        base_info = {
            "금융회사": base_cols[1].text.strip(),
            "상품명": base_cols[2].text.strip(),
            "주택종류": base_cols[3].text.strip(),
            "금리방식": base_cols[4].text.strip(),
            "상환방식": base_cols[5].text.strip(),
            "최저금리": base_cols[6].text.strip(),
            "최고금리": base_cols[7].text.strip(),
            "전월평균금리": base_cols[8].text.strip(),
            "월평균상환액": base_cols[9].text.strip(),
            "문의전화": base_cols[10].text.strip()
        }

        # 상세정보는 바로 다음 tr.detailList에 있음
        detail_row = soup.select("tbody tr.detailList")[idx]
        if detail_row:
            for tr in detail_row.select("table tbody tr"):
                tds = tr.select("td")
                if len(tds) >= 2:
                    key = tds[0].text.strip()
                    val = tds[1].text.strip()
                    base_info[key] = val

        results.append(base_info)

    except Exception as e:
        print(f"[{idx}] 오류 발생: {e}")
        continue

driver.quit()

# CSV 저장
all_keys = set().union(*(r.keys() for r in results))
with open("주택담보대출_전체_상세정보.csv", "w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(f, fieldnames=list(all_keys))
    writer.writeheader()
    writer.writerows(results)

print("CSV 저장 완료")
