#type: 대출상품 Loan products 


import pandas as pd
import json
from pathlib import Path

# CSV 데이터를 수동 입력 대신 예제에 따라 생성
# 실제로는 pd.read_csv()로 불러와야 함

data = pd.read_csv("C:/Users/sj/Documents/CAPD/langgraph_rag_agent/db_preprocessing/rawdata/금융위원회_서민금융상품기본정보.csv")

json_docs = []
for idx, row in data.iterrows():
    id_str = f"people_{idx:05d}"
    product_name = row["finPrdNm"]
    bank = row["ofrInstNm"]
    summary = f"기관: {bank}, 상품명: {product_name}, 금리: {row['irt']}, 대출한도: {row['lnLmt']}, 기간: {row['maxTotLnTrm']}"
    
    if row["prdNm"] == "대출상품":
        doc = {
            "id": id_str,
            "bank": bank, #은행명
            "product_name": product_name, #상품명
            "type": row["prdNm"],
            "content": (
                f"기관명: {bank}\n상품명: {product_name}\n대출한도: {row['lnLmt']}\n"
                f"금리: {row['irtCtg']} ({row['irt']})\n대출기간: {row['maxTotLnTrm']} "
                f"(거치 {row['maxDfrmTrm']} / 상환 {row['maxRdptTrm']})\n상환방법: {row['rdptMthd']}\n"
                f"용도: {row['usge']}\n대상: {row['trgt']}\n가입방법: {row['jnMthd']}\n"
                f"우대/가산 조건: {row['prftAddIrtCond']}\n기타사항: {row['etcRefSbjc']}"
            ),
            "key_summary": summary,
            "metadata": {
                "기관명": bank,
                "상품명": product_name,
                "대출한도": row["lnLmt"],
                "금리구분": row["irtCtg"],
                "금리": row["irt"],
                "총대출기간": row["maxTotLnTrm"],
                "거치기간": row["maxDfrmTrm"],
                "상환기간": row["maxRdptTrm"],
                "상환방법": row["rdptMthd"],
                "용도": row["usge"],
                "대상": row["trgt"],
                "가입방법": row["jnMthd"],
                "우대조건": row["prftAddIrtCond"],
                "기타참고사항": row["etcRefSbjc"],
                "취급기관": row["hdlInst"],
                "연락처": row["cnpl"],
                "관련사이트": row["rltSite"],
                "상품존재여부": row["prdExisYn"],
                "key_summary": summary,
                "PDF 문서 링크":"-" #추후 수정 필요
            }
        }

    json_docs.append(doc)

# 저장
output_path = Path("C:/Users/sj/Documents/CAPD/langgraph_rag_agent/findata/loan_people.json")
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(json_docs, f, ensure_ascii=False, indent=2)

output_path.name