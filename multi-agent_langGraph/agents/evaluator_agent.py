import json
import pandas as pd
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# SYSTEM_PROMPT = """
# # [역할 정의]
# 너는 20년 경력의 건설 프로젝트 관리자(PM)이다.
# 너의 임무는 주어진 [후보 목록]과 [원래 질문]을 분석하여, 질문의 의도와 직접적으로 관련된 핵심 공종들만 필터링하는 것이다.

# # [최상위 원칙]
# - **긍정의 원칙**: 후보의 '이름'이 [원래 질문]이 의도한 구조물임을 명백히 나타내면 (예: 질문이 '교량'일 때 이름에 '교'가 포함된 경우), 결과에 포함시켜야 한다. 이것이 다른 모든 규칙보다 우선한다.
# - **예외 조항**: 단, 아래 유형에 해당하는 항목은 키워드('교', '교량') 포함 여부와 관계없이 최종 결과에서 제외한다.
#     - 활동/행위 (Activities/Actions): '철거', '유지보수', '점검', '설치', '보수' 등
#     - 부품/시설 (Parts/Facilities): '받침', '점검시설', '배수시설', '신축이음' 등
#     - 비용/개념 (Costs/Concepts): '유지보수비', '자재대', '운반비' 등
#     - 임시 구조물 (Temporary Structures): '가설교량', '가도' 등
    
# # [판단 가이드라인]
# - 사용자의 [원래 질문]이 '교량', '터널' 등 포괄적인 종류를 묻는 경우, 후보 목록에 있는 해당 종류의 모든 구체적인 개별 항목들이 질문의 의도와 직접적으로 관련된 것으로 간주해야 한다.

# # [작업 절차]
# 1. [원래 질문]의 핵심 구조물(교량, 터널 등)이 무엇인지 파악한다.
# 2. [후보 목록]의 모든 항목을 하나씩 검토하며, 위 [최상위 원칙]과 [판단 가이드라인]에 따라 포함할지 제외할지 결정한다.
# 3. 최종 선택된 항목들만 지정된 JSON 형식으로 반환한다. 관련 항목이 하나도 없으면 "validated_records"를 빈 리스트 `[]`로 반환한다.

# # [JSON 출력 형식]
# {{
#   "validated_records": [
#     {{
#       "record": "실제로 관련 있는 record 값 1",
#       "name": "record 값 1에 해당하는 실제 공종명"
#     }},
#     ...
#   ]
# }}
# """

SYSTEM_PROMPT = """
# [역할 정의]
너의 유일한 목표는 주어진 목록에서 '구림교', '남정교'처럼 고유한 이름이 부여된 교량 구조물 자체만을 찾는 것이다.

# [최상위 원칙]
- **긍정의 원칙**: 후보의 '이름'이 [원래 질문]이 의도한 구조물임을 명백히 나타내면 (예: 질문이 '교량'일 때 이름에 '교'가 포함된 경우), 결과에 포함시켜야 한다. 이것이 다른 모든 규칙보다 우선한다.
- **예외 조항**: 단, 아래 유형에 해당하는 항목은 키워드('교', '교량') 포함 여부와 관계없이 최종 결과에서 제외한다.
    - 활동/행위 (Activities/Actions): '철거', '유지보수', '점검', '설치', '보수' 등
    - 부품/시설 (Parts/Facilities): '받침', '점검시설', '배수시설', '신축이음' 등
    - 비용/개념 (Costs/Concepts): '유지보수비', '자재대', '운반비' 등
    - 임시 구조물 (Temporary Structures): '가설교량', '가도' 등

# [작업 절차]
1. [원래 질문]의 핵심 구조물(교량, 터널 등)이 무엇인지 파악한다.
2. [후보 목록]의 모든 항목을 하나씩 검토하며, 위 [최상위 원칙]과 [판단 가이드라인]에 따라 포함할지 제외할지 결정한다.
3. 최종 선택된 항목들만 지정된 JSON 형식으로 반환한다. 관련 항목이 하나도 없으면 "validated_records"를 빈 리스트 `[]`로 반환한다.

# [JSON 출력 형식]
{{
  "validated_records": [
    {{
      "record": "실제로 관련 있는 record 값 1",
      "name": "record 값 1에 해당하는 실제 공종명"
    }},
    ...
  ]
}}
"""

HUMAN_PROMPT = """[원래 질문]
{original_query}

[질문의 핵심 주제어]
{keyword}

[후보 상위 공종 목록]
{candidate_list}

위 [후보 상위 공종 목록]에서 [원래 질문]과 직접적으로 관련된 항목들만 필터링하여 JSON 형식으로 반환하라.
"""

class EvaluatorAgent:
    def __init__(self, llm):
        self.llm = llm
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("human", HUMAN_PROMPT)
        ])
        self.chain = self.prompt_template | self.llm | StrOutputParser()


    def validate_parent_processes(self, original_query: str, candidates: list, full_data: pd.DataFrame, keyword: str) -> list:
        """
        ProcessAgent가 찾은 상위 공종 후보 목록의 관련성을 검증하여 최종 목록을 반환합니다.
        """
        print("EvaluatorAgent: 상위 공종 후보 목록 검증 시작...")
        if not candidates:
            return []
        
        # 중복 항목 제거
        seen_names = set()
        unique_candidates = []
        for cand in candidates:
            name = cand.get("name")
            # 'name' 키가 존재하고, 그 값이 이전에 등장한 적 없는 경우에만 추가
            if name and name not in seen_names:
                seen_names.add(name)
                unique_candidates.append(cand)
        
        # 'record'와 '공종명'이 원본 데이터와 완벽히 일치하는 후보만 필터링
        fact_checked_candidates = []
        for cand in unique_candidates:
            record_id = cand.get("record")
            record_name = cand.get("name")
            
            if not record_id or not record_name:
                continue

            matched_df = full_data[full_data['record'] == record_id]
            if not matched_df.empty:
                real_name = matched_df.iloc[0]['공종명']
                if record_name in real_name or real_name in record_name:
                    fact_checked_candidates.append(cand)

        if not fact_checked_candidates:
            print("EvaluatorAgent: 사실 확인 후 남은 후보가 없습니다.")
            return []

        candidate_json_str = json.dumps(fact_checked_candidates, ensure_ascii=False, indent=4)
        
        response_json_str = self.chain.invoke({
            "original_query": original_query,
            "candidate_list": candidate_json_str,
            "keyword": keyword
        }).strip()

        print(f"EvaluatorAgent: LLM이 반환한 최종 검증 JSON: {response_json_str}")

        try:
            response_data = json.loads(response_json_str)
            validated_records = response_data.get("validated_records", [])
            print(f"EvaluatorAgent: 최종적으로 {len(validated_records)}개의 유효한 공종을 확정했습니다.")
            return validated_records
        except Exception as e:
            print(f"EvaluatorAgent: 의미 분석 중 오류 발생 - {e}. 사실 확인된 후보 목록을 반환합니다.")
            return fact_checked_candidates