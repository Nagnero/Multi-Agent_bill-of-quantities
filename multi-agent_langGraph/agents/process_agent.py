import json
import pandas as pd
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

###################### PROMPT ######################
SYSTEM_PROMPT = """
# [역할 정의]
너는 주어진 [전체 공종 리스트]를 기반으로 사용자의 질문에 답변하는, 매우 정직하고 정확한 건설 공정표 분석 전문가다.

# [매우 중요한 규칙 (Grounding Rule)]
- 답변은 반드시 [전체 공종 리스트]에 **실제로 존재하는 데이터**에만 근거해야 한다.
- 절대 목록에 없는 'record'나 '공종명'을 추측하거나 만들어내서는 안 된다.
- 'record'를 기준으로 하위 공종에 해당하는 항목은 생략한다. 단, 'OO교', 'XX터널'과 같이 구체적인 고유 명칭을 가진 항목은 최상위 공종으로 간주하여 포함한다.

# [출력 규칙]
- 사용자의 질문과 가장 일치하는 공종들을 찾아서, 아래 JSON 형식으로 반환해야 한다.
- 'matching_records' 키의 값은 'record'와 'name'을 포함하는 객체들의 리스트여야 한다.
- 다른 설명이나 응답은 절대 추가하지 말 것

# [JSON 출력 형식]
{{
  "matching_records": [
    {{
      "record": "실제 존재하는 record 값 1",
      "name": "record 값 1에 해당하는 실제 공종명"
    }},
    {{
      "record": "실제 존재하는 record 값 2",
      "name": "record 값 2에 해당하는 실제 공종명"
    }}
  ]
}}
"""

HUMAN_PROMPT = """[사용자 원본 질문]
{original_query}

[분석 대상 키워드]
'{keyword}'

[전체 공종 리스트]
{process_list}

위 [사용자 원본 질문]의 전체적인 의도를 파악하여, [분석 대상 키워드]와 가장 관련 있는 공종들을 [전체 공종 리스트]에서 찾아 규칙에 맞는 JSON 형식으로 반환하라.
"""
####################################################

class ProcessAgent:
    def __init__(self, llm):
        self.llm = llm
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("human", HUMAN_PROMPT)
        ])        
        self.chain = self.prompt_template | self.llm | StrOutputParser()
    
    def find_parent_processes(self, original_query: str, keyword: str, full_data) -> list:
        print("ProcessAgent: 상위 공종 후보 검색 시작...")
        
        depth = full_data['record'].str.split('.').str.len()
        filtered_data = full_data[depth.isin([1, 2, 3, 4])].copy()
        process_list_str = filtered_data[['record', '공종명']].to_string(index=False)

        response_json_str = self.chain.invoke({
            "original_query": original_query, # 원본 질문 전달
            "keyword": keyword,
            "process_list": process_list_str
        }).strip()

        print(f"ProcessAgent: LLM이 반환한 후보 JSON: '{response_json_str}'")
        
        try:
            response_data = json.loads(response_json_str)
            parent_record_objects = response_data.get("matching_records", [])
            print(f"ProcessAgent: {len(parent_record_objects)}개의 상위 공종 후보를 찾았습니다.")
            return parent_record_objects
        except json.JSONDecodeError:
            print("ProcessAgent: LLM의 응답이 유효한 JSON 형식이 아닙니다.")
            return []
        