import os
import json
import pandas as pd
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

###################### PROMPT ######################
SYSTEM_PROMPT = """
# [역할 정의]
너는 건설 프로젝트 분석을 위한 멀티 에이전트 시스템의 총괄 지휘관(Orchestrator)이다.
너의 유일한 임무는 사용자의 질문을 분석하여, 어떤 전문 에이전트가 처리해야 할 작업인지 판단하고, 해당 작업을 수행하는 데 필요한 정보들을 추출하여 명확한 JSON 형식으로 출력하는 것이다.

# [핵심 지시사항]
1.  **의도 분석**: 사용자의 질문을 분석하여 아래에 정의된 '작업 종류' 중 하나로 분류한다.
2.  **파일 특정**: 사용자가 언급하는 프로젝트나 문서 설명을 [사용 가능한 파일 목록]과 비교하여, 가장 일치하는 **정확한 전체 파일명**을 찾아낸다. 사용자가 파일을 특정하지 않으면 "None"을 반환한다.
3.  **정보 추출**: 작업을 수행하는 데 필요한 모든 파라미터를 추출한다.
4.  **용어 정제**: 만약 사용자가 '다리', '터널'처럼 대상만 언급하고 명확한 '공사'를 지칭하지 않으면, 이를 '다리 공사', '터널 공사'와 같이 하위 에이전트가 이해하기 쉬운 명확한 공종명으로 변환하여 `process_name`을 추출한다.
5.  **JSON 출력**: 분석 결과를 지정된 JSON 형식으로만 출력한다.
"""

HUMAN_PROMPT = """
[사용 가능한 파일 목록]
{available_files}

# [작업 종류]
1. `sub_process_extraction`: 특정 파일에서 특정 공사의 세부 공종 목록을 추출하는 작업.
    - 필수 파라미터: `file_name`, `process_name`
2. `general_cost_analysis`: 여러 파일을 기반으로 특정 공사의 일반적인 비용을 분석하는 작업.
    - 필수 파라미터: `process_name`

# [JSON 출력 형식]
{{
  "task": "작업 종류 이름",
  "parameters": {{
    "파라미터 이름": "추출한 값"
  }}
}}

# [요청 분석 예시]
[사용자 질문]: "북일-남일1 Q1 공사에서 교량공사의 세부 공종이 궁금해."
[출력]:
{{
  "task": "sub_process_extraction",
  "parameters": {{
    "file_name": "20161013327_북일-남일1_Q1_S6_E1412_조달청_조사내역서.txt",
    "process_name": "교량공사"
  }}
}}

[사용자 질문]: "북일-남일1 Q2에서 다리는 어떻게 돼가?"
[출력]:
{{
  "task": "sub_process_extraction",
  "parameters": {{
    "file_name": "20200827108_북일-남일2_Q2_S6_E1412_조달청_조사내역서.txt",
    "process_name": "다리 공사"
  }}
}}

[사용자 질문]: "보통 '배수 공사'를 하는 데 비용이 일반적으로 얼마나 들어?"
[출력]:
{{
  "task": "general_cost_analysis",
  "parameters": {{
    "process_name": "배수 공사"
  }}
}}

# [실제 분석 요청]
[사용자 질문]: "{user_query}"
[출력]:
"""
####################################################

class Orchestrator:
    """
    전체 에이전트 시스템을 지휘하는 오케스트레이터 클래스
    데이터를 로드하고, 사용자 쿼리에 따라 적절한 에이전트를 호출
    """
    def __init__(self, data_dir: str):
        # LLM 모델 설정
        self.llm = ChatOpenAI(
            model="openai/gpt-oss-120b",
            openai_api_key="EMPTY",
            openai_api_base="", # 사용할 LLM 모델
            max_tokens=128000,
            temperature=0, # 결과의 일관성을 위해 0으로 설정
        )
        
        self.data_dir = data_dir
        self.available_files = self._get_file_list(data_dir)
        
        # 데이터를 미리 로딩하지 않고, 필요할 때 로딩
        self.loaded_data_cache = {}
        
        # 프롬프트와 체인 초기화
        self.task_planning_chain = self._create_task_planning_chain()
        
    def _get_file_list(self, data_dir: str) -> list:
        try:
            return [f for f in os.listdir(data_dir) if f.endswith('.txt')]
        except FileNotFoundError:
            print(f"오류: 데이터 디렉토리를 찾을 수 없습니다 - {data_dir}")
            return []
        
    def _create_task_planning_chain(self):
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("human", HUMAN_PROMPT)
        ])
        return prompt_template | self.llm | StrOutputParser()
        
    def _load_data(self, file_name: str) -> pd.DataFrame:
        # 캐시에 파일이 이미 로드되어 있는지 확인
        if file_name in self.loaded_data_cache:
            return self.loaded_data_cache[file_name]
        
        file_path = os.path.join(self.data_dir, file_name)
        
        records = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line: continue
                    
                    # 데이터 파싱
                    parts = line.split(':', 1)
                    if len(parts) != 2: continue
                    record_id = parts[0].strip()
                    
                    rest_parts = parts[1].split(';')
                    
                    process_name = rest_parts[0].strip() if len(rest_parts) > 0 else "None"
                    spec = rest_parts[1].strip() if len(rest_parts) > 1 else "None"
                    total_cost = rest_parts[2].strip() if len(rest_parts) > 2 else "0"
                    
                    records.append({
                        'record': record_id, 
                        '공종명': process_name, 
                        'spec': spec,
                        'total_cost': total_cost
                    })
            
            df = pd.DataFrame(records)
            self.loaded_data_cache[file_name] = df # 로드한 데이터 캐싱
                 
            print(f"데이터 로딩 완료: 총 {len(records)}개의 공종을 불러왔습니다.")
            return df
        
        except FileNotFoundError:
            print(f"오류: 데이터 파일을 찾을 수 없습니다 - {file_path}")
            return None
        except Exception as e:
            print(f"데이터 로딩 중 오류 발생: {e}")
            return None
    
    # 사용자 쿼리를 받아 LLM 체인을 실행하고, 작업 계획(JSON)을 반환
    def plan_task(self, query: str, available_files: list) -> dict:
        print(f"Orchestrator: '{query}'에 대한 의도 분석 시작...")
        if not available_files:
            return {"task": "error", "parameters": {"message": "사용 가능한 파일이 없습니다."}}

        file_list_str = "\n".join(available_files)
        response_json_str = self.task_planning_chain.invoke({
            "available_files": file_list_str,
            "user_query": query
        })

        try:
            task_plan = json.loads(response_json_str)
            print(f"Orchestrator: '{task_plan.get('task')}' 작업으로 판단됨.")
            return task_plan
        except json.JSONDecodeError:
            print(f"LLM 응답 파싱 오류: {response_json_str}")
            return {"task": "error", "parameters": {"message": "LLM 응답 파싱 오류."}}