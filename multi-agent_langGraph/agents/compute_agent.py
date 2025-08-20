import os
import re
import pandas as pd
from tqdm import tqdm

from typing import TypedDict, List, Set, Optional
from langgraph.graph import StateGraph, END

from agents.process_agent import ProcessAgent
from agents.evaluator_agent import EvaluatorAgent

# state 데이터 관리
class ComputeState(TypedDict):
    original_query: str
    target_process_name: str
    available_files: List[str]
    data_dir: str
    current_file_index: int

    all_item_costs: List[int]
    files_with_data: Set[str]
    # current_file_index: int
    # total_cost: int
    # total_item_count: int
    
    current_data: Optional[pd.DataFrame]
    candidates: Optional[List[dict]]
    validated_processes: Optional[List[dict]]
    
    final_result: str

class ComputeAgent:
    def __init__(self, process_agent: ProcessAgent, evaluator_agent: EvaluatorAgent):
        self.process_agent = process_agent
        self.evaluator_agent = evaluator_agent
        self.graph = self._create_graph()

    def _load_and_parse_data(self, file_path: str) -> pd.DataFrame:
        records = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line: continue
                    
                    parts = line.split(':', 1)
                    if len(parts) != 2: continue
                    record_id = parts[0].strip()
                    
                    rest_parts = parts[1].split(';')
                    
                    process_name = rest_parts[0].strip() if len(rest_parts) > 0 else "None"
                    spec = rest_parts[1].strip() if len(rest_parts) > 1 else "None"
                    total_cost = rest_parts[2].strip()
                    
                    records.append({
                        'record': record_id, 
                        '공종명': process_name, 
                        'spec': spec,
                        'total_cost': total_cost
                    })
            return pd.DataFrame(records)
        except Exception as e:
            print(f" -> 파일 파싱 오류: {os.path.basename(file_path)} 처리 중 오류 발생 - {e}")
            return pd.DataFrame()

    def _start_computation(self, state: ComputeState) -> ComputeState:
        print("--- Compute Subgraph: 계산 시작 ---")
        return {
            "current_file_index": 0,
            "total_cost": 0,
            "total_item_count": 0,
            "files_with_data": set(),
        }

    def _load_data_node(self, state: ComputeState) -> ComputeState:
        idx = state['current_file_index']
        file_name = state['available_files'][idx]
        file_path = os.path.join(state['data_dir'], file_name)
        print(f"\n--- 루프 {idx + 1}: 파일 로딩 ({file_name}) ---")
        current_data = self._load_and_parse_data(file_path)
        return {"current_data": current_data}

    def _process_agent(self, state: ComputeState) -> ComputeState:
        print(" -> ProcessAgent 호출")
        if state['current_data'].empty:
            return {"candidates": []}
        
        candidates = self.process_agent.find_parent_processes(
            original_query=state['original_query'],
            keyword=state['target_process_name'],
            full_data=state['current_data']
        )
        return {"candidates": candidates}

    def _evaluator_agent(self, state: ComputeState) -> ComputeState:
        print(" -> EvaluatorAgent 호출")
        if not state['candidates']:
            return {"validated_processes": []}
            
        validated = self.evaluator_agent.validate_parent_processes(
            original_query=state['original_query'],
            candidates=state['candidates'],
            full_data=state['current_data'],
            keyword=state['target_process_name']
        )
        return {"validated_processes": validated}

    def _aggregate_results(self, state: ComputeState) -> ComputeState:
        all_item_costs = state.get('all_item_costs', [])
        files_with_data = state.get('files_with_data', set())
        current_file_index = state['current_file_index']
        
        costs_from_this_file = []
        
        if state['validated_processes']:
            file_name = state['available_files'][state['current_file_index']]
            state['files_with_data'].add(file_name)
            
            for parent_process in state['validated_processes']:
                parent_id = parent_process['record']
                parent_name = parent_process['name']

                # 하위 공종들의 비용 합계
                sub_processes_df = state['current_data'][
                    state['current_data']['record'].str.startswith(parent_id + '.')
                ]
                
                if not sub_processes_df.empty:
                    numeric_costs = pd.to_numeric(sub_processes_df['total_cost'])
                    total_cost_for_parent = numeric_costs.sum()
                else:
                    total_cost_for_parent = 0
                    
                # 예외처리) 하위 공종 비용 합계가 0이면 부모 공종의 비용으로 처리
                if total_cost_for_parent == 0:
                    parent_df = state['current_data'][state['current_data']['record'] == parent_id]
                    own_cost = pd.to_numeric(parent_df['total_cost']).iloc[0]
                    total_cost_for_parent = own_cost

                print(f"   - '{parent_name}' ({parent_id}): 하위 공종 비용 합계 = {total_cost_for_parent:,.0f}원")
                costs_from_this_file.append(total_cost_for_parent)

        next_index = current_file_index + 1
        updated_costs = all_item_costs + costs_from_this_file
        
        return {
            "all_item_costs": updated_costs,
            "files_with_data": files_with_data,
            "current_file_index": next_index 
        }


    def _check_if_done(self, state: ComputeState) -> str:
        if state['current_file_index'] >= len(state['available_files']):
            print("--- Compute Subgraph: 모든 파일 처리 완료 ---")
            return "finalize"
        else:
            return "continue"

    def _finalize_computation(self, state: ComputeState) -> ComputeState:
        print("--- Compute Subgraph: 최종 결과 생성 ---")
        all_costs = state.get('all_item_costs', [])
        
        if not all_costs:
            result = f"분석 결과, '{state['target_process_name']}'에 대한 유효한 비용 데이터를 찾을 수 없습니다."
        else:
            total_item_count = len(all_costs)
            total_cost = sum(all_costs)
            average_cost = total_cost / total_item_count
            
            result = (
                f"'{state['target_process_name']}'에 대한 비용 분석 결과:\n"
                f" - 총 {len(state['files_with_data'])}개 프로젝트(파일)에서 관련 데이터 발견\n"
                f" - 분석된 총 유효 공종(다리 등) 수: {total_item_count}개\n"
                f" - 평균 비용: {average_cost:,.0f}원"
            )
        return {"final_result": result}

    def _create_graph(self) -> StateGraph:
        """내부 로직을 수행하는 서브그래프를 생성하고 컴파일"""
        workflow = StateGraph(ComputeState)

        workflow.add_node("start", self._start_computation)
        workflow.add_node("load_data_node", self._load_data_node)
        workflow.add_node("process_agent", self._process_agent)
        workflow.add_node("evaluator_agent", self._evaluator_agent)
        workflow.add_node("aggregate_results", self._aggregate_results)
        workflow.add_node("finalize", self._finalize_computation)

        workflow.set_entry_point("start")
        workflow.add_edge("start", "load_data_node")
        workflow.add_edge("load_data_node", "process_agent")
        workflow.add_edge("process_agent", "evaluator_agent")
        workflow.add_edge("evaluator_agent", "aggregate_results")
        
        workflow.add_conditional_edges(
            "aggregate_results",
            self._check_if_done,
            {
                "continue": "load_data_node", 
                "finalize": "finalize"      
            }
        )
        workflow.add_edge("finalize", END)
        
        return workflow.compile()