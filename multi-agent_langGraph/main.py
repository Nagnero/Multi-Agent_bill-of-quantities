import os
import json
import pandas as pd
from pathlib import Path
from typing import TypedDict, List, Optional

from langgraph.graph import StateGraph, END

from agents.orchestrator import Orchestrator
from agents.process_agent import ProcessAgent
from agents.evaluator_agent import EvaluatorAgent
from agents.compute_agent import ComputeAgent

current_file_path = Path(__file__).resolve()
current_dir = current_file_path.parent

orchestrator = Orchestrator(data_dir=current_dir/"data")
process_agent = ProcessAgent(llm=orchestrator.llm)
evaluator_agent = EvaluatorAgent(llm=orchestrator.llm)
compute_agent = ComputeAgent(process_agent=process_agent, evaluator_agent=evaluator_agent)

########################## 그래프 상태 정의 ##########################
class AgentState(TypedDict):
    user_query: str
    available_files: List[str]
    task: Optional[str]
    parameters: Optional[dict]
    target_data: Optional[pd.DataFrame]
    parent_candidates: Optional[list]
    validated_parents: Optional[list]
    final_result: Optional[str]

########################## 노드(Node) 함수 정의 ##########################
def orchestrate_node(state: AgentState):
    print("--- 노드 실행: Orchestrator ---")
    task_plan = orchestrator.plan_task(state['user_query'], state['available_files'])
    return {"task": task_plan.get("task"), "parameters": task_plan.get("parameters")}

def load_data_node(state: AgentState):
    print("--- 노드 실행: 데이터 로딩 ---")
    file_name = state["parameters"].get("file_name")
    if not file_name:
        raise ValueError("데이터 로딩 노드: 파일명이 없습니다.")
    data = orchestrator._load_data(file_name) # Orchestrator의 로딩 기능 재사용
    return {"target_data": data}

def process_node(state: AgentState):
    print("--- 노드 실행: ProcessAgent ---")
    candidates = process_agent.find_parent_processes(
        original_query=state['user_query'],
        keyword=state['parameters'].get("process_name"),
        full_data=state['target_data']
    )
    return {"parent_candidates": candidates}

def evaluate_node(state: AgentState):
    print("--- 노드 실행: EvaluatorAgent ---")
    validated = evaluator_agent.validate_parent_processes(
        original_query=state['user_query'],
        candidates=state['parent_candidates'],
        full_data=state['target_data'],
        keyword=state['parameters'].get("process_name")
    )
    
    # ############################ 결과 출력 코드 ################################
    # # 출력 결과를 파일로 저장하는 코드
    # try:
    #     file_name = state["parameters"].get("file_name")
    #     if file_name:
    #         output_dir = current_dir/"교량/output4"
    #         os.makedirs(output_dir, exist_ok=True)
            
    #         results_to_save = {
    #             "parent_candidates": state.get('parent_candidates', []),
    #             "validated_parents": validated
    #         }

    #         base_filename = os.path.splitext(file_name)[0]
    #         output_filepath = os.path.join(output_dir, f"{base_filename}_results.json")
            
    #         with open(output_filepath, 'w', encoding='utf-8') as f:
    #             json.dump(results_to_save, f, ensure_ascii=False, indent=4)
    #         print(f"--- 중간 결과 저장 완료: {output_filepath} ---")
            
    # except Exception as e:
    #     print(f"--- 중간 결과 저장 중 오류 발생: {e} ---")
    # ##########################################################################
    return {"validated_parents": validated}

def finalize_sub_process_node(state: AgentState):
    validated_parents = state.get("validated_parents")
    if not validated_parents:
        return {"final_result": f"'{state['parameters'].get('process_name')}'과 직접 관련된 공종을 찾을 수 없습니다."}

    all_sub_processes = []
    target_data = state['target_data']
    # 검증된 각 상위 공종 및 그 하위 공종들을 모두 찾아서 리스트에 추가
    for parent in validated_parents:
        parent_id = parent.get("record")
        if parent_id:
            parent_df = target_data[target_data['record'] == parent_id]
            sub_df = target_data[target_data['record'].str.startswith(parent_id + '.')]
            if not parent_df.empty: all_sub_processes.append(parent_df)
            if not sub_df.empty: all_sub_processes.append(sub_df)
    
    if not all_sub_processes:
        return {"final_result": "관련된 세부 공종 내역이 없습니다."}
    
    final_df = pd.concat(all_sub_processes)
    return {"final_result": final_df.to_string()}

def compute_node(state: AgentState):
    print("--- 노드 실행: ComputeAgent Subgraph ---")
    subgraph_input = {
        "original_query": state['user_query'],
        "target_process_name": state['parameters'].get('process_name'),
        "available_files": state['available_files'],
        "data_dir": orchestrator.data_dir,
    }
    subgraph_final_state = compute_agent.graph.invoke(subgraph_input)
    result_string = subgraph_final_state.get("final_result", "서브그래프에서 결과를 가져오는 데 실패했습니다.")
    return {"final_result": result_string}


########################### 그래프 흐름 정의 및 컴파일 ##########################
def route_task(state: AgentState):
    # Orchestrator의 결과에 따라 다음 노드 결정
    if state['task'] == "sub_process_extraction":
        return "load_data_node"
    elif state['task'] == "general_cost_analysis":
        return "compute_node"
    else:
        return END

workflow = StateGraph(AgentState)

# 함수들을 그래프의 노드로 추가
workflow.add_node("orchestrator", orchestrate_node)
workflow.add_node("load_data_node", load_data_node)
workflow.add_node("process_agent", process_node)
workflow.add_node("evaluator_agent", evaluate_node)
workflow.add_node("finalize_sub_process", finalize_sub_process_node)
workflow.add_node("compute_node", compute_node)

# 그래프의 시작점을 'orchestrator' 노드로 설정
workflow.set_entry_point("orchestrator")

# 'orchestrator' 노드 다음에 어떤 노드로 갈지 'route_task' 함수를 통해 조건부로 결정
workflow.add_conditional_edges("orchestrator", route_task, {
    "load_data_node": "load_data_node",
    "compute_node": "compute_node",
    END: END
})

############################ 경로 정의 ############################
# 세부 공종 추출 경로
workflow.add_edge('load_data_node', 'process_agent')
workflow.add_edge('process_agent', 'evaluator_agent')
workflow.add_edge('evaluator_agent', 'finalize_sub_process')
workflow.add_edge('finalize_sub_process', END)

# 일반 비용 분석 경로
workflow.add_edge('compute_node', END)

# 정의된 워크플로우를 실행 가능한 객체로 컴파일
app = workflow.compile()


############################ 메인 실행 블록 ############################
if __name__ == "__main__":
    ##################### 그래프 구조 이미지로 출력 #####################
    # from IPython.display import display, Image
    # display(Image(app.get_graph().draw_mermaid_png(output_file_path=current_dir/"maingraph.png")))
    # display(Image(compute_agent.graph.get_graph().draw_mermaid_png(output_file_path=current_dir/"compute_subgraph.png")))
    ###############################################################
    
    while True:
        print("\n어떤 공종에 대해 알아보고 싶으신가요? (예: 북일-남일1 Q1 공사에서 다리공사, 일반적인 토공 비용 등)")
        query = input("입력: ")
        if query.lower() in ["exit", "quit"]:
            break

        initial_state = {
            "user_query": query,
            "available_files": orchestrator.available_files
        }
        config = {"recursion_limit": 200}        
        final_state = app.invoke(initial_state, config=config)

        print("\n--- 최종 결과 ---")
        result_message = final_state.get("final_result", "결과를 가져오는 데 실패했습니다.")
        print(result_message)