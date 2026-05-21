# 국제전기 프로세스 마이닝 분석 및 데이터 추출 시스템
import pandas as pd
import pm4py
from pm4py.objects.conversion.log import converter as log_converter
import os
import time

# [시작점] 실행 시간 기록
script_start_time = time.time()
output_dir = os.getcwd()

print("=========================================================")
print(" PM4Py 프로세스 마이닝 분석 및 데이터 추출 시스템")
print(f" 시작 시간: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(script_start_time))}")
print("=========================================================\n")

# =============================================================================
# 1. 데이터 로드 및 전처리
# =============================================================================
print("[STEP 1] 데이터 로드 및 전처리 중...")

file_path = 'total_event_log.csv'

try:
    df = pd.read_csv(file_path)
except FileNotFoundError:
    print(f"\n[오류] '{file_path}' 파일을 찾을 수 없습니다.")
    exit()

# 시간 형식 오류 수정 함수
def fix_24_hour_time(ts):
    ts_str = str(ts)
    if "24:00:00" in ts_str:
        return pd.to_datetime(ts_str.replace("24:00:00", "00:00:00")) + pd.Timedelta(days=1)
    else:
        return pd.to_datetime(ts_str, errors='coerce')

df['Timestamp'] = df['Timestamp'].apply(fix_24_hour_time)
df = df.dropna(subset=['Timestamp'])

# PM4Py 표준 컬럼 매핑
df = df.rename(columns={
    'Case ID': 'case:concept:name',
    'Activity': 'concept:name',
    'Timestamp': 'time:timestamp',
    'Lifecycle': 'lifecycle:transition',
    'Resource': 'org:resource'
})
df['lifecycle:transition'] = df['lifecycle:transition'].replace({'end': 'complete'})
df = df.sort_values(by=['case:concept:name', 'time:timestamp'])

# 이벤트 로그 객체 변환
event_log = log_converter.apply(df)
log_complete = pm4py.filter_event_attribute_values(
    event_log, attribute_key='lifecycle:transition', values=['complete'], level='event', retain=True
)

# =============================================================================
# 2. 모델 도출 (PNG 시각화)
# =============================================================================
print("[STEP 2] 표준 프로세스 모델(Petri Net) 도출 및 저장...")
net, initial_marking, final_marking = pm4py.discover_petri_net_inductive(log_complete)
pm4py.save_vis_petri_net(net, initial_marking, final_marking, os.path.join(output_dir, "2026_ver1_표준공정모델.png"))

# =============================================================================
# 3. 병목 구간 분석 (PNG 시각화 및 순수 대기 시간 CSV 저장)
# =============================================================================
print("[STEP 3] 병목 구간 히트맵 저장 및 순수 대기 시간 데이터 추출...")

# 히트맵 이미지 저장
dfg, start_activities, end_activities = pm4py.discover_performance_dfg(log_complete)
pm4py.save_vis_performance_dfg(dfg, start_activities, end_activities, os.path.join(output_dir, "병목분석_히트맵.png"))

# PNG에 표시된 엣지 값(mean)을 그대로 추출
dfg_rows = []
for (frm, to), perf_val in dfg.items():
    mean_sec = perf_val.get('mean', 0) if isinstance(perf_val, dict) else (float(perf_val) if perf_val else 0)
    dfg_rows.append({'From': frm, 'To': to, 'DFG_Mean_Sec': round(mean_sec, 2)})

path_analysis = pd.DataFrame(dfg_rows).sort_values(['From', 'To']).reset_index(drop=True)

# 병목 여부: PNG 색상 기준과 동일하게 정규화 후 상위 50% 이상
_min_dfg = path_analysis['DFG_Mean_Sec'].min()
_max_dfg = path_analysis['DFG_Mean_Sec'].max()
path_analysis['Is_Bottleneck'] = (path_analysis['DFG_Mean_Sec'] - _min_dfg) / (_max_dfg - _min_dfg + 1e-6) > 0.5

# [2단계] 순수 대기 시간(Waiting Time) 계산: (현재 Start - 이전 Complete)
temp_df = df.copy().sort_values(by=['case:concept:name', 'time:timestamp'])
temp_df['prev_activity'] = temp_df.groupby('case:concept:name')['concept:name'].shift(1)
temp_df['prev_timestamp'] = temp_df.groupby('case:concept:name')['time:timestamp'].shift(1)
temp_df['prev_lifecycle'] = temp_df.groupby('case:concept:name')['lifecycle:transition'].shift(1)

waiting_mask = (temp_df['lifecycle:transition'] == 'start') & (temp_df['prev_lifecycle'] == 'complete')
waiting_df = temp_df[waiting_mask].copy()
waiting_df['wait_hours'] = (waiting_df['time:timestamp'] - waiting_df['prev_timestamp']).dt.total_seconds() / 3600

wait_agg = waiting_df.groupby(['prev_activity', 'concept:name']).agg(
    Avg_Wait_Hours=('wait_hours', 'mean'),
    Max_Wait_Hours=('wait_hours', 'max'),
).reset_index().rename(columns={'prev_activity': 'From', 'concept:name': 'To'})

# DFG 전체 엣지에 순수 대기시간 병합 (없으면 NaN)
path_analysis = path_analysis.merge(wait_agg, on=['From', 'To'], how='left')

# 컬럼 순서 정리
path_analysis = path_analysis[['From', 'To', 'DFG_Mean_Sec', 'Is_Bottleneck', 'Avg_Wait_Hours', 'Max_Wait_Hours']]

path_analysis.to_csv(os.path.join(output_dir, "pure_waiting_time_analysis.csv"), index=False, encoding='utf-8-sig')

# =============================================================================
# 4. 모듈별 순수 작업 효율 분석 (CSV 저장)
# =============================================================================
print("[STEP 4] 모듈별 순수 작업 시간 분석 및 저장...")
df_start = df[df['lifecycle:transition'] == 'start'][['case:concept:name', 'concept:name', 'time:timestamp']]
df_end = df[df['lifecycle:transition'] == 'complete'][['case:concept:name', 'concept:name', 'time:timestamp']]
df_service = pd.merge(df_start, df_end, on=['case:concept:name', 'concept:name'], suffixes=('_start', '_end'))

df_service['duration_sec'] = (df_service['time:timestamp_end'] - df_service['time:timestamp_start']).dt.total_seconds()
module_stats = df_service.groupby('concept:name')['duration_sec'].agg(['mean', 'max', 'count']).reset_index()
module_stats.to_csv(os.path.join(output_dir, "module_efficiency_stats.csv"), index=False, encoding='utf-8-sig')

print("\n=========================================================")
print(" [완료] 모든 분석이 완료되었습니다.")
print(f" ⏱ 총 소요 시간: {time.time() - script_start_time:.2f} 초")
print(" 생성된 파일: 2026_ver1_표준공정모델.png, 병목분석_히트맵.png")
print("             pure_waiting_time_analysis.csv, module_efficiency_stats.csv")
print("=========================================================")