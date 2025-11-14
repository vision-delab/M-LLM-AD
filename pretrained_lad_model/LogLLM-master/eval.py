import os
import re
import argparse
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from model import LogLLM, LogLLM_not_ft
from customDataset import CustomDataset, CustomCollator
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import matplotlib.pyplot as plt
from openpyxl.drawing.image import Image # Excel에 이미지 삽입하는 용도

ANSI_ESCAPE_PATTERN = re.compile(r'\x1B\[[0-?]*[ -/]*[@-~]')

parser = argparse.ArgumentParser(description="LogLLM Eval Code")
parser.add_argument('--dataset_name', type=str, default='BGL', help = 'Dataset Name(HDFS_v1, BGL, Thunderbird)')
args = parser.parse_args()

max_content_len = 100
max_seq_len = 128
batch_size = 32

Bert_path = "/scratch/jyy1551/LAD/LogLLM/Models/bert-base-uncased"
Llama_path = "/scratch/jyy1551/LAD/LogLLM/Models/Meta-Llama-3-8B"

ROOT_DIR = Path(__file__).parent
# ft_path = os.path.join(ROOT_DIR, r"ft_model_{}".format(dataset_name))

device = torch.device("cuda:0")


def evalModel(model, dataloader):
    preds = []
    labels = []
    results_list = []

    output_paths = {} # 반환 파일 저장 dictionary

    model.eval()
    with torch.no_grad():
        for bathc_i in tqdm(dataloader):
            inputs = bathc_i['inputs']
            seq_positions = bathc_i['seq_positions']

            inputs = inputs.to(device)
            seq_positions = seq_positions

            outputs_ids = model(inputs,seq_positions)
            outputs_raw_text = model.Llama_tokenizer.batch_decode(outputs_ids)

            original_logs = bathc_i['Content']

            # print(outputs)
            # (추가) abnormal raw data 추출하는 코드 추가
            for i in range(len(outputs_raw_text)):
                raw_prediction = outputs_raw_text[i] # 모델의 원본 예측 텍스트
                original_log_content = original_logs[i] # 원본 로그 템플릿 묶음
                match = re.search(r'normal|anomalous', raw_prediction, re.IGNORECASE)
                clean_prediction = ""
                if match:
                    clean_prediction = match.group().lower()
                    preds.append(clean_prediction)
                else:
                    clean_prediction = "ERROR_OUTPUT"
                    print(f'error :{raw_prediction}')
                    preds.append(clean_prediction)
                
                clean_log_content = ANSI_ESCAPE_PATTERN.sub('', original_log_content)
                results_list.append({
                    'Original_Log_Content': clean_log_content,
                    'Model_Clean_Prediction': clean_prediction
                })
    df_results=pd.DataFrame(results_list)

    # 원본 로그/예측 결과 csv 저장 
    # highlighting anomalous data
    csv_fileName = f"/home/jyy1551/LAD/pretrained_lad_model/results/logllm_{dataset_name}_detailed_logs.csv"
    df_results.to_csv(csv_filename, index=False)
    output_paths['detailed_logs_csv'] = csv_filename
    print(f"Detailed logs saved to {csv_filename}")



    def highlight_anomalous_row(row):
        if row['Model_Clean_Prediction'] == 'anomalous':
            return ['background-color: #FFCDD2']*len(row)
        else:
            return ['']*len(row)
    
    styled_df = df_results.style.apply(highlight_anomalous_row, axis=1)
    excel_filename=f"/home/jyy1551/LAD/pretrained_lad_model/results/logllm_{dataset_name}_block_30_predictions_styled_2.xlsx"
    styled_df.to_excel(excel_filename, index=False, engine='openpyxl')
    output_paths['styled_logs_excel'] = excel_filename
    print(f"Styled results saved to {excel_filename}")


    preds_copy = np.array(preds)
    preds = np.zeros_like(preds_copy,dtype=int)
    preds[preds_copy == 'anomalous'] = 1
    preds[preds_copy != 'anomalous'] = 0
    
    pred_num_anomalous = (preds == 1).sum()
    pred_num_normal =  (preds == 0).sum()

    print(f'Number of detected anomalous seqs: {pred_num_anomalous}; number of detected normal seqs: {pred_num_normal}')

    # 1. 총 분석 블록 수 (preds는 현재 numpy 정수 배열)
    total_blocks = len(preds)
    
    # 2. 탐지된 비정상 블록 수 (이미 계산된 'pred_num_anomalous' 값 사용)
    detected_anomalous_blocks = pred_num_anomalous
    
    # 3. 총 분석 로그 라인 수 (루프 초반에 저장한 results_list 사용)
    total_log_lines = 0
    for item in results_list: # 'results_list' (list of dicts) 사용
        # ' ;-; ' 구분자를 기준으로 로그 라인 수 계산
        total_log_lines += item['Original_Log_Content'].count(' ;-; ') + 1
        
    # 4. 비정상 블록 비율
    anomalous_ratio_percent = 0.0
    if total_blocks > 0:
        anomalous_ratio_percent = (detected_anomalous_blocks / total_blocks) * 100
    
    print("\n--- Generating Integrated Excel Report ---")

    # === 1. 시각화 데이터 준비 ===
    
    # KPI 데이터 (파이 차트용)
    total_normal_blocks = total_blocks - detected_anomalous_blocks
    pie_sizes = [total_normal_blocks, detected_anomalous_blocks]
    pie_labels = ['Normal Blocks', 'Anomalous Blocks']
    pie_colors = ['#8FDEBD', '#FF9999'] # (정상:초록, 비정상:빨강)

    # 핫스팟 데이터 (롤링 평균 플롯용)
    # preds는 0과 1로 구성된 NumPy 배열
    df_preds = pd.DataFrame({'Anomaly': preds})
    
    # 100개 블록 단위로 비정상 비율을 계산 (창 크기는 조절 가능)
    window_size = 100 
    df_preds['AnomalyRate'] = df_preds['Anomaly'].rolling(
        window=window_size, min_periods=1
    ).mean()

    # === 2. 그래프 생성 및 이미지 파일로 저장 ===
    
    # (A) KPI 파이 차트
    fig_pie, ax_pie = plt.subplots()
    ax_pie.pie(pie_sizes, 
            labels=pie_labels, 
            colors=pie_colors, 
            autopct='%1.1f%%', 
            startangle=90)
    ax_pie.axis('equal')
    ax_pie.set_title('Anomaly Detection Summary')
    pie_chart_filename = f"/home/jyy1551/LAD/pretrained_lad_model/results/logllm_{dataset_name}_summary_pie.png"
    fig_pie.savefig(pie_chart_filename)
    output_paths['summary_pie_chart'] = pie_chart_filename # (수정) 경로 저장
    plt.close(fig_pie)
    print(f"Pie chart saved to {pie_chart_filename}")

    # (B) 핫스팟 롤링 플롯
    fig_hotspot, ax_hotspot = plt.subplots(figsize=(15, 6))
    df_preds['AnomalyRate'].plot(ax=ax_hotspot, 
                                color='red', 
                                lw=2,
                                title=f'Anomaly Hotspots (Rolling Average over {window_size} blocks)')
    ax_hotspot.set_xlabel('Block Index (Sequence)')
    ax_hotspot.set_ylabel('Anomaly Rate in Window')
    ax_hotspot.grid(True)
    hotspot_plot_filename = f"/home/jyy1551/LAD/pretrained_lad_model/results/logllm_{dataset_name}_hotspot_plot.png"
    fig_hotspot.savefig(hotspot_plot_filename)
    output_paths['hotspot_plot'] = hotspot_plot_filename
    plt.close(fig_hotspot)
    print(f"Hotspot plot saved to {hotspot_plot_filename}")

    hotspot_data_filename = f"logllm_{dataset_name}_hotspot_data.csv"
    df_preds.to_csv(hotspot_data_filename, index=False)
    output_paths['hotspot_data_csv'] = hotspot_data_filename
    print(f"Hotspot plot data saved to {hotspot_data_filename}")

    # === 3. ExcelWriter로 다중 시트 리포트 생성 ===
    
    report_excel_filename = f"/home/jyy1551/LAD/pretrained_lad_model/results/logllm_{dataset_name}_visual_report.xlsx"
    output_paths['integrated_report_excel'] = report_excel_filename
    with pd.ExcelWriter(report_excel_filename, engine='openpyxl') as writer:
        
        # --- 시트 1: Dashboard ---
        # KPI 요약 데이터를 DataFrame으로 생성
        kpi_data = {
            'Metric': [
                "Total Log Lines Analyzed",
                "Total Log Blocks Analyzed",
                "Detected Anomalous Blocks",
                "Anomalous Block Ratio",
            ],
            'Value': [
                f"{total_log_lines:,}",
                f"{total_blocks:,}",
                f"{detected_anomalous_blocks:,}",
                f"{anomalous_ratio_percent:.2f} %",
            ]
        }
        df_kpi = pd.DataFrame(kpi_data)
        kpi_csv_filename = f"/home/jyy1551/LAD/pretrained_lad_model/results/logllm_{dataset_name}_kpi_summary.csv"
        df_kpi.to_csv(kpi_csv_filename, index=False)
        output_paths['kpi_summary_csv'] = kpi_csv_filename
        print(f"KPI summary data saved to {kpi_csv_filename}")
        
        # 'Dashboard' 시트에 KPI 데이터 작성
        df_kpi.to_excel(writer, sheet_name='Dashboard', startrow=1, startcol=1, index=False)
        
        # 'Dashboard' 시트 가져오기
        sheet_dashboard = writer.sheets['Dashboard']
        
        # 시트에 파이 차트 이미지 삽입
        img_pie = Image(pie_chart_filename)
        img_pie.anchor = 'E2' # (E열 2행 근처에 이미지 위치)
        sheet_dashboard.add_image(img_pie)
        
        # 시트에 핫스팟 플롯 이미지 삽입
        img_hotspot = Image(hotspot_plot_filename)
        img_hotspot.anchor = 'B15' # (B열 15행 근처에 이미지 위치)
        sheet_dashboard.add_image(img_hotspot)
        
        # (선택) 시트 컬럼 너비 자동 조절
        sheet_dashboard.column_dimensions['B'].width = 30
        sheet_dashboard.column_dimensions['C'].width = 20

        # --- 시트 2: Detailed Logs (하이라이트 적용) ---
        # 이전에 만든 'styled_df' 객체를 사용합니다.
        styled_df.to_excel(writer, sheet_name='Detailed_Logs', index=False)
        
        # --- (선택) 시트 3: 핫스팟 플롯 원본 데이터 ---
        df_preds.to_excel(writer, sheet_name='Hotspot_Plot_Data', index=False)

    print(f"\nIntegration report successfully saved to {report_excel_filename}")
    print(f"output_paths: {output_paths}")
    return output_paths


if __name__ == '__main__':

    # 2. [수정] argparse로 받은 값으로 전역 변수를 *먼저* 덮어씁니다.
    dataset_name = args.dataset_name
    data_path = r'/scratch/jyy1551/LAD/LogLLM/Datasets/pred/{dataset_name}/pred.csv'.format(dataset_name=dataset_name)
    ft_path = os.path.join(ROOT_DIR, r"ft_model_{}".format(dataset_name))
    
    # 3. [수정] 이제 정의된 변수들을 사용합니다.
    print(
    f'--- Running Configuration ---\n'
    f'dataset_name: {dataset_name}\n'
    f'data_path: {data_path}\n'
    f'ft_path: {ft_path}\n'
    f'batch_size: {batch_size}\n'
    f'max_content_len: {max_content_len}\n'
    f'max_seq_len: {max_seq_len}\n'
    f'device: {device}\n'
    f'-----------------------------')
    
    dataset = CustomDataset(data_path) # 4. 올바른 data_path로 로드
    
    # (ft/not_ft)
    model = LogLLM(Bert_path, Llama_path, ft_path=ft_path, is_train_mode=False, device=device,
                   max_content_len=max_content_len, max_seq_len=max_seq_len)

    tokenizer = model.Bert_tokenizer
    collator = CustomCollator(tokenizer, max_seq_len=max_seq_len, max_content_len=max_content_len)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collator,
        num_workers=4,
        shuffle=False,
        drop_last=False
    )

    evalModel(model, dataloader)