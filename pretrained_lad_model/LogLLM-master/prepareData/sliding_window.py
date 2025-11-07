import os.path
import numpy as np
import pandas as pd
from helper import sliding_window, fixedSize_window, structure_log
import sys
#### for Thunderbird, Liberty, BGL

try:
    dataset_name = sys.argv[1]
except IndexError:
    print("dataset 이름을 인자로 전달해야 함.")
    sys.exit(1)

data_dir = f'/scratch/jyy1551/LAD/LogLLM/Datasets/pred/{dataset_name}'
log_name = f"{dataset_name}.log"

start_line = 0
end_line = None

# # Liberty
# start_line = 40000000
# end_line = 45000000

# # thunderbird
# start_line = 160000000
# end_line = 170000000

output_dir = data_dir



if __name__ == '__main__':
    # group_type = 'time_sliding'

    window_size = 1 # block의 크기(1개의 로그 템플릿을 한 묶음으로 만듦.)
    step_size = 1 # 100개짜리 block을 만든 뒤, 100칸을 건너뛰어 다음 블록을 만듦.

    # (수정 필요) 데이터셋 자체에서 레이블 제거 후 format 부분도 수정 필요. -> <Label> 삭제 완.
    if 'thunderbird' in log_name.lower() or 'spirit' in log_name.lower() or 'liberty' in log_name.lower():
        log_format = '<Id> <Date> <Admin> <Month> <Day> <Time> <AdminAddr> <Content>'   #thunderbird  , spirit, liberty
    elif 'bgl' in log_name.lower():
        log_format = '<Id> <Date> <Code1> <Time> <Code2> <Component1> <Component2> <Level> <Content>'  #bgl
    else:
        raise Exception('missing valid log format') 
    print(f'Auto log_format: {log_format}')

    structure_log(data_dir, output_dir, log_name, log_format, start_line = start_line, end_line = end_line)

    print(f'window_size: {window_size}; step_size: {step_size}')

    # train_ratio = 0.8

    df = pd.read_csv(os.path.join(output_dir,f'{log_name}_structured.csv'))

    print(len(df))

    # data preprocess
    print('Start grouping.')

    # grouping with fixedSize window
    session_pred_df = fixedSize_window(
        df[['Content']],
        window_size=window_size, step_size=step_size
    )


    col = ['Content']
    spliter=' ;-; '

    session_pred_df = session_pred_df[col]
    session_pred_df['session_length'] = session_pred_df["Content"].apply(len)
    session_pred_df["Content"] = session_pred_df["Content"].apply(lambda x: spliter.join(x))

    mean_session_pred_len = session_pred_df['session_length'].mean()
    max_session_pred_len = session_pred_df['session_length'].max()

    session_pred_df.to_csv(os.path.join(output_dir, 'pred.csv'),index=False)

    print('Pred dataset info:')
    print(f"max session length: {max_session_pred_len}; mean session length: {mean_session_pred_len}\n")
   
    with open(os.path.join(output_dir, 'pred_info.txt'), 'w') as file:
        file.write(f"max session length: {max_session_pred_len}; mean session length: {mean_session_pred_len}\n")