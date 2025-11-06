import os
import re
from collections import defaultdict

import pandas as pd
from tqdm import tqdm

from helper import structure_log

data_dir = '/scratch/jyy1551/LAD/LogLLM/Datasets/pred/HDFS_v1'
log_name = "HDFS_v1.log"

output_dir = data_dir


if __name__ == '__main__':
    log_format = '<Date> <Time> <Pid> <Level> <Component>: <Content>'  # HDFS log format
    structure_log(data_dir, output_dir, log_name, log_format)

    # spliter = ' ;-; '
    # train_ratio = 0.8

    log_structured_file = os.path.join(output_dir, log_name + "_structured.csv")

    df = pd.read_csv(log_structured_file, engine='c',
            na_filter=False, memory_map=True, dtype={'Date':object, "Time": object})

    print(f'number of messages in {log_structured_file} is {len(df)}')
    # df = df[:100000]

    # data_dict_content = defaultdict(list) #preserve insertion order of items
    # for idx, row in tqdm(df.iterrows(),total=len(df)):
    #     blkId_list = re.findall(r'(blk_-?\d+)', row['Content'])
    #     blkId_set = set(blkId_list)
    #     for blk_Id in blkId_set:
    #         data_dict_content[blk_Id].append(row["Content"])

    # data_df = pd.DataFrame(list(data_dict_content.items()), columns=['BlockId', 'Content'])

    # train_len = int(train_ratio * len(data_df))

    # data_df = data_df.sample(frac=1).reset_index(drop=True)  ##shuffle

    # session_pred_df = data_df
    # session_pred_df = session_pred_df.reset_index(drop=True)

    # session_pred_df['session_length'] = session_pred_df["Content"].apply(len)
    # session_pred_df["Content"] = session_pred_df["Content"].apply(lambda x: spliter.join(x))

    session_pred_df = df[['Content']].copy()
    session_pred_df['session_length'] = 1

    mean_session_pred_len = session_pred_df['session_length'].mean()
    max_session_pred_len = session_pred_df['session_length'].max()
    session_pred_df.to_csv(os.path.join(output_dir, 'pred.csv'), index=False)

   
    print('Pred dataset info:')
    print(f"max session length: {max_session_pred_len}; mean session length: {mean_session_pred_len}\n")

    with open(os.path.join(output_dir, 'pred_info.txt'), 'w') as file:
        file.write(f"max session length: {max_session_pred_len}; mean session length: {mean_session_pred_len}\n")
        file.write(f"number of total sessions: {len(session_pred_df)}\n")

