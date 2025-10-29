import argparse
from config import update_config
import torch
from dataset import get_data
from util import str2bool, make_folder
from fastprogress import progress_bar
from eval import fsm_evaluation, fsm_visualization
from llm_set.basic_func import prompt_selection, llm_selection, print_prompt, fsm_inference
import json
import warnings
warnings.filterwarnings('ignore')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--dataset', default='synthetic', type=str)
    parser.add_argument('--class_name', default='point', type=str, help='flat-trend, freq, point, etc')
    parser.add_argument('--set_image', default=None, type=str2bool, nargs='?', const=True)
    parser.add_argument('--anomaly_detect', default=True, type=str2bool, nargs='?', const=True)
    parser.add_argument('--visualization', default=False, type=str2bool, nargs='?', const=True)
    parser.add_argument('--prompt_ver', default=2, type=int, help='0: default, 1: CoT, 2: Segment')
    parser.add_argument('--llm_model', default=0, type=int, help='0: Qwen')
    parser.add_argument('--top_k', default=3, type=int, help='number of periods')
    parser.add_argument('--group_length', default=6, type=int, help='length of semgnt group')
    parser.add_argument('--interval', default=100, type=int, help='interval of periods')
    parser.add_argument('--grouping', default=True, type=str2bool, nargs='?', const=True, help='inference per segment group')

    args = parser.parse_args()
    cfg = update_config(args)
    cfg.print_cfg()
    make_folder(cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    print(device)

    '''
    ==============================
    Dataset Setting
    ==============================
    '''
    value_list, label_list, data_length = get_data(cfg)

    '''
    ==============================
    Anomaly Detection
    ==============================
    '''
    if cfg.grouping:
        predict_file_name = f'experiment/{cfg.dataset_name}/{cfg.llm_model}/fsm_grp{str(cfg.group_length)}_{cfg.class_name}.json'
    else:
        predict_file_name = f'experiment/{cfg.dataset_name}/{cfg.llm_model}/fsm_seg_{cfg.class_name}.json'

    if cfg.anomaly_detect:
        print_check = True
        prompt = prompt_selection(cfg)
        model_set = llm_selection(cfg, device)
        dict_arr = []

        print(predict_file_name)
        with open(predict_file_name, 'w') as file:
            for num in progress_bar(range(data_length), total=data_length):
                print_check = print_prompt(print_check, prompt)
                test_value = value_list[num]
                output_dict = fsm_inference(cfg, model_set, test_value, device)
                dict_arr.append(output_dict)
                print(num, output_dict['interval'])
            json.dump(dict_arr, file, indent=4)

    # evaluation
    best_a = fsm_evaluation(cfg, label_list, predict_file_name, data_length)

    # visualization
    if cfg.visualization:
        fsm_visualization(cfg, value_list, label_list, predict_file_name, data_length, best_a)

if __name__=="__main__":
    main()
    
