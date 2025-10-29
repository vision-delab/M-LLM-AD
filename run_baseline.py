import argparse
from config import update_config
import torch
from dataset import get_data_w_image, make_image_data
from util import str2bool, make_folder
from fastprogress import progress_bar
from eval import evaluation
from llm_set.basic_func import prompt_selection, llm_selection, print_prompt, inference
import json
import warnings
warnings.filterwarnings('ignore')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--dataset', default='synthetic', type=str)
    parser.add_argument('--class_name', default='point', type=str, help='flat-trend, freq, point, etc')
    parser.add_argument('--set_image', default=False, type=str2bool, nargs='?', const=True)
    parser.add_argument('--anomaly_detect', default=True, type=str2bool, nargs='?', const=True)
    parser.add_argument('--visualization', default=False, type=str2bool, nargs='?', const=True)
    parser.add_argument('--prompt_ver', default=0, type=int, help='0: default, 1: CoT')
    parser.add_argument('--llm_model', default=0, type=int, help='0: Qwen, 1: MiniCPM')
    parser.add_argument('--top_k', default=-1, type=int, help='x')
    parser.add_argument('--group_length', default=-1, type=int, help='x')
    parser.add_argument('--interval', default=-1, type=int, help='x')
    parser.add_argument('--grouping', default=False, type=str2bool, nargs='?', const=True, help='x')

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
    value_list, label_list, image_path_list, data_length = get_data_w_image(cfg)
    if cfg.set_image:
        make_image_data(image_path_list, value_list, data_length)

    '''
    ==============================
    Anomaly Detection
    ==============================
    '''
    predict_file_name = f'experiment/{cfg.dataset_name}/{cfg.llm_model}/{cfg.class_name}.json'

    if cfg.anomaly_detect:
        print_check = True
        prompt = prompt_selection(cfg)
        model_set = llm_selection(cfg, device)
        dict_arr = []

        print(predict_file_name)
        with open(predict_file_name, 'w') as file:
            for num in progress_bar(range(data_length), total=data_length):
                image_path = image_path_list[num]
                print_check = print_prompt(print_check, prompt)
                json_output = inference(cfg, model_set, prompt, image_path, device)
                dict_arr.append(json_output)
                print(num, image_path, json_output)
            json.dump(dict_arr, file, indent=4)

    # evaluation
    evaluation(cfg, value_list, label_list, predict_file_name, data_length)


if __name__=="__main__":
    main()
    
