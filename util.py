import os 
import numpy as np
import matplotlib.pyplot as plt
import argparse
from PIL import Image
import json
import torch


def make_folder(cfg):
    if not os.path.exists(f'./experiment'):
        os.mkdir(f'./experiment')
    if not os.path.exists(f'./experiment/{cfg.dataset_name}'):
        os.mkdir(f'./experiment/{cfg.dataset_name}')
    if not os.path.exists(f'./experiment/{cfg.dataset_name}/{cfg.llm_model}'):
        os.mkdir(f'./experiment/{cfg.dataset_name}/{cfg.llm_model}')
    if not os.path.exists(f'./experiment/{cfg.dataset_name}/{cfg.llm_model}/{cfg.class_name}'):
        os.mkdir(f'./experiment/{cfg.dataset_name}/{cfg.llm_model}/{cfg.class_name}')
    if not os.path.exists(f'./Images'):
        os.mkdir(f'./Images')
    if not os.path.exists(f'./Images/{cfg.dataset_name}'):
        os.mkdir(f'./Images/{cfg.dataset_name}')
    if not os.path.exists(f'./Images/{cfg.dataset_name}/{cfg.class_name}'):
        os.mkdir(f'./Images/{cfg.dataset_name}/{cfg.class_name}')


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def show_pil_image(pil_image):
    plt.figure(figsize=(6, 6))
    plt.imshow(pil_image)
    plt.axis('off')  
    plt.show()


def make_ts_image(image_path, length, values):
    plt.clf()
    plt.figure(figsize=(12, 2))
    plt.plot([num for num in range(length)],[score for score in values]) 
    plt.tight_layout()
    plt.savefig(image_path)


def make_ts_image_for_testing(cfg, image_path, length, values, predicted, label, precision, recall, f1score):
    predicted_idx = [i for i,l in enumerate(predicted) if l==1] 
    anomalies_idx = [i for i,l in enumerate(label) if l==1] 

    plt.clf()
    plt.figure(figsize=(12, 4))
    plt.plot([num for num in range(length)],[score for score in values]) 
    plt.bar(predicted_idx, 1, bottom=0, width=1, color='red',alpha=0.5, label=f'Predicted') 
    plt.bar(anomalies_idx, 2, bottom=-1, width=1, color='green',alpha=0.5, label='Ground-truth') 
    plt.plot([], [], ' ', label=f'Precision: {precision:.2f}')
    plt.plot([], [], ' ', label=f'Recall: {recall:.2f}')
    plt.plot([], [], ' ', label=f'F1 Score: {f1score:.2f}')

    if cfg.llm_model == 0:
        plt.title("Qwen (7B)")  
    elif cfg.llm_model == 1:
        plt.title("MiniCPM (8B)")  
    plt.xlabel("Time")
    plt.ylabel("Value")

    plt.legend(fontsize=12, loc='upper left')
    plt.tight_layout()
    plt.savefig(image_path)


def get_path_range_set(json_file_path='Guide/anomaly_guide.json'):
    path_range_set = []
    for i in range(20):
        range_list = []
        length = 50*(i+1)
        path = f'Guide/anomaly_guide_{50*(i+1)}.png'

        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            item = data[(length//50)-1]
            for i in range(4):
                guide_range = json.loads(item[f'range_{i+1}'])
                range_list.append(guide_range)

        path_range_set.append((path, range_list))
    return path_range_set


def get_path_range(path_range_set, value):
    length_list = [50*(i+1) for i in range(20)]
    min_value = 1000
    min_idx = 0
    for i in range(20):
        length = length_list[i]
        diff = abs(length-value)
        if abs(length-value) < min_value:
            min_value = diff
            min_idx = i
    return path_range_set[min_idx]


def get_normal_set(normal_guide_bag, normal_caption_bag, value, moment_model, device):
    length_list = [50*(i+1) for i in range(20)]
    min_value = 1000
    min_idx = 0
    for i in range(20):
        length = length_list[i]
        diff = abs(length-value)
        if abs(length-value) < min_value:
            min_value = diff
            min_idx = i

    normal_guide_list = normal_guide_bag[min_idx]
    normal_caption_list = normal_caption_bag[min_idx]

    # get normal embedding
    normal_guide_tensor = torch.from_numpy(np.array(normal_guide_list))
    normal_guide_tensor = normal_guide_tensor.unsqueeze(1).float().to(device)
    normal_embedding = moment_model(x_enc=normal_guide_tensor).embeddings.to(device) # [n, 1024]

    return normal_caption_list, normal_embedding


def get_optimal_normal_guide(class_name, group, normal_caption_list, normal_embedding, model, device):
    # get group embedding
    group = np.array(group)
    mean_tensor = torch.from_numpy(np.mean(group, axis=0))
    mean_tensor = mean_tensor.view(1, 1, -1).float().to(device)
    mean_embedding = model(x_enc=mean_tensor).embeddings  # [1, 1024]

    # find similarity
    similarity = -torch.norm(normal_embedding - mean_embedding, dim=1)
    most_similar_index = int(torch.argmax(similarity).cpu())
    
    # get optimal guide
    optimal_normal_length = normal_caption_list['range']
    optimal_normal_caption = normal_caption_list['desc'][most_similar_index]
    optimal_normal_caption = json.loads(optimal_normal_caption)
    optimal_normal_img = f'Guide/{class_name}/normal_range/{optimal_normal_length}/{optimal_normal_length}_{most_similar_index}.png'

    return optimal_normal_caption, optimal_normal_img