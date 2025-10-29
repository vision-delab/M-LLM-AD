import os
import pickle
from util import make_ts_image
from fastprogress import progress_bar


def get_data(cfg):
    file_path = cfg.data_path
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
        value_list = []
        label_list = []

        if cfg.dataset_name == 'synthetic':
            data_length = len(data['series'])

            for num in range(data_length):
                org_values = data['series'][num]
                values = [value[0] for value in org_values]

                org_answers = data['anom'][num]
                labels = [0 for _ in range(len(values))]
                for answer in org_answers[0]:
                    start, end = answer[0], answer[1] # interval 
                    labels[start:end] = [1 for _ in range(start, end)]

                value_list.append(values)
                label_list.append(labels)

    return value_list, label_list, data_length


def get_data_w_image(cfg):
    file_path = cfg.data_path
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
        value_list = []
        label_list = []
        image_path_list = []

        if cfg.dataset_name == 'synthetic':
            data_length = len(data['series'])

            for num in range(data_length):
                org_values = data['series'][num]
                values = [value[0] for value in org_values]

                org_answers = data['anom'][num]
                labels = [0 for _ in range(len(values))]
                for answer in org_answers[0]:
                    start, end = answer[0], answer[1] # interval 
                    labels[start:end] = [1 for _ in range(start, end)]

                image_path = f'./Images/{cfg.dataset_name}/{cfg.class_name}/{cfg.class_name}_{num}.jpg'

                value_list.append(values)
                label_list.append(labels)
                image_path_list.append(image_path)

    return value_list, label_list, image_path_list, data_length


def make_image_data(image_path_list, value_list, data_length):
    print('set image...')
    for num in progress_bar(range(data_length), total=data_length):
        image_path = image_path_list[num]
        length = len(value_list[num])
        values = value_list[num]
        make_ts_image(image_path, length, values)