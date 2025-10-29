import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO
import numpy as np
from fastprogress import progress_bar
import re
import json
import torch
from llm_set.qwen_func import qwen_make_messages_one_image, qwen_make_messages_images, qwen_inference
from util import get_optimal_normal_guide


def to_rgb(pil_image: Image.Image) -> Image.Image:
      if pil_image.mode == 'RGBA':
          white_background = Image.new("RGB", pil_image.size, (255, 255, 255))
          white_background.paste(pil_image, mask=pil_image.split()[3])  # Use alpha channel as mask
          return white_background
      else:
          return pil_image.convert("RGB")


def segment_image_process(values, xlim_arr, ylim_arr):
    plt.close()
    plt.figure(figsize=(12, 2))
    plt.plot(
        range(xlim_arr[0], xlim_arr[1]),
        values
    )
    plt.xlim(xlim_arr)
    plt.ylim(ylim_arr)

    if (xlim_arr[1]-xlim_arr[0]) < 5:
        app_range = 1
    else:
        app_range = (xlim_arr[1]-xlim_arr[0])//5

    xticks = list(range(xlim_arr[0], xlim_arr[1]+1, app_range))
    plt.xticks(xticks)
    plt.xticks(fontsize=20) 
    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight') 
    buf.seek(0)
    pil_image = to_rgb(Image.open(buf))
    return pil_image


def stack_images_vertically(image_list):
    widths, heights = zip(*(img.size for img in image_list))
    max_width = max(widths)
    total_height = sum(heights)
    new_image = Image.new('RGB', (max_width, total_height), (255, 255, 255))
    y_offset = 0
    for img in image_list:
        new_image.paste(img, (0, y_offset))
        y_offset += img.height
    return new_image

# ============================================================================================


# 1. Period calculation
def get_period_arr(values, time_length, top_k=2, interval=100):
    half = len(values) // 2
    fft_result = np.fft.fft(values)
    magnitude = np.abs(fft_result)[:half]
    m_tuple = [(i+1, v) for i, v in enumerate(magnitude[1:])] # [(idx, magnitude), ...]
    sm_tuple = sorted(m_tuple, key=lambda x:x[1])[::-1] # sorted: [(idx, magnitude), ...]
    high_indice = [v[0] for v in sm_tuple] # [idx_1, idx_2, ...]
    magnitude_arr = [v[1] for v in sm_tuple] # [mag_1, mag_2, ...]
    period_arr = [round(time_length/i) for i in high_indice] # [period_1, period_2, ...]

    if interval > 0 and top_k > 0: 
        set_arr = [(magnitude_arr[i], period_arr[i]) for i in range(len(period_arr))]
        selected = []
        selected_periods = []

        for magnitude, period in set_arr:
            if period > 8:
                is_valid = True
                for p in selected_periods:
                    if abs(period - p) < interval:
                        is_valid = False
                        break  
                if is_valid:
                    selected.append((magnitude, period))
                    selected_periods.append(period)
                    if len(selected) == top_k:
                        break
            
        magnitude_arr = [v[0] for v in selected]
        period_arr = [v[1] for v in selected]
    
    else:
        magnitude_arr = period_arr = []

    return magnitude_arr, period_arr


# 2. Time Segmentation
def time_segmentation(values, time_length, period_arr, top_k=2):
    all_segment_list = []
    for k in range(top_k):
        segment_list_size = time_length//period_arr[k]
        segment_length = time_length//segment_list_size
        segment_list = [values[i*segment_length:i*segment_length+segment_length] for i in range(0, segment_list_size)]
        all_segment_list.append(segment_list)
    return all_segment_list


# 3. Segment Grouping
def seg_grouping(segment_list, group_length=4):
    group_list = []
    group = []
    segment_list_size = len(segment_list)
    for i in range(segment_list_size):
        group.append(segment_list[i])
        if i != 0 and (i+1) % group_length == 0:
            group_list.append(group)
            group = []
        elif i+1 == segment_list_size:
            group_list.append(group)
    return group_list


# 4. Inference per Segment Group 
def group_inference(model, processor, prompt, group_list, time_length, group_length=4, device='cuda'):
    output_list = []

    for i, group in enumerate(group_list):
        pil_image_list = []
        for j, segment in enumerate(group):
            segment_length = len(segment)
            index = group_length*i+j
            xticks_range =  (index*segment_length, index*segment_length+segment_length)
            yticks_range = (-1, 1)
            pil_image = segment_image_process(segment, xticks_range, yticks_range)
            pil_image_list.append(pil_image)
        group_image = stack_images_vertically(pil_image_list)
        
        messages = qwen_make_messages_one_image(prompt, group_image)
        output = qwen_inference(model, processor, messages, device)
        # print(f'{i}/{len(group_list)}-th group [period: {len(group[0])}]:', output)
        try:
            output = re.findall(r'\[.*?\]', output)[0]
        except:
            output = '[]'
        
        # parsing
        try:
            parsed = json.loads(output)
            for item in parsed:
                if item["start"] >= 0 and item["end"] <= time_length:
                    temp_dict = {}
                    temp_dict['start'] = item["start"]
                    temp_dict['end'] = item["end"]
                    output_list.append(temp_dict)
        except:
            pass

    return output_list


def segment_inference(model, processor, prompt, segment_list, time_length, device='cuda'):
    output_list = []
    segment_length = len(segment_list[0])

    # inference per segment about k-th frequency 
    for i, segment in enumerate(segment_list):
        pil_image = segment_image_process(segment, (i*segment_length, i*segment_length+segment_length), (-1, 1))
        messages = qwen_make_messages_one_image(prompt, pil_image)
        output = qwen_inference(model, processor, messages, device)

        try:
            output = re.findall(r'\[.*?\]', output)[0]
        except:
            output = '[]'
        
        # parsing
        try:
            parsed = json.loads(output)
            for item in parsed:
                if item["start"] >= 0 and item["end"] <= time_length:
                    temp_dict = {}
                    temp_dict['start'] = item["start"]
                    temp_dict['end'] = item["end"]
                    output_list.append(temp_dict)
        except:
            pass
    return output_list


# ============================================================================================


def two_guide_group_inference(class_name, model, processor, moment_model, normal_caption_list, normal_embedding, anomaly_guide_image, group_list, time_length, group_length=4, final=False, device='cuda'):
    from llm_set.basic_func import prompt_with_guides, prompt_with_guides_and_group
    output_list = []

    for i, group in enumerate(group_list):
        normal_guide_caption, normal_guide_image = get_optimal_normal_guide(class_name, group, normal_caption_list, normal_embedding, moment_model, device)

        pil_image_list = []
        for j, segment in enumerate(group):
            segment_length = len(segment)
            index = group_length*i+j
            pil_image = segment_image_process(segment, (index*segment_length, index*segment_length+segment_length), (-1, 1))
            pil_image_list.append(pil_image)
        group_image = stack_images_vertically(pil_image_list)

        if final:
            prompt = prompt_with_guides(normal_guide_caption)
        else:
            prompt = prompt_with_guides_and_group(normal_guide_caption)
        messages = qwen_make_messages_images(prompt, normal_guide_image, anomaly_guide_image, group_image)
        output = qwen_inference(model, processor, messages)
        try:
            output = re.findall(r'\[.*?\]', output)[0]
        except:
            output = '[]'
        
        # parsing
        try:
            parsed = json.loads(output)
            for item in parsed:
                if item["start"] >= 0 and item["end"] <= time_length:
                    temp_dict = {}
                    temp_dict['start'] = item["start"]
                    temp_dict['end'] = item["end"]
                    output_list.append(temp_dict)
        except:
            pass

    return output_list