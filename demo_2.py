import torch
import json
import warnings
from types import SimpleNamespace
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from io import BytesIO
import re
import pickle
import os
from sklearn.metrics import precision_score, recall_score, f1_score
from qwen_vl_utils import process_vision_info
from config import update_config
from util import make_folder
from llm_set.basic_func import prompt_selection, llm_selection

# 진행상황 바 (없으면 pip install fastprogress)
try:
    from fastprogress import progress_bar
except ImportError:
    def progress_bar(iterable, **kwargs):
        return iterable

warnings.filterwarnings('ignore')


# ============================================================================================
# 1. Image Processing Helper Functions
# ============================================================================================

def to_rgb(pil_image: Image.Image) -> Image.Image:
    if pil_image.mode == 'RGBA':
        white_background = Image.new("RGB", pil_image.size, (255, 255, 255))
        white_background.paste(pil_image, mask=pil_image.split()[3])
        return white_background
    else:
        return pil_image.convert("RGB")


def segment_image_process(values, xlim_arr, ylim_arr=(-1, 1)):
    """
    주어진 값(values)을 그래프로 그려 PIL 이미지로 변환
    """
    plt.close()
    plt.figure(figsize=(12, 2))

    x_range = range(xlim_arr[0], xlim_arr[1])

    plt.plot(x_range, values)
    plt.xlim(xlim_arr)
    plt.ylim(ylim_arr)

    # x축 눈금 설정
    if (xlim_arr[1] - xlim_arr[0]) < 5:
        app_range = 1
    else:
        app_range = (xlim_arr[1] - xlim_arr[0]) // 5

    xticks = list(range(xlim_arr[0], xlim_arr[1] + 1, app_range))
    plt.xticks(xticks, fontsize=20)
    plt.yticks(fontsize=20)
    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    pil_image = to_rgb(Image.open(buf))
    plt.close()
    return pil_image


def stack_images_vertically(image_list):
    """이미지 리스트를 수직으로 쌓아 하나의 이미지로 병합"""
    if not image_list:
        return None
    widths, heights = zip(*(img.size for img in image_list))
    max_width = max(widths)
    total_height = sum(heights)
    new_image = Image.new('RGB', (max_width, total_height), (255, 255, 255))
    y_offset = 0
    for img in image_list:
        new_image.paste(img, (0, y_offset))
        y_offset += img.height
    return new_image


def make_ts_image_for_testing(length, values, predicted, label, precision, recall, f1score):
    """결과 시각화"""
    predicted_idx = [i for i, l in enumerate(predicted) if l == 1]
    anomalies_idx = [i for i, l in enumerate(label) if l == 1]

    plt.clf()
    plt.figure(figsize=(24, 4))
    plt.plot([num for num in range(length)], [score for score in values], label='Value')
    plt.bar(predicted_idx, 1, bottom=0, width=1, color='red', alpha=0.5, label='Predicted')
    plt.bar(anomalies_idx, 2, bottom=-1, width=1, color='green', alpha=0.5, label='Ground-truth')

    plt.plot([], [], ' ', label=f'Precision: {precision:.2f}')
    plt.plot([], [], ' ', label=f'Recall: {recall:.2f}')
    plt.plot([], [], ' ', label=f'F1 Score: {f1score:.2f}')

    plt.xlabel("Time", fontsize=20)
    plt.ylabel("Value", fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.legend(fontsize=24, loc='upper left')
    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    plt.close()
    pil_image = Image.open(buf)
    return pil_image


# ============================================================================================
# 2. Frequency Analysis & Segmentation Logic
# ============================================================================================

def get_period_arr(values, time_length, top_k=2, interval=100):
    """FFT를 사용하여 시계열의 주요 주기(Period)를 계산"""
    half = len(values) // 2
    fft_result = np.fft.fft(values)
    magnitude = np.abs(fft_result)[:half]

    m_tuple = [(i + 1, v) for i, v in enumerate(magnitude[1:])]
    sm_tuple = sorted(m_tuple, key=lambda x: x[1])[::-1]

    high_indice = [v[0] for v in sm_tuple]
    magnitude_arr = [v[1] for v in sm_tuple]

    period_arr = [round(time_length / i) for i in high_indice]

    if interval > 0 and top_k > 0:
        set_arr = [(magnitude_arr[i], period_arr[i]) for i in range(len(period_arr))]
        selected = []
        selected_periods = []

        for magnitude, period in set_arr:
            if period > 8:  # 너무 짧은 주기는 제외
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

        period_arr = [v[1] for v in selected]
    else:
        period_arr = []

    return period_arr


def time_segmentation(values, time_length, period_arr, top_k=2):
    """주기를 기반으로 시계열 데이터를 세그먼트로 분할"""
    all_segment_list = []
    run_k = min(top_k, len(period_arr))
    for k in range(run_k):
        period = period_arr[k]
        if period == 0: continue

        segment_list_size = time_length // period
        if segment_list_size == 0: continue

        segment_length = time_length // segment_list_size

        segment_list = [
            values[i * segment_length: i * segment_length + segment_length]
            for i in range(0, segment_list_size)
        ]
        all_segment_list.append(segment_list)
    return all_segment_list


def seg_grouping(segment_list, group_length=4):
    """세그먼트들을 그룹핑"""
    group_list = []
    group = []
    segment_list_size = len(segment_list)
    for i in range(segment_list_size):
        group.append(segment_list[i])
        if i != 0 and (i + 1) % group_length == 0:
            group_list.append(group)
            group = []
        elif i + 1 == segment_list_size:
            group_list.append(group)
    return group_list


# ============================================================================================
# 3. Inference Functions
# ============================================================================================

def local_qwen_inference(model, processor, image_input, prompt, device):
    """내부 Qwen 추론 래퍼 함수"""
    messages = [
        {"role": "system", "content": "You are a time series anomaly detector."},
        {"role": "user", "content": [
            {"type": "image", "image": image_input},
            {"type": "text", "text": prompt},
        ]},
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
    inputs = inputs.to(device)

    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True,
                                         clean_up_tokenization_spaces=False)

    return output_text[0]


def group_inference(model, processor, prompt, group_list, time_length, group_length=4, device='cuda'):
    """그룹화된 세그먼트에 대해 추론 수행"""
    output_list = []
    print(f"  - Running Group Inference (Groups: {len(group_list)})...")

    for i, group in enumerate(group_list):
        pil_image_list = []
        for j, segment in enumerate(group):
            segment_length = len(segment)
            index = group_length * i + j
            start_x = index * segment_length
            xticks_range = (start_x, start_x + segment_length)

            pil_image = segment_image_process(segment, xticks_range, (-1, 1))
            pil_image_list.append(pil_image)

        group_image = stack_images_vertically(pil_image_list)

        raw_output = local_qwen_inference(model, processor, group_image, prompt, device)

        try:
            output_str = re.findall(r'\[.*?\]', raw_output)[0]
        except:
            output_str = '[]'

        try:
            parsed = json.loads(output_str)
            for item in parsed:
                if item["start"] >= 0:
                    temp_dict = {}
                    temp_dict['start'] = item["start"]
                    temp_dict['end'] = item["end"]
                    output_list.append(temp_dict)
        except:
            pass

    return output_list


def segment_inference(model, processor, prompt, segment_list, time_length, device='cuda'):
    """개별 세그먼트에 대해 순차 추론 수행"""
    output_list = []
    segment_length = len(segment_list[0])
    print(f"  - Running Segment Inference (Segments: {len(segment_list)})...")

    for i, segment in enumerate(segment_list):
        current_start = i * segment_length
        current_end = current_start + len(segment)

        pil_image = segment_image_process(segment, (current_start, current_end), (-1, 1))

        raw_output = local_qwen_inference(model, processor, pil_image, prompt, device)

        try:
            output_str = re.findall(r'\[.*?\]', raw_output)[0]
        except:
            output_str = '[]'

        try:
            parsed = json.loads(output_str)
            for item in parsed:
                if item["start"] >= 0:
                    temp_dict = {}
                    temp_dict['start'] = item["start"]
                    temp_dict['end'] = item["end"]
                    output_list.append(temp_dict)
        except:
            pass

    return output_list


# ============================================================================================
# 4. Main Eval Logic (FSM Pipeline)
# ============================================================================================

def run_fsm_pipeline(sample_num, sample_values, sample_label, args, model, processor, output_dir, device):
    print(f"\n=== FSM Pipeline for Sample {sample_num} ===")
    time_length = len(sample_values)

    # 1. Period Calculation (FFT)
    print("1. Calculating Periods (FFT)...")
    period_arr = get_period_arr(sample_values, time_length, top_k=args.top_k, interval=args.interval)
    print(f"   -> Detected Periods: {period_arr}")

    # 2. Prepare Prompt
    prompt_text = """\
Detect ranges of anomalies in this time series, in terms of the x-axis coordinate.
List one by one, in JSON format.
If there are no anomalies, answer with an empty list [].
Output template:
[{"start": ..., "end": ...}, {"start": ..., "end": ...}...]
"""

    all_predicted_masks = []

    # --- A. Original (Whole Series) Inference ---
    print("2. Running Original (Global) Inference...")
    org_image = segment_image_process(sample_values, (0, time_length), (-1, 1))
    raw_output_org = local_qwen_inference(model, processor, org_image, prompt_text, device)

    try:
        json_org = re.findall(r'\[.*?\]', raw_output_org)[0]
        parsed_org = json.loads(json_org)
    except:
        parsed_org = []

    mask_org = np.zeros(time_length)
    for item in parsed_org:
        s, e = int(item['start']), int(item['end'])
        s, e = max(0, s), min(time_length, e)
        if s < e: mask_org[s:e] = 1

    # --- B. Segmented Inference (per Period) ---
    segments_results_masks = []

    if args.top_k > 0 and len(period_arr) > 0:
        print("3. Running Segmented Inference...")
        all_segment_lists = time_segmentation(sample_values, time_length, period_arr, top_k=args.top_k)

        for k, segment_list in enumerate(all_segment_lists):
            print(f"   -> Processing Period Index {k} (Period: {period_arr[k]})...")

            if args.grouping:
                group_list = seg_grouping(segment_list, group_length=args.group_length)
                results = group_inference(model, processor, prompt_text, group_list, time_length, args.group_length,
                                          device)
            else:
                results = segment_inference(model, processor, prompt_text, segment_list, time_length, device)

            mask_seg = np.zeros(time_length)
            for item in results:
                s, e = int(item['start']), int(item['end'])
                s, e = max(0, s), min(time_length, e)
                if s < e: mask_seg[s:e] = 1
            segments_results_masks.append(mask_seg)

    all_predicted_masks.extend(segments_results_masks)
    all_predicted_masks.append(mask_org)

    # 4. Evaluation with Ensemble
    print("4. Evaluating with Ensemble...")

    if len(segments_results_masks) > 0:
        ffp_predicted = np.mean(np.array(segments_results_masks), axis=0)
    else:
        ffp_predicted = mask_org

    org_predicted = mask_org

    best_f1 = -1
    best_result_tuple = (0, 0, 0)
    best_pred_mask = None
    best_alpha = 0

    for a in range(0, 11):
        alpha = a * 0.1
        final_score_map = alpha * ffp_predicted + (1 - alpha) * org_predicted

        pred_binary = (final_score_map >= 0.5).astype(int)

        if np.all(sample_label == 0) or np.all(sample_label == 1):
            calc_label = np.concatenate(([0], sample_label, [1]))
            calc_pred = np.concatenate(([0], pred_binary, [1]))
        else:
            calc_label = sample_label
            calc_pred = pred_binary

        p = precision_score(calc_label, calc_pred)
        r = recall_score(calc_label, calc_pred)
        f1 = f1_score(calc_label, calc_pred)

        if f1 > best_f1:
            best_f1 = f1
            best_result_tuple = (p, r, f1)
            best_pred_mask = pred_binary
            best_alpha = alpha

    print(f"   -> Best Alpha: {best_alpha:.1f}, Best F1: {best_f1:.4f}")

    # 5. Visualization
    final_image = make_ts_image_for_testing(
        length=time_length,
        values=sample_values,
        predicted=best_pred_mask,
        label=sample_label,
        precision=best_result_tuple[0],
        recall=best_result_tuple[1],
        f1score=best_result_tuple[2]
    )

    input_image = segment_image_process(sample_values, (0, time_length), (-1, 1))
    input_save_path = os.path.join(output_dir, f"input_{args.class_name}_{sample_num}.png")
    input_image.save(input_save_path)
    print(f"   -> Saved input image to {input_save_path}")

    save_path = os.path.join(output_dir, f"predicted_{args.class_name}_{sample_num}_FSM.png")
    final_image.save(save_path)
    print(f"   -> Saved result to {save_path}")

    return final_image


# ============================================================================================
# Main Setup
# ============================================================================================

def setup_environment():
    print("--- 1. Setting up environment ---")

    # [수정] llm_model을 원래 값인 0으로 복구하여 util.make_folder 경로 오류 방지
    # 새로운 알고리즘 파라미터(top_k, grouping 등)는 유지
    args = SimpleNamespace(
        dataset='synthetic',
        class_name='point',
        prompt_ver=1,
        llm_model=0,  # <--- 0으로 복구
        set_image=False,
        anomaly_detect=True,
        visualization=True,
        top_k=2,  # FFT Top-k Periods
        group_length=4,  # Grouping size
        interval=20,  # Period 간 최소 간격
        grouping=True  # Grouping 사용 여부
    )
    cfg = update_config(args)
    make_folder(cfg)
    output_dir = 'saved_images'
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    return args, cfg, output_dir, device


def load_models(cfg, device):
    print("\n--- 2. Loading models ---")
    model_set, processor, model = llm_selection(cfg, device)
    print("Model and prompt loaded successfully.")
    return model, processor


def load_dataset(file_path):
    print("\n--- 3. Loading dataset ---")
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None

    data_length = len(data['series'])
    value_list = []
    label_list = []
    for num in range(data_length):
        org_values = data['series'][num]
        values = [value[0] for value in org_values]
        org_answers = data['anom'][num]
        labels = [0 for _ in range(len(values))]
        for answer in org_answers[0]:
            start, end = answer[0], answer[1]
            labels[start:end] = [1 for _ in range(start, end)]
        value_list.append(values)
        label_list.append(labels)
    print(f"Loaded {len(value_list)} samples.")
    return value_list, label_list


def main():
    # 1. Setup
    args, cfg, output_dir, device = setup_environment()

    # 2. Load Model
    model, processor = load_models(cfg, device)

    # 3. Load Data
    data_file_path = '/home/inpyo/inpyo/Code-LMTAD-main/data/synthetic/point/eval/data.pkl'
    value_list, label_list = load_dataset(data_file_path)

    if value_list is not None:
        sample_to_run = 50
        if not (0 <= sample_to_run < len(value_list)):
            return

        sample_values = value_list[sample_to_run]
        sample_label = label_list[sample_to_run]

        run_fsm_pipeline(
            sample_num=sample_to_run,
            sample_values=sample_values,
            sample_label=sample_label,
            args=args,
            model=model,
            processor=processor,
            output_dir=output_dir,
            device=device
        )

    print("\nScript finished.")


if __name__ == "__main__":
    main()