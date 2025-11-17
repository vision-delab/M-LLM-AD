import torch
import json
import warnings
from types import SimpleNamespace
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import matplotlib.image as mpimg
from io import BytesIO
import re
import pickle
import os
from sklearn.metrics import precision_score, recall_score, f1_score

# ✨ Qwen 유틸리티 직접 import (중요)
from qwen_vl_utils import process_vision_info
from config import update_config
from util import str2bool, make_folder

# ✨ 1. (수정됨) 'inference' import 제거
from llm_set.basic_func import prompt_selection, llm_selection, print_prompt

warnings.filterwarnings('ignore')


# --- Helper, Image Gen Functions (변경 없음) ---
def show_pil_image(image):
    plt.close()
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.axis('off')
    plt.show()


def make_ts_image(length, values, label=None, show_gt=False):
    plt.clf()
    plt.figure(figsize=(12, 2))
    plt.plot([num for num in range(length)], [score for score in values])
    if show_gt:
        anomalies_idx = [i for i, l in enumerate(label) if l == 1]
        plt.bar(anomalies_idx, 2, bottom=-1, width=1, color='green', alpha=0.5, label='Ground-truth')
    plt.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    plt.close()
    pil_image = Image.open(buf)
    return pil_image


def make_ts_image_for_testing(length, values, predicted, label, precision, recall, f1score):
    predicted_idx = [i for i, l in enumerate(predicted) if l == 1]
    anomalies_idx = [i for i, l in enumerate(label) if l == 1]
    plt.clf()
    plt.figure(figsize=(24, 4))
    plt.plot([num for num in range(length)], [score for score in values])
    plt.bar(predicted_idx, 1, bottom=0, width=1, color='red', alpha=0.5, label=f'Predicted')
    plt.bar(anomalies_idx, 2, bottom=-1, width=1, color='green', alpha=0.5, label='Ground-truth')
    plt.plot([], [], ' ', label=f'Precision: {precision:.2f}')
    plt.plot([], [], ' ', label=f'Recall: {recall:.2f}')
    plt.plot([], [], ' ', label=f'F1 Score: {f1score:.2f}')
    plt.xlabel("Time", fontsize=20);
    plt.ylabel("Value", fontsize=20)
    plt.xticks(fontsize=20);
    plt.yticks(fontsize=20)
    plt.legend(fontsize=24, loc='upper left')
    plt.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    plt.close()
    pil_image = Image.open(buf)
    return pil_image


# --- (변경 없음) 입력 이미지 생성 ---
def generate_and_save_input_image(sample_num, sample_values, sample_label, class_name, output_dir):
    print(f"\n--- Generating Input Image for Sample {sample_num} ---")
    sample_length = len(sample_label)
    sample_image = make_ts_image(sample_length, sample_values, sample_label, False)
    input_save_path = os.path.join(output_dir, f"input_{class_name}_{sample_num}.png")
    sample_image.save(input_save_path)
    print(f"Input image saved to: {input_save_path}")
    return input_save_path


# --- ✨ 2. (수정됨) 'run_inference' 함수 ---
# demo_1.py의 자체 추론 로직으로 복원 (가장 안정적)
# (단, 'prompt'를 인자로 받도록 수정)

def run_inference(input_image_path, model, processor, device, prompt):
    """
    지정된 이미지 경로의 파일을 열고,
    demo_1.py와 동일한 방식으로 'model.generate'를 직접 호출합니다.
    """
    print(f"\n--- Running Inference on {input_image_path} ---")
    try:
        sample_image = Image.open(input_image_path)
    except FileNotFoundError:
        print(f"Error: Input image not found at {input_image_path}")
        return "[]"
    except Exception as e:
        print(f"Error loading image: {e}")
        return "[]"

    # demo_1.py와 같이 'messages' 객체를 직접 생성
    messages = [
        {"role": "system", "content": "You are a time series anomaly detector."},
        {"role": "user", "content": [
            {"type": "image", "image": sample_image},  # PIL 이미지 객체
            {"type": "text", "text": prompt},  # 'prompt_selection'에서 로드한 프롬프트 텍스트
        ]},
    ]

    print("Running inference... (Using model.generate directly)")

    # demo_1.py의 추론 로직
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)  # Qwen 유틸리티 사용
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
    inputs = inputs.to(device)

    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True,
                                         clean_up_tokenization_spaces=False)
    try:
        qwen_output = re.findall(r'\[.*?\]', output_text[0])[0]
    except:
        qwen_output = '[]'

    print(f"Model Raw Output: {qwen_output}")
    return qwen_output


# --- (변경 없음) 평가 및 시각화 ---
def evaluate_and_visualize(prediction_json, sample_num, sample_values, sample_label, args, output_dir):
    # ... (내부 로직 demo_1.py와 동일) ...
    print(f"\n--- Evaluating and Visualizing Sample {sample_num} ---")
    sample_length = len(sample_label)
    current_values = list(sample_values);
    current_labels = list(sample_label)
    print("Parsing prediction and evaluating...")
    try:
        parsed = json.loads(prediction_json)
        result = [(item["start"], item["end"]) for item in parsed]
    except Exception as e:
        print(f"Error parsing JSON output: {e}");
        result = []
    predicted = [0 for _ in range(sample_length)]
    for answer in result:
        try:
            start, end = answer[0], answer[1];
            start = max(0, int(start));
            end = min(sample_length, int(end))
            if start < end: predicted[start:end] = [1 for _ in range(start, end)]
        except (IndexError, TypeError, ValueError) as e:
            print(f"Skipping invalid prediction item: {answer} (Error: {e})")
    if np.all(np.array(current_labels) == 0) or np.all(np.array(current_labels) == 1):
        predicted = [0] + predicted + [1];
        current_labels = [0] + current_labels + [1]
        current_values = [current_values[0]] + current_values + [current_values[-1]]
        sample_length = len(current_labels)
    precision = precision_score(current_labels, predicted);
    recall = recall_score(current_labels, predicted);
    f1score = f1_score(current_labels, predicted)
    print('precision:', precision);
    print('recall:', recall);
    print('f1_score:', f1score)

    pred_image = make_ts_image_for_testing(length=sample_length, values=current_values, predicted=predicted,
                                           label=current_labels, precision=precision, recall=recall, f1score=f1score)
    pred_save_path = os.path.join(output_dir, f"predicted_{args.class_name}_{sample_num}.png")
    pred_image.save(pred_save_path)
    print(f"Prediction image saved to: {pred_save_path}")

    return pred_image, (precision, recall, f1score)


# --- (변경 없음) 환경 설정 ---
def setup_environment():
    print("--- 1. Setting up environment ---")
    args = SimpleNamespace(
        dataset='synthetic', class_name='point', prompt_ver=0, llm_model=0,
        set_image=False, anomaly_detect=True, visualization=True,
        top_k=-1, group_length=-1, interval=-1, grouping=False
    )
    cfg = update_config(args)
    make_folder(cfg)
    output_dir = 'saved_images'
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    return args, cfg, output_dir, device


# --- ✨ 3. (수정됨) 모델 로드 ---
# 'prompt' 객체를 'prompt_selection'을 통해 로드하고 반환
# (model_set 반환 제거)
def load_models(cfg, device):
    print("\n--- 2. Loading models ---")

    # prompt_selection을 통해 프롬프트 로드
    prompt = prompt_selection(cfg)

    model_set, processor, model = llm_selection(cfg, device)
    print("Model and prompt loaded successfully.")

    # 'prompt' 객체도 함께 반환
    return model, processor, prompt


# --- (변경 없음) 데이터셋 로드 ---
def load_dataset(file_path):
    print("\n--- 3. Loading dataset ---")
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
    except FileNotFoundError:
        print(f"Error: Data file not found at {file_path}")
        return None, None
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None
    data_length = len(data['series'])
    value_list = []
    label_list = []
    print("Loading and processing dataset...")
    for num in range(data_length):
        org_values = data['series'][num];
        values = [value[0] for value in org_values]
        org_answers = data['anom'][num];
        labels = [0 for _ in range(len(values))]
        for answer in org_answers[0]:
            start, end = answer[0], answer[1];
            labels[start:end] = [1 for _ in range(start, end)]
        value_list.append(values);
        label_list.append(labels)
    print(f"Loaded {len(value_list)} samples.")
    return value_list, label_list


# --- ✨ 4. (수정됨) 추론 및 평가 래퍼 ---
# 'prompt' 객체를 인자로 받아 'run_inference'에 전달
# (cfg, model_set 전달 제거)
# --- ✨ 4. (수정됨) 추론 및 평가 래퍼 ---
# 'output_dir'을 인자로 받도록 수정

def run_inference_and_eval(input_path, sample_to_run, sample_values, sample_label, args, model, processor, device,
                           prompt, output_dir): # ✨ output_dir 인자 추가
    print(f"\n--- 4. Running inference and evaluation for sample {sample_to_run} ---")

    # Step 4b: 저장된 이미지로 추론 실행
    prediction_json_output = run_inference(
        input_image_path=input_path,
        model=model,
        processor=processor,
        device=device,
        prompt=prompt
    )

    # Step 4c: 추론 결과로 평가 및 시각화
    output_image, (p, r, f1) = evaluate_and_visualize(
        prediction_json=prediction_json_output,
        sample_num=sample_to_run,
        sample_values=sample_values,
        sample_label=sample_label,
        args=args,
        output_dir=output_dir
    )

    print(f"\nOverall metrics for sample {sample_to_run}: Precision={p:.4f}, Recall={r:.4f}, F1-Score={f1:.4f}")

    return output_image


# --- ✨ 5. (수정됨) Main 함수 ---
# 'load_models'에서 'prompt'를 받아 하위 함수로 전달
# (cfg, model_set 전달 제거)
def main():
    """
    스크립트의 전체 실행 흐름을 관리합니다.
    (설정 -> 모델/프롬프트 로드 -> 데이터 로드 -> [이미지 생성] -> 추론/평가)
    """

    # 1. 설정 및 환경 준비
    args, cfg, output_dir, device = setup_environment()

    # 2. 모델 로드 (✨ 'prompt' 반환 받음)
    model, processor, prompt = load_models(cfg, device)

    # 3. 데이터 로드
    data_file_path = '/home/inpyo/inpyo/Code-LMTAD-main/data/synthetic/point/eval/data.pkl'
    value_list, label_list = load_dataset(data_file_path)

    # 4. 파이프라인 실행 (데이터 로드 성공 시)
    if value_list is not None and label_list is not None:
        sample_to_run = 50  # 테스트할 샘플 번호 지정

        if not (0 <= sample_to_run < len(value_list)):
            print(f"Error: Sample number {sample_to_run} is out of range. Max index is {len(value_list) - 1}.")
            return

        sample_values = value_list[sample_to_run]
        sample_label = label_list[sample_to_run]

        # --- 4a. (선택 사항) 입력 이미지 생성 ---
        print("\n--- 4a. Generating input image (optional) ---")
        input_path = generate_and_save_input_image(
            sample_num=sample_to_run,
            sample_values=sample_values,
            sample_label=sample_label,
            class_name=args.class_name,
            output_dir=output_dir
        )
        # -----------------------------------------------

        input_path = os.path.join(output_dir, f"input_{args.class_name}_{sample_to_run}.png")

        if 'input_path' not in locals():
            print("Error: 'input_path' is not defined.")
            return

        # --- 4b. 추론 및 평가 실행 ---
        output_image = run_inference_and_eval(
            input_path=input_path,
            sample_to_run=sample_to_run,
            sample_values=sample_values,
            sample_label=sample_label,
            args=args,
            model=model,
            processor=processor,
            device=device,
            prompt=prompt,
            output_dir=output_dir
        )

        if output_image:
            print(f"\nSuccessfully received output image object for sample {sample_to_run}.")

    print("\nScript finished.")


if __name__ == "__main__":
    main()