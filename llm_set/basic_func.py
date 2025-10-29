import textwrap
import torch
import re
import json
from util import get_path_range, get_normal_set
from llm_set.qwen_func import qwen_make_messages_one_image, qwen_make_messages_images, qwen_inference
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from fsm_func import get_period_arr, time_segmentation, seg_grouping, group_inference, segment_inference, two_guide_group_inference
import warnings
warnings.filterwarnings('ignore')


def print_long_string(long_string, width=70):
    wrapped_string = textwrap.fill(long_string, width)
    print(wrapped_string)


def print_prompt(check, prompt):
    if check: 
        print('==========================================')
        print(prompt)
        print('==========================================')
    return False


def llm_selection(cfg, device):
    if cfg.llm_model == 0:
        model = Qwen2VLForConditionalGeneration.from_pretrained(
        "llm_model/Qwen2-VL-7B-Instruct", torch_dtype="auto", device_map="auto")
        processor = AutoProcessor.from_pretrained("llm_model/Qwen2-VL-7B-Instruct")
        model_set = (model, processor)
    return model_set, processor, model


def inference(cfg, model_set, prompt, image_path, device):
    if cfg.llm_model == 0:
        model = model_set[0]
        processor = model_set[1]
        messages = qwen_make_messages_one_image(prompt, image_path)
        output = qwen_inference(model, processor, messages, device)
        
    try:
        output = re.findall(r'\[.*?\]', output)[0]
    except:
        output = '[]'
    return output


def fsm_inference(cfg, model_set, test_value, device):
    top_k = cfg.top_k
    group_length = cfg.group_length
    interval = cfg.interval
    model, processor = model_set[0], model_set[1] 
    test_length = len(test_value)

    # period calculation
    magnitude_arr, period_arr = get_period_arr(values=test_value, time_length=test_length, top_k=top_k, interval=interval)
    period_arr.append(test_length)
    # time segmentation
    all_segment_list = time_segmentation(values=test_value, time_length=test_length, period_arr=period_arr, top_k=top_k+1)

    # inference 
    all_output_list = []
    for k in range(top_k+1):
        if k == top_k:
            prompt = prompt_for_fsm(True)
        else:
            if cfg.grouping:
                prompt = prompt_for_fsm(False)
            else:
                prompt = prompt_for_fsm(True)

        segment_list = all_segment_list[k]
        # inference per segment group 
        if cfg.grouping == True:
            group_list = seg_grouping(segment_list, group_length=group_length)
            output_list = group_inference(model, processor, prompt, group_list, test_length, group_length=group_length, device=device)
        # Inference per segment
        else:
            output_list = segment_inference(model, processor, prompt, segment_list, test_length)
        all_output_list.append(output_list)

    output_dict = {}
    output_dict['magnitude'] = json.dumps(magnitude_arr)
    output_dict['interval'] = json.dumps(all_output_list)
    return output_dict


def fsm_two_guide_inference(cfg, model_set, moment_model, normal_guide_bag, normal_caption_bag, path_range_set, test_value, device):
    top_k = cfg.top_k
    group_length = cfg.group_length
    interval = cfg.interval
    model, processor = model_set[0], model_set[1] 
    test_length = len(test_value)
    data_type = cfg.class_name

    # period calculation
    magnitude_arr, period_arr = get_period_arr(values=test_value, time_length=test_length, top_k=top_k, interval=interval)
    period_arr.append(test_length)
    # time segmentation
    all_segment_list = time_segmentation(values=test_value, time_length=test_length, period_arr=period_arr, top_k=top_k+1)

    # inference 
    all_output_list = []
    for k in range(top_k+1):
        segment_list = all_segment_list[k]
        segment_length = len(segment_list[0])

        # inference per segment group 
        if cfg.grouping == True:
            group_list = seg_grouping(segment_list, group_length=group_length)
            normal_caption_list, normal_embedding = get_normal_set(normal_guide_bag, normal_caption_bag, segment_length, moment_model, device)

            path_range_list = get_path_range(path_range_set, segment_length)
            anomaly_guide_image = path_range_list[0]

            if k == top_k:
                output_list = two_guide_group_inference(data_type, model, processor, moment_model, normal_caption_list, normal_embedding, anomaly_guide_image, group_list, test_length, group_length, True, device)
            else:
                output_list = two_guide_group_inference(data_type, model, processor, moment_model, normal_caption_list, normal_embedding, anomaly_guide_image, group_list, test_length, group_length, False, device)

        # Inference per segment
        else:
            raise ValueError("error")
        all_output_list.append(output_list)

    output_dict = {}
    output_dict['magnitude'] = json.dumps(magnitude_arr)
    output_dict['interval'] = json.dumps(all_output_list)
    return output_dict


# ============================================================================================


def prompt_selection(cfg):
    # default
    if cfg.prompt_ver == 0 and cfg.grouping==False:
        prompt = """\
        Detect ranges of anomalies in this time series, in terms of the x-axis coordinate.
        List one by one, in JSON format. 
        If there are no anomalies, answer with an empty list [].

        Output template:
        [{"start": ..., "end": ...}, {"start": ..., "end": ...}...]
        please do not provide any additional text or explanation.
        """

    # CoT
    elif cfg.prompt_ver == 1:
        prompt = """\
        Detect ranges of anomalies in this time series, in terms of the x-axis coordinate.
        List one by one, in JSON format. 
        If there are no anomalies, answer with an empty list [].

        Output template:
        [{"start": ..., "end": ..., "reason": ...}, {"start": ..., "end": ..., , "reason": ...}...]
        
        **Consideration**:
        In the "reason" field, explain step by step why this range is considered anomalous (e.g. trend, frequency, point, out-of-range, etc).
        """

    # Segment group
    elif cfg.prompt_ver == 2:
        prompt = """\
        The image shows a univariate time series split into multiple horizontal segments for compact visualization.
        Each row corresponds to a consecutive segment of the original time series, arranged from left to right and then top to bottom.

        Detect ranges of anomalies in this time series, based on visual inconsistency or deviation compared to the other segments.
        Return the detected anomaly ranges as a list of dictionaries, one per anomaly, in terms of the x-axis coordinate.
        If there are no anomalies, return an empty list [].
        
        Output format:
        [{"start": ..., "end": ...}, {"start": ..., "end": ...}]
        Do not provide any additional explanation or text.
        """
    # Prompt Tuning (point)
    elif cfg.prompt_ver == 3:
        prompt = """\
       Detect ranges of point anomalies (i.e., extreme global outliers, spikes, or dips) in this time series, in terms of the x-axis coordinate.
       List one by one, in JSON format.
       For a single point anomaly, 'start' and 'end' can be the same x-coordinate or a very small range covering the point.
       If there are no such anomalies, answer with an empty list [].
       
       Output template:
       [{"start": ..., "end": ...}, {"start": ..., "end": ...}...]
       please do not provide any additional text or explanation.
        """

    # Prompt Tuning (range)
    elif cfg.prompt_ver == 4:
        prompt = """\
       Detect anomalous ranges or subsequences (e.g., periods of sudden high/low volatility, flatlines, or segments that clearly break the local pattern) in this time series, in terms of the x-axis coordinate.
       List one by one, in JSON format.
       If there are no such anomalies, answer with an empty list [].

       Output template:
       [{"start": ..., "end": ...}, {"start": ..., "end": ...}...]
       please do not provide any additional text or explanation.
        """

    # Prompt Tuning (trend)
    elif cfg.prompt_ver == 5:
        prompt = """\
       Detect ranges exhibiting an anomalous trend (e.g., a sudden and sustained change in the slope, a sharp level shift, or a segment that trends counter to the established pattern) in this time series, in terms of the x-axis coordinate.
       List one by one, in JSON format.
       If there are no such trend anomalies, answer with an empty list [].

       Output template:
       [{"start": ..., "end": ...}, {"start": ..., "end": ...}...]
       please do not provide any additional text or explanation.
        """

    # Prompt Tuning (freq)
    elif cfg.prompt_ver == 6:
        prompt = """\
       Detect ranges with anomalous frequencies (e.g., segments that start oscillating much faster or slower than the rest of the series) in this time series, in terms of the x-axis coordinate.
       List one by one, in JSON format.
       If there are no such trend anomalies, answer with an empty list [].

       Output template:
       [{"start": ..., "end": ...}, {"start": ..., "end": ...}...]
       please do not provide any additional text or explanation.
        """

    return prompt


def prompt_for_fsm(default=True):
    # default
    if default:
        prompt = """\
        Detect ranges of anomalies in this time series, in terms of the x-axis coordinate.
        List one by one, in JSON format. 
        If there are no anomalies, answer with an empty list [].

        Output template:
        [{"start": ..., "end": ...}, {"start": ..., "end": ...}...]
        please do not provide any additional text or explanation.
        """

    # Segment group
    else:
        prompt = """\
        The image shows a univariate time series split into multiple horizontal segments for compact visualization.
        Each row corresponds to a consecutive segment of the original time series, arranged from left to right and then top to bottom.

        Detect ranges of anomalies in this time series, based on visual inconsistency or deviation compared to the other segments.
        Return the detected anomaly ranges as a list of dictionaries, one per anomaly, in terms of the x-axis coordinate.
        If there are no anomalies, return an empty list [].
        
        Output format:
        [{"start": ..., "end": ...}, {"start": ..., "end": ...}]
        Do not provide any additional explanation or text.
        """
    return prompt


def prompt_with_guides(guide_dict):
    prompt = f"""
    You are given three images.

    The first image serves as a visual guide, depicting a normal time series pattern based on the following description:
    {{
        "trend": {guide_dict['trend']},
        "seasonality": {guide_dict['seasonality']},
        "shape": {guide_dict['shape']},
        "state": {guide_dict['state']},
    }} 

    The second image is a visual guide showing four types of univariate time series anomalies.
    Each row corresponds to one anomaly type (from top to bottom): 
    (1) Frequency anomaly: unexpected changes in periodic patterns.
    (2) Trend anomaly: changes in the gradient of the time series, such as acceleration, deceleration, or reversal. 
    (3) Point anomaly: individual points deviating significantly from the surrounding pattern. 
    (4) Out-of-range anomaly: values that strongly deviate from the normal range. 
    Green regions in the second image indicate the anomaly ranges for each type.

    The third image shows a univariate time series for anomaly detection.    

    Identify time ranges in the third image that deviate from the normal pattern shown in the first image, or that match any of the four anomaly types illustrated in the second image.  
    Return the detected anomaly ranges as a list of dictionaries, one per anomaly, in terms of the x-axis coordinate.
    If there are no anomalies, return an empty list [].
    
    Output format only: [{{"start": ..., "end": ...}}, {{"start": ..., "end": ...}}...] 
    Do not provide any additional explanation or text.
    """
    return prompt


def prompt_with_guides_and_group(guide_dict):
    prompt = f"""
    You are given three images.

    The first image serves as a visual guide, depicting a normal time series pattern based on the following description:
    {{
        "trend": {guide_dict['trend']}
        "seasonality": {guide_dict['seasonality']}
        "shape": {guide_dict['shape']}
        "state": {guide_dict['state']}
    }} 

    The second image is a visual guide showing four types of univariate time series anomalies.
    Each row corresponds to one anomaly type (from top to bottom): 
    (1) Frequency anomaly: unexpected changes in periodic patterns. 
    (2) Trend anomaly: changes in the gradient of the time series, such as acceleration, deceleration, or reversal. 
    (3) Point anomaly: individual points deviating significantly from the surrounding pattern. 
    (4) Out-of-range anomaly: values that strongly deviate from the normal range. 
    Green regions in the second image indicate the anomaly ranges for each type.

    The third image shows a univariate time series split into multiple horizontal segments for compact visualization. 
    Each row corresponds to a consecutive segment of the original time series, arranged from left to right and then top to bottom.

    Identify time ranges in the third image that deviate from the normal pattern shown in the first image,
    or that match any of the four anomaly types illustrated in the second image.  
    Pay attention not only to matches with reference patterns, but also to inconsistencies or deviations that stand out compared to other segments.    
    Return the detected anomaly ranges as a list of dictionaries, one per anomaly, in terms of the x-axis coordinate.
    If there are no anomalies, return an empty list [].

    Output format only: [{{"start": ..., "end": ...}}, {{"start": ..., "end": ...}}...] 
    Do not provide any additional explanation or text.
   """
    return prompt