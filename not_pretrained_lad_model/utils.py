import os

import torch


def read_log_file(path, chunk_size=50):
    if not os.path.exists(path):
        raise FileNotFoundError(f"File does not exist: {path}")

    batches = []
    buffer = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            buffer.append(line)

            if len(buffer) >= chunk_size:
                batches.append(buffer)
                buffer = []

        if buffer:
            batches.append(buffer)

    return batches


def build_user_defined_messages(log_lines, user_anomaly_definition):
    indexed_logs = "\n".join([f"{i+1}. {line}" for i, line in enumerate(log_lines)])

    prompt_text = f"""
You are an expert system log anomaly detector.

Analyze each log entry and determine whether it represents NORMAL or ABNORMAL system behavior based solely on the following user-defined rule:
{user_anomaly_definition}

If a line is ABNORMAL, provide a short and clear explanation of why it is abnormal, including the main cause or affected component. 
Keep the explanation to one or two short sentences.

Output format (strictly follow this, no explanations beyond what is requested):
1. normal
2. abnormal - explanation
3. normal
4. abnormal - explanation
...

Logs:
{indexed_logs}

Now produce the classifications following the format above.
""".strip()

    messages = [
        {"role": "system", "content": "You are Qwen, a helpful assistant specialized in system log anomaly detection."},
        {"role": "user", "content": prompt_text}
    ]
    return messages


def build_log_only_messages(log_lines):
    indexed_logs = "\n".join([f"{i+1}. {line}" for i, line in enumerate(log_lines)])
    
    prompt_text = f"""
You are an expert system log anomaly detector.

Analyze each log entry and determine whether it represents NORMAL or ABNORMAL system behavior.

Rules:
• NORMAL: Regular operational events such as successful startups, connections, completions, or expected state updates.
• ABNORMAL: Any indication of errors, failures, crashes, exceptions, timeouts, unexpected shutdowns, or suspicious activities.
• If a line is ABNORMAL, provide a brief but informative explanation describing the reason for the anomaly. 
  - Include the main cause or affected component.
  - Limit the explanation to one or two short sentences; do not write long paragraphs.

For each input line, output exactly one line.

Output format (strictly follow this, no explanations beyond what is requested):
1. normal
2. abnormal - brief explanation
3. normal
4. abnormal - brief explanation
...

Logs:
{indexed_logs}

Now produce the classifications including a brief explanation for abnormal lines.
""".strip()

    messages = [
        {
            "role": "system",
            "content": "You are Qwen, a helpful assistant specialized in system log anomaly detection."
        },
        {
            "role": "user",
            "content": prompt_text
        }
    ]
    return messages


def generate_user_defined_result(model, tokenizer, dataset_path, user_defined_prompt):
    all_responses = []
    log_batches = read_log_file(dataset_path)

    for i, log_batch in enumerate(log_batches):
        messages = build_user_defined_messages(log_batch, user_defined_prompt)

        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

        with torch.no_grad():
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=512
            )

        generated_ids = [
            output_ids[len(input_ids):] 
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        all_responses.append(response)

    with open("llm_results.txt", "w", encoding="utf-8") as f:
        for i, r in enumerate(all_responses):
            for line in r.splitlines():
                f.write(line + "\n")

    return all_responses


def generate_log_only_result(model, tokenizer, dataset_path):
    all_responses = []
    log_batches = read_log_file(dataset_path)

    for i, log_batch in enumerate(log_batches):
        messages = build_log_only_messages(log_batch)

        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

        with torch.no_grad():
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=512
            )

        generated_ids = [
            output_ids[len(input_ids):] 
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        all_responses.append(response)

    with open("llm_results.txt", "w", encoding="utf-8") as f:
        for i, r in enumerate(all_responses):
            for line in r.splitlines():
                f.write(line + "\n")

    return all_responses
