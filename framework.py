from transformers import AutoModelForCausalLM, AutoTokenizer

import sys
import subprocess

from utils import generate_log_only_result, generate_user_defined_result


dataset = "BGL" # example (-> user input으로 받아야 할 부분)

# dataset_path = "dataset/user-defined_dataset/android_keyguardservice.txt"
# user_defined_prompt = "An anomaly occurs when a user or process attempts an action without the necessary permission or access rights."

dataset_path = "dataset/sampled_logs/Android_labeling_sample.txt"
user_defined_prompt = ""


if user_defined_prompt:
    model_name = "Qwen/Qwen2.5-7B-Instruct"
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype="auto",
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    user_defined_results = generate_user_defined_result(
        model = model,
        tokenizer = tokenizer,
        dataset_path = dataset_path,
        user_defined_prompt = user_defined_prompt
    )
else:
    if dataset in ["BGL", "Thunderbird", "HDFS_v1"]:
        try:
            if dataset in ["BGL", "Thunderbird"]:
                subprocess.run(
                    [sys.executable, "/home/jyy1551/LAD/pretrained_lad_model/LogLLM-master/prepareData/sliding_window.py", dataset],
                    check=True
                )
                print(f"data preprocessing 스크립트 실행 완료.")
            else:
                subprocess.run(
                    [sys.executable, "/home/jyy1551/LAD/pretrained_lad_model/LogLLM-master/prepareData/session_window.py"],
                    check=True
                )
        except subprocess.CalledProcessError as e:
            print(f"data preprocessing 스크립트 실행 오류: {e}")
            sys.exit(1)
        
        sbatch_command = [
            "sbatch",
            "-o",
            f"/home/jyy1551/LAD/pretrained_lad_model/log/LogLLM_{dataset}_ft_%j.log",
            "bash.sh",
            dataset  # [BGL, Thunderbird, HDFS_v1]
        ]
        # /home/jyy1551/LAD/pretrained_lad_model/LogLLM-master/eval.py 코드 실행 (sbatch 작업 제출)
        try:
            subprocess.run(
                sbatch_command,
                cwd = "/home/jyy1551/LAD/pretrained_lad_model",
                check=True
            )
            print(f"eval 스크립트 실행 완료.")
        except subprocess.CalledProcessError as e:
            print(f"bash 스크립트 실행 오류: {e}")
            sys.exit(1)
    else:
        dataset_type = "others"  # hdfs, bgl, thunderbird, others

        if dataset_type in ["BGL", "Thunderbird", "HDFS_v1"]:
            try:
                if dataset_type in ["BGL", "Thunderbird"]:
                    subprocess.run(
                        [sys.executable, "/home/jyy1551/LAD/pretrained_lad_model/LogLLM-master/prepareData/sliding_window.py", dataset_type],
                        check=True
                    )
                    print(f"data preprocessing 스크립트 실행 완료.")
                else:
                    subprocess.run(
                        [sys.executable, "/home/jyy1551/LAD/pretrained_lad_model/LogLLM-master/prepareData/session_window.py"],
                        check=True
                    )
            except subprocess.CalledProcessError as e:
                print(f"data preprocessing 스크립트 실행 오류: {e}")
                sys.exit(1)
            
            sbatch_command = [
                "sbatch",
                "-o",
                f"/home/jyy1551/LAD/pretrained_lad_model/log/LogLLM_{dataset_type}_ft_%j.log",
                "bash.sh",
                dataset_type  # [BGL, Thunderbird, HDFS_v1]
            ]
            # /home/jyy1551/LAD/pretrained_lad_model/LogLLM-master/eval.py 코드 실행 (sbatch 작업 제출)
            try:
                subprocess.run(
                    sbatch_command,
                    cwd = "/home/jyy1551/LAD/pretrained_lad_model",
                    check=True
                )
                print(f"eval 스크립트 실행 완료.")
            except subprocess.CalledProcessError as e:
                print(f"bash 스크립트 실행 오류: {e}")
                sys.exit(1)

        else:  # others
            model_name = "Qwen/Qwen2.5-7B-Instruct"
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                dtype="auto",
                device_map="auto"
            )
            tokenizer = AutoTokenizer.from_pretrained(model_name)

            log_only_results = generate_log_only_result(
                model = model,
                tokenizer = tokenizer,
                dataset_path = dataset_path
            )


    





