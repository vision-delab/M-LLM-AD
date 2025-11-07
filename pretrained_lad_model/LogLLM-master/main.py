import subprocess
import sys

dataset = "BGL" # example

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
        f"/home/jyy1551/LAD/LogLLM/log/LogLLM_{dataset}_ft_%j.log",
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
    print("구현 예정")
    # others
    



    