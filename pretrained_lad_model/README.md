# LogLLM: Log-based Anomaly Detection Using Large Language Models #

# 가상환경 세팅
```
conda create -n logllm python=3.9
conda activate logllm

conda install pip
pip install -r requirements.txt
```

# 실행 가이드
📄 https://www.notion.so/Pre-trained-Model-2a0551eaeee88040b8f9fa2995400091?source=copy_link

# LogLLM 구현 상세
📄 https://www.notion.so/LogLLM-Implementation-299551eaeee880419675e33fb2b5142f

- 일부 log dataset 크기가 매우 커서, datacenter의 scratch 경로에 데이터 저장해두었음.
- eval 코드 실행 시, Huggingface Account 정보 입력 필요함.
- datacenter, huggingface 계정 정보, 실행 방법 등 자세한 내용은 위 notion 링크 참조(초대받은 사용자만 열람 가능)


## Datasets

The statistics of datasets used in the experiments.

|             |                    |                     |    Training Data    |  Training Data  |   Training Data   |    Testing Data     |  Testing Data   |   Testing Data    |
|:-----------:|:------------------:|:-------------------:|:-------------------:|:---------------:|:-----------------:|:-------------------:|:---------------:|:-----------------:|
|             | **# Log Messages** | **# Log Sequences** | **# Log Sequences** | **# Anomalies** | **Anomaly Ratio** | **# Log Sequences** | **# Anomalies** | **Anomaly Ratio** |
|    HDFS     |     11,175,629     |       575,061       |       460,048       |      13497      |       2.93%       |       115013        |      3341       |       2.90%       |
|     BGL     |     4,747,963      |       47,135        |       37,708        |      4009       |      10.63%       |        9427         |       817       |       8.67%       |
| Thunderbird |     10,000,000     |       99,997        |       79,997        |       837       |       1.05%       |        20000        |       29        |       0.15%       |

