# Code-LMTAD
Harnessing Large Language Models for Multivariate Time Series Anomaly Detection [[View](https://shacoding.com/2018/07/29/harnessing-large-language-models-for-multivariate-time-series-anomaly-detection/)]  
```password: delab_llm```

## 1. Datasets
We downloaded the synthetic datasets and moved each folder to the ```data/synthetic``` directory. As a result, the datasets are stored in paths such as ```data/synthetic/freq``` and ```data/synthetic/flat-trend```.  
For the real-world datasets, we followed the preprocessing steps provided in the **TranAD** GitHub and moved them to the ```data/real-world``` directory.

- **Synthetic dataset** ```AnomLLM, ICLR'25```: [[Paper](https://arxiv.org/pdf/2410.05440)][[GitHub](https://github.com/rose-stl-lab/anomllm?tab=readme-ov-file)][[Google Drive](https://drive.google.com/file/d/19KNCiOm3UI_JXkzBAWOdqXwM0VH3xOwi/view)]  
- **Real-world dataset** ```TranAD, VLDB'22```: [[Paper](https://arxiv.org/pdf/2201.07284)][[GitHub](https://github.com/imperial-qore/TranAD)] 


## 2. Requirements and Installation
- **M-LLM->** ```Qwne2-VL```: [[Huggingface]](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct)
- **Pretrained encoder->** ```MOMENT```: [[Paper](https://arxiv.org/pdf/2402.03885)][[GitHub]](https://github.com/moment-timeseries-foundation-model/moment)
- **Install required packages**:
```bash
conda create -n qwen python=3.10.16

git lfs clone  https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct

pip install git+https://github.com/huggingface/transformers
qwen-vl-utils
torch==2.6.0
matplotlib
fastprogress
scikit-learn
numpy==1.24.3
accelerate
torchvision
momentfm
```
We downloaded **Qwen2-VL** using the command above and moved the LLM files to ```llm_model/Qwen2-VL-7B-Instruct```.  
We used the **MOMENT (ICML'24)** model to generate and utilize guides. For more details, please refer to the GitHub link above.

## 3. Command
```class_name (synthetic)```: [freq, point, range, trend]
- run_baseline.py
```bash
python run_baseline.py --dataset='synthetic' --class_name='freq' --set_image=True  # set image -> inference -> evaluation
python run_baseline.py --dataset='synthetic' --class_name='freq' # inference -> evaluation
python run_baseline.py --dataset='synthetic' --class_name='freq' --anomaly_detect=False # only evaluation
```
- run_fsm.py
```bash
python run_fsm.py --dataset='synthetic' --class_name='freq' 
```
- run_two_guide_fsm.py
```bash
python run_two_guide_fsm.py --dataset='synthetic' --class_name='freq' 
```
