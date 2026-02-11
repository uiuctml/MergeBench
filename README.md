# MergeBench: A Benchmark for Merging Domain-Specialized LLMs

<p align="center">
  <a href="https://yifei-he.github.io/mergebench/">
    <img src="https://img.shields.io/badge/Project-Website-blue?style=flat-square&logo=google-chrome&logoColor=white" alt="Project Website"></a>
  &nbsp;
  <a href="https://arxiv.org/abs/2505.10833">
    <img src="https://img.shields.io/badge/arXiv-Paper-red?style=flat-square&logo=arxiv" alt="arXiv Paper"></a>
  &nbsp;
  <a href="https://github.com/uiuctml/MergeBench">
    <img src="https://img.shields.io/badge/GitHub-Project-181717?style=flat-square&logo=github" alt="GitHub Project"></a>
  &nbsp;
  <a href="https://huggingface.co/MergeBench">
    <img src="https://img.shields.io/badge/Hugging%20Face-Dataset-FFD21E?style=flat-square&logo=huggingface" alt="HF Models"></a>
</p>

This is the official repo of MergeBench in the paper ["MergeBench: A Benchmark for Merging Domain-Specialized LLMs"](https://arxiv.org/abs/2505.10833) at NeurIPS 2025 Datasets and Benchmarks Track.


![alt text](MergeBench.png "MergeBench")

## Abstract
Model merging provides a scalable alternative to multi-task training by combining specialized finetuned models through parameter arithmetic, enabling efficient deployment without the need for joint training or access to all task data. While recent methods have shown promise, existing evaluations are limited in both model scale and task diversity, leaving open questions about their applicability to large, domain-specialized LLMs. To tackle the challenges, we introduce MergeBench, a comprehensive evaluation suite designed to assess model merging at scale. MergeBench builds on state-of-the-art open-source language models, including Llama and Gemma families at 2B to 9B scales, and covers five key domains: instruction following, mathematics, multilingual understanding, coding and safety. We standardize finetuning and evaluation protocols, and assess eight representative merging methods across multi-task performance, forgetting and runtime efficiency. Based on extensive experiments, we provide practical guidelines for algorithm selection and share insights showing that model merging tends to perform better on stronger base models, with techniques such as merging coefficient tuning and sparsification improving knowledge retention. However, several challenges remain, including the computational cost on large models, the gap for in-domain performance compared to multi-task models, and the underexplored role of model merging in standard LLM training pipelines. We hope MergeBench provides a foundation for future research to advance the understanding and practical application of model merging.

## Merging Algorithms
All of the constituent model checkpoints are available at https://huggingface.co/MergeBench. We provide further details in the readme file of the `merging` folder.

## Evaluation
We utilize three existing evaluation packages, and we recommend creating separate environments for each evaluation.
### lm-eval
```
conda create -n lmeval python=3.10.9
conda activate lmeval

git clone --depth 1 https://github.com/EleutherAI/lm-evaluation-harness
cd lm-evaluation-harness
pip install -e .

pip3 install torch==2.1.2 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

pip install langdetect
pip install immutabledict
```

### bigcode-eval
```
conda create -n bigcode python=3.10.9
conda activate bigcode

git clone https://github.com/bigcode-project/bigcode-evaluation-harness.git
cd bigcode-evaluation-harness

pip install -e .
pip3 install torch==2.1.2 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install numpy==1.24.1
```

### safety-eval
To install the evaluation
```
git clone https://github.com/nouhadziri/safety-eval-fork
conda create -n safety-eval python=3.10 && conda activate safety-eval
pip install -e .
pip install -r requirements.txt
pip install vllm==0.4.2
```
Running the evaluation necessitates a value for openai API key as some tasks in the benchmark suite requires openai API. However, for the ones we test on, it is not required, and you can put the placeholder as follows
```
export OPENAI_API_KEY=''
```

To perform the full evaluation on all five task categories on the base `Llama-3.2-3B` model with GPU 0 and save the results in the folder `results/llama-3.2-3b`, run the following command:
```
bash scripts/evaluate.sh meta-llama/Llama-3.2-3B 0 results/llama-3.2-3b
```

## Citation
```
@inproceedings{
    he2025mergebench,
    title={MergeBench: A Benchmark for Merging Domain-Specialized {LLM}s},
    author={Yifei He and Siqi Zeng and Yuzheng Hu and Rui Yang and Tong Zhang and Han Zhao},
    booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems Datasets and Benchmarks Track},
    year={2025},
    url={https://openreview.net/forum?id=rw50iUoyLu}
}
```