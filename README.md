# Unlearning or Obfuscating? Jogging the Memory of Unlearned LLMs via Benign Relearning

This repository contains the code and experiments for the manuscript:

> [Unlearning or Obfuscating? Jogging the Memory of Unlearned LLMs via Benign Relearning](https://arxiv.org/abs/2406.13356)
>

Machine unlearning is a promising approach to mitigate undesirable memorization of training data in ML models. However, in this work we show that existing approaches for unlearning in LLMs are surprisingly susceptible to a simple set of *benign relearning attacks*. With access to only a small and potentially loosely related set of data, we find that we can ``jog'' the memory of unlearned models to reverse the effects of unlearning. For example, we show that relearning on public medical articles can lead an unlearned LLM to output harmful knowledge about bioweapons, and relearning general wiki information about the book series Harry Potter can force the model to output verbatim memorized text. We formalize this unlearning-relearning pipeline, explore the attack across three popular unlearning benchmarks, and discuss future directions and guidelines that result from our study. We show that current approximate unlearning methods simply suppress the model outputs and fail to robustly forget target knowledge in the LLMs.

## The code is adapted partially from [TOFU](https://github.com/locuslab/tofu)


## Installation

```
conda create -n synthetic python=3.10
conda activate synthetic
conda install pytorch pytorch-cuda=11.8 -c pytorch -c nvidia
conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
```

## Synthetic Data Experiments

Please move to `synthetic` folder for detailed code to reproduce experiments in our paper.

## WMDP Experiments

Please move to `wmdp` folder for detailed code to reproduce experiments in our paper.

## WHP Experiments

Coming soon... We will update these codes shortly.

## Citation
If you find our work / repo useful, please cite [our work]((https://arxiv.org/abs/2406.13356))
```
@article{hu2024jogging,
  title={Jogging the Memory of Unlearned LLMs Through Targeted Relearning Attacks},
  author={Hu, Shengyuan and Fu, Yiwei and Wu, Zhiwei Steven and Smith, Virginia},
  journal={arXiv preprint arXiv:2406.13356},
  year={2024}
}
```