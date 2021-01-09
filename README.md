# Learning Content and Context with Language Bias for VQA

This repo contains code to run the VQA-CP and VQA v2 experiments from our paper ["Learning Content and Context with Language Bias for Visual Question Answering"](https://arxiv.org/pdf/2012.11134.pdf). In particular, it contains code to train a VQA model so that it can utilize question-type priors to learn content and context for answering questions.

This repo contains code modified from [here](https://github.com/yanxinzju/CSS-VQA),many thanks!

### Prerequisites

Make sure you are on a machine with a NVIDIA GPU and Python 2.7 with about 100 GB disk space. <br>
h5py==2.7.1 <br>
pytorch==1.1.0 <br>
Click==7.0 <br>
numpy==1.16.5 <br>
tqdm==4.54.0 <br>

### Data Setup

You can use

```
bash tools/download.sh
```

to download the data <br> and the rest of the data can be obtained from [BaiduYun](https://pan.baidu.com/s/1oHdwYDSJXC1mlmvu8cQhKw)(passwd:3jot) or [GoogleDrive](https://drive.google.com/drive/folders/13e-b76otJukupbjfC-n1s05L202PaFKQ?usp=sharing) unzip feature1.zip and feature2.zip and merge them into data/rcnn_feature/ <br> (model.pth is the trained model from [CSS-VQA](https://github.com/yanxinzju/CSS-VQA))

Our trained models can be obtained from [BaiduYun](https://pan.baidu.com/s/1jdIh5hNqhe_InfufJu79qg)(passwd:muma) 

You can use

```
bash tools/process.sh 
```

to process the data <br>

All data should be downloaded to a 'data/' directory in the root directory of this repository.

### Training

Run

```
python main.py --dataset cpv2 --mode updn --debias CCB_loss  --output [] --seed 0
```

to train a model with our CCB learning strategy on VQA-CP v2.


Run 

```
python main.py --dataset cpv2 --mode q_v_debias --debias CCB_loss --topq 1 --topv -1 --qvp 0 --output [] --seed 0
```
to train our model with [Counterfactual Samples Synthesizing] (https://arxiv.org/pdf/2003.06576.pdf)


### Testing

Run

```
python eval.py --dataset cpv2 --debias CCB_loss --model_state []
```

to eval a model

### Code Changes

We have tried to minimizes changes to the original codebase, the main changes are:

1. Some detailed version changes are made to improve the stability and scalability of the code
2. Our model is based on [Bottom-up and Top-down model](https://arxiv.org/abs/1707.07998)
3. Our learning strategy is added in `vqa_debiasing_objectives.py`.
4. In addition, we visualize our predictions, attention weights and the bias in `visualize.py and visualize_bias.py,`.

## Citation

If you find this code useful, please cite the following paper:

  ```
@article{yang2020learning,
  title={Learning content and context with language bias for Visual Question Answering},
  author={Yang, Chao and Feng, Su and Li, Dongsheng and Shen, Huawei and Wang, Guoqing and Jiang, Bin},
  journal={arXiv preprint arXiv:2012.11134},
  year={2020}
}
  ```


