





# Learning Content and Context with Language Bias for VQA

This repo contains code for our paper ["Learning Content and Context with Language Bias for Visual Question Answering"](https://arxiv.org/pdf/2012.11134.pdf)
This repo contains code modified from [here](https://github.com/yanxinzju/CSS-VQA),many thanks!



This repo contains code to run the VQA-CP experiments from our paper ["Don’t Take the Easy Way Out: Ensemble Based Methods for Avoiding Known Dataset Biases"](https://arxiv.org/abs/1909.03683). In particular, it contains code to a train VQA model so that it does not make use of question-type priors when answering questions, and evaluate it on [VQA-CP v2.0](https://www.cc.gatech.edu/~aagrawal307/vqa-cp/).

This repo is a fork of [this](https://github.com/hengyuan-hu/bottom-up-attention-vqa/) implementation of the [BottomUpTopDown VQA model](https://arxiv.org/abs/1707.07998). This fork extends the implementation so it can be used on [VQA-CP v2.0](https://www.cc.gatech.edu/~aagrawal307/vqa-cp/), and supports the debiasing methods from our paper.



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

to download the data <br>
and the rest of the data and trained model can be obtained from [BaiduYun](https://pan.baidu.com/s/1jdIh5hNqhe_InfufJu79qg)(passwd:muma) or [GoogleDrive](https://drive.google.com/drive/folders/13e-b76otJukupbjfC-n1s05L202PaFKQ?usp=sharing)

to download the data <br>
and the rest of the data and trained model can be obtained from [BaiduYun](https://pan.baidu.com/s/1oHdwYDSJXC1mlmvu8cQhKw)(passwd:3jot) or [GoogleDrive](https://drive.google.com/drive/folders/13e-b76otJukupbjfC-n1s05L202PaFKQ?usp=sharing)
unzip feature1.zip and feature2.zip and merge them into data/rcnn_feature/ <br>

use

```
bash tools/process.sh 
```

to process the data <br>

All data should be downloaded to a 'data/' directory in the root directory of this repository.

The easiest way to download the data is to run the provided script `tools/download.sh` from the repository root. The features are provided by and downloaded from the original authors' [repo](https://github.com/peteanderson80/bottom-up-attention). If the script does not work, it should be easy to examine the script and modify the steps outlined in it according to your needs. Then run `tools/process.sh` from the repository root to process the data to the correct format.



### Training

Run

```
CUDA_VISIBLE_DEVICES=0 python main.py --dataset cpv2 --mode updn --debias CCB_loss  --output [] --seed 0
```

to train a model

Run `python main.py --output_dir /path/to/output --seed 0` to start training our Learned-Mixin +H VQA-CP model, see the command line options for how to use other ensemble method, or how to train on non-changing priors VQA 2.0.



```
CUDA_VISIBLE_DEVICES=0 python main.py --dataset cpv2 --mode q_v_debias --debias CCB_loss --topq 1 --topv -1 --qvp 0 --output [] --seed 0
```







### Testing

Run

```
CUDA_VISIBLE_DEVICES=0 python eval.py --dataset cpv2 --debias CCB_loss --model_state []
```

to eval a model



The scores reported by the script are very close (within a hundredth of a percent in my experience) to the results reported by the official evaluation metric, but can be slightly different because the answer normalization process of the official script is not fully accounted for. To get the official numbers, you can run `python save_predictions.py /path/to/model /path/to/output_file` and the run the official VQA 2.0 evaluation [script](https://github.com/GT-Vision-Lab/VQA/blob/master/PythonEvaluationTools/vqaEvalDemo.py) on the resulting file.

### 





###Code Changes



In general we have tried to minimizes changes to the original codebase to reduce the risk of adding bugs, the main changes are:

1. The download and preprocessing script also setup [VQA-CP 2.0](https://www.cc.gatech.edu/~aagrawal307/vqa-cp/)
2. We use the filesystem, instead of HDF5, to store image feature. On my machine this is about a 1.5-3.0x speed up
3. Support dynamically loading the image features from disk during training so models can be trained on machines with less RAM
4. Debiasing objectives are added in `vqa_debiasing_objectives.py`
5. Some additional arguments are added to `main.py` that control the debiasing objective
6. Minor quality of life improvements and tqdm progress monitoring

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


