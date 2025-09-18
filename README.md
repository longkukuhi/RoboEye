<h1 align="center">
RoboEye: Enhancing 2D Robotic Object Identification with Selective 3D Geometric Keypoint Matching</h1>

<img src="./arch.png" width="70%" height="70%" />

* **Official PyTorch implementation for paper:  "RoboEye: Enhancing 2D Robotic Object Identification with Selective 3D Geometric Keypoint Matching."** <br>

## 📰 Updates
- [2025/09/17] The code of RoboEye is released! 🎉

## 🗞️ Contents
- [Setup](#setup)
- [Download pre-trained 2D feature extractor](#download-pre-trained-2d-feature-extractor)
- [Download pre-trained robot 3D retrieval transformer](#download-pre-trained-robot-3d-retrieval-transformer)
- [Download Text Tokenizer](#download-text-tokenizer)
- [Download our preprocessed json files](#download-our-preprocessed-json-files)
- [(Optional) Download our checkpoints](#optional-download-our-checkpoints)
- [Object Identification in ARMBench](#object-identification-in-armbench)

## 🛠️ Setup

First, clone this repository to your local machine, and install the dependencies.
```bash
conda create -n roboeye python=3.11 -y
conda activate roboeye
cd RoboEye
pip install -r requirements.txt
```
❗ You can modify the PyTorch version to suit your machine.

## 📍 Download pre-trained 2D Feature Extractor

   - [`BEiT3-Base`](https://conversationhub.blob.core.windows.net/beit-share-public/beit3/pretraining/beit3_base_patch16_224.pth?sv=2021-10-04&st=2023-06-08T11%3A16%3A02Z&se=2033-06-09T11%3A16%3A00Z&sr=c&sp=r&sig=N4pfCVmSeq4L4tS8QbrFVsX6f6q844eft8xSuXdxU48%3D): #layer=12; hidden=768; FFN factor=4x; #head=12; patch=16x16; #parameters: 222M

## 📍 Download pre-trained Robot 3D Retrieval Transformer

Download the model weights [here](https://huggingface.co/facebook/VGGT-1B/blob/main/model.pt) and load, or:

```python
model = VGGT()
_URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
model.load_state_dict(torch.hub.load_state_dict_from_url(_URL))
```


## 📍 Download Text Tokenizer

[beit3.spm](https://conversationhub.blob.core.windows.net/beit-share-public/beit3/sentencepiece/beit3.spm) is the sentencepiece model used for tokenizing texts.
```
from transformers import XLMRobertaTokenizer
tokenizer = XLMRobertaTokenizer("/your_beit3_model_path/beit3.spm")
```

## 📦 Download our preprocessed json files

The dataset will be made publicly available upon acceptance of the paper.

## 🗂️ (Optional) Download our checkpoints

The checkpoints will be made publicly available upon acceptance of the paper.

## 🎮 Object Identification in ARMBench
First extract positive and negative examples for the adapter-based training.
```bash
python armbench/ID.py --model 'beit3_base_patch16_224' --input_size 224 --task 'armbenchpick1extract' --batch_size 128 \
 --layer_decay 0.65 --lr 2e-4 --epochs 30 --warmup_epochs 3 --drop_path 0.2 --sentencepiece_model 'beit3.spm' \
 --data_path 'path/to/your/dataset' --output_dir '/your_output_path/' --log_dir '/your_log_path/' --weight_decay 0.05  \
 --save_ckpt_freq 1 --finetune 'path/to/ckpt/beit3_base_patch16_224.pth' -vggt_path 'path/to/ckpt/model.pt' --eval
```
- `model` specifics the name of model we use in this experiments. 
- `log_dir` is the folder dir that stores the ouput log.
- `task`  specifics using armbenchpick1extract for only use pre-pick images, armbench3t1extract for use both pre-pick and post-pick images. 
- `data_path` is the folder dir that stores the datasets.
- `finetune` specifics the dir to pre-trained weight of BEiT-3 model.
- `vggt_path` specifics the dir to pre-trained weight of VGGT model.


Train the 3D keypoint-based retrieval matcher with adapter-based strategy.
```bash
python armbench/rerank.py --model 'vggt' --input_size 224 --task 'rerank' --batch_size 128 \
 --layer_decay 0.65 --lr 5e-5 --epochs 30 --warmup_epochs 3 --drop_path 0.2 --sentencepiece_model 'beit3.spm' \
 --data_path 'path/to/your/dataset' --output_dir '/your_output_path/' --log_dir '/your_log_path/' --weight_decay 0.05  \
 --save_ckpt_freq 1 -vggt_path 'path/to/ckpt/model.pt'
```
- `model` specifics the name of model we use in this experiments. 
- `log_dir` is the folder dir that stores the ouput log.
- `task`  specifics using rerank for training with only use pre-pick images, rerank3t1 for training with both pre-pick and post-pick images.
- `data_path` is the folder dir that stores the datasets.
- `vggt_path` specifics the dir to pre-trained weight of VGGT model.


Sample positive and negative examples for the MRR-driven 3D-awareness training.
```bash
python armbench/ID.py --model 'beit3_base_patch16_224' --input_size 224 --task 'armbenchpick1sample' --batch_size 128 \
 --layer_decay 0.65 --lr 2e-4 --epochs 30 --warmup_epochs 3 --drop_path 0.2 --sentencepiece_model 'beit3.spm' \
 --data_path 'path/to/your/dataset' --output_dir '/your_output_path/' --log_dir '/your_log_path/' --weight_decay 0.05  \
 --save_ckpt_freq 1 --finetune 'path/to/ckpt/beit3_base_patch16_224.pth' -vggt_path 'path/to/ckpt/model.pth' --eval
```
- `model` specifics the name of model we use in this experiments. 
- `log_dir` is the folder dir that stores the ouput log.
- `task`  specifics using armbenchpick1sample for only use pre-pick images, armbench3t1sample for use both pre-pick and post-pick images. 
- `data_path` is the folder dir that stores the datasets.
- `finetune` specifics the dir to pre-trained weight of BEiT-3 model.
- `vggt_path` specifics the dir to re-trained weight of VGGT model.


Train the 3D-feature-awareness module with MRR-driven 3D-awareness training.
```bash
python armbench/rerank.py --model 'vggt' --input_size 224 --task 'classifier' --batch_size 256 \
 --layer_decay 0.65 --lr 1e-3 --epochs 30 --warmup_epochs 3 --drop_path 0.2 --sentencepiece_model 'beit3.spm' \
 --data_path 'path/to/your/dataset' --output_dir '/your_output_path/' --log_dir '/your_log_path/' --weight_decay 0.05  \
 --save_ckpt_freq 1 -vggt_path 'path/to/ckpt/model.pth'
```
- `model` specifics the name of model we use in this experiments. 
- `log_dir` is the folder dir that stores the ouput log.
- `task`  specifics using classifier for selective training objective.
- `data_path` is the folder dir that stores the datasets.
- `vggt_path` specifics the dir to re-trained weight of VGGT model.


Final inference of the two-stage paradigm.
```bash
python armbench/ID.py --model 'beit3_base_patch16_224' --input_size 224 --task 'armbenchpick1' --batch_size 128 \
 --layer_decay 0.65 --lr 2e-4 --epochs 30 --warmup_epochs 3 --drop_path 0.2 --sentencepiece_model 'beit3.spm' \
 --data_path 'path/to/your/dataset' --output_dir '/your_output_path/' --log_dir '/your_log_path/' --weight_decay 0.05  \
 --save_ckpt_freq 1 --finetune 'path/to/ckpt/beit3_base_patch16_224.pth' -vggt_path 'path/to/ckpt/model.pth' --eval
```
- `model` specifics the name of model we use in this experiments. 
- `log_dir` is the folder dir that stores the ouput log.
- `task`  specifics using armbenchpick1t for only use pre-pick images, armbench3t1 for use both pre-pick and post-pick images.
- `data_path` is the folder dir that stores the datasets.
- `finetune` specifics the dir to pre-trained weight of BEiT-3 model.
- `vggt_path` specifics the dir to re-trained weight of VGGT model.



## 🙏 Acknowledgement

This repository is built using the [VGGT](https://github.com/facebookresearch/vggt), the [BEiT](https://github.com/microsoft/unilm/tree/master/beit), the [BEiTv2](https://github.com/microsoft/unilm/tree/master/beit2), the [BEiTv3](https://github.com/microsoft/unilm/tree/master/beit3), the [CLIP](https://github.com/openai/CLIP), the [open_clip](https://github.com/mlfoundations/open_clip), the [Oscar](https://github.com/microsoft/Oscar), the [DeiT](https://github.com/facebookresearch/deit) repository, the [timm](https://github.com/rwightman/pytorch-image-models) library and many other inspiring works in the community.


## 📄 License
This project is licensed under the license found in the LICENSE file in the root directory of this source tree.
[Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct)

