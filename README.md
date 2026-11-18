# RoboEye

RoboEye identifies an object picked by a robot. A 2D retrieval model first finds likely matches in the ARMBench reference gallery. A learned classifier selects cases in which 3D keypoint matching can improve their ranking. [ICRA 2026 conference program](https://ras.papercept.net/conferences/conferences/ICRA26/program/ICRA26_ContentListWeb_4.html).

![RoboEye architecture for selective 3D matching](assets/arch.png)

## Prepare the environment and inputs

```bash
conda create -n roboeye python=3.11 -y
conda activate roboeye
python -m pip install -r requirements.txt
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
```

Run the commands from the repository root on a CUDA system. [requirements.txt](requirements.txt) specifies PyTorch 2.7.1 and torchvision 0.22.1. Use the bundled [BEiT-3 tokenizer](beit3.spm), the [BEiT-3 Base checkpoint](https://conversationhub.blob.core.windows.net/beit-share-public/beit3/pretraining/beit3_base_patch16_224.pth?sv=2021-10-04&st=2023-06-08T11%3A16%3A02Z&se=2033-06-09T11%3A16%3A00Z&sr=c&sp=r&sig=N4pfCVmSeq4L4tS8QbrFVsX6f6q844eft8xSuXdxU48%3D) for 2D retrieval and the [VGGT-1B checkpoint](https://huggingface.co/facebook/VGGT-1B/blob/main/model.pt) for initial 3D feature extraction.

Prepare ARMBench pick and reference images with the split files expected by [the identification loader](armbench/armbench_datasets.py). For example, the multi-view loader reads `<split>_data_3t1.json` and `all_labels.json`. This release provides no download links for its preprocessed split files or task-specific fine-tuned checkpoints. The public BEiT-3 and VGGT downloads initialize individual stages.

## Follow an object through the pipeline

| Stage | Entry point | Pre-pick alone | Pre- and post-pick views |
| --- | --- | --- | --- |
| Extract candidate training pairs | [ID.py](armbench/ID.py) with `--eval` | `armbenchpick1extract` | `armbench3t1extract` |
| Train the geometric matcher | [rerank.py](armbench/rerank.py) | `rerank` | `rerank3t1` |
| Sample features and labels for the selector | ID.py with `--eval` | `armbenchpick1sample` | `armbench3t1sample` |
| Train the selector | rerank.py | See filename note below | `classifier` |
| Evaluate identification | ID.py with `--eval` | `armbenchpick1` | `armbench3t1` |

For example, extract multi-view pairs with the upstream VGGT checkpoint. Set `DATA`, `BEIT` and `VGGT` to local paths:

```bash
DATA=/path/to/armbench
BEIT=/path/to/beit3_base_patch16_224.pth
VGGT=/path/to/model.pt

python armbench/ID.py --model beit3_base_patch16_224 --task armbench3t1extract \
  --input_size 224 --batch_size 128 --sentencepiece_model beit3.spm \
  --data_path "$DATA" --finetune "$BEIT" --vggt_path "$VGGT" \
  --output_dir runs/extract --log_dir logs/extract --eval
```

Extraction writes `sample_paths_full_3t1.json` in the working directory. Place a copy under `$DATA`, where the matcher reads it:

```bash
cp sample_paths_full_3t1.json "$DATA/"
python armbench/rerank.py --model vggt --task rerank3t1 \
  --data_path "$DATA" --vggt_path "$VGGT" --output_dir runs/matcher --log_dir logs/matcher \
  --batch_size 128 --epochs 30 --lr 5e-5 --layer_decay 0.65 --drop_path 0.2 \
  --warmup_epochs 3 --weight_decay 0.05 --save_ckpt_freq 1
```

For multi-view sampling, repeat the first command with `--task armbench3t1sample`, point `--vggt_path` at the trained matcher and use a new output directory. [The selector loader](armbench/classifier_datasets.py) reads `features_3t1.npy` and `labels_3t1.npy` from the data root. The single-view sampler produces `features.npy` and `labels.npy`, so that route requires a corresponding loader change.

```bash
python armbench/rerank.py --model vggt --task classifier \
  --data_path "$DATA" --output_dir runs/selector --log_dir logs/selector \
  --batch_size 256 --epochs 30 --lr 1e-3 --layer_decay 0.65 --drop_path 0.2 \
  --warmup_epochs 3 --weight_decay 0.05 --save_ckpt_freq 1
```

For final inference, `ID.py` accepts the selector checkpoint via `--classifier_path`. Its explicit `--vggt_path` loading branches cover extraction and sampling only. Check the geometric matcher loaded by the final task handler before interpreting its scores as trained-matcher results. [Checkpoint logic](armbench/ID.py) · [Task handlers](armbench/engine.py).

## Acknowledgments

This repository is built using the [VGGT](https://github.com/facebookresearch/vggt), the [BEiT](https://github.com/microsoft/unilm/tree/master/beit), the [BEiTv2](https://github.com/microsoft/unilm/tree/master/beit2), the [BEiTv3](https://github.com/microsoft/unilm/tree/master/beit3), the [CLIP](https://github.com/openai/CLIP), the [open_clip](https://github.com/mlfoundations/open_clip), the [Oscar](https://github.com/microsoft/Oscar), the [DeiT](https://github.com/facebookresearch/deit) repository, the [timm](https://github.com/rwightman/pytorch-image-models) library and many other inspiring works in the community.


The source code was released in September 2025. The original release links Microsoft's [Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct). The default branch has no root license file.
