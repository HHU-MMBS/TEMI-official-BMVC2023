# Official code for paper Exploring the Limits of Deep Image Clustering using Pretrained Models
This is the official code to reproduce our results as published in [BMVC2023](https://bmvc2023.org/)

[![arXiv](https://img.shields.io/badge/YouTube-red)](https://www.youtube.com/watch?v=Z-1HpVcjzYM) [![arXiv](https://img.shields.io/badge/BMVC-2023-blue)](https://proceedings.bmvc2023.org/297/) [![arXiv](https://img.shields.io/badge/arXiv-2303.17896-red)](https://arxiv.org/abs/2303.17896)

## Set up instructions

#### Install virtual env with clip via:
```
conda create -n temi python=3.7
conda activate temi

pip install -r requirements.txt
```

#### Available model names
```
dino_resnet50, dino_vits16, dino_vitb16, timm_resnet50,timm_vit_small_patch16_224, timm_vit_base_patch16_224 timm_vit_large_patch16_224, convnext_small, convnext_base, convnext_large, msn_vit_small, msn_vit_base msn_vit_large, mocov3_vit_small, mocov3_vit_base, clip_ViT-B/16, clip_ViT-L/14, clip_RN50, mae_vit_base, mae_vit_large, mae_vit_huge
```
#### Available dataset names (IN1K and its subsets need the imagenet (IN1K) path to be passed with `--datapath` where `./data` is used by default):
```
CIFAR10, CIFAR100, STL10, CIFAR20, IN50, IN100, IN200, IN1K
```

## How to generate image embeddings for different models and datasets
```
python gen_embeds.py --arch clip_ViT-B/32 --dataset CIFAR10 --batch_size 256
```


## TEMI: how to train the head and evaluate cl
```
export CUDA_VISIBLE_DEVICES=0; outdir=$"./experiments/TEMI-output-test" ; clusters=10 ; dataset=$"CIFAR10"; 
python train_main.py  --precomputed --arch clip_ViT-B/32  --batch_size=1024 --use_fp16=false --max_momentum_teacher=0.996 \
--lr=1e-4 --warmup_epochs=20 --min_lr=1e-4 --epochs=100 --output_dir $outdir --dataset $dataset  --knn=50 
--out_dim=$clusters  --num_heads=16 --loss TEMI --loss-args  beta=0.6 \

python eval_experiment.py --ckpt_folder $outdir 
```


## Overclustering experiments on IN1K
Don't forget to generate the image embeddings first and fix the imagenet paths (`--datapth`).
```
dataset=$"IN1K"; clusters=$25000 ; model=dino_vitb16; head=$16; knn=$25; 
echo "clusters:"  $clusters "dataset:" $dataset "heads" $head "knn-pairs" $knn "model" $model 
outdir=$"./experiments/overclustering/$indist-$model/"
python train_main.py --disable_ddp --precomputed --embed_norm --arch $model \
--batch_size=128 --use_fp16=false --max_momentum_teacher=0.996 \
--lr=1e-4 --warmup_epochs=20 --min_lr=1e-4 --epochs=100 \
--output_dir $outdir --dataset $dataset  \
--knn=$knn --out_dim=$clusters  --num_heads=$head \
--loss TEMI  --loss-args beta=$beta  \
                        
python eval_experiment.py --ckpt_folder $outdir
```





## How to cite our work
```bibtex
@inproceedings{Adaloglou_2023_BMVC,
author    = {Nikolas Adaloglou and Felix Michels and Hamza Kalisch and Markus Kollmann},
title     = {Exploring the Limits of Deep Image Clustering using Pretrained Models},
booktitle = {34th British Machine Vision Conference 2023, {BMVC} 2023, Aberdeen, UK, November 20-24, 2023},
publisher = {BMVA},
year      = {2023},
url       = {https://papers.bmvc2023.org/0297.pdf}
}
```

## Licence and credits
The codebase was developed based on FAIR's [DINO repository](https://github.com/facebookresearch/dino), which has an Apache License 2.0.
For the clustering evaluations, we used the function from [SSCN](https://github.com/elad-amrani/self-classifier)


### Linear Probing
```
python linear_evaluation.py --arch=clip_ViT-B/32 --dataset CIFAR10
```

### K-means baseline
Note: Multiple architectures can be passed in `--archs`
```
python baseline_kmeans.py --dataset CIFAR10 --archs clip_ViT-B/32
```

