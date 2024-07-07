#!/bin/bash


#Set job requirements
#SBATCH -n 1
#SBATCH -t 20:00:00
#SBATCH -p gpu
#SBATCH --gpus-per-node=1



source activate prompt

gpu_id=0
bz=64
lr=0.0001


python train_gps.py /gpfs/work5/0/prjs0370/zhizhang/dataset/FGVC \
        --dataset dogs \
        --num-classes 120 --simple-aug\
        --model vit_base_patch16_224_in21k \
        --epochs 100 \
        --batch-size $bz \
        --opt adam  --weight-decay 0 \
        --warmup-lr 1e-7 --warmup-epochs 10 \
        --lr $lr --min-lr 1e-8 \
        --drop-path 0 --img-size 224 \
        --model-ema --model-ema-decay 0.9998 \
        --output output/ \
        --amp --tuning-mode part --pretrained \
        --pruning --pruning_method gradient_perCell \
        --times_para 1 \
        --gpu_id $gpu_id \
        --log-wandb \
        --experiment fgvc \
        --run_name "" \
        --no-prefetcher \
        --contrast-aug \
        --contrastive
