#!/bin/bash
#SBATCH -t 2-00:00:00
#SBATCH -p public
#SBATCH -q public
#SBATCH -o /scratch/echang32/2D_Adamv2/finetune/full_transfer/drishti_gs/log_new.out
#SBATCH --cpus-per-task=10 
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=80G

module load mamba/latest
source activate LabEnv

python /home/echang32/2D_Adamv2/official_finetune/main.py --output_dir /scratch/echang32/2D_Adamv2/finetune/ --weights /home/echang32/2D_Adamv2/weights/pretrain/resnet50_fundus.pth --task segmentation --mode full_transfer --dataset drishti_gs --datasets_config ../datasets/datasets_config.yaml --pretrain adam --epochs 200 --patience 50 --trials 3 --batch_size_per_gpu 1 --lr 1e-3 --workers 10 --attempt Official_new