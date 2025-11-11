#!/bin/bash
#SBATCH -t 1-00:00:00
#SBATCH -p public
#SBATCH -q public
#SBATCH -o /scratch/echang32/2D_Adamv2/finetune/full_transfer/shenzhen/log_new.out
#SBATCH --cpus-per-task=16 
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64G

module load mamba/latest
source activate LabEnv

python /home/echang32/2D_Adamv2/official_finetune/main.py --output_dir /scratch/echang32/2D_Adamv2/finetune/ --weights /home/echang32/2D_Adamv2/weights/pretrain/resnet50_cxr.pth --task classification --mode full_transfer --use_fp16 True --dataset shenzhen --datasets_config ../datasets/datasets_config.yaml --epochs 200 --patience 50 --trials 10 --batch_size_per_gpu 16 --lr 2.5e-4 --first_beta 0.90 --second_beta 0.95 --weight_decay 5e-2 --workers 10 --attempt Official_new
