#!/bin/bash
#SBATCH -t 2-00:00:00
#SBATCH -p general
#SBATCH -q public
#SBATCH -o /scratch/echang32/2D_Adamv2/finetune/fewshot_5/siim_acr_ptx/log.out
#SBATCH --cpus-per-task=16 
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=80G

module load mamba/latest
source activate LabEnv

python /home/echang32/2D_Adamv2/finetune/main.py --output_dir /scratch/echang32/2D_Adamv2/finetune/ --weights /scratch/echang32/2D_Adamv2/pretrain/adamv2/checkpoint.pth --task segmentation --mode fewshot_5 --dataset siim_acr_ptx --datasets_config ../datasets_config.yaml --epochs 200 --patience 50 --trials 10 --batch_size_per_gpu 16 --workers 10 --attempt Official
