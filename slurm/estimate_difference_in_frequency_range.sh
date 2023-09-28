#!/bin/bash
#SBATCH -o /home/sist/luoxin/projects/DualFormer/slurm_logs/job.%j.out
#SBATCH -p dongliu # 分区
#SBATCH -J estimate_difference_in_frequency_range

#SBATCH --nodes=1 # 节点数
#SBATCH --ntasks=1 # 任务数
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G # 内存大小
#SBATCH --gres=gpu:a100:1

nvidia-smi

python /home/sist/luoxin/projects/DualFormer/scripts/estimate_difference_in_frequency_range.py --model_path experiments/pretrained_models/RealESRNet_x4plus.pth

python /home/sist/luoxin/projects/DualFormer/scripts/estimate_difference_in_frequency_range.py --model_path experiments/pretrained_models/RealESRGAN_x4plus.pth
