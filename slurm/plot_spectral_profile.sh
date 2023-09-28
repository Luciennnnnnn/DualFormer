#!/bin/bash
#SBATCH -o /home/sist/luoxin/projects/DualFormer/slurm_logs/job.%j.out
#SBATCH -p dongliu # 分区
#SBATCH -J plot_spectral_profile

#SBATCH --nodes=1 # 节点数
#SBATCH --ntasks=1 # 任务数
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G # 内存大小
#SBATCH --gres=gpu:a100:1

nvidia-smi

python /home/sist/luoxin/projects/DualFormer/scripts/plot_spectral_profile.py