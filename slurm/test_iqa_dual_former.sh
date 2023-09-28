#!/bin/bash
#SBATCH -o /home/sist/luoxin/projects/DualFormer/slurm_logs/job.%j.out
#SBATCH -p dongliu # 分区
#SBATCH -J test_iqa_dual_former

#SBATCH --nodes=1 # 节点数
#SBATCH --ntasks=1 # 任务数
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G # 内存大小
#SBATCH --gres=gpu:a100:1

nvidia-smi

bash /home/sist/luoxin/projects/DualFormer/scripts/test/test_iqa_dual_former.sh