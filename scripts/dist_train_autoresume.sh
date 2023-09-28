#!/usr/bin/env bash

GPUS=$1
CONFIG=$2

# usage
if [ $# -lt 2 ] ;then
    echo "usage:"
    echo "./scripts/dist_train.sh [number of gpu] [path to option file]"
    exit
fi

if [ -n "${PBS_JOBID+set}" ]; then
  default_port=$(echo $PBS_JOBID | cut -d. -f1)
  default_port=${default_port: -4}
  default_port=$((default_port))
  default_port=$((default_port + 10000))
else
  if [ -n "${SLURM_JOB_ID+set}" ]; then
    default_port=$SLURM_JOB_ID
    default_port=${default_port: -4}
    default_port=$((default_port))
    default_port=$((default_port + 10000))
  else
    default_port=4321
  fi
fi

echo $default_port

PYTHONPATH="$(dirname $0)/..:${PYTHONPATH}" \
torchrun --nproc_per_node=$GPUS --master_port=$default_port \
    basicsr/train.py -opt $CONFIG --auto_resume --launcher pytorch ${@:3}

# python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
#     basicsr/train.py -opt $CONFIG --auto_resume --launcher pytorch ${@:3}