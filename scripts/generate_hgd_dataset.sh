# !/bin/bash
SHELL_FOLDER=$(cd "$(dirname "$0")";pwd)
cd $(dirname $SHELL_FOLDER)

python scripts/generate_hgd_dataset.py \
--input datasets/DIV2K/DIV2K_valid_HR \
--hr_folder datasets/DIV2K/HD/HR/X4 \
--lr_folder datasets/DIV2K/HD/LR/X4 \
--scale 4

python scripts/generate_hgd_dataset.py \
--input datasets/Urban100/HR \
--hr_folder datasets/Urban100/HD/HR/X4 \
--lr_folder datasets/Urban100/HD/LR/X4 \
--scale 4

python scripts/generate_hgd_dataset.py \
--input datasets/B100/HR \
--hr_folder datasets/B100/HD/HR/X4 \
--lr_folder datasets/B100/HD/LR/X4 \
--scale 4
