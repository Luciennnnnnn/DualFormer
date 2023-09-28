# !/bin/bash
SHELL_FOLDER=$(cd "$(dirname "$0")";pwd)
cd $(dirname $SHELL_FOLDER)

python scripts/generate_realesrgan_dataset.py --input datasets/DIV2K/DIV2K_valid_HR --hr_folder datasets/DIV2K/RealESRGAN/LR/X4 --lr_folder datasets/DIV2K/RealESRGAN/HR/X4 --scale 4