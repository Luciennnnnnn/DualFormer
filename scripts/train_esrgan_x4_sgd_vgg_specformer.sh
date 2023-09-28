# !/bin/bash
SHELL_FOLDER=$(cd "$(dirname "$0")";pwd)
cd $(dirname $SHELL_FOLDER)

# cd ../

# if ! pip list | grep basicsr >/dev/null
# then
#     echo "basicsr is not installed, installing it..."
pip install --user -e .
# fi

CUDA_VISIBLE_DEVICES=0,1 \
scripts/dist_train_autoresume.sh 2 options/train/train_esrgan_x4_sgd_vgg_specformer.yml