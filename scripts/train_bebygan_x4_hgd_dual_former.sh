# !/bin/bash
SHELL_FOLDER=$(cd "$(dirname "$0")";pwd)
cd $(dirname $SHELL_FOLDER)

# cd ../

# if ! pip list | grep basicsr >/dev/null
# then
#     echo "basicsr is not installed, installing it..."
pip install --user -e .
# fi

CUDA_VISIBLE_DEVICES=0,1,2,3 \
scripts/dist_train_autoresume.sh 4 options/train/train_bebygan_x4_hgd_dual_former.yml