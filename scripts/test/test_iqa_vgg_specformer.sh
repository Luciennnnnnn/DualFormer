# !/bin/bash
SHELL_FOLDER=$(cd "$(dirname "$0")";pwd)
cd $(dirname $SHELL_FOLDER)

test_sets=(
datasets/CLIVE/ # 500 * 500
datasets/KonIQ-10k/1024x768/
)

test_set_guiders=(
datasets/IQA_guider/CLIVE_all_pair_resize2img.txt
datasets/IQA_guider/KonIQ-10k_all_pair.txt
)

test_set_names=(
CLIVE
KonIQ-10k
)

experiments=(
pretrained_models/NRIQA_vgg_specformer_CLIVE_KonIQ-10k
pretrained_models/NRIQA_vgg_specformer_CLIVE_KonIQ-10k
)

for((i=0;i<${#experiments[@]};i++))
do
    for((j=0;j<${#test_sets[@]};j++))
    do
        python test_iqa.py --experiment_name ${experiments[i]} \
        --img_size 384 384 \
        --test_protocol_spatial random \
        --spectral_window_size 6 \
        --test_protocol_spectral random \
        --batch_size 20 \
        --test_set ${test_sets[j]} --test_set_guider ${test_set_guiders[j]} --test_set_name ${test_set_names[j]}
    done
done

##
test_sets=(
datasets/PIPAL/train
datasets/PIPAL/train
datasets/PIPAL/train
datasets/PIPAL/train # 288
)

test_set_guiders=(
datasets/IQA_guider/PIPAL_train_TradSR_pair.txt
datasets/IQA_guider/PIPAL_train_PsnrSR_pair.txt
datasets/IQA_guider/PIPAL_train_GANbSR_pair.txt
datasets/IQA_guider/PIPAL_train_AllSR_pair.txt
)

test_set_names=(
PIPAL_train_TradSR
PIPAL_train_PsnrSR
PIPAL_train_GANbSR
PIPAL_train
)

experiments=(
pretrained_models/NRIQA_vgg_specformer_PIPAL
)

for((i=0;i<${#experiments[@]};i++))
do
    for((j=0;j<${#test_sets[@]};j++))
    do
        python test_iqa.py --experiment_name ${experiments[i]} \
        --img_size 192 192 \
        --test_protocol_spatial random \
        --test_protocol_spatial patchify \
        --batch_size 20 \
        --test_set ${test_sets[j]} --test_set_guider ${test_set_guiders[j]} --test_set_name ${test_set_names[j]}
    done
done