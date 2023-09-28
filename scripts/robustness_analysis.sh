# !/bin/bash
SHELL_FOLDER=$(cd "$(dirname "$0")";pwd)
cd $(dirname $SHELL_FOLDER)

# spatial transformer
python scripts/robustness_analysis_under_frequency_perturbation.py \
--experiment_name pretrained_models/Spatial_Transformer \
--net spatial_transformer

# spectral transformer
python scripts/robustness_analysis_under_frequency_perturbation.py \
--experiment_name pretrained_models/Spectral_Transformer \
--net spectral_transformer

# spatial mlpmixer
python scripts/robustness_analysis_under_frequency_perturbation.py \
--experiment_name pretrained_models/Spatial_MLPMixer \
--net spatial_mlpmixer

# spectral mlpmixer
python scripts/robustness_analysis_under_frequency_perturbation.py \
--experiment_name pretrained_models/Spectral_MLPMixer \
--net spectral_mlpmixer