#!/bin/bash
#SBATCH -J X-TMR
#SBATCH -p cvr
#SBATCH -N 1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:hgx:4
#SBATCH --mem 300GB

source activate temos

# python -m train --cfg configs/config_vae_motionx_versionhuamnml_smplx212.yaml --cfg_assets configs/assets.yaml --batch_size 512 --nodebug
# python -m train --cfg configs/config_vae_motionx_version1_smplx212.yaml --cfg_assets configs/assets.yaml --batch_size 128 --nodebug
# python -m train --cfg configs/config_vae_motionx_version1_smplx212_no_joint.yaml --cfg_assets configs/assets.yaml --nodebug
# python -m train --cfg configs/config_vae_motionx_versionhumanml_vector263.yaml --cfg_assets configs/assets.yaml --batch_size 256 --nodebug
# python -m train --cfg configs/configs_temos/config_temos_kit_1e_4.yaml --cfg_assets configs/assets.yaml --nodebug
# python -m train --cfg configs/configs_temos/config_temos_motionx_1e-5_infonce_multi_gpu_nce1e-1_layer4_head6_wo-kl.yaml --cfg_assets configs/assets.yaml --nodebug
# python -m train --cfg configs/configs_temos/config_temos_humanml3d_1e-5_woinfonce_multi_gpu_nce1e-1_layer6_head6-newmetrics_neg256-norecliploss-V2-norm.yaml --cfg_assets configs/assets.yaml --nodebug
# python -m train --cfg configs/configs_temos/config_temos_humanml3d_1e-5_woinfonce_multi_gpu_nce1e-1_layer6_head6-newmetrics_neg256-recliploss_pretrain-V2-norm.yaml --cfg_assets configs/assets.yaml --nodebug
# python -m train --cfg configs/configs_temos/config_temos_unimocap_1e-5_infonce_multi_gpu_nce1e-1_layer6_head6-V2.yaml --cfg_assets configs/assets.yaml --nodebug
python -m train --cfg configs/configs_temos/config_temos_humanml3d_1e-5_infonce_multi_gpu_nce1e-1_layer4_head6_wo-kl.yaml --cfg_assets configs/assets.yaml --nodebug