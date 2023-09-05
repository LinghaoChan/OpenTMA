#!/bin/bash
#SBATCH -J Uni-TMR
#SBATCH -p cvr
#SBATCH -N 1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:rtx:4
#SBATCH --mem 80GB

source activate temos

# python -m train --cfg configs/configs_temos/MotionX-TMR.yaml --cfg_assets configs/assets.yaml --nodebug
python -m train --cfg configs/configs_temos/UniMocap-TMR.yaml --cfg_assets configs/assets.yaml --nodebug
# python -m train --cfg configs/configs_temos/H3D-TMR.yaml --cfg_assets configs/assets.yaml --nodebug