#!/bin/bash

path1="/comp_robot/chenlinghao/LSL/experiments/temos/config_temos_motionx_1e-5_infonce_multi_gpu_nce1e-1_layer4_head6_wo-kl/embeddings/val/epoch_99/"
path2="/comp_robot/chenlinghao/LSL/experiments/temos/config_temos_motionx_1e-5_infonce_multi_gpu_nce1e-1_layer4_head6_wo-kl/embeddings/val/epoch_599/"
path3="/comp_robot/chenlinghao/LSL/experiments/temos/config_temos_motionx_1e-5_infonce_multi_gpu_nce1e-1_layer4_head6_wo-kl/embeddings/val/epoch_999/"


for protocal in A B D
do
    echo "**protocal" $protocal"**"
    for retrieval_type in T2M M2T
    do
        echo $retrieval_type
        python retrieval.py --retrieval_type $retrieval_type --protocal $protocal --expdirs $path1 $path2 $path3 
    done
done
