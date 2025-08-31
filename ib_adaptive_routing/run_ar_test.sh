#!/bin/bash
# Usage: bash run_ar_test.sh

# Need set 
# for open Adaptive Routing
# NCCL_IB_AR_THRESHOLD=8192
# for close Adaptive Routing
# NCCL_IB_AR_THRESHOLD=1073741824
# ref https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#nccl-ib-ar-threshold

export HOSTFILE=/data/konghaoran/ib_cluster_test/ib_adaptive_routing/hostfile_31_62

AR_THRESHOLD_OPTIONS=(0 8192 1073741824)

for threshold in ${AR_THRESHOLD_OPTIONS[@]}; do
    echo "Running test with NCCL_IB_AR_THRESHOLD=$threshold"
    export NCCL_IB_AR_THRESHOLD=$threshold
    export NSYS_OUTFILE="ar_test_threshold_${threshold}"
    PROFILE=1 bash ../oyhd_dist_ib_launch.sh python3 simple_ar_test.py > ar_test_threshold_${threshold}.log 2>&1
    echo "Completed test with NCCL_IB_AR_THRESHOLD=$threshold, log saved to ar_test_threshold_${threshold}.log"
done
