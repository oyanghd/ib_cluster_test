#!/bin/bash
# NCCL test script for two nodes with output redirected
# Usage: bash comm_size_div_n.sh

OPS=("all_reduce_perf" "all_gather_perf" "alltoall_perf" "reduce_scatter_perf")
LOCAL_REGISTERS=(0 1 2)

MIN_BYTES=2
MAX_BYTES=16G
STEP_FACTOR=2

NCCL_TESTS_PATH=/data/konghaoran/nccl-tests/build
OUT_DIR="./nccl_logs"
mkdir -p $OUT_DIR

for op in "${OPS[@]}"; do
    for R in "${LOCAL_REGISTERS[@]}"; do
        log_no_profile="${OUT_DIR}/${op}_R${R}_profile0.log"
        log_profile="${OUT_DIR}/${op}_R${R}_profile1.log"

        echo "=============================="
        echo " Running $op with -R $R (no profile) "
        echo "=============================="
        PROFILE=0 bash ../oyhd_dist_ib_launch_tiny.sh \
            /data/konghaoran/nccl-tests/build/${op} \
            -b $MIN_BYTES -e $MAX_BYTES -f $STEP_FACTOR -g 1 -R $R \
            > $log_no_profile 2>&1
        echo "Finished $op with -R $R (no profile)"
        echo "Log saved to $log_no_profile"
        echo "-----------------------------------"

        echo "=============================="
        echo " Running $op with -R $R (with profile) "
        echo "=============================="
        PROFILE=1 bash ../oyhd_dist_ib_launch_tiny.sh \
            ${NCCL_TESTS_PATH}/${op} \
            -b $MIN_BYTES -e $MAX_BYTES -f $STEP_FACTOR -g 1 -R $R \
            > $log_profile 2>&1
        echo "Finished $op with -R $R (with profile)"
        echo "Log saved to $log_profile"
        echo "-----------------------------------"
    done
done