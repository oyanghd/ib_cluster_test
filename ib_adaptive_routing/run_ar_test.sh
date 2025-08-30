#!/bin/bash
# Usage: bash run_ar_test.sh

# Need set 
# for open Adaptive Routing
# NCCL_IB_AR_THRESHOLD=8192
# for close Adaptive Routing
# NCCL_IB_AR_THRESHOLD=1073741824
# ref https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#nccl-ib-ar-threshold
PROFILE=1 bash ../oyhd_dist_ib_launch.sh python3 simple_ar_test.py
