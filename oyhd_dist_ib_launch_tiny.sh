#!/bin/bash

sed 's/slots=8/slots=1/g' ./hostfile > hostfile_tiny
WORLD_SIZE=$(awk -F= '/slots=/ {sum += $2} END {print sum}' hostfile_tiny)

PROFILE=${PROFILE:-0} # nsys profile
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)

MASTER_ADDR=$(awk 'NR==1 {print $1}' hostfile_tiny)
MASTER_PORT=35766

# PROFILE_WRAPPER
if [ $PROFILE == 1 ]; then PROFILE_WRAPPER="$SCRIPT_DIR/nsys_profile_all_ranks.sh";
else PROFILE_WRAPPER=; fi
mpirun --allow-run-as-root -np $WORLD_SIZE \
    --hostfile hostfile_tiny \
    -bind-to none  -map-by slot \
    -mca pml ob1 \
    -mca btl ^openib \
    -mca plm_rsh_num_concurrent 300 \
    -mca routed_radix 600 \
    -mca plm_rsh_no_tree_spawn 1 \
    -x NCCL_IB_DISABLE=0 \
    -x NCCL_IB_GID_INDEX=0 \
    -x NCCL_IB_QPS_PER_CONNECTION=8 \
    -x NCCL_IB_TIMEOUT=22 \
    -x NCCL_DEBUG=WARN \
    -x NCCL_PLUGIN_P2P=ucx \
    -x NCCL_SOCKET_IFNAME=eth0 \
    -x NCCL_COLLNET_ENABLE=1 \
    -x NCCL_IB_AR_THRESHOLD=0 \
    -x PATH=./:$PATH \
    -x PYTHONPATH=./:$PYTHONPATH \
    -x MASTER_ADDR=$MASTER_ADDR \
    -x MASTER_PORT=$MASTER_PORT \
    -x CUDA_VISIBLE_DEVICES=0 \
    -x CUDA_DEVICE_MAX_CONNECTIONS=1 \
    $PROFILE_WRAPPER \
    ${@:1} \

# -x LD_PRELOAD=/nlp_group/ouyanghaodong/python_code/comm_compute_tests_1sm/nccl-kai_kccl_1sm_libnccl.so.2.27.5.1sm.fix2 \
# -x PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \

