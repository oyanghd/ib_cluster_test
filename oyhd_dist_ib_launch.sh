#!/bin/bash

if [ -z "$HOSTFILE" ]; then
    HOSTFILE="/etc/mpi/hostfile"
fi

sed 's/slots=1/slots=8/g' "$HOSTFILE" > hostfile_dist
WORLD_SIZE=$(awk -F= '/slots=/ {sum += $2} END {print sum}' hostfile_dist)

PROFILE=${PROFILE:-0} # nsys profile
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)

MASTER_ADDR=$(awk 'NR==1 {print $1}' hostfile_dist)
MASTER_PORT=35788

# PROFILE_WRAPPER
if [ $PROFILE == 1 ]; then PROFILE_WRAPPER="$SCRIPT_DIR/nsys_profile_rank.sh";
else PROFILE_WRAPPER=; fi
mpirun --allow-run-as-root -np $WORLD_SIZE \
    --hostfile hostfile_dist \
    -bind-to none  -map-by slot \
    -mca pml ob1 \
    -mca btl ^openib \
    -mca plm_rsh_num_concurrent 300 \
    -mca routed_radix 600 \
    -mca plm_rsh_no_tree_spawn 1 \
    --map-by ppr:8:node \
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
    -x CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    -x CUDA_DEVICE_MAX_CONNECTIONS=1 \
    $PROFILE_WRAPPER \
    ${@:1} \

# -x LD_PRELOAD=/nlp_group/ouyanghaodong/python_code/comm_compute_tests_1sm/nccl-kai_kccl_1sm_libnccl.so.2.27.5.1sm.fix2 \
# -x PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \

