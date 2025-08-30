#!/bin/bash
# Usage: bash ib_sharp_performance_test.sh hostfile
HOSTFILE=$1
if [[ -z "$HOSTFILE" ]]; then
    echo "Usage: $0 <hostfile>"
    exit 1
fi

NODES=($(awk '{print $1}' "$HOSTFILE"))

NCCL_TESTS_PATH=/data/konghaoran/nccl-tests/build
OUT_DIR="./nccl_logs"
mkdir -p $OUT_DIR

POWERS=(2 4 8 16 32 64 125)

OPS=("all_reduce_perf" "all_gather_perf" "reduce_scatter_perf")
NVLS_AND_SHARP_ENABLE=(0 1)

PROFILE=${PROFILE:-0} # nsys profile
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)

if [ $PROFILE == 1 ]; then PROFILE_WRAPPER="$SCRIPT_DIR/../nsys_profile_rank0.sh";
else PROFILE_WRAPPER=; fi

for op in "${OPS[@]}"; do
    for R in "${NVLS_AND_SHARP_ENABLE[@]}"; do
        for NUM_NODES in "${POWERS[@]}"; do
            if (( NUM_NODES > ${#NODES[@]} )); then
                break
            fi

            # 取前 NUM_NODES 节点
            HOSTS=("${NODES[@]:0:$NUM_NODES}")

            # 生成 host 参数
            HOST_ARG=$(printf "%s:1," "${HOSTS[@]}")
            HOST_ARG=${HOST_ARG%,}  # 去掉最后的逗号

            log_profile="${OUT_DIR}/${op}_NVLS_AND_SHARP_ENABLE_${R}_profile0.log"
            profile_out_file="${op}_NVLS_AND_SHARP_ENABLE_${R}"

            echo "=================================================="
            echo "Running NCCL SHARP test on $NUM_NODES nodes..."
            echo "Hosts: $HOST_ARG"

            mpirun -np $NUM_NODES -host $HOST_ARG \
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
                -x NCCL_SHARP_ENABLE=$R \
                -x NCCL_NVLS_ENABLE=$R \
                $NCCL_TESTS_PATH/${op} -b 8 -e 16G -f 2 -g 1 -R 0 \
                > $log_profile 2>&1
            
            mpirun -np $NUM_NODES -host $HOST_ARG \
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
                -x NCCL_SHARP_ENABLE=$R \
                -x NCCL_NVLS_ENABLE=$R \
                $PROFILE_WRAPPER \
                -o $profile_out_file \
                $NCCL_TESTS_PATH/${op} -b 8 -e 16G -f 2 -g 1 -R 0
        done
    done
done