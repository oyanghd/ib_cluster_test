#!/bin/bash
# Usage: bash run_alltoall_with_nccl_bg_loop.sh

set -e

# ================== Configs ==================
ALL_HOSTFILE="/etc/mpi/hostfile"
NUM_CARDS_PER_NODE=8
ALLTOALL_SCOPE=(3 4 8 2 1)        # all-to-all group size
REPEAT=2                          # every scope repeat times
NCCL_TESTS_BIN="/data/konghaoran/nccl-tests/build/reduce_scatter_perf"
PYTHON_SCRIPT="test_2stage_alltoall.py"
LAUNCH_SCRIPT="../oyhd_dist_ib_launch.sh"
# =============================================

mapfile -t NODES < <(awk '{print $1}' $ALL_HOSTFILE | sort -u)
TOTAL_NODES=${#NODES[@]}

for SCOPE_SIZE in "${ALLTOALL_SCOPE[@]}"; do
    if [ $SCOPE_SIZE -gt $TOTAL_NODES ]; then
        echo "Skipping scope $SCOPE_SIZE: more than total nodes"
        continue
    fi

    for ((trial=1; trial<=REPEAT; trial++)); do
        echo "=== All-to-All Test: Scope=$SCOPE_SIZE, Trial=$trial ==="

        ALLTOALL_NODES=($(shuf -e "${NODES[@]}" -n $SCOPE_SIZE))
        echo "Selected all-to-all nodes: ${ALLTOALL_NODES[@]}"

        ALLTOALL_HOSTFILE="hostfile_alltoall"
        printf "%s\n" "${ALLTOALL_NODES[@]}" > $ALLTOALL_HOSTFILE

        BG_NODES=($(comm -23 <(printf "%s\n" "${NODES[@]}" | sort) <(printf "%s\n" "${ALLTOALL_NODES[@]}" | sort)))
        echo "Background nodes: ${BG_NODES[@]}"
        BG_HOSTFILE="hostfile_bg"
        printf "%s\n" "${BG_NODES[@]}" > $BG_HOSTFILE

        if [ ${#BG_NODES[@]} -gt 0 ]; then
            echo "Starting background all-reduce on nodes: ${BG_NODES[@]}"
            mpirun --allow-run-as-root -bind-to none \
                --hostfile $BG_HOSTFILE \
                -map-by slot \
                -mca pml ob1 \
                -mca btl ^openib \
                -mca plm_rsh_num_concurrent 300 \
                -mca routed_radix 600 \
                -mca plm_rsh_no_tree_spawn 1 \
                -x NCCL_SOCKET_IFNAME=eth0 \
                -x NCCL_PLUGIN_P2P=ucx \
                --map-by ppr:8:node \
                $NCCL_TESTS_BIN -b 4096M -e 4096M -g 1 -N 3000 \
                > bg_allreduce.out 2>&1 &
            BG_PID=$!
        fi

        sleep 20  # wait for background to stabilize

        export NSYS_OUTFILE="NODES_${SCOPE_SIZE}_trial${trial}"
        export HOSTFILE=$ALLTOALL_HOSTFILE
        PROFILE=1 bash $LAUNCH_SCRIPT python3 $PYTHON_SCRIPT > reduce_scatter_scope${SCOPE_SIZE}_trial${trial}.log 2>&1

        if [ ! -z "$BG_PID" ]; then
            echo "Killing background all-reduce process $BG_PID"
            kill $BG_PID || true
            wait $BG_PID 2>/dev/null || true
        fi

        sleep 20

        echo "=== Trial $trial for scope $SCOPE_SIZE completed ==="
    done
done
