#!/bin/bash

set -e

if [ $OMPI_COMM_WORLD_RANK == 0 ]; then
    cmd=$@
    prog=$(basename "$1")
    Rval=$NCCL_NVLS_ENABLE

    outfile="${prog}_NODES_${OMPI_COMM_WORLD_SIZE}_NVLS_AND_SHARP_ENABLE_${Rval}_rank${OMPI_COMM_WORLD_RANK}"

    echo "[nsys_profile_rank.sh] num nodes ${OMPI_COMM_WORLD_SIZE} profiling ${prog} with NVLS and SHARP ENABLE ${Rval}, rank=${OMPI_COMM_WORLD_RANK}"

    nsys profile -t cuda,nvtx -s none --cpuctxsw none --python-sampling true --python-sampling-frequency 1000 --force-overwrite true -o ${outfile} ${cmd} || true
else
    $@ || true
fi