#!/bin/bash

set -e

if [ $OMPI_COMM_WORLD_RANK == 3 ]; then
    cmd=$@
    prog=$(basename "$1")

    echo "NSYS_OUTFILE before check: $NSYS_OUTFILE"
    if [ -z "$NSYS_OUTFILE" ]; then
        NSYS_OUTFILE="NODES_${OMPI_COMM_WORLD_SIZE}_rank${OMPI_COMM_WORLD_RANK}"
    fi

    echo "[nsys_profile_rank.sh] num nodes ${OMPI_COMM_WORLD_SIZE} profiling ${prog} rank=${OMPI_COMM_WORLD_RANK} outfile=${NSYS_OUTFILE}"

    nsys profile -t cuda,nvtx -s none --cpuctxsw none --python-sampling true --python-sampling-frequency 1000 --force-overwrite true -o ${NSYS_OUTFILE} ${cmd} || true
else
    $@ || true
fi