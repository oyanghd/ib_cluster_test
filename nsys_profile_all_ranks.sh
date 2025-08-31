#!/bin/bash

set -e

cmd=$@
prog=$(basename "$1")
Rval=$(echo "$@" | grep -oP "(?<=-R )\d+" || echo "NA")

outfile="${prog}_R${Rval}_rank${OMPI_COMM_WORLD_RANK}"

echo "[nsys_profile_rank.sh] profiling ${prog} with -R ${Rval}, rank=${OMPI_COMM_WORLD_RANK}"

nsys profile -t cuda,nvtx -s none --cpuctxsw none --python-sampling true --python-sampling-frequency 1000 --force-overwrite true -o ${outfile} ${cmd} || true
