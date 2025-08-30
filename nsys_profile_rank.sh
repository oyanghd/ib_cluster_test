#!/bin/bash

set -e

if [ $OMPI_COMM_WORLD_RANK == 3 ]; then
    nsys profile -t cuda,nvtx -s none --cpuctxsw none --python-sampling true --python-sampling-frequency 1000 $@ || true
else
    $@ || true
fi