#!/usr/bin/env bash
# find_same_switch_gpu.sh hostfile
# Parse ibnetdiscover output and find two nodes with the same GPU index
# connected to the same switch. Generate a new hostfile with only those two.

hostfile=$1
ibout=$(mktemp)

# Discover the full InfiniBand fabric topology
ibnetdiscover > $ibout

# Read all nodes from hostfile (OpenMPI format: "node slots=8")
nodes=($(awk '{print $1}' $hostfile))

declare -A mapping   # key = switch_id:gpu  val = "nodeX nodeY ..."

current_switch=""
gpu=""
hca=""

while read -r line; do
    if [[ $line =~ ^Ca ]]; then
        # Example: "Ca      : 0x248a070300eb6f00 ports 1"
        continue
    elif [[ $line =~ ^\ \ 0 ]]; then
        # Example: "  0   1   0x248a070300eb6f00 1 [mlx5_0/1]"
        iface=$(echo $line | grep -o '\[.*\]' | tr -d '[]')
        gpu=$(echo $iface | grep -o '[0-9]*$')   # mlx5_0 -> 0
        hca=$iface
    elif [[ $line =~ ^Switch ]]; then
        # Example: "Switch  : 0x508f9a0300a3f6c0 ports 36"
        current_switch=$(echo $line | awk '{print $3}')
        key="$current_switch:$gpu"

        # Try to map iface to node name from hostfile
        for n in "${nodes[@]}"; do
            if [[ $hca == *$n* ]] || [[ $n == $(hostname) ]]; then
                mapping[$key]="${mapping[$key]} $n"
            fi
        done
    fi
done < $ibout

# Select the first pair of nodes with same GPU index on the same switch
pair_nodes=()
for key in "${!mapping[@]}"; do
    arr=(${mapping[$key]})
    if [ ${#arr[@]} -gt 1 ]; then
        pair_nodes=("${arr[0]}" "${arr[1]}")
        break
    fi
done

if [ ${#pair_nodes[@]} -eq 2 ]; then
    echo "Found pair: ${pair_nodes[0]} and ${pair_nodes[1]}"
    out_file="hostfile_no_cross_track"
    echo "${pair_nodes[0]} slots=8" > $out_file
    echo "${pair_nodes[1]} slots=8" >> $out_file
    echo "Generated $out_file"
else
    echo "No suitable pair of nodes found."
fi
