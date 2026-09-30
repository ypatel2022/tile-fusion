#!/bin/bash
set -euo pipefail

if [ "$#" -lt 2 ] || [ "$#" -gt 4 ]; then
    echo "Usage: $0 <benchmark-executable> <log-folder> [runs=100] [suite=ratios|tridiagonal]" >&2
    exit 1
fi

BINLIB=$1
LOGS=$2
RUNS=${3:-100}
SUITE=${4:-ratios}
if [ ! -x "$BINLIB" ]; then
    echo "Benchmark executable not found: $BINLIB" >&2
    exit 1
fi
case "$SUITE" in
    ratios) PERCENTAGES=(0 25 50 75 100) ;;
    tridiagonal) PERCENTAGES=(tridiagonal) ;;
    *) echo "Unknown suite: $SUITE" >&2; exit 1 ;;
esac

mkdir -p "$LOGS"
for file in "$LOGS"/spmm_spmm_graph_*.csv; do
    if [ -e "$file" ]; then
        echo "Benchmark CSVs already exist in $LOGS; choose a new log folder." >&2
        exit 1
    fi
done
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

for order in 0 1 2; do
    for index in "${!PERCENTAGES[@]}"; do
        percent=${PERCENTAGES[$(((index + 2 * order) % ${#PERCENTAGES[@]}))]}
        args=(32 "$RUNS" "$order")
        if [ "$percent" != tridiagonal ]; then
            args+=("$percent")
        fi
        echo "Running $percent, process order $order"
        "$BINLIB" "${args[@]}" > "$LOGS/spmm_spmm_graph_${percent}_${order}.csv"
    done
done
