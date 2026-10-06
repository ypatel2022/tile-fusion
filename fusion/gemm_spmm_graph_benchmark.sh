#!/bin/bash
set -euo pipefail

if [ "$#" -lt 2 ] || [ "$#" -gt 4 ]; then
    echo "Usage: $0 <benchmark-executable> <log-folder> [runs=100] [rows=all]" >&2
    exit 1
fi

BINLIB=$1
LOGS=$2
RUNS=${3:-100}
ROWS=${4:-all}
if [ ! -x "$BINLIB" ]; then
    echo "Benchmark executable not found: $BINLIB" >&2
    exit 1
fi
mkdir -p "$LOGS"
for file in "$LOGS"/gemm_spmm_graph_*.csv "$LOGS/run_status.csv"; do
    if [ -e "$file" ]; then
        echo "Benchmark files already exist in $LOGS; choose a new log folder." >&2
        exit 1
    fi
done
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
printf 'file,device,threads,process_order,suite,rows,exit_code,operation,tile_rows\n' > "$LOGS/run_status.csv"
failed=0
for order in 0 1 2; do
    stem="gemm_spmm_graph_${order}"
    echo "Running GEMM-SpMM, process order $order"
    status=0
    "$BINLIB" "$RUNS" "$order" "$ROWS" > "$LOGS/$stem.csv" 2> "$LOGS/$stem.err" || status=$?
    printf '%s,GPU,1,%s,structured,%s,%s,GEMMSpMM,0\n' "$stem.csv" "$order" "$ROWS" "$status" >> "$LOGS/run_status.csv"
    if [ "$status" -ne 0 ]; then
        echo "$stem exited with status $status; preserved CSV and stderr." >&2
        failed=1
    fi
done
exit "$failed"
