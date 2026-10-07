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
for file in "$LOGS"/spmm_spmm_cpu_*.csv "$LOGS/run_status.csv"; do
    if [ -e "$file" ]; then
        echo "Benchmark files already exist in $LOGS; choose a new log folder." >&2
        exit 1
    fi
done
export OMP_DYNAMIC=FALSE MKL_DYNAMIC=FALSE
printf 'file,device,threads,process_order,suite,rows,exit_code\n' > "$LOGS/run_status.csv"
failed=0
thread_counts=(1 4 32)
for order in 0 1 2; do
    for index in "${!thread_counts[@]}"; do
        threads=${thread_counts[$(((index + order) % ${#thread_counts[@]}))]}
        export OMP_NUM_THREADS=$threads MKL_NUM_THREADS=$threads
        stem="spmm_spmm_cpu_${threads}_${order}"
        echo "Running $threads CPU threads, process order $order"
        status=0
        "$BINLIB" "$threads" "$RUNS" "$order" "$ROWS" > "$LOGS/$stem.csv" 2> "$LOGS/$stem.err" || status=$?
        printf '%s,CPU,%s,%s,structured,%s,%s\n' "$stem.csv" "$threads" "$order" "$ROWS" "$status" >> "$LOGS/run_status.csv"
        if [ "$status" -ne 0 ]; then
            echo "$stem exited with status $status; preserved CSV and stderr." >&2
            failed=1
        fi
    done
done
exit "$failed"
