#!/bin/bash
set -euo pipefail

if [ "$#" -lt 2 ] || [ "$#" -gt 5 ]; then
    echo "Usage: $0 <benchmark-executable> <log-folder> [runs=100] [suite=ratios|tridiagonal|structured] [rows=all]" >&2
    exit 1
fi

BINLIB=$1
LOGS=$2
RUNS=${3:-100}
SUITE=${4:-ratios}
ROWS=${5:-all}
if [ ! -x "$BINLIB" ]; then
    echo "Benchmark executable not found: $BINLIB" >&2
    exit 1
fi
case "$SUITE" in
    ratios) CASES=(0 25 50 75 100) ;;
    tridiagonal) CASES=(tridiagonal) ;;
    structured) CASES=(structured) ;;
    *) echo "Unknown suite: $SUITE" >&2; exit 1 ;;
esac
if [ "$SUITE" != structured ] && [ "$ROWS" != all ]; then
    echo "The row filter is available only for the structured suite." >&2
    exit 1
fi

mkdir -p "$LOGS"
for file in "$LOGS"/spmm_spmm_graph_*.csv "$LOGS/run_status.csv"; do
    if [ -e "$file" ]; then
        echo "Benchmark files already exist in $LOGS; choose a new log folder." >&2
        exit 1
    fi
done
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
printf 'file,device,threads,process_order,suite,rows,exit_code\n' > "$LOGS/run_status.csv"
failed=0
for order in 0 1 2; do
    for index in "${!CASES[@]}"; do
        matrix_case=${CASES[$(((index + 2 * order) % ${#CASES[@]}))]}
        if [ "$SUITE" = structured ]; then
            args=(--structured "$RUNS" "$order" "$ROWS")
        else
            args=(32 "$RUNS" "$order")
            if [ "$matrix_case" != tridiagonal ]; then
                args+=("$matrix_case")
            fi
        fi
        stem="spmm_spmm_graph_${matrix_case}_${order}"
        echo "Running $matrix_case, process order $order"
        status=0
        "$BINLIB" "${args[@]}" > "$LOGS/$stem.csv" 2> "$LOGS/$stem.err" || status=$?
        printf '%s,GPU,1,%s,%s,%s,%s\n' "$stem.csv" "$order" "$SUITE" "$ROWS" "$status" >> "$LOGS/run_status.csv"
        if [ "$status" -ne 0 ]; then
            echo "$stem exited with status $status; preserved CSV and stderr." >&2
            failed=1
        fi
    done
done
exit "$failed"
