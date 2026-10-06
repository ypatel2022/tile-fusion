#!/bin/bash
set -euo pipefail

if [ "$#" -lt 2 ] || [ "$#" -gt 6 ]; then
    echo "Usage: $0 <benchmark-executable> <log-folder> [runs=100] [suite=ratios|tridiagonal|structured] [rows=all] [tile-rows=4]" >&2
    exit 1
fi

BINLIB=$1
LOGS=$2
RUNS=${3:-100}
SUITE=${4:-ratios}
ROWS=${5:-all}
TILES=${6:-4}
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
if [ "$SUITE" != structured ] && { [ "$ROWS" != all ] || [ "$TILES" != 4 ]; }; then
    echo "Row and tile filters are available only for the structured suite." >&2
    exit 1
fi
IFS=, read -r -a tile_rows <<< "$TILES"
if [ "$SUITE" = structured ]; then
    if [[ ! "$TILES" =~ ^(1|4|16)(,(1|4|16))*$ ]]; then
        echo "Tile rows must be a comma-separated list of 1,4,16." >&2
        exit 1
    fi
    CASES=("${tile_rows[@]}")
    selected=,
    for tile in "${CASES[@]}"; do
        if [[ "$selected" == *",$tile,"* ]]; then
            echo "Tile rows must be distinct." >&2
            exit 1
        fi
        selected+="$tile,"
    done
fi

mkdir -p "$LOGS"
for file in "$LOGS"/spmm_spmm_graph_*.csv "$LOGS/run_status.csv"; do
    if [ -e "$file" ]; then
        echo "Benchmark files already exist in $LOGS; choose a new log folder." >&2
        exit 1
    fi
done
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
printf 'file,device,threads,process_order,suite,rows,exit_code,operation,tile_rows\n' > "$LOGS/run_status.csv"
failed=0
for order in 0 1 2; do
    for index in "${!CASES[@]}"; do
        matrix_case=${CASES[$(((index + 2 * order) % ${#CASES[@]}))]}
        if [ "$SUITE" = structured ]; then
            args=(--structured "$RUNS" "$order" "$ROWS" "$matrix_case")
            stem="spmm_spmm_graph_structured_tile${matrix_case}_${order}"
            tile=$matrix_case
        else
            args=(32 "$RUNS" "$order")
            if [ "$matrix_case" != tridiagonal ]; then
                args+=("$matrix_case")
            fi
            stem="spmm_spmm_graph_${matrix_case}_${order}"
            tile=4
        fi
        echo "Running $SUITE case $matrix_case, process order $order"
        status=0
        "$BINLIB" "${args[@]}" > "$LOGS/$stem.csv" 2> "$LOGS/$stem.err" || status=$?
        printf '%s,GPU,1,%s,%s,%s,%s,SpMMSpMM,%s\n' "$stem.csv" "$order" "$SUITE" "$ROWS" "$status" "$tile" >> "$LOGS/run_status.csv"
        if [ "$status" -ne 0 ]; then
            echo "$stem exited with status $status; preserved CSV and stderr." >&2
            failed=1
        fi
    done
done
exit "$failed"
