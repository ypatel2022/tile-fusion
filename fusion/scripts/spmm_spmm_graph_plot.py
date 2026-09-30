import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METHODS = {
    "direct": "Direct (wait after each kernel)",
    "direct_same_stream": "Same-stream direct (wait after pair)",
    "graph": "CUDA Graph (wait after pair)",
}
COLORS = {"direct": "#4C72B0", "direct_same_stream": "#DD8452", "graph": "#55A868"}
CASE_COLUMNS = ["Matrix Name", "nRows", "NNZ", "bCols",
                "Target Fused Percent", "Fused Ratio"]


def read_process_medians(log_folder):
    files = sorted(log_folder.glob("spmm_spmm_graph_*.csv"))
    if not files:
        raise ValueError(f"No spmm_spmm_graph_*.csv files in {log_folder}")
    data = pd.concat([pd.read_csv(path) for path in files], ignore_index=True)
    medians = []
    for _, row in data.iterrows():
        trials = int(row["Number of Trials"])
        warmup = int(row["Warmup Trials"])
        if not 0 <= warmup < trials:
            raise ValueError(f"Invalid trial counts for {row['Matrix Name']}")
        times = row[[f"Trial{i} Subregion0 Executor" for i in range(trials)]]
        times = times.to_numpy(dtype=float)
        correct = row[[f"Correct{i}" for i in range(trials)]].to_numpy(dtype=float)
        errors = row[[f"Error{i}" for i in range(trials)]].to_numpy(dtype=float)
        if not (np.all(correct == 1) and np.all(np.isfinite(errors))):
            raise ValueError(f"Failed output verification for {row['Matrix Name']}")
        if not (np.all(np.isfinite(times)) and np.all(times > 0)):
            raise ValueError(f"Invalid GPU timing for {row['Matrix Name']}")
        medians.append(np.median(times[warmup:]) * 1e6)
    data = data[CASE_COLUMNS + ["Process Order", "Implementation Name",
                                "Number of Trials", "Warmup Trials"]].copy()
    data["gpu_us"] = medians
    return data


def summarize(data):
    records = []
    for case, rows in data.groupby(CASE_COLUMNS, dropna=False):
        if len(rows[["Number of Trials", "Warmup Trials"]].drop_duplicates()) != 1:
            raise ValueError(f"Inconsistent trial counts for {case[0]}")
        if rows.duplicated(["Process Order", "Implementation Name"]).any():
            raise ValueError(f"Duplicate process/method results for {case[0]}")
        times = rows.pivot(index="Process Order", columns="Implementation Name",
                           values="gpu_us")
        if (set(times.index) != {0, 1, 2} or set(times.columns) != set(METHODS)
                or times.isna().any().any()):
            raise ValueError(f"Expected all three methods and process orders for {case[0]}")
        matrix, size, nnz, features, percent, fused_ratio = case
        record = {"matrix": matrix, "rows": int(size), "nnz": int(nnz),
                  "features": int(features), "target_fused_percent": int(percent),
                  "fused_ratio": fused_ratio}
        for method in METHODS:
            record[f"{method}_us"] = times[method].median()
        # Pair ratios within each process before combining independent repeats.
        for baseline, label in [("direct", "direct"), ("direct_same_stream", "same_stream")]:
            ratios = times[baseline] / times["graph"]
            record[f"graph_vs_{label}"] = ratios.median()
            record[f"graph_vs_{label}_min"] = ratios.min()
            record[f"graph_vs_{label}_max"] = ratios.max()
        records.append(record)
    return pd.DataFrame(records).sort_values(["features", "target_fused_percent", "rows"])


def plot_results(summary, log_folder):
    groups = list(summary.groupby(["features", "target_fused_percent"], sort=True))
    columns = min(3, len(groups))
    rows = (len(groups) + columns - 1) // columns
    for kind in ["latency", "speedup"]:
        fig, axes = plt.subplots(rows, columns, figsize=(5 * columns, 3.8 * rows),
                                 squeeze=False, sharey=True)
        for ax, ((features, percent), data) in zip(axes.flat, groups):
            data = data.sort_values("rows")
            sizes = data["rows"].to_numpy()
            if kind == "latency":
                for method, label in METHODS.items():
                    ax.plot(sizes, data[f"{method}_us"], "o-", label=label,
                            color=COLORS[method], markersize=4)
                ax.set_yscale("log")
                ax.set_ylabel("Time per SpMM pair (µs)")
            else:
                for baseline, label in [("direct", "direct"),
                                        ("direct_same_stream", "same_stream")]:
                    speedup = data[f"graph_vs_{label}"].to_numpy()
                    low = data[f"graph_vs_{label}_min"].to_numpy()
                    high = data[f"graph_vs_{label}_max"].to_numpy()
                    ax.errorbar(sizes, speedup, yerr=[speedup - low, high - speedup],
                                fmt="o-", capsize=3, markersize=4,
                                label=("Graph vs direct" if baseline == "direct" else
                                       "Graph vs same-stream direct"),
                                color=COLORS[baseline])
                ax.axhline(1, color="black", linestyle="--", linewidth=1)
                ax.set_ylabel("Baseline time / graph time (×)")
            title = "Original tridiagonal" if percent == -1 else f"{percent:g}% fused rows"
            ax.set_title(f"{title} · {features:g} features")
            ax.set_xscale("log")
            ax.set_xticks(sizes)
            ax.set_xticklabels([f"{int(size):,}" for size in sizes], rotation=30, ha="right")
            ax.set_xlabel("Sparse rows / columns")
            ax.grid(alpha=0.25)
            ax.spines[["top", "right"]].set_visible(False)
        for ax in list(axes.flat)[len(groups):]:
            ax.set_visible(False)
        handles, labels = axes.flat[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.93),
                   ncol=1 if columns == 1 else len(labels), frameon=False)
        title = ("Tile-fused SpMM–SpMM execution time" if kind == "latency" else
                 "CUDA Graph speedup — above 1× means graph is faster")
        fig.suptitle(title, fontsize=14)
        note = ("Lower is better. Median of three process medians;\n"
                "warmup excluded." if kind == "latency" else
                "Median paired speedups; bars show the three-process min–max range.\n"
                "These ranges are not confidence intervals.")
        fig.text(0.5, 0.015, note, ha="center", fontsize=9, wrap=True)
        fig.tight_layout(rect=(0, 0.06, 1, 0.83 if columns == 1 else 0.88))
        for extension in ["png", "pdf"]:
            fig.savefig(log_folder / f"spmm_spmm_graph_{kind}.{extension}",
                        dpi=180, bbox_inches="tight")
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Summarize and plot SpMM-SpMM CUDA Graph CSVs.")
    parser.add_argument("log_folder", type=Path)
    args = parser.parse_args()
    summary = summarize(read_process_medians(args.log_folder))
    summary.to_csv(args.log_folder / "summary.csv", index=False)
    columns = ["rows", "features", "target_fused_percent", "direct_us",
               "direct_same_stream_us", "graph_us", "graph_vs_direct", "graph_vs_same_stream"]
    print(summary[columns].to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print("Speedup = baseline time / graph time; above 1 means graph is faster.")
    plot_results(summary, args.log_folder)
    print(f"Saved summary.csv and latency/speedup PNG and PDF figures in {args.log_folder}")


if __name__ == "__main__":
    main()
