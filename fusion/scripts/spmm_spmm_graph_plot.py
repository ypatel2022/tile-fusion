import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


LEGACY_METHODS = {
    "direct": "Direct (wait after each kernel)",
    "direct_same_stream": "Same-stream direct (wait after pair)",
    "graph": "CUDA Graph (wait after pair)",
}
GPU_METHODS = {
    "unfused_direct": "Unfused direct",
    "unfused_graph": "Unfused graph",
    "cusparse_alg2_direct": "cuSPARSE ALG2 direct",
    "cusparse_alg2_graph": "cuSPARSE ALG2 graph",
    "cusparse_alg3_direct": "cuSPARSE ALG3 direct",
    "cusparse_alg3_graph": "cuSPARSE ALG3 graph",
    "direct_same_stream": "Tile-fused direct",
    "graph": "Tile-fused graph",
}
CPU_METHODS = {
    "cpu_unfused": "Ordinary parallel SpMM",
    "cpu_mkl": "MKL",
    "cpu_avx2_unfused": "Unfused AVX2",
    "cpu_avx2_fused": "Tile-fused AVX2",
}
CASE_COLUMNS = ["Matrix Name", "nRows", "NNZ", "bCols", "Target Fused Percent",
                "Device", "Number of Threads", "Matrix Family", "Structure Size"]
SIZES = [64, 512, 4096, 32768, 262144, 1048576]
STRUCTURES = [("banded", 3), ("banded", 5), ("banded", 9),
              ("block_diagonal", 4), ("block_diagonal", 16)]


def methods_for(device, family):
    if device == "CPU":
        return CPU_METHODS
    return GPU_METHODS if family in {"banded", "block_diagonal"} else LEGACY_METHODS


def read_process_medians(log_folder):
    files = sorted(log_folder.glob("spmm_spmm_graph_*.csv"))
    files += sorted(log_folder.glob("spmm_spmm_cpu_*.csv"))
    status_path = log_folder / "run_status.csv"
    statuses = pd.read_csv(status_path) if status_path.exists() else pd.DataFrame()
    if not files and statuses.empty:
        raise ValueError(f"No benchmark CSVs in {log_folder}")
    records, issues = [], []
    for path in files:
        try:
            data = pd.read_csv(path)
        except (pd.errors.EmptyDataError, pd.errors.ParserError) as error:
            issues.append({"file": str(path), "status": f"unreadable CSV: {error}"})
            continue
        for _, row in data.iterrows():
            record = row.to_dict()
            # Archived native CSVs predate the structured metadata columns.
            record.setdefault("Device", "GPU")
            record["Device"] = str(record["Device"]).upper()
            record.setdefault("Matrix Family", "tridiagonal" if row["Target Fused Percent"] == -1 else "rewired")
            record.setdefault("Structure Size", -1)
            record.setdefault("Tile Eligible Ratio", row["Fused Ratio"])
            record["file"] = str(path)
            required = CASE_COLUMNS + ["Process Order", "Implementation Name",
                                       "Number of Trials", "Warmup Trials"]
            if any(pd.isna(record.get(key)) for key in required):
                issues.append({"file": str(path), "status": "incomplete CSV row"})
                continue
            record["executor_us"] = np.nan
            record["status"] = "ok"
            try:
                trials, warmup = int(row["Number of Trials"]), int(row["Warmup Trials"])
                if not 0 <= warmup < trials:
                    raise ValueError("invalid trial counts")
                times = row[[f"Trial{i} Subregion0 Executor" for i in range(trials)]].to_numpy(dtype=float)
                correct = row[[f"Correct{i}" for i in range(trials)]].to_numpy(dtype=float)
                errors = row[[f"Error{i}" for i in range(trials)]].to_numpy(dtype=float)
                if not (np.all(correct == 1) and np.all(np.isfinite(errors))):
                    raise ValueError("failed output verification")
                if not (np.all(np.isfinite(times)) and np.all(times > 0)):
                    raise ValueError("invalid executor timing")
                record["executor_us"] = np.median(times[warmup:]) * 1e6
            except (KeyError, TypeError, ValueError) as error:
                record["status"] = str(error)
            records.append(record)
    for _, row in statuses.iterrows():
        path = log_folder / row["file"]
        if int(row["exit_code"]) != 0 or not path.exists():
            issues.append({"file": str(path), "Device": row["device"],
                           "Number of Threads": row["threads"], "Process Order": row["process_order"],
                           "status": f"process exit {row['exit_code']}" if path.exists() else "missing CSV"})
    result = pd.DataFrame(records)
    result.attrs["issues"] = issues
    result.attrs["statuses"] = statuses
    # Retain the old field for scripts that inspect archived process medians.
    if not result.empty:
        result["gpu_us"] = result["executor_us"]
    return result


def comparisons_for(device, family):
    if device == "CPU":
        return [("cpu_unfused", "cpu_avx2_fused", "fusion_vs_unfused"),
                ("cpu_mkl", "cpu_avx2_fused", "fusion_vs_mkl"),
                ("cpu_avx2_unfused", "cpu_avx2_fused", "fusion_vs_avx2")]
    if family not in {"banded", "block_diagonal"}:
        return [("direct", "graph", "graph_vs_direct"),
                ("direct_same_stream", "graph", "graph_vs_same_stream")]
    pairs = [("unfused_direct", "unfused_graph", "unfused_graph_vs_direct"),
             ("cusparse_alg2_direct", "cusparse_alg2_graph", "alg2_graph_vs_direct"),
             ("cusparse_alg3_direct", "cusparse_alg3_graph", "alg3_graph_vs_direct"),
             ("direct_same_stream", "graph", "graph_vs_same_stream")]
    for mode in ["direct", "graph"]:
        fused = "direct_same_stream" if mode == "direct" else "graph"
        for baseline in ["unfused", "cusparse_alg2", "cusparse_alg3"]:
            pairs.append((f"{baseline}_{mode}", fused, f"fusion_{mode}_vs_{baseline}"))
    return pairs


def expected_cases(data):
    cases = [] if data.empty else data[CASE_COLUMNS].drop_duplicates().to_dict("records")
    known = {(case["Matrix Name"], case["Device"], int(case["Number of Threads"])) for case in cases}
    for _, run in data.attrs["statuses"].iterrows():
        if run["suite"] != "structured":
            continue
        sizes = SIZES if str(run["rows"]) == "all" else [int(run["rows"])]
        device, threads = str(run["device"]).upper(), int(run["threads"])
        for size in sizes:
            for family, structure in STRUCTURES:
                name = f"{family}_{structure}_{size}"
                if (name, device, threads) in known:
                    continue
                nnz = (size * structure - (structure // 2) * (structure // 2 + 1)
                       if family == "banded" else size * structure)
                cases.append(dict(zip(CASE_COLUMNS, [name, size, nnz, 32, -1,
                                                     device, threads, family, structure])))
                known.add((name, device, threads))
    return cases


def summarize(data):
    records, issues = [], list(data.attrs["issues"])
    for case in expected_cases(data):
        rows = data
        for key, value in case.items():
            if not rows.empty:
                rows = rows[rows[key] == value]
        device, family = case["Device"], case["Matrix Family"]
        methods = methods_for(device, family)
        record = {"matrix": case["Matrix Name"], "rows": int(case["nRows"]),
                  "nnz": int(case["NNZ"]), "features": int(case["bCols"]),
                  "target_fused_percent": int(case["Target Fused Percent"]),
                  "device": device, "threads": int(case["Number of Threads"]),
                  "family": family, "structure_size": int(case["Structure Size"])}
        record["fused_ratio"] = rows["Fused Ratio"].max() if not rows.empty else np.nan
        record["tile_eligible_ratio"] = rows["Tile Eligible Ratio"].iloc[0] if not rows.empty else np.nan
        valid = {}
        for method in methods:
            method_rows = rows[rows["Implementation Name"] == method] if not rows.empty else rows
            values = {}
            for order in [0, 1, 2]:
                process = method_rows[method_rows["Process Order"] == order] if not method_rows.empty else method_rows
                status = "missing result" if process.empty else "duplicate result" if len(process) != 1 else process.iloc[0]["status"]
                if status == "ok":
                    values[order] = process.iloc[0]["executor_us"]
                else:
                    issues.append({"file": process.iloc[0]["file"] if not process.empty else "",
                                   "Matrix Name": case["Matrix Name"], "Device": device,
                                   "Number of Threads": record["threads"], "Implementation Name": method,
                                   "Process Order": order, "status": status})
            valid[method] = pd.Series(values, dtype=float)
            record[f"{method}_valid_processes"] = len(values)
            # Incomplete methods remain in the raw table, but do not enter plots or rankings.
            record[f"{method}_us"] = valid[method].median() if len(values) == 3 else np.nan
        for baseline, candidate, label in comparisons_for(device, family):
            ratios = valid[baseline] / valid[candidate]
            complete = len(valid[baseline]) == len(valid[candidate]) == 3
            record[label] = ratios.median() if complete else np.nan
            record[f"{label}_min"] = ratios.min() if complete else np.nan
            record[f"{label}_max"] = ratios.max() if complete else np.nan
        records.append(record)
    summary = pd.DataFrame(records)
    summary.attrs["issues"] = issues
    return summary.sort_values(["device", "threads", "features", "target_fused_percent",
                                "family", "structure_size", "rows"]) if not summary.empty else summary


def plot_panels(summary, folder, filename, title, series, speedup=False):
    groups = list(summary.groupby(["features", "family", "structure_size", "target_fused_percent"], sort=True))
    if not groups:
        return
    columns = min(3, len(groups))
    rows = (len(groups) + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(5 * columns, 3.8 * rows), squeeze=False)
    handles = {}
    for ax, ((features, family, structure, percent), data) in zip(axes.flat, groups):
        data = data.sort_values("rows")
        sizes = data["rows"].to_numpy()
        for index, (key, label) in enumerate(series.items()):
            if key not in data:
                continue
            values = data[key].to_numpy(dtype=float)
            if not np.isfinite(values).any():
                continue
            color = plt.get_cmap("tab10")(index)
            if speedup:
                low, high = data[f"{key}_min"].to_numpy(), data[f"{key}_max"].to_numpy()
                line = ax.errorbar(sizes, values, yerr=[values - low, high - values], fmt="o-",
                                   color=color, capsize=3, markersize=4, label=label)
            else:
                line, = ax.plot(sizes, values, "o-", color=color, markersize=4, label=label)
            handles[label] = line
        if speedup:
            ax.axhline(1, color="black", linestyle="--", linewidth=1)
            ax.set_ylabel("Baseline time / candidate time (×)")
        else:
            ax.set_yscale("log")
            ax.set_ylabel("Time per SpMM pair (µs)")
        if family == "banded":
            name = f"Banded · {structure} diagonals"
        elif family == "block_diagonal":
            name = f"Block diagonal · {structure} rows/block"
        else:
            name = "Original tridiagonal" if percent == -1 else f"{percent:g}% fused rows"
        ax.set_title(f"{name} · {features:g} features")
        ax.set_xscale("log")
        ax.set_xticks(sizes)
        ax.set_xticklabels([f"{int(size):,}" for size in sizes], rotation=30, ha="right")
        ax.set_xlabel("Sparse rows / columns")
        ax.grid(alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
    for ax in list(axes.flat)[len(groups):]:
        ax.set_visible(False)
    fig.legend(list(handles.values()), list(handles), loc="upper center", bbox_to_anchor=(0.5, 0.945),
               ncol=min(3, max(1, len(handles))), frameon=False, fontsize=9)
    fig.suptitle(title, fontsize=14)
    note = ("Median paired speedups; bars show observed process min–max, not confidence intervals."
            if speedup else "Median of three process medians; warmup excluded. Lower is better.")
    fig.text(0.5, 0.01, note + "\nMethods with incomplete or incorrect results are omitted; see failed_results.csv.",
             ha="center", fontsize=8)
    fig.tight_layout(rect=(0, 0.07, 1, 0.83))
    for extension in ["png", "pdf"]:
        fig.savefig(folder / f"{filename}.{extension}", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_results(summary, folder):
    for (device, threads), rows in summary.groupby(["device", "threads"]):
        if device == "CPU":
            plot_panels(rows, folder, f"spmm_spmm_cpu_latency_threads{threads}",
                        f"CPU execution · {threads} threads", {f"{key}_us": label for key, label in CPU_METHODS.items()})
            plot_panels(rows, folder, f"spmm_spmm_cpu_fusion_threads{threads}",
                        f"CPU tile-fusion speedup · {threads} threads",
                        {"fusion_vs_unfused": "Fusion vs ordinary SpMM", "fusion_vs_mkl": "Fusion vs MKL",
                         "fusion_vs_avx2": "Fusion vs unfused AVX2"}, speedup=True)
            continue
        legacy = rows[~rows["family"].isin(["banded", "block_diagonal"])]
        if not legacy.empty:
            plot_panels(legacy, folder, "spmm_spmm_graph_latency", "Tile-fused SpMM–SpMM execution time",
                        {f"{key}_us": label for key, label in LEGACY_METHODS.items()})
            plot_panels(legacy, folder, "spmm_spmm_graph_speedup", "CUDA Graph speedup",
                        {"graph_vs_direct": "Graph vs direct", "graph_vs_same_stream": "Graph vs same-stream direct"}, True)
        structured = rows[rows["family"].isin(["banded", "block_diagonal"])]
        if structured.empty:
            continue
        plot_panels(structured, folder, "spmm_spmm_structured_gpu_latency", "GPU SpMM–SpMM execution time",
                    {f"{key}_us": label for key, label in GPU_METHODS.items()})
        plot_panels(structured, folder, "spmm_spmm_structured_graph_gain", "CUDA Graph replay speedup",
                    {"unfused_graph_vs_direct": "Unfused", "alg2_graph_vs_direct": "cuSPARSE ALG2",
                     "alg3_graph_vs_direct": "cuSPARSE ALG3", "graph_vs_same_stream": "Tile-fused"}, True)
        for mode in ["direct", "graph"]:
            plot_panels(structured, folder, f"spmm_spmm_structured_fusion_{mode}", f"GPU tile-fusion speedup · {mode}",
                        {f"fusion_{mode}_vs_unfused": "Fusion vs ordinary SpMM",
                         f"fusion_{mode}_vs_cusparse_alg2": "Fusion vs cuSPARSE ALG2",
                         f"fusion_{mode}_vs_cusparse_alg3": "Fusion vs cuSPARSE ALG3"}, True)


def plot_overview(gpu, cpu, folder):
    joined = pd.concat([gpu, cpu], ignore_index=True)
    records = []
    for _, case in joined.iterrows():
        methods = methods_for(case["device"], case["family"])
        for method, label in methods.items():
            if f"{method}_us" not in case:
                continue
            records.append({"matrix": case["matrix"], "rows": case["rows"], "nnz": case["nnz"],
                            "features": case["features"], "family": case["family"],
                            "structure_size": case["structure_size"], "device": case["device"],
                            "threads": case["threads"], "method": method, "method_label": label,
                            "executor_us": case[f"{method}_us"]})
    pd.DataFrame(records).to_csv(folder / "cpu_gpu_latency.csv", index=False)
    # Plot the same named methods for every case.
    view = gpu.copy()
    series = {"graph_us": "GPU tile-fused graph", "unfused_graph_us": "GPU unfused graph",
              "cusparse_alg2_graph_us": "GPU cuSPARSE ALG2 graph", "cusparse_alg3_graph_us": "GPU cuSPARSE ALG3 graph"}
    keys = ["matrix", "rows", "nnz", "features", "family", "structure_size"]
    for threads in [1, 4, 32]:
        columns = keys + ["cpu_avx2_fused_us", "cpu_mkl_us"]
        selected = cpu[cpu["threads"] == threads][columns].rename(columns={
            "cpu_avx2_fused_us": f"cpu_fused_{threads}_us", "cpu_mkl_us": f"cpu_mkl_{threads}_us"})
        view = view.merge(selected, on=keys, how="left", validate="one_to_one")
        series[f"cpu_fused_{threads}_us"] = f"CPU tile-fused · {threads} threads"
        series[f"cpu_mkl_{threads}_us"] = f"CPU MKL · {threads} threads"
    plot_panels(view, folder, "spmm_spmm_cpu_gpu_latency", "CPU/GPU executor latency · transfers and setup excluded", series)


def write_summary(folder):
    data = read_process_medians(folder)
    summary = summarize(data)
    data.to_csv(folder / "process_medians.csv", index=False)
    pd.DataFrame(summary.attrs["issues"], columns=["file", "Matrix Name", "Device", "Number of Threads",
                                                  "Implementation Name", "Process Order", "status"]).to_csv(folder / "failed_results.csv", index=False)
    summary.to_csv(folder / "summary.csv", index=False)
    if not summary.empty:
        plot_results(summary, folder)
    failures = len(summary.attrs["issues"])
    print(f"{folder}: {len(summary)} case/configuration summaries; {failures} failure or missing-result records.")
    print("Speedups require three valid paired processes. Raw CSVs are preserved.")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Summarize native CPU/GPU SpMM–SpMM benchmark CSVs.")
    parser.add_argument("log_folder", type=Path)
    parser.add_argument("--cpu-log-folder", type=Path, help="Also summarize a CPU sweep and compare executor latencies.")
    args = parser.parse_args()
    summary = write_summary(args.log_folder)
    if args.cpu_log_folder:
        cpu = write_summary(args.cpu_log_folder)
        if not summary.empty and not cpu.empty:
            gpu = summary[summary["device"] == "GPU"]
            if not gpu.empty:
                plot_overview(gpu, cpu[cpu["device"] == "CPU"], args.log_folder)
    print("Saved summary.csv, process_medians.csv, failed_results.csv and available PNG/PDF figures.")


if __name__ == "__main__":
    main()
