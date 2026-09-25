"""Run every declared case, retain raw SWBench CSVs, and summarize all results."""
import argparse
import csv
import json
import math
from pathlib import Path
import statistics
import subprocess

MODES = ("unfused_mkl", "unfused_avx2", "tile_fused_avx2")


def read_result(path, trials):
    with path.open() as source:
        rows = list(csv.DictReader(source))
    if len(rows) != 3 or {r["Implementation Name"] for r in rows} != set(MODES):
        raise ValueError(f"Missing benchmark modes: {path}")
    result = {}
    for row in rows:
        if None in row or int(row["Number of Trials"]) != trials:
            raise ValueError(f"Malformed CSV: {path}")
        samples = []
        for trial in range(trials):
            if row[f"Correct{trial}"] != "1" or not math.isfinite(float(row[f"Error{trial}"])):
                raise ValueError(f"Correctness failure: {path}")
            time = float(row[f"Trial{trial} Subregion0 Executor"])
            if not math.isfinite(time) or time <= 0:
                raise ValueError(f"Invalid timing: {path}")
            samples.append(time)
        result[row["Implementation Name"]] = {
            "median": statistics.median(samples),
            "inspection": float(row["Trial0 Subregion0 Inspection"]),
            "fused_fraction": float(row["FusedRows0"]) / float(row["Nodes0"]),
            "nodes": int(float(row["Nodes0"])),
            "nnz": int(float(row["Nonzeros0"])),
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("executable", type=Path)
    parser.add_argument("inputs", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--threads", type=int, nargs="+", default=[1, 8])
    parser.add_argument("--widths", type=int, nargs="+", default=[64, 128])
    parser.add_argument("--cache-bytes", type=int, nargs="+", default=[1048576])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--trials", type=int, default=7)
    parser.add_argument("--warmups", type=int, default=3)
    args = parser.parse_args()
    if min(args.threads + args.widths + args.cache_bytes + [args.repeats, args.trials, args.warmups]) <= 0:
        parser.error("All counts and dimensions must be positive")
    if args.output.exists():
        parser.error("Use a new output directory to preserve earlier measurements")
    args.output.mkdir(parents=True)
    cases = ["local_65536", "local_262144", "shuffled_65536", "pubmed"]
    manifest = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    manifest["cases"] = cases
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    summary = []
    for case in cases:
        for width in args.widths:
            for threads in args.threads:
                for cache in args.cache_bytes:
                    runs = []
                    for repeat in range(args.repeats):
                        name = f"{case}-w{width}-t{threads}-c{cache}-r{repeat + 1}"
                        command = [str(args.executable.resolve()), str((args.inputs / f"{case}.mtx").resolve()),
                                   str(width), str(width), str(threads), str(cache),
                                   str(args.trials), str(args.warmups), str(repeat % 3 + 1)]
                        print(name, flush=True)
                        with (args.output / f"{name}.csv").open("w") as out, (args.output / f"{name}.err").open("w") as err:
                            subprocess.run(command, stdout=out, stderr=err, check=True)
                        runs.append(read_result(args.output / f"{name}.csv", args.trials))
                    medians = {mode: statistics.median(run[mode]["median"] for run in runs) for mode in MODES}
                    inspection = statistics.median(run["tile_fused_avx2"]["inspection"] for run in runs)
                    saving = medians["unfused_avx2"] - medians["tile_fused_avx2"]
                    ratios = [run["unfused_avx2"]["median"] / run["tile_fused_avx2"]["median"] for run in runs]
                    summary.append(dict(case=case, width=width, threads=threads, cache_bytes=cache,
                                        nodes=runs[0]["tile_fused_avx2"]["nodes"],
                                        nnz=runs[0]["tile_fused_avx2"]["nnz"],
                                        **{f"{mode}_ms": medians[mode] * 1000 for mode in MODES},
                                        speedup_vs_avx2=medians["unfused_avx2"] / medians["tile_fused_avx2"],
                                        speedup_vs_mkl=medians["unfused_mkl"] / medians["tile_fused_avx2"],
                                        min_repeat_speedup=min(ratios), max_repeat_speedup=max(ratios),
                                        inspection_ms=inspection * 1000,
                                        break_even_reuses=math.ceil(inspection / saving) if saving > 0 else "never",
                                        fused_fraction=runs[0]["tile_fused_avx2"]["fused_fraction"]))
    with (args.output / "summary.csv").open("w") as output:
        writer = csv.DictWriter(output, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    report = ["# GNN linear-layer benchmark results", "",
              "Operation: H = X W, Y = normalized(A) H; float32 CPU kernels. "
              "Features and weights are deterministic synthetic data, including for PubMed. "
              "This measures the linear operations, not full GNN training.", "",
              f"{args.repeats} independent processes per case; each has {args.warmups} warmups and "
              f"{args.trials} timed trials per implementation. Implementation order rotates between processes. "
              "Times below are medians of process medians; speedup > 1 favors fusion. "
              "Every warmup and trial checks the full output against MKL, whose reference also "
              "receives a scalar double-precision check on 16 sampled rows.", "",
              "| Graph | Width | Threads | Cache KiB | MKL ms | Unfused AVX2 ms | Fused ms | Speedup vs AVX2 | Repeat range | Inspection ms | Break-even reuses | Fused rows |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in summary:
        report.append(f"| {r['case']} | {r['width']} | {r['threads']} | {r['cache_bytes'] / 1024:g} "
                      f"| {r['unfused_mkl_ms']:.3f} | {r['unfused_avx2_ms']:.3f} | {r['tile_fused_avx2_ms']:.3f} "
                      f"| {r['speedup_vs_avx2']:.2f}x | {r['min_repeat_speedup']:.2f}–{r['max_repeat_speedup']:.2f}x "
                      f"| {r['inspection_ms']:.3f} | {r['break_even_reuses']} | {r['fused_fraction']:.1%} |")
    report += ["", "Inspection is excluded from execution. Break-even is ceil(inspection / "
               "(unfused AVX2 execution − fused execution)); it estimates how many uses of the same "
               "graph repay scheduling, assuming these execution times persist. Input loading, allocation, "
               "normalization, buffer clearing, warmup, and verification are outside execution timing. "
               "Buffers are reused. Cache sizes are explicit inspector budgets, not measured cache occupancy.", "",
               "The ordered synthetic graph is deliberately favorable to locality. The shuffled case "
               "has identical topology with relabeled vertices. PubMed uses its bundled node ordering, "
               "undirected edges, and self-loops. All declared cases, including slowdowns, are included.", "",
               "This comparison includes changes in GEMM tiling and threading as well as fusion. "
               "A speedup with zero fused rows is not evidence of inter-operation data reuse.", ""]
    (args.output / "RESULTS.md").write_text("\n".join(report))
    print("\n".join(report), flush=True)


if __name__ == "__main__":
    main()
