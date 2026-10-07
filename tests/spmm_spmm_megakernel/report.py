#!/usr/bin/env python3
"""Summarize the fixed sweep using paired process medians."""
import argparse
import csv
import json
import math
from pathlib import Path
from statistics import median

BASELINES = ["original_waited", "direct_same_stream", "graph", "unfused_direct",
             "unfused_graph", "cusparse_alg2_direct", "cusparse_alg2_graph",
             "cusparse_alg3_direct", "cusparse_alg3_graph"]
CANDIDATES = ["megakernel_barrier", "megakernel_events_global", "megakernel_events_shared"]
CONDITION = ["Rows", "Family", "Structure", "Tile Rows", "Regime"]
MEMORY = ["Slots", "Calls", "CSR Bytes", "Dense Bytes", "Workspace Bytes", "X Pool Bytes",
          "L2 Bytes", "X Bytes", "H Bytes", "Y Bytes", "Schedule Bytes", "Event Bytes",
          "Library Workspace Bytes", "Global H Write Bytes per Call", "H Fully Written"]
SETUP = ["Device Malloc Seconds", "Initial Copy Seconds", "Inspection and Schedule Seconds",
         "Library Preparation Seconds", "Graph Preparation Seconds"]
REFERENCES = ["graph", "direct_same_stream", "unfused_graph", "selected_vendor_graph"]
PHASES = {"steady": 7, "resident_end_to_end": 7, "cold_end_to_end": 1}


def finite(text):
    return math.isfinite(float(text))


def write_csv(path, rows, fields):
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", type=Path, nargs="+", help="run_suite output directories")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    records, failures, expected, process_status, manifests = {}, [], set(), {}, []
    for directory in args.runs:
        manifest = json.loads((directory / "manifest.json").read_text())
        manifests.append(manifest)
        if (directory / "INVALID-EVALUATION-CHANGED.txt").exists():
            raise SystemExit("Invalid evaluation: " + str(directory))
        names = BASELINES if manifest["phase"] == "baseline-validation" else BASELINES + CANDIDATES
        if manifest["method_filter"] != "all":
            names = [manifest["method_filter"]]
        for n in manifest["rows"]:
            for family, structure in manifest["structures"]:
                for tile in manifest["tile_rows"]:
                    for regime in manifest["regimes"]:
                        if regime != "rotating" or n >= 4096:
                            condition = tuple(map(str, [n, family, structure, tile, regime]))
                            for phase in PHASES:
                                for name in names:
                                    expected.add((phase, *condition, name))
        with (directory / "run_status.csv").open() as stream:
            for status in csv.DictReader(stream):
                path = directory / status["file"]
                if int(status["exit_code"]):
                    failures.append({"source": str(path), "method": "process", "status": status["exit_code"]})
                if not path.exists():
                    failures.append({"source": str(path), "method": "process", "status": "missing file"})
                    continue
                with path.open() as raw:
                    for row in csv.DictReader(raw):
                        condition = tuple(row[field] for field in CONDITION)
                        key = (row["Phase"], *condition, row["Method"])
                        process = int(row["Process Order"])
                        identity = (*condition, row["Method"], process)
                        valid = row["Correct"] == "1" and row["Status"] == "ok" and finite(row["Max Error"])
                        process_status[identity] = process_status.get(identity, True) and valid
                        if not valid:
                            failures.append({"source": str(path), "method": row["Method"], "status": row["Status"]})
                        records.setdefault((key, process), []).append(row)
    hashes = {json.dumps(m["evaluation_sha256"], sort_keys=True) for m in manifests}
    if len(hashes) != 1:
        raise SystemExit("Runs use different evaluation files; analyze them separately.")
    commits = {m["implementation_commit"] for m in manifests}
    if len(commits) != 1:
        raise SystemExit("Runs use different implementation commits; analyze them separately.")

    processes = {}
    for (key, process), rows in records.items():
        phase, *condition, name = key
        if phase not in PHASES:
            continue
        identity = (*condition, name, process)
        complete = len(rows) == PHASES[phase] and len({r["Trial"] for r in rows}) == PHASES[phase]
        valid = process_status[identity] and complete
        if not valid:
            continue
        processes[(key, process)] = {
            "device": median(float(r["Device Seconds"]) for r in rows),
            "wall": median(float(r["Wall Seconds"]) for r in rows),
            "setup": median(float(r["Setup Seconds"]) for r in rows),
            "allocation": median(float(r["Allocation Seconds"]) for r in rows),
            "preparation": median(float(r["Preparation Seconds"]) for r in rows),
            "error": max(float(r["Max Error"]) for r in rows), "row": rows[0],
        }

    summaries = []
    for key in sorted(expected):
        phase, *condition, name = key
        group = {p: processes[(key, p)] for p in [0, 1, 2] if (key, p) in processes}
        row = dict(zip(["Phase", *CONDITION, "Method"], key))
        row.update({"Correct Processes": len(group), "Status": "complete" if len(group) == 3 else "incomplete"})
        # Incomplete rows remain visible, but carry no aggregate or ranking.
        if len(group) == 3:
            for metric in ["device", "wall", "setup", "allocation", "preparation"]:
                values = [x[metric] for x in group.values()]
                row[metric + " seconds"] = median(values)
                row[metric + " min"] = min(values)
                row[metric + " max"] = max(values)
            row["Max Error"] = max(x["error"] for x in group.values())
            for field in MEMORY:
                values = {x["row"][field] for x in group.values()}
                row[field] = next(iter(values)) if len(values) == 1 else "varies"
            for field in SETUP:
                row[field] = median(float(x["row"][field]) for x in group.values())
            first = next(iter(group.values()))["row"]
            slots = int(first["Slots"])
            row["CSR + dense bytes per slot"] = int(first["CSR Bytes"]) + int(first["Dense Bytes"]) // slots
            row["Allocated pool bytes"] = int(first["CSR Bytes"]) + int(first["Dense Bytes"]) + int(first["Workspace Bytes"])
            for reference in REFERENCES:
                pairs = []
                setup_differences = []
                for p, sample in group.items():
                    labels = [reference] if reference != "selected_vendor_graph" else ["cusparse_alg2_graph", "cusparse_alg3_graph"]
                    options = [processes.get(((phase, *condition, label), p)) for label in labels]
                    if any(x is None for x in options):
                        break
                    metric = "wall" if phase == "cold_end_to_end" else "device"
                    baseline = min(options, key=lambda x: x[metric])
                    if sample[metric] <= 0:
                        break
                    pairs.append(baseline[metric] / sample[metric])
                    saving = baseline[metric] - sample[metric]
                    extra_setup = sample["setup"] - baseline["setup"]
                    setup_differences.append(max(0.0, extra_setup) / saving if saving > 0 else math.inf)
                if len(pairs) == 3:
                    row["speedup vs " + reference] = median(pairs)
                    row["speedup min vs " + reference] = min(pairs)
                    row["speedup max vs " + reference] = max(pairs)
                    if phase == "steady" and all(math.isfinite(x) for x in setup_differences):
                        row["extra setup breakeven calls vs " + reference] = math.ceil(median(setup_differences))
        summaries.append(row)
    fields = ["Phase", *CONDITION, "Method", "Correct Processes", "Status"]
    fields += [f"{metric} {suffix}" for metric in ["device", "wall", "setup", "allocation", "preparation"]
               for suffix in ["seconds", "min", "max"]]
    fields += ["Max Error", *MEMORY, *SETUP, "CSR + dense bytes per slot", "Allocated pool bytes"]
    fields += [prefix + reference for reference in REFERENCES
               for prefix in ["speedup vs ", "speedup min vs ", "speedup max vs ", "extra setup breakeven calls vs "]]
    write_csv(args.output / "summary.csv", summaries, fields)
    write_csv(args.output / "failures.csv", failures, ["source", "method", "status"])
    (args.output / "provenance.json").write_text(json.dumps({"runs": list(map(str, args.runs)), "manifests": manifests}, indent=2) + "\n")
    complete = sum(row["Status"] == "complete" for row in summaries)
    text = (f"# SpMM–SpMM sweep\n\n{complete}/{len(summaries)} method/condition/phase rows have three correct processes. "
            f"{len(failures)} recorded failures. Full results are in `summary.csv`; failures remain in `failures.csv`.\n\n"
            "Times are medians of process medians. Ranges are the observed process minimum and maximum, not confidence intervals. "
            "Speedups pair process orders. `selected_vendor_graph` selects the faster ALG2/ALG3 graph per process and is labeled tuning. "
            "Incomplete rows have no aggregate or ranking.\n\n"
            "Steady timings include submission and completion. Resident end-to-end adds X upload and Y readback. "
            "Cold end-to-end includes the entire buffer ring allocation/copy/preparation and first output readback, excluding input generation and the CPU oracle. "
            "Setup totals combine allocation/initial copies and method preparation. Component columns distinguish device allocation, copying, inspection/schedule upload, library preparation and graph preparation. "
            "Extra-setup break-even estimates divide additional setup by steady time saved and exclude transfers and other application work.\n\n"
            "Memory columns describe allocated device arrays. Graph/runtime internal allocations are not available from these APIs and are excluded. "
            "Allocation size does not establish cache hits; profiler measurements are separate.\n")
    (args.output / "RESULTS.md").write_text(text)
    print(f"{complete}/{len(summaries)} complete rows; {len(failures)} recorded failures")


if __name__ == "__main__":
    main()
