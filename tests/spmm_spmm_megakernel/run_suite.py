#!/usr/bin/env python3
"""Run the fixed GPU sweep and retain failed processes."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import random
import subprocess
import time

SIZES = [64, 512, 4096, 32768, 131072, 262144, 1048576]
STRUCTURES = [("banded", 3), ("banded", 5), ("banded", 9),
              ("block_diagonal", 4), ("block_diagonal", 16)]
FROZEN_FILES = ["evaluation.h", "evaluation.cu", "CMakeLists.txt", "EVALUATION.md",
                "run_suite.py", "report.py"]


def integers(text, allowed):
    values = allowed if text == "all" else [int(x) for x in text.split(",")]
    if not values or len(set(values)) != len(values) or any(x not in allowed for x in values):
        raise argparse.ArgumentTypeError("Select distinct values from " + str(allowed))
    return values


def git(directory, *arguments):
    result = subprocess.run(["git", "-C", str(directory), *arguments],
                            text=True, capture_output=True)
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("executable", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--rows", default="all")
    parser.add_argument("--tiles", default="all")
    parser.add_argument("--orders", default="all")
    parser.add_argument("--regimes", default="repeated,rotating")
    parser.add_argument("--method", default="all", help="Single method for validation/profiling, or all.")
    parser.add_argument("--phase", choices=["baseline-validation", "performance"], default="performance")
    args = parser.parse_args()
    rows = integers(args.rows, SIZES)
    tiles = integers(args.tiles, [1, 4, 16])
    orders = integers(args.orders, [0, 1, 2])
    regimes = args.regimes.split(",")
    if not regimes or len(set(regimes)) != len(regimes) or any(x not in {"repeated", "rotating"} for x in regimes):
        parser.error("Regimes must be repeated, rotating, or repeated,rotating.")
    executable = args.executable.resolve()
    if not executable.is_file():
        parser.error("Executable does not exist.")
    args.output.mkdir(parents=True, exist_ok=False)
    source = Path(__file__).resolve().parent
    frozen_paths = [source / name for name in FROZEN_FILES]
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in frozen_paths}
    manifest = {
        "phase": args.phase, "implementation_commit": git(source, "rev-parse", "HEAD"),
        "evaluation_commit": git(source, "log", "-1", "--format=%H", "--", *FROZEN_FILES),
        "evaluation_sha256": hashes, "binary_sha256": hashlib.sha256(executable.read_bytes()).hexdigest(),
        "rows": rows, "structures": STRUCTURES, "tile_rows": tiles, "regimes": regimes,
        "process_orders": orders, "method_filter": args.method,
        "trials": 7, "replays": "max(100,2*slots)",
        "rotation": "N>=4096; slots=max(2,floor(2*L2/X_bytes)+1)",
        "case_shuffle_seed": 20261007, "command": __import__("sys").argv,
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    info = subprocess.run([str(executable), "--device-info"], text=True, capture_output=True)
    (args.output / "device-info.txt").write_text(info.stdout + info.stderr)
    if info.returncode:
        raise SystemExit(info.returncode)
    failed = False
    fields = ["file", "rows", "family", "structure", "tile_rows", "regime", "order", "exit_code", "wall_seconds", "command"]
    with (args.output / "run_status.csv").open("w", newline="") as statuses:
        writer = csv.DictWriter(statuses, fieldnames=fields)
        writer.writeheader()
        for order in orders:
            cases = [(n, family, structure, tile, regime) for n in rows
                     for family, structure in STRUCTURES for tile in tiles for regime in regimes
                     if regime != "rotating" or n >= 4096]
            random.Random(20261007 + order).shuffle(cases)
            for n, family, structure, tile, regime in cases:
                stem = f"{family}_{structure}_{n}_tile{tile}_{regime}_order{order}"
                command = [str(executable), "--benchmark", str(n), family, str(structure),
                           str(tile), regime, str(order), args.method, "--stats",
                           str((args.output / f"{stem}.stats.csv").resolve())]
                start = time.monotonic()
                with (args.output / f"{stem}.csv").open("w") as stdout, (args.output / f"{stem}.err").open("w") as stderr:
                    result = subprocess.run(command, stdout=stdout, stderr=stderr)
                writer.writerow(dict(zip(fields, [stem + ".csv", n, family, structure, tile, regime,
                                                   order, result.returncode, time.monotonic()-start,
                                                   json.dumps(command)])))
                statuses.flush()
                failed |= result.returncode != 0
                print(f"{stem}: exit {result.returncode}", flush=True)
    current = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in frozen_paths}
    if current != hashes:
        (args.output / "INVALID-EVALUATION-CHANGED.txt").write_text("Evaluation files changed during this run. Results are invalid.\n")
        raise SystemExit(2)
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
