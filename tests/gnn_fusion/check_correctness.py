"""Small regression cases, including the all-fused empty trailing tile."""
import argparse
from pathlib import Path
import subprocess
import tempfile

from prepare_inputs import banded, permute, write_matrix
from run_suite import read_result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("executable", type=Path)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="gnn-correctness-") as directory:
        folder = Path(directory)
        # Unequal feature/output dimensions also check matrix strides.
        cases = [("single", [{0}], 17, 96, 1048576, 1),
                 ("all_fused", banded(64), 17, 96, 1048576, 1),
                 ("unfused", permute(banded(1024)), 17, 32, 65536, 0)]
        for name, rows, features, hidden, cache, expected_fraction in cases:
            matrix = folder / f"{name}.mtx"
            csv = folder / f"{name}.csv"
            write_matrix(matrix, rows)
            with csv.open("w") as output:
                subprocess.run([str(args.executable.resolve()), str(matrix), str(features),
                                str(hidden), str(args.threads), str(cache), "3", "2", "3"],
                               stdout=output, check=True)
            results = read_result(csv, 3)
            if results["tile_fused_avx2"]["fused_fraction"] != expected_fraction:
                raise AssertionError(f"{name}: regression case did not exercise intended schedule")
            print(f"PASS: {name} ({args.threads} threads)", flush=True)


if __name__ == "__main__":
    main()
