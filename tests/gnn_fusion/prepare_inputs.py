"""Generate a locality experiment and convert the bundled PubMed graph (stdlib only)."""
import argparse
from pathlib import Path
import random


def write_matrix(path, rows):
    with path.open("w") as output:
        output.write("%%MatrixMarket matrix coordinate real general\n")
        output.write(f"{len(rows)} {len(rows)} {sum(map(len, rows))}\n")
        for row, neighbors in enumerate(rows):
            for col in sorted(neighbors):
                output.write(f"{row + 1} {col + 1} 1\n")


def banded(n, radius=8):
    # A ring with 16 neighbors plus a self-loop: all degrees are exactly 17.
    return [{(i + offset) % n for offset in range(-radius, radius + 1)} for i in range(n)]


def permute(rows):
    labels = list(range(len(rows)))
    random.Random(42).shuffle(labels)
    result = [set() for _ in rows]
    for i, neighbors in enumerate(rows):
        result[labels[i]] = {labels[j] for j in neighbors}
    return result


def pubmed(repo):
    folder = repo / "modeling/pubmed-data"
    with (folder / "Pubmed-Diabetes.NODE.paper.tab").open() as source:
        next(source)  # NODE paper
        next(source)  # Feature schema
        ids = [line.split("\t", 1)[0] for line in source if line.strip()]
    mapping = {node: i for i, node in enumerate(ids)}
    rows = [{i} for i in range(len(ids))]
    with (folder / "Pubmed-Diabetes.DIRECTED.cites.tab").open() as source:
        next(source)  # DIRECTED cites
        next(source)  # NO_FEATURES
        for line in source:
            if line.startswith("#"):
                continue
            nodes = [field.removeprefix("paper:") for field in line.split() if field.startswith("paper:")]
            if len(nodes) != 2:
                raise ValueError(f"Unexpected PubMed edge: {line}")
            a, b = (mapping[node] for node in nodes)
            rows[a].add(b)
            rows[b].add(a)
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    repo = Path(__file__).resolve().parents[2]
    write_matrix(args.output / "smoke.mtx", banded(64))
    for n in (65536, 262144):
        rows = banded(n)
        write_matrix(args.output / f"local_{n}.mtx", rows)
        if n == 65536:
            write_matrix(args.output / f"shuffled_{n}.mtx", permute(rows))
    write_matrix(args.output / "pubmed.mtx", pubmed(repo))
