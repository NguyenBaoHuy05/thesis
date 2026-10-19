"""Degree-preserving shuffles of the action-concept matrix for ablation K2 (model/concept_logic.py).

  python tools/shuffle_concepts.py [--matrix annotations/concepts/ntu_action_concept_v2_4.csv]
      [--seeds 0 1 2] [--out-dir annotations/concepts/shuffled] [--swaps-per-one 10]

Checkerboard swaps: rows r1, r2 and columns c1, c2 with M[r1, c1] = M[r2, c2] = 1 and
M[r1, c2] = M[r2, c1] = 0 become M[r1, c2] = M[r2, c1] = 1, M[r1, c1] = M[r2, c2] = 0. Every action
keeps its number of concepts and every concept its number of actions; swapping continues until all
rows are distinct again (as in the real matrix), so a shuffled matrix still names every class. Writes
<out-dir>/<matrix stem>_shuffle_s<seed>.csv with the same header and row order.
"""

import argparse
import csv
import os
import random


def read(path):
    with open(path, encoding='utf-8', newline='') as handle:
        rows = list(csv.reader(handle))
    return rows[0], [r[0] for r in rows[1:]], [[int(float(v)) for v in r[1:]] for r in rows[1:]]


def shuffle(M, rng, swaps):
    M = [row[:] for row in M]
    ones = [(r, c) for r, row in enumerate(M) for c, v in enumerate(row) if v]
    done, attempts = 0, 0
    while done < swaps or len({tuple(row) for row in M}) < len(M):
        attempts += 1
        if attempts > 1000 * max(swaps, 1):
            raise RuntimeError('could not reach distinct rows')
        i, j = rng.randrange(len(ones)), rng.randrange(len(ones))
        (r1, c1), (r2, c2) = ones[i], ones[j]
        if r1 == r2 or c1 == c2 or M[r1][c2] or M[r2][c1]:
            continue
        M[r1][c1] = M[r2][c2] = 0
        M[r1][c2] = M[r2][c1] = 1
        ones[i], ones[j] = (r1, c2), (r2, c1)
        done += 1
    return M


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parser.add_argument('--matrix', default=os.path.join(repo, 'annotations', 'concepts', 'ntu_action_concept_v2_4.csv'))
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    parser.add_argument('--out-dir', default=os.path.join(repo, 'annotations', 'concepts', 'shuffled'))
    parser.add_argument('--swaps-per-one', type=int, default=10)
    args = parser.parse_args()

    header, actions, M = read(args.matrix)
    n_ones = sum(map(sum, M))
    os.makedirs(args.out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(args.matrix))[0]
    for seed in args.seeds:
        S = shuffle(M, random.Random(seed), args.swaps_per_one * n_ones)
        assert [sum(r) for r in S] == [sum(r) for r in M]
        assert [sum(col) for col in zip(*S)] == [sum(col) for col in zip(*M)]
        kept = sum(a == b == 1 for ra, rb in zip(M, S) for a, b in zip(ra, rb))
        path = os.path.join(args.out_dir, f'{stem}_shuffle_s{seed}.csv')
        with open(path, 'w', encoding='utf-8', newline='') as handle:
            writer = csv.writer(handle, quoting=csv.QUOTE_ALL)
            writer.writerow(header)
            for name, row in zip(actions, S):
                writer.writerow([name] + [str(v) for v in row])
        print(f'seed {seed}: {kept}/{n_ones} ones still in place -> {path}')
