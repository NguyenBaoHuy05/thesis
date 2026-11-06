"""Build the concept–action logic graph (model/concept_graph.py) and print a report to read by hand.

  python tools/build_concept_graph.py [--matrix annotations/concepts/ntu_action_concept_v2_4.csv]
      [--out annotations/concepts/graph_v1.json]
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.concept_graph import graph_from_matrix


def report(graph):
    c, a = graph.concepts, graph.actions
    lines = [json.dumps(graph.summary(), indent=1), '', 'motifs (members: actions using it):']
    users = {g: [a[k] for k, ms in enumerate(graph.action_motifs) if g in ms] for g in range(len(graph.motifs))}
    for g, members in enumerate(graph.motifs):
        lines.append(f'  m{g}: {" & ".join(c[j] for j in members)}  ({len(users[g])}: {", ".join(users[g][:6])}'
                     + (', ...' if len(users[g]) > 6 else '') + ')')
    lines.append('OR groups: ' + '; '.join(' | '.join(c[j] for j in o) for o in graph.ors))
    lines.append(f'pairs ({len(graph.pairs)}), NOT(j|i) nodes ({len(graph.nots)}):')
    for i, j in graph.pairs:
        for x, y in ((i, j), (j, i)):
            diff = next((d for p, q, d in graph.nots if (p, q) == (x, y)), None)
            text = ', '.join(c[k] for k in diff) if diff else '(empty: no node)'
            lines.append(f'  NOT({a[y]} | {a[x]}) = 1 - AND({text})')
    return '\n'.join(lines)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--matrix', default='annotations/concepts/ntu_action_concept_v2_4.csv')
    parser.add_argument('--out', default='annotations/concepts/graph_v1.json')
    args = parser.parse_args()
    graph = graph_from_matrix(args.matrix)
    graph.save(args.out)
    print(report(graph))
    print(f'\nSaved {args.out}')
