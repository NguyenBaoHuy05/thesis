import re
from collections import Counter

with open(r'd:\Thesis\CLIR-SkateFormer\work_dir\ntu60_cs\CLIR_SkateFormer_j_A4_K2\rules_best\dnf_rules.txt', 'r', encoding='utf-8') as f:
    text = f.read()

actions = [a.strip() for a in text.strip().split('────────────────────────────────────────────────────────────') if a.strip()]
print('Total actions parsed:', len(actions))

clause_types = Counter()
num_clauses_per_action = []
weights_by_type = {'AND1': [], 'OR1': [], 'AND2': [], 'OR2': []}
literals = Counter()
pos_count = 0
neg_count = 0
clauses_per_act_map = {}

for act in actions:
    lines = [l.strip() for l in act.split('\n')]
    act_name = lines[0]
    c_count = 0
    for l in lines:
        m = re.search(r'Clause \d+ \[(AND1|AND2|OR1|OR2).*?w=([\d\.]+)', l)
        if m:
            ctype = m.group(1)
            w = float(m.group(2))
            clause_types[ctype] += 1
            weights_by_type[ctype].append(w)
            c_count += 1
        if l.startswith('('):
            subclauses = l.split(' ∨ ')
            for sc in subclauses:
                clean_sc = sc.strip('() ')
                parts = [p.strip() for p in clean_sc.split('∧')]
                for p in parts:
                    if not p:
                        continue
                    if p.startswith('NOT '):
                        neg_count += 1
                        literals[p] += 1
                    else:
                        pos_count += 1
                        literals[p] += 1
    num_clauses_per_action.append(c_count)
    clauses_per_act_map[act_name] = c_count

print('Clause types distribution:', clause_types)
for k, v in weights_by_type.items():
    if v:
        print(f'{k}: count={len(v)}, min={min(v):.3f}, max={max(v):.3f}, mean={sum(v)/len(v):.3f}')

print(f'Clauses per action: min={min(num_clauses_per_action)}, max={max(num_clauses_per_action)}, mean={sum(num_clauses_per_action)/len(num_clauses_per_action):.2f}')
total_literals = pos_count + neg_count
print(f'Literals: total={total_literals}, pos={pos_count} ({pos_count/total_literals*100:.1f}%), neg={neg_count} ({neg_count/total_literals*100:.1f}%)')

print('\nTop 10 positive literals:')
pos_lits = [(k, v) for k, v in literals.items() if not k.startswith('NOT ')]
pos_lits.sort(key=lambda x: x[1], reverse=True)
for lit, c in pos_lits[:10]:
    print(f'  {lit}: {c}')

print('\nTop 10 negative literals:')
neg_lits = [(k, v) for k, v in literals.items() if k.startswith('NOT ')]
neg_lits.sort(key=lambda x: x[1], reverse=True)
for lit, c in neg_lits[:10]:
    print(f'  {lit}: {c}')

# Check which actions have OR1
print('\nActions with OR1 clauses:')
for act in actions:
    lines = [l.strip() for l in act.split('\n')]
    has_or1 = any('OR1' in l for l in lines)
    if has_or1:
        print(' ', lines[0])
