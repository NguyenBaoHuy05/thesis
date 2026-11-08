// Rule discriminability of a KG. Usage: node kgcmp.js <primitive.json> <kg_rules.json>
const fs = require('fs');
const P = JSON.parse(fs.readFileSync(process.argv[2], 'utf8'));
const R = JSON.parse(fs.readFileSync(process.argv[3], 'utf8')).rules;
const UNSEEN = {'55/5': [10, 11, 19, 26, 56], '48/12': [3, 5, 9, 12, 15, 40, 42, 47, 51, 56, 58, 59],
  '110/10': [4, 13, 37, 43, 49, 65, 88, 95, 99, 106],
  '96/24': [5, 9, 11, 16, 18, 20, 22, 29, 35, 39, 45, 49, 59, 68, 70, 81, 84, 87, 93, 94, 104, 113, 114, 119]};
const byIdx = {}; for (const r of R) byIdx[r.id - 1] = r;
const seqKey = r => (r.sequence || []).join('>');
const hard = (a, b) => a.require.filter(k => b.forbid.includes(k)).length + b.require.filter(k => a.forbid.includes(k)).length;
const soft = (a, b) => a.require.filter(k => !b.require.includes(k)).length + b.require.filter(k => !a.require.includes(k)).length;
// a pair is "weak" if no hard separator, and either identical require sets (only sequence may differ) or <= 2 soft separators
function status(a, b) {
  const h = hard(a, b), s = soft(a, b);
  if (h > 0) return 'hard';
  if (s === 0) return seqKey(a) && seqKey(b) && seqKey(a) !== seqKey(b) ? 'seq-only' : 'IDENTICAL';
  return s <= 2 ? 'weak' : 'soft';
}
const used = new Set(R.flatMap(r => r.require));
console.log(`primitives ${P.length}, never required: ${P.filter(p => !used.has(p.id)).map(p => p.id).join(' ') || '-'}`);
const cnt = {}; for (const r of R) for (const k of r.require) cnt[k] = (cnt[k] || 0) + 1;
console.log(`avg require ${(R.reduce((s, r) => s + r.require.length, 0) / R.length).toFixed(2)}, avg forbid ${(R.reduce((s, r) => s + r.forbid.length, 0) / R.length).toFixed(2)}`);
function report(name, cls) {
  const c = {hard: 0, soft: 0, weak: 0, 'seq-only': 0, IDENTICAL: 0}, bad = [];
  for (let i = 0; i < cls.length; i++) for (let j = i + 1; j < cls.length; j++) {
    const st = status(cls[i], cls[j]); c[st]++;
    if (st === 'weak' || st === 'IDENTICAL') bad.push(`${cls[i].class} ~ ${cls[j].class} [${st}]`);
  }
  const n = Object.values(c).reduce((a, b) => a + b, 0);
  console.log(`${name.padEnd(8)} pairs ${String(n).padStart(4)} | hard ${c.hard} soft ${c.soft} weak ${c.weak} seq-only ${c['seq-only']} IDENTICAL ${c.IDENTICAL}`);
  return bad;
}
for (const [sp, U] of Object.entries(UNSEEN)) { const b = report(sp, U.map(i => byIdx[i])); for (const x of b) console.log('    ', x); }
const b = report('NTU60', R.filter(r => r.id <= 60)); console.log('     NTU60 weak/identical:', b.length);
const b2 = report('NTU120', R); for (const x of b2) console.log('    ', x);
