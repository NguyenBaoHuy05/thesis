// Build zeroshot/primitive_v2.json and zeroshot/kg_rules_v2.json from v1 (v1 files are left untouched).
// Usage: node make_kg_v2.js <zeroshot dir>
const fs = require('fs');
const path = require('path');
const dir = process.argv[2];
const P = JSON.parse(fs.readFileSync(path.join(dir, 'primitive.json'), 'utf8'));
const K = JSON.parse(fs.readFileSync(path.join(dir, 'kg_rules.json'), 'utf8'));
const byId = Object.fromEntries(P.map(p => [p.id, p]));

// ------------------------------------------------------------------ vocabulary
// changed definitions / groups
Object.assign(byId['head.nod'], {
  definition: 'Sagittal angle of (p3-p2) relative to the trunk axis (p20-p0) increases by > 10 deg and returns by > 7 deg within 2 s (>= 1 down-up cycle; repeated cycles = nodding, one slow cycle = bow)',
  description: 'Head dips down and comes back up (nod or bow).',
});
byId['head.lowered'].exclusive_group = 'head_pitch';
byId['hands.touching'].exclusive_group = 'hands_distance';

const NEW = [
  {id: 'hands.both_active', part: 'hand', type: 'motion', joints: [1, 7, 11, 21, 23],
   definition: 'Speeds of p7 and p11 relative to p1 both exceed 0.3 L/s in overlapping windows covering >= 0.5 s',
   exclusive_group: 'hand_activity', description: 'Both hands are moving at the same time (bimanual action).'},
  {id: 'hands.one_active', part: 'hand', type: 'motion', joints: [1, 7, 11, 21, 23],
   definition: 'One hand moves relative to p1 (speed > 0.3 L/s for >= 0.5 s) while the other hand stays still (speed < 0.1 L/s) over the same window',
   exclusive_group: 'hand_activity', description: 'Only one hand is moving while the other hand stays still.'},
  {id: 'hands.near', part: 'hand', type: 'relation', joints: [7, 11, 21, 23],
   definition: '0.12L <= min(dist(p7,p11), dist(p21,p23)) < 0.35L',
   exclusive_group: 'hands_distance', description: 'The two hands are close to each other without touching, as if holding an object between them.'},
  {id: 'hands.pull_apart', part: 'hand', type: 'motion', joints: [7, 11, 21, 23],
   definition: 'dist(p7,p11) goes from < 0.15L to > 0.35L within 1 s',
   exclusive_group: null, description: 'The hands start together and quickly pull apart.'},
  {id: 'hand.static_hold', part: 'hand', type: 'state', joints: [0, 1, 7, 11, 21, 23],
   definition: 'For any hand with yH > y0: speed of pH relative to p1 < 0.08 L/s for >= 1 s, after the hand moved > 0.3L to reach that position',
   exclusive_group: null, description: 'A hand is raised to a position and held still there (holding a gesture).'},
  {id: 'hand.wrist_twist', part: 'hand', type: 'motion', joints: [6, 7, 10, 11, 21, 22, 23, 24],
   definition: 'For any hand: the hand-plane normal (pTip - pH) x (pThumb - pH) rotates about the forearm axis (pH - pW) by > 60 deg with >= 2 reversals within 2 s',
   exclusive_group: null, description: 'A wrist twists back and forth, as when turning a cap or a cube.'},
  {id: 'hand.thumb_index_touch', part: 'hand', type: 'state', joints: [7, 11, 21, 22, 23, 24],
   definition: 'For any hand: dist(p_thumb, p_tip) < 0.04L (joints 22-21 or 24-23) for >= 3 frames',
   exclusive_group: null, description: 'The thumb tip touches the finger tip of the same hand.'},
  {id: 'head.tilt_back', part: 'head', type: 'state', joints: [0, 2, 3, 20],
   definition: 'Angle between (p3-p2) and (p20-p0) > 15 deg with backward component (p3-p2)·f < 0',
   exclusive_group: 'head_pitch', description: 'Head is tilted back, face turned upward.'},
  {id: 'body.brief_motion', part: 'body', type: 'motion', joints: [],
   definition: 'Total time with mean joint speed relative to p1 > 0.3 L/s is < 1.2 s',
   exclusive_group: 'motion_duration', description: 'The movement is short and happens once.'},
  {id: 'body.sustained_motion', part: 'body', type: 'motion', joints: [],
   definition: 'Total time with mean joint speed relative to p1 > 0.3 L/s is > 2.5 s',
   exclusive_group: 'motion_duration', description: 'The movement goes on for a long time.'},
  {id: 'inter.mutual_reach', part: 'interaction', type: 'motion', joints: [0, 1, 7, 11],
   definition: 'Within overlapping 1 s windows, a hand of A and a hand of B each move toward the other person (distance to the other p1 decreases by > 0.3L)',
   exclusive_group: null, description: 'Both persons reach a hand toward each other.'},
];
for (const p of NEW) if (byId[p.id]) throw new Error('dup ' + p.id);
const P2 = [...P, ...NEW];
const ids = new Set(P2.map(p => p.id));

// ------------------------------------------------------------------ rules
const rules = JSON.parse(JSON.stringify(K.rules));
const R = Object.fromEntries(rules.map(r => [r.id, r]));
function set(id, {require, forbid, sequence}) {
  const r = R[id];
  if (require) r.require = require;
  if (forbid) r.forbid = forbid;
  if (sequence) r.sequence = sequence;
}
const add = (id, key, ...ks) => { for (const k of ks) if (!R[id][key].includes(k)) R[id][key].push(k); };
const del = (id, key, ...ks) => { R[id][key] = R[id][key].filter(k => !ks.includes(k)); };

// writing / typing / cutting paper / counting money: one vs both hands, contact
add(12, 'require', 'hands.one_active');               add(12, 'forbid', 'hands.both_active');
add(30, 'require', 'hands.both_active');              add(30, 'forbid', 'hands.one_active');
set(76, {require: ['hand.small_fast_oscillation', 'head.lowered', 'hands.near', 'hands.both_active'],
         forbid: ['inter.two_person', 'hands.clapping', 'hands.touching', 'hands.one_active']});
set(74, {require: ['hand.small_fast_oscillation', 'head.lowered', 'hand.torso_level', 'hands.touching'],
         forbid: ['inter.two_person', 'hand.above_head', 'hands.one_active']});
// reading / check time / phone: still hands, object held between hands
add(11, 'require', 'hand.static_hold', 'hands.near');
add(33, 'require', 'hand.static_hold');               add(33, 'forbid', 'hands.near');
add(29, 'require', 'hands.near');
// bimanual manipulations of a held object
set(13, {require: ['hands.touching', 'hands.pull_apart', 'hand.torso_level'],
         forbid: ['inter.two_person', 'hands.clapping', 'hands.rubbing', 'hand.wrist_twist'],
         sequence: ['hands.touching', 'hands.pull_apart']});
set(78, {require: ['hands.touching', 'hand.wrist_twist', 'hand.torso_level'],
         forbid: ['inter.two_person', 'hands.clapping', 'hands.pull_apart']});
set(75, {require: ['hands.touching', 'head.lowered', 'hand.torso_level', 'hands.one_active', 'body.sustained_motion'],
         forbid: ['inter.two_person', 'hands.clapping', 'hands.rubbing', 'hands.pull_apart', 'hand.wrist_twist']});
set(82, {require: ['hands.touching', 'head.lowered', 'hand.torso_level', 'hands.both_active'],
         forbid: ['inter.two_person', 'hand.small_fast_oscillation', 'hands.one_active', 'hand.at_face']});
set(73, {require: ['hands.touching', 'head.lowered', 'body.brief_motion'],
         forbid: ['inter.two_person', 'hands.rubbing', 'body.sustained_motion']});
add(83, 'require', 'hands.both_active');
set(39, {require: ['hands.touching', 'hand.torso_level', 'hand.static_hold'],
         forbid: ['inter.two_person', 'hands.rubbing', 'hands.clapping', 'hands.pull_apart']});
add(84, 'require', 'hand.wrist_twist', 'hands.both_active');
// hand at face: hold vs brief vs head back
set(67, {require: ['hand.at_face', 'arm.flexed', 'hand.static_hold'],
         forbid: ['inter.two_person', 'hand.vertical_oscillation', 'hand.small_fast_oscillation', 'torso.forward_jerk', 'head.tilt_back']});
set(79, {require: ['hand.at_face', 'arm.flexed', 'body.brief_motion'],
         forbid: ['inter.two_person', 'torso.forward_jerk', 'hand.vertical_oscillation', 'hand.static_hold', 'head.tilt_back']});
set(103, {require: ['hand.at_face', 'head.tilt_back'],
          forbid: ['inter.two_person', 'torso.forward_jerk', 'hand.small_fast_oscillation', 'hand.vertical_oscillation', 'head.lowered']});
add(105, 'require', 'body.brief_motion');            add(105, 'forbid', 'hand.small_fast_oscillation');
// hand signs
add(69, 'require', 'hand.static_hold');               add(69, 'forbid', 'hand.thumb_index_touch');
add(70, 'require', 'hand.static_hold');               add(70, 'forbid', 'hand.thumb_index_touch');
set(71, {require: ['hand.in_front_space', 'arm.flexed', 'hand.thumb_index_touch', 'hand.static_hold'],
         forbid: ['inter.two_person', 'hand.thumb_down', 'hand.thumb_up', 'hand.lateral_oscillation']});
add(72, 'require', 'hand.static_hold');               add(72, 'forbid', 'hand.thumb_index_touch', 'hand.thumb_up');
set(77, {require: ['hand.in_front_space', 'arm.flexed', 'hand.thumb_index_touch', 'body.brief_motion'],
         forbid: ['inter.two_person', 'hands.touching', 'hand.lateral_oscillation', 'hand.static_hold']});
// drop vs juggling
set(5, {require: ['hand.in_front_space', 'hand.torso_level', 'body.brief_motion'],
        forbid: ['inter.two_person', 'hand.at_foot', 'torso.bend_deep', 'hand.above_head', 'hand.vertical_oscillation']});
add(66, 'require', 'body.sustained_motion');          add(66, 'forbid', 'body.brief_motion');
// two-person: one-way vs mutual, contact vs no contact
add(56, 'forbid', 'inter.mutual_reach');
add(118, 'require', 'inter.mutual_reach');
add(107, 'forbid', 'inter.hand_to_other_head_shoulder', 'inter.hand_to_other_trunk');
// fixes
set(35, {require: ['head.nod', 'head.lowered']});                    // head.nod was never used
del(22, 'require', 'hand.above_head');                              // implied by both_hands.above_head

// checks
for (const r of rules) {
  for (const k of [...r.require, ...r.forbid, ...(r.sequence || [])]) if (!ids.has(k)) throw new Error(`${r.class}: unknown ${k}`);
  if (r.require.some(k => r.forbid.includes(k))) throw new Error(`${r.class}: require & forbid overlap`);
  if ((r.sequence || []).some(k => !r.require.includes(k))) throw new Error(`${r.class}: sequence not in require`);
}

// constraints
const impl = [...K.constraints.implication,
  ['inter.mutual_reach', 'inter.two_person'],
  ['inter.arms_around_other', 'inter.hand_to_other_back'],
  ['both_hands.above_head', 'hand.above_head'],
  ['hands.rubbing', 'hands.touching'],
  ['hands.clapping', 'hands.both_active'],
  ['hands.rubbing', 'hands.both_active'],
];

const meta = {...K.meta,
  generator: K.meta.generator + '. v2: Claude Opus 5.5, 2026-11-08, revised from v1 using only a rule-level analysis (classes with identical require sets, class pairs separated by <= 2 primitives) and class names; no experiment result or test data used',
  vocabulary: `zeroshot/primitive_v2.json (${P2.length} primitives: v1 + ${NEW.length} new, head.nod redefined to include bow)`,
};

// ------------------------------------------------------------------ write (one entry per line, like v1)
const line = o => JSON.stringify(o);
fs.writeFileSync(path.join(dir, 'primitive_v2.json'), '[\n' + P2.map(p => '  ' + line(p)).join(',\n') + '\n]\n');
const out = '{\n  "meta": ' + JSON.stringify(meta, null, 4).replace(/\n/g, '\n  ') +
  ',\n  "constraints": {\n    "implication": [\n' + impl.map(x => '      ' + line(x)).join(',\n') + '\n    ]\n  },\n' +
  '  "rules": [\n' + rules.map(r => '    ' + JSON.stringify(r).replace(/","/g, '", "').replace(/":/g, '": ').replace(/,"/g, ', "')).join(',\n') + '\n  ]\n}\n';
fs.writeFileSync(path.join(dir, 'kg_rules_v2.json'), out);
console.log(`wrote ${P2.length} primitives, ${rules.length} rules, ${impl.length} implications`);
