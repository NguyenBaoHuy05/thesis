"""Add temporal-order (THEN) attributes to the action KG (A1-G, THEN extension).

  python tools/add_then_attributes.py [--kg annotations/action_kg.json] [--out annotations/action_kg_then.json]
  python tools/add_then_attributes.py --reverse --out annotations/action_kg_then_reversed.json

Each THEN attribute is "event A, then event B" (model/predicates.py: EVENTS, THEN). Edges come from
the `phases` text (start / middle / end) the LLM wrote for each action in the KG, read by hand:
- has: the phases say A, then B;
- lacks: only on the time-reversed partner whose phases say B, then A (an action whose phases do not
  settle the order stays unknown, never lacks).
All edges have w = 2 (the order is what defines the action) and record the phase text they come from.
The output keeps every other node and edge of the input, records `then_from` (input hash) and gets a
new hash; build its shuffled KGs (K2) with tools/shuffle_kg.py --kg <out>.

--reverse swaps has and lacks of every THEN edge (the rest of the KG unchanged): a targeted control.
If the order evidence is real, the reversed KG must do worse on the time-reversed pairs than the real one.
"""

import argparse
import copy
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.shuffle_kg import kg_hash

EVENT_TEXT = {
    'hand_at_head': 'a hand is at the head',
    'above_head': 'a hand is above the head',
    'behind': 'a hand is behind the torso',
    'chest': 'a hand is in front of the chest',
    'low': 'the active hand is low, at or below the waist',
}
BODY_PARTS = {
    'hand_at_head': ['hand_R', 'hand_L', 'head'], 'above_head': ['hand_R', 'hand_L', 'head'],
    'behind': ['hand_R', 'hand_L', 'torso'], 'chest': ['hand_R', 'hand_L', 'torso'],
    'low': ['hand_R', 'hand_L'],
}

# attribute -> (event A, event B, has, lacks); every action listed is checked against its phases below.
SPEC = {
    'then_hand_at_head_to_above_head': ('hand_at_head', 'above_head',
                                        ['take_off_a_hat_cap', 'take_off_headphone', 'throw_up_cap_hat'],
                                        ['put_on_a_hat_cap', 'put_on_headphone']),
    'then_above_head_to_hand_at_head': ('above_head', 'hand_at_head',
                                        ['put_on_a_hat_cap', 'put_on_headphone'],
                                        ['take_off_a_hat_cap', 'take_off_headphone']),
    'then_behind_to_chest': ('behind', 'chest', ['put_on_jacket', 'put_on_bag'], ['take_off_bag']),
    'then_chest_to_behind': ('chest', 'behind', ['take_off_jacket', 'take_off_bag'], ['put_on_jacket']),
    'then_low_to_chest': ('low', 'chest', ['take_object_out_of_bag', 'put_on_bag'], ['put_object_into_bag']),
    'then_chest_to_low': ('chest', 'low', ['put_object_into_bag', 'take_off_bag'],
                          ['take_object_out_of_bag', 'put_on_bag']),
}


def phases_text(action):
    p = action['phases']
    return f"{p['start']} | {p['middle']} | {p['end']}"


def add_then(kg, reverse=False):
    if 'shuffled' in kg:
        raise ValueError('add THEN attributes to the real KG, then shuffle the result')
    out = copy.deepcopy(kg)
    for name, (first, second, has, lacks) in SPEC.items():
        if name in out['attributes']:
            raise ValueError(f'{name} is already an attribute')
        out['attributes'][name] = dict(
            group='temporal_order',
            text=f'{EVENT_TEXT[first].capitalize()}, then {EVENT_TEXT[second]}.',
            definition=(f'First {EVENT_TEXT[first]}; later in the action {EVENT_TEXT[second]}. '
                        f'The reverse order does not count.'),
            body_parts=sorted(set(BODY_PARTS[first]) | set(BODY_PARTS[second])),
            observable='direct', events=[first, second],
            positive_examples=has, negative_examples=lacks,
            source='THEN extension: edges read from the KG phases (tools/add_then_attributes.py)')
        if reverse:
            has, lacks = lacks, has
        for kind, actions in (('has', has), ('lacks', lacks)):
            for action in actions:
                if action not in out['actions']:
                    raise ValueError(f'{name}: unknown action {action}')
                out['actions'][action][kind][name] = dict(
                    w=2, agree=1.0, evidence=f'phases: {phases_text(out["actions"][action])}')
    out['then_from'] = kg['hash']
    if reverse:
        out['then_reversed'] = True
    out['hash'] = kg_hash(out)
    return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parser.add_argument('--kg', default=os.path.join(repo, 'annotations', 'action_kg.json'))
    parser.add_argument('--out', default=os.path.join(repo, 'annotations', 'action_kg_then.json'))
    parser.add_argument('--reverse', action='store_true', help='swap has / lacks of the THEN edges (control)')
    args = parser.parse_args()
    with open(args.kg, encoding='utf-8') as handle:
        kg = json.load(handle)
    out = add_then(kg, args.reverse)
    with open(args.out, 'w', encoding='utf-8') as handle:
        json.dump(out, handle, indent=1, ensure_ascii=False)
    print(('REVERSED ' if args.reverse else '') + f'{len(SPEC)} THEN attributes, '
          f'{sum(len(s[2]) + len(s[3]) for s in SPEC.values())} edges -> {args.out} ({out["hash"][:19]}...)')
    for name, (first, second, has, lacks) in SPEC.items():
        print(f'  {name}: has {", ".join(has)}; lacks {", ".join(lacks)}')
