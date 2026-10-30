"""
Rule-based negative pair generator for task-oriented dialogue summarization.

Generates three types of hard negatives based on empirical failure analysis
of fine-tuned SLLM models on MultiWOZ:

  Type 1 - Remove booking reference numbers  (BART 66/150, Qwen 31/100)
  Type 2 - Generalize contact/phone numbers  (BART 50/150, Qwen 12/100)
  Type 3 - Mutate quantities (people/nights) (BART 9/150,  Qwen 12/100)

Strategy options:
  priority  Apply first applicable type in order Type1 > Type2 > Type3
  random    Randomly pick one applicable type
  all       Apply all applicable types sequentially

Usage:
    python generate_negative_pairs.py \
        --input  data/train.json \
        --output data/train_with_negatives.json \
        --source reference \
        --strategy priority \
        --seed 42
"""

import json
import re
import random
import argparse
import copy
from pathlib import Path


# ── Patterns ──────────────────────────────────────────────────────────────────

# Booking reference: 6-8 uppercase alphanumeric, standalone word
REF_PATTERN      = re.compile(r'\b([A-Z][A-Z0-9]{5,7})\b')

# Contact / phone: 5+ consecutive digits
CONTACT_PATTERN  = re.compile(r'(?<!\d)(\d{5,})(?!\d)')

# Quantities: integer + unit (people, nights, rooms …)
QUANTITY_PATTERN = re.compile(
    r'\b(\d+)\s+(people|persons?|nights?|days?|rooms?)',
    re.IGNORECASE
)

# Phrases used as replacements
GENERIC_REF_PHRASES = [
    "booking reference provided",
    "reservation confirmed",
    "booking confirmed",
    "confirmation provided",
]

GENERIC_CONTACT_PHRASES = [
    "contact details provided",
    "contact number provided",
    "contact information shared",
    "driver contact provided",
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _clean(text: str) -> str:
    """Remove common artifacts left after replacement."""
    text = re.sub(r':\s*[,.]',     '.',  text)        # ": ." → "."
    text = re.sub(r'\s+[,.]',      '.',  text)        # " ."  → "."
    text = re.sub(r'\bwith\s+\.',  '.',  text)        # "with ." → "."
    text = re.sub(r'\band\s+\.',   '.',  text)        # "and ." → "."
    text = re.sub(r'\bproviding\s+\.',  '.', text)
    text = re.sub(r'(\b\w+\b) \1', r'\1', text)  # remove consecutive duplicate words
    text = re.sub(r'\b(\w+) \1\b',  r'\1', text)  # consecutive duplicate words
    text = re.sub(r'  +',          ' ',  text)        # double spaces
    text = re.sub(r' +\n',         '\n', text)        # trailing spaces
    text = re.sub(r'\n{3,}',       '\n\n', text)      # excess blank lines
    # Remove bullet lines that became empty after substitution
    text = re.sub(r'^[-–*•]\s*[,.]?\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'\n{3,}',       '\n\n', text)
    return text.strip()


def _is_valid_negative(original: str, modified: str) -> bool:
    """Reject modifications that are identical or too short."""
    return modified != original and len(modified.strip()) >= 30


# ── Type 1: Remove booking reference numbers ──────────────────────────────────

def remove_booking_reference(summary: str, rng: random.Random) -> str | None:
    """
    Replace booking/reservation reference codes with a generic phrase.

    Before: "Provided booking confirmation with reference number 81E1ITXC."
    """
    if not REF_PATTERN.search(summary):
        return None

    result = summary

    # Pattern: "reference number CODE" or "reference CODE"
    result = re.sub(
        r'\breference\s+(?:number\s+)?[A-Z][A-Z0-9]{5,7}\b',
        rng.choice(GENERIC_REF_PHRASES),
        result,
        flags=re.IGNORECASE
    )
    # Pattern: "booking reference: CODE"
    result = re.sub(
        r'\b(?:booking\s+)?reference\s*:\s*[A-Z][A-Z0-9]{5,7}\b',
        rng.choice(GENERIC_REF_PHRASES),
        result,
        flags=re.IGNORECASE
    )
    # Remove any remaining standalone codes
    result = REF_PATTERN.sub('', result)

    result = _clean(result)
    return result if _is_valid_negative(summary, result) else None


# ── Type 2: Generalize contact / phone numbers ────────────────────────────────

def generalize_contact_numbers(summary: str, rng: random.Random) -> str | None:
    """
    Replace specific phone/contact numbers with a generic phrase.

    Before: "...contact number 07341216795."
    After:  "...contact details provided."
    """
    if not CONTACT_PATTERN.search(summary):
        return None

    result = summary

    # Pattern: "contact number XXXXX" / "phone number XXXXX" / "number XXXXX"
    result = re.sub(
        r'(?:contact\s+|phone\s+)?(?:number|phone)\s+\d{5,}',
        rng.choice(GENERIC_CONTACT_PHRASES),
        result,
        flags=re.IGNORECASE
    )

    # Pattern: "number: XXXXX"
    result = re.sub(
        r'(?:number|phone)\s*:\s*\d{5,}',
        rng.choice(GENERIC_CONTACT_PHRASES),
        result,
        flags=re.IGNORECASE
    )

    # Remove remaining standalone long digit sequences
    result = CONTACT_PATTERN.sub('', result)

    result = _clean(result)
    return result if _is_valid_negative(summary, result) else None


# ── Type 3: Mutate quantities ─────────────────────────────────────────────────

def mutate_quantities(summary: str, rng: random.Random) -> str | None:
    """
    Change numeric quantities (people, nights, rooms) by ±1 or ±2.

    Before: "Booked for 7 people for 3 nights."
    After:  "Booked for 6 people for 2 nights."
    """
    matches = list(QUANTITY_PATTERN.finditer(summary))
    if not matches:
        return None

    result = summary
    offset = 0
    mutated_any = False

    for match in matches:
        original_num = int(match.group(1))
        unit = match.group(2)

        # Choose delta: prefer ±1, ensure result > 0
        deltas = [1, -1, 2, -2]
        rng.shuffle(deltas)
        delta = next((d for d in deltas if original_num + d > 0), None)
        if delta is None:
            continue

        new_num   = original_num + delta
        old_text  = match.group(0)
        new_text  = f"{new_num} {unit}"

        start = match.start() + offset
        end   = match.end()   + offset
        result = result[:start] + new_text + result[end:]
        offset += len(new_text) - len(old_text)
        mutated_any = True

    return result if (mutated_any and _is_valid_negative(summary, result)) else None


# ── Strategy dispatcher ───────────────────────────────────────────────────────

def generate_negative(
    positive: str,
    strategy: str,
    rng: random.Random
) -> dict:
    """
    Generate one negative summary from a positive summary.

    Returns:
        negative   : str | None  — the generated negative (None if not possible)
        type_used  : str         — type applied, e.g. 'type1_ref'
        applicable : list[str]   — all types that could have been applied
    """
    t1 = remove_booking_reference(positive, rng)
    t2 = generalize_contact_numbers(positive, rng)
    t3 = mutate_quantities(positive, rng)

    applicable = []
    if t1: applicable.append('type1_ref')
    if t2: applicable.append('type2_contact')
    if t3: applicable.append('type3_quantity')

    if strategy == 'priority':
        for neg, name in [(t1, 'type1_ref'), (t2, 'type2_contact'), (t3, 'type3_quantity')]:
            if neg:
                return {'negative': neg, 'type_used': name, 'applicable': applicable}
        return {'negative': None, 'type_used': 'none', 'applicable': applicable}

    elif strategy == 'random':
        valid = [(neg, name) for neg, name in
                 [(t1,'type1_ref'), (t2,'type2_contact'), (t3,'type3_quantity')]
                 if neg is not None]
        if not valid:
            return {'negative': None, 'type_used': 'none', 'applicable': applicable}
        neg, name = rng.choice(valid)
        return {'negative': neg, 'type_used': name, 'applicable': applicable}

    elif strategy == 'all':
        # Apply all applicable types in sequence
        result = positive
        used   = []
        for fn, name in [
            (remove_booking_reference,   'type1_ref'),
            (generalize_contact_numbers, 'type2_contact'),
            (mutate_quantities,          'type3_quantity'),
        ]:
            modified = fn(result, rng)
            if modified:
                result = modified
                used.append(name)
        if not used:
            return {'negative': None, 'type_used': 'none', 'applicable': applicable}
        return {'negative': result, 'type_used': '+'.join(used), 'applicable': applicable}

    return {'negative': None, 'type_used': 'none', 'applicable': applicable}


# ── Main ──────────────────────────────────────────────────────────────────────

def process_dataset(
    input_path: str,
    output_path: str,
    source_field: str = 'reference',
    strategy: str    = 'priority',
    seed: int        = 42,
):
    rng = random.Random(seed)

    with open(input_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]

    results = []
    stats   = {'total': len(data), 'generated': 0, 'skipped': 0}
    type_counts = {}

    for item in data:
        positive = item.get(source_field, '').strip()
        if not positive:
            stats['skipped'] += 1
            continue

        gen      = generate_negative(positive, strategy, rng)
        negative = gen['negative']
        tused    = gen['type_used']

        out = copy.deepcopy(item)
        out['summaries'] = []
        pair = {'pos' : positive, 'neg' : negative}
        if negative:
            out['summaries'].append(pair)
        # out['summaries']['positive']            = positive
        # out['summaries']['negative']            = negative
        # out['negative_type']       = tused
        # out['negative_applicable'] = gen['applicable']
        results.append(out)

        if negative:
            stats['generated'] += 1
            type_counts[tused] = type_counts.get(tused, 0) + 1
        else:
            stats['skipped'] += 1
            type_counts['none'] = type_counts.get('none', 0) + 1

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    total = stats['total']
    print(f'\n{"="*50}')
    print(f'Negative pair generation complete')
    print(f'{"="*50}')
    print(f'Total samples   : {total}')
    print(f'Generated       : {stats["generated"]}  ({stats["generated"]/total*100:.1f}%)')
    print(f'Skipped         : {stats["skipped"]}')
    print(f'\nBreakdown by type:')
    for k, v in sorted(type_counts.items()):
        print(f'  {k:<30}: {v}')
    print(f'\nSaved → {output_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Rule-based negative pair generator for contrastive learning'
    )
    parser.add_argument('--input',    required=True)
    parser.add_argument('--output',   required=True)
    parser.add_argument('--source',   default='reference',
                        help='Field used as positive summary (default: reference)')
    parser.add_argument('--strategy', default='priority',
                        choices=['priority', 'random', 'all'],
                        help='Negative selection strategy (default: priority)')
    parser.add_argument('--seed',     default=42, type=int)
    args = parser.parse_args()

    process_dataset(
        input_path   = args.input,
        output_path  = args.output,
        source_field = args.source,
        strategy     = args.strategy,
        seed         = args.seed,
    )