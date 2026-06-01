#!/usr/bin/env python3
"""
self_consistency_vote.py
========================
Aggregate k sampled completions per question into a single voted answer.

Designed for the Qwen3-4B math benchmark pipeline. This is PURE POST-PROCESSING
of the model's own outputs (allowed under the rules — no external solver touches
the math; we only pick among the model's samples).

WHAT IT DOES
------------
For each question, given k completions:
  * extracts the FINAL answer from each sample (last \\boxed{...}, with brace
    matching; falls back to N separate boxes for multi-answer; drops samples
    that never closed </think> or never produced a box -> truncation handling),
  * normalizes answers for voting (numbers rounded to 6 dp with round-half-up,
    NOT truncated; symbolic forms canonicalized with sympy when possible),
  * votes:
       - MCQ            -> mode of the chosen letter(s)
       - single free    -> mode of the normalized value
       - multi-answer   -> mode PER SLOT (each [ANS] voted independently),
                           then reassembled into one comma box,
  * emits a reconstructed `response` containing the voted \\boxed{...} so it
    flows through your EXISTING answer-extractor / CSV step unchanged,
  * (optional) if `gold` is present, estimates accuracy with a grader that
    approximates the real one (numeric tolerance + sympy symbolic equality).

INPUT FORMATS (pick one)
------------------------
A) One combined JSONL, each line:
     {"id":int, "is_mcq":bool, "n_ans":int(optional),
      "gold":[...](optional), "samples":["<resp1>", "<resp2>", ...]}

B) k separate JSONL files in the SAME shape as your inference output
   (each line {"id","is_mcq","gold","response",...}).  Use --merge to fold
   them into form (A) by id.  Example:
     python self_consistency_vote.py --merge run1.jsonl run2.jsonl run3.jsonl \
            run4.jsonl run5.jsonl --out voted.jsonl

USAGE
-----
  # vote + write submission-ready responses
  python self_consistency_vote.py --in combined.jsonl --out voted.jsonl

  # merge k single-sample files first, then vote, then estimate public acc
  python self_consistency_vote.py --merge r1.jsonl r2.jsonl r3.jsonl \
         --out voted.jsonl --grade

  # restrict voting to free-form + multi only (MCQ passes through 1st sample)
  python self_consistency_vote.py --in combined.jsonl --out voted.jsonl \
         --skip-mcq-vote
"""

import argparse, json, re, sys
from collections import Counter, defaultdict

# ----------------------------------------------------------------------------
# optional symbolic canonicalization
# ----------------------------------------------------------------------------
try:
    import sympy
    from sympy.parsing.sympy_parser import (
        parse_expr, standard_transformations, implicit_multiplication_application,
        convert_xor,
    )
    _TRANSF = standard_transformations + (
        implicit_multiplication_application, convert_xor)
    _HAVE_SYMPY = True
except Exception:
    _HAVE_SYMPY = False


# ----------------------------------------------------------------------------
# extraction helpers
# ----------------------------------------------------------------------------
def all_boxed(text):
    """Return list of \\boxed{...} contents, in order, using brace matching."""
    outs, i = [], 0
    while True:
        j = text.find('\\boxed', i)
        if j < 0:
            break
        k = text.find('{', j)
        if k < 0:
            break
        depth, m = 0, k
        while m < len(text):
            c = text[m]
            if c == '{':
                depth += 1
            elif c == '}':
                depth -= 1
                if depth == 0:
                    break
            m += 1
        outs.append(text[k + 1:m].strip())
        i = m + 1
    return outs


def split_top_level_commas(s):
    """Split on commas that are NOT inside (), [], {} — so '(1, 2)' stays whole
    and '1,000' inside parens isn't broken. Robust for multi-answer slots."""
    parts, depth, buf = [], 0, []
    for ch in s:
        if ch in '([{':
            depth += 1; buf.append(ch)
        elif ch in ')]}':
            depth = max(0, depth - 1); buf.append(ch)
        elif ch == ',' and depth == 0:
            parts.append(''.join(buf)); buf = []
        else:
            buf.append(ch)
    if buf:
        parts.append(''.join(buf))
    return [p.strip() for p in parts if p.strip() != '']


def is_truncated(resp):
    """Model never finished reasoning / never committed an answer -> abstain."""
    return ('</think>' not in resp) or (not all_boxed(resp))


_LETTER_RE = re.compile(r'\b([A-J])\b')

def extract_mcq_letters(resp, expected_n=1):
    """Pull answer letter(s) from the final box (or last letters seen)."""
    boxes = all_boxed(resp)
    src = boxes[-1] if boxes else resp[-200:]
    parts = split_top_level_commas(src) if ',' in src else [src]
    letters = []
    for p in parts:
        m = _LETTER_RE.findall(p.upper())
        if m:
            letters.append(m[-1])
    if not letters:  # last resort: scan tail
        m = _LETTER_RE.findall(src.upper())
        letters = m[-expected_n:] if m else []
    return letters


def extract_slots(resp, expected_n):
    """Return a list of `expected_n` raw sub-answer strings, or None if it
    can't be aligned (sample then abstains for this question)."""
    boxes = all_boxed(resp)
    if not boxes:
        return None
    last = boxes[-1]
    inside = split_top_level_commas(last)
    # Case 1: single box already holds the right number of comma parts
    if expected_n is None:
        # infer later by mode; just return the split of the last box
        return inside if inside else [last]
    if len(inside) == expected_n:
        return inside
    # Case 2: model used N separate \boxed{} -> take the last N
    if len(boxes) >= expected_n:
        tail = boxes[-expected_n:]
        # only accept if those boxes are atomic (not themselves comma-lists)
        if all(len(split_top_level_commas(b)) == 1 for b in tail):
            return [b.strip() for b in tail]
    # Case 3: count mismatch -> abstain (don't pollute the vote)
    return None


# ----------------------------------------------------------------------------
# normalization (for grouping equal answers when voting)
# ----------------------------------------------------------------------------
def _clean(s):
    s = s.strip()
    s = s.replace('\\left', '').replace('\\right', '')
    s = s.replace('\\,', '').replace('\\!', '').replace('\\ ', ' ')
    s = s.replace('$', '').strip()
    s = re.sub(r'\\text\{([^}]*)\}', r'\1', s)
    s = re.sub(r'\\mathrm\{([^}]*)\}', r'\1', s)
    return s.strip()

def to_float(s):
    t = _clean(s).replace(',', '').replace('\\', '')
    # bare fraction a/b
    m = re.fullmatch(r'\s*(-?\d+(?:\.\d+)?)\s*/\s*(-?\d+(?:\.\d+)?)\s*', t)
    if m:
        try:
            return float(m.group(1)) / float(m.group(2))
        except ZeroDivisionError:
            return None
    try:
        return float(t)
    except ValueError:
        return None

def round_half_up(x, nd=6):
    """Round (not truncate) to nd decimals — fixes the 52.666666 vs ...667 bug."""
    from decimal import Decimal, ROUND_HALF_UP
    try:
        q = Decimal(10) ** (-nd)
        return float(Decimal(repr(x)).quantize(q, rounding=ROUND_HALF_UP))
    except Exception:
        return round(x, nd)

def canon_key(s):
    """Canonical key used to GROUP equal answers for voting."""
    f = to_float(s)
    if f is not None:
        return ('num', round_half_up(f, 6))
    cl = _clean(s)
    if _HAVE_SYMPY:
        try:
            e = parse_expr(cl.replace('^', '**'), transformations=_TRANSF, evaluate=True)
            return ('sym', sympy.srepr(sympy.simplify(e)))
        except Exception:
            pass
    return ('str', re.sub(r'\s+', '', cl).lower())


def representative(raw_list):
    """Given raw strings that share the winning key, return the one to emit.
    Prefer a clean numeric form rounded to 6dp; else the most common raw."""
    f = to_float(raw_list[0])
    import math
    if f is not None and math.isfinite(f):
        r = round_half_up(f, 6)
        if not math.isfinite(r):
            return Counter(raw_list).most_common(1)[0][0]
        # integer -> no decimal point (matches the prompt rule the grader liked)
        if abs(r - round(r)) < 1e-9:
            return str(int(round(r)))
        return ('%.6f' % r).rstrip('0').rstrip('.')
    return Counter(raw_list).most_common(1)[0][0]


# ----------------------------------------------------------------------------
# voting
# ----------------------------------------------------------------------------
def vote_single(samples):
    groups = defaultdict(list)
    for resp in samples:
        if is_truncated(resp):
            continue
        boxes = all_boxed(resp)
        ans = boxes[-1]
        groups[canon_key(ans)].append(ans)
    if not groups:
        return ''  # everything truncated
    best = max(groups.values(), key=len)
    return representative(best)

def vote_mcq(samples, expected_n=1):
    cols = defaultdict(Counter)
    for resp in samples:
        if is_truncated(resp):
            continue
        letters = extract_mcq_letters(resp, expected_n)
        for idx, L in enumerate(letters[:expected_n]):
            cols[idx][L] += 1
    if not cols:
        return ''
    n = (max(cols) + 1) if expected_n is None else expected_n
    out = []
    for idx in range(n):
        if cols.get(idx):
            out.append(cols[idx].most_common(1)[0][0])
    return ', '.join(out)

def vote_multi(samples, expected_n):
    # infer slot count if unknown: mode of part-counts among finished samples
    if expected_n is None:
        counts = Counter()
        for resp in samples:
            if is_truncated(resp):
                continue
            slots = extract_slots(resp, None)
            if slots:
                counts[len(slots)] += 1
        if not counts:
            return ''
        expected_n = counts.most_common(1)[0][0]
    cols = [defaultdict(list) for _ in range(expected_n)]
    used = 0
    for resp in samples:
        if is_truncated(resp):
            continue
        slots = extract_slots(resp, expected_n)
        if slots is None or len(slots) != expected_n:
            continue
        used += 1
        for i, sl in enumerate(slots):
            cols[i][canon_key(sl)].append(sl)
    if used == 0:
        return ''
    out = []
    for i in range(expected_n):
        if not cols[i]:
            out.append('')
            continue
        best = max(cols[i].values(), key=len)
        out.append(representative(best))
    return ', '.join(out)


def aggregate(rec, skip_mcq_vote=False):
    samples = rec['samples']
    is_mcq = rec.get('is_mcq', False)
    n_ans = rec.get('n_ans')  # may be None
    gold = rec.get('gold')
    if n_ans is None and isinstance(gold, list):
        n_ans = len(gold)  # public set only; private set won't have this

    if is_mcq:
        if skip_mcq_vote:
            # pass through the first finished sample's letter(s)
            for resp in samples:
                if not is_truncated(resp):
                    return ', '.join(extract_mcq_letters(resp, n_ans or 1))
            voted = vote_mcq(samples, n_ans or 1)
        else:
            voted = vote_mcq(samples, n_ans or 1)
    elif (n_ans or 1) > 1:
        voted = vote_multi(samples, n_ans)
    else:
        voted = vote_single(samples)
    return voted


# ----------------------------------------------------------------------------
# approximate grader (only for estimating accuracy on the PUBLIC set)
# ----------------------------------------------------------------------------
def grade_one(pred, gold, is_mcq):
    """Approximation of the real grader. For estimation only."""
    gold = gold if isinstance(gold, list) else [gold]
    preds = split_top_level_commas(pred) if ',' in pred else [pred]
    if len(preds) != len(gold):
        return False
    for p, g in zip(preds, gold):
        if is_mcq:
            if p.strip().upper() != str(g).strip().upper():
                return False
            continue
        pf, gf = to_float(p), to_float(g)
        if pf is not None and gf is not None:
            denom = max(1.0, abs(gf))
            if abs(pf - gf) > 1e-4 and abs(pf - gf) / denom > 1e-4:
                return False
        else:
            if canon_key(p) != canon_key(g):
                return False
    return True


# ----------------------------------------------------------------------------
# I/O
# ----------------------------------------------------------------------------
def load_jsonl(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def merge_files(paths):
    """Fold k single-sample inference files (id,response,...) into combined form."""
    by_id = {}
    for p in paths:
        for r in load_jsonl(p):
            i = r['id']
            if i not in by_id:
                by_id[i] = {'id': i, 'is_mcq': r.get('is_mcq', False),
                            'samples': []}
                if 'gold' in r:
                    by_id[i]['gold'] = r['gold']
                if 'n_ans' in r:
                    by_id[i]['n_ans'] = r['n_ans']
            by_id[i]['samples'].append(r['response'])
    return [by_id[i] for i in sorted(by_id)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in', dest='inp', help='combined JSONL (form A)')
    ap.add_argument('--merge', nargs='+', help='k single-sample JSONL files (form B)')
    ap.add_argument('--out', required=True, help='output JSONL (voted)')
    ap.add_argument('--skip-mcq-vote', action='store_true',
                    help='pass MCQ through first sample (spend k only on free/multi)')
    ap.add_argument('--grade', action='store_true',
                    help='estimate accuracy if gold present (PUBLIC set only)')
    args = ap.parse_args()

    if args.merge:
        recs = merge_files(args.merge)
    elif args.inp:
        recs = load_jsonl(args.inp)
    else:
        sys.exit('provide --in or --merge')

    out_rows, n_correct, n_graded = [], 0, 0
    for rec in recs:
        voted = aggregate(rec, skip_mcq_vote=args.skip_mcq_vote)
        row = {'id': rec['id'], 'is_mcq': rec.get('is_mcq', False),
               'voted_answer': voted,
               # reconstructed response -> feeds your existing extractor/CSV
               'response': '</think>\n\\boxed{%s}' % voted}
        if 'gold' in rec:
            row['gold'] = rec['gold']
            if args.grade:
                ok = grade_one(voted, rec['gold'], rec.get('is_mcq', False))
                row['correct'] = ok
                n_correct += int(ok); n_graded += 1
        out_rows.append(row)

    with open(args.out, 'w') as f:
        for r in out_rows:
            f.write(json.dumps(r) + '\n')

    print('wrote %d voted answers -> %s' % (len(out_rows), args.out))
    if args.grade and n_graded:
        print('ESTIMATED accuracy (approx grader): %d/%d = %.2f%%' %
              (n_correct, n_graded, 100 * n_correct / n_graded))
        print('(approximate — confirm against the official scorer)')


if __name__ == '__main__':
    main()