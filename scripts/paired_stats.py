"""Paired comparison of catch success on the same throws (same seeds) between two methods, per condition.

For each condition: success of A and B, discordant throws (A only, B only), difference A - B in points with a 95 %
Wald interval for paired proportions, and the exact two-sided McNemar p-value (binomial test on discordant throws).
Input: an evaluation JSON with "rows" (method, condition, seed, success), e.g. runs/eval_v2_final.json; several files
can be given, rows are merged. Methods are matched by exact name or by a unique prefix.
Usage: .venv/bin/python scripts/paired_stats.py runs/eval_v2_final.json --a rl:hi2_cse --b 'fixed:{"ratio":0.45'
"""
import argparse
import json
from math import comb, sqrt


def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def resolve(name, methods):
    if name in methods:
        return name
    hits = [m for m in methods if m.startswith(name)]
    if len(hits) != 1:
        raise SystemExit(f"method {name!r}: {len(hits)} matches among {sorted(methods)}")
    return hits[0]


def compare(rows, a, b):
    out = {}
    for cond in dict.fromkeys(r["condition"] for r in rows):
        ra = {r["seed"]: bool(r["success"]) for r in rows if r["method"] == a and r["condition"] == cond}
        rb = {r["seed"]: bool(r["success"]) for r in rows if r["method"] == b and r["condition"] == cond}
        seeds = sorted(set(ra) & set(rb))
        if not seeds:
            continue
        n = len(seeds)
        only_a = sum(ra[s] and not rb[s] for s in seeds)
        only_b = sum(rb[s] and not ra[s] for s in seeds)
        d = (only_a - only_b) / n
        se = sqrt(max(only_a + only_b - (only_a - only_b) ** 2 / n, 0.0)) / n
        out[cond] = dict(n=n, succ_a=sum(ra[s] for s in seeds) / n, succ_b=sum(rb[s] for s in seeds) / n,
                         only_a=only_a, only_b=only_b, diff=d, lo=d - 1.96 * se, hi=d + 1.96 * se,
                         p=mcnemar_exact(only_a, only_b))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--json", action="store_true", help="print JSON instead of a table")
    a = ap.parse_args()
    rows = [r for f in a.files for r in json.load(open(f))["rows"]]
    methods = {r["method"] for r in rows}
    ma, mb = resolve(a.a, methods), resolve(a.b, methods)
    res = compare(rows, ma, mb)
    if a.json:
        print(json.dumps(dict(a=ma, b=mb, conditions=res), indent=1))
        return
    print(f"A = {ma}\nB = {mb}")
    for cond, s in res.items():
        print(f"{cond:12s} n {s['n']}  A {100 * s['succ_a']:5.1f} %  B {100 * s['succ_b']:5.1f} %  "
              f"A-B {100 * s['diff']:+5.1f} [{100 * s['lo']:+5.1f}, {100 * s['hi']:+5.1f}]  "
              f"discordant {s['only_a']}/{s['only_b']}  McNemar p {s['p']:.3g}")


if __name__ == "__main__":
    main()
