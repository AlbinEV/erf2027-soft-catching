"""Markdown tables (Italian decimals) of the v2 comparison: success with 95 % Wilson interval, median parcel
acceleration and contact force on successful catches, mean reward, per condition and method.
Usage: .venv/bin/python scripts/table_v2.py runs/eval_v2_final.json runs/methods_v2.json
"""
import json
import sys

import numpy as np


def wilson(k, n, z=1.96):
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return c - h, c + h


def it(x, nd=1):
    return "–" if x is None else f"{x:.{nd}f}".replace(".", ",")


def main():
    d = json.load(open(sys.argv[1]))
    methods = json.load(open(sys.argv[2]))
    conds = list(d["conditions"])
    for c in conds:
        print(f"\n**{c}**\n")
        print("| Metodo | Prese | IC 95 % | Accel. pacco (g) | Forza (N) | Ricompensa |")
        print("|---|---|---|---|---|---|")
        for m, lab_it, _ in methods:
            s = d["table"].get(f"{c} | {m}")
            if s is None:
                continue
            k = round(s["success"] * s["n"])
            lo, hi = wilson(k, s["n"])
            print(f"| {lab_it} | {it(100 * s['success'])} % | {it(100 * lo)}–{it(100 * hi)} | {it(s['acc_median'])} | "
                  f"{it(s['force_median'], 0)} | {it(s['reward'], 3)} |")


if __name__ == "__main__":
    main()
