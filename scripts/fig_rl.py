"""Figures of the RL comparison (catch-parameter policies v2), Italian and English.

Input: runs/eval_v2_final.json (scripts/eval_high2.py). Output: figures/fig_rl_successo_<lang>, fig_rl_acc_<lang>,
fig_rl_adatta_<lang> (PDF and PNG, 12 cm wide, TrueType fonts).
Usage: .venv/bin/python scripts/fig_rl.py --methods-json runs/methods_v2.json
"""
import argparse
import json
from pathlib import Path

import matplotlib
import matplotlib.ticker
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "pdf.fonttype": 42, "ps.fonttype": 42,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6})
CM = 1 / 2.54
COLS = ["#BAC7C1", "#8FB3A8", "#5E8F82", "#367266", "#243333", "#0F1A1A"]
COND = {"it": dict(nominal="nominali", broad="variabili\n(addestramento)", heavy="pacchi\n0,4–0,8 kg",
                   slow_motor="motore lento\n(0,13 m/s)", fast="lanci veloci", very_slow="motore molto\nlento (0,10 m/s)"),
        "en": dict(nominal="nominal", broad="varied\n(training)", heavy="parcels\n0.4–0.8 kg",
                   slow_motor="slow motor\n(0.13 m/s)", fast="fast throws", very_slow="very slow\nmotor (0.10 m/s)")}
LAB = {"it": dict(success="Prese riuscite (%)", acc="Accelerazione del pacco,\nmediana (g)",
                  ratio="Rapporto di chiusura scelto", vmotor="Velocità del motore (m/s)",
                  turn="Rotazione palmo in su scelta", mass="Massa del pacco (kg)", rule="regola adattiva"),
       "en": dict(success="Successful catches (%)", acc="Parcel acceleration,\nmedian (g)",
                  ratio="Chosen closing ratio", vmotor="Motor speed (m/s)", turn="Chosen palm-up turn gain",
                  mass="Parcel mass (kg)", rule="adaptive rule")}


def save(fig, name, lang):
    OUT.mkdir(exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig_{name}_{lang}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="runs/eval_v2_final.json")
    ap.add_argument("--methods-json", required=True, help="JSON list of [method spec, it label, en label]")
    ap.add_argument("--adapt-method", default=None, help="method spec whose choices are plotted")
    a = ap.parse_args()
    d = json.loads((ROOT / a.data).read_text())
    methods = json.loads((ROOT / a.methods_json).read_text())
    conds = [c for c in COND["it"] if any(k.startswith(c + " |") for k in d["table"])]
    for lang in ("it", "en"):
        li = 1 if lang == "it" else 2
        for key, name, ylab in (("success", "rl_successo", LAB[lang]["success"]), ("acc_median", "rl_acc", LAB[lang]["acc"])):
            fig, ax = plt.subplots(figsize=(12 * CM, 5.0 * CM))
            w = 0.8 / len(methods)
            for j, m in enumerate(methods):
                vals = []
                for c in conds:
                    s = d["table"].get(f"{c} | {m[0]}")
                    v = None if s is None else s[key]
                    vals.append(np.nan if v is None else (100 * v if key == "success" else v))
                ax.bar(np.arange(len(conds)) + (j - (len(methods) - 1) / 2) * w, vals, width=w, color=COLS[j % len(COLS)],
                       label=m[li])
            ax.set_xticks(range(len(conds)), [COND[lang][c] for c in conds], fontsize=7)
            ax.set_ylabel(ylab)
            if key == "success":
                ax.set_ylim(0, 100)
            ax.legend(frameon=False, ncol=3, fontsize=6.5, loc="lower center", bbox_to_anchor=(0.5, 1.0))
            save(fig, name, lang)
        if a.adapt_method:
            import sys
            sys.path.insert(0, str(ROOT))
            from catch.env import closing_time
            from catch.theta2 import to_kwargs
            rows = [r for r in d["rows"] if r["method"] == a.adapt_method and r["condition"] == "broad"]
            if rows:
                k = [to_kwargs(r["u"]) for r in rows]
                v = np.array([r["speed_motor"] for r in rows])
                mass = np.array([r["mass"] for r in rows])
                fig, (a1, a2) = plt.subplots(1, 2, figsize=(12 * CM, 4.6 * CM))
                a1.scatter(v, [x["ratio"] for x in k], s=3, color=COLS[3], alpha=0.6)
                vv = np.linspace(v.min(), v.max(), 50)
                a1.plot(vv, [min(0.95, 0.55 * closing_time(x) / closing_time(0.25)) for x in vv], color=COLS[5],
                        lw=0.8, ls="--", label=LAB[lang]["rule"])
                a1.set_xlabel(LAB[lang]["vmotor"])
                a1.set_ylabel(LAB[lang]["ratio"])
                a1.legend(frameon=False, fontsize=6.5)
                a2.scatter(mass, [x["turn_gain"] for x in k], s=3, color=COLS[4], alpha=0.6)
                a2.set_xlabel(LAB[lang]["mass"])
                a2.set_ylabel(LAB[lang]["turn"])
                if lang == "it":
                    for ax_ in (a1, a2):
                        ax_.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: f"{x:g}".replace(".", ",")))
                        ax_.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: f"{x:g}".replace(".", ",")))
                fig.tight_layout()
                save(fig, "rl_adatta", lang)
    print("figures written")


if __name__ == "__main__":
    main()
