"""Combined figures of the ERF2027 paper (Italian and English), same style as scripts/make_figures.py.

fig_requisiti: (a) closing ratio, (b) hand closing time, (c) parcel mass for three tendon tensions (teacher).
fig_cedevolezza: rigid versus compliant hand on the 500 test throws: success, median peak contact force, median peak
    parcel acceleration (teacher, classical pipeline, student).
fig_rl_delta: paired difference in catch success with respect to the nominal fixed rule, per condition, with 95 %
    interval (hand-made adaptive rule, RL supervisors, rule distilled from the supervisor; seed-1 supervisors as open
    markers when available).
Inputs: runs/sweeps/*.json, runs/closing_times.json, runs/eval_rigid_500000.json (or eval_base_500000.json),
runs/eval_compliant_500000.json, runs/eval_v2_final.json, runs/eval_v2_distilled.json, runs/eval_v2_seed1.json.
Usage: .venv/bin/python scripts/paper_figures.py
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_figures import CM, C, OUT, RUNS, decimal_axis, load, plt, save, teacher_rate, wilson  # noqa: E402
from paired_stats import compare  # noqa: E402

L = {
    "it": dict(success="Prese riuscite (%)", ratio="Rapporto di chiusura",
               t95="Tempo di chiusura (s)", mass="Massa del pacco (kg)", tension="tensione per braccio",
               rigid="mano rigida", compliant="palmo e polso cedevoli", force="Forza mediana (N)",
               acc="Accelerazione\nmediana (g)", teacher="Insegnante", classical="Pipeline classica",
               student="Studente", delta="Differenza di prese rispetto alla regola nominale (punti percentuali)",
               adaptive="regola adattiva (encoder)", rl_cam="RL: solo telecamera",
               rl_cse="RL: + dimensione ed encoder", seed1="secondo seme di addestramento",
               distilled="regola ricavata dal supervisore (encoder)",
               cond=dict(nominal="nominali", broad="variabili (come in addestramento)", heavy="pacchi 0,4–0,8 kg",
                         slow_motor="motore lento (0,13 m/s)", very_slow="motore 0,10 m/s (fuori addestramento)",
                         fast="lanci veloci (0,45–0,55 s)")),
    "en": dict(success="Successful catches (%)", ratio="Closing ratio",
               t95="Closing time (s)", mass="Parcel mass (kg)", tension="tension per arm",
               rigid="rigid hand", compliant="compliant palm and wrist", force="Median force (N)",
               acc="Median parcel\nacceleration (g)", teacher="Teacher", classical="Classical pipeline",
               student="Student", delta="Difference in catches from the nominal rule (percentage points)",
               adaptive="adaptive rule (encoder)", rl_cam="RL: camera only", rl_cse="RL: + size and encoder",
               distilled="rule distilled from the supervisor (encoder)",
               seed1="second training seed",
               cond=dict(nominal="nominal", broad="varied (as in training)", heavy="parcels 0.4–0.8 kg",
                         slow_motor="slow motor (0.13 m/s)", very_slow="motor 0.10 m/s (outside training)",
                         fast="fast throws (0.45–0.55 s)")),
}
NOMINAL_RULE = 'fixed:{"ratio":0.45,"absorb":0.4,"brake_time":0.0,"turn_gain":0.0,"motor_hold":1.0}'
ADAPTIVE_RULE = 'adaptive:{"ratio":0.55,"absorb":0.4,"brake_time":0.4,"turn_gain":4.0,"motor_hold":1.0}'
# rule distilled from the RL supervisor: its mean parameters on the 300 selection throws (seeds 100000+), with the
# closing ratio scaled by the encoder-measured motor speed as in the adaptive rule
DISTILLED_RULE = 'adaptive:{"ratio":0.56,"absorb":0.52,"brake_time":0.25,"turn_gain":6.0,"motor_hold":0.83}'


def fmt(v, lang, nd=2):
    s = f"{v:.{nd}f}"
    return s.replace(".", ",") if lang == "it" else s


def fig_requisiti(lang):
    t = L[lang]
    fig, axs = plt.subplots(1, 3, figsize=(12 * CM, 3.6 * CM), sharey=True)
    # (a) closing ratio
    ratios = [0.1, 0.2, 0.3, 0.45, 0.6, 0.71, 0.9]
    pts = [(r, teacher_rate(f"ratio_{r}")) for r in ratios]
    pts = [(r, kn) for r, kn in pts if kn]
    ax = axs[0]
    x = [r for r, _ in pts]
    ci = [wilson(k, n) for _, (k, n) in pts]
    ax.fill_between(x, [100 * a for a, _ in ci], [100 * b for _, b in ci], color=C["classical"], lw=0)
    ax.plot(x, [100 * k / n for _, (k, n) in pts], "o-", color=C["teacher"], ms=3, lw=1)
    ax.set_xlabel(t["ratio"])
    ax.set_xticks([0.1, 0.3, 0.5, 0.7, 0.9])
    decimal_axis(ax, lang)
    ax.set_ylabel(t["success"])
    ax.set_ylim(0, 102)
    # (b) closing time (best ratio among 0.3, 0.45, 0.6 for each motor speed)
    ct = load("closing_times.json")
    pts = []
    for v in (0.25, 0.125, 0.0625, 0.03, 0.01):
        best = None
        for r in (0.3, 0.45, 0.6):
            kn = teacher_rate(f"ratio_{r}") if v == 0.25 else teacher_rate(f"motor_{v}_ratio_{r}")
            if kn and (best is None or kn[0] > best[0]):
                best = kn
        if best:
            pts.append((ct[str(v)]["0.95"], best))
    ax = axs[1]
    x = [a for a, _ in pts]
    ci = [wilson(k, n) for _, (k, n) in pts]
    ax.fill_between(x, [100 * a for a, _ in ci], [100 * b for _, b in ci], color=C["classical"], lw=0)
    ax.plot(x, [100 * k / n for _, (k, n) in pts], "o-", color=C["teacher"], ms=3, lw=1)
    ax.set_xscale("log")
    ax.set_xticks([0.2, 0.4, 0.8, 2.0], [fmt(v, lang, 1) for v in (0.2, 0.4, 0.8, 2.0)])
    ax.minorticks_off()
    ax.set_xlabel(t["t95"])
    # (c) parcel mass and tendon tension
    ax = axs[2]
    bins = np.arange(0.1, 0.81, 0.1)
    for T_, col in ((20, C["classical"]), (40, C["student"]), (80, C["teacher"])):
        d = load(f"sweeps/mass_T{T_}.json")
        if d is None:
            continue
        r = d["results"]
        xs, ys = [], []
        for lo, hi in zip(bins[:-1], bins[1:]):
            sel = [e for e in r if lo <= e["mass"] < hi]
            if sel:
                xs.append((lo + hi) / 2)
                ys.append(100 * np.mean([e["success"] for e in sel]))
        ax.plot(xs, ys, "o-", color=col, ms=3, lw=1, label=f"{T_} N")
    ax.set_xlabel(t["mass"])
    ax.set_xticks([0.2, 0.4, 0.6, 0.8])
    decimal_axis(ax, lang)
    ax.legend(frameon=False, loc="lower left", fontsize=6, handlelength=1.0, title=t["tension"], title_fontsize=6,
              borderaxespad=0.1, ncol=3, columnspacing=0.6, handletextpad=0.3)
    for ax, lab in zip(axs, "abc"):
        ax.text(-0.02, 1.04, f"({lab})", transform=ax.transAxes, ha="left", va="bottom", fontweight="bold")
    fig.tight_layout(w_pad=0.6)
    save(fig, "requisiti", lang)


def _per_method(d):
    out = {}
    for e in d["episodes"]:
        out.setdefault(e["method"], []).append(e)
    return out


def fig_cedevolezza(lang):
    rig = load("eval_rigid_500000.json") or load("eval_base_500000.json")
    com = load("eval_compliant_500000.json")
    if rig is None or com is None:
        return
    t = L[lang]
    hands = (("rigid", _per_method(rig), C["classical"]), ("compliant", _per_method(com), C["student"]))
    methods = (("teacher", "teacher"), ("student", "dagger_v1"), ("classical", "classical"))
    fig, axs = plt.subplots(1, 3, figsize=(12 * CM, 3.4 * CM))
    w = 0.38
    for j, (hk, eps, col) in enumerate(hands):
        for i, (mk, m) in enumerate(methods):
            es = eps[m]
            ok = [e for e in es if e["success"]]
            k, n = len(ok), len(es)
            lo, hi = wilson(k, n)
            v = 100 * k / n
            xi = i + (j - 0.5) * w
            axs[0].bar(xi, v, w, color=col, label=t[hk] if i == 0 else None)
            axs[0].errorbar(xi, v, yerr=[[max(v - 100 * lo, 0)], [max(100 * hi - v, 0)]], color="black",
                            capsize=1.5, lw=0.6)
            axs[1].bar(xi, np.median([e["max_force"] for e in ok]), w, color=col)
            acc = [e["max_acc_g"] for e in ok if e.get("max_acc_g") is not None]
            if acc:
                axs[2].bar(xi, np.median(acc), w, color=col)
    for ax, lab in zip(axs, (t["success"], t["force"], t["acc"])):
        ax.set_xticks(range(len(methods)), [t[mk] for mk, _ in methods], fontsize=6.5, rotation=30, ha="right",
                      rotation_mode="anchor")
        ax.set_ylabel(lab)
    axs[0].set_ylim(80, 101)
    fig.legend(*axs[0].get_legend_handles_labels(), frameon=False, ncol=2, loc="lower center",
               bbox_to_anchor=(0.5, 1.0))
    for ax, lab in zip(axs, "abc"):
        ax.text(-0.02, 1.04, f"({lab})", transform=ax.transAxes, ha="left", va="bottom", fontweight="bold")
    fig.tight_layout(w_pad=0.8)
    save(fig, "cedevolezza", lang)


def fig_rl_delta(lang):
    d0 = load("eval_v2_final.json")
    if d0 is None:
        return
    d1, dd = load("eval_v2_seed1.json"), load("eval_v2_distilled.json")
    t = L[lang]
    rows = d0["rows"] + (dd["rows"] if dd else []) + (d1["rows"] if d1 else [])
    methods = {r["method"] for r in rows}
    series = [(ADAPTIVE_RULE, "adaptive", C["classical"], "s"), ("rl:hi2_cam", "rl_cam", "#8FB3A8", "^"),
              ("rl:hi2_cse", "rl_cse", C["teacher"], "o"), (DISTILLED_RULE, "distilled", C["student"], "D")]
    series = [x for x in series if x[0] in methods]
    conds = ["nominal", "fast", "broad", "heavy", "slow_motor", "very_slow"]
    fig, ax = plt.subplots(figsize=(12 * CM, 4.3 * CM))
    offs = np.linspace(-0.27, 0.27, len(series))
    for si, (m, key, col, mk) in enumerate(series):
        res = compare(rows, m, NOMINAL_RULE)
        for ci, c in enumerate(conds):
            s = res[c]
            y = ci + offs[si]
            ax.errorbar(100 * s["diff"], y, xerr=[[100 * (s["diff"] - s["lo"])], [100 * (s["hi"] - s["diff"])]],
                        fmt=mk, color=col, ms=3.2, lw=0.8, capsize=1.5, label=t[key] if ci == 0 else None)
        if m.startswith("rl:") and m + "_s1" in methods:
            res1 = compare(rows, m + "_s1", NOMINAL_RULE)
            for ci, c in enumerate(conds):
                if c in res1:
                    ax.plot(100 * res1[c]["diff"], ci + offs[si], mk, ms=4.5, mfc="none", mec="black", mew=0.5,
                            label=t["seed1"] if (ci == 0 and m == "rl:hi2_cse") else None)
    ax.axvline(0, color="black", lw=0.6)
    ax.set_yticks(range(len(conds)), [t["cond"][c] for c in conds])
    ax.invert_yaxis()
    ax.set_xlabel(t["delta"])
    decimal_axis(ax, lang)
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.4, 1.0), ncol=2, fontsize=6.5)
    save(fig, "rl_delta", lang)


if __name__ == "__main__":
    for lang in ("it", "en"):
        for f in (fig_requisiti, fig_cedevolezza, fig_rl_delta):
            f(lang)
    print(sorted(p.name for p in OUT.glob("fig_*_it.pdf")))
