"""Paper figures (Italian and English) from the evaluation files in runs/. Missing data: the figure is skipped.

Output: figures/fig_<name>_<lang>.pdf and .png, 12 cm wide (Springer SPAR single column), TrueType fonts.
Usage: .venv/bin/python scripts/make_figures.py
"""
import json
from pathlib import Path

import matplotlib
import matplotlib.ticker
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RUNS, OUT = ROOT / "runs", ROOT / "figures"
OUT.mkdir(exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "pdf.fonttype": 42, "ps.fonttype": 42,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
                     "xtick.major.width": 0.6, "ytick.major.width": 0.6})
CM = 1 / 2.54
C = {"teacher": "#243333", "classical": "#BAC7C1", "student": "#367266", "student_dr": "#8FB3A8"}
T = {
    "it": dict(teacher="Insegnante\n(stato vero)", classical="Pipeline classica\n(telecamera\n+ filtro)",
               student="Studente\n(solo\ntelecamera)", student_dr="Studente con\ncondizioni\ncasuali",
               success="Prese riuscite (%)", ratio="Rapporto di chiusura", rho_slow="ρ = 1/√2",
               t95="Tempo di chiusura della mano (s)", proto="prototipo (circa 10 mm/s)",
               aim="Errore di mira dell'operatore (cm)", base="condizioni nominali",
               cond=dict(aim006="mira σ = 6 cm", base="nominale (σ = 12 cm)", aim018="mira σ = 18 cm",
                         aim024="mira σ = 24 cm", camnoise010="rumore telecamera 1 cm",
                         camnoise020="rumore telecamera 2 cm", latency066="ritardo 66 ms", latency100="ritardo 100 ms",
                         camrate30="telecamera a 30 Hz", small_parcels="pacchi 5–8 cm",
                         heavy_parcels="pacchi 0,4–0,8 kg", spin15="rotazione fino a 15 rad/s",
                         fast_throws="lanci veloci (0,4–0,5 s)"), seeds="semi di addestramento",
               mass="Massa del pacco (kg)", tension="tensione per braccio"),
    "en": dict(teacher="Teacher\n(true state)", classical="Classical\npipeline\n(camera + filter)",
               student="Student\n(camera only)", student_dr="Student with\nrandomised\nconditions",
               success="Successful catches (%)", ratio="Closing ratio", rho_slow="ρ = 1/√2",
               t95="Hand closing time (s)", proto="prototype (about 10 mm/s)",
               aim="Operator aiming error (cm)", base="nominal conditions",
               cond=dict(aim006="aim σ = 6 cm", base="nominal (σ = 12 cm)", aim018="aim σ = 18 cm",
                         aim024="aim σ = 24 cm", camnoise010="camera noise 1 cm", camnoise020="camera noise 2 cm",
                         latency066="latency 66 ms", latency100="latency 100 ms", camrate30="camera at 30 Hz",
                         small_parcels="parcels 5–8 cm", heavy_parcels="parcels 0.4–0.8 kg",
                         spin15="spin up to 15 rad/s", fast_throws="fast throws (0.4–0.5 s)"),
               seeds="training seeds", mass="Parcel mass (kg)", tension="tension per arm"),
}


def load(p):
    p = RUNS / p
    return json.loads(p.read_text()) if p.exists() else None


def wilson(k, n, z=1.96):
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return c - h, c + h


def decimal_axis(ax, lang, axis="x"):
    if lang == "it":
        f = matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}".replace(".", ","))
        (ax.xaxis if axis == "x" else ax.yaxis).set_major_formatter(f)


def save(fig, name, lang):
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig_{name}_{lang}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig_methods(lang):
    base, stud = load("eval_base_500000.json"), load("eval_students_500000.json")
    if base is None:
        return
    t = T[lang]
    rows = [("teacher", base["table"]["teacher"]), ("classical", base["table"]["classical"]),
            ("student", base["table"]["dagger_v1"])]
    if stud and "dagger_dr" in stud["table"]:
        rows.append(("student_dr", stud["table"]["dagger_dr"]))
    fig, ax = plt.subplots(figsize=(12 * CM, 4.6 * CM))
    for i, (k, r) in enumerate(rows):
        lo, hi = wilson(r["success"], r["n"])
        v = 100 * r["success"] / r["n"]
        ax.bar(i, v, color=C[k], width=0.6)
        ax.errorbar(i, v, yerr=[[v - 100 * lo], [100 * hi - v]], color="black", capsize=3, lw=0.8)
        ax.text(i, 100 * hi + 1.0, f"{v:.1f}".replace(".", "," if lang == "it" else "."), ha="center", va="bottom")
    if stud:
        seeds = [100 * stud["table"][s]["success"] / stud["table"][s]["n"] for s in ("dagger_s1", "dagger_s2")
                 if s in stud["table"]]
        ax.plot([2] * len(seeds), seeds, "o", ms=3, mfc="white", mec="black", mew=0.6, label=t["seeds"])
        ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.02))
    ax.set_xticks(range(len(rows)), [t[k] for k, _ in rows], fontsize=7)
    ax.set_ylim(70, 103)
    ax.set_ylabel(t["success"])
    save(fig, "metodi", lang)


def teacher_rate(name):
    d = load(f"sweeps/{name}.json")
    if d is None:
        return None
    r = d["results"]
    return sum(x["success"] for x in r), len(r)


def fig_ratio(lang):
    ratios = [0.1, 0.2, 0.3, 0.45, 0.6, 0.71, 0.9]
    pts = [(r, teacher_rate(f"ratio_{r}")) for r in ratios]
    pts = [(r, kn) for r, kn in pts if kn]
    if len(pts) < 3:
        return
    t = T[lang]
    fig, ax = plt.subplots(figsize=(12 * CM, 4.4 * CM))
    x = [r for r, _ in pts]
    y = [100 * k / n for _, (k, n) in pts]
    ci = [wilson(k, n) for _, (k, n) in pts]
    ax.fill_between(x, [100 * a for a, _ in ci], [100 * b for _, b in ci], color=C["classical"], lw=0)
    ax.plot(x, y, "o-", color=C["teacher"], ms=3.5, lw=1)
    ax.axvline(1 / np.sqrt(2), color=C["student"], ls="--", lw=0.8)
    ax.text(1 / np.sqrt(2) + 0.015, 35, t["rho_slow"], color=C["student"], ha="left")
    ax.set_xlabel(t["ratio"])
    decimal_axis(ax, lang)
    ax.set_ylabel(t["success"])
    ax.set_ylim(0, 102)
    save(fig, "chiusura", lang)


def fig_motor(lang):
    ct = load("closing_times.json")
    if ct is None:
        return
    pts = []
    for v, t95 in ((0.25, ct["0.25"]["0.95"]), (0.125, ct["0.125"]["0.95"]), (0.0625, ct["0.0625"]["0.95"]),
                   (0.03, ct["0.03"]["0.95"]), (0.01, ct["0.01"]["0.95"])):
        best = None
        for r in (0.3, 0.45, 0.6):
            kn = teacher_rate(f"ratio_{r}") if v == 0.25 else teacher_rate(f"motor_{v}_ratio_{r}")
            if kn and (best is None or kn[0] > best[0]):
                best = kn
        if best:
            pts.append((t95, best))
    if len(pts) < 3:
        return
    t = T[lang]
    fig, ax = plt.subplots(figsize=(12 * CM, 4.4 * CM))
    x = [a for a, _ in pts]
    y = [100 * k / n for _, (k, n) in pts]
    ci = [wilson(k, n) for _, (k, n) in pts]
    ax.fill_between(x, [100 * a for a, _ in ci], [100 * b for _, b in ci], color=C["classical"], lw=0)
    ax.plot(x, y, "o-", color=C["teacher"], ms=3.5, lw=1)
    ax.set_xscale("log")
    ax.set_xticks(x, [f"{a:.2f}".replace(".", "," if lang == "it" else ".") for a in x])
    ax.minorticks_off()
    ax.annotate(t["proto"], (x[-1], y[-1]), xytext=(0, 40), textcoords="offset points", ha="right",
                arrowprops=dict(arrowstyle="-", lw=0.6))
    ax.set_xlabel(t["t95"])
    ax.set_ylabel(t["success"])
    ax.set_ylim(0, 102)
    save(fig, "motore", lang)


def fig_robust(lang):
    conds = ["base", "aim006", "aim018", "aim024", "camnoise010", "camnoise020", "latency066", "latency100",
             "camrate30", "fast_throws", "spin15", "heavy_parcels", "small_parcels"]
    base = load("eval_base_500000.json")
    stud = load("eval_students_500000.json")
    if base is None:
        return
    data = {}
    for c in conds:
        d = base if c == "base" else load(f"sweeps/{c}.json")
        ddr = stud if c == "base" else load(f"sweeps/{c}_dr.json")
        if d is None:
            continue
        row = {k: d["table"][m] for k, m in (("teacher", "teacher"), ("classical", "classical"),
                                              ("student", "dagger_v1"))}
        if ddr and "dagger_dr" in ddr["table"]:
            row["student_dr"] = ddr["table"]["dagger_dr"]
        data[c] = row
    if len(data) < 3:
        return
    t = T[lang]
    keys = ["teacher", "classical", "student", "student_dr"]
    fig, ax = plt.subplots(figsize=(12 * CM, 0.55 * CM * len(data) + 1.6 * CM))
    h = 0.2
    for i, c in enumerate(data):
        for j, k in enumerate(keys):
            if k in data[c]:
                r = data[c][k]
                ax.barh(i + (j - 1.5) * h, 100 * r["success"] / r["n"], height=h, color=C[k],
                        label=t[k].replace("\n", " ") if i == 0 else None)
    ax.set_yticks(range(len(data)), [t["cond"][c] for c in data])
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel(t["success"])
    ax.legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.45, -0.12 - 0.02 * (13 - len(data))))
    save(fig, "robustezza", lang)


def fig_aim(lang):
    base = load("eval_base_500000.json")
    extra = [load(f"sweeps/{c}.json") for c in ("aim018", "aim024")]
    if base is None:
        return
    eps = base["episodes"] + [e for d in extra if d for e in d["episodes"]]
    bins = [0, 10, 20, 30, 40, 60]
    t = T[lang]
    fig, ax = plt.subplots(figsize=(12 * CM, 4.4 * CM))
    for k, m in (("teacher", "teacher"), ("classical", "classical"), ("student", "dagger_v1")):
        xs, ys = [], []
        for lo, hi in zip(bins[:-1], bins[1:]):
            sel = [e for e in eps if e["method"] == m and lo <= 100 * e["aim_err"] < hi]
            if len(sel) >= 20:
                xs.append((lo + hi) / 2)
                ys.append(100 * np.mean([e["success"] for e in sel]))
        ax.plot(xs, ys, "o-", color=C[k], ms=3.5, lw=1, label=t[k].replace("\n", " "))
    ax.set_xlabel(t["aim"])
    ax.set_ylabel(t["success"])
    ax.set_ylim(0, 102)
    ax.legend(frameon=False, loc="lower left")
    save(fig, "mira", lang)


def fig_mass(lang):
    t = T[lang]
    fig, ax = plt.subplots(figsize=(12 * CM, 4.4 * CM))
    bins = np.arange(0.1, 0.81, 0.1)
    n_ok = 0
    for T_, col in ((20, C["classical"]), (40, C["student"]), (80, C["teacher"])):
        d = load(f"sweeps/mass_T{T_}.json")
        if d is None:
            continue
        n_ok += 1
        r = d["results"]
        xs, ys = [], []
        for lo, hi in zip(bins[:-1], bins[1:]):
            sel = [x for x in r if lo <= x["mass"] < hi]
            if sel:
                xs.append((lo + hi) / 2)
                ys.append(100 * np.mean([x["success"] for x in sel]))
        ax.plot(xs, ys, "o-", color=col, ms=3.5, lw=1, label=f"{T_} N {t['tension']}")
    if n_ok == 0:
        plt.close(fig)
        return
    ax.set_xlabel(t["mass"])
    ax.set_ylabel(t["success"])
    ax.set_ylim(0, 102)
    decimal_axis(ax, lang)
    ax.legend(frameon=False, loc="lower left")
    save(fig, "massa", lang)


if __name__ == "__main__":
    for lang in ("it", "en"):
        for f in (fig_methods, fig_ratio, fig_motor, fig_robust, fig_aim, fig_mass):
            f(lang)
    print(sorted(p.name for p in OUT.glob("*.pdf")))
