"""Figure of the ERF2027 paper (Italian and English): how far the catch generalises, designed hand (19-20 Oct 2026).

For each test condition: classical pipeline with its Kalman filter tuned on the selection throws (measurement noise
6 cm), student trained in restricted conditions (nominal throws, ideal camera or the camera at one position) and student
trained in varied conditions that cover the test, with 95 % Wilson intervals; the teacher (true state) as a vertical
tick, the classical pipeline with the untuned filter (measurement noise equal to the camera noise) as a cross and,
where available, the classical pipeline tuned for that condition (open diamond).
Inputs: runs/night/{cam,campos,wide,widecam}/*.json (runs/chain_2026-10-19_night*.sh), runs/explore/kf_tuned/ and
runs/explore/kf_cond/ (runs/chain_2026-10-20_explore6.sh, explore7.sh).
Usage: .venv/bin/python scripts/fig_generalizzazione.py
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_figures import CM, C, RUNS, decimal_axis, plt, save, wilson  # noqa: E402

N = RUNS
# condition key, (file, method) for classical, restricted student, varied student, teacher (or None)
ROWS = [
    ("nominal", ("explore/kf_tuned/m0.06_nominal", "classical"), ("night/cam/ideal_base", "dagger_proto"),
     ("night/widecam/nominal_ideal", "dagger_proto_all"), ("night/cam/ideal_base", "teacher"), None),
    ("camera", ("explore/kf_tuned/m0.06_pinhole", "classical"), ("night/cam/pinhole_base", "dagger_proto"),
     ("night/campos/behind", "dagger_proto_campos"), ("night/cam/pinhole_base", "teacher"), None),
    ("camera_worst", ("explore/kf_tuned/m0.06_pinhole_worst", "classical"), ("night/cam/worst_base", "dagger_proto"),
     ("night/cam/worst_student_cam", "dagger_proto_cam"), ("night/cam/worst_base", "teacher"),
     ("explore/kf_cond/pinhole_worst_test_m*", "classical")),
    ("camera_side", ("explore/kf_tuned/m0.06_side", "classical"), ("night/campos/side", "dagger_proto_cam"),
     ("night/campos/side", "dagger_proto_campos"), ("night/cam/side_base", "teacher"), None),
    ("camera_random", ("explore/kf_tuned/m0.06_random_worst", "classical"),
     ("night/campos/random_worst", "dagger_proto_cam"), ("night/campos/random_worst", "dagger_proto_campos"), None,
     None),
    ("wide", ("explore/kf_tuned/m0.06_wide", "classical"), ("night/wide/wide_base", "dagger_proto"),
     ("night/wide/wide_student_wide", "dagger_proto_wide"), ("night/wide/wide_base", "teacher"),
     ("explore/kf_cond/wide_test_m*", "classical")),
    ("wide_camera", ("explore/kf_tuned/m0.06_wide_randompos", "classical"),
     ("night/widecam/wide_randompos", "dagger_proto_widecam"),
     ("night/widecam/wide_randompos_long", "dagger_proto_all_long"), ("night/wide/wide_base", "teacher"),
     ("explore/kf_cond/wide_randompos_test_m*", "classical")),
]
# classical pipeline with the untuned filter (same throws as the row)
UNTUNED = {"nominal": "night/cam/ideal_base", "camera": "night/cam/pinhole_base", "camera_worst": "night/cam/worst_base",
           "camera_side": "night/campos/side", "camera_random": "night/campos/random_worst", "wide": "night/wide/wide_base",
           "wide_camera": "night/widecam/wide_randompos"}
L = {
    "it": dict(x="Prese riuscite (%)", classical="Pipeline classica (filtro tarato)",
               narrow="Studente, addestramento ristretto", broad="Studente, addestramento variato",
               teacher="Insegnante (stato vero)", cond_tuned="Classica tarata per la condizione",
               untuned="Classica, filtro non tarato",
               cond=dict(nominal="lanci nominali, telecamera ideale",
                         camera="telecamera realistica dietro il robot",
                         camera_worst="+ calibrazione 2 cm e 2°, rumore doppio,\nritardo 66 ms, 10 % fotogrammi persi",
                         camera_side="telecamera spostata a lato",
                         camera_random="telecamera in posizione casuale, degradata",
                         wide="lanci vari (±45°, 1–4 m, volo 0,3–1 s)",
                         wide_camera="lanci vari, telecamera in posizione casuale")),
    "en": dict(x="Successful catches (%)", classical="Classical pipeline (tuned filter)",
               narrow="Student, restricted training", broad="Student, varied training",
               teacher="Teacher (true state)", cond_tuned="Classical tuned for the condition",
               untuned="Classical, untuned filter",
               cond=dict(nominal="nominal throws, ideal camera",
                         camera="realistic camera behind the robot",
                         camera_worst="+ calibration 2 cm and 2°, double noise,\n66 ms latency, 10 % frames lost",
                         camera_side="camera moved to the side",
                         camera_random="camera at random positions, degraded",
                         wide="varied throws (±45°, 1–4 m, flight 0.3–1 s)",
                         wide_camera="varied throws, camera at random positions")),
}


def rate(spec):
    f, m = spec
    if "*" in f:
        f = str(sorted(N.glob(f + ".json"))[0].relative_to(N))[:-5]
    d = json.loads((N / f"{f}.json").read_text())
    e = [x for x in d["episodes"] if x["method"] == m]
    k = sum(x["success"] for x in e)
    return k, len(e)


def fig(lang):
    t = L[lang]
    fig_, ax = plt.subplots(figsize=(12 * CM, 5.1 * CM))
    series = [(1, "classical", C["classical"]), (2, "narrow", C["student_dr"]), (3, "broad", C["student"])]
    h = 0.24
    for ri, row in enumerate(ROWS):
        for si, key, col in series:
            k, n = rate(row[si])
            lo, hi = wilson(k, n)
            y = ri + (si - 2) * h
            ax.barh(y, 100 * k / n, height=h * 0.9, color=col, label=t[key] if ri == 0 else None)
            ax.errorbar(100 * k / n, y, xerr=[[max(0.0, 100 * (k / n - lo))], [max(0.0, 100 * (hi - k / n))]], fmt="none",
                        ecolor="black", lw=0.6, capsize=1.2)
        if row[4] is not None:
            k, n = rate(row[4])
            ax.plot([100 * k / n] * 2, [ri - 1.6 * h, ri + 1.6 * h], color=C["teacher"], lw=1.2,
                    label=t["teacher"] if ri == 0 else None)
        k, n = rate((UNTUNED[row[0]], "classical"))
        ax.plot(100 * k / n, ri - h, "x", ms=3.6, mec="black", mew=0.8, label=t["untuned"] if ri == 0 else None)
        if row[5] is not None:
            k, n = rate(row[5])
            ax.plot(100 * k / n, ri - h, "D", ms=3.2, mfc="white", mec="black", mew=0.7,
                    label=t["cond_tuned"] if ri == 2 else None)
    ax.set_yticks(range(len(ROWS)), [t["cond"][r[0]] for r in ROWS], fontsize=6.8)
    ax.invert_yaxis()
    ax.set_xlim(60, 101)
    ax.set_xlabel(t["x"])
    decimal_axis(ax, lang)
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.35, 1.0), ncol=2, fontsize=6.5)
    save(fig_, "generalizzazione", lang)


if __name__ == "__main__":
    for lang in ("it", "en"):
        fig(lang)
    for row in ROWS:
        print(row[0], [round(100 * rate(s)[0] / rate(s)[1], 1) if s else None for s in row[1:]])
