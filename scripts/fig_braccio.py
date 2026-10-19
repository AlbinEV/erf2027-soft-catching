"""Fig. 2: geometry of one arm and the closing sequence, Italian and English.

(a) anatomy of one arm drawn to scale from catch/config.py: rigid phalanges on the spine, closing tendon at 7 mm from
    the spine and anchored at the tip, bevels that encode the joint stops (20 deg at the root, 3 x 30 deg at the
    corner); the same arm straight (1) and with every joint at its stop (2);
(b) section of the hand during closing (every joint at the same angle until its stop; geometry only, no loads,
    no contact) around a 10 cm cube on the palm.
Usage: .venv/bin/python scripts/fig_braccio.py
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Polygon, Rectangle  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from catch.config import Params  # noqa: E402

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7, "pdf.fonttype": 42})
INK, PHAL, SPINE, TENDON, ACC, CUBE = "#243333", "#E6EBE9", "#C8762E", "#367266", "#367266", "#F2E3CF"
DORSAL = "#B9B3D6"
D_MM = 5.0          # depth of the dorsal stop blocks drawn under the spine (schematic)
T = {"it": dict(a="(a) anatomia di un braccio", b="(b) chiusura attorno a un cubo di 10 cm", spine="spina",
                tendon="tendine di chiusura", phal="falange", stops="battute", tip="ancoraggio",
                dorsal="battute dorsali (−30°)",
                s1="diritto", s2="in battuta", open="aperto", half="a metà", root="radice in battuta",
                closed="tutte le battute", palm="palmo", x="x (mm)", z="z (mm)"),
     "en": dict(a="(a) anatomy of one arm", b="(b) closing around a 10 cm cube", spine="spine",
                tendon="closing tendon", phal="phalanx", stops="stops", tip="anchor",
                dorsal="dorsal stops (−30°)",
                s1="straight", s2="at the stops", open="open", half="halfway", root="root at stop",
                closed="all stops", palm="palm", x="x (mm)", z="z (mm)")}


def draw_arm(ax, p, origin, angles):
    """Phalanges as trapezoids whose bevels encode the stops, spine on one face, tendon at guide_m on the other;
    under the spine, dorsal blocks whose bevels encode the outward (negative) stop of every joint, outward_limit_deg
    in the model. angles: joint angles (rad), positive towards the tendon side. Returns the tendon anchor."""
    h, g = p.phalanx_h_m * 1000, p.guide_m * 1000
    pitch, stops = np.asarray(p.pitch_m) * 1000, np.radians(p.stop_deg)
    out = np.radians(p.outward_limit_deg)
    o, th, gap = np.array(origin, float), 0.0, 0.4
    spine, tendon = [o.copy()], []
    for i, L in enumerate(pitch):
        th += angles[i]
        R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
        loc = lambda pts, o=o, R=R: [tuple(o + R @ np.array(q)) for q in pts]  # noqa: E731
        c0 = h * np.tan(stops[i] / 2)
        c1 = h * np.tan(stops[i + 1] / 2) if i + 1 < len(pitch) else 0.0
        ax.add_patch(Polygon(loc([(gap, 0), (L - gap, 0), (L - gap - c1, h), (gap + c0, h)]), closed=True,
                             facecolor=PHAL, edgecolor=INK, lw=0.45, zorder=2))
        d0 = D_MM * np.tan(out / 2)                       # every joint, root included, stops at -outward_limit
        d1 = D_MM * np.tan(out / 2) if i + 1 < len(pitch) else 0.0
        ax.add_patch(Polygon(loc([(gap, -0.9), (L - gap, -0.9), (L - gap - d1, -0.9 - D_MM), (gap + d0, -0.9 - D_MM)]),
                             closed=True, facecolor=DORSAL, edgecolor=INK, lw=0.35, zorder=2))
        tendon += loc([(0, g), (L, g)])
        o = o + R @ np.array([L, 0.0])
        spine.append(o.copy())
    spine, tendon = np.array(spine), np.array(tendon)
    ax.plot(spine[:, 0], spine[:, 1], color=SPINE, lw=1.4, solid_capstyle="butt", zorder=3)
    ax.plot(tendon[:, 0], tendon[:, 1], color=TENDON, lw=0.9, ls=(0, (3, 1.5)), zorder=4)
    ax.plot(*tendon[-1], "o", ms=2.3, color=TENDON, zorder=4)
    return tendon[-1]


def section(p, theta):
    """Spine polyline of one arm in the hand section (x outward, z along the hand axis), every joint at
    min(theta, stop); the open arm leans outwards by mount_deg."""
    phi, x, z = np.radians(p.mount_deg), p.root_radius_m * 1000, 0.0
    pts = [(x, z)]
    for L, stop in zip(np.asarray(p.pitch_m) * 1000, p.stop_deg):
        phi -= np.radians(min(theta, stop))
        x, z = x + L * np.sin(phi), z + L * np.cos(phi)
        pts.append((x, z))
    return np.array(pts)


def main():
    p = Params()
    h = p.phalanx_h_m * 1000
    stops = np.radians(p.stop_deg)
    for lang, t in T.items():
        fig, (ax, bx) = plt.subplots(1, 2, figsize=(12 / 2.54, 4.6 / 2.54), gridspec_kw=dict(width_ratios=[1.55, 1]))
        # (a) anatomy: straight arm (1) and the same arm at its stops (2)
        draw_arm(ax, p, (0, 0), np.zeros(len(stops)))
        xc = 205.0
        draw_arm(ax, p, (xc, 0), stops)
        pitch = np.asarray(p.pitch_m) * 1000
        x_root, x_corner = pitch[0], pitch[:5].sum()
        ax.text(1.0, h + 3, "20°", ha="left", va="bottom", fontsize=6, color=ACC)
        ax.text(x_corner + 1.5 * pitch[5], h + 3, f"3 × 30° ({t['stops']})", ha="center", va="bottom", fontsize=6,
                color=ACC)
        ax.annotate(t["tendon"], (45, p.guide_m * 1000), (45, h + 18), ha="center", va="bottom", fontsize=6,
                    color=TENDON, arrowprops=dict(arrowstyle="-", lw=0.5, color=TENDON))
        ax.annotate(t["phal"], (pitch[0] + 20, h * 0.5), (14, -14), ha="center", va="top", fontsize=6,
                    arrowprops=dict(arrowstyle="-", lw=0.5, color=INK))
        ax.annotate(t["spine"], (62, 0), (62, -14), ha="center", va="top", fontsize=6, color=SPINE,
                    arrowprops=dict(arrowstyle="-", lw=0.5, color=SPINE))
        xg = pitch[:6].sum()
        ax.annotate(t["dorsal"], (xg, -0.9 - D_MM), (xg + 5, -24), ha="center", va="top", fontsize=6,
                    color="#5B5291", arrowprops=dict(arrowstyle="-", lw=0.5, color="#5B5291"))
        ax.text(pitch.sum() + 3, -4, t["tip"], ha="left", va="top", fontsize=5.5, color=TENDON)
        for x0, n, lab in ((0, "1", t["s1"]), (xc, "2", t["s2"])):
            ax.text(x0, -42, f"({n}) {lab}", ha="left", va="center", fontsize=6.5, color=INK)
        ax.set_xlim(-5, xc + 125)
        ax.set_ylim(-48, 120)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(t["a"], loc="left", fontsize=7)
        # (b) closing sequence around a 10 cm cube on the palm
        bx.add_patch(Rectangle((-50, 0), 100, 100, facecolor=CUBE, edgecolor=INK, lw=0.6, zorder=1))
        bx.add_patch(Rectangle((-p.palm_half_m * 1000, -p.palm_thick_m * 1000), p.palm_half_m * 2000,
                               p.palm_thick_m * 1000, facecolor=TENDON, edgecolor="none", zorder=1))
        for th, col, lab in ((0, "#C9D1D6", t["open"]), (10, "#9AA6AF", t["half"]), (20, "#5E8F82", t["root"]),
                             (30, INK, t["closed"])):
            pts = section(p, th)
            bx.plot(pts[:, 0], pts[:, 1], color=col, lw=1.3, label=lab)
            bx.plot(-pts[:, 0], pts[:, 1], color=col, lw=1.3)
        bx.text(0, -16, t["palm"], ha="center", va="top", fontsize=6, color=TENDON)
        bx.set_aspect("equal")
        bx.set_xlim(-150, 150)
        bx.set_ylim(-28, 210)
        bx.axis("off")
        bx.legend(frameon=False, fontsize=5.5, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.02),
                  handlelength=1.2, columnspacing=0.8)
        bx.set_title(t["b"], loc="left", fontsize=7)
        fig.tight_layout(w_pad=0.5)
        for ext in ("pdf", "png"):
            fig.savefig(ROOT / "figures" / f"fig_braccio_{lang}.{ext}", dpi=300, bbox_inches="tight", pad_inches=0.01)
        plt.close(fig)
    print("ok")


if __name__ == "__main__":
    main()
