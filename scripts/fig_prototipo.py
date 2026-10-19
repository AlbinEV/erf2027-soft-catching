"""Photos of the printed hand prototype for the end of the ERF2027 paper. A photo and a video frame of
12 Oct 2026 (source files not distributed; set PHOTO_DIR),
cropped without faces, lettering or the date stamp of the camera. A manifest records paths, hashes and crops.
Usage: .venv/bin/python scripts/fig_prototipo.py
"""
import hashlib
import os
import io
import json
import subprocess
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image, ImageFilter, ImageOps  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SRC = Path(os.environ.get("PHOTO_DIR", "photos"))
# (source file, crop box (left, top, right, bottom), regions to blur, video time in s or None for a photo)
PHOTOS = [
    ("braccio_blu_singolo_piegato_2026-10-12.jpeg", (180, 330, 950, 1560), [], None),
    ("braccio_singolo_chiusura_tendine_2026-10-12.mp4", (110, 30, 390, 470), [], 10.0),
]
# Of the first selection only the single straight arm was approved; the second panel is a frame
# of the video of the same arm closed by pulling the tendon (cropped to the arm: no face, no shirt lettering).
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7, "pdf.fonttype": 42})


def main():
    imgs, meta = [], []
    for name, box, blur, t_video in PHOTOS:
        f = SRC / name
        if t_video is None:
            im = ImageOps.exif_transpose(Image.open(f)).convert("RGB")
        else:
            png = subprocess.run(["ffmpeg", "-v", "error", "-ss", str(t_video), "-i", str(f), "-frames:v", "1", "-f",
                                  "image2pipe", "-vcodec", "png", "-"], capture_output=True, check=True).stdout
            im = Image.open(io.BytesIO(png)).convert("RGB")
        for r in blur:
            im.paste(im.crop(r).filter(ImageFilter.GaussianBlur(25)), r[:2])
        im = im.crop(box)
        imgs.append(im)
        meta.append(dict(path=str(f), sha256=hashlib.sha256(f.read_bytes()).hexdigest(), crop=list(box),
                         blurred=[list(r) for r in blur], video_time_s=t_video))
    out = ROOT / "figures"
    (out / "fig_prototipo_manifest.json").write_text(json.dumps(meta, indent=1))
    ratios = [im.size[0] / im.size[1] for im in imgs]              # equal heights
    fig, axs = plt.subplots(1, len(imgs), figsize=(12 / 2.54, 12 / 2.54 / sum(ratios)),
                            gridspec_kw=dict(wspace=0.03, width_ratios=ratios))
    for i, (ax, im) in enumerate(zip(axs, imgs)):
        ax.imshow(im)
        ax.set_axis_off()
        ax.text(0.02, 0.97, f"({'abc'[i]})", transform=ax.transAxes, va="top", fontsize=7,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.0))
    for lang in ("it", "en"):        # no text besides the panel letters: the same image for both languages
        for ext in ("pdf", "png"):
            fig.savefig(out / f"fig_prototipo_{lang}.{ext}", dpi=300, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
