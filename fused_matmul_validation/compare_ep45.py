"""Compare the new cosine->0 epoch-45 sample against the 60-run's mid-decay epoch-40
and epoch-50, same prompts, to confirm the level."""
import os, glob
from PIL import Image, ImageDraw, ImageFont

OUT = r"C:/Concord/overfit/phase2_out"
COLS = [("epoch40", "ep40 (60-run)"), ("epoch45", "ep45 cosine->0"), ("epoch50", "ep50 (60-run)")]
PROMPTS = {0: "bureaucrat", 8: "kalbat", 10: "yaSattra", 11: "yaTsatsa"}
THUMB, PAD, LBL_H, LBL_W = 320, 8, 30, 96


def find(folder, idx):
    for pat in (f"{idx:02d}_*", f"p{idx:02d}.*", f"p{idx:02d}_*"):
        g = sorted(glob.glob(os.path.join(OUT, folder, pat)))
        if g:
            return g[0]
    return None


def font(sz):
    for p in (r"C:/Windows/Fonts/arial.ttf",):
        try:
            return ImageFont.truetype(p, sz)
        except Exception:
            pass
    return ImageFont.load_default()


rows = list(PROMPTS.items())
W = LBL_W + len(COLS) * (THUMB + PAD) + PAD
H = LBL_H + len(rows) * (THUMB + PAD) + PAD
canvas = Image.new("RGB", (W, H), (250, 250, 250))
draw = ImageDraw.Draw(canvas)
fh, fl = font(18), font(15)
for c, (_, label) in enumerate(COLS):
    draw.text((LBL_W + c * (THUMB + PAD) + PAD + 70, 7), label, fill=(0, 0, 0), font=fh)
for r, (idx, plabel) in enumerate(rows):
    y0 = LBL_H + r * (THUMB + PAD) + PAD
    draw.text((6, y0 + THUMB // 2 - 8), plabel, fill=(0, 0, 0), font=fl)
    for c, (folder, _) in enumerate(COLS):
        x0 = LBL_W + c * (THUMB + PAD) + PAD
        fp = find(folder, idx)
        if fp:
            im = Image.open(fp).convert("RGB")
            im.thumbnail((THUMB, THUMB))
            canvas.paste(im, (x0, y0 + (THUMB - im.size[1]) // 2))
        else:
            draw.rectangle([x0, y0, x0 + THUMB, y0 + THUMB], outline=(200, 0, 0), width=2)
canvas.save(os.path.join(OUT, "compare_ep45.jpg"), quality=88)
print("saved compare_ep45.jpg", canvas.size)
