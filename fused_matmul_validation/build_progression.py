import os, glob
from PIL import Image, ImageDraw, ImageFont

OUT = r"C:/Concord/overfit/phase2_out"
EPOCHS = [10, 20, 30, 40, 50, 60]
# prompt index -> label (matches overfit_samples.json order)
PROMPTS = {0: "bureaucrat", 8: "kalbat", 10: "yaSattra", 2: "yaDon", 11: "yaTsatsa"}
THUMB, PAD, LBL_H, LBL_W = 300, 8, 28, 96


def find(epoch, idx):
    d = os.path.join(OUT, f"epoch{epoch:02d}")
    for pat in (f"{idx:02d}_*", f"p{idx:02d}.*", f"p{idx:02d}_*"):
        g = sorted(glob.glob(os.path.join(d, pat)))
        if g:
            return g[0]
    return None


def font(sz):
    for p in (r"C:/Windows/Fonts/arial.ttf", r"C:/Windows/Fonts/segoeui.ttf"):
        try:
            return ImageFont.truetype(p, sz)
        except Exception:
            pass
    return ImageFont.load_default()


rows = list(PROMPTS.items())
ncol, nrow = len(EPOCHS), len(rows)
W = LBL_W + ncol * (THUMB + PAD) + PAD
H = LBL_H + nrow * (THUMB + PAD) + PAD
canvas = Image.new("RGB", (W, H), (250, 250, 250))
draw = ImageDraw.Draw(canvas)
f_hdr, f_lbl = font(18), font(15)

for c, e in enumerate(EPOCHS):
    x = LBL_W + c * (THUMB + PAD) + PAD + THUMB // 2 - 28
    draw.text((x, 6), f"epoch {e}", fill=(0, 0, 0), font=f_hdr)

missing = []
for r, (idx, label) in enumerate(rows):
    y0 = LBL_H + r * (THUMB + PAD) + PAD
    draw.text((6, y0 + THUMB // 2 - 8), label, fill=(0, 0, 0), font=f_lbl)
    for c, e in enumerate(EPOCHS):
        x0 = LBL_W + c * (THUMB + PAD) + PAD
        fp = find(e, idx)
        if fp:
            im = Image.open(fp).convert("RGB")
            im.thumbnail((THUMB, THUMB))
            canvas.paste(im, (x0, y0 + (THUMB - im.size[1]) // 2))
        else:
            draw.rectangle([x0, y0, x0 + THUMB, y0 + THUMB], outline=(200, 0, 0), width=2)
            missing.append((e, idx))

path = os.path.join(OUT, "progression.jpg")
canvas.save(path, quality=88)
print(f"saved {path} size={canvas.size} missing={missing}")
