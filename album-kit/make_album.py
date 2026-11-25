"""Diamond SC photo-album builder.

Grades a folder of match/training photos to the club's look, crops them to 4:5 (1080x1350)
and builds an editorial cover: a huge condensed word in Diamond yellow-gold placed BEHIND the
player (subject cut out with an AI mask), cream secondary text bottom-left, crest bottom-right.

Usage:
  python make_album.py SRC_DIR OUT_DIR --cover IMG_1234.jpg --word "FIRST HALF" \
      --sub "DIAMOND SC" --tag "MATCH DAY" [--y 150] [--cover-top 0.0] [--bright]

  python make_album.py SRC_DIR OUT_DIR --contact      # numbered contact sheet to pick photos
  python make_album.py SRC_DIR OUT_DIR --cover X.jpg --only-cover --y 180   # iterate on the cover

Requires: pip install pillow numpy onnxruntime
The background-removal model (RMBG-1.4, ~176 MB) is downloaded once into ./models/.
"""
import argparse, glob, os, sys, urllib.request
import numpy as np
from PIL import Image, ImageOps, ImageDraw, ImageFont, ImageFilter, ImageEnhance, ImageCms

KIT = os.path.dirname(os.path.abspath(__file__))
W, H = 1080, 1350
YEL = (247, 196, 20)          # Diamond yellow-gold
CREAM = (250, 236, 222)
BIG = os.path.join(KIT, "fonts", "C_Anton.ttf")
SMALL = os.path.join(KIT, "fonts", "C_LeagueGothic.ttf")
CREST = os.path.join(KIT, "assets", "logo_crest.png")
MODEL = os.path.join(KIT, "models", "rmbg14.onnx")
MODEL_URL = "https://huggingface.co/briaai/RMBG-1.4/resolve/main/onnx/model.onnx"

# LAB mean / std of the photographer-edited training set: the target look.
REF_MEAN = np.array([123.42, -0.25, 22.57], np.float32)
REF_STD = np.array([58.02, 20.45, 18.58], np.float32)

_srgb, _lab = ImageCms.createProfile("sRGB"), ImageCms.createProfile("LAB")
_to_lab = ImageCms.buildTransformFromOpenProfiles(_srgb, _lab, "RGB", "LAB")
_to_rgb = ImageCms.buildTransformFromOpenProfiles(_lab, _srgb, "LAB", "RGB")


def load(path):
    return ImageOps.exif_transpose(Image.open(path)).convert("RGB")


def grade(im, bright=True, k=0.45):
    """Match the training-photo look. Gentle: a/b variance is clamped so skin never goes red."""
    raw = np.asarray(ImageCms.applyTransform(im, _to_lab))
    a = raw.astype(np.float32)
    a[..., 1:] = raw[..., 1:].view(np.int8).astype(np.float32)
    flat = a.reshape(-1, 3)
    m, s = flat.mean(0), flat.std(0)
    sc = REF_STD / (s + 1e-6); sc[1:] = np.clip(sc[1:], 0.9, 1.1)
    target_mean = REF_MEAN.copy()
    if bright:  # never pull exposure down: keep the brighter of photo / reference
        target_mean[0] = max(m[0], REF_MEAN[0]) + 4
    t = (a - m) * sc + target_mean
    out = a * (1 - k) + t * k
    o8 = np.empty(out.shape, np.uint8)
    o8[..., 0] = np.clip(out[..., 0], 0, 255).astype(np.uint8)
    o8[..., 1:] = np.clip(np.round(out[..., 1:]), -127, 127).astype(np.int8).view(np.uint8)
    rgb = ImageCms.applyTransform(Image.fromarray(o8, "LAB"), _to_rgb)
    rgb = ImageEnhance.Color(rgb).enhance(1.06)
    rgb = ImageEnhance.Contrast(rgb).enhance(1.04)
    return rgb.filter(ImageFilter.UnsharpMask(radius=2, percent=60, threshold=3))


def crop45(im, top=None):
    """4:5 crop. top = 0..1 vertical position of the crop window (None = centre).
    Landscape shots are centre-cropped horizontally."""
    w, h = im.size
    if h / w >= 5 / 4:
        ch = round(w * 5 / 4); y = round((h - ch) * (0.5 if top is None else top))
        im = im.crop((0, y, w, y + ch))
    else:
        cw = round(h * 4 / 5); x = (w - cw) // 2
        im = im.crop((x, 0, x + cw, h))
    return im.resize((W, H), Image.LANCZOS)


_sess = None
def subject_mask(im):
    global _sess
    if _sess is None:
        import onnxruntime as ort
        if not os.path.exists(MODEL):
            os.makedirs(os.path.dirname(MODEL), exist_ok=True)
            print("downloading background-removal model (~176 MB)...")
            urllib.request.urlretrieve(MODEL_URL, MODEL)
        _sess = ort.InferenceSession(MODEL, providers=["CPUExecutionProvider"])
    x = np.asarray(im.resize((1024, 1024), Image.BILINEAR), np.float32) / 255 - 0.5
    o = _sess.run(None, {_sess.get_inputs()[0].name: x.transpose(2, 0, 1)[None]})[0][0, 0]
    o = (o - o.min()) / (o.max() - o.min() + 1e-6)
    m = Image.fromarray((o * 255).astype(np.uint8)).resize(im.size, Image.BILINEAR)
    return np.clip((np.asarray(m, np.float32) / 255 - 0.3) / 0.4, 0, 1)


def big_word(word, width, color=YEL, stretch=1.35):
    f = ImageFont.truetype(BIG, 400)
    bb = f.getbbox(word)
    im = Image.new("RGBA", (bb[2] - bb[0] + 20, bb[3] - bb[1] + 20), (0, 0, 0, 0))
    ImageDraw.Draw(im).text((10 - bb[0], 10 - bb[1]), word, font=f, fill=color + (255,))
    im = im.crop(im.split()[3].getbbox())
    return im.resize((width, round(im.height * width / im.width * stretch)), Image.LANCZOS)


def small_text(lines, size, color=CREAM):
    f = ImageFont.truetype(SMALL, size); lh = int(size * 0.92)
    w = max(int(f.getlength(l)) for l in lines)
    im = Image.new("RGBA", (w + 10, lh * len(lines) + 20), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    for i, l in enumerate(lines): d.text((0, i * lh), l, font=f, fill=color + (255,))
    return im.crop(im.split()[3].getbbox())


def make_cover(base, word, y, sub, tag, word_w=930):
    m = subject_mask(base)
    lines = word.split("|")                       # "FIRST|HALF" -> two stacked lines
    L = Image.new("RGBA", (W, H), (0, 0, 0, 0)); yy = y
    for ln in lines:
        t = big_word(ln, word_w); L.alpha_composite(t, ((W - t.width) // 2, yy)); yy += t.height + 18
    a = np.asarray(L).copy(); a[..., 3] = (a[..., 3] * (1 - m)).astype(np.uint8)
    out = base.convert("RGBA"); out.alpha_composite(Image.fromarray(a, "RGBA"))
    # light bottom scrim only where the small text sits (keeps the photo bright)
    sc = np.zeros((H, W, 4), np.uint8)
    ramp = np.clip((np.arange(H) - H * 0.78) / (H * 0.22), 0, 1) ** 1.5
    sc[..., 3] = (ramp * 90)[:, None].astype(np.uint8)
    out.alpha_composite(Image.fromarray(sc, "RGBA"))
    st = small_text(sub.split("|"), 150); out.alpha_composite(st, (90, H - 110 - st.height))
    if tag:
        tg = small_text([tag], 44); out.alpha_composite(tg, (90 + st.width + 40, H - 110 - tg.height))
    crest = Image.open(CREST).convert("RGBA"); ch = 190
    crest = crest.resize((round(crest.width * ch / crest.height), ch), Image.LANCZOS)
    out.alpha_composite(crest, (W - 90 - crest.width, H - 110 - ch))
    return out.convert("RGB")


def contact_sheet(files, out):
    th = []
    for i, f in enumerate(files):
        im = load(f); im.thumbnail((300, 300))
        c = Image.new("RGB", (300, 300), (20, 20, 20)); c.paste(im, ((300 - im.width) // 2, (300 - im.height) // 2))
        ImageDraw.Draw(c).text((6, 4), f"{i} {os.path.basename(f)[:18]}", fill=(255, 255, 0)); th.append(c)
    for p in range(0, len(th), 30):
        sheet = Image.new("RGB", (1800, 1500))
        for k, c in enumerate(th[p:p + 30]): sheet.paste(c, ((k % 6) * 300, (k // 6) * 300))
        sheet.save(os.path.join(out, f"_contact_{p // 30}.jpg"), quality=85)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src"); ap.add_argument("out")
    ap.add_argument("--cover"); ap.add_argument("--word", default="FIRST HALF")
    ap.add_argument("--sub", default="DIAMOND|SC"); ap.add_argument("--tag", default="MATCH DAY")
    ap.add_argument("--y", type=int, default=120); ap.add_argument("--cover-top", type=float, default=0.0)
    ap.add_argument("--files", help="comma-separated subset/order of filenames for the album")
    ap.add_argument("--contact", action="store_true"); ap.add_argument("--only-cover", action="store_true")
    ap.add_argument("--no-bright", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    exts = ("*.jpg", "*.JPG", "*.jpeg", "*.JPEG", "*.png", "*.PNG")
    files = sorted({f for e in exts for f in glob.glob(os.path.join(a.src, e))})
    if a.files: files = [os.path.join(a.src, n.strip()) for n in a.files.split(",")]
    if a.contact: contact_sheet(files, a.out); print("contact sheets in", a.out); return
    bright = not a.no_bright
    if a.cover:
        base = crop45(grade(load(os.path.join(a.src, a.cover)), bright), a.cover_top)
        make_cover(base, a.word, a.y, a.sub, a.tag).save(os.path.join(a.out, "01-cover.jpg"), quality=95, subsampling=0)
        print("cover done")
        if a.only_cover: return
    k = 2
    for f in files:
        if a.cover and os.path.basename(f) == a.cover: continue
        crop45(grade(load(f), bright)).save(os.path.join(a.out, f"{k:02d}.jpg"), quality=95, subsampling=0)
        k += 1
    print("album images:", k - 1)


if __name__ == "__main__":
    main()
