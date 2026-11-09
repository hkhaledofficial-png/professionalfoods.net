#!/usr/bin/env python3
"""
Professional Ready Mix — promo video renderer.

Builds vertical 9:16 (1080x1920) brand promos from the site footage & photos,
in the same visual language as Professional Foods (deep dark base, warm gold
accents, cinematic grade, short tagline headings) but focused on ready-mix
concrete.

Usage:
    python3 render.py --media <dir with v/ and imgs/> --fonts <dir> --out <dir> [--only 1,2,3] [--preview]
"""
import argparse, io, math, os, subprocess, sys, wave
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageChops

W, H, FPS = 1080, 1920, 30
LAND = False  # True for the 16:9 (YouTube) cut

# ---- Brand palette ------------------------------------------------------------
# Same art direction as Professional Foods (dark premium base, one strong accent,
# spaced-caps kickers, cinematic grade) but the accent is the Ready Mix logo blue.
ACC = (48, 117, 253)         # logo blue  #3075FD
ACC_HI = (150, 192, 255)
ACC_LO = (28, 86, 214)
CREAM = (250, 251, 253)
SAND = (214, 224, 238)       # cool concrete-white for sub lines
CHARCOAL = (10, 13, 18)      # concrete-dark base
CHARCOAL_2 = (30, 33, 36)
LOGO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "logo-ready-mix.png")
LOGO_WHITE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "logo-ready-mix-white.png")

ARGS = None


def font(name, size):
    return ImageFont.truetype(os.path.join(ARGS.fonts, name), size)


F_AR_BLACK = "Cairo-Black.ttf"
F_AR_BOLD = "Cairo-Bold.ttf"
F_AR_REG = "Tajawal-Medium.ttf"
F_EN_XB = "Inter-ExtraBold.ttf"
F_EN_SB = "Inter-SemiBold.ttf"


# ---- easing -----------------------------------------------------------------
def clamp(x, a=0.0, b=1.0):
    return max(a, min(b, x))


def ease_out(x):
    x = clamp(x)
    return 1 - (1 - x) ** 3


def ease_in_out(x):
    x = clamp(x)
    return 3 * x * x - 2 * x * x * x


# ---- text layers ------------------------------------------------------------
def accent_fill(mask):
    """Vertical gold gradient through a mask -> RGBA."""
    w, h = mask.size
    y = np.linspace(0, 1, h)[:, None]
    top, mid, bot = np.array((185, 214, 255)), np.array((112, 164, 255)), np.array((60, 128, 255))
    g = np.where(y < 0.5, top + (mid - top) * (y / 0.5), mid + (bot - mid) * ((y - 0.5) / 0.5))
    g = np.repeat(g[:, None, :], w, axis=1).reshape(h, w, 3)
    rgba = np.dstack([g, np.array(mask)]).astype(np.uint8)
    return Image.fromarray(rgba, "RGBA")


def solid_fill(mask, color):
    img = Image.new("RGBA", mask.size, color + (0,))
    img.putalpha(mask)
    return img


def text_mask(text, fnt, rtl=False, tracking=0):
    """Render text to an L mask. Tracking (px) applied for LTR only."""
    if rtl:
        kw = dict(direction="rtl", language="ar", features=["-liga"] if False else None)
        kw = {k: v for k, v in kw.items() if v is not None}
        bbox = fnt.getbbox(text, **kw)
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        pad = int(fnt.size * 0.35)
        m = Image.new("L", (w + pad * 2, h + pad * 2), 0)
        ImageDraw.Draw(m).text((pad - bbox[0], pad - bbox[1]), text, font=fnt, fill=255, **kw)
        return m
    if tracking == 0:
        bbox = fnt.getbbox(text)
        pad = int(fnt.size * 0.3)
        m = Image.new("L", (bbox[2] - bbox[0] + pad * 2, bbox[3] - bbox[1] + pad * 2), 0)
        ImageDraw.Draw(m).text((pad - bbox[0], pad - bbox[1]), text, font=fnt, fill=255)
        return m
    widths = [fnt.getlength(c) for c in text]
    total = sum(widths) + tracking * (len(text) - 1)
    asc, desc = fnt.getmetrics()
    pad = int(fnt.size * 0.3)
    m = Image.new("L", (int(total) + pad * 2, asc + desc + pad * 2), 0)
    d = ImageDraw.Draw(m)
    x = pad
    for c, cw in zip(text, widths):
        d.text((x, pad), c, font=fnt, fill=255)
        x += cw + tracking
    return m.crop(m.getbbox()) if m.getbbox() else m


def with_shadow(layer, radius=18, opacity=0.65):
    a = layer.getchannel("A")
    pad = radius * 2
    big = Image.new("RGBA", (layer.width + pad * 2, layer.height + pad * 2), (0, 0, 0, 0))
    sh = Image.new("L", big.size, 0)
    sh.paste(a, (pad, pad + 4))
    sh = sh.filter(ImageFilter.GaussianBlur(radius)).point(lambda v: int(v * opacity))
    shadow = Image.new("RGBA", big.size, (0, 0, 0, 0))
    shadow.putalpha(sh)
    big.alpha_composite(shadow)
    big.alpha_composite(layer, (pad, pad))
    return big


class Text:
    """A pre-rendered text element with an in/out animation.

    kind: 'head' (Arabic headline, cream), 'sub' (Arabic sub, sand),
          'kicker' (English spaced caps, gold), 'gold' (Arabic headline, gold)
    """

    def __init__(self, text, kind, y, t_in, t_out, size=None, anim="rise"):
        self.y, self.t_in, self.t_out, self.anim = y, t_in, t_out, anim
        if kind == "head":
            m = text_mask(text, font(F_AR_BLACK, size or 104), rtl=True)
            layer = solid_fill(m, CREAM)
        elif kind == "accent":
            m = text_mask(text, font(F_AR_BLACK, size or 104), rtl=True)
            layer = accent_fill(m)
        elif kind == "sub":
            m = text_mask(text, font(F_AR_BOLD, size or 54), rtl=True)
            layer = solid_fill(m, SAND)
        elif kind == "kicker":
            m = text_mask(text, font(F_EN_XB, size or 34), tracking=int((size or 34) * 0.42))
            layer = accent_fill(m)
        else:
            raise ValueError(kind)
        self.layer = with_shadow(layer, radius=16 if kind != "kicker" else 8,
                                 opacity=0.9 if kind in ("accent", "kicker") else 0.7)
        self.kind = kind

    def draw(self, canvas, t):
        if t < self.t_in or t > self.t_out:
            return
        p_in = ease_out((t - self.t_in) / 0.55)
        p_out = 1 - ease_in_out((t - (self.t_out - 0.35)) / 0.35)
        a = clamp(min(p_in, p_out))
        if a <= 0:
            return
        dy = int((1 - p_in) * 46) if self.anim == "rise" else 0
        lay = self.layer
        if self.anim == "scale":
            s = 0.92 + 0.08 * p_in
            lay = lay.resize((int(lay.width * s), int(lay.height * s)), Image.LANCZOS)
        if a < 1:
            lay = lay.copy()
            lay.putalpha(lay.getchannel("A").point(lambda v: int(v * a)))
        x = (W - lay.width) // 2
        canvas.alpha_composite(lay, (x, self.y - lay.height // 2 + dy))


class Rule:
    """Thin gold divider that grows from the centre."""

    def __init__(self, y, width, t_in, t_out, thick=3):
        self.y, self.width, self.t_in, self.t_out, self.thick = y, width, t_in, t_out, thick

    def draw(self, canvas, t):
        if t < self.t_in or t > self.t_out:
            return
        p = ease_out((t - self.t_in) / 0.7)
        a = clamp(1 - ease_in_out((t - (self.t_out - 0.35)) / 0.35))
        w = int(self.width * p)
        if w < 2:
            return
        bar = Image.new("RGBA", (w, self.thick), ACC + (int(255 * a),))
        # soft fade at the ends
        al = np.array(bar.getchannel("A"), dtype=np.float32)
        ramp = np.minimum(1, np.minimum(np.arange(w), np.arange(w)[::-1]) / max(1, w * 0.18))
        al = al * ramp[None, :]
        bar.putalpha(Image.fromarray(al.astype(np.uint8)))
        canvas.alpha_composite(bar, ((W - w) // 2, self.y))


class Chips:
    """Location chips that pop in one after another (RTL order)."""

    def __init__(self, labels, y, t_in, t_out, gap=0.28):
        self.items = []
        f = font(F_AR_BOLD, 46)
        for lab in labels:
            m = text_mask(lab, f, rtl=True)
            tw, th = m.size
            pw, ph = tw + 30, th + 2
            chip = Image.new("RGBA", (pw, ph), (0, 0, 0, 0))
            d = ImageDraw.Draw(chip)
            d.rounded_rectangle((1, 1, pw - 2, ph - 2), radius=ph // 2, fill=(14, 16, 18, 170), outline=ACC + (255,), width=3)
            chip.alpha_composite(solid_fill(m, CREAM), (15, 1))
            self.items.append(chip)
        self.y, self.t_in, self.t_out, self.gap = y, t_in, t_out, gap
        spacing = 22
        total = sum(c.width for c in self.items) + spacing * (len(self.items) - 1)
        self.xs = []
        x = (W + total) // 2  # RTL: first item on the right
        for c in self.items:
            x -= c.width
            self.xs.append(x)
            x -= spacing

    def draw(self, canvas, t):
        if t < self.t_in or t > self.t_out:
            return
        p_out = 1 - ease_in_out((t - (self.t_out - 0.35)) / 0.35)
        for i, (c, x) in enumerate(zip(self.items, self.xs)):
            p = ease_out((t - self.t_in - i * self.gap) / 0.45)
            a = clamp(min(p, p_out))
            if a <= 0:
                continue
            lay = c.copy()
            lay.putalpha(lay.getchannel("A").point(lambda v: int(v * a)))
            canvas.alpha_composite(lay, (x, self.y - c.height // 2 + int((1 - p) * 30)))


# ---- media sources ------------------------------------------------------------
def decode(path, start, dur, vf=""):
    """Decode a video segment to a list of PIL RGB frames at FPS."""
    probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                            "stream=width,height:stream_side_data=rotation", "-of", "default=nw=1", path],
                           capture_output=True, text=True).stdout
    vals = dict(l.split("=", 1) for l in probe.split() if "=" in l)
    w, h = int(vals["width"]), int(vals["height"])
    if vals.get("rotation") in ("-90", "90", "270", "-270"):
        w, h = h, w
    chain = [f"fps={FPS}"] + ([vf] if vf else [])
    if "transpose" in vf:
        w, h = h, w
    cmd = ["ffmpeg", "-v", "error", "-ss", str(start), "-i", path, "-t", str(dur + 0.2),
           "-vf", ",".join(chain), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    raw = subprocess.run(cmd, capture_output=True).stdout
    n = len(raw) // (w * h * 3)
    arr = np.frombuffer(raw[: n * w * h * 3], np.uint8).reshape(n, h, w, 3)
    return [Image.fromarray(a) for a in arr]


class Media:
    def __init__(self, kind, path, start=0.0, vf="", speed=1.0):
        self.kind, self.path, self.start, self.vf, self.speed = kind, path, start, vf, speed
        self.frames = None
        self.img = None

    def load(self, dur):
        if self.kind == "img":
            self.img = Image.open(self.path).convert("RGB")
        else:
            self.frames = decode(self.path, self.start, dur * self.speed + 0.3, self.vf)

    def unload(self):
        self.frames, self.img = None, None

    def get(self, t):
        if self.kind == "img":
            return self.img
        i = int(t * self.speed * FPS)
        return self.frames[min(i, len(self.frames) - 1)]

    def size(self):
        return self.img.size if self.kind == "img" else self.frames[0].size


def cover(img, cw, ch, zoom=1.0, fx=0.5, fy=0.5):
    """Crop img to cover cw x ch, with zoom and focus point (0..1)."""
    iw, ih = img.size
    s = max(cw / iw, ch / ih) * zoom
    vw, vh = cw / s, ch / s
    x0 = (iw - vw) * fx
    y0 = (ih - vh) * fy
    return img.resize((cw, ch), Image.BICUBIC, box=(x0, y0, x0 + vw, y0 + vh))


# ---- grade / finishing --------------------------------------------------------
def make_luts():
    x = np.arange(256) / 255.0
    # gentle S-curve + warm split: lifted warm shadows, golden highlights
    s = x + 0.10 * np.sin((x - 0.5) * 2 * math.pi) * -0.5 * 0 + 0.0
    s = 0.5 + (x - 0.5) * 1.10
    s = np.clip(s, 0, 1)
    r = np.clip(s * 0.99 + 0.004, 0, 1)
    g = np.clip(s * 1.00 + 0.006, 0, 1)
    b = np.clip(s * 1.02 + 0.016, 0, 1)
    return [(c * 255).astype(np.uint8).tolist() for c in (r, g, b)]


LUTS = None
VIGNETTE = None
TEXT_GRAD = None
GRAIN = []


def init_finish():
    global LUTS, VIGNETTE, TEXT_GRAD, GRAIN
    LUTS = sum(make_luts(), [])
    yy, xx = np.mgrid[0:H, 0:W]
    d = np.sqrt(((xx - W / 2) / (W * 0.72)) ** 2 + ((yy - H / 2) / (H * 0.62)) ** 2)
    v = np.clip(1 - 0.55 * np.clip(d - 0.35, 0, 1) ** 1.6, 0, 1)
    VIGNETTE = Image.fromarray((v * 255).astype(np.uint8))
    # dark gradient bottom & top for text legibility
    g = np.zeros(H)
    yb = np.linspace(0, 1, H)
    g += np.clip((yb - 0.52) / 0.48, 0, 1) ** 1.3 * 0.78
    g += np.clip((0.16 - yb) / 0.16, 0, 1) ** 1.5 * 0.45
    a = np.repeat((np.clip(g, 0, 1) * 255).astype(np.uint8)[:, None], W, axis=1)
    TEXT_GRAD = Image.new("RGBA", (W, H), CHARCOAL + (0,))
    TEXT_GRAD.putalpha(Image.fromarray(a))
    rng = np.random.default_rng(7)
    for _ in range(6):
        n = rng.normal(0, 1, (H // 2, W // 2))
        n = np.clip(128 + n * 6, 0, 255).astype(np.uint8)
        GRAIN.append(Image.fromarray(n).resize((W, H), Image.BILINEAR))


def finish(frame, fi, grade=True, textgrad=True):
    img = frame.convert("RGB")
    if grade:
        img = img.point(LUTS)
    img = ImageChops.multiply(img, Image.merge("RGB", [VIGNETTE] * 3))
    if textgrad:
        base = img.convert("RGBA")
        base.alpha_composite(TEXT_GRAD)
        img = base.convert("RGB")
    # grain (overlay-ish via add/sub around mid-grey)
    gr = GRAIN[fi % len(GRAIN)]
    img = ImageChops.add(img, Image.merge("RGB", [gr] * 3), scale=1.0, offset=-128)
    return img


# ---- scenes ------------------------------------------------------------------
class Scene:
    def __init__(self, dur, media=None, mode="full", kb=(1.0, 1.12), focus=((0.5, 0.5), (0.5, 0.5)),
                 elements=(), frame_y=None, motion=None, grade=True, vo=None, watermark=True):
        self.media, self.mode, self.kb, self.focus = media, mode, kb, focus
        self.elements, self.frame_y, self.motion, self.grade = list(elements), frame_y, motion, grade
        self.vo, self.vo_wav, self.watermark = vo, None, watermark and motion is None
        self.set_dur(dur)

    def set_dur(self, d):
        self.dur = d
        for e in self.elements:
            if e.t_out < 50:
                e.t_out = d

    def render_bg(self, t):
        p = t / self.dur
        if self.motion:
            return self.motion(t, self.dur)
        src = self.media.get(t)
        z = self.kb[0] + (self.kb[1] - self.kb[0]) * ease_in_out(p) if self.mode == "full" else 1.0
        (fx0, fy0), (fx1, fy1) = self.focus
        fx, fy = fx0 + (fx1 - fx0) * ease_in_out(p), fy0 + (fy1 - fy0) * ease_in_out(p)
        portrait_in_land = LAND and src.size[0] < src.size[1]
        if self.mode == "full" and not portrait_in_land:
            return cover(src, W, H, z, fx, fy).convert("RGBA")
        # "frame": landscape footage shown wide, over a blurred, darkened copy
        small = cover(src, W // 8, H // 8, 1.0, 0.5, 0.5).filter(ImageFilter.GaussianBlur(5))
        bg = small.resize((W, H), Image.BILINEAR)
        bg = Image.blend(bg, Image.new("RGB", (W, H), CHARCOAL), 0.55).convert("RGBA")
        iw, ih = src.size
        z2 = self.kb[0] + (self.kb[1] - self.kb[0]) * ease_in_out(p)
        if LAND:
            if iw / ih > 1.2:  # landscape footage fills a 16:9 frame natively
                return cover(src, W, H, z2, fx, fy).convert("RGBA")
            fh = H  # portrait source: pillarbox over the blurred copy
            fw = int(H * iw / ih)
            inner = cover(src, fw, fh, z2, fx, fy).convert("RGBA")
            x = (W - fw) // 2
            bg.alpha_composite(inner, (x, 0))
            d = ImageDraw.Draw(bg)
            d.line([(x - 2, 0), (x - 2, H)], fill=ACC + (200,), width=3)
            d.line([(x + fw, 0), (x + fw, H)], fill=ACC + (200,), width=3)
            return bg
        fw = W
        fh = int(fw * 9 / 16) if iw / ih > 1.2 else int(fw * ih / iw)
        inner = cover(src, fw, fh, z2, fx, fy).convert("RGBA")
        y = self.frame_y if self.frame_y is not None else (H - fh) // 2
        bg.alpha_composite(inner, ((W - fw) // 2, y))
        d = ImageDraw.Draw(bg)
        d.line([(0, y - 1), (W, y - 1)], fill=ACC + (200,), width=3)
        d.line([(0, y + fh), (W, y + fh)], fill=ACC + (200,), width=3)
        return bg


def flash_alpha(t_into_scene):
    """Short warm light-leak at the start of a scene (0..1)."""
    x = t_into_scene / 0.35
    return clamp(1 - x) ** 2 * 0.35


LEAK = None


def leak_layer():
    global LEAK
    if LEAK is None:
        yy, xx = np.mgrid[0:H // 4, 0:W // 4]
        d = np.sqrt(((xx - W / 4 * 0.85) / (W / 4 * 0.9)) ** 2 + ((yy - H / 4 * 0.15) / (H / 4 * 0.7)) ** 2)
        a = np.clip(1 - d, 0, 1) ** 1.5
        rgb = np.dstack([np.full_like(a, 170), np.full_like(a, 205), np.full_like(a, 255)])
        img = Image.fromarray(np.dstack([rgb, a * 255]).astype(np.uint8), "RGBA").resize((W, H), Image.BILINEAR)
        LEAK = img
    return LEAK


def render_video(name, scenes, outdir, music, xfade=0.4, preview=False):
    starts, t = [], 0.0
    for i, s in enumerate(scenes):
        starts.append(t)
        t += s.dur - (xfade if i < len(scenes) - 1 else 0)
    total = t
    nframes = int(round(total * FPS))
    silent = os.path.join(outdir, f".{name}_silent.mp4")
    enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                            "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "21", "-maxrate", "9M", "-bufsize", "18M",
                            "-pix_fmt", "yuv420p", "-movflags", "+faststart", silent], stdin=subprocess.PIPE)
    loaded = set()
    stills = []
    for fi in range(nframes):
        tg = fi / FPS
        active = [i for i, s in enumerate(scenes) if starts[i] <= tg < starts[i] + s.dur]
        for i in active:
            if i not in loaded and scenes[i].media:
                scenes[i].media.load(scenes[i].dur)
                loaded.add(i)
        for i in list(loaded):
            if tg >= starts[i] + scenes[i].dur:
                scenes[i].media.unload()
                loaded.discard(i)
        layers = []
        for i in active:
            s = scenes[i]
            lt = tg - starts[i]
            fr = s.render_bg(lt)
            fr = finish(fr, fi, grade=s.grade).convert("RGBA")
            if i > 0:
                la = flash_alpha(lt)
                if la > 0.01:
                    lk = leak_layer().copy()
                    lk.putalpha(lk.getchannel("A").point(lambda v: int(v * la)))
                    fr.alpha_composite(lk)
            if s.watermark:
                fr.alpha_composite(watermark(), ((W - watermark().width) // 2, 36 if LAND else 70))
            for e in s.elements:
                e.draw(fr, lt)
            alpha = 1.0
            if i > 0 and lt < xfade:
                alpha = ease_in_out(lt / xfade)
            layers.append((fr, alpha))
        out = layers[0][0]
        for fr, a in layers[1:]:
            out = Image.blend(out, fr, a)
        # global fade in / out
        g = min(1.0, tg / 0.4, (total - tg) / 0.5)
        if g < 1:
            out = Image.blend(Image.new("RGBA", (W, H), (0, 0, 0, 255)), out, clamp(g))
        rgb = out.convert("RGB")
        enc.stdin.write(rgb.tobytes())
        if preview and fi % FPS == 15:
            stills.append(rgb.resize((W // 4, H // 4)))
        if fi % 90 == 0:
            print(f"  {name}: {fi}/{nframes}", flush=True)
    enc.stdin.close()
    enc.wait()
    final = os.path.join(outdir, f"{name}.mp4")
    wav = os.path.join(outdir, f".{name}.wav")
    music(total, wav, starts)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", silent, "-i", wav, "-c:v", "copy", "-c:a", "aac",
                    "-b:a", "192k", "-shortest", "-movflags", "+faststart", final], check=True)
    os.remove(silent)
    os.remove(wav)
    if stills:
        cols = 8
        rows = math.ceil(len(stills) / cols)
        sheet = Image.new("RGB", (cols * (W // 4), rows * (H // 4)), "black")
        for k, st in enumerate(stills):
            sheet.paste(st, ((k % cols) * (W // 4), (k // cols) * (H // 4)))
        sheet.save(os.path.join(outdir, f".{name}_sheet.jpg"), quality=82)
    print(f"done {final} ({total:.1f}s)")
    return final


# ---- music (original, synthesized) -------------------------------------------
SR = 44100


def synth_music(total, bpm=96, root=55.0, hits=()):
    n = int(total * SR) + SR
    t = np.arange(n) / SR
    out = np.zeros((n, 2))
    # warm pad: root, fifth, octave, minor third — slowly swelling, slight detune for width
    env = np.clip(t / 2.5, 0, 1) * np.clip((total + 0.6 - t) / 1.6, 0, 1)
    for mult, amp in ((1, 0.20), (1.5, 0.10), (2, 0.08), (2.378, 0.05), (3, 0.03)):
        f = root * mult
        l = np.sin(2 * np.pi * f * t + 0.4 * np.sin(2 * np.pi * 0.11 * t))
        r = np.sin(2 * np.pi * f * 1.003 * t + 0.4 * np.sin(2 * np.pi * 0.13 * t))
        lfo = 0.75 + 0.25 * np.sin(2 * np.pi * 0.2 * t + mult)
        out[:, 0] += l * amp * lfo
        out[:, 1] += r * amp * lfo
    out *= env[:, None]
    beat = 60 / bpm
    rng = np.random.default_rng(3)

    def add(sig, at, gain=1.0):
        i = int(at * SR)
        j = min(n, i + len(sig))
        if i < n:
            out[i:j] += sig[: j - i, None] * gain if sig.ndim == 1 else sig[: j - i] * gain

    # kick (pitch-swept sine) — starts after the intro, four-on-the-floor
    kl = int(0.45 * SR)
    kt = np.arange(kl) / SR
    kick = np.sin(2 * np.pi * (45 * kt + (110 / 18) * (1 - np.exp(-18 * kt)))) * np.exp(-7 * kt)
    hat_l = int(0.05 * SR)
    hat = rng.normal(0, 1, hat_l) * np.exp(-np.arange(hat_l) / SR * 90)
    hat = np.diff(np.concatenate([[0], hat]))  # crude high-pass
    k = 0
    tb = 2.0
    while tb < total - 2.2:
        add(kick, tb, 0.55)
        tb += beat
        k += 1
    # cinematic impacts at scene changes / end card
    il = int(2.5 * SR)
    it = np.arange(il) / SR
    boom = np.sin(2 * np.pi * (38 * it + 60 / 6 * (1 - np.exp(-6 * it)))) * np.exp(-2.2 * it)
    noise = rng.normal(0, 1, il)
    # smoothed noise for the 'thud' body
    noise = np.convolve(noise, np.ones(60) / 60, mode="same") * np.exp(-5 * it) * 2.5
    impact = boom * 0.9 + noise * 0.15
    for h_at, g in hits:
        add(impact, h_at, g)
    # riser into end card
    if hits:
        last = hits[-1][0]
        rl = int(1.6 * SR)
        rt = np.arange(rl) / SR
        rise = rng.normal(0, 1, rl)
        rise = np.convolve(rise, np.ones(8) / 8, mode="same") * (rt / 1.6) ** 2.5 * 0.06
        add(rise, max(0, last - 1.6), 1.0)
    # simple stereo 'reverb': feedback delays
    for dly, g in ((0.083, 0.25), (0.137, 0.18), (0.211, 0.12)):
        d = int(dly * SR)
        out[d:, 0] += out[:-d, 1] * g
        out[d:, 1] += out[:-d, 0] * g
    out = out[: int(total * SR)]
    fade = np.clip((total - t[: len(out)]) / 1.2, 0, 1)
    out *= fade[:, None]
    out /= max(1e-6, np.abs(out).max())
    return out


def read_wav(path):
    """Load a WAV as float mono at SR (resampled with ffmpeg)."""
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-ac", "1", "-ar", str(SR), "-f", "s16le", "-"],
                         capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.int16).astype(np.float32) / 32768


def mix_audio(total, music, vo_events, path):
    """Music bed ducked under the voice-over; VO slightly warmed & centred."""
    n = int(total * SR)
    music = music[:n]
    vo = np.zeros(n, np.float32)
    for at, clip in vo_events:
        i = int(at * SR)
        j = min(n, i + len(clip))
        if i < n:
            vo[i:j] += clip[: j - i]
    # ducking envelope from VO energy (attack 60ms / release 400ms)
    env = np.abs(vo)
    win = int(0.03 * SR)
    env = np.convolve(env, np.ones(win) / win, mode="same")
    active = (env > 0.01).astype(np.float32)
    duck = np.empty(n, np.float32)
    a_att, a_rel = 1 / (0.06 * SR), 1 / (0.4 * SR)
    v = 0.0
    for k in range(0, n, 64):  # block-wise smoothing is plenty here
        target = active[k]
        v += (target - v) * min(1.0, (a_att if target > v else a_rel) * 64)
        duck[k:k + 64] = v
    music_gain = 0.55 * (1 - 0.6 * duck)  # ~-8 dB under the voice
    out = music * music_gain[:, None] + vo[:, None] * 0.95
    out /= max(1e-6, np.abs(out).max()) / 0.89
    pcm = (out * 32767).astype(np.int16)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(SR)
        wf.writeframes(pcm.tobytes())


# ---- end card & motion graphics ---------------------------------------------
_ENDBG = {}


def endcard_bg(t, dur):
    key = "bg"
    if key not in _ENDBG:
        yy, xx = np.mgrid[0:H, 0:W]
        d = np.sqrt(((xx - W / 2) / W) ** 2 + ((yy - H * 0.46) / (H * 0.55)) ** 2)
        glow = np.clip(1 - d * 1.5, 0, 1) ** 2
        base = np.array(CHARCOAL, dtype=np.float32)
        col = base[None, None, :] + glow[..., None] * (np.array(ACC, dtype=np.float32) - base)[None, None, :] * 0.22
        # faint concrete texture
        rng = np.random.default_rng(11)
        tex = rng.normal(0, 1, (H // 6, W // 6))
        tex = np.array(Image.fromarray(((tex * 20) + 128).clip(0, 255).astype(np.uint8)).resize((W, H), Image.BICUBIC), dtype=np.float32) - 128
        col += tex[..., None] * 0.12
        _ENDBG[key] = Image.fromarray(col.clip(0, 255).astype(np.uint8)).convert("RGBA")
    return _ENDBG[key].copy()


def reach_bg(t, dur):
    """Motion graphic: gold rings radiating from Cairo, dots lighting up across the country."""
    img = endcard_bg(0, 1)
    d = ImageDraw.Draw(img, "RGBA")
    cx, cy = W // 2, int(H * (0.6 if LAND else 0.47))
    sx, sy = (1.9, 0.42) if LAND else (1.0, 1.0)
    rmax = 420 if LAND else 520
    # grid lines (engineering feel)
    for gx in range(0, W, 90):
        d.line([(gx, 0), (gx, H)], fill=(255, 248, 235, 10), width=1)
    for gy in range(0, H, 90):
        d.line([(0, gy), (W, gy)], fill=(255, 248, 235, 10), width=1)
    # pulsing rings
    for k in range(4):
        ph = (t * 0.55 + k / 4) % 1.0
        r = 40 + ph * rmax
        a = int(120 * (1 - ph))
        d.ellipse((cx - r, cy - r, cx + r, cy + r), outline=ACC + (a,), width=3)
    # destinations (abstract, not a literal map)
    pts = [(-330, -420), (300, -360), (-380, 120), (360, 210), (-160, 420), (210, -120),
           (-260, -150), (120, 380), (420, -40), (-60, -380), (-430, -260), (300, 470)]
    for i, (px, py) in enumerate(pts):
        appear = 0.5 + i * 0.16
        p = ease_out((t - appear) / 0.6)
        if p <= 0:
            continue
        x, y = cx + int(px * sx), cy + int(py * sy)
        # line from centre
        lx = cx + (x - cx) * p
        ly = cy + (y - cy) * p
        d.line([(cx, cy), (lx, ly)], fill=ACC + (int(110 * p),), width=2)
        if p > 0.95:
            pulse = 0.5 + 0.5 * math.sin((t - appear) * 5)
            rr = 9 + pulse * 5
            d.ellipse((x - rr * 2.2, y - rr * 2.2, x + rr * 2.2, y + rr * 2.2), fill=ACC + (40,))
            d.ellipse((x - rr, y - rr, x + rr, y + rr), fill=ACC_HI + (255,))
    # centre
    d.ellipse((cx - 22, cy - 22, cx + 22, cy + 22), fill=ACC + (255,))
    d.ellipse((cx - 44, cy - 44, cx + 44, cy + 44), outline=ACC_HI + (200,), width=3)
    return img


_WM = None


def watermark():
    """Small white logo mark shown at the top of footage scenes."""
    global _WM
    if _WM is None:
        lg = Image.open(LOGO_WHITE)
        hh = 92 if LAND else 120
        lg = lg.resize((int(lg.width * hh / lg.height), hh), Image.LANCZOS)
        lg.putalpha(lg.getchannel("A").point(lambda v: int(v * 0.9)))
        _WM = with_shadow(lg, radius=10, opacity=0.5)
    return _WM


class LogoReveal:
    """Animated reveal of the real Professional Ready Mix logo, part by part."""

    # row bands of the source logo (after cropping to its bbox)
    BANDS = [(0, 614, 0.10), (640, 674, 0.55), (704, 793, 0.80), (820, 878, 1.00)]

    def __init__(self, y_center, width=720):
        lg = Image.open(LOGO).convert("RGBA")
        s = width / lg.width
        self.parts = []
        for (y0, y1, t0) in self.BANDS:
            part = lg.crop((0, y0, lg.width, y1))
            part = part.resize((width, max(1, int(part.height * s))), Image.LANCZOS)
            self.parts.append((part, int(y0 * s), t0))
        self.h = int(lg.height * s)
        self.top = y_center - self.h // 2
        self.width = width
        # soft blue glow behind the emblem
        gw = int(width * 1.3)
        yy, xx = np.mgrid[0:gw, 0:gw]
        d = np.sqrt((xx - gw / 2) ** 2 + (yy - gw / 2) ** 2) / (gw / 2)
        a = (np.clip(1 - d, 0, 1) ** 2.2 * 120).astype(np.uint8)
        self.glow = Image.new("RGBA", (gw, gw), ACC + (0,))
        self.glow.putalpha(Image.fromarray(a))
        self.t_in, self.t_out = 0, 99

    def draw(self, canvas, t):
        ga = clamp((t - 0.05) / 0.8)
        if ga > 0:
            g = self.glow.copy()
            g.putalpha(g.getchannel("A").point(lambda v: int(v * ga)))
            canvas.alpha_composite(g, ((W - g.width) // 2, self.top + int(self.parts[0][0].height * 0.5) - g.height // 2))
        x0 = (W - self.width) // 2
        for i, (part, py, t0) in enumerate(self.parts):
            p = ease_out((t - t0) / 0.6)
            if p <= 0:
                continue
            lay = part
            if i == 0:  # emblem: scale up + fade
                sc = 0.85 + 0.15 * p
                lay = part.resize((max(1, int(part.width * sc)), max(1, int(part.height * sc))), Image.LANCZOS)
                lay = lay.copy()
                lay.putalpha(lay.getchannel("A").point(lambda v: int(v * p)))
                canvas.alpha_composite(lay, ((W - lay.width) // 2, self.top + py + (part.height - lay.height) // 2))
            elif i == 1:  # rule grows from the centre
                w = max(2, int(part.width * p))
                lay = part.crop(((part.width - w) // 2, 0, (part.width + w) // 2, part.height))
                canvas.alpha_composite(lay, ((W - w) // 2, self.top + py))
            else:  # wordmark lines: fade + rise
                lay = part.copy()
                lay.putalpha(lay.getchannel("A").point(lambda v: int(v * p)))
                canvas.alpha_composite(lay, (x0, self.top + py + int((1 - p) * 24)))
        # light sweep across the logo once everything is in
        sp = (t - 1.6) / 0.9
        if 0 < sp < 1:
            band = Image.new("L", (W, self.h), 0)
            d = ImageDraw.Draw(band)
            cx = int(-200 + sp * (W + 400))
            d.polygon([(cx - 60, 0), (cx + 20, 0), (cx - 40, self.h), (cx - 120, self.h)], fill=150)
            band = band.filter(ImageFilter.GaussianBlur(18))
            mask = Image.new("L", (W, self.h), 0)
            for part, py, _ in self.parts:
                mask.paste(part.getchannel("A"), (x0, py))
            m = ImageChops.multiply(band, mask)
            shine = Image.new("RGBA", (W, self.h), (255, 255, 255, 0))
            shine.putalpha(m)
            canvas.alpha_composite(shine, (0, self.top))


def endcard_elements(tagline_ar, tagline_en):
    return [
        LogoReveal(400 if LAND else 800, width=500 if LAND else 700),
        Text("بروفيشنال ريدي ميكس", "head", 760 if LAND else 1330, 1.3, 99, size=72),
        Text(tagline_ar, "sub", 860 if LAND else 1430, 1.6, 99, size=48),
        Text(tagline_en, "kicker", 1000 if LAND else 1700, 2.0, 99, size=24),
    ]


# ---- the three promos -----------------------------------------------------------
def build(media_dir):
    V = lambda n: os.path.join(media_dir, "v", f"v{n}.mp4")
    I = {
        "wide_pour": "23cc1059-a89f-4810-8786-1043bb42428f.JPG",
        "boom_mixer": "24610597-73cb-4dec-b1b1-d80bada5ec1d.JPG",
        "mixers": "42c4208a-ae45-4660-8329-ea146478ffe3.JPG",
        "slab_truck": "54b23395-2329-456e-92e4-b27d76cd30ad.JPG",
        "slab_truck2": "55731a47-592d-4055-a443-0d76592254f7.JPG",
        "boom_pink": "5e7a20ff-b0f0-442f-b279-bd2c9312d4a3.JPG",
        "excav": "67b38655-b239-4042-ab06-155842fe36bc.JPG",
        "night_port": "6b6308cb-3e9f-4865-ae43-bbd831bfd9b4.JPG",
        "night": "7ba76962-24ab-4894-8b9d-a6c5f54450db.JPG",
        "yellow_boom": "8514c73b-a14e-4cda-85eb-6a4f2c1e8f8a.JPG",
        "slab_truck3": "8a7aafc7-d0c4-45e8-837c-9f0ae2e17144.JPG",
        "boom_slab_pink": "8b8e9ca7-4509-4eb8-b386-e0495e085f74.JPG",
        "deep_found": "a1a12486-2c9d-4ed4-a9ea-14d46460270f.JPG",
        "found_grid": "a6e66f76-a768-43f1-bf94-43b47168cbc8.JPG",
    }
    img = lambda k: Media("img", os.path.join(media_dir, "imgs", I[k]))
    sideways = "transpose=2"  # these clips were filmed with the phone on its side

    TOP, MID, BOT, BOT2, KICK = 300, 1120, 1480, 1600, 1370
    KF = 380  # kicker position in framed scenes
    if LAND:
        TOP, MID, BOT, BOT2, KICK = 120, 540, 830, 935, 715
        KF = KICK
    end_dur = 4.2
    videos = {}

    # 1 — Identity: strength & fleet
    v1 = [
        Scene(3.0, img("wide_pour"), kb=(1.05, 1.2), focus=((0.15, 0.5), (0.35, 0.5)), elements=[
            Text("PROFESSIONAL READY MIX", "kicker", KICK, 0.3, 3.0),
            Text("كل مشروع كبير", "head", BOT, 0.45, 3.0)]),
        Scene(3.0, img("mixers"), kb=(1.12, 1.0), focus=((0.6, 0.55), (0.4, 0.55)), elements=[
            Text("بيبدأ من", "sub", BOT - 110, 0.25, 3.0, size=58),
            Text("أساس متين", "accent", BOT + 10, 0.4, 3.0, size=120)]),
        Scene(3.6, Media("vid", V(4), 3.0, sideways), mode="frame", frame_y=560, kb=(1.0, 1.08), elements=[
            Text("OUR FLEET", "kicker", KF, 0.2, 3.6),
            Text("أسطول مضخات وخلاطات", "head", BOT, 0.3, 3.6, size=84),
            Text("جاهز يوصل لموقعك في ميعادك", "sub", BOT2 + 20, 0.6, 3.6, size=48)]),
        Scene(3.2, img("slab_truck"), kb=(1.0, 1.14), focus=((0.2, 0.5), (0.45, 0.5)), elements=[
            Text("صبّ دقيق", "head", BOT, 0.3, 3.2),
            Text("وجودة ثابتة في كل خلطة", "sub", BOT2 + 10, 0.55, 3.2)]),
        Scene(3.6, Media("vid", V(5), 4.0), mode="frame", frame_y=560, kb=(1.0, 1.1), elements=[
            Text("DAY & NIGHT", "kicker", KF, 0.2, 3.6),
            Text("شغالين ليل ونهار", "head", BOT, 0.3, 3.6),
            Text("عشان مشروعك ما يستناش", "sub", BOT2 + 10, 0.6, 3.6)]),
        Scene(3.0, img("boom_pink"), kb=(1.18, 1.04), focus=((0.55, 0.4), (0.4, 0.5)), elements=[
            Text("الخرسانة الجاهزة", "head", BOT, 0.3, 3.0),
            Text("بمعايير احترافية", "accent", BOT2 + 20, 0.5, 3.0, size=80)]),
        Scene(end_dur, motion=endcard_bg, grade=False,
              elements=endcard_elements("شريكك من أول صبة", "BUILT ON STRENGTH")),
    ]
    for sc, line in zip(v1, ['كل مشروع كبير..', 'بيبدأ من أساس متين.', 'في بروفيشنال ريدي ميكس، عندنا أسطول مضخات وخلاطات، جاهز يوصلّك في ميعادك.', 'صبّ دقيق.. وجودة ثابتة في كل خلطة.', 'شغّالين ليل ونهار.. عشان مشروعك ما يستناش.', 'خرسانة جاهزة.. بمعايير احترافية.', 'بروفيشنال ريدي ميكس.. شريكك من أول صبّة.']):
        sc.vo = line
    videos["professional-ready-mix-01-strength"] = (v1, [(5.6, 0.5), (14.6, 0.5)])

    # 2 — Projects across West Cairo & government work
    v2 = [
        Scene(3.0, img("night"), kb=(1.05, 1.18), focus=((0.5, 0.45), (0.6, 0.5)), elements=[
            Text("OUR PROJECTS", "kicker", KICK, 0.3, 3.0),
            Text("مشاريعنا بتتكلم", "head", BOT, 0.45, 3.0)]),
        Scene(3.8, Media("vid", V(4), 15.0, sideways), mode="frame", frame_y=560, kb=(1.0, 1.08), elements=[
            Text("WEST CAIRO", "kicker", KF, 0.2, 3.8),
            Text("غرب القاهرة", "accent", BOT - 30, 0.3, 3.8, size=112),
            Chips(["٦ أكتوبر", "الشيخ زايد", "الجيزة"], BOT2 + (35 if LAND else 50), 0.8, 3.8)]),
        Scene(3.0, img("excav"), kb=(1.0, 1.15), focus=((0.25, 0.5), (0.5, 0.5)), elements=[
            Text("مشاريع سكنية", "head", BOT, 0.3, 3.0),
            Text("وتجارية وصناعية", "sub", BOT2 + 10, 0.55, 3.0)]),
        Scene(3.2, img("found_grid"), kb=(1.02, 1.15), focus=((0.5, 0.6), (0.5, 0.4)), elements=[
            Text("GOVERNMENT PROJECTS", "kicker", KICK, 0.2, 3.2),
            Text("ومشاريع حكومية كبرى", "head", BOT, 0.35, 3.2, size=92)]),
        Scene(3.6, Media("vid", V(6), 8.0, sideways), mode="frame", frame_y=560, kb=(1.0, 1.1), elements=[
            Text("FROM FOUNDATION UP", "kicker", KF, 0.2, 3.6),
            Text("من الأساسات", "head", BOT, 0.3, 3.6),
            Text("لحد آخر صبة", "accent", BOT2 + 20, 0.6, 3.6, size=80)]),
        Scene(3.0, img("deep_found"), kb=(1.12, 1.0), focus=((0.5, 0.5), (0.5, 0.5)), elements=[
            Text("خبرة على أرض الواقع", "head", BOT, 0.3, 3.0, size=92)]),
        Scene(end_dur, motion=endcard_bg, grade=False,
              elements=endcard_elements("ثقة بتتبني صبة ورا صبة", "WEST CAIRO  •  GIZA  •  EGYPT")),
    ]
    for sc, line in zip(v2, ['مشاريعنا هي اللي بتتكلم عننا.', 'في غرب القاهرة.. من أكتوبر، للشيخ زايد، للجيزة.', 'مشاريع سكنية، وتجارية، وصناعية..', 'ومشاريع حكومية كبرى.', 'من أول الأساسات.. لحد آخر صبّة.', 'خبرة حقيقية على أرض الواقع.', 'بروفيشنال ريدي ميكس.. ثقة بتتبني صبّة ورا صبّة.']):
        sc.vo = line
    videos["professional-ready-mix-02-projects"] = (v2, [(2.6, 0.45), (15.8, 0.45)])

    # 3 — Reach beyond Cairo & Giza
    v3 = [
        Scene(3.0, img("boom_mixer"), kb=(1.05, 1.2), focus=((0.45, 0.4), (0.6, 0.45)), elements=[
            Text("BEYOND CAIRO & GIZA", "kicker", KICK, 0.3, 3.0),
            Text("مش بس القاهرة والجيزة", "head", BOT, 0.45, 3.0, size=90)]),
        Scene(4.0, motion=reach_bg, grade=False, elements=[
            Text("شغلنا وصل", "head", 105 if LAND else 300, 0.4, 4.0),
            Text("لمحافظات كتير", "accent", 225 if LAND else 430, 0.7, 4.0, size=110),
            Text("ACROSS EGYPT'S GOVERNORATES", "kicker", 1020 if LAND else 1650, 1.2, 4.0)]),
        Scene(3.4, Media("vid", V(4), 20.0, sideways), mode="frame", frame_y=560, kb=(1.0, 1.08), elements=[
            Text("SAME STANDARD", "kicker", KF, 0.2, 3.4),
            Text("نفس الجودة", "head", BOT, 0.3, 3.4),
            Text("ونفس الالتزام في كل موقع", "sub", BOT2 + 10, 0.6, 3.4)]),
        Scene(3.0, img("yellow_boom"), kb=(1.0, 1.15), focus=((0.3, 0.5), (0.55, 0.45)), elements=[
            Text("أسطول جاهز للتنقل", "head", BOT, 0.3, 3.0, size=92),
            Text("لأي مكان مشروعك فيه", "sub", BOT2 + 10, 0.55, 3.0)]),
        Scene(3.4, Media("vid", V(5), 12.0), mode="frame", frame_y=560, kb=(1.0, 1.1), elements=[
            Text("ON TIME", "kicker", KF, 0.2, 3.4),
            Text("مهما كانت المسافة", "head", BOT, 0.3, 3.4, size=92),
            Text("الصبة في ميعادها", "accent", BOT2 + 20, 0.6, 3.4, size=80)]),
        Scene(3.0, img("boom_slab_pink"), kb=(1.15, 1.02), focus=((0.6, 0.5), (0.4, 0.5)), elements=[
            Text("في كل مكان بنبني", "head", BOT, 0.3, 3.0, size=92),
            Text("إحنا موجودين", "sub", BOT2 + 10, 0.55, 3.0)]),
        Scene(end_dur, motion=endcard_bg, grade=False,
              elements=endcard_elements("شريكك في كل محافظة", "READY MIX CONCRETE  •  EGYPT")),
    ]
    for sc, line in zip(v3, ['شغلنا مش بس في القاهرة والجيزة..', 'إحنا وصلنا لمحافظات كتير في مصر.', 'بنفس الجودة.. ونفس الالتزام في كل موقع.', 'أسطولنا جاهز يتحرك لأي مكان مشروعك فيه.', 'ومهما كانت المسافة.. الصبّة في ميعادها.', 'في كل مكان بنبني.. إحنا موجودين.', 'بروفيشنال ريدي ميكس.. شريكك في كل محافظة.']):
        sc.vo = line
    videos["professional-ready-mix-03-reach"] = (v3, [(2.6, 0.5), (16.4, 0.45)])
    return videos


def main():
    global ARGS
    ap = argparse.ArgumentParser()
    ap.add_argument("--media", required=True)
    ap.add_argument("--fonts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", default="1,2,3")
    ap.add_argument("--preview", action="store_true")
    ap.add_argument("--voice", default="Algenib")
    ap.add_argument("--aspect", default="9x16", choices=["9x16", "16x9"])
    ARGS = ap.parse_args()
    global W, H, LAND
    if ARGS.aspect == "16x9":
        W, H, LAND = 1920, 1080, True
    os.makedirs(ARGS.out, exist_ok=True)
    init_finish()
    vids = build(ARGS.media)
    wanted = {int(x) for x in ARGS.only.split(",")}
    import hashlib
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from tts import tts
    vo_dir = os.path.join(ARGS.out, ".vo")
    os.makedirs(vo_dir, exist_ok=True)
    VO_LEAD, VO_TAIL = 0.3, 0.35
    for k, (name, (scenes, _)) in enumerate(vids.items(), 1):
        if k not in wanted:
            continue
        xf = 0.4
        # 1) voice-over per scene (cached), trimmed of silence; scene stretches to fit its line
        for s in scenes:
            if not s.vo:
                continue
            h = hashlib.sha1((s.vo + ARGS.voice).encode()).hexdigest()[:12]
            raw, clean = os.path.join(vo_dir, h + "_raw.wav"), os.path.join(vo_dir, h + ".wav")
            if not os.path.exists(clean):
                for attempt in range(0 if os.path.exists(raw) else 4):
                    tts(s.vo, raw, voice=ARGS.voice)
                    d = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of",
                                              "csv=p=0", raw], capture_output=True, text=True).stdout)
                    if d < 1.2 + len(s.vo) * 0.11:  # guard against the model reading the direction aloud
                        break
                    print("  VO too long, retrying:", s.vo, d)
                subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", raw, "-af",
                                "silenceremove=start_periods=1:start_threshold=-45dB,areverse,"
                                "silenceremove=start_periods=1:start_threshold=-45dB,areverse,"
                                "highpass=f=70,acompressor=threshold=-18dB:ratio=3:attack=5:release=120,"
                                "equalizer=f=180:t=q:w=1:g=2,atempo=1.08", clean], check=True)
            s.vo_wav = clean
            s.vo_clip = read_wav(clean)
            need = VO_LEAD + len(s.vo_clip) / SR + VO_TAIL + (1.4 if s.motion is endcard_bg else 0)
            if need > s.dur:
                s.set_dur(round(need, 2))
        total = sum(s.dur for s in scenes) - xf * (len(scenes) - 1)

        def music(tot, path, starts, scenes=scenes, k=k):
            hits = [(starts[1], 0.45), (starts[len(scenes) // 2], 0.4), (starts[-1] + 0.1, 0.9)]
            bed = synth_music(tot, bpm=(96, 100, 92)[k - 1], root=(55.0, 49.0, 52.0)[k - 1], hits=hits)
            events = [(st + VO_LEAD + (0.5 if sc.motion is endcard_bg else 0), sc.vo_clip)
                      for st, sc in zip(starts, scenes) if sc.vo]
            mix_audio(tot, bed, events, path)

        suffix = "-16x9" if LAND else ""
        render_video(name + suffix, scenes, ARGS.out, music, xfade=xf, preview=ARGS.preview)


if __name__ == "__main__":
    main()
