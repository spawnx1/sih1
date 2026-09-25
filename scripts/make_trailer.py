"""
make_trailer.py -- renders the MULE CONNECTION cinematic explainer to a real .mp4.

Frame-by-frame with Pillow -> H.264 via imageio-ffmpeg (bundled ffmpeg), then a
generated minimal soundscape (numpy) is muxed in. No browser, no system ffmpeg.

    .venv\\Scripts\\python.exe scripts\\make_trailer.py
    -> docs/mule_connection.mp4
"""
import math, os, subprocess, wave, struct, random
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageChops
import imageio.v2 as imageio
import imageio_ffmpeg

W, H, FPS = 1920, 1080, 30
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "mule_connection.mp4")
BG = (5, 7, 10)
INK = (238, 242, 246); DIM = (140, 151, 163); FAINT = (95, 108, 118)
CYAN = (86, 186, 205); AMBER = (214, 169, 92); VIOLET = (140, 128, 224)
RED = (224, 86, 108); GREEN = (70, 196, 149)
FDIR = "C:/Windows/Fonts/"
_fc = {}
def F(size, w="reg"):
    key = (size, w)
    if key not in _fc:
        fn = {"light": "segoeuil.ttf", "reg": "segoeui.ttf", "semi": "segoeuisl.ttf",
              "bold": "segoeuib.ttf", "mono": "consola.ttf"}.get(w, "segoeui.ttf")
        p = FDIR + fn
        if not os.path.exists(p):
            p = FDIR + "segoeui.ttf"
        _fc[key] = ImageFont.truetype(p, size)
    return _fc[key]

def sat(x): return 0.0 if x < 0 else 1.0 if x > 1 else x
def ss(x):  x = sat(x); return x * x * (3 - 2 * x)                      # smoothstep
def eo(x):  return 1 - (1 - sat(x)) ** 3                                # ease-out cubic
def A(c, a):
    a = sat(a); return (int(c[0] * a), int(c[1] * a), int(c[2] * a))    # fade toward black

# radial glow sprite (L)
_G = 256
_gx = np.linspace(-1, 1, _G)
_gxx, _gyy = np.meshgrid(_gx, _gx)
_gr = np.sqrt(_gxx**2 + _gyy**2)
_gl = np.clip(1 - _gr, 0, 1) ** 2.2
GLOW = Image.fromarray((_gl * 255).astype("uint8"), "L")

def node(base, x, y, r, color, a=1.0, core=True):
    a = sat(a)
    if a <= 0: return
    s = max(2, int(r * 4.2))
    m = GLOW.resize((s, s))
    x0, y0 = int(x - s / 2), int(y - s / 2)
    base.paste(A(color, 0.85 * a), (x0, y0, x0 + s, y0 + s), m)
    if core:
        d = ImageDraw.Draw(base)
        rr = r * 0.55
        d.ellipse([x - rr, y - rr, x + rr, y + rr], fill=A(color, min(1, a * 1.25)))

def line(base, p1, p2, color, a=1.0, w=2, prog=1.0):
    a = sat(a)
    if a <= 0: return
    x1, y1 = p1; x2, y2 = p2
    prog = sat(prog)
    x2 = x1 + (x2 - x1) * prog; y2 = y1 + (y2 - y1) * prog
    ImageDraw.Draw(base).line([x1, y1, x2, y2], fill=A(color, a), width=w)

def text(base, s, cx, y, size, color=INK, a=1.0, w="light", track=0, anchor="mm"):
    a = sat(a)
    if a <= 0 or not s: return
    d = ImageDraw.Draw(base); f = F(size, w); col = A(color, a)
    if track == 0:
        d.text((cx, y), s, font=f, fill=col, anchor=anchor)
        return
    widths = [d.textlength(ch, font=f) + track for ch in s]
    total = sum(widths) - track
    x = cx - total / 2
    for ch, wd in zip(s, widths):
        d.text((x, y), ch, font=f, fill=col, anchor="lm")
        x += wd

def block(base, x0, y0, x1, y1, color, a):
    ImageDraw.Draw(base).rectangle([x0, y0, x1, y1], fill=A(color, a))

# --------------------------------------------------------------------------
# deterministic layout for the network scene
# --------------------------------------------------------------------------
random.seed(26184)
NET_N = 34
_cx, _cy = W * 0.5, H * 0.54
NET = []
for i in range(NET_N):
    ang = random.uniform(0, 2 * math.pi); rad = 60 + 300 * math.sqrt(random.random())
    grp = random.choice([CYAN, CYAN, AMBER, VIOLET])
    NET.append({"x": _cx + math.cos(ang) * rad * 1.35, "y": _cy + math.sin(ang) * rad,
                "r": random.uniform(9, 22), "c": grp, "d": random.uniform(0, 1),
                "red": random.random() < 0.14})
NET_E = []
for i in range(NET_N):
    for _ in range(random.randint(1, 2)):
        j = random.randrange(NET_N)
        if j != i: NET_E.append((i, j, random.uniform(0, 1)))

# --------------------------------------------------------------------------
# SCENES  (name, start, dur)  -- fade-through-black between them
# --------------------------------------------------------------------------
SCN = [("cold", 0.0, 5.0), ("flow", 5.0, 10.0), ("asof", 15.0, 9.0),
       ("fore", 24.0, 11.0), ("mule", 35.0, 10.0), ("net", 45.0, 11.0),
       ("act", 56.0, 5.0), ("logo", 61.0, 6.0)]
TOTAL = SCN[-1][1] + SCN[-1][2]
FIN, FOUT = 0.7, 0.7

def draw_cold(b, t, dur):
    a = ss((t - .4) / 1.4)
    text(b, "Behind every transaction", W/2, H/2 - 44, 62, INK, a, "light", 1)
    text(b, "is a connection.", W/2, H/2 + 34, 62, INK, ss((t - 1.1) / 1.4), "light", 1)

def flow_nodes():
    xs = [W*0.13, W*0.30, W*0.47, W*0.64, W*0.81]; y = H*0.5
    labels = ["VICTIM", "MULE", "MULE", "MULE", "ATM"]
    cols = [CYAN, DIM, DIM, AMBER, RED]
    return xs, y, labels, cols

def draw_flow(b, t, dur):
    xs, y, labels, cols = flow_nodes()
    text(b, "THE PROBLEM", W/2, H*0.2, 20, FAINT, ss(t/0.8), "semi", 6)
    amts = ["\u20b985,000", "\u20b978,000", "\u20b971,000", "\u20b966,000", "CASH"]
    for i in range(5):
        na = ss((t - 0.6 - i * 0.7) / 0.7)
        if i < 4:
            lp = eo((t - 1.0 - i * 0.7) / 0.7)
            line(b, (xs[i], y), (xs[i+1], y), CYAN if i < 3 else RED, 0.5 * na, 3, lp)
        node(b, xs[i], y, 26 if i in (0,) else 22, cols[i], na)
        text(b, labels[i], xs[i], y + 52, 17, DIM, na, "reg", 2)
        text(b, amts[i], xs[i], y - 50, 18, INK if i < 4 else RED, na, "mono")
    text(b, "Stolen money is layered through mule accounts and cashed out in minutes.",
         W/2, H*0.78, 27, DIM, ss((t - 3.6) / 1.0), "light")

def draw_asof(b, t, dur):
    y = H*0.52; x0, x1 = W*0.14, W*0.86; axf = eo(t/1.2)
    line(b, (x0, y), (x0 + (x1 - x0) * axf, y), (60, 74, 84), 1.0, 2)
    lx = W*0.6
    la = ss((t - 1.0) / 0.8)
    ImageDraw.Draw(b).line([lx, y - 150, lx, y + 150], fill=A(RED, 0.9 * la), width=3)
    text(b, "as_of", lx, y - 172, 20, RED, la, "mono", 2)
    # past features (left, cyan dots) fade in; future greyed
    random.seed(7)
    for i in range(9):
        px = x0 + (lx - x0) * (0.08 + 0.84 * i / 8); dot = ss((t - 1.4 - i * 0.12) / 0.6)
        node(b, px, y, 8, CYAN, dot * 0.9, core=True)
    text(b, "PAST", (x0 + lx) / 2, y + 60, 18, DIM, ss((t-1.4)/0.8), "reg", 4)
    text(b, "FUTURE", (lx + x1) / 2, y + 60, 18, FAINT, ss((t-1.4)/0.8), "reg", 4)
    text(b, "Every forecast uses only the past.", W/2, H*0.2, 40, INK, ss((t-0.4)/1.0), "light", 1)
    text(b, "f(account, as_of)  \u00b7  no leakage \u2014 verified bit-for-bit by an automated test.",
         W/2, H*0.8, 24, DIM, ss((t-2.6)/1.0), "reg")

def draw_fore(b, t, dur):
    text(b, "Will it cash out?   When?   Where?", W/2, H*0.18, 34, INK, ss((t-0.3)/1.0), "light", 1)
    # big percent
    pa = ss((t - 1.0) / 0.8); pct = int(eo((t - 1.0) / 2.2) * 73)
    text(b, f"{pct}", W*0.31, H*0.52, 190, RED if t > 3.4 else CYAN, pa, "light")
    text(b, "%", W*0.31 + 150, H*0.44, 60, DIM, pa, "light")
    text(b, "chance of cash-out in 10 min", W*0.31, H*0.66, 22, DIM, ss((t-1.6)/0.8), "reg")
    # hazard curve
    ox, oy, cw, ch = W*0.54, H*0.62, W*0.34, H*0.26
    ImageDraw.Draw(b).line([ox, oy, ox + cw, oy], fill=A(FAINT, 0.5 * ss((t-1)/0.6)), width=2)
    ImageDraw.Draw(b).line([ox, oy, ox, oy - ch], fill=A(FAINT, 0.5 * ss((t-1)/0.6)), width=2)
    pts = [(0, .04), (.25, .42), (.5, .66), (.72, .80), (1, .88)]
    cp = eo((t - 1.4) / 2.4)
    prev = None
    for k in range(1, 61):
        f = k / 60.0
        if f > cp: break
        # sample piecewise
        seg = min(f, 1.0)
        xx = ox + cw * seg
        # interp y
        yy = 0.04
        for a2 in range(len(pts) - 1):
            if pts[a2][0] <= seg <= pts[a2+1][0]:
                tt = (seg - pts[a2][0]) / (pts[a2+1][0] - pts[a2][0] + 1e-6)
                yy = pts[a2][1] + (pts[a2+1][1] - pts[a2][1]) * tt
        py = oy - ch * yy
        if prev: line(b, prev, (xx, py), CYAN, 0.95, 4)
        prev = (xx, py)
    text(b, "cash-out hazard curve", ox + cw/2, oy + 30, 18, FAINT, ss((t-1.4)/0.8), "reg", 2)
    # three model tags
    tags = [("M1", "will \u00b7 when"), ("M2", "how"), ("M3", "where")]
    for i, (m, s) in enumerate(tags):
        ta = ss((t - 2.4 - i*0.3)/0.7); tx = W*0.54 + i*W*0.11
        text(b, m, tx, H*0.30, 30, CYAN, ta, "semi", 2)
        text(b, s, tx, H*0.35, 18, DIM, ta, "reg", 1)
    # RED tier flip
    if t > 3.6:
        ra = ss((t-3.6)/0.5)
        block(b, W*0.40, H*0.86, W*0.60, H*0.92, RED, 0.14*ra)
        text(b, "TIER  RED  \u00b7  FREEZE", W/2, H*0.89, 26, RED, ra, "semi", 3)
    text(b, "XGBoost hazard \u00b7 hierarchical abstention \u00b7 ROC-AUC 0.913",
         W/2, H*0.955, 20, FAINT, ss((t-4.4)/1.0), "mono")

def draw_mule(b, t, dur):
    text(b, "One account.", W/2, H*0.16, 46, INK, ss((t-0.3)/0.9), "light", 1)
    mx, my = W*0.66, H*0.55
    vs = [(W*0.24, H*0.35), (W*0.22, H*0.55), (W*0.24, H*0.75)]
    for i, v in enumerate(vs):
        na = ss((t - 0.8 - i*0.35)/0.7); lp = eo((t - 1.4 - i*0.35)/0.8)
        line(b, v, (mx, my), CYAN, 0.5*na, 3, lp)
        node(b, v[0], v[1], 20, CYAN, na)
        text(b, f"VICTIM {i+1}", v[0], v[1] + 42, 16, DIM, na, "reg", 2)
    ma = ss((t-1.0)/0.8)
    node(b, mx, my, 40, AMBER, ma)
    text(b, "NOMAD-2405", mx, my - 66, 26, INK, ss((t-1.6)/0.8), "semi", 2)
    text(b, "cash-out mule", mx, my + 60, 18, AMBER, ss((t-1.8)/0.8), "reg", 1)
    # victims counter
    vc = int(eo((t-2.6)/2.0) * 11)
    text(b, f"{vc}", W*0.66, H*0.20, 64, RED, ss((t-2.6)/0.6), "light")
    text(b, "victims, one identity", W/2, H*0.20 + 4, 46, INK, ss((t-2.2)/0.9), "light", 1) if False else None
    text(b, "Eleven victims. One identity.", W/2, H*0.86, 30, DIM, ss((t-4.0)/1.0), "light")

def draw_net(b, t, dur):
    # edges then nodes fade in over the scene
    for (i, j, d) in NET_E:
        ea = ss((t - 0.6 - d*2.2)/1.2) * 0.5
        n1, n2 = NET[i], NET[j]
        c = RED if (n1["red"] and n2["red"]) else (60, 78, 88)
        line(b, (n1["x"], n1["y"]), (n2["x"], n2["y"]), c, ea, 2)
    for n in NET:
        na = ss((t - 0.4 - n["d"]*2.4)/1.0)
        c = RED if n["red"] else n["c"]
        node(b, n["x"], n["y"], n["r"], c, na)
    text(b, "Five hundred accounts.", W/2, H*0.12, 40, INK, ss((t-0.3)/1.0), "light", 1)
    text(b, "One intelligence map.", W/2, H*0.12 + 52, 40, INK, ss((t-1.0)/1.0), "light", 1) if t < 100 else None
    text(b, "behaviour clusters \u00b7 shared-case links \u00b7 codenamed \u00b7 click to investigate",
         W/2, H*0.9, 22, DIM, ss((t-3.2)/1.2), "reg")

def draw_act(b, t, dur):
    x, y = W/2, H*0.48
    pulse = 0.5 + 0.5*math.sin(t*7)
    node(b, x, y, 48 + pulse*8, RED, ss(t/0.6))
    sa = ss((t-0.8)/0.5)
    text(b, "FREEZE", x, y, 40, INK, sa, "bold", 6)
    text(b, "Freeze the account before the golden hour closes.",
         W/2, H*0.74, 30, DIM, ss((t-1.6)/1.0), "light")

def draw_logo(b, t, dur):
    text(b, "MULE CONNECTION", W/2, H/2 - 10, 66, INK, ss((t-0.4)/1.2), "light", 8)
    lw = ss((t-1.2)/0.8)
    ImageDraw.Draw(b).line([W/2 - 180*lw, H/2 + 40, W/2 + 180*lw, H/2 + 40], fill=A(CYAN, 0.8*lw), width=2)
    text(b, "See the connections.  Stop the network.", W/2, H/2 + 84, 26, DIM, ss((t-2.0)/1.2), "reg", 1)

DISPATCH = {"cold": draw_cold, "flow": draw_flow, "asof": draw_asof, "fore": draw_fore,
            "mule": draw_mule, "net": draw_net, "act": draw_act, "logo": draw_logo}

# vignette
_vy, _vx = np.mgrid[0:H, 0:W]
_vd = np.sqrt(((_vx - W/2)/(W*0.62))**2 + ((_vy - H/2)/(H*0.62))**2)
VIG = Image.fromarray((np.clip(1 - _vd*0.55, 0.25, 1) * 255).astype("uint8"), "L").convert("RGB")

def render_frame(t):
    b = Image.new("RGB", (W, H), BG)
    for name, s0, dur in SCN:
        if s0 <= t < s0 + dur:
            tl = t - s0
            DISPATCH[name](b, tl, dur)
            env = 1.0
            if tl < FIN: env = ss(tl / FIN)
            elif tl > dur - FOUT: env = ss((dur - tl) / FOUT)
            b = ImageChops.multiply(b, VIG)
            if env < 1:
                b = Image.eval(b, lambda p: int(p * env))
            return b
    return ImageChops.multiply(b, VIG)

def make_audio(path):
    sr = 44100; n = int(TOTAL * sr); tt = np.arange(n) / sr
    out = np.zeros(n)
    lfo = 0.5 + 0.5*np.sin(2*np.pi*0.08*tt)
    out += 0.05 * np.sin(2*np.pi*48*tt) * (0.6 + 0.4*lfo)
    out += 0.03 * np.sin(2*np.pi*96*tt) * (0.5 + 0.5*lfo)
    def blip(at, freq, dur, amp, kind="tick"):
        i0 = int(at*sr); i1 = min(n, i0 + int(dur*sr))
        if i0 >= n: return
        lt = np.arange(i1 - i0)/sr
        env = np.exp(-lt/ (dur*0.4))
        w = np.sin(2*np.pi*freq*lt)
        if kind == "tick": w = np.sign(np.sin(2*np.pi*freq*lt))*0.4 + w*0.6
        out[i0:i1] += amp*env*w
    # ticks at scene starts + beats
    for name, s0, dur in SCN:
        blip(s0, 1400, 0.06, 0.08)
    for i in range(5): blip(6.0 + i*0.7, 1600, 0.05, 0.06)     # flow draws
    blip(15.2, 1200, 0.08, 0.07)
    # impacts
    for at in [17.6, 27.6, 38.0, 45.2, 56.2, 61.2]:
        blip(at, 55, 0.5, 0.28)
    # rising tension into red flip (~t 33)
    r0, r1 = int(30*sr), int(34*sr)
    rt = np.arange(r1-r0)/sr
    out[r0:r1] += 0.06*np.sin(2*np.pi*(120+40*rt)*rt)*np.linspace(0,1,r1-r0)
    out = out / (np.max(np.abs(out)) + 1e-6) * 0.7
    pcm = (out*32767).astype(np.int16)
    with wave.open(path, "w") as wv:
        wv.setnchannels(1); wv.setsampwidth(2); wv.setframerate(sr)
        wv.writeframes(pcm.tobytes())

def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    tmp = OUT.replace(".mp4", "_silent.mp4")
    wav = OUT.replace(".mp4", ".wav")
    nfr = int(TOTAL * FPS)
    print(f"rendering {nfr} frames @ {FPS}fps ({TOTAL:.1f}s)...")
    wr = imageio.get_writer(tmp, fps=FPS, codec="libx264", quality=7,
                            macro_block_size=8, ffmpeg_log_level="error",
                            output_params=["-pix_fmt", "yuv420p"])
    for i in range(nfr):
        fr = render_frame(i / FPS)
        wr.append_data(np.asarray(fr))
        if i % 60 == 0: print(f"  {i}/{nfr}")
    wr.close()
    print("audio...")
    make_audio(wav)
    print("mux...")
    exe = imageio_ffmpeg.get_ffmpeg_exe()
    subprocess.run([exe, "-y", "-i", tmp, "-i", wav, "-c:v", "copy",
                    "-c:a", "aac", "-b:a", "160k", "-shortest", OUT],
                   check=True, capture_output=True)
    os.remove(tmp); os.remove(wav)
    print("done ->", OUT, f"({os.path.getsize(OUT)//1024} KB)")

if __name__ == "__main__":
    main()
