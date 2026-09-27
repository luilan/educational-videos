"""Video 11 — Training: Learning from Mistakes.

Render from the repo root:  ./render.sh how-llms-work v11
"""
import json
import os

import numpy as np
from PIL import Image, ImageDraw
from manim import *

from common import MONO, code_panel, finish, highlight, next_up_card, token, token_row
from intro import play_token_intro
from v11_script import NEXT, TAGLINE, TITLE
from voiced_scene import VoicedScene

LOSS_COLOR = ORANGE
GRAD_COLOR = YELLOW
WEIGHT_COLOR = BLUE_C
MODEL_COLOR = PURPLE_B
TRAIN_COLOR = TEAL_C
VAL_COLOR = YELLOW

HERE = os.path.dirname(os.path.abspath(__file__))
TRAIN_LOG = os.path.join(HERE, "tiny_gpt", "training_log.json")

GIBBERISH = "lGk'kFYAe:cLs:QzoYSky"   # real start of the tiny GPT's step-0 sample
SENT = ["The", "cat", "sat", "on", "the", "mat"]

# Section 4: an illustrative batch with one (invented) loss per position
BATCH = [(["The", "cat", "sat", "on", "the", "mat"], [4.8, 3.6, 2.1, 0.9, 0.5, 1.4]),
         (["To", "be", "or", "not", "to", "be"], [5.2, 2.7, 1.2, 0.8, 0.3, 0.1]),
         (["I", "think", ",", "therefore", "I", "am"], [3.9, 4.1, 2.6, 1.9, 0.6, 0.2])]
BATCH_MEAN = float(np.mean([v for _, vals in BATCH for v in vals]))

CODE = """for step in range(max_steps):
    x, y = get_batch(train_data)     # text, and the next tokens
    logits = model(x)                # forward pass
    loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
    opt.zero_grad()
    loss.backward()                  # backpropagation
    opt.step()                       # nudge every weight"""


# ---------------------------------------------------------------------------- helpers
def label(text, color=GREY_B, font_size=24, **kw):
    return Text(text, font_size=font_size, color=color, **kw)


def caption(text):
    return Text(text, font_size=20, color=GREY_B).to_corner(DR, buff=0.3)


def arrow(start, end, color=GREY_B, width=4, ratio=0.25):
    return Arrow(start, end, buff=0, color=color, stroke_width=width, max_tip_length_to_length_ratio=ratio,
                 max_stroke_width_to_length_ratio=10)


def box_to(tok, color, opacity=0.3):
    """Recolour a token's box only (never the whole box+label group)."""
    return tok[0].animate.set_stroke(color=color).set_fill(color=color, opacity=opacity)


def labelled_box(text, color, width, height=0.8, font_size=26, opacity=0.2):
    box = RoundedRectangle(corner_radius=0.14, width=width, height=height, stroke_color=color, stroke_width=3,
                           fill_color=color, fill_opacity=opacity)
    return VGroup(box, Text(text, font_size=font_size).move_to(box))


def noise_color(rng):
    hue = [BLUE_C, BLUE_E, GREY_B, TEAL_E, GREY_D][rng.integers(5)]
    return interpolate_color(GREY_E, hue, rng.uniform(0.1, 1.0))


def calm_color(r, c):
    v = 0.5 + 0.35 * np.sin(0.45 * c + 0.6 * r)
    return interpolate_color(BLUE_E, BLUE_B, v)


def neg_log(p):
    return -float(np.log(p))


# ---------------------------------------------------------------------------- loss landscape
LAND_TH, LAND_EL = np.radians(-32), np.radians(42)   # view rotation and elevation
LAND_S, LAND_ZS = 1.05, 0.95                         # scene units per weight unit, height scale
LAND_PPU = 135                                       # pixels per scene unit at 1080p
LAND_CENTER = np.array([0.0, -0.55, 0.0])
LAND_START = (-1.6, -1.1)
LAND_LR = 0.5


def land_g(a, b):
    """Illustrative loss surface in view-aligned coordinates (a: screen right, b: depth)."""
    return (1.5
            - 1.55 * np.exp(-((a - 1.3) ** 2 / 2.2 + (b - 0.5) ** 2 / 1.3))
            - 0.5 * np.exp(-((a + 0.7) ** 2 / 0.6 + (b - 2.3) ** 2 / 0.6))
            + 1.0 * np.exp(-((a + 2.5) ** 2 / 1.3 + (b + 0.1) ** 2 / 1.6))
            + 0.45 * np.exp(-((a - 0.4) ** 2 / 1.2 + (b + 2.6) ** 2 / 0.7))
            + 0.8 * np.exp(-((a - 2.8) ** 2 / 1.2 + (b - 2.6) ** 2 / 1.0))
            + 0.03 * (a ** 2 + b ** 2)
            + 0.07 * np.sin(1.4 * a + 0.5) * np.cos(1.2 * b))


def land_f(x, y):
    return land_g(x * np.cos(LAND_TH) - y * np.sin(LAND_TH), x * np.sin(LAND_TH) + y * np.cos(LAND_TH))


def land_grad(x, y, h=1e-4):
    return np.array([(land_f(x + h, y) - land_f(x - h, y)) / (2 * h),
                     (land_f(x, y + h) - land_f(x, y - h)) / (2 * h)])


def land_uv(x, y, z):
    """Projected (u, v) in scene units plus a depth value for painter's ordering."""
    xr = x * np.cos(LAND_TH) - y * np.sin(LAND_TH)
    yr = x * np.sin(LAND_TH) + y * np.cos(LAND_TH)
    zz = z * LAND_ZS
    return (LAND_S * xr, LAND_S * (yr * np.sin(LAND_EL) + zz * np.cos(LAND_EL)),
            yr * np.cos(LAND_EL) - zz * np.sin(LAND_EL))


class Landscape:
    """The surface is rasterised once with PIL (shaded quads, back to front); overlays use point()."""

    def __init__(self, n=90, extent=3.0, wire=6):
        xs = np.linspace(-extent, extent, n + 1)
        X, Y = np.meshgrid(xs, xs, indexing="ij")
        Z = land_f(X, Y)
        U, V, D = land_uv(X, Y, Z)
        pad = 0.1
        umin, umax, vmin, vmax = U.min() - pad, U.max() + pad, V.min() - pad, V.max() + pad
        k = 2 * LAND_PPU
        W, H = int((umax - umin) * k), int((vmax - vmin) * k)
        img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        zmin, zmax = Z.min(), Z.max()
        lo, mid, hi = np.array([20, 70, 110]), np.array([90, 60, 120]), np.array([200, 90, 50])
        light = np.array([-0.4, 0.5, 0.8])
        light /= np.linalg.norm(light)
        step = xs[1] - xs[0]
        quads = sorted(((D[i:i + 2, j:j + 2].mean(), i, j) for i in range(n) for j in range(n)), reverse=True)
        line = (20, 20, 30, 255)
        for _, i, j in quads:
            pts = [((U[a, b] - umin) * k, (vmax - V[a, b]) * k)
                   for a, b in [(i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1)]]
            t = (Z[i:i + 2, j:j + 2].mean() - zmin) / (zmax - zmin)
            c = lo + (mid - lo) * (t / 0.5) if t < 0.5 else mid + (hi - mid) * ((t - 0.5) / 0.5)
            dzx = (Z[i + 1, j] - Z[i, j] + Z[i + 1, j + 1] - Z[i, j + 1]) / 2 / step
            dzy = (Z[i, j + 1] - Z[i, j] + Z[i + 1, j + 1] - Z[i + 1, j]) / 2 / step
            nrm = np.array([-dzx * LAND_ZS, -dzy * LAND_ZS, 1.0])
            shade = 0.55 + 0.6 * max(0.0, float(nrm @ light) / np.linalg.norm(nrm))
            col = tuple(int(min(255, v * shade)) for v in c) + (255,)
            draw.polygon(pts, fill=col, outline=col)
            if i % wire == 0:
                draw.line([pts[0], pts[3]], fill=line, width=2)
            if (i + 1) % wire == 0:
                draw.line([pts[1], pts[2]], fill=line, width=2)
            if j % wire == 0:
                draw.line([pts[0], pts[1]], fill=line, width=2)
            if (j + 1) % wire == 0:
                draw.line([pts[3], pts[2]], fill=line, width=2)
        img = img.resize((W // 2, H // 2), Image.LANCZOS)
        self.offset = LAND_CENTER - np.array([(umin + umax) / 2, (vmin + vmax) / 2, 0])
        self.mob = ImageMobject(np.array(img))
        self.mob.stretch_to_fit_width(umax - umin).stretch_to_fit_height(vmax - vmin).move_to(LAND_CENTER)

    def point(self, x, y):
        u, v, _ = land_uv(x, y, land_f(x, y))
        return np.array([u, v, 0.0]) + self.offset

    def surface_line(self, p, q, n=24):
        """Points of the straight weight-space segment p -> q, lifted onto the surface."""
        return [self.point(*(np.array(p) + (np.array(q) - np.array(p)) * s)) for s in np.linspace(0, 1, n)]


def descent_path(start, lr, steps):
    p = np.array(start, dtype=float)
    path = [p.copy()]
    for _ in range(steps):
        p = p - lr * land_grad(*p)
        path.append(p.copy())
    return path


# ---------------------------------------------------------------------------- real training log
def load_log():
    with open(TRAIN_LOG) as f:
        log = json.load(f)
    steps = np.array([s for s, _ in log["train"]], dtype=float)
    loss = np.array([v for _, v in log["train"]], dtype=float)
    n = len(loss)
    smooth = np.empty(n)
    for k in range(n):   # centred 50-step moving average, window shrinking at the ends
        h = min(k, 25, n - 1 - k)
        smooth[k] = loss[k - h:k + h + 1].mean()
    val = np.array(log["val"], dtype=float)
    return steps, loss, smooth, val, log["params"]


# ---------------------------------------------------------------------------- series icons (section 12)
def icon_tokens():
    return token_row(["The", "cat"], buff=0.08, font_size=22)


def icon_embeddings():
    vals = [0.9, 0.3, 0.7, 0.15, 0.55]
    return VGroup(*[Square(0.24, stroke_color=BLUE_B, stroke_width=1.5,
                           fill_color=interpolate_color(BLACK, BLUE_C, 0.25 + 0.75 * v), fill_opacity=1)
                    for v in vals]).arrange(DOWN, buff=0.04)


def icon_attention():
    dots = VGroup(*[Dot(radius=0.09, color=WHITE) for _ in range(4)]).arrange(RIGHT, buff=0.35)
    arcs = VGroup(*[ArcBetweenPoints(dots[3].get_center() + 0.1 * UP, dots[i].get_center() + 0.1 * UP,
                                     angle=PI * 0.6).set_stroke(TEAL_C, 3)
                    for i in range(3)])
    return VGroup(arcs, dots)


def icon_mlp():
    layers = [3, 4, 3]
    cols = [VGroup(*[Dot(radius=0.07, color=GREEN_C) for _ in range(k)]).arrange(DOWN, buff=0.16) for k in layers]
    net = VGroup(*cols).arrange(RIGHT, buff=0.42)
    lines = VGroup(*[Line(a.get_center(), b.get_center(), stroke_width=1.2, stroke_color=GREEN_E)
                     for c1, c2 in zip(cols[:-1], cols[1:]) for a in c1 for b in c2])
    return VGroup(lines, net)


def icon_block():
    plates = VGroup(*[RoundedRectangle(corner_radius=0.08, width=1.1, height=0.32, stroke_color=PURPLE_B,
                                       stroke_width=2, fill_color=PURPLE_E, fill_opacity=0.8) for _ in range(3)])
    for i, p in enumerate(plates):
        p.shift(i * (0.16 * UP + 0.1 * RIGHT))
    return plates


def icon_output():
    hs = [0.25, 0.9, 0.4, 0.15, 0.3]
    bars = VGroup(*[Rectangle(width=0.18, height=h, stroke_width=0,
                              fill_color=YELLOW if h == max(hs) else GREY_B, fill_opacity=1) for h in hs])
    bars.arrange(RIGHT, buff=0.06, aligned_edge=DOWN)
    return bars


class TrainingVideo(VoicedScene):
    VIDEO = "v11"

    def clear_anims(self):
        anims = []
        for m in list(self.mobjects):
            m.clear_updaters()
            if isinstance(m, ValueTracker):
                self.remove(m)
            else:
                anims.append(FadeOut(m))
        return anims

    def clear(self, run_time=0.4):
        anims = self.clear_anims()
        if anims:
            self.play(*anims, run_time=run_time)

    def construct(self):
        play_token_intro(self, TITLE, 11, TAGLINE)
        self.s1_noise()
        self.s2_data()
        self.s3_cross_entropy()
        self.s4_average()
        self.s5_landscape()
        self.s6_descent()
        self.s7_learning_rate()
        self.s8_backprop()
        self.s9_loop()
        self.s10_code()
        self.s11_curve()
        self.s12_outro()
        finish(self)

    # 1. A new model is noise ----------------------------------------------------------------
    def s1_noise(self):
        self.section(1)
        rng = np.random.default_rng(7)
        rows, cols = 6, 14
        grid = VGroup(*[Square(0.4, stroke_width=0, fill_opacity=1) for _ in range(rows * cols)])
        grid.arrange_in_grid(rows, cols, buff=0.08).move_to([0, 0.75, 0])
        for c in grid:
            c.set_fill(noise_color(rng))
        head = Text("a freshly created model", font_size=30).next_to(grid, UP, buff=0.4)
        self.play(FadeIn(grid), FadeIn(head, shift=0.2 * DOWN), run_time=0.8)

        clock = {"t": 0.0}

        def flicker(m, dt):
            clock["t"] += dt
            if clock["t"] >= 0.08:
                clock["t"] = 0.0
                for c in m:
                    c.set_fill(noise_color(rng))

        self.at("noise")
        grid.add_updater(flicker)
        sub = label("every weight: random", font_size=26).next_to(grid, DOWN, buff=0.35)
        self.at("random")
        self.play(FadeIn(sub, shift=0.15 * UP), run_time=0.5)

        out = VGroup(label("output:", font_size=26),
                     Text(GIBBERISH, font=MONO, font_size=30, color=RED_B)).arrange(RIGHT, buff=0.3)
        out.move_to([0, -2.2, 0])
        note = label("real sample from the untrained model", font_size=20).next_to(out, DOWN, buff=0.25)
        self.at("gibberish")
        self.play(FadeIn(out, shift=0.15 * UP), FadeIn(note), run_time=0.6)

        calm = label("after training: useful numbers", font_size=26, color=BLUE_B).move_to(sub)
        self.at("useful")
        grid.clear_updaters()
        self.play(*[c.animate.set_fill(calm_color(k // cols, k % cols)) for k, c in enumerate(grid)],
                  Succession(FadeOut(sub), FadeIn(calm)), run_time=0.8)
        self.end_section()

    # 2. The text is its own answer key -------------------------------------------------------
    def s2_data(self):
        self.section(2)
        self.clear()
        lines = ["… the rain had stopped by noon, and the house was quiet.",
                 "The cat sat on the mat and watched the empty street.",
                 "Nobody came. The old clock in the hall kept ticking …"]
        para = VGroup(*[Text(t, font_size=22, color=GREY_C) for t in lines]).arrange(DOWN, aligned_edge=LEFT,
                                                                                    buff=0.16)
        para.move_to([0, 2.85, 0])
        para[1][:17].set_color(WHITE)          # "The cat sat on the mat" (no glyphs for spaces)
        self.at("take")
        self.play(FadeIn(para, shift=0.15 * DOWN), run_time=0.6)

        row = token_row(SENT, buff=0.14).move_to([0, 1.2, 0])
        self.at("text")
        self.play(LaggedStart(*[FadeIn(t, shift=0.2 * DOWN) for t in row], lag_ratio=0.1), run_time=0.6)

        mat = row[5]
        cover = VGroup(RoundedRectangle(corner_radius=0.12, width=mat.width, height=mat.height, stroke_color=GREY_B,
                                        fill_color=GREY_E, fill_opacity=1),
                       Text("?", font_size=32, color=YELLOW)).move_to(mat)
        self.at("hide")
        self.play(FadeIn(cover, scale=1.2), run_time=0.4)

        model = labelled_box("model", MODEL_COLOR, 2.2, 0.9, font_size=30).move_to([-1.2, -0.75, 0])
        prefix = VGroup(*row[:5])
        a_in = arrow(prefix.get_bottom() + 0.1 * DOWN + 0.3 * RIGHT, model.get_top() + 0.08 * UP)
        self.at("ask")
        self.play(GrowArrow(a_in), FadeIn(model, shift=0.15 * DOWN), run_time=0.5)

        guess = token("rug").move_to([mat.get_x(), -0.75, 0])
        a_out = arrow(model.get_right() + 0.1 * RIGHT, guess.get_left() + 0.12 * LEFT)
        guess_lab = label("guess", font_size=22).next_to(guess, DOWN, buff=0.2)
        self.at("predict")
        self.play(GrowArrow(a_out), FadeIn(guess, shift=0.2 * RIGHT), FadeIn(guess_lab), run_time=0.6)

        vs = DoubleArrow(guess.get_top() + 0.08 * UP, cover.get_bottom() + 0.08 * DOWN, buff=0, color=GREY_B,
                         stroke_width=3, tip_length=0.18)
        vs_lab = label("compare", font_size=22).next_to(vs, RIGHT, buff=0.2)
        self.at("compare")
        self.play(GrowFromCenter(vs), FadeIn(vs_lab), run_time=0.5)

        cross = Text("✗", font_size=44, color=RED).next_to(guess, RIGHT, buff=0.25)
        truth_lab = label("truth", font_size=22, color=GREEN_B).next_to(mat, UP, buff=0.18)
        self.at("truth")
        self.play(FadeOut(cover, shift=0.3 * UP), box_to(mat, GREEN_C), FadeIn(truth_lab),
                  FadeIn(cross, scale=1.4), run_time=0.6)

        self.at("answers")
        self.play(Indicate(para[1][14:17], color=GREEN_B, scale_factor=1.3),
                  Indicate(mat, color=GREEN_B, scale_factor=1.12), run_time=0.8)

        key = Text("the text is its own answer key", font_size=32).move_to([0, -2.6, 0])
        self.at("label")
        self.play(FadeIn(key, shift=0.2 * UP), run_time=0.6)
        self.end_section()

    # 3. Cross-entropy -------------------------------------------------------------------------
    def s3_cross_entropy(self):
        self.section(3)
        self.clear()
        title = Text("cross-entropy loss", font_size=36).move_to([0, 3.25, 0])
        self.at("cross")
        self.play(FadeIn(title, shift=0.2 * DOWN), run_time=0.5)

        formula = Text("loss = −log p(correct token)", font_size=36).move_to([0, 2.25, 0])
        formula[0:5].set_color(LOSS_COLOR)      # glyphs: "loss=" | "−log" | "p(correcttoken)"
        formula[9:].set_color(BLUE_B)
        self.at("negative")
        self.play(FadeIn(formula[0:9], shift=0.15 * UP), run_time=0.5)
        self.at("probability")
        self.play(FadeIn(formula[9:], shift=0.15 * UP), run_time=0.5)

        # right: the curve loss = -log p
        O, XL, YS = np.array([1.4, -2.55, 0]), 4.4, 0.72          # origin, x length, units per loss unit

        def cp(p, l):
            return O + np.array([XL * p, YS * l, 0])

        x_ax = Line(O, cp(1.08, 0), stroke_color=GREY_B, stroke_width=2)
        y_ax = Line(O, cp(0, 5.1), stroke_color=GREY_B, stroke_width=2)
        ticks = VGroup(label("0", font_size=20).next_to(O, DOWN, buff=0.15),
                       label("1", font_size=20).next_to(cp(1, 0), DOWN, buff=0.15))
        x_lab = label("p(correct token)", font_size=22).next_to(cp(0.5, 0), DOWN, buff=0.45)
        y_lab = label("loss", font_size=22, color=LOSS_COLOR).next_to(cp(0, 5.1), UP, buff=0.12)
        ps = np.linspace(np.exp(-5), 1, 200)
        curve = VMobject(stroke_color=LOSS_COLOR, stroke_width=4).set_points_smoothly([cp(p, neg_log(p)) for p in ps])
        self.at("correct")
        self.play(Create(x_ax), Create(y_ax), FadeIn(ticks), FadeIn(x_lab), FadeIn(y_lab), run_time=0.5)
        self.play(Create(curve), run_time=0.8)

        # left: two predictions for "mat"
        ctx = Text("The cat sat on the  →  mat", font_size=28).move_to([-3.6, 1.15, 0])
        ctx[-3:].set_color(BLUE_B)
        self.at("matt")
        self.play(FadeIn(ctx, shift=0.15 * DOWN), run_time=0.5)

        def bar_row(p, text, y):
            frame = Rectangle(width=3.0, height=0.42, stroke_color=GREY_B, stroke_width=2)
            frame.move_to([-6.2 + 1.5, y, 0])
            fill = Rectangle(width=max(3.0 * p, 0.03), height=0.42, stroke_width=0, fill_color=BLUE_C,
                             fill_opacity=0.9).align_to(frame, LEFT).match_y(frame)
            head = Text(text, font_size=26).next_to(frame, UP, buff=0.18).align_to(frame, LEFT)
            return frame, fill, head

        f90, b90, h90 = bar_row(0.9, "p(mat) = 90%", -0.1)
        dot90 = Dot(cp(0.9, neg_log(0.9)), radius=0.09, color=WHITE)
        self.at("90")
        self.play(FadeIn(f90), GrowFromEdge(b90, LEFT), FadeIn(h90), FadeIn(dot90, scale=1.5), run_time=0.7)
        l90 = Text("loss ≈ 0.1", font_size=30, color=LOSS_COLOR).next_to(f90, RIGHT, buff=0.4)
        v90 = label("0.1", font_size=22, color=LOSS_COLOR).next_to(dot90, UP, buff=0.15)
        self.at("0")
        self.play(FadeIn(l90, shift=0.2 * LEFT), FadeIn(v90), run_time=0.5)

        f1, b1, h1 = bar_row(0.01, "p(mat) = 1%", -1.9)
        dot1 = Dot(cp(0.01, neg_log(0.01)), radius=0.09, color=WHITE)
        self.at("just")
        self.at("1")
        self.play(FadeIn(f1), GrowFromEdge(b1, LEFT), FadeIn(h1), FadeIn(dot1, scale=1.5), run_time=0.7)
        l1 = Text("loss ≈ 4.6", font_size=30, color=LOSS_COLOR).next_to(f1, RIGHT, buff=0.4)
        v1 = label("4.6", font_size=22, color=LOSS_COLOR).next_to(dot1, RIGHT, buff=0.2).shift(0.12 * UP)
        self.at("4")
        self.play(FadeIn(l1, shift=0.2 * LEFT), FadeIn(v1), run_time=0.5)
        self.end_section()

    # 4. Average over a batch -----------------------------------------------------------------
    def s4_average(self):
        self.section(4)
        self.clear(0.3)
        rows, nums = VGroup(), VGroup()
        for k, (words, vals) in enumerate(BATCH):
            r = token_row(words, buff=0.1, font_size=24).move_to([0, 1.9 - 1.45 * k, 0]).align_to([-6.3, 0, 0], LEFT)
            rows.add(r)
            nums.add(VGroup(*[Text(f"{v:.1f}", font_size=22, color=LOSS_COLOR).next_to(t, DOWN, buff=0.15)
                              for t, v in zip(r, vals)]))
        self.play(LaggedStart(*[FadeIn(r, shift=0.15 * DOWN) for r in rows], lag_ratio=0.2), run_time=0.4)
        self.at("loss")
        self.play(LaggedStart(*[FadeIn(n, shift=0.1 * UP) for n in nums], lag_ratio=0.2), run_time=0.6)

        head = Text("average loss", font_size=30).move_to([4.2, 1.5, 0])
        value = Text(f"{BATCH_MEAN:.2f}", font_size=64, color=LOSS_COLOR).next_to(head, DOWN, buff=0.3)
        batch_lab = label("a batch of text", font_size=24).move_to([0, 3.25, 0]).align_to(rows, LEFT)
        flying = VGroup(*[n.copy() for row in nums for n in row])
        self.at("batch")
        self.add(flying)
        self.play(FadeIn(batch_lab), FadeIn(head),
                  LaggedStart(*[n.animate.move_to(value).scale(0.8).set_opacity(0) for n in flying], lag_ratio=0.03),
                  run_time=0.8)
        self.remove(flying)
        self.play(FadeIn(value, scale=1.3), run_time=0.3)

        down = arrow(value.get_bottom() + 0.25 * DOWN + 0.9 * LEFT, value.get_bottom() + 1.15 * DOWN + 0.9 * LEFT,
                     color=GREEN_C, width=6, ratio=0.35)
        better = label("lower is better", font_size=28, color=GREEN_B).next_to(down, RIGHT, buff=0.25)
        self.at("lower")
        self.play(GrowArrow(down), FadeIn(better, shift=0.15 * LEFT), run_time=0.6)

        goal = Text("training = searching for weights that make this small", font_size=28).move_to([0, -2.85, 0])
        illus = caption("illustrative numbers")
        self.at("search")
        self.play(FadeIn(goal, shift=0.2 * UP), FadeIn(illus), run_time=0.6)
        self.at("small")
        self.play(Indicate(value, color=LOSS_COLOR, scale_factor=1.15), run_time=0.6)
        self.end_section()

    # 5. The loss landscape --------------------------------------------------------------------
    def s5_landscape(self):
        self.section(5)
        self.clear()
        land = self.land = Landscape()
        path = self.path = descent_path(LAND_START, LAND_LR, 40)
        title = self.land_title = Text("the loss landscape", font_size=34).move_to([0, 3.3, 0])
        # weight axes along the front edges, loss upward at the left corner
        e1 = land.point(0, -3) + 0.45 * (DOWN + 0.4 * LEFT)
        e2 = land.point(3, 0) + 0.45 * (DOWN + 0.8 * RIGHT)
        w1 = label("weight 1", font_size=22).rotate(np.arctan2(-np.sin(LAND_EL) * 0.53, 0.85)).move_to(e1)
        w2 = label("weight 2", font_size=22).rotate(np.arctan2(np.sin(LAND_EL) * 0.85, 0.53)).move_to(e2)
        corner = land.point(-3, -3) + 0.25 * LEFT
        up = arrow(corner, corner + 1.3 * UP, color=LOSS_COLOR, width=4)
        loss_lab = label("loss", font_size=22, color=LOSS_COLOR).next_to(up, LEFT, buff=0.15)
        illus = caption("illustrative landscape")
        self.at("landscape")
        self.play(FadeIn(land.mob), FadeIn(title, shift=0.2 * DOWN), FadeIn(w1), FadeIn(w2), GrowArrow(up),
                  FadeIn(loss_lab), FadeIn(illus), run_time=1.0)

        dot = self.dot = Dot(land.point(*path[0]), radius=0.11, color=WHITE).set_z_index(5)
        dot.set_stroke(BLACK, 2, background=True)
        self.at("point")
        self.play(FadeIn(dot, scale=2.0), run_time=0.4)
        setting = label("one setting of all the weights", font_size=24, color=WHITE).move_to([-4.2, 2.3, 0])
        pointer = Line(setting.get_bottom() + 0.1 * DOWN, dot.get_center() + 0.15 * UP + 0.05 * LEFT,
                       stroke_color=GREY_B, stroke_width=2)
        self.at("one")
        self.play(FadeIn(setting, shift=0.15 * DOWN), Create(pointer), run_time=0.6)

        ring = DashedVMobject(Ellipse(width=1.6, height=0.6, stroke_color=GREEN_B, stroke_width=4),
                              num_dashes=24).move_to(land.point(*path[-1]))
        valley = label("deep valley", font_size=26, color=GREEN_B).next_to(ring, RIGHT, buff=0.15).shift(0.5 * UP)
        valley.add_background_rectangle(opacity=0.6, buff=0.08)
        self.at("valley")
        self.play(Create(ring), FadeIn(valley, shift=0.15 * UP), run_time=0.7)

        dims = label("really: millions of dimensions", font_size=26, color=GREY_A).move_to([4.2, 2.3, 0])
        self.at("millions")
        self.play(FadeIn(dims, shift=0.15 * DOWN), run_time=0.5)

        self.at("downhill")
        trail = VMobject(stroke_color=WHITE, stroke_width=3, stroke_opacity=0.7).set_points_smoothly(
            land.surface_line(path[0], path[1]))
        self.play(MoveAlongPath(dot, trail.copy()), FadeOut(pointer), run_time=0.9)
        self.s5_extras = VGroup(setting, dims)
        self.end_section()

    # 6. Gradient descent ------------------------------------------------------------------------
    def s6_descent(self):
        self.section(6)
        land, path, dot = self.land, self.path, self.dot

        def grad_arrow(p, length=0.9):
            g = land_grad(*p)
            q = p + length * g / np.linalg.norm(g)
            return arrow(land.point(*p), land.point(*q), color=GRAD_COLOR, width=6, ratio=0.3)

        def step_arrow(p, q):
            return arrow(land.point(*p), land.point(*q), color=WHITE, width=5, ratio=0.35)

        g_arrow = grad_arrow(path[1], 1.1)
        g_lab = label("gradient ∇", font_size=28, color=GRAD_COLOR).next_to(g_arrow.get_end(), UP, buff=0.2)
        g_lab.add_background_rectangle(opacity=0.6, buff=0.08)
        self.at("gradient")
        self.play(FadeOut(self.s5_extras), GrowArrow(g_arrow), FadeIn(g_lab, shift=0.1 * UP), run_time=0.6)

        per = label("one slope per weight: which way is up, how steep", font_size=24, color=GREY_A)
        per.move_to([0, 2.75, 0])
        self.at("weight")
        self.play(FadeIn(per, shift=0.15 * DOWN), run_time=0.5)
        self.at("up")
        self.play(Indicate(g_arrow, color=GRAD_COLOR, scale_factor=1.15), run_time=0.6)
        self.at("steeply")
        self.play(g_arrow.animate(rate_func=there_and_back).scale(1.4, about_point=g_arrow.get_start()),
                  run_time=0.8)

        s_arrow = step_arrow(path[1], path[2])
        s_lab = label("step", font_size=26, color=WHITE).next_to(s_arrow.get_end(), DOWN, buff=0.2)
        s_lab.add_background_rectangle(opacity=0.6, buff=0.08)
        self.at("other")
        self.play(GrowArrow(s_arrow), FadeIn(s_lab), run_time=0.5)

        def hop(i, j, run_time):
            crumb = Dot(land.point(*path[i]), radius=0.05, color=GREY_A).set_z_index(4)
            return AnimationGroup(FadeIn(crumb, run_time=0.01), dot.animate(run_time=run_time).move_to(land.point(*path[j])))

        rule = label("every weight:  w ← w − step size × gradient", font_size=26, color=GREY_A).move_to(per)
        self.at("every")
        self.play(hop(1, 2, 0.6), FadeOut(g_arrow), FadeOut(g_lab), FadeOut(s_arrow), FadeOut(s_lab),
                  FadeOut(per), FadeIn(rule), run_time=0.6)

        self.at("against")
        for i in (2, 3):
            ga = grad_arrow(path[i], 0.8)
            self.play(GrowArrow(ga), run_time=0.25)
            self.play(hop(i, i + 1, 0.35), FadeOut(ga), run_time=0.35)

        gd = Text("gradient descent", font_size=38, color=GRAD_COLOR).move_to(self.land_title)
        self.at("thats")
        self.play(FadeTransform(self.land_title, gd), run_time=0.5)
        self.at("descent")
        idx = [4, 5, 6, 7, 8, 9]
        self.play(Succession(*[hop(a, b, 0.18) for a, b in zip(idx[:-1], idx[1:])]), run_time=0.18 * (len(idx) - 1))
        self.end_section()

    # 7. Learning rate ----------------------------------------------------------------------------
    def s7_learning_rate(self):
        self.section(7)
        self.clear()
        A = 0.42

        def bowl(cx):
            cy = -1.9
            fn = FunctionGraph(lambda x: A * x ** 2, x_range=[-2.5, 2.5], color=LOSS_COLOR, stroke_width=4)
            fn.shift([cx, cy, 0])
            return fn, (lambda x: np.array([cx + x, cy + A * x ** 2, 0]))

        left, lp = bowl(-3.4)
        right, rp = bowl(3.4)
        self.play(Create(left), Create(right), run_time=0.8)
        title = Text("learning rate = step size", font_size=36).move_to([0, 3.2, 0])
        self.at("learning")
        self.play(FadeIn(title, shift=0.2 * DOWN), run_time=0.5)

        small = label("too small", font_size=30, color=WHITE).move_to([-3.4, 1.9, 0])
        ball = Dot(lp(-2.3), radius=0.12, color=WHITE).set_z_index(3)
        self.at("small")
        self.play(FadeIn(small, shift=0.15 * DOWN), FadeIn(ball, scale=1.5), run_time=0.4)
        x = -2.3
        hops = []
        for _ in range(13):
            nx = 0.94 * x
            crumb = Dot(lp(x), radius=0.045, color=GREY_B)
            hops.append(AnimationGroup(FadeIn(crumb, run_time=0.01), ball.animate.move_to(lp(nx))))
            x = nx
        self.play(Succession(*hops), run_time=0.14 * len(hops))
        slow = label("…forever", font_size=24).next_to(small, DOWN, buff=0.2)
        self.play(FadeIn(slow), run_time=0.3)

        large = label("too large", font_size=30, color=WHITE).move_to([3.4, 1.9, 0])
        bounce = label("overshoots, bounces around", font_size=24).next_to(large, DOWN, buff=0.2)
        ball2 = Dot(rp(-1.1), radius=0.12, color=WHITE).set_z_index(3)
        self.at("large")
        self.play(FadeIn(large, shift=0.15 * DOWN), FadeIn(bounce), FadeIn(ball2, scale=1.5), run_time=0.35)
        x = -1.1
        jumps = []
        for _ in range(6):
            nx = -1.13 * x
            arc = ArcBetweenPoints(rp(x), rp(nx), angle=(-1 if nx > x else 1) * PI / 2.4)
            trail = DashedVMobject(arc.copy().set_stroke(GREY_B, 2, 0.6), num_dashes=12)
            jumps.append(AnimationGroup(MoveAlongPath(ball2, arc), Create(trail)))
            x = nx
        self.play(Succession(*jumps), run_time=0.45 * len(jumps))
        self.end_section()

    # 8. Backpropagation ---------------------------------------------------------------------------
    def s8_backprop(self):
        self.section(8)
        self.clear(0.35)
        names = ["embedding", "block", "block", "logits", "loss"]
        boxes = VGroup(*[labelled_box(n, LOSS_COLOR if n == "loss" else GREY_B, 2.2 if n == "embedding" else 1.65,
                                      0.8, font_size=24)
                         for n in names]).arrange(RIGHT, buff=0.55).move_to([0, 1.5, 0])
        links = VGroup(*[arrow(a.get_right() + 0.06 * RIGHT, b.get_left() + 0.06 * LEFT, width=3, ratio=0.4)
                         for a, b in zip(boxes[:-1], boxes[1:])])
        self.play(LaggedStart(*[FadeIn(b, shift=0.15 * DOWN) for b in boxes], lag_ratio=0.1), FadeIn(links),
                  run_time=0.6)
        q = Text("gradients for millions of weights?", font_size=34).move_to([0, 3.25, 0])
        self.at("millions")
        self.play(FadeIn(q, shift=0.15 * DOWN), run_time=0.5)

        title = Text("backpropagation", font_size=38).move_to([0, 3.25, 0])
        fwd = arrow([-5.4, 2.35, 0], [5.4, 2.35, 0], color=BLUE_B, width=5, ratio=0.04)
        fwd_lab = label("forward pass", font_size=22, color=BLUE_B).move_to([-4.4, 2.68, 0])
        chips = VGroup()
        for b in boxes[:-1]:
            chip = VGroup(*[Square(0.2, stroke_color=GREY_B, stroke_width=1.5, fill_color=GREY_D, fill_opacity=1)
                            for _ in range(3)]).arrange(RIGHT, buff=0.04)
            chips.add(chip.move_to([b.get_x(), 0.75, 0]))
        self.at("backpropagation")
        self.play(FadeTransform(q, title), GrowArrow(fwd), FadeIn(fwd_lab),
                  LaggedStart(*[FadeIn(c, shift=0.1 * DOWN) for c in chips], lag_ratio=0.25), run_time=1.2)

        chain = Text("chain rule: multiply the slopes, layer by layer", font_size=26).move_to([0, -0.95, 0])
        self.at("chain")
        self.play(FadeIn(chain, shift=0.15 * UP), run_time=0.5)

        self.at("loss")
        self.play(Indicate(boxes[-1], color=LOSS_COLOR, scale_factor=1.12), run_time=0.5)

        back = arrow([5.4, 0.2, 0], [-5.4, 0.2, 0], color=RED, width=6, ratio=0.04)
        back_lab = label("backward pass: gradients", font_size=22, color=RED).move_to([4.0, -0.15, 0])
        self.at("backwards")
        self.play(GrowArrow(back), FadeIn(back_lab),
                  LaggedStart(*[b[0].animate(rate_func=there_and_back).set_stroke(RED, 6) for b in boxes[::-1]],
                              lag_ratio=0.2), run_time=1.2)

        reuse = label("saved forward results are reused", font_size=24, color=YELLOW).move_to([0, -1.5, 0])
        self.at("reusing")
        self.play(*[sq.animate.set_fill(YELLOW_D, 1).set_stroke(YELLOW) for c in chips for sq in c],
                  FadeIn(reuse, shift=0.15 * UP), run_time=0.7)

        f_lab = label("forward", font_size=24, color=BLUE_B).move_to([-5.2, -2.35, 0])
        b_lab = label("backward", font_size=24, color=RED).move_to([-5.2, -3.05, 0])
        f_bar = Rectangle(width=2.0, height=0.4, stroke_width=0, fill_color=BLUE_C, fill_opacity=0.9)
        f_bar.move_to([-3.9 + 1.0, -2.35, 0])
        b_bar = Rectangle(width=4.0, height=0.4, stroke_width=0, fill_color=RED, fill_opacity=0.9)
        b_bar.move_to([-3.9 + 2.0, -3.05, 0])
        f_val = Text("1× compute", font_size=26).next_to(f_bar, RIGHT, buff=0.25)
        b_val = Text("≈ 2× compute", font_size=26).next_to(b_bar, RIGHT, buff=0.25)
        self.at("costs")
        self.play(FadeIn(f_lab), GrowFromEdge(f_bar, LEFT), FadeIn(f_val), run_time=0.5)
        self.at("twice")
        self.play(FadeIn(b_lab), GrowFromEdge(b_bar, LEFT), FadeIn(b_val), run_time=0.8)
        self.end_section()

    # 9. The loop, and Adam ------------------------------------------------------------------------
    def s9_loop(self):
        self.section(9)
        self.clear(0.35)
        C, R = np.array([-2.2, -0.3, 0]), 2.55
        spec = [("batch", BLUE_D, 90), ("forward", BLUE_B, 18), ("loss", LOSS_COLOR, -54),
                ("backward", RED, -126), ("step", WEIGHT_COLOR, -198)]
        nodes, arcs = VGroup(), VGroup()
        for name, color, ang in spec:
            n = labelled_box(name, GREY_D, 1.75, 0.72, font_size=26, opacity=0.0)
            n[1].set_color(GREY_B)
            nodes.add(n.move_to(C + R * np.array([np.cos(np.radians(ang)), np.sin(np.radians(ang)), 0])))
        for k in range(5):
            a0 = np.radians(spec[k][2]) - np.radians(21)
            arc = Arc(radius=R, start_angle=a0, angle=-np.radians(72 - 42), arc_center=C, stroke_color=GREY_C,
                      stroke_width=3)
            arc.add_tip(tip_length=0.18, tip_width=0.18)
            arcs.add(arc)
        self.play(FadeIn(nodes), FadeIn(arcs), run_time=0.6)

        def light(k):
            n = nodes[k]
            color = spec[k][1]
            return [n[0].animate.set_stroke(color, 3).set_fill(color, 0.25), n[1].animate.set_color(WHITE),
                    Indicate(arcs[k - 1], color=WHITE, scale_factor=1.0)]

        for k, cue in enumerate(["batch", "forward", "loss", "backward", "stepover"]):
            self.at(cue)
            self.play(*light(k), run_time=0.4)

        runner = Dot(radius=0.1, color=YELLOW).set_z_index(3)
        state = {"a": np.radians(90)}

        def orbit(m, dt):
            state["a"] -= dt * TAU / 1.6
            m.move_to(C + R * np.array([np.cos(state["a"]), np.sin(state["a"]), 0]))

        orbit(runner, 0)
        self.at("over")
        runner.add_updater(orbit)
        self.add(runner)

        big = Text("billions of tokens", font_size=32).move_to([4.3, 1.9, 0])
        self.at("billions")
        self.play(FadeIn(big, shift=0.15 * DOWN), run_time=0.5)

        adam = Text("Adam: a step size per weight", font_size=26, color=BLUE_B).move_to([4.3, 0.55, 0])
        self.at("adam")
        self.play(FadeIn(adam, shift=0.15 * DOWN), Indicate(nodes[4], color=WEIGHT_COLOR, scale_factor=1.1),
                  run_time=0.6)
        sizes = [0.9, 2.2, 0.5, 1.6, 1.1]
        bars = VGroup()
        for k, s in enumerate(sizes):
            w = label(f"weight {k + 1}", font_size=20)
            b = Rectangle(width=s, height=0.24, stroke_width=0, fill_color=WEIGHT_COLOR, fill_opacity=0.9)
            bars.add(VGroup(w, b).arrange(RIGHT, buff=0.25))
        bars.arrange(DOWN, buff=0.18, aligned_edge=LEFT).move_to([4.3, -1.35, 0])
        for g in bars:
            g[1].align_to([bars[0][1].get_left()[0], 0, 0], LEFT)
        illus = caption("illustrative step sizes")
        self.at("adapts")
        self.play(LaggedStart(*[GrowFromEdge(g[1], LEFT) for g in bars], lag_ratio=0.12),
                  FadeIn(VGroup(*[g[0] for g in bars])), FadeIn(illus), run_time=0.8)
        self.end_section()

    # 10. The loop in PyTorch ------------------------------------------------------------------------
    def s10_code(self):
        self.section(10)
        self.clear(0.35)
        # Pango spaces 20-pt mono tightly (22 pt jumps to much wider spacing), so build at 20 and scale the
        # panel up to the full 12.8 width: glyphs end up the size of ~23 pt.
        code, hl = code_panel(CODE, font_size=20)
        VGroup(code, hl).scale(12.8 / code.width).move_to([0, -0.3, 0])
        hl.match_y(code.line_numbers[2])
        head = Text("a training loop in PyTorch", font_size=32).move_to([0, 2.7, 0])
        self.play(FadeIn(code), run_time=0.5)
        self.at("pytorch")
        self.play(FadeIn(head, shift=0.15 * DOWN), run_time=0.5)
        self.at("forward")
        self.play(Create(hl), run_time=0.35)
        self.at("loss")
        self.play(highlight(hl, code, 3), run_time=0.3)
        self.at("backward")
        self.play(highlight(hl, code, 5), run_time=0.3)
        self.at("step")
        self.play(highlight(hl, code, 6), run_time=0.3)
        self.end_section()

    # 11. A real loss curve ------------------------------------------------------------------------
    def s11_curve(self):
        self.section(11)
        self.clear(0.35)
        steps, raw, smooth, val, params = load_log()
        X0, W, Y0, H, LMIN, LMAX = -5.0, 10.4, -2.45, 4.5, 1.0, 4.6

        def cp(step, loss):
            return np.array([X0 + W * step / 5000.0, Y0 + H * (loss - LMIN) / (LMAX - LMIN), 0])

        x_ax = Line(cp(0, LMIN), cp(5000, LMIN), stroke_color=GREY_B, stroke_width=2)
        y_ax = Line(cp(0, LMIN), cp(0, LMAX), stroke_color=GREY_B, stroke_width=2)
        ticks, tick_labels = VGroup(), VGroup()
        for s in range(0, 5001, 1000):
            ticks.add(Line(cp(s, LMIN), cp(s, LMIN) + 0.1 * DOWN, stroke_color=GREY_B, stroke_width=2))
            tick_labels.add(label(str(s), font_size=20).next_to(cp(s, LMIN), DOWN, buff=0.18))
        for v in (1, 2, 3, 4):
            ticks.add(Line(cp(0, v), cp(0, v) + 0.1 * LEFT, stroke_color=GREY_B, stroke_width=2))
            tick_labels.add(label(str(v), font_size=20).next_to(cp(0, v), LEFT, buff=0.18))
        x_title = label("training step", font_size=22).next_to(cp(2500, LMIN), DOWN, buff=0.6)
        y_title = label("loss", font_size=22, color=LOSS_COLOR).rotate(PI / 2).move_to([X0 - 0.95, cp(0, 2.8)[1], 0])
        head = Text("real training run", font_size=30).move_to([0, 3.35, 0])
        self.at("real")
        self.play(Create(x_ax), Create(y_ax), FadeIn(ticks), FadeIn(tick_labels), FadeIn(x_title), FadeIn(y_title),
                  FadeIn(head, shift=0.15 * DOWN), run_time=0.55)

        keep = np.r_[np.arange(0, 100), np.arange(100, 5001, 10)]
        raw_line = VMobject(stroke_color=TRAIN_COLOR, stroke_width=1.2, stroke_opacity=0.18)
        raw_line.set_points_as_corners([cp(steps[k], raw[k]) for k in range(0, 5001, 4)])
        train = VMobject(stroke_color=TRAIN_COLOR, stroke_width=4).set_points_as_corners(
            [cp(steps[k], smooth[k]) for k in keep])
        val_line = VMobject(stroke_color=VAL_COLOR, stroke_width=3).set_points_as_corners(
            [cp(s, v) for s, v in val])
        val_dots = VGroup(*[Dot(cp(s, v), radius=0.05, color=VAL_COLOR) for s, v in val])
        legend = VGroup(
            VGroup(Line(ORIGIN, 0.5 * RIGHT, stroke_color=TRAIN_COLOR, stroke_width=4),
                   label("training loss (50-step average)", font_size=20)).arrange(RIGHT, buff=0.2),
            VGroup(VGroup(Line(ORIGIN, 0.5 * RIGHT, stroke_color=VAL_COLOR, stroke_width=3),
                          Dot(0.25 * RIGHT, radius=0.05, color=VAL_COLOR)),
                   label("validation loss", font_size=20)).arrange(RIGHT, buff=0.2),
        ).arrange(DOWN, buff=0.2, aligned_edge=LEFT).move_to([3.3, 0.5, 0])
        self.at("curve")
        self.play(Create(raw_line), Create(train), Create(val_line), FadeIn(val_dots, lag_ratio=0.1),
                  FadeIn(legend), run_time=0.85)

        sub = label(f"tiny GPT from the next video · {params:,} parameters", font_size=22).move_to([0, 2.9, 0])
        self.at("tiny")
        self.play(FadeIn(sub, shift=0.15 * DOWN), run_time=0.5)

        start = Dot(cp(0, val[0][1]), radius=0.1, color=WHITE).set_z_index(3)
        start_lab = label(f"start ≈ {val[0][1]:.1f}", font_size=24, color=WHITE).next_to(start, RIGHT, buff=0.25)
        start_lab.shift(0.25 * UP)
        self.at("4")
        self.play(FadeIn(start, scale=1.6), FadeIn(start_lab, shift=0.15 * LEFT), run_time=0.5)

        guess = float(np.log(65))
        dash = DashedLine(cp(0, guess), cp(5000, guess), dash_length=0.12, stroke_color=GREY_B, stroke_width=2)
        dash_lab = label(f"random guessing over 65 characters ≈ {guess:.2f}", font_size=22).next_to(
            cp(5000, guess), DOWN, buff=0.15).align_to(cp(5000, guess), RIGHT)
        self.at("guessing")
        self.play(Create(dash), FadeIn(dash_lab), run_time=0.6)

        early = VMobject(stroke_color=LOSS_COLOR, stroke_width=7).set_points_as_corners(
            [cp(steps[k], smooth[k]) for k in keep if k <= 400])
        fast = label("fast", font_size=26, color=LOSS_COLOR).move_to(cp(750, 3.0))
        self.at("fast")
        self.play(Create(early), FadeIn(fast), run_time=0.6)

        tail = VMobject(stroke_color=GREY_A, stroke_width=7).set_points_as_corners(
            [cp(steps[k], smooth[k]) for k in keep if k >= 1000])
        slower = label("then slower", font_size=26, color=GREY_A).move_to(cp(1900, 2.15))
        self.at("slower")
        self.play(FadeOut(early), Create(tail), FadeIn(slower), run_time=0.7)

        end = Dot(cp(5000, val[-1][1]), radius=0.1, color=VAL_COLOR).set_z_index(3)
        end_lab = label(f"{val[-1][1]:.2f}", font_size=26, color=VAL_COLOR).next_to(end, RIGHT, buff=0.18)
        self.at("1")
        self.play(FadeIn(end, scale=1.6), FadeIn(end_lab), FadeOut(tail), FadeOut(fast), FadeOut(slower),
                  run_time=0.5)

        never = label("validation: text it never trained on", font_size=26, color=VAL_COLOR).move_to(cp(3500, 2.25))
        self.at("never")
        self.play(val_line.animate.set_stroke(width=7), FadeIn(never, shift=0.15 * UP), run_time=0.6)
        self.end_section()

    # 12. Outro ------------------------------------------------------------------------------------
    def s12_outro(self):
        self.section(12)
        self.clear(0.35)
        makers = [icon_tokens, icon_embeddings, icon_attention, icon_mlp, icon_block, icon_output]
        names = ["tokens", "embeddings", "attention", "MLP", "blocks", "output"]
        icons = VGroup()
        for k, (mk, nm) in enumerate(zip(makers, names)):
            ic = mk()
            if ic.width > 1.4:
                ic.scale_to_fit_width(1.4)
            ic.move_to([-5.1 + 2.04 * k, 0.4, 0])
            icons.add(VGroup(ic, label(nm, font_size=22).move_to([-5.1 + 2.04 * k, -0.75, 0])))
        self.at("every")
        self.play(LaggedStart(*[FadeIn(i, shift=0.3 * UP) for i in icons], lag_ratio=0.15), run_time=1.2)

        pics = VGroup(*[i[0] for i in icons])
        self.at("together")
        self.play(FadeOut(VGroup(*[i[1] for i in icons])),
                  pics.animate.arrange(RIGHT, buff=0.35).move_to([0, 0.2, 0]), run_time=0.7)
        frame = SurroundingRectangle(pics, buff=0.35, corner_radius=0.25, color=WHITE, stroke_width=3)
        self.play(Create(frame), run_time=0.5)

        card = next_up_card(NEXT)
        self.at("gpt")
        self.play(*self.clear_anims(), run_time=0.35)
        self.play(FadeIn(card, shift=0.2 * UP), run_time=0.6)
        self.end_section()
