"""Foundations F14 (extra) — Train vs Validation Data.

Render from the repo root:  ./render.sh foundations f14
"""
import json
import os
from types import SimpleNamespace

import numpy as np
from manim import *

from common import code_panel, finish, highlight, next_up_card, used_in_card
from f14_script import LABEL, NEXT, TAGLINE, TITLE, USED_IN
from intro import play_token_intro
from voiced_scene import VoicedScene

TRAIN_COLOR = TEAL_C
VAL_COLOR = YELLOW
TEST_COLOR = GREY_B
MODEL_COLOR = PURPLE_B

HERE = os.path.dirname(os.path.abspath(__file__))
TINY = os.path.join(HERE, "..", "how-llms-work", "tiny_gpt")

# ---------------------------------------------------------------------------- real data (tiny GPT, episode 12)
with open(os.path.join(TINY, "input.txt")) as f:
    N_CHARS = len(f.read())
SPLIT = int(0.9 * N_CHARS)
N_VAL = N_CHARS - SPLIT
assert (N_CHARS, SPLIT, N_VAL) == (1_115_394, 1_003_854, 111_540)

with open(os.path.join(TINY, "training_log.json")) as f:
    _log = json.load(f)
STEPS = np.array([s for s, _ in _log["train"]], dtype=float)
RAW = np.array([v for _, v in _log["train"]], dtype=float)
SMOOTH = np.array([RAW[max(0, k - 49):k + 1].mean() for k in range(len(RAW))])  # trailing 50-step average
VAL = np.array(_log["val"], dtype=float)
TRAIN_END, VAL_END = float(SMOOTH[-1]), float(VAL[-1, 1])
assert len(RAW) == 5001 and VAL[-1, 0] == 5000
assert f"{RAW[-100:].mean():.3f}" == "1.326" and f"{TRAIN_END:.2f}" == "1.33" and f"{VAL_END:.2f}" == "1.59"
assert 1.2 < SMOOTH[2500:].min() and SMOOTH[2500:].max() < 2.0 and 1.2 < VAL[10:, 1].min() and VAL[10:, 1].max() < 2.0


# ---------------------------------------------------------------------------- illustrative overfitting curves
def ill_train(t):
    return 0.3 + 3.2 * np.exp(-0.45 * t)


def ill_val(t):
    return 1.0 + 2.8 * np.exp(-0.5 * t) + 0.12 * t


T_MIN = 2 * np.log(1.4 / 0.12)          # where d/dt ill_val = -1.4·e^(-t/2) + 0.12 = 0
_tt = np.linspace(0, 10, 1001)
assert (ill_val(_tt) > ill_train(_tt)).all() and abs(_tt[np.argmin(ill_val(_tt))] - T_MIN) < 0.01
assert np.all(np.diff(ill_train(_tt)) < 0) and np.all(np.diff(ill_val(_tt) - ill_train(_tt))[300:] > 0)

CODE = """data = torch.tensor(encode(text))
split = int(0.9 * len(data))
train_data = data[:split]       # 1,003,854 characters
val_data = data[split:]         # 111,540 characters"""
assert f"{SPLIT:,}" in CODE and f"{N_VAL:,}" in CODE


# ---------------------------------------------------------------------------- helpers
def text(s, size=26, color=WHITE, **kw):
    return Text(s, font_size=size, color=color, **kw)


def caption(s):
    return Text(s, font_size=20, color=GREY_B).to_corner(DR, buff=0.3)


def arrow(start, end, color=GREY_B, width=4):
    return Arrow(start, end, buff=0.1, color=color, stroke_width=width, max_tip_length_to_length_ratio=0.3,
                 max_stroke_width_to_length_ratio=10)


def model_box():
    box = RoundedRectangle(corner_radius=0.15, width=1.6, height=0.9, stroke_color=MODEL_COLOR, stroke_width=3,
                           fill_color=MODEL_COLOR, fill_opacity=0.25)
    return VGroup(box, text("model", 26).move_to(box))


def lock_icon(color=GREY_B, scale=1.0):
    body = RoundedRectangle(corner_radius=0.06, width=0.46, height=0.36, stroke_color=color, stroke_width=2,
                            fill_color=color, fill_opacity=1)
    top, r, leg = body.get_top()[1], 0.14, 0.1
    shackle = VGroup(Line([-r, top, 0], [-r, top + leg, 0]), Line([r, top, 0], [r, top + leg, 0]),
                     Arc(radius=r, start_angle=0, angle=PI, arc_center=[0, top + leg, 0]))
    shackle.set_stroke(color=color, width=6)
    hole = Dot(body.get_center(), radius=0.05, color=BLACK)
    return VGroup(shackle, body, hole).scale(scale)


def ruler_icon(width=0.9, color=WHITE):
    body = Rectangle(width=width, height=0.3, stroke_color=color, stroke_width=2.5, fill_color=BLACK,
                     fill_opacity=0.7)
    ticks = VGroup()
    for k in range(9):
        x = body.get_left()[0] + width * (k + 1) / 10
        ln = 0.14 if k % 2 == 1 else 0.08
        ticks.add(Line([x, body.get_top()[1], 0], [x, body.get_top()[1] - ln, 0], stroke_color=color,
                       stroke_width=2))
    return VGroup(body, ticks)


def bar_rect(x0, x1, y, color, h=0.9, opacity=0.35):
    return Rectangle(width=x1 - x0, height=h, stroke_color=color, stroke_width=2.5, fill_color=color,
                     fill_opacity=opacity).move_to([(x0 + x1) / 2, y, 0])


def real_chart(s0, s1, l0, l1, x0, w, y0, h, xticks, yticks, yfmt):
    """Axes (Text tick labels) plus the real smoothed training curve and validation points in [s0, s1]."""
    def cp(s, v):
        return np.array([x0 + w * (s - s0) / (s1 - s0), y0 + h * (v - l0) / (l1 - l0), 0])

    frame = VGroup(Line(cp(s0, l0), cp(s1, l0), stroke_color=GREY_B, stroke_width=2),
                   Line(cp(s0, l0), cp(s0, l1), stroke_color=GREY_B, stroke_width=2))
    for s in xticks:
        frame.add(Line(cp(s, l0), cp(s, l0) + 0.1 * DOWN, stroke_color=GREY_B, stroke_width=2),
                  text(str(s), 20, GREY_B).next_to(cp(s, l0), DOWN, buff=0.18))
    for v in yticks:
        frame.add(Line(cp(s0, v), cp(s0, v) + 0.1 * LEFT, stroke_color=GREY_B, stroke_width=2),
                  text(yfmt.format(v), 20, GREY_B).next_to(cp(s0, v), LEFT, buff=0.18))
    frame.add(text("training step", 22, GREY_B).next_to(cp((s0 + s1) / 2, l0), DOWN, buff=0.6))
    frame.add(text("loss", 22, GREY_B).rotate(PI / 2).move_to([x0 - 1.0, cp(s0, (l0 + l1) / 2)[1], 0]))

    ks = [k for k in range(len(STEPS)) if s0 <= STEPS[k] <= s1 and (k < 100 or k % 5 == 0)]
    train = VMobject(stroke_color=TRAIN_COLOR, stroke_width=4).set_points_as_corners(
        [cp(STEPS[k], SMOOTH[k]) for k in ks])
    pts = [(s, v) for s, v in VAL if s0 <= s <= s1]
    val_line = VMobject(stroke_color=VAL_COLOR, stroke_width=3).set_points_as_corners([cp(s, v) for s, v in pts])
    val_dots = VGroup(*[Dot(cp(s, v), radius=0.05, color=VAL_COLOR) for s, v in pts])
    return SimpleNamespace(cp=cp, frame=frame, train=train, val_line=val_line, val_dots=val_dots)


def full_chart():
    return real_chart(0, 5000, 0.0, 4.6, -5.4, 10.0, -2.5, 5.0, range(0, 5001, 1000), range(0, 5), "{:d}")


def legend():
    return VGroup(
        VGroup(Line(ORIGIN, 0.5 * RIGHT, stroke_color=TRAIN_COLOR, stroke_width=4),
               text("training loss (50-step average)", 22)).arrange(RIGHT, buff=0.2),
        VGroup(VGroup(Line(ORIGIN, 0.5 * RIGHT, stroke_color=VAL_COLOR, stroke_width=3),
                      Dot(0.25 * RIGHT, radius=0.05, color=VAL_COLOR)),
               text("validation loss", 22)).arrange(RIGHT, buff=0.2),
    ).arrange(DOWN, buff=0.22, aligned_edge=LEFT)


class TrainValVideo(VoicedScene):
    VIDEO = "f14"

    # ------------------------------------------------------------------ stage helpers
    def on_stage(self, keep=()):
        top = list(self.mobjects)
        inner = set()
        for m in top:
            for d in m.get_family()[1:]:
                inner.add(id(d))
        keep_ids = {id(d) for k in keep for d in k.get_family()}
        return [m for m in top if id(m) not in inner and id(m) not in keep_ids]

    def clear_anims(self, keep=()):
        return [FadeOut(m) for m in self.on_stage(keep)]

    def construct(self):
        play_token_intro(self, TITLE, 0, TAGLINE, label=LABEL)
        self.s1_two_curves()
        self.s2_memorize()
        self.s3_split()
        self.s4_tiny_gpt()
        self.s5_both()
        self.s6_gap()
        self.s7_overfitting()
        self.s8_test()
        self.s9_code()
        self.s10_outro()
        self.end_section()
        finish(self)

    # 1. Two real curves ---------------------------------------------------------------------------
    def s1_two_curves(self):
        self.section(1)
        ch = full_chart()
        head = text("real training run (episode 11)", 30).move_to([0, 3.35, 0])
        leg = legend().move_to([3.0, 2.0, 0])
        self.at("episode")
        self.play(FadeIn(ch.frame), FadeIn(head, shift=0.15 * DOWN), run_time=0.6)
        self.at("curves")
        self.play(Create(ch.train), Create(ch.val_line), FadeIn(ch.val_dots, lag_ratio=0.1), FadeIn(leg),
                  run_time=1.0)
        self.at("training")
        self.play(Indicate(leg[0], color=TRAIN_COLOR, scale_factor=1.12), ch.train.animate.set_stroke(width=7),
                  run_time=0.7)
        self.at("validation")
        self.play(Indicate(leg[1], color=VAL_COLOR, scale_factor=1.12), ch.train.animate.set_stroke(width=4),
                  ch.val_line.animate.set_stroke(width=6), run_time=0.7)
        self.at("why")
        self.play(ch.val_line.animate.set_stroke(width=3), run_time=0.3)
        end = Dot(ch.cp(5000, TRAIN_END), radius=0.09, color=TRAIN_COLOR).set_z_index(3)
        self.at("low")
        self.play(ch.train.animate.set_stroke(width=7), FadeIn(end, scale=1.5), run_time=0.5)
        q = text("?", 64, RED).next_to(end, RIGHT, buff=0.3).shift(0.25 * DOWN)
        self.at("fool")
        self.play(FadeIn(q, scale=1.6), run_time=0.4)
        self.end_section()

    # 2. A memorizer -------------------------------------------------------------------------------
    def s2_memorize(self):
        self.section(2)

        def row(prompt_s, y, tag_s, color):
            p = text(prompt_s, 26)
            box = RoundedRectangle(corner_radius=0.12, width=p.width + 0.4, height=0.7, stroke_color=color,
                                   stroke_width=2.5, fill_color=color, fill_opacity=0.15)
            prompt = VGroup(box, p.move_to(box)).move_to([0, y, 0]).to_edge(LEFT, buff=0.5)
            tag = text(tag_s, 22, color).next_to(prompt, UP, buff=0.15).align_to(prompt, LEFT)
            model = model_box().move_to([-1.9, y, 0])
            a1 = arrow(prompt.get_right(), model.get_left())
            a2 = arrow(model.get_right(), model.get_right() + 0.8 * RIGHT)
            return prompt, tag, model, a1, a2

        y1, y2 = 1.3, -1.4
        p1, tag1, m1, a11, a12 = row("First Citizen:", y1, "training text", TRAIN_COLOR)
        out1 = VGroup(text("Before we proceed any further,", 24), text("hear me speak.", 24)).arrange(
            DOWN, aligned_edge=LEFT, buff=0.15).next_to(a12, RIGHT, buff=0.15)
        ok = text("✓", 36, GREEN).next_to(out1, RIGHT, buff=0.2)
        zero = text("training loss ≈ 0", 26, GREEN).next_to(out1, DOWN, buff=0.3).align_to(out1, LEFT)
        p2, tag2, m2, a21, a22 = row("To be, or not to be,", y2, "new text", GREY_A)
        out2 = text("aurst Citizen: befo, hee spe ak", 24, GREY_A).next_to(a22, RIGHT, buff=0.15)
        bad = text("✗", 36, RED).next_to(out2, RIGHT, buff=0.2)
        useless = text("useless on new text", 26, RED).next_to(out2, DOWN, buff=0.3).align_to(out2, LEFT)
        allg = VGroup(p1, tag1, m1, a11, a12, out1, ok, zero, p2, tag2, m2, a21, a22, out2, bad, useless)
        allg.shift(-allg.get_center()[0] * RIGHT)

        self.play(*self.clear_anims(), FadeIn(m1), run_time=0.5)
        self.at("memorize")
        self.play(FadeIn(tag1), FadeIn(p1), GrowArrow(a11), run_time=0.5)
        self.at("word")
        self.play(GrowArrow(a12), run_time=0.2)
        self.play(AddTextLetterByLetter(out1[0]), run_time=0.7)
        self.play(AddTextLetterByLetter(out1[1]), run_time=0.4)
        self.at("perfectly")
        self.play(FadeIn(ok, scale=1.5), FadeIn(zero, shift=0.1 * UP), run_time=0.4)
        self.at("useless")
        self.play(FadeIn(tag2), FadeIn(p2), GrowArrow(a21), FadeIn(m2), GrowArrow(a22),
                  FadeIn(caption("illustrative")), run_time=0.4)
        self.at("new")
        self.play(FadeIn(out2), FadeIn(bad, scale=1.5), FadeIn(useless, shift=0.1 * UP), run_time=0.45)
        self.end_section()

    # 3. Split the data ------------------------------------------------------------------------------
    def s3_split(self):
        self.section(3)
        L, R, Y, G = -5.5, 5.5, 0.6, 0.1
        xs = L + 0.9 * (R - L)
        whole = bar_rect(L, R, Y, GREY_B, opacity=0.25)
        data_lab = text("the data", 28, GREY_A).next_to(whole, UP, buff=0.3).align_to(whole, LEFT)
        left = bar_rect(L, xs - G, Y, GREY_B, opacity=0.25)
        right = bar_rect(xs + G, R, Y, GREY_B, opacity=0.25)
        cut = DashedLine([xs, Y + 0.8, 0], [xs, Y - 0.8, 0], color=WHITE, stroke_width=3)
        self.at("split")
        self.play(*self.clear_anims(), FadeIn(whole), FadeIn(data_lab), run_time=0.5)
        self.play(Create(cut), run_time=0.3)
        self.play(FadeOut(whole), FadeOut(cut), FadeIn(left), FadeIn(right), run_time=0.4)

        train_lab = text("training set", 32).move_to(left)
        learn = text("used for learning", 24, GREY_B).next_to(left, DOWN, buff=0.3)
        self.at("training")
        self.play(left.animate.set_fill(TRAIN_COLOR, 0.35).set_stroke(TRAIN_COLOR), FadeIn(train_lab), run_time=0.5)
        self.at("learning")
        self.play(FadeIn(learn, shift=0.1 * UP), run_time=0.4)
        self.at("slice")
        self.play(Indicate(right, color=WHITE, scale_factor=1.25), run_time=0.6)

        val_lab = text("validation set", 28, VAL_COLOR).next_to(right, UP, buff=0.3).align_to(right, RIGHT)
        self.at("validation")
        self.play(right.animate.set_fill(VAL_COLOR, 0.35).set_stroke(VAL_COLOR), FadeIn(val_lab), run_time=0.5)

        shift = np.array([0.35, -2.2, 0])
        lock = lock_icon(GREY_A, 1.1).move_to(right.get_center() + shift + 1.05 * LEFT)
        never = text("never trained on", 26, GREY_A).next_to(right.get_center() + shift, DOWN, buff=0.75)
        never.align_to(right.get_right() + shift, RIGHT)
        self.at("aside")
        self.play(right.animate.shift(shift), val_lab.animate.shift(shift), run_time=0.5)
        self.play(FadeIn(lock, scale=1.4), FadeIn(never, shift=0.1 * UP), run_time=0.4)
        self.end_section()

    # 4. The tiny GPT's split ------------------------------------------------------------------------
    def s4_tiny_gpt(self):
        self.section(4)
        L, R, Y, G = -5.5, 5.5, 0.6, 0.06
        xs = L + 0.9 * (R - L)
        head = text("tiny GPT's data: the Shakespeare text", 32).move_to([0, 2.95, 0])
        whole = bar_rect(L, R, Y, GREY_B, opacity=0.25)
        top_brace = Brace(whole, UP, buff=0.12, color=GREY_B)
        total = text(f"{N_CHARS:,} characters", 28).next_to(top_brace, UP, buff=0.12)
        self.play(*self.clear_anims(), run_time=0.4)
        self.at("gpt")
        self.play(FadeIn(head, shift=0.15 * DOWN), FadeIn(whole), GrowFromCenter(top_brace), FadeIn(total),
                  run_time=0.7)

        left = bar_rect(L, xs - G, Y, TRAIN_COLOR)
        right = bar_rect(xs + G, R, Y, VAL_COLOR)
        lb = Brace(left, DOWN, buff=0.12, color=TRAIN_COLOR)
        rb = Brace(right, DOWN, buff=0.12, color=VAL_COLOR)
        tl = VGroup(text("train (90 %)", 28, TRAIN_COLOR), text(f"{SPLIT:,}", 28)).arrange(DOWN, buff=0.14)
        tl.next_to(lb, DOWN, buff=0.15)
        vl = VGroup(text("validation (10 %)", 28, VAL_COLOR), text(f"{N_VAL:,}", 28)).arrange(
            DOWN, buff=0.14, aligned_edge=RIGHT)
        vl.next_to(rb, DOWN, buff=0.15).align_to(right, RIGHT)
        self.at("90")
        grey_slice = bar_rect(xs + G, R, Y, GREY_B, opacity=0.25)
        self.play(FadeOut(whole), FadeIn(left), FadeIn(grey_slice), GrowFromCenter(lb), FadeIn(tl, shift=0.1 * UP),
                  run_time=0.5)
        million = text("≈ a million", 24, GREY_B).next_to(tl, DOWN, buff=0.14)
        self.at("million")
        self.play(FadeIn(million), run_time=0.4)
        self.at("10")
        self.play(FadeOut(grey_slice), FadeIn(right), GrowFromCenter(rb), FadeIn(vl, shift=0.1 * UP), run_time=0.5)

        ruler = ruler_icon(0.85).move_to(right)
        measure = text("only used to measure", 24, GREY_B).next_to(vl, DOWN, buff=0.2).align_to(right, RIGHT)
        self.at("measure")
        self.play(FadeIn(ruler, shift=0.15 * DOWN), FadeIn(measure), run_time=0.4)
        self.end_section()

    # 5. Loss on both --------------------------------------------------------------------------------
    def s5_both(self):
        self.section(5)
        ch = full_chart()
        self.ch = ch
        head = text("real training run", 30).move_to([0, 3.35, 0])
        leg = legend().move_to([3.0, 2.0, 0])
        self.at("during")
        self.play(*self.clear_anims(), FadeIn(ch.frame), FadeIn(head), FadeIn(leg), run_time=0.5)
        self.play(Create(ch.train), Create(ch.val_line), FadeIn(ch.val_dots, lag_ratio=0.1), run_time=0.9)
        self.at("both")
        self.play(ch.train.animate.set_stroke(width=7), ch.val_line.animate.set_stroke(width=6), run_time=0.6)
        self.at("training")
        self.play(ch.val_line.animate.set_stroke(width=3), run_time=0.3)
        seen = text("training loss: fits what it has seen", 26, TRAIN_COLOR).move_to([0.6, -1.7, 0])
        self.at("seen")
        self.play(FadeIn(seen, shift=0.1 * UP), run_time=0.4)
        self.at("validation")
        self.play(ch.train.animate.set_stroke(width=4), ch.val_line.animate.set_stroke(width=6), run_time=0.4)
        never = text("validation loss: text it has never seen", 26, VAL_COLOR).move_to([0.6, 0.4, 0])
        self.at("never")
        self.play(FadeIn(never, shift=0.1 * DOWN), run_time=0.4)
        self.end_section()

    # 6. Zoom on the end: the gap --------------------------------------------------------------------
    def s6_gap(self):
        self.section(6)
        ch = self.ch
        zr = Rectangle(width=ch.cp(5000, 0)[0] - ch.cp(2500, 0)[0], height=ch.cp(0, 2.0)[1] - ch.cp(0, 1.2)[1],
                       stroke_color=WHITE, stroke_width=3).move_to((ch.cp(2500, 1.2) + ch.cp(5000, 2.0)) / 2)
        self.at("our")
        self.play(Create(zr), run_time=0.4)

        z = real_chart(2500, 5000, 1.2, 2.0, -5.0, 7.2, -2.4, 4.8, range(2500, 5001, 500),
                       [1.2, 1.4, 1.6, 1.8, 2.0], "{:.1f}")
        z.val_line.set_stroke(width=4)
        target = Rectangle(width=7.2, height=4.8, stroke_color=WHITE, stroke_width=3, stroke_opacity=0).move_to(
            [-5.0 + 3.6, -2.4 + 2.4, 0])
        head = text("zoom: the last 2,500 steps", 26, GREY_B).move_to([0, 3.35, 0])
        self.at("ended")
        self.play(*self.clear_anims(keep=[zr]), Transform(zr, target), FadeIn(z.frame), FadeIn(z.train),
                  FadeIn(z.val_line), FadeIn(z.val_dots), FadeIn(head), run_time=0.75)
        self.remove(zr)

        tdot = Dot(z.cp(5000, TRAIN_END), radius=0.1, color=TRAIN_COLOR).set_z_index(3)
        vdot = Dot(z.cp(5000, VAL_END), radius=0.1, color=VAL_COLOR).set_z_index(3)
        tlab = text(f"train ≈ {TRAIN_END:.2f}", 30, TRAIN_COLOR).next_to(tdot, RIGHT, buff=0.3).shift(0.55 * DOWN)
        vlab = text(f"validation ≈ {VAL_END:.2f}", 30, VAL_COLOR).next_to(vdot, RIGHT, buff=0.3).shift(0.55 * UP)
        self.at("1")
        self.play(FadeIn(tdot, scale=1.6), FadeIn(tlab, shift=0.1 * LEFT), run_time=0.45)
        self.at("6")
        self.play(FadeIn(vdot, scale=1.6), FadeIn(vlab, shift=0.1 * LEFT), run_time=0.45)

        br = Brace(Line(tdot.get_center() + 0.12 * UP, vdot.get_center() + 0.12 * DOWN), RIGHT, buff=0.25,
                   color=WHITE)
        gap = text("gap", 32).next_to(br, RIGHT, buff=0.2).shift(0.25 * UP)
        normal = text("normal ✓", 30, GREEN).next_to(gap, DOWN, buff=0.25).align_to(gap, LEFT)
        self.at("gap")
        self.play(GrowFromCenter(br), FadeIn(gap), run_time=0.4)
        self.at("normal")
        self.play(FadeIn(normal, shift=0.1 * UP), run_time=0.4)
        self.end_section()

    # 7. Overfitting (illustrative) ------------------------------------------------------------------
    def s7_overfitting(self):
        self.section(7)
        X0, W, Y0, H = -5.2, 9.6, -2.4, 4.6

        def cp(t, v):
            return np.array([X0 + W * t / 10, Y0 + H * v / 4.0, 0])

        axes = VGroup(Line(cp(0, 0), cp(10, 0), stroke_color=GREY_B, stroke_width=2),
                      Line(cp(0, 0), cp(0, 4.0), stroke_color=GREY_B, stroke_width=2),
                      text("training time →", 22, GREY_B).next_to(cp(5, 0), DOWN, buff=0.3),
                      text("loss", 22, GREY_B).rotate(PI / 2).next_to(cp(0, 2.0), LEFT, buff=0.3))
        ts = np.linspace(0, 10, 201)
        train = VMobject(stroke_color=TRAIN_COLOR, stroke_width=4).set_points_smoothly(
            [cp(t, ill_train(t)) for t in ts])
        val = VMobject(stroke_color=VAL_COLOR, stroke_width=4).set_points_smoothly([cp(t, ill_val(t)) for t in ts])
        tname = text("training", 24, TRAIN_COLOR).next_to(cp(10, ill_train(10)), RIGHT, buff=0.15)
        vname = text("validation", 24, VAL_COLOR).next_to(cp(10, ill_val(10)), RIGHT, buff=0.15)
        cap = caption("illustrative")
        self.at("but")
        self.play(*self.clear_anims(), FadeIn(axes), Create(train), Create(val), FadeIn(tname), FadeIn(vname),
                  FadeIn(cap), run_time=0.8)

        shade = Polygon(*[cp(t, ill_val(t)) for t in ts], *[cp(t, ill_train(t)) for t in ts[::-1]],
                        stroke_width=0, fill_color=GREY_B, fill_opacity=0.25).set_z_index(-1)
        self.at("growing")
        self.play(FadeIn(shade), run_time=0.6)

        vmin = Dot(cp(T_MIN, ill_val(T_MIN)), radius=0.09, color=WHITE).set_z_index(3)
        self.at("stops")
        self.play(FadeIn(vmin, scale=1.6), run_time=0.35)

        tu = ts[ts >= T_MIN]
        up = VMobject(stroke_color=RED, stroke_width=7).set_points_smoothly([cp(t, ill_val(t)) for t in tu])
        self.at("rises")
        self.play(Create(up), run_time=0.6)

        red_zone = Polygon(*[cp(t, ill_val(t)) for t in tu], *[cp(t, ill_train(t)) for t in tu[::-1]],
                           stroke_width=0, fill_color=RED, fill_opacity=0.2).set_z_index(-1)
        over = text("overfitting: memorizing", 32, RED).move_to([2.6, 1.7, 0])
        self.at("overfitting")
        self.play(FadeIn(red_zone), FadeIn(over, shift=0.1 * DOWN), run_time=0.5)

        stop = DashedLine(cp(T_MIN, 0), cp(T_MIN, 4.2), dash_length=0.12, stroke_color=GREEN, stroke_width=3)
        stop_lab = text("stop here", 28, GREEN).next_to(stop, UP, buff=0.15)
        self.at("stop")
        self.play(Create(stop), FadeIn(stop_lab, shift=0.1 * DOWN), run_time=0.5)
        self.end_section()

    # 8. A third slice: the test set -----------------------------------------------------------------
    def s8_test(self):
        self.section(8)
        L, R, Y, G = -6.0, 6.0, 0.9, 0.06
        at = lambda f: L + f * (R - L)
        train = bar_rect(L, at(0.9) - G, Y, TRAIN_COLOR)
        val = bar_rect(at(0.9) + G, R, Y, VAL_COLOR)
        tl = text("train", 28, TRAIN_COLOR).next_to(train, DOWN, buff=0.25)
        vl = text("validation", 24, VAL_COLOR).next_to(val, DOWN, buff=0.25)
        self.at("careful")
        self.play(*self.clear_anims(), FadeIn(train), FadeIn(val), FadeIn(tl), FadeIn(vl), run_time=0.5)

        train2 = bar_rect(L, at(0.76) - G, Y, TRAIN_COLOR)
        val2 = bar_rect(at(0.76) + G, at(0.88) - G, Y, VAL_COLOR)
        test = bar_rect(at(0.88) + G, R, Y, TEST_COLOR)
        lock = lock_icon(GREY_A, 1.0).next_to(test, UP, buff=0.25)
        testl = text("test", 24, TEST_COLOR).next_to(test, DOWN, buff=0.25)
        self.at("third")
        self.play(Transform(train, train2), Transform(val, val2), tl.animate.next_to(train2, DOWN, buff=0.25),
                  vl.animate.next_to(val2, DOWN, buff=0.25), FadeIn(test, shift=0.2 * LEFT), FadeIn(lock),
                  FadeIn(testl), run_time=0.6)
        self.at("test")
        self.play(Indicate(test, color=WHITE, scale_factor=1.2), run_time=0.5)

        untouched = text("test: untouched until the very end", 30, GREY_A).move_to([0, -1.0, 0])
        self.at("end")
        self.play(FadeIn(untouched, shift=0.1 * UP), run_time=0.4)
        final = text("one final, honest score", 36).move_to([0, -2.0, 0])
        self.at("final")
        self.play(FadeIn(final, shift=0.1 * UP), run_time=0.4)
        self.end_section()

    # 9. The code ------------------------------------------------------------------------------------
    def s9_code(self):
        self.section(9)
        code, hl = code_panel(CODE, 26)
        code.move_to([0, 0.2, 0])
        hl.match_y(code.line_numbers[1])
        hl.align_to(code.code_lines, LEFT).shift(0.12 * LEFT)
        self.at("code")
        self.play(*self.clear_anims(), FadeIn(code, shift=0.2 * UP), run_time=0.5)
        self.at("split")
        self.play(Create(hl), run_time=0.3)
        self.at("training")
        self.play(highlight(hl, code, 2), run_time=0.3)
        self.at("validation")
        self.play(highlight(hl, code, 3), run_time=0.3)
        self.end_section()

    # 10. Outro --------------------------------------------------------------------------------------
    def s10_outro(self):
        self.section(10)
        card = used_in_card(USED_IN).move_to([0, 0.5, 0])
        rows = card[1]
        self.at("see")
        self.play(*self.clear_anims(), FadeIn(card, shift=0.2 * UP), run_time=0.6)
        box = SurroundingRectangle(rows[0], color=YELLOW, buff=0.12)
        self.at("11")
        self.play(Create(box), Indicate(rows[0], color=YELLOW, scale_factor=1.08), run_time=0.5)
        self.at("12")
        self.play(Transform(box, SurroundingRectangle(rows[1], color=YELLOW, buff=0.12)),
                  Indicate(rows[1], color=YELLOW, scale_factor=1.08), run_time=0.5)
        done = text("Foundations complete ✓", 36, GREEN).next_to(card, DOWN, buff=0.8)
        self.at("completes")
        self.play(FadeIn(done, shift=0.15 * UP), run_time=0.5)
        nxt = next_up_card(NEXT)
        self.at("next")
        self.play(*self.clear_anims(), FadeIn(nxt, shift=0.2 * UP), run_time=0.6)
