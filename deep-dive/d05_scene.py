"""How LLMs Work: Deep Dive, episode 5 — RoPE: Rotating Vectors to Encode Order.

Render from the repo root:  ./render.sh deep-dive d05
Every score, frequency and difference on screen comes from code/d05_rope/rope.py (head size 64, base 1,000,000 from
Qwen2.5-0.5B's config; checked against transformers' Qwen2 rotary embedding).
"""
import math

from manim import *

from common import MODEL_COLOR, MONO, TOKEN_COLOR, code_panel, finish, highlight, next_up_card, token
from intro import play_token_intro
from d05_script import NEXT, TAGLINE, TITLE
from voiced_scene import VoicedScene

SERIES_LABEL = "HOW LLMs WORK: DEEP DIVE  ·  EPISODE 5"
SCORES = [(5, 2, "+5.924725"), (105, 102, "+5.924725"), (10_005, 10_002, "+5.924725"), (5, 4, "+8.349663"),
          (105, 104, "+8.349663")]
CLOCKS = [(0, 1.0, "6.3"), (8, 3.16e-2, "199"), (16, 1.0e-3, "6,283"), (31, 1.54e-6, "4,080,185")]
CODE = """freqs = base ** (-torch.arange(0, d, 2) / d)       # one speed per pair

def rope(x, position):
    angle = position * freqs
    x1, x2 = x[: d // 2], x[d // 2:]                    # split into pairs
    return torch.cat([x1 * cos(angle) - x2 * sin(angle),
                      x1 * sin(angle) + x2 * cos(angle)])   # rotate each pair"""


def fit_right(m, width=6.4):
    """Keep right-column text inside the frame."""
    return m.scale_to_fit_width(width) if m.width > width else m


def plane(center, radius=1.6, color=GREY_D):
    return VGroup(Circle(radius=radius, color=color, stroke_width=2).move_to(center),
                  Line(center + radius * LEFT, center + radius * RIGHT, color=color, stroke_width=1),
                  Line(center + radius * DOWN, center + radius * UP, color=color, stroke_width=1))


class RopeVideo(VoicedScene):
    VIDEO = "d05"

    def construct(self):
        play_token_intro(self, TITLE, 5, TAGLINE, label=SERIES_LABEL)
        self.hook()          # 1
        self.rotate2d()      # 2
        self.why()           # 3
        self.check()         # 4
        self.clocks()        # 5
        self.where()         # 6
        self.stretch()       # 7
        self.code()          # 8
        self.outro()         # 9

    def clear_stage(self, *keep, run_time=0.6):
        self.play(*[FadeOut(m) for m in self.mobjects if m not in keep], run_time=run_time)

    # ------------------------------------------------------------------ 1. hook
    def hook(self):
        self.section(1)
        table = VGroup(*[Text(t, font=MONO, font_size=24, color=c) for t, c in
                         [("1 → vector", GREY_B), ("…", GREY_B), ("1,024 → vector", GREY_B), ("1,025 → ?", RED)]])
        table.arrange(DOWN, aligned_edge=LEFT, buff=0.2).move_to([-3.6, 0.4, 0])
        self.at("1024")
        self.play(FadeIn(table), run_time=0.5)
        models = Text("Llama · Mistral · Qwen", font_size=28, color=GREY_A).move_to([3.0, 2.0, 0])
        self.at("llama")
        self.play(FadeIn(models), run_time=0.4)
        name = Text("RoPE: rotary position embedding", font_size=32, color=YELLOW).move_to([3.0, 1.0, 0])
        self.at("rope")
        self.play(FadeIn(name), run_time=0.5)
        c = np.array([3.0, -1.2, 0])
        pl = plane(c, 1.1)
        arrow = Arrow(c, c + 1.1 * RIGHT, buff=0, color=BLUE_C, stroke_width=6)
        self.at("rotating")
        self.play(Create(pl), GrowArrow(arrow), run_time=0.5)
        self.play(Rotate(arrow, angle=1.2, about_point=c), run_time=0.8)
        self.end_section()

    # ------------------------------------------------------------------ 2. rotation in 2-D
    def rotate2d(self):
        self.section(2)
        self.clear_stage()
        c = np.array([-2.0, 0.0, 0])
        pl = plane(c, 2.2)
        theta = 0.5
        vec = np.array([1.9, 0.6, 0])
        arrow = Arrow(c, c + vec, buff=0, color=BLUE_C, stroke_width=6)
        label = Text("position 0", font_size=28, color=BLUE_B).move_to([3.4, 1.4, 0])
        self.at("arrow")
        self.play(Create(pl), GrowArrow(arrow), FadeIn(label), run_time=0.7)
        for k, cue in [(1, "one"), (2, "two")]:
            self.at(cue)
            new_label = Text(f"position {k}: rotate {k} × θ", font_size=28, color=BLUE_B).move_to(label)
            self.play(Rotate(arrow, angle=theta, about_point=c), Transform(label, new_label), run_time=0.6)
        ghosts = VGroup(*[Arrow(c, c + rotate_vector(vec, theta * k), buff=0, color=BLUE_E, stroke_width=3)
                          for k in range(3)])
        self.add(ghosts)
        length = Text("length unchanged:\n|q| = 8.960271 at every position", font=MONO, font_size=20,
                      line_spacing=0.8).move_to([3.6, 0.0, 0])
        self.at("length")
        self.play(FadeIn(length), run_time=0.4)
        dirn = Text("only the direction carries the position", font_size=26, color=YELLOW).move_to([3.4, -1.4, 0])
        self.at("direction")
        self.play(FadeIn(dirn), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 3. why it works
    def why(self):
        self.section(3)
        self.clear_stage()
        c = np.array([-3.0, -0.2, 0])
        pl = plane(c, 2.2)
        theta = 0.35
        q0, k0 = np.array([2.0, 0.3, 0]), np.array([1.6, 1.2, 0])
        m, n = 5, 2
        q = Arrow(c, c + rotate_vector(q0, m * theta), buff=0, color=BLUE_C, stroke_width=6)
        k = Arrow(c, c + rotate_vector(k0, n * theta), buff=0, color=GOLD, stroke_width=6)
        ql = Text("query at m", font_size=22, color=BLUE_B).next_to(q.get_end(), UP, buff=0.1)
        kl = Text("key at n", font_size=22, color=GOLD).next_to(k.get_end(), RIGHT, buff=0.1)
        self.at("dot")
        self.play(Create(pl), run_time=0.4)
        dotf = fit_right(Text("score = q · k = |q| |k| cos(angle)", font=MONO, font_size=22)).move_to([3.2, 2.2, 0])
        self.at("angle")
        self.play(FadeIn(dotf), run_time=0.4)
        self.at("query")
        self.play(GrowArrow(q), FadeIn(ql), run_time=0.5)
        self.at("key")
        self.play(GrowArrow(k), FadeIn(kl), run_time=0.5)
        rel = fit_right(Text("the angle between them changes by (m − n) × θ", font_size=24, color=YELLOW)).move_to([3.2, 1.0, 0])
        self.at("minus")
        self.play(FadeIn(rel), run_time=0.4)
        shift = Text("move both by +100 positions:\nsame angle between them", font_size=24, line_spacing=0.8)
        shift.move_to([3.2, -0.4, 0])
        self.play(FadeIn(shift), Rotate(VGroup(q, k), angle=1.4, about_point=c), FadeOut(ql), FadeOut(kl),
                  run_time=1.0)
        apart = Text("score depends only on how far apart", font_size=28, color=GREEN_B).move_to([3.2, -1.8, 0])
        self.at("apart")
        self.play(FadeIn(apart), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 4. real check
    def check(self):
        self.section(4)
        self.clear_stage()
        head = Text("random query and key, head size 64: score = rope(q, m) · rope(k, n)", font_size=26).to_edge(UP, buff=0.5)
        self.at("64")
        self.play(FadeIn(head), run_time=0.4)
        rows = VGroup()
        for m, n, s in SCORES:
            rows.add(VGroup(Text(f"m = {m:>6,}", font=MONO, font_size=26), Text(f"n = {n:>6,}", font=MONO, font_size=26),
                            Text(f"m − n = {m - n}", font=MONO, font_size=26, color=GREY_B),
                            Text(s, font=MONO, font_size=26, color=YELLOW if m - n == 3 else TEAL_B)).arrange(RIGHT, buff=0.6))
        rows.arrange(DOWN, aligned_edge=LEFT, buff=0.35).move_to([0, -0.2, 0])
        for k, cue in [(0, "5"), (1, "105"), (2, "10")]:
            self.at(cue)
            self.play(FadeIn(rows[k], shift=0.2 * RIGHT), run_time=0.4)
        same = SurroundingRectangle(VGroup(*[r[3] for r in rows[:3]]), color=YELLOW, buff=0.1)
        self.at("decimal")
        self.play(Create(same), run_time=0.4)
        self.at("distance")
        self.play(FadeIn(rows[3], shift=0.2 * RIGHT), run_time=0.4)
        self.at("everywhere")
        self.play(FadeIn(rows[4], shift=0.2 * RIGHT), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 5. many frequencies
    def clocks(self):
        self.section(5)
        self.clear_stage()
        head = Text("64 numbers = 32 pairs, each rotating at its own speed (Qwen2.5)", font_size=26).to_edge(UP, buff=0.4)
        self.at("32")
        self.play(FadeIn(head), run_time=0.4)
        pos = ValueTracker(0)
        clocks = VGroup()
        for i, (pair, f, wl) in enumerate(CLOCKS):
            c = np.array([-4.8 + i * 3.2, 0.4, 0])
            face = Circle(radius=1.1, color=GREY_C, stroke_width=3).move_to(c)
            hand = always_redraw(lambda c=c, f=f: Line(c, c + 1.0 * np.array([math.cos(pos.get_value() * f + PI / 2),
                                                                                    math.sin(pos.get_value() * f + PI / 2), 0]),
                                                       color=YELLOW, stroke_width=6))
            lab = VGroup(Text(f"pair {pair}", font_size=22), Text(f"one turn every\n{wl} tokens", font_size=20,
                                                                    color=GREY_B, line_spacing=0.8)).arrange(DOWN, buff=0.1)
            lab.next_to(face, DOWN, buff=0.25)
            clocks.add(VGroup(face, hand, lab))
        counter = always_redraw(lambda: Text(f"position {int(pos.get_value()):>3}", font=MONO, font_size=28,
                                             color=BLUE_B).move_to([0, -2.9, 0]))
        self.at("clock")
        self.play(*[FadeIn(cl) for cl in clocks], FadeIn(counter), run_time=0.6)
        self.at("fastest")
        self.play(pos.animate.set_value(60), run_time=4.0, rate_func=linear)
        for cue, k in [("200", 1), ("6", 2), ("4", 3)]:
            self.at(cue)
            self.play(Indicate(clocks[k][2], color=YELLOW), run_time=0.4)
        fast = Text("fast pairs: nearby order · slow pairs: long distances", font_size=24, color=YELLOW)
        fast.next_to(head, DOWN, buff=0.25)
        self.at("nearby")
        self.play(FadeIn(fast), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 6. where it's applied
    def where(self):
        self.section(6)
        self.clear_stage()
        boxes = VGroup(*[VGroup(RoundedRectangle(corner_radius=0.15, width=2.4, height=0.9, stroke_color=c,
                                                 fill_color=c, fill_opacity=0.25), Text(t, font_size=28))
                         for t, c in [("query", BLUE_C), ("key", GOLD), ("value", GREEN_C)]]).arrange(RIGHT, buff=0.8)
        for b in boxes:
            b[1].move_to(b[0])
        boxes.move_to([0, 0.6, 0])
        rope_tags = VGroup(*[Text("RoPE ↻", font_size=24, color=YELLOW).next_to(b, UP, buff=0.2) for b in boxes[:2]])
        no = Text("not rotated", font_size=22, color=GREY_B).next_to(boxes[2], UP, buff=0.2)
        layer = Text("inside every attention layer", font_size=26).move_to([0, 2.6, 0])
        self.at("queries")
        self.play(FadeIn(layer), FadeIn(boxes[:2]), FadeIn(rope_tags), run_time=0.6)
        emb = Text("token embeddings: nothing added", font_size=26, color=GREY_A).move_to([0, -1.0, 0])
        self.at("embeddings")
        self.play(FadeIn(emb), run_time=0.4)
        self.at("values")
        self.play(FadeIn(boxes[2]), FadeIn(no), run_time=0.4)
        whom = Text("position only shapes who attends to whom", font_size=28, color=YELLOW).move_to([0, -2.2, 0])
        self.at("whom")
        self.play(FadeIn(whom), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 7. stretching
    def stretch(self):
        self.section(7)
        self.clear_stage()
        ax = NumberLine(x_range=[0, 8, 1], length=10, color=GREY_B, include_numbers=False).move_to([0, 0.8, 0])
        trained = Line(ax.n2p(0), ax.n2p(4), color=GREEN, stroke_width=10)
        tl = Text("angles seen in training", font_size=22, color=GREEN_B).next_to(trained, UP, buff=0.2)
        self.at("rescaled")
        self.play(Create(ax), Create(trained), FadeIn(tl), run_time=0.6)
        longer = Line(ax.n2p(0), ax.n2p(8), color=YELLOW, stroke_width=10).shift(0.8 * DOWN)
        ll = Text("slow the rotations down: the same angles cover twice the text", font_size=24, color=YELLOW)
        ll.next_to(longer, DOWN, buff=0.25)
        self.at("slow")
        self.play(TransformFromCopy(trained, longer), run_time=0.8)
        self.at("longer")
        self.play(FadeIn(ll), run_time=0.4)
        st = Text("how models stretch their context window (next episode)", font_size=24, color=GREY_A).to_edge(DOWN, buff=0.6)
        self.at("stretch")
        self.play(FadeIn(st), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 8. code
    def code(self):
        self.section(8)
        self.clear_stage()
        code, hl = code_panel(CODE, font_size=22)
        code.move_to([0, 0.5, 0])
        self.play(FadeIn(code, shift=0.2 * UP), run_time=0.5)
        hl.match_y(code.line_numbers[3])
        self.at("angles")
        self.play(Create(hl), run_time=0.3)
        self.at("split")
        self.play(highlight(hl, code, 4), run_time=0.4)
        self.at("rotate")
        self.play(highlight(hl, code, 5), run_time=0.4)
        check = Text("matches transformers' Qwen2 RoPE to within 2 × 10⁻⁷", font_size=24, color=GREEN_B)
        check.next_to(code, DOWN, buff=0.35)
        self.at("matches")
        self.play(FadeIn(check), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 9. outro
    def outro(self):
        self.section(9)
        self.clear_stage(run_time=0.4)
        card = next_up_card(NEXT)
        self.at("next")
        self.play(FadeIn(card, shift=0.2 * UP), run_time=0.6)
        self.end_section()
        finish(self, hold=1.5)
