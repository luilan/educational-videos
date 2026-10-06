"""How LLMs Work: Deep Dive, episode 43 — Superposition.

Render from the repo root:  ./render.sh deep-dive d43
Every number on screen comes from code/d43_superposition/superposition.py (toy model x' = ReLU(W^T W x + b), trained
with Adam; part 1: 5 features into 2 dimensions, importance 0.8^i; part 2: 100 equally important features into 20).
assets/d43/toy_5x2.json holds the trained 5 × 2 directions for each sparsity.
"""
import json
from pathlib import Path

from manim import *

from common import MODEL_COLOR, MONO, TOKEN_COLOR, code_panel, finish, highlight, next_up_card
from intro import play_token_intro
from d43_script import NEXT, TAGLINE, TITLE
from voiced_scene import VoicedScene

SERIES_LABEL = "HOW LLMs WORK: DEEP DIVE  ·  EPISODE 43"
TOY = json.loads((Path(__file__).parent / "assets/d43/toy_5x2.json").read_text())
FEAT_COLORS = [RED_C, GOLD, GREEN_C, BLUE_C, PURPLE_B]
BIG = [("0", 14), ("0.5", 40), ("0.8", 58), ("0.9", 95), ("0.95", 100), ("0.99", 100)]
CODE = """x = torch.rand(B, n) * (torch.rand(B, n) > S)   # sparse
x_hat = torch.relu(x @ W.T @ W + b)             # m numbers inside"""


def panel(vectors, title, scale=1.1):
    circ = Circle(radius=scale, color=GREY_D, stroke_width=1)
    axes = VGroup(Line(LEFT * scale * 1.2, RIGHT * scale * 1.2, color=GREY_D, stroke_width=1),
                  Line(DOWN * scale * 1.2, UP * scale * 1.2, color=GREY_D, stroke_width=1))
    arrows = VGroup()
    for i, (x, y) in enumerate(vectors):
        if (x * x + y * y) ** 0.5 < 0.2:
            continue
        arrows.add(Arrow(ORIGIN, [x * scale, y * scale, 0], buff=0, color=FEAT_COLORS[i], stroke_width=5,
                         max_tip_length_to_length_ratio=0.18))
    t = Text(title, font_size=20).next_to(circ, DOWN, buff=0.35)
    return VGroup(circ, axes, arrows, t)


class SuperpositionVideo(VoicedScene):
    VIDEO = "d43"

    def construct(self):
        play_token_intro(self, TITLE, 43, TAGLINE, label=SERIES_LABEL)
        self.hook()          # 1
        self.toy()           # 2
        self.dense()         # 3
        self.sparse()        # 4
        self.why()           # 5
        self.bigger()        # 6
        self.consequence()   # 7
        self.code()          # 8
        self.outro()         # 9

    def clear_stage(self, *keep, run_time=0.6):
        self.play(*[FadeOut(m) for m in self.mobjects if m not in keep], run_time=run_time)

    # ------------------------------------------------------------------ 1. hook
    def hook(self):
        self.section(1)
        n = Circle(radius=0.95, color=MODEL_COLOR, fill_opacity=0.4).move_to([0, 0.3, 0])
        nl = Text("one neuron", font_size=18).move_to(n)
        self.at("neurons")
        self.play(FadeIn(n), FadeIn(nl), run_time=0.4)
        tags = ["legal text", "DNA", "Korean", "base64", "sports scores"]
        pos = [[-3.5, 1.8, 0], [3.4, 1.8, 0], [-3.8, -1.0, 0], [3.6, -1.0, 0], [0, -2.0, 0]]
        tg = VGroup(*[Text(t, font_size=22, color=FEAT_COLORS[i]).move_to(p) for i, (t, p) in enumerate(zip(tags, pos))])
        lines = VGroup(*[Line(n.get_center(), t.get_center(), color=GREY_C, stroke_width=1.5, buff=0.4).set_z_index(-1)
                         for t in tg])
        self.at("unrelated")
        self.play(LaggedStart(*[AnimationGroup(Create(l), FadeIn(t)) for l, t in zip(lines, tg)], lag_ratio=0.2), run_time=1.0)
        s = Text("superposition: more features than dimensions", font_size=26, color=YELLOW).move_to([0, -3.0, 0])
        self.at("superposition")
        self.play(FadeIn(s), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 2. toy
    def toy(self):
        self.section(2)
        self.clear_stage()
        ins = VGroup(*[Circle(radius=0.22, color=c, fill_opacity=0.6) for c in FEAT_COLORS]).arrange(DOWN, buff=0.25).move_to([-4, 0, 0])
        mid = VGroup(*[Circle(radius=0.28, color=WHITE, fill_opacity=0.2) for _ in range(2)]).arrange(DOWN, buff=0.6).move_to([0, 0, 0])
        outs = VGroup(*[Circle(radius=0.22, color=c, fill_opacity=0.6) for c in FEAT_COLORS]).arrange(DOWN, buff=0.25).move_to([4, 0, 0])
        e1 = VGroup(*[Line(a.get_right(), b.get_left(), stroke_width=1, color=GREY_C) for a in ins for b in mid])
        e2 = VGroup(*[Line(a.get_right(), b.get_left(), stroke_width=1, color=GREY_C) for a in mid for b in outs])
        l1 = Text("5 features", font_size=20).next_to(ins, UP)
        l2 = Text("2 numbers", font_size=20).next_to(mid, UP, buff=0.6)
        l3 = Text("ReLU(Wᵀh + b)", font_size=20).next_to(outs, UP)
        self.at("five")
        self.play(FadeIn(ins), FadeIn(l1), run_time=0.4)
        self.at("squeezes")
        self.play(Create(e1), FadeIn(mid), FadeIn(l2), run_time=0.6)
        self.at("read")
        self.play(Create(e2), FadeIn(outs), FadeIn(l3), run_time=0.6)
        imp = Text("importance 1, 0.8, 0.64, 0.51, 0.41", font_size=20, color=GREY_B).move_to([0, -2.6, 0])
        self.at("matter")
        self.play(FadeIn(imp), run_time=0.3)
        self.end_section()

    # ------------------------------------------------------------------ 3. dense
    def dense(self):
        self.section(3)
        self.clear_stage()
        self.panels = VGroup(panel(TOY["0.0"], "dense (sparsity 0)"), panel(TOY["0.7"], "sparsity 0.7"),
                             panel(TOY["0.97"], "sparsity 0.97")).arrange(RIGHT, buff=1.2).move_to([0, 0.4, 0])
        self.at("active")
        self.play(FadeIn(self.panels[0][:2]), FadeIn(self.panels[0][3]), run_time=0.4)
        self.at("two")
        self.play(*[GrowArrow(a) for a in self.panels[0][2]], run_time=0.6)
        t = Text("2 features, at right angles; 3 dropped", font_size=22, color=YELLOW).move_to([0, -2.7, 0])
        self.at("drops")
        self.play(FadeIn(t), run_time=0.3)
        self.dense_note = t
        self.end_section()

    # ------------------------------------------------------------------ 4. sparse
    def sparse(self):
        self.section(4)
        self.play(FadeOut(self.dense_note), run_time=0.3)
        self.at("70")
        self.play(FadeIn(self.panels[1][:2]), FadeIn(self.panels[1][3]), *[GrowArrow(a) for a in self.panels[1][2]], run_time=0.7)
        l = Text("4 features", font_size=22, color=YELLOW).next_to(self.panels[1], DOWN, buff=0.2)
        self.play(FadeIn(l), run_time=0.3)
        self.at("97")
        self.play(FadeIn(self.panels[2][:2]), FadeIn(self.panels[2][3]), *[GrowArrow(a) for a in self.panels[2][2]], run_time=0.7)
        l2 = Text("all 5: a pentagon", font_size=22, color=YELLOW).next_to(self.panels[2], DOWN, buff=0.2)
        self.at("pentagon")
        self.play(FadeIn(l2), run_time=0.3)
        self.end_section()

    # ------------------------------------------------------------------ 5. why
    def why(self):
        self.section(5)
        self.clear_stage()
        a = Arrow(ORIGIN, [2.0, 0.0, 0], buff=0, color=RED_C, stroke_width=6)
        b = Arrow(ORIGIN, [0.62, 1.9, 0], buff=0, color=GOLD, stroke_width=6)
        g = VGroup(a, b).move_to([-3.0, 0.2, 0])
        proj = DashedLine(b.get_end(), [b.get_end()[0], a.get_start()[1], 0], color=GREY_B)
        self.at("overlap")
        self.play(GrowArrow(a), GrowArrow(b), run_time=0.5)
        self.play(Create(proj), run_time=0.4)
        t1 = Text("reading feature 1 picks up\na bit of feature 2", font_size=22, line_spacing=0.85).move_to([2.6, 1.4, 0])
        self.play(FadeIn(t1), run_time=0.3)
        t2 = Text("rarely active together →\nrare interference", font_size=22, color=GREEN_B, line_spacing=0.85).move_to([2.6, 0.0, 0])
        self.at("rarely")
        self.play(FadeIn(t2), run_time=0.4)
        t3 = Text("ReLU + negative bias\nfilter small leftovers", font_size=22, color=YELLOW, line_spacing=0.85).move_to([2.6, -1.4, 0])
        self.at("filter")
        self.play(FadeIn(t3), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 6. bigger
    def bigger(self):
        self.section(6)
        self.clear_stage()
        head = Text("100 equally important features → 20 dimensions", font_size=26).to_edge(UP, buff=0.5)
        self.at("scale")
        self.play(FadeIn(head), run_time=0.3)
        unit, x0, base = 0.032, -4.2, -2.3
        bars = VGroup()
        for k, (S, n) in enumerate(BIG):
            x = x0 + 1.7 * k
            b = Rectangle(width=0.9, height=n * unit, stroke_width=0, fill_color=GREEN_C, fill_opacity=0.85).move_to([x, base, 0], aligned_edge=DOWN)
            bars.add(VGroup(b, Text(str(n), font=MONO, font_size=20).next_to(b, UP, buff=0.08),
                            Text(S, font_size=20).move_to([x, base - 0.3, 0])))
        xl = Text("sparsity", font_size=20, color=GREY_B).move_to([x0 - 1.2, base - 0.3, 0])
        dims = DashedLine([x0 - 0.7, base + 20 * unit, 0], [x0 + 1.7 * 5 + 0.7, base + 20 * unit, 0], color=YELLOW)
        dl = Text("20 dimensions", font_size=18, color=YELLOW).next_to(dims, RIGHT, buff=0.1)
        yl = Text("features represented", font_size=20, color=GREY_B).move_to([0, 2.3, 0])
        self.play(FadeIn(xl), FadeIn(yl), Create(dims), FadeIn(dl), run_time=0.4)
        for k, cue in enumerate(("dense", "half", None, "90", "and", None)):
            if cue:
                self.at(cue)
            self.play(FadeIn(bars[k]), run_time=0.3)
        p = Text("at 0.5: pairs of features pointing in opposite directions", font_size=20, color=GREY_A).move_to([0, 1.8, 0])
        self.play(FadeIn(p), run_time=0.3)
        self.end_section()

    # ------------------------------------------------------------------ 7. consequence
    def consequence(self):
        self.section(7)
        self.clear_stage()
        t = Text("real features are sparse: a topic, a language, a grammar rule", font_size=24).move_to([0, 2.0, 0])
        self.at("real")
        self.play(FadeIn(t), run_time=0.4)
        neurons = VGroup(*[Circle(radius=0.3, color=MODEL_COLOR, fill_opacity=0.4) for _ in range(6)]).arrange(RIGHT, buff=0.5)
        neurons.move_to([0, -0.2, 0])
        feats = VGroup(*[Dot(radius=0.07, color=FEAT_COLORS[i % 5]) for i in range(18)]).arrange(RIGHT, buff=0.32).move_to([0, 1.0, 0])
        edges = VGroup(*[Line(f.get_center(), neurons[(i * 7) % 6].get_top(), stroke_width=1, color=FEAT_COLORS[i % 5])
                         for i, f in enumerate(feats)])
        self.at("pack")
        self.play(FadeIn(neurons), FadeIn(feats), Create(edges), run_time=0.8)
        p = Text("each neuron shared: polysemantic", font_size=24, color=YELLOW).move_to([0, -1.4, 0])
        self.at("polysemantic")
        self.play(FadeIn(p), run_time=0.3)
        u = Text("to read the features, we need to unpack them", font_size=24, color=GREEN_B).move_to([0, -2.4, 0])
        self.at("unpack")
        self.play(FadeIn(u), run_time=0.3)
        self.end_section()

    # ------------------------------------------------------------------ 8. code
    def code(self):
        self.section(8)
        self.clear_stage()
        code, hl = code_panel(CODE, font_size=26)
        code.move_to([0, 0.3, 0])
        self.at("code")
        self.play(FadeIn(code, shift=0.2 * UP), run_time=0.5)
        hl.match_y(code.line_numbers[1])
        self.at("multiply")
        self.play(Create(hl), run_time=0.3)
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
