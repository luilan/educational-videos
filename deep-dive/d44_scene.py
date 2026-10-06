"""How LLMs Work: Deep Dive, episode 44 — Sparse Autoencoders.

Render from the repo root:  ./render.sh deep-dive d44
Every number on screen comes from code/d44_sae/sae.py (GPT-2 small, residual stream after layer 6, 768 → 6,144
features, L1 2.0, 3,000 Adam steps on Tiny Shakespeare activations, seeded). The trade-off panel also uses the first run
(L1 coefficient 0.23: 923 active, 95.0%) and the L1 sweep (coefficient 8, 800 steps: 4.1 active, 1.5%).
"""
from manim import *

from common import MONO, code_panel, finish, highlight, next_up_card
from intro import play_token_intro
from d44_script import NEXT, TAGLINE, TITLE
from voiced_scene import VoicedScene

SERIES_LABEL = "HOW LLMs WORK: DEEP DIVE  ·  EPISODE 44"
CODE = """f = torch.relu((x - b_dec) @ W_enc.T + b_enc)
x_hat = f @ W_dec.T + b_dec
l1 = (f * W_dec.norm(dim=0)).sum(1).mean()
loss = ((x_hat - x) ** 2).sum(1).mean() + L1 * l1"""
FEATURES = [("feature 4747 · “chief”", ["Help, three o' the chief", "be well winged with our chief", "That valour is the chief"], GREEN_B),
            ("feature 2637 · “Mess(enger):”", ["you fragments!⏎⏎Mess", "true to you.⏎⏎Mess", "these tedious nights?⏎⏎Mess"], BLUE_B),
            ("feature 766 · learned words", ["odds beyond arithmetic", "my soul recorded⏎The history", "Give me a calendar",
                                             "with a piece of scripture"], GOLD)]


def chip(text, color, w=None, fs=20):
    t = Text(text, font_size=fs)
    r = RoundedRectangle(width=w or t.width + 0.4, height=t.height + 0.35, corner_radius=0.1, color=color, fill_opacity=0.2)
    return VGroup(r, t.move_to(r))


def bars(rows, x0, scale, y0, dy, fmt):
    g = VGroup()
    for k, (lab, v, col) in enumerate(rows):
        b = Rectangle(width=max(v * scale, 0.02), height=0.5, stroke_width=0, fill_color=col, fill_opacity=0.85)
        b.move_to([x0, y0 - dy * k, 0], aligned_edge=LEFT)
        g.add(VGroup(Text(lab, font_size=22).next_to(b, LEFT, buff=0.25).align_to([x0 - 0.25, 0, 0], RIGHT), b,
                     Text(fmt(v), font=MONO, font_size=22).next_to(b, RIGHT, buff=0.15)))
    return g


class SAEVideo(VoicedScene):
    VIDEO = "d44"

    def construct(self):
        play_token_intro(self, TITLE, 44, TAGLINE, label=SERIES_LABEL)
        self.hook()          # 1
        self.architecture()  # 2
        self.tradeoff()      # 3
        self.reconstruction()  # 4
        self.splice()        # 5
        self.features()      # 6
        self.neurons()       # 7
        self.code()          # 8
        self.outro()         # 9

    def clear_stage(self, *keep, run_time=0.6):
        self.play(*[FadeOut(m) for m in self.mobjects if m not in keep], run_time=run_time)

    # ------------------------------------------------------------------ 1. hook
    def hook(self):
        self.section(1)
        n = Circle(0.6, color=GREY_B, fill_opacity=0.15).move_to([-3.5, 0, 0])
        nl = Text("one neuron (illustration)", font_size=20, color=GREY_B).next_to(n, DOWN, buff=0.2)
        feats = VGroup(*[chip(t, c, w=3.0) for t, c in (("speaker names", BLUE_B), ("“thou”", GREEN_B), ("“And” at line start", GOLD),
                                                         ("“chief”", RED_B))]).arrange(DOWN, buff=0.25).move_to([3.2, 0, 0])
        arrows = VGroup(*[Arrow(n.get_right(), f.get_left(), buff=0.1, stroke_width=2, color=GREY_B) for f in feats])
        self.at("neurons")
        self.play(FadeIn(n), FadeIn(nl), run_time=0.4)
        self.play(GrowFromPoint(arrows, n.get_right()), FadeIn(feats, lag_ratio=0.1), run_time=0.8)
        t = Text("a sparse autoencoder: unpack them into separate, readable features", font_size=24, color=YELLOW).move_to([0, -2.6, 0])
        self.at("sparse")
        self.play(FadeIn(t), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 2. architecture
    def architecture(self):
        self.section(2)
        self.clear_stage()

        def block(label, sub, h, color, x):
            r = Rectangle(width=1.3, height=h, color=color, fill_opacity=0.2).move_to([x, 0.6, 0])
            return VGroup(r, Text(label, font=MONO, font_size=24).move_to(r), Text(sub, font_size=18, color=GREY_A).next_to(r, DOWN, buff=0.2))

        x_in = block("768", "residual stream\nafter layer 6", 1.6, BLUE_B, -4.5)
        f = block("6,144", "features\n(8× more, mostly 0)", 3.6, GOLD, 0)
        x_out = block("768", "rebuilt", 1.6, BLUE_B, 4.5)
        e = Arrow(x_in[0].get_right(), f[0].get_left(), buff=0.1, color=GREY_B)
        el = Text("encode + ReLU", font_size=18).next_to(e, UP, buff=0.1)
        d = Arrow(f[0].get_right(), x_out[0].get_left(), buff=0.1, color=GREY_B)
        dl = Text("decode", font_size=18).next_to(d, UP, buff=0.1)
        self.at("residual")
        self.play(FadeIn(x_in), run_time=0.4)
        self.at("encode")
        self.play(GrowArrow(e), FadeIn(el), FadeIn(f), run_time=0.6)
        self.at("decode")
        self.play(GrowArrow(d), FadeIn(dl), FadeIn(x_out), run_time=0.6)
        l = Text("loss = |x − x̂|²  +  L1 · Σ features", font=MONO, font_size=24, color=YELLOW).move_to([0, -2.6, 0])
        self.at("loss")
        self.play(FadeIn(l), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 3. trade-off
    def tradeoff(self):
        self.section(3)
        self.clear_stage()
        h = Text("the penalty sets the trade-off", font_size=26).to_edge(UP, buff=0.6)
        self.at("penalty")
        self.play(FadeIn(h), run_time=0.3)
        cols = [("too weak", "923 active / token", "95% variance explained", "nothing readable", RED_B),
                ("too strong", "4 active / token", "2% variance explained", "gives up", RED_B),
                ("in between", "30 active / token", "80% variance explained", "sparse and faithful", GREEN_B)]
        panels = VGroup()
        for k, (a, b, c, d, col) in enumerate(cols):
            g = VGroup(Text(a, font_size=26, color=col), Text(b, font=MONO, font_size=20), Text(c, font=MONO, font_size=18, color=GREY_A),
                       Text(d, font_size=20, color=col)).arrange(DOWN, buff=0.25)
            box = SurroundingRectangle(g, buff=0.3, color=col, corner_radius=0.1)
            panels.add(VGroup(box, g))
        panels.arrange(RIGHT, buff=0.4).move_to([0, -0.1, 0])
        for p in panels:
            p[0].stretch_to_fit_height(panels[0][0].height)
        self.at("900")
        self.play(FadeIn(panels[0]), run_time=0.4)
        self.at("strong")
        self.play(FadeIn(panels[1]), run_time=0.4)
        self.at("between")
        self.play(FadeIn(panels[2]), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 4. reconstruction
    def reconstruction(self):
        self.section(4)
        self.clear_stage()
        h = Text("GPT-2 activations over Tiny Shakespeare · 322,580 tokens", font_size=24).to_edge(UP, buff=0.6)
        self.at("trained")
        self.play(FadeIn(h), run_time=0.3)
        tiles = VGroup(*[VGroup(Text(v, font=MONO, font_size=40, color=c), Text(l, font_size=20, color=GREY_A)).arrange(DOWN, buff=0.25)
                         for v, l, c in (("80%", "variance explained\n(held-out text)", GREEN_B),
                                         ("30 / 6,144", "features active per token", GOLD),
                                         ("0", "dead features", BLUE_B))]).arrange(RIGHT, buff=1.2, aligned_edge=UP).move_to([0, 0, 0])
        for k, cue in enumerate(("80", "30", "dead")):
            self.at(cue)
            self.play(FadeIn(tiles[k]), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 5. splice
    def splice(self):
        self.section(5)
        self.clear_stage()
        h = Text("put the reconstruction back into GPT-2 at layer 6", font_size=26).to_edge(UP, buff=0.6)
        self.at("real")
        self.play(FadeIn(h), run_time=0.3)
        rows = [("GPT-2", 4.686, GREY_B), ("with the SAE", 4.903, GREEN_C), ("layer → its average", 7.861, RED_C)]
        b = bars(rows, -0.6, 0.75, 1.2, 0.9, lambda v: f"{v:.2f}")
        lab = Text("next-token loss on held-out text", font_size=18, color=GREY_B).move_to([2.3, 1.9, 0])
        self.at("rises")
        self.play(FadeIn(lab), FadeIn(b[:2]), run_time=0.5)
        self.at("average")
        self.play(FadeIn(b[2]), run_time=0.4)
        t = Text("93% of the layer's contribution kept", font_size=26, color=YELLOW).move_to([0, -2.0, 0])
        self.at("93")
        self.play(FadeIn(t), run_time=0.3)
        self.end_section()

    # ------------------------------------------------------------------ 6. features
    def features(self):
        self.section(6)
        self.clear_stage()
        h = Text("where random features fire most", font_size=26).to_edge(UP, buff=0.5)
        self.at("random")
        self.play(FadeIn(h), run_time=0.3)
        groups = VGroup()
        for k, (title, ctx, col) in enumerate(FEATURES):
            t = Text(title, font_size=22, color=col)
            c = VGroup(*[Text(s.replace("⏎", " ⏎ "), font=MONO, font_size=16, color=GREY_A) for s in ctx]).arrange(DOWN, buff=0.08, aligned_edge=LEFT)
            groups.add(VGroup(t, c).arrange(DOWN, buff=0.15, aligned_edge=LEFT))
        groups.arrange(DOWN, buff=0.35, aligned_edge=LEFT).move_to([0, -0.2, 0])
        for g, cue in zip(groups, ("chief", "mess", "learned")):
            self.at(cue)
            self.play(FadeIn(g), run_time=0.4)
        n = Text("not every feature is this clean", font_size=22, color=GREY_B).to_edge(DOWN, buff=0.4)
        self.at("clean")
        self.play(FadeIn(n), run_time=0.3)
        self.end_section()

    # ------------------------------------------------------------------ 7. neurons
    def neurons(self):
        self.section(7)
        self.clear_stage()
        h = Text("top-20 activations: how often the same token?", font_size=26).to_edge(UP, buff=0.6)
        self.at("compare")
        self.play(FadeIn(h), run_time=0.3)
        l1 = Text("share of the top 20 that are one token (average)", font_size=20, color=GREY_B).move_to([0, 1.8, 0])
        a = bars([("SAE features", 0.49, GREEN_C), ("raw dimensions", 0.33, GREY_B)], -0.8, 6, 1.1, 0.75, lambda v: f"{v * 100:.0f}%")
        l2 = Text("units whose top 20 are all one token", font_size=20, color=GREY_B).move_to([0, -0.6, 0])
        b = bars([("SAE features", 0.14, GREEN_C), ("raw dimensions", 0.01, GREY_B)], -0.8, 6, -1.3, 0.75, lambda v: f"{v * 100:.0f}%")
        self.at("half")
        self.play(FadeIn(l1), FadeIn(a[0]), run_time=0.4)
        self.at("third")
        self.play(FadeIn(a[1]), run_time=0.3)
        self.at("14")
        self.play(FadeIn(l2), FadeIn(b[0]), run_time=0.4)
        self.at("1")
        self.play(FadeIn(b[1]), run_time=0.3)
        self.end_section()

    # ------------------------------------------------------------------ 8. code
    def code(self):
        self.section(8)
        self.clear_stage()
        code, hl = code_panel(CODE, font_size=24)
        code.move_to([0, 0.3, 0])
        self.play(FadeIn(code, shift=0.2 * UP), run_time=0.5)
        hl.match_y(code.line_numbers[0])
        self.at("relu")
        self.play(Create(hl), run_time=0.3)
        self.at("matrix")
        self.play(highlight(hl, code, 1), run_time=0.3)
        self.at("l1")
        self.play(highlight(hl, code, 2), run_time=0.3)
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
