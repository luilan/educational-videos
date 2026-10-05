"""How LLMs Work: Deep Dive, episode 22 — Mixed Precision.

Render from the repo root:  ./render.sh deep-dive d22
Every number on screen comes from code/d22_mixed_precision/mixed_precision.py (torch.finfo for the formats; rounding and
underflow examples; a 4-layer tiny GPT trained 2,000 steps three ways; GPT-2 small in float32 and bfloat16).
"""
from manim import *

from common import MODEL_COLOR, MONO, TOKEN_COLOR, code_panel, finish, highlight, next_up_card, token
from intro import play_token_intro
from d22_script import NEXT, TAGLINE, TITLE
from voiced_scene import VoicedScene

SERIES_LABEL = "HOW LLMs WORK: DEEP DIVE  ·  EPISODE 22"
F32, F16, BF16 = BLUE_C, ORANGE, GREEN_C
CODE = """with torch.autocast("cpu", dtype=torch.bfloat16):
    logits = model(x)                  # math in bfloat16
loss = F.cross_entropy(logits.float().view(-1, V), y.view(-1))
loss.backward()                        # gradients: float32
opt.step()                             # float32 master weights"""


def bits(sign, exp, man, colour, unit=0.18):
    parts = VGroup()
    for k, (n, c) in enumerate(((sign, GREY_B), (exp, colour), (man, interpolate_color(colour, BLACK, 0.45)))):
        parts.add(Rectangle(width=n * unit, height=0.4, stroke_width=1, stroke_color=BLACK, fill_color=c, fill_opacity=0.9))
    return parts.arrange(RIGHT, buff=0)


class MixedPrecisionVideo(VoicedScene):
    VIDEO = "d22"

    def construct(self):
        play_token_intro(self, TITLE, 22, TAGLINE, label=SERIES_LABEL)
        self.hook()          # 1
        self.formats()       # 2
        self.rounding()      # 3
        self.underflow()     # 4
        self.mixed()         # 5
        self.experiment()    # 6
        self.gpt2()          # 7
        self.code()          # 8
        self.outro()         # 9

    def clear_stage(self, *keep, run_time=0.6):
        self.play(*[FadeOut(m) for m in self.mobjects if m not in keep], run_time=run_time)

    # ------------------------------------------------------------------ 1. hook
    def hook(self):
        self.section(1)
        a = bits(1, 8, 23, F32).move_to([0, 1.2, 0])
        al = Text("32 bits", font=MONO, font_size=24, color=F32).next_to(a, LEFT, buff=0.3)
        b = bits(1, 8, 7, BF16).move_to([0, 0.2, 0]).align_to(a, LEFT)
        bl = Text("16 bits", font=MONO, font_size=24, color=BF16).next_to(b, LEFT, buff=0.3)
        self.at("16")
        self.play(FadeIn(a), FadeIn(al), FadeIn(b), FadeIn(bl), run_time=0.6)
        h = Text("half the memory, much faster math on modern GPUs", font_size=26).move_to([0, -1.2, 0])
        self.at("half")
        self.play(FadeIn(h), run_time=0.4)
        q = Text("but what breaks?", font_size=28, color=YELLOW).move_to([0, -2.2, 0])
        self.at("breaks")
        self.play(FadeIn(q), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 2. formats
    def formats(self):
        self.section(2)
        self.clear_stage()
        xs = [-5.0, -1.6, 1.4, 4.4]
        hdr = VGroup(*[Text(t, font_size=20, color=GREY_B).move_to([x, 2.4, 0]) for t, x in
                       zip(("format (bits)", "largest value", "step after 1.0", "smallest normal"), xs)])
        self.play(FadeIn(hdr), run_time=0.3)
        rows = [("float32", (1, 8, 23), F32, "3.4 × 10³⁸", "1.2 × 10⁻⁷", "1.2 × 10⁻³⁸", "32"),
                ("float16", (1, 5, 10), F16, "65,504", "0.00098", "6.1 × 10⁻⁵", "65"),
                ("bfloat16", (1, 8, 7), BF16, "3.4 × 10³⁸", "0.0078 (1/128)", "1.2 × 10⁻³⁸", "128")]
        for k, (name, b, col, big, step, tiny, cue) in enumerate(rows):
            y = 1.4 - 1.2 * k
            g = VGroup(Text(name, font=MONO, font_size=22, color=col), bits(*b, col, unit=0.1)).arrange(DOWN, buff=0.12)
            g.move_to([xs[0], y, 0])
            vals = VGroup(*[Text(v, font=MONO, font_size=22, color=col).move_to([x, y, 0]) for v, x in
                            zip((big, step, tiny), xs[1:])])
            self.at(cue, cue + "th")
            self.play(FadeIn(g), FadeIn(vals), run_time=0.5)
        self.end_section()

    # ------------------------------------------------------------------ 3. rounding
    def rounding(self):
        self.section(3)
        self.clear_stage()
        head = Text("a weight of 1.0 plus a small update", font_size=28).to_edge(UP, buff=0.6)
        self.at("swallow", "core", "coarse", "steps")
        self.play(FadeIn(head), run_time=0.4)
        xs = [-4.4, -1.4, 1.6, 4.6]
        hdr = VGroup(*[Text(t, font=MONO, font_size=22, color=c).move_to([x, 1.6, 0]) for t, x, c in
                       zip(("float32", "float16", "bfloat16"), xs[1:], (F32, F16, BF16))])
        rows = [("1.0 + 0.001", "1.0010000", "1.0009766", "1.0000000"), ("1.0 + 0.0001", "1.0001000", "1.0000000",
                                                                          "1.0000000")]
        tab = VGroup()
        for k, r in enumerate(rows):
            y = 0.6 - 0.9 * k
            tab.add(VGroup(*[Text(v, font=MONO, font_size=24, color=RED_B if (v == "1.0000000") else WHITE)
                             .move_to([x, y, 0]) for v, x in zip(r, xs)]))
        self.play(FadeIn(hdr), run_time=0.3)
        self.at("thousandth")
        self.play(FadeIn(tab[0][:3]), run_time=0.4)
        self.at("rounds")
        self.play(FadeIn(tab[0][3]), run_time=0.4)
        self.at("ten")
        self.play(FadeIn(tab[1]), run_time=0.4)
        n = Text("red: the update vanished", font_size=22, color=RED_B).move_to([0, -1.9, 0])
        self.play(FadeIn(n), run_time=0.3)
        self.end_section()

    # ------------------------------------------------------------------ 4. underflow
    def underflow(self):
        self.section(4)
        self.clear_stage()
        head = Text("a gradient of 10⁻⁹", font_size=30).to_edge(UP, buff=0.6)
        self.at("tiny")
        self.play(FadeIn(head), run_time=0.4)
        rows = [("stored in float16", "0.0", F16, RED_B, "0"), ("× 1024 in float16, then ÷ 1024", "9.9 × 10⁻¹⁰", F16,
                                                                    GREEN_B, "survives"),
                ("stored in bfloat16", "1.0 × 10⁻⁹", BF16, GREEN_B, "bfloat16s")]
        for k, (name, v, col, vcol, cue) in enumerate(rows):
            y = 1.2 - 1.1 * k
            lab = Text(name, font_size=24, color=col).move_to([-0.6, y, 0], aligned_edge=RIGHT)
            val = Text(v, font=MONO, font_size=26, color=vcol).move_to([0.0, y, 0], aligned_edge=LEFT)
            self.at(cue, "bfloat", "b") if k == 2 else self.at(cue)
            self.play(FadeIn(lab), FadeIn(val), run_time=0.4)
        ls = Text("loss scaling", font_size=26, color=YELLOW).move_to([4.6, 0.1, 0])
        self.play(FadeIn(ls), run_time=0.3)
        self.end_section()

    # ------------------------------------------------------------------ 5. mixed
    def mixed(self):
        self.section(5)
        self.clear_stage()
        master = RoundedRectangle(width=3.4, height=1.0, corner_radius=0.15, color=F32, fill_opacity=0.3).move_to([-3.6, 0.8, 0])
        ml = Text("master weights\nfloat32", font_size=22, line_spacing=0.85).move_to(master)
        comp = RoundedRectangle(width=3.4, height=1.0, corner_radius=0.15, color=BF16, fill_opacity=0.3).move_to([3.6, 0.8, 0])
        cl = Text("matrix math\nbfloat16", font_size=22, line_spacing=0.85).move_to(comp)
        a1 = Arrow(master.get_right(), comp.get_left(), buff=0.15)
        a1l = Text("cast down", font_size=18, color=GREY_B).next_to(a1, UP, buff=0.08)
        upd = CurvedArrow(comp.get_bottom() + 0.1 * DOWN, master.get_bottom() + 0.1 * DOWN, angle=-1.0, color=YELLOW)
        ul = Text("gradients → update in float32", font_size=20, color=YELLOW).move_to([0, -1.6, 0])
        self.at("mixes")
        self.play(FadeIn(comp), FadeIn(cl), run_time=0.5)
        self.at("master")
        self.play(FadeIn(master), FadeIn(ml), GrowArrow(a1), FadeIn(a1l), run_time=0.6)
        self.at("rounded")
        self.play(Create(upd), FadeIn(ul), run_time=0.6)
        self.end_section()

    # ------------------------------------------------------------------ 6. experiment
    def experiment(self):
        self.section(6)
        self.clear_stage()
        head = Text("tiny GPT, 4 layers, 2,000 steps: validation loss", font_size=26).to_edge(UP, buff=0.6)
        self.play(FadeIn(head), run_time=0.3)
        rows = [("all float32", 1.644, F32, "644"), ("bfloat16 math + float32 weights", 1.645, BF16, "645"),
                ("pure bfloat16 weights", 1.677, RED_C, "677")]
        out = VGroup()
        for k, (name, v, col, cue) in enumerate(rows):
            y = 1.2 - 1.0 * k
            lab = Text(name, font_size=22, color=col).move_to([-0.6, y, 0], aligned_edge=RIGHT)
            b = Rectangle(width=(v - 1.5) * 25, height=0.55, stroke_width=0, fill_color=col, fill_opacity=0.85)
            b.move_to([-0.3, y, 0], aligned_edge=LEFT)
            out.add(VGroup(lab, b, Text(f"{v:.3f}", font=MONO, font_size=22).next_to(b, RIGHT, buff=0.15)))
            self.at(cue)
            self.play(FadeIn(out[k]), run_time=0.4)
        cap = Text("bars start at 1.5", font_size=16, color=GREY_B).next_to(out, DOWN, buff=0.3)
        n = Text("likely reason: small updates rounded away", font_size=24, color=YELLOW).move_to([0, -2.4, 0])
        self.at("rounded")
        self.play(FadeIn(cap), FadeIn(n), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 7. GPT-2
    def gpt2(self):
        self.section(7)
        self.clear_stage()
        head = Text("GPT-2 small on 1,024 tokens of Shakespeare", font_size=26).to_edge(UP, buff=0.6)
        self.play(FadeIn(head), run_time=0.3)
        rows = [("float32", 475, 3.9996, F32), ("bfloat16", 237, 3.9897, BF16)]
        out = VGroup()
        for k, (name, mib, loss, col) in enumerate(rows):
            y = 1.0 - 1.1 * k
            lab = Text(name, font=MONO, font_size=24, color=col).move_to([-3.0, y, 0], aligned_edge=RIGHT)
            b = Rectangle(width=mib / 100, height=0.55, stroke_width=0, fill_color=col, fill_opacity=0.85)
            b.move_to([-2.7, y, 0], aligned_edge=LEFT)
            out.add(VGroup(lab, b, Text(f"{mib} MiB   loss {loss:.2f}", font=MONO, font_size=22).next_to(b, RIGHT, buff=0.2)))
        self.at("237")
        self.play(FadeIn(out[1]), run_time=0.4)
        self.at("475")
        self.play(FadeIn(out[0]), run_time=0.4)
        n = Text("half the memory, loss barely moves", font_size=26, color=YELLOW).move_to([0, -1.8, 0])
        self.at("barely")
        self.play(FadeIn(n), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 8. code
    def code(self):
        self.section(8)
        self.clear_stage()
        code, hl = code_panel(CODE, font_size=24)
        code.move_to([0, 0.3, 0])
        self.at("code")
        self.play(FadeIn(code, shift=0.2 * UP), run_time=0.5)
        hl.match_y(code.line_numbers[0])
        self.at("context")
        self.play(Create(hl), run_time=0.3)
        self.at("optimizer")
        self.play(highlight(hl, code, 4), run_time=0.3)
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
