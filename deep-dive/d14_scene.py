"""How LLMs Work: Deep Dive, episode 14 — LayerNorm vs RMSNorm.

Render from the repo root:  ./render.sh deep-dive d14
Every number on screen comes from code/d14_norms/norms.py (GPT-2 small's LayerNorm and Qwen2.5-0.5B's RMSNorm by hand;
an 8-layer tiny GPT trained 1,500 steps with LayerNorm, RMSNorm and no norm at two learning rates; CPU timings).
"""
from manim import *

from common import MODEL_COLOR, MONO, TOKEN_COLOR, code_panel, finish, highlight, next_up_card, token
from intro import play_token_intro
from d14_script import NEXT, TAGLINE, TITLE
from voiced_scene import VoicedScene

SERIES_LABEL = "HOW LLMs WORK: DEEP DIVE  ·  EPISODE 14"
LAYERS = [0, 3, 6, 9, 11]
STREAM = [5.2, 63.2, 82.4, 132.7, 253.3]
LN_COL, RMS_COL, NONE_COL = BLUE_C, GOLD, RED_C
CODE = """class RMSNorm(nn.Module):
    def forward(self, x):
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * rms * self.weight

# LayerNorm, for comparison:
#   (x - mean(x)) / sqrt(var(x) + eps) * weight + bias"""


def vec(values, colour=WHITE, size=24):
    return Text("[" + ", ".join(f"{v:+.3f}" for v in values) + ", …]", font=MONO, font_size=size, color=colour)


class NormsVideo(VoicedScene):
    VIDEO = "d14"

    def construct(self):
        play_token_intro(self, TITLE, 14, TAGLINE, label=SERIES_LABEL)
        self.hook()          # 1
        self.why()           # 2
        self.layernorm()     # 3
        self.rmsnorm()       # 4
        self.same()          # 5
        self.stability()     # 6
        self.why_rms()       # 7
        self.code()          # 8
        self.outro()         # 9

    def clear_stage(self, *keep, run_time=0.6):
        self.play(*[FadeOut(m) for m in self.mobjects if m not in keep], run_time=run_time)

    # ------------------------------------------------------------------ 1. hook
    def hook(self):
        self.section(1)
        stream = Arrow([-6, -0.8, 0], [6, -0.8, 0], buff=0, stroke_width=12, color=BLUE_D,
                       max_tip_length_to_length_ratio=0.03)
        blk = Rectangle(width=1.6, height=0.9, color=GREEN_C, fill_opacity=0.4).move_to([0, 1.0, 0])
        bl = Text("block", font_size=22).move_to(blk)
        nm = RoundedRectangle(width=1.4, height=0.5, corner_radius=0.1, color=YELLOW, fill_opacity=0.3).move_to([0, 0.0, 0])
        nl = Text("norm", font_size=20).move_to(nm)
        a1 = Arrow([0, -0.7, 0], nm.get_bottom(), buff=0.05, stroke_width=3)
        a2 = Arrow(nm.get_top(), blk.get_bottom(), buff=0.05, stroke_width=3)
        self.at("normalization")
        self.play(GrowArrow(stream), FadeIn(blk), FadeIn(bl), FadeIn(nm), FadeIn(nl), GrowArrow(a1), GrowArrow(a2),
                  run_time=0.8)
        g = Text("GPT-2: LayerNorm", font_size=28, color=LN_COL).move_to([-3.5, 2.5, 0])
        r = Text("Llama · Mistral · Qwen: RMSNorm", font_size=28, color=RMS_COL).move_to([3.0, 2.5, 0])
        self.at("layer")
        self.play(FadeIn(g), run_time=0.4)
        self.at("rms")
        self.play(FadeIn(r), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 2. why normalize
    def why(self):
        self.section(2)
        self.clear_stage()
        head = Text("GPT-2, average token", font_size=26).to_edge(UP, buff=0.5)
        self.play(FadeIn(head), run_time=0.3)
        scale = 4.4 / 253.3
        rows = VGroup()
        for k, (l, s) in enumerate(zip(LAYERS, STREAM)):
            x = -4.8 + 2.2 * k
            b1 = Rectangle(width=0.6, height=max(0.05, s * scale), stroke_width=0, fill_color=BLUE_D, fill_opacity=0.85)
            b1.move_to([x - 0.36, -2.4 + b1.height / 2, 0])
            b2 = Rectangle(width=0.6, height=27.7 * scale, stroke_width=0, fill_color=YELLOW, fill_opacity=0.85)
            b2.move_to([x + 0.36, -2.4 + b2.height / 2, 0])
            t1 = Text(f"{s:g}", font=MONO, font_size=16).next_to(b1, UP, buff=0.06)
            t2 = Text("27.7", font=MONO, font_size=16, color=YELLOW).next_to(b2, UP, buff=0.06)
            lab = Text(f"layer {l}", font_size=18, color=GREY_B).move_to([x, -2.7, 0])
            rows.add(VGroup(b1, t1, b2, t2, lab))
        leg = VGroup(Square(0.22, color=BLUE_D, fill_opacity=0.85), Text("the stream", font_size=18),
                     Square(0.22, color=YELLOW, fill_opacity=0.85), Text("what the block reads (normalized)",
                                                                         font_size=18)).arrange(RIGHT, buff=0.15)
        leg.move_to([0, 2.6, 0])
        self.at("grows")
        self.play(FadeIn(leg[:2]), *[FadeIn(VGroup(r[0], r[1], r[4])) for r in rows], run_time=0.8)
        self.at("normalization")
        self.play(FadeIn(leg[2:]), *[FadeIn(VGroup(r[2], r[3])) for r in rows], run_time=0.6)
        sq = Text("always √768 = 27.7", font_size=26, color=YELLOW).move_to([0, 1.7, 0])
        self.at("27")
        self.play(FadeIn(sq), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 3. LayerNorm
    def layernorm(self):
        self.section(3)
        self.clear_stage()
        head = Text("LayerNorm, by hand: “ sat” entering GPT-2 layer 6", font_size=26, color=LN_COL).to_edge(UP, buff=0.5)
        self.at("sat")
        self.play(FadeIn(head), run_time=0.4)
        x = vec([-0.316, 3.483, -0.591, 0.704]).move_to([0, 2.0, 0])
        xl = Text("768 numbers", font_size=18, color=GREY_B).next_to(x, DOWN, buff=0.1)
        self.play(FadeIn(x), FadeIn(xl), run_time=0.4)
        stats = Text("mean 0.088 · standard deviation 3.465", font=MONO, font_size=24).move_to([0, 0.9, 0])
        self.at("mean")
        self.play(FadeIn(stats), run_time=0.4)
        step = Text("(x − mean) ÷ std", font=MONO, font_size=26, color=YELLOW).move_to([0, 0.1, 0])
        nx = vec([-0.117, 0.980, -0.196, 0.178], YELLOW).move_to([0, -0.6, 0])
        self.at("subtract")
        self.play(FadeIn(step), run_time=0.4)
        self.play(FadeIn(nx), run_time=0.4)
        ss = Text("× learned scale  + learned shift   (768 + 768 numbers)", font=MONO, font_size=22,
                  color=LN_COL).move_to([0, -1.5, 0])
        self.at("learned")
        self.play(FadeIn(ss), run_time=0.4)
        ok = Text("matches GPT-2's LayerNorm to 4.8 × 10⁻⁷", font_size=24, color=GREEN_B).move_to([0, -2.5, 0])
        self.at("matches")
        self.play(FadeIn(ok), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 4. RMSNorm
    def rmsnorm(self):
        self.section(4)
        self.clear_stage()
        head = Text("RMSNorm: skip the mean", font_size=30, color=RMS_COL).to_edge(UP, buff=0.5)
        self.at("skips")
        self.play(FadeIn(head), run_time=0.4)
        f1 = Text("x ÷ √(mean(x²) + ε)", font=MONO, font_size=30).move_to([0, 1.6, 0])
        self.at("root")
        self.play(FadeIn(f1), run_time=0.4)
        f2 = Text("× learned scale", font=MONO, font_size=26, color=RMS_COL).move_to([0, 0.8, 0])
        self.at("scale")
        self.play(FadeIn(f2), run_time=0.4)
        no = Text("no centering, no shift", font_size=24, color=GREY_A).move_to([0, 0.0, 0])
        self.at("centering")
        self.play(FadeIn(no), run_time=0.3)
        q = Text("Qwen2.5-0.5B, layer 6 (896 numbers): mean 0.022, RMS 0.445", font=MONO, font_size=22).move_to([0, -1.0, 0])
        self.at("022")
        self.play(FadeIn(q), run_time=0.4)
        ok = Text("matches Qwen's RMSNorm to 4.8 × 10⁻⁷", font_size=24, color=GREEN_B).move_to([0, -2.0, 0])
        self.at("matches")
        self.play(FadeIn(ok), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ helpers for 5 and 6
    def loss_rows(self, values, y0):
        rows = VGroup()
        for k, (name, v, col) in enumerate(values):
            y = y0 - 0.9 * k
            lab = Text(name, font_size=24, color=col).move_to([-2.4, y, 0], aligned_edge=RIGHT)
            if v is None:
                b = Text("✗ diverged (NaN)", font_size=24, color=RED_B).move_to([-2.0, y, 0], aligned_edge=LEFT)
                rows.add(VGroup(lab, b))
                continue
            bar = Rectangle(width=v * 3.0, height=0.5, stroke_width=0, fill_color=col, fill_opacity=0.85)
            bar.move_to([-2.0, y, 0], aligned_edge=LEFT)
            num = Text(f"{v:.2f}", font=MONO, font_size=22).next_to(bar, RIGHT, buff=0.15)
            rows.add(VGroup(lab, bar, num))
        return rows

    # ------------------------------------------------------------------ 5. same results
    def same(self):
        self.section(5)
        self.clear_stage()
        head = Text("tiny GPT, 8 layers, 1,500 steps, learning rate 0.001", font_size=26).to_edge(UP, buff=0.6)
        self.at("three")
        self.play(FadeIn(head), run_time=0.4)
        rows = self.loss_rows([("LayerNorm", 1.69, LN_COL), ("RMSNorm", 1.69, RMS_COL), ("no norm", 1.71, NONE_COL)], 1.2)
        cap = Text("validation loss", font_size=20, color=GREY_B).move_to([0, -1.6, 0])
        self.at("69")
        self.play(FadeIn(rows[0]), FadeIn(cap), run_time=0.3)
        self.at("69")
        self.play(FadeIn(rows[1]), run_time=0.3)
        self.at("71")
        self.play(FadeIn(rows[2]), run_time=0.3)
        bd = Text("barely any difference", font_size=28, color=YELLOW).move_to([0, -2.5, 0])
        self.at("barely")
        self.play(FadeIn(bd), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 6. stability
    def stability(self):
        self.section(6)
        self.clear_stage()
        head = Text("learning rate × 10 (0.01)", font_size=28).to_edge(UP, buff=0.6)
        self.at("raise")
        self.play(FadeIn(head), run_time=0.4)
        rows = self.loss_rows([("LayerNorm", 1.81, LN_COL), ("RMSNorm", 1.81, RMS_COL), ("no norm", None, NONE_COL)], 1.2)
        self.at("81")
        self.play(FadeIn(rows[0]), FadeIn(rows[1]), run_time=0.4)
        self.at("number")
        self.play(FadeIn(rows[2], scale=1.2), run_time=0.4)
        st = Text("normalization keeps training stable", font_size=28, color=YELLOW).move_to([0, -2.2, 0])
        self.at("stable")
        self.play(FadeIn(st), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 7. why RMSNorm
    def why_rms(self):
        self.section(7)
        self.clear_stage()
        hdr = VGroup(Dot(radius=0.01, fill_opacity=0), Text("LayerNorm", font_size=26, color=LN_COL),
                     Text("RMSNorm", font_size=26, color=RMS_COL))
        rows = [("statistics", "mean + spread", "root mean square"), ("learned", "scale + shift", "scale only"),
                ("this CPU (PyTorch)", "15 ms", "46 ms")]
        xs = [-3.8, 0.4, 4.0]
        table = VGroup()
        for m, x in zip(hdr[1:], xs[1:]):
            m.move_to([x, 2.0, 0])
        for k, r in enumerate(rows):
            table.add(VGroup(*[Text(v, font_size=24, color=GREY_B if i == 0 else WHITE).move_to([x, 1.0 - 0.9 * k, 0])
                               for i, (v, x) in enumerate(zip(r, xs))]))
        self.at("simpler")
        self.play(FadeIn(hdr[1:]), FadeIn(table[0]), run_time=0.5)
        self.at("shift")
        self.play(FadeIn(table[1]), run_time=0.4)
        c = Text("centering turned out to matter little", font_size=24, color=GREY_A).move_to([0, -2.0, 0])
        self.at("centering")
        self.play(FadeIn(c), run_time=0.4)
        self.at("implementation")
        self.play(FadeIn(table[2]), run_time=0.4)
        note = Text("equally optimized: RMSNorm does slightly less work", font_size=24, color=YELLOW).move_to([0, -2.8, 0])
        self.at("equally")
        self.play(FadeIn(note), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 8. code
    def code(self):
        self.section(8)
        self.clear_stage()
        code, hl = code_panel(CODE, font_size=22)
        code.move_to([0, 0.3, 0])
        self.at("code")
        self.play(FadeIn(code, shift=0.2 * UP), run_time=0.5)
        hl.match_y(code.line_numbers[2])
        self.at("square")
        self.play(Create(hl), run_time=0.3)
        self.at("weight")
        self.play(highlight(hl, code, 3), run_time=0.3)
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
