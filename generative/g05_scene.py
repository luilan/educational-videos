"""How Generative Models Work: Images and Sound, episode 5 — Diffusion I: Adding Noise.

Render from the repo root:  ./render.sh generative g05
Numbers and pictures come from code/g05_adding_noise (forward_noise.py → results.json and out/*.png).
"""
import json

import numpy as np
from manim import *

from common import MONO, code_panel, finish, highlight, next_up_card
from gen_common import frame, tile
from intro import play_token_intro
from g05_script import NEXT, TAGLINE, TITLE
from voiced_scene import VoicedScene

SERIES_LABEL = "HOW GENERATIVE MODELS WORK  ·  EPISODE 5"
CODE_DIR = "code/g05_adding_noise"
OUT = f"{CODE_DIR}/out"
R = json.load(open(f"{CODE_DIR}/results.json"))
SHOW = [0, 50, 100, 200, 300, 500, 700, 1000]
LIN_C, COS_C, NOISE_C, SIG_C = BLUE_C, ORANGE, GREY_B, YELLOW
CODE = """beta = torch.linspace(1e-4, 0.02, 1000)
alpha_bar = torch.cumprod(1 - beta, dim=0)
x_t = alpha_bar[t].sqrt() * x0 + (1 - alpha_bar[t]).sqrt() * torch.randn_like(x0)"""


def row_of(path, height=1.3, buff=0.12):
    return Group(*[tile(path, i, height) for i in range(len(SHOW))]).arrange(RIGHT, buff=buff)


class AddingNoiseVideo(VoicedScene):
    VIDEO = "g05"

    def construct(self):
        play_token_intro(self, TITLE, 5, TAGLINE, label=SERIES_LABEL)
        self.hook()        # 1
        self.one_step()    # 2
        self.thousand()    # 3
        self.jump()        # 4
        self.snr()         # 5
        self.schedules()   # 6
        self.why()         # 7
        self.code()        # 8
        self.outro()       # 9

    def clear_stage(self, *keep, run_time=0.6):
        self.play(*[FadeOut(m) for m in self.mobjects if m not in keep], run_time=run_time)

    # ------------------------------------------------------------------ 1. hook
    def hook(self):
        self.section(1)
        vae = tile("code/g03_variational_autoencoders/out/random_vae.png", 1, 1.8).move_to([-4.2, 1.2, 0])
        lv = Text("VAEs: blur", font_size=22, color=GREY_B).next_to(vae, DOWN, buff=0.2)
        self.at("vaes")
        self.play(FadeIn(vae), FadeIn(frame(vae)), FadeIn(lv), run_time=0.5)
        lg = Text("GANs: hard to train", font_size=22, color=GREY_B).move_to([-4.2, -1.3, 0])
        self.at("gans")
        self.play(FadeIn(lg), run_time=0.4)
        noise = tile(f"{OUT}/noising_linear.png", 7, 1.8).move_to([0.8, 0.3, 0])
        clean = tile(f"{OUT}/noising_linear.png", 0, 1.8).move_to([4.6, 0.3, 0])
        arr = Arrow(noise.get_right(), clean.get_left(), buff=0.2, color=ORANGE)
        al = Text("undo noise", font_size=22, color=ORANGE).next_to(arr, UP, buff=0.1)
        self.at("undo")
        self.play(FadeIn(noise), FadeIn(frame(noise)), GrowArrow(arr), FadeIn(al), FadeIn(clean), FadeIn(frame(clean)), run_time=0.8)
        first = Text("first, the easy half: adding it", font_size=24, color=YELLOW).move_to([2.7, -1.8, 0])
        self.at("easy")
        self.play(FadeIn(first), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 2. one step
    def one_step(self):
        self.section(2)
        self.clear_stage()
        rng = VGroup(NumberLine(x_range=[-1, 1, 1], length=4, include_numbers=False, color=GREY_B),)
        nums = VGroup(*[Text(t, font=MONO, font_size=18).next_to(rng[0].n2p(v), DOWN, buff=0.15) for t, v in [("−1", -1), ("0", 0), ("+1", 1)]])
        lr = Text("pixel values", font_size=20, color=GREY_B).next_to(rng, UP, buff=0.15)
        g = VGroup(lr, rng, nums).move_to([0, 2.4, 0])
        self.at("pixels")
        self.play(FadeIn(g), run_time=0.5)
        f = VGroup(Text("xₜ = ", font=MONO, font_size=32), Text("√(1 − βₜ) · xₜ₋₁", font=MONO, font_size=32, color=SIG_C),
                   Text("  +  ", font=MONO, font_size=32), Text("√βₜ · ε", font=MONO, font_size=32, color=NOISE_C)).arrange(RIGHT, buff=0.1).move_to([0, 0.7, 0])
        shrink = Text("yellow: shrink the picture a tiny bit", font_size=20, color=SIG_C)
        add = Text("grey: add a tiny bit of random noise ε (standard normal)", font_size=20, color=NOISE_C)
        VGroup(shrink, add).arrange(DOWN, aligned_edge=LEFT, buff=0.15).next_to(f, DOWN, buff=0.3)
        self.at("shrinks")
        self.play(FadeIn(f), FadeIn(shrink), run_time=0.6)
        self.at("adds")
        self.play(FadeIn(add), run_time=0.4)
        ax = Axes(x_range=[0, 1000, 250], y_range=[0, 0.02, 0.01], x_length=5.5, y_length=1.6, tips=False,
                  axis_config={"color": GREY_C, "include_numbers": False}).move_to([0, -2.2, 0])
        line = ax.plot(lambda t: 1e-4 + (0.02 - 1e-4) * t / 1000, color=COS_C)
        l0 = Text("β = 0.0001", font=MONO, font_size=16, color=COS_C).next_to(ax.c2p(0, 0), LEFT, buff=0.15)
        l1 = Text("0.02", font=MONO, font_size=16, color=COS_C).next_to(ax.c2p(1000, 0.02), RIGHT, buff=0.15)
        lt = Text("step 0 → 1,000", font_size=16, color=GREY_B).next_to(ax, DOWN, buff=0.1)
        self.at("beta")
        self.play(Create(ax), Create(line), FadeIn(l0), FadeIn(l1), FadeIn(lt), run_time=0.8)
        self.end_section()

    # ------------------------------------------------------------------ 3. a thousand steps
    def thousand(self):
        self.section(3)
        self.clear_stage()
        row = row_of(f"{OUT}/noising_linear.png").move_to([0, 0.8, 0])
        labels = VGroup(*[Text(f"t = {t}", font=MONO, font_size=16, color=GREY_B).next_to(im, DOWN, buff=0.15) for t, im in zip(SHOW, row)])
        self.at("thousand")
        self.play(FadeIn(row[0]), FadeIn(labels[0]), run_time=0.3)
        cues = {1: "50", 4: "300", 6: "700", 7: "end"}
        k = 1
        for idx, cue in cues.items():
            self.at(cue)
            self.play(*[FadeIn(row[j]) for j in range(k, idx + 1)], *[FadeIn(labels[j]) for j in range(k, idx + 1)], run_time=0.5)
            k = idx + 1
        ab = R["linear"]["final_alpha_bar"]
        n = Text(f"ᾱ at step 1,000 = {ab:.5f}: what's left is almost exactly pure noise", font_size=22, color=YELLOW).move_to([0, -1.3, 0])
        self.at("pure")
        self.play(FadeIn(n), run_time=0.4)
        s = Text("→ generating will start from a standard normal", font_size=22, color=ORANGE).next_to(n, DOWN, buff=0.3)
        self.at("generating")
        self.play(FadeIn(s), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 4. the jump
    def jump(self):
        self.section(4)
        self.clear_stage()
        ab = Text("ᾱₜ = (1 − β₁)(1 − β₂) … (1 − βₜ)", font=MONO, font_size=26).move_to([0, 2.5, 0])
        self.at("multiply")
        self.play(FadeIn(ab), run_time=0.5)
        jf = Text("xₜ = √ᾱₜ · x₀  +  √(1 − ᾱₜ) · ε", font=MONO, font_size=32, color=YELLOW).move_to([0, 1.2, 0])
        parts = VGroup(Text("a scaled copy of the picture", font_size=20, color=SIG_C), Text("plus scaled noise", font_size=20, color=NOISE_C)).arrange(RIGHT, buff=1.4).next_to(jf, DOWN, buff=0.3)
        self.at("jump")
        self.play(FadeIn(jf), run_time=0.5)
        self.at("scaled")
        self.play(FadeIn(parts[0]), run_time=0.3)
        self.at("plus")
        self.play(FadeIn(parts[1]), run_time=0.3)
        c = R["step_vs_jump_t500"]
        rows = VGroup(Text("the same picture noised 2,000 times, to step 500:", font_size=20, color=GREY_B),
                      Text(f"500 small steps   spread {c['steps_mean_std']:.4f}", font=MONO, font_size=20),
                      Text(f"one jump          spread {c['jump_mean_std']:.4f}", font=MONO, font_size=20),
                      Text(f"formula √(1 − ᾱ)  spread {c['expected_std']:.4f}", font=MONO, font_size=20, color=YELLOW)).arrange(DOWN, aligned_edge=LEFT, buff=0.15).move_to([0, -1.6, 0])
        self.at("checked")
        self.play(FadeIn(rows[0]), run_time=0.4)
        self.at("500")
        self.play(FadeIn(rows[1]), run_time=0.3)
        self.at("jump")
        self.play(FadeIn(rows[2]), run_time=0.3)
        self.at("formula")
        self.play(FadeIn(rows[3]), run_time=0.3)
        self.end_section()

    # ------------------------------------------------------------------ SNR axes
    def snr_axes(self):
        ax = Axes(x_range=[0, 1000, 250], y_range=[-50, 50, 25], x_length=8.0, y_length=4.4, tips=False,
                  axis_config={"color": GREY_C}).move_to([-1.2, -0.2, 0])
        xl = VGroup(*[Text(str(t), font=MONO, font_size=14, color=GREY_B).next_to(ax.c2p(t, -50), DOWN, buff=0.12) for t in (0, 250, 500, 750, 1000)])
        yl = VGroup(*[Text(f"{v:+d} dB" if v else "0 dB", font=MONO, font_size=14, color=GREY_B).next_to(ax.c2p(0, v), LEFT, buff=0.12) for v in (-50, -25, 0, 25, 50)])
        zero = DashedLine(ax.c2p(0, 0), ax.c2p(1000, 0), color=GREY_D)
        t = Text("step", font_size=16, color=GREY_B).next_to(xl, DOWN, buff=0.1)
        return ax, VGroup(ax, xl, yl, zero, t)

    def curve(self, ax, name, color):
        pts = [(c["t"], c["snr_db"]) for c in R[name]["curve"] if c["snr_db"] is not None]
        pts = [(t, max(v, -50)) for t, v in pts]
        return VMobject(color=color, stroke_width=4).set_points_smoothly([ax.c2p(t, v) for t, v in pts])

    # ------------------------------------------------------------------ 5. signal to noise
    def snr(self):
        self.section(5)
        self.clear_stage()
        h = Text("signal-to-noise ratio  ᾱ / (1 − ᾱ)", font_size=26).to_edge(UP, buff=0.4)
        self.at("signal")
        self.play(FadeIn(h), run_time=0.4)
        ax, g = self.snr_axes()
        self.play(FadeIn(g), run_time=0.5)
        lin = self.curve(ax, "linear", LIN_C)
        self.at("starts")
        self.play(Create(lin), run_time=2.0)
        d = Dot(ax.c2p(260, 0), radius=0.08, color=YELLOW)
        dl = Text("step 260: signal = noise", font_size=18, color=YELLOW).next_to(d, UR, buff=0.1)
        self.at("260")
        self.play(FadeIn(d, scale=2), FadeIn(dl), run_time=0.5)
        e = Text("−44 dB", font=MONO, font_size=18, color=LIN_C).next_to(ax.c2p(1000, -44), RIGHT, buff=0.1)
        self.at("44")
        self.play(FadeIn(e), run_time=0.3)
        self.lin, self.ax, self.snr_group = lin, ax, VGroup(g, lin, d, dl, e, h)
        self.end_section()

    # ------------------------------------------------------------------ 6. schedules
    def schedules(self):
        self.section(6)
        ax = self.ax
        band = Rectangle(width=abs(ax.c2p(1000, 0)[0] - ax.c2p(0, 0)[0]), height=abs(ax.c2p(0, -20)[1] - ax.c2p(0, -50)[1]),
                         stroke_width=0, fill_color=RED_E, fill_opacity=0.35).move_to(ax.c2p(500, -35))
        bl = Text("almost pure noise (below −20 dB)", font_size=16, color=RED_B).move_to(ax.c2p(250, -42))
        self.at("wasteful")
        self.play(FadeIn(band), FadeIn(bl), run_time=0.5)
        legend = VGroup(Text(f"linear: {R['linear']['steps_with_snr_below_minus20db']} steps", font_size=20, color=LIN_C),
                        Text(f"cosine: {R['cosine']['steps_with_snr_below_minus20db']} steps", font_size=20, color=COS_C)).arrange(DOWN, aligned_edge=LEFT, buff=0.25).move_to([4.8, 1.6, 0])
        self.at("326")
        self.play(FadeIn(legend[0]), run_time=0.4)
        cos = self.curve(ax, "cosine", COS_C)
        self.at("cosine")
        self.play(Create(cos), run_time=1.5)
        self.at("64")
        self.play(FadeIn(legend[1]), run_time=0.4)
        jl = Text("judge reads half\nthe pictures until:", font_size=18, color=GREY_B, line_spacing=0.9).move_to([4.8, -0.3, 0])
        marks = VGroup()
        for name, col in (("linear", LIN_C), ("cosine", COS_C)):
            t = R[name]["last_t_half_read"]
            v = next(c["snr_db"] for c in R[name]["curve"] if c["t"] == t)
            d = Dot(ax.c2p(t, v), radius=0.08, color=col)
            lab = Text(f"step {t}", font=MONO, font_size=16, color=col)
            lab.next_to(d, DL if name == "linear" else UR, buff=0.08)
            marks.add(VGroup(d, lab))
        self.at("judge")
        self.play(FadeIn(jl), run_time=0.3)
        self.at("75")
        self.play(FadeIn(marks[0]), run_time=0.4)
        self.at("175")
        self.play(FadeIn(marks[1]), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 7. why bother
    def why(self):
        self.section(7)
        self.clear_stage()
        row = row_of(f"{OUT}/noising_cosine.png", height=1.25)
        rev = Group(*reversed(list(row))).arrange(RIGHT, buff=0.12).move_to([0, 0.9, 0])
        self.at("destroy")
        self.play(FadeIn(rev), run_time=0.6)
        arrows = VGroup(*[Arrow(rev[i].get_bottom() + DOWN * 0.1, rev[i + 1].get_bottom() + DOWN * 0.1, buff=0.05, color=ORANGE, stroke_width=3,
                                max_tip_length_to_length_ratio=0.3).shift(DOWN * 0.15) for i in range(7)])
        small = Text("each step: undo a little noise (an easy problem)", font_size=22, color=ORANGE).move_to([0, -1.0, 0])
        self.at("undoing")
        self.play(FadeIn(small), run_time=0.4)
        self.at("walk")
        self.play(LaggedStart(*[GrowArrow(a) for a in arrows], lag_ratio=0.25), run_time=1.6)
        new = Text("…to a brand new picture", font_size=24, color=YELLOW).move_to([0, -2.0, 0])
        self.at("brand")
        self.play(FadeIn(new), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 8. code
    def code(self):
        self.section(8)
        self.clear_stage()
        code, hl = code_panel(CODE, font_size=20)
        code.move_to([0, 0.3, 0])
        self.play(FadeIn(code, shift=0.2 * UP), run_time=0.5)
        self.at("betas")
        self.play(Create(hl), run_time=0.3)
        self.at("running")
        self.play(highlight(hl, code, 1), run_time=0.3)
        self.at("jump")
        self.play(highlight(hl, code, 2), run_time=0.3)
        self.end_section()

    # ------------------------------------------------------------------ 9. outro
    def outro(self):
        self.section(9)
        self.clear_stage(run_time=0.4)
        card_ = next_up_card(NEXT)
        self.at("next")
        self.play(FadeIn(card_, shift=0.2 * UP), run_time=0.6)
        self.end_section()
        finish(self, hold=1.5)
