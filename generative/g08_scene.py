"""How Generative Models Work: Images and Sound, episode 8 — The Denoiser: U-Nets and Time.

Render from the repo root:  ./render.sh generative g08
Numbers and pictures come from code/g08_unets_and_time (train_ablation.py, eval_full.py, eval_samplers.py).
"""
import json
import math

import numpy as np
from manim import *

from common import MONO, code_panel, finish, highlight, next_up_card
from gen_common import frame, tile
from intro import play_token_intro
from g08_script import NEXT, TAGLINE, TITLE
from voiced_scene import VoicedScene

SERIES_LABEL = "HOW GENERATIVE MODELS WORK  ·  EPISODE 8"
CODE_DIR = "code/g08_unets_and_time"
OUT = f"{CODE_DIR}/out"
G06 = "code/g06_learning_to_denoise/out"
RES = {v: json.load(open(f"{CODE_DIR}/results_{v}.json")) for v in ("noskip", "notime", "full")}
SAMP = json.load(open(f"{CODE_DIR}/results_samplers.json"))["results"]
LEVELS = [10, 50, 100, 200, 400, 600, 800, 950]
ERR = {"full": [RES["full"]["eps_mse_by_step"][str(l)] for l in LEVELS],
       "noskip": [RES["noskip"]["history"][-1]["eps_mse_by_step"][str(l)] for l in LEVELS],
       "notime": [RES["notime"]["history"][-1]["eps_mse_by_step"][str(l)] for l in LEVELS]}
COL = {"full": GREEN_C, "noskip": RED_C, "notime": PURPLE_B}
NAME = {"full": "full U-Net", "noskip": "no skip connections", "notime": "no time input"}
SKIP_C, TIME_C, NET_C = YELLOW, BLUE_C, TEAL_C
CODE = """def timestep_embedding(t, dim):
    freqs = torch.exp(-math.log(10_000) * torch.arange(dim // 2) / (dim // 2))
    args = t[:, None] * freqs[None]
    return torch.cat([torch.sin(args), torch.cos(args)], dim=1)

# inside every block:
h = h + self.time(temb)[:, :, None, None]"""


def row(path, n=8, h=1.0, r=0, buff=0.1):
    return Group(*[tile(path, i, h, row=r) for i in range(n)]).arrange(RIGHT, buff=buff)


def pct(v):
    return f"{v * 100:.0f}%" if v >= 0.1 or v == 0 else f"{v * 100:.1f}%"


class UNetsAndTimeVideo(VoicedScene):
    VIDEO = "g08"

    def construct(self):
        play_token_intro(self, TITLE, 8, TAGLINE, label=SERIES_LABEL)
        self.hook()          # 1
        self.job()           # 2
        self.unet()          # 3
        self.no_skips()      # 4
        self.time_input()    # 5
        self.no_time_step()  # 6
        self.no_time_walk()  # 7
        self.at_scale()      # 8
        self.code()          # 9
        self.outro()         # 10

    def clear_stage(self, *keep, run_time=0.6):
        self.play(*[FadeOut(m) for m in self.mobjects if m not in keep], run_time=run_time)

    def error_chart(self, center, w, h, which):
        """Noise-guess error by noise level, log scale."""
        lo, hi = -3.3, -1.5
        ax = Axes(x_range=[0, 7, 1], y_range=[0, hi - lo, 0.5], x_length=w, y_length=h, tips=False,
                  axis_config={"color": GREY_C, "include_ticks": False}).move_to(center)
        Y = lambda v: math.log10(v) - lo
        xl = VGroup(*[Text(f"{l}", font=MONO, font_size=14, color=GREY_B).next_to(ax.c2p(i, 0), DOWN, buff=0.1)
                      for i, l in enumerate(LEVELS)])
        yl = VGroup(*[Text(s, font=MONO, font_size=14, color=GREY_B).next_to(ax.c2p(0, Y(v)), LEFT, buff=0.1)
                      for v, s in ((0.001, "0.001"), (0.01, "0.01"))])
        xt = Text("noise level (step t)", font_size=16, color=GREY_B).next_to(xl, DOWN, buff=0.1)
        yt = Text("noise-guess error", font_size=16, color=GREY_B).rotate(PI / 2).next_to(yl, LEFT, buff=0.15)
        lines = {k: VMobject(color=COL[k], stroke_width=4).set_points_as_corners(
            [ax.c2p(i, Y(v)) for i, v in enumerate(ERR[k])]) for k in which}
        return ax, VGroup(ax, xl, yl, xt, yt), lines, Y

    # ------------------------------------------------------------------ 1. hook
    def hook(self):
        self.section(1)
        rows_ = [(f"{G06}/samples_step20000.png", "full"), (f"{OUT}/noskip_samples_step20000.png", "noskip"),
                 (f"{OUT}/notime_samples_step20000.png", "notime")]
        g = Group(*[row(p, h=1.0) for p, _ in rows_]).arrange(DOWN, buff=0.55).move_to([1.0, -0.3, 0])
        labels = [VGroup(Text(pct(SAMP[k]["eta0"]["recognizable"]), font=MONO, font_size=30, color=COL[k]),
                         Text(NAME[k], font_size=16, color=GREY_B)).arrange(DOWN, buff=0.08).next_to(g[i], LEFT, buff=0.35)
                  for i, (_, k) in enumerate(rows_)]
        head = Text("recognized by the judge (DDIM, 50 steps)", font_size=20, color=GREY_B).to_edge(UP, buff=0.5)
        self.at("70")
        self.play(FadeIn(head), FadeIn(g[0]), FadeIn(labels[0]), run_time=0.6)
        self.at("remove")
        self.play(FadeIn(g[1]), run_time=0.5)
        self.at("0")
        self.play(FadeIn(labels[1], scale=1.2), run_time=0.4)
        self.at("hide")
        self.play(FadeIn(g[2]), run_time=0.5)
        self.at("1")
        self.play(FadeIn(labels[2], scale=1.2), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 2. the job
    def job(self):
        self.section(2)
        self.clear_stage()
        h = 1.8
        xt = tile(f"{G06}/denoise_step20000.png", 2, h, row=0).move_to([-4.6, 0.4, 0])
        lx = Text("noisy picture", font_size=20, color=GREY_B).next_to(xt, DOWN, buff=0.15)
        lt = Text("step t = 400", font=MONO, font_size=20, color=TIME_C).next_to(lx, DOWN, buff=0.15)
        net = VGroup(RoundedRectangle(width=2.4, height=1.2, corner_radius=0.15, color=NET_C, fill_color=NET_C, fill_opacity=0.15),
                     Text("denoiser", font_size=26, color=NET_C)).move_to([0, 0.4, 0])
        net[1].move_to(net[0])
        a1 = Arrow(xt.get_right(), net.get_left(), buff=0.15, color=GREY_B)
        a0 = Arrow(lt.get_right(), net.get_left() + DOWN * 0.3, buff=0.15, color=TIME_C)
        out = Square(side_length=h, color=GREY_B).move_to([4.6, 0.4, 0])
        out_t = Text("noise\nguess", font_size=24, line_spacing=0.9).move_to(out)
        out_s = Text("3 × 32 × 32", font=MONO, font_size=18, color=GREY_B).next_to(out, DOWN, buff=0.15)
        a2 = Arrow(net.get_right(), out.get_left(), buff=0.15, color=GREY_B)
        self.at("job")
        self.play(FadeIn(net), run_time=0.4)
        self.at("noisy")
        self.play(FadeIn(xt), FadeIn(frame(xt)), FadeIn(lx), GrowArrow(a1), run_time=0.5)
        self.at("step")
        self.play(FadeIn(lt), GrowArrow(a0), run_time=0.4)
        self.at("returns")
        self.play(GrowArrow(a2), Create(out), FadeIn(out_t), run_time=0.5)
        self.at("same")
        self.play(FadeIn(out_s), run_time=0.3)
        s = Text("same size in, same size out: an answer for every pixel", font_size=22, color=YELLOW).move_to([0, -2.4, 0])
        self.at("pixel")
        self.play(FadeIn(s), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 3. the U
    def build_u(self):
        spec = [(-4.6, 1.3, 2.4, 0.35, "32×32"), (-2.4, 0.0, 1.3, 0.6, "16×16"), (0.0, -1.3, 0.75, 0.9, "8×8"),
                (2.4, 0.0, 1.3, 0.6, "16×16"), (4.6, 1.3, 2.4, 0.35, "32×32")]
        boxes = VGroup(*[Rectangle(width=w, height=hh, color=NET_C, fill_color=NET_C, fill_opacity=0.25).move_to([x, y, 0])
                         for x, y, hh, w, _ in spec])
        labs = VGroup(*[Text(s, font=MONO, font_size=16, color=GREY_B).next_to(boxes[i], DOWN, buff=0.12)
                        for i, (*_, s) in enumerate(spec)])
        moves = VGroup(*[Arrow(boxes[i].get_right(), boxes[i + 1].get_left(), buff=0.1, color=GREY_B, stroke_width=3)
                         for i in range(4)])
        skips = VGroup(DashedLine(boxes[0].get_right() + RIGHT * 0.1, boxes[4].get_left() + LEFT * 0.1, color=SKIP_C).add_tip(tip_length=0.2),
                       DashedLine(boxes[1].get_right() + RIGHT * 0.1, boxes[3].get_left() + LEFT * 0.1, color=SKIP_C).add_tip(tip_length=0.2))
        skips[0].shift(UP * 0.6)
        return boxes, labs, moves, skips

    def unet(self):
        self.section(3)
        self.clear_stage()
        boxes, labs, moves, skips = self.build_u()
        self.u = (boxes, labs, moves, skips)
        down = Text("down: see the whole shape", font_size=20, color=GREY_B).move_to([-3.4, -2.4, 0])
        up = Text("up: back to every pixel", font_size=20, color=GREY_B).move_to([3.4, -2.4, 0])
        self.at("halves")
        self.play(FadeIn(boxes[0]), FadeIn(labs[0]), run_time=0.4)
        self.at("down")
        self.play(GrowArrow(moves[0]), FadeIn(boxes[1]), FadeIn(labs[1]), run_time=0.5)
        self.at("8x8", "eight")
        self.play(GrowArrow(moves[1]), FadeIn(boxes[2]), FadeIn(labs[2]), run_time=0.5)
        self.at("wider")
        self.play(FadeIn(down), run_time=0.4)
        self.at("up")
        self.play(GrowArrow(moves[2]), FadeIn(boxes[3]), FadeIn(labs[3]), GrowArrow(moves[3]), FadeIn(boxes[4]), FadeIn(labs[4]),
                  FadeIn(up), run_time=0.8)
        sl = Text("skip connections: fine detail straight across", font_size=20, color=SKIP_C).move_to([0, 2.6, 0])
        self.at("skip")
        self.play(Create(skips), FadeIn(sl), run_time=0.8)
        self.end_section()

    # ------------------------------------------------------------------ 4. without skips
    def no_skips(self):
        self.section(4)
        boxes, labs, moves, skips = self.u
        crosses = VGroup(*[Cross(Square(side_length=0.45).move_to(s.get_center()), stroke_color=RED, stroke_width=6)
                           for s in skips])
        self.at("remove")
        self.play(skips.animate.set_opacity(0.25), FadeIn(crosses), run_time=0.5)
        sq = SurroundingRectangle(boxes[2], color=RED, buff=0.12)
        self.at("squeeze")
        self.play(Create(sq), run_time=0.4)
        self.at("100")
        self.clear_stage(run_time=0.4)
        ax, chart, lines, Y = self.error_chart([0.3, 0.75, 0], 8.5, 3.1, ("full", "noskip"))
        leg = VGroup(*[Text(NAME[k], font_size=18, color=COL[k]) for k in ("full", "noskip")]).arrange(RIGHT, buff=0.6).next_to(chart, UP, buff=0.15)
        self.play(FadeIn(chart), Create(lines["full"]), Create(lines["noskip"]), FadeIn(leg), run_time=0.8)
        i = LEVELS.index(100)
        ratio = ERR["noskip"][i] / ERR["full"][i]
        br = DoubleArrow(ax.c2p(i, Y(ERR["full"][i])), ax.c2p(i, Y(ERR["noskip"][i])), buff=0.05,
                         color=YELLOW, stroke_width=3, tip_length=0.15)
        bl = Text(f"{ratio:.1f}×", font=MONO, font_size=20, color=YELLOW).next_to(br, RIGHT, buff=0.1)
        self.at("worse")
        self.play(GrowFromCenter(br), FadeIn(bl), run_time=0.4)
        a = row(f"{OUT}/noskip_eta1_samples.png", h=0.95, r=0).move_to([0.9, -2.55, 0])
        la = Text("no skips", font_size=18, color=COL["noskip"]).next_to(a, LEFT, buff=0.3)
        self.at("blobs")
        self.play(FadeIn(a), FadeIn(la), run_time=0.5)
        n = Text(f"DDIM {pct(SAMP['noskip']['eta0']['recognizable'])}   ·   fresh noise {pct(SAMP['noskip']['eta1']['recognizable'])}",
                 font=MONO, font_size=18, color=COL["noskip"]).next_to(a, UP, buff=0.12)
        self.at("recognizable")
        self.play(FadeIn(n), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 5. time
    def time_input(self):
        self.section(5)
        self.clear_stage()
        head = Text("the step number becomes a vector", font_size=24).to_edge(UP, buff=0.5)
        freqs = np.exp(-math.log(10_000) * np.arange(32) / 32)
        strips = VGroup()
        for t in (50, 400, 900):
            vals = np.concatenate([np.sin(t * freqs), np.cos(t * freqs)])
            cells = VGroup(*[Rectangle(width=0.15, height=0.45, stroke_width=0,
                                    fill_color=interpolate_color(BLUE_D, RED_D, (v + 1) / 2), fill_opacity=1) for v in vals]).arrange(RIGHT, buff=0.02)
            lab = Text(f"t = {t}", font=MONO, font_size=18, color=TIME_C).next_to(cells, LEFT, buff=0.3)
            strips.add(VGroup(lab, cells))
        strips.arrange(DOWN, buff=0.35, aligned_edge=RIGHT).move_to([0, 0.6, 0])
        cap = Text("sines │ cosines, fast to slow", font_size=16, color=GREY_B).next_to(strips, DOWN, buff=0.2)
        self.at("time")
        self.play(FadeIn(head), run_time=0.4)
        self.at("sines")
        self.play(LaggedStart(*[FadeIn(s) for s in strips], lag_ratio=0.3), FadeIn(cap), run_time=1.2)
        pos = Text("(like a transformer's position code)", font_size=18, color=GREY_B).next_to(head, DOWN, buff=0.15)
        self.at("transformers", "transformer")
        self.play(FadeIn(pos), run_time=0.3)
        blk = Text("added into every block  →  “expect this much noise”", font_size=22, color=TIME_C).move_to([0, -2.3, 0])
        self.at("block")
        self.play(FadeIn(blk), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 6. without time: one step
    def no_time_step(self):
        self.section(6)
        self.clear_stage()
        path = f"{OUT}/notime_denoise_step20000.png"
        top = Group(*[tile(path, i, 0.95, row=0) for i in range(5)]).arrange(RIGHT, buff=0.12)
        bot = Group(*[tile(path, i, 0.95, row=1) for i in range(5)]).arrange(RIGHT, buff=0.12)
        g = Group(top, bot).arrange(DOWN, buff=0.15).move_to([2.6, 1.6, 0])
        lt = Text("noisy input\n(t = 50 … 900)", font_size=16, color=GREY_B, line_spacing=0.9).next_to(top, LEFT, buff=0.3)
        lb = Text("clean guess,\nno time input", font_size=16, color=COL["notime"], line_spacing=0.9).next_to(bot, LEFT, buff=0.3)
        self.at("cleaning")
        self.play(FadeIn(top), FadeIn(lt), run_time=0.4)
        self.play(FadeIn(bot), FadeIn(lb), run_time=0.4)
        ax, chart, lines, Y = self.error_chart([0.3, -1.5, 0], 8.0, 2.2, ("full", "noskip", "notime"))
        lines["noskip"].set_stroke(opacity=0.3)
        leg = VGroup(*[Text(NAME[k], font_size=16, color=COL[k]) for k in ("full", "notime", "noskip")]).arrange(DOWN, aligned_edge=LEFT, buff=0.1)
        leg[2].set_opacity(0.4)
        leg.next_to(chart, RIGHT, buff=0.1).shift(UP * 0.4)
        self.at("looking")
        self.play(FadeIn(chart), Create(lines["full"]), Create(lines["noskip"]), FadeIn(leg), run_time=0.6)
        self.play(Create(lines["notime"]), run_time=0.6)
        worst = max(a / b for a, b in zip(ERR["notime"], ERR["full"]))
        n = Text(f"at most {worst:.1f}× the full model's error", font_size=18, color=COL["notime"]).next_to(chart, UP, buff=0.05).shift(LEFT * 1.2)
        self.at("60")
        self.play(FadeIn(n), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 7. without time: the walk
    def no_time_walk(self):
        self.section(7)
        self.clear_stage()
        a = row(f"{OUT}/notime_samples_step20000.png", h=1.1, r=0).move_to([1.0, 1.5, 0])
        b = row(f"{OUT}/notime_eta1_samples.png", h=1.1, r=1).move_to([1.0, -0.3, 0])
        la = VGroup(Text(pct(SAMP["notime"]["eta0"]["recognizable"]), font=MONO, font_size=28, color=COL["notime"]),
                    Text("DDIM", font_size=16, color=GREY_B)).arrange(DOWN, buff=0.08).next_to(a, LEFT, buff=0.35)
        lb = VGroup(Text(pct(SAMP["notime"]["eta1"]["recognizable"]), font=MONO, font_size=28, color=COL["notime"]),
                    Text("fresh noise", font_size=16, color=GREY_B)).arrange(DOWN, buff=0.08).next_to(b, LEFT, buff=0.35)
        head = Text("no time input: samples after 50 steps", font_size=20, color=GREY_B).to_edge(UP, buff=0.4)
        self.at("fails")
        self.play(FadeIn(head), FadeIn(a), run_time=0.5)
        self.at("1")
        self.play(FadeIn(la), run_time=0.3)
        self.at("33")
        self.play(FadeIn(b), FadeIn(lb), run_time=0.5)
        noisy = VGroup(*[SurroundingRectangle(a[i], color=RED, buff=0.04) for i in (0, 6, 7)],
                       *[SurroundingRectangle(b[i], color=RED, buff=0.04) for i in (1, 2)])
        self.at("covered")
        self.play(Create(noisy), run_time=0.5)
        m = Text("misjudge the noise level → remove the wrong amount → errors build up", font_size=20).move_to([0, -1.9, 0])
        self.at("misjudgment")
        self.play(FadeIn(m), run_time=0.4)
        k = Text("right at every step of the walk, not just on average", font_size=24, color=YELLOW).move_to([0, -2.8, 0])
        self.at("right")
        self.play(FadeIn(k), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 8. at scale
    def at_scale(self):
        self.section(8)
        self.clear_stage()
        ref = VGroup(Text("Sun et al., 2025", font_size=26, color=TIME_C),
                     Text("“Is Noise Conditioning Necessary for Denoising Generative Models?”", font_size=20, slant=ITALIC, color=GREY_B)
                     ).arrange(DOWN, buff=0.15).move_to([0, 1.9, 0])
        b1 = Text("•  many larger models work almost as well without the time input", font_size=22)
        b2 = Text("•  deterministic DDIM sampling broke worst, as it did here", font_size=22)
        b3 = Text("•  standard models still pass the time in: cheap, and every step gets easier", font_size=22, color=YELLOW)
        bl = VGroup(b1, b2, b3).arrange(DOWN, aligned_edge=LEFT, buff=0.45).move_to([0, -0.6, 0])
        self.at("bigger")
        self.play(FadeIn(ref), run_time=0.5)
        self.at("study")
        self.play(FadeIn(b1), run_time=0.4)
        self.at("dim")
        self.play(FadeIn(b2), run_time=0.4)
        self.at("standard")
        self.play(FadeIn(b3), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 9. code
    def code(self):
        self.section(9)
        self.clear_stage()
        code, hl = code_panel(CODE, font_size=18)
        code.move_to([0, 0.3, 0])
        self.play(FadeIn(code, shift=0.2 * UP), run_time=0.4)
        self.at("turn")
        self.play(Create(hl), run_time=0.2)
        self.play(highlight(hl, code, 3), run_time=0.3)
        self.at("add")
        self.play(highlight(hl, code, 6), run_time=0.3)
        self.end_section()

    # ------------------------------------------------------------------ 10. outro
    def outro(self):
        self.section(10)
        self.clear_stage(run_time=0.4)
        card_ = next_up_card(NEXT)
        self.at("next")
        self.play(FadeIn(card_, shift=0.2 * UP), run_time=0.6)
        self.end_section()
        finish(self, hold=1.5)
