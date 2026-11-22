"""How Generative Models Work: Images and Sound, episode 9 — Flow Matching.

Render from the repo root:  ./render.sh generative g09
Numbers and pictures come from code/g09_flow_matching (train_flow.py, diffusion_fresh_noise.py) and episode 7's results.
"""
import json
import math

import numpy as np
from manim import *
from PIL import Image

from common import MONO, code_panel, finish, highlight, next_up_card
from gen_common import dots, frame, plane, tile
from intro import play_token_intro
from g09_script import NEXT, TAGLINE, TITLE
from voiced_scene import VoicedScene

SERIES_LABEL = "HOW GENERATIVE MODELS WORK  ·  EPISODE 9"
CODE_DIR = "code/g09_flow_matching"
OUT = f"{CODE_DIR}/out"
TOY = json.load(open(f"{CODE_DIR}/results_toy.json"))
SH = json.load(open(f"{CODE_DIR}/results_shapes.json"))
FRESH = json.load(open(f"{CODE_DIR}/results_fresh.json"))
G07 = {(r["eta"], r["steps"]): r for r in json.load(open("code/g07_sampling/results.json"))["runs"]}
FINAL = SH["history"][-1]
FLOW = {k: FINAL[f"euler{k}"]["recognizable"] for k in (5, 10, 20, 50)}
DIFF = {5: FRESH["eta1_steps5"]["recognizable"], 10: G07[(1.0, 10)]["recognizable"],
        20: G07[(1.0, 20)]["recognizable"], 50: G07[(1.0, 50)]["recognizable"]}
BY = {b["steps"]: b for b in TOY["by_steps"]}
DATA = np.array(TOY["data"])
FLOW_C, DIFF_C, REFLOW_C, DATA_C = TEAL_C, ORANGE, YELLOW, GREY_D
CODE = """t = torch.rand(len(x0)).view(-1, 1, 1, 1)
x_t = (1 - t) * x0 + t * eps                           # a straight line
loss = ((model(x_t, t * 1000) - (eps - x0)) ** 2).mean()  # its direction
# generate: from t = 1 down to 0 in small steps dt
x = x - dt * model(x, t * 1000)"""


def row(path, n=8, h=1.0, r=0, buff=0.1):
    return Group(*[tile(path, i, h, row=r) for i in range(n)]).arrange(RIGHT, buff=buff)


def img(arr, h):
    m = ImageMobject(np.clip(arr * 255, 0, 255).astype(np.uint8))
    m.set_resampling_algorithm(RESAMPLING_ALGORITHMS["nearest"])
    return m.scale_to_fit_height(h)


class FlowMatchingVideo(VoicedScene):
    VIDEO = "g09"

    def construct(self):
        play_token_intro(self, TITLE, 9, TAGLINE, label=SERIES_LABEL)
        self.hook()          # 1
        self.idea()          # 2
        self.versus()        # 3
        self.toy()           # 4
        self.not_straight()  # 5
        self.reflow()        # 6
        self.shapes()        # 7
        self.practice()      # 8
        self.code()          # 9
        self.outro()         # 10

    def clear_stage(self, *keep, run_time=0.6):
        self.play(*[FadeOut(m) for m in self.mobjects if m not in keep], run_time=run_time)

    def toy_plane(self, center, title, color=GREY_B, size=3.6, lo=-2.4, hi=2.4):
        box, ax = plane(center, size=size, lo=lo, hi=hi)
        t = Text(title, font_size=18, color=color).next_to(box, UP, buff=0.12)
        return box, ax, t

    # ------------------------------------------------------------------ 1. hook
    def hook(self):
        self.section(1)
        a = row(f"{OUT}/diffusion_eta1_05.png", h=1.15).move_to([0.9, 0.9, 0])
        b = row(f"{OUT}/flow_euler05_step20000.png", h=1.15).move_to([0.9, -1.4, 0])
        la = VGroup(Text(f"{DIFF[5]:.0%}", font=MONO, font_size=32, color=DIFF_C),
                    Text("diffusion", font_size=16, color=GREY_B)).arrange(DOWN, buff=0.08).next_to(a, LEFT, buff=0.35)
        lb = VGroup(Text(f"{FLOW[5]:.0%}", font=MONO, font_size=32, color=FLOW_C),
                    Text("flow matching", font_size=16, color=GREY_B)).arrange(DOWN, buff=0.08).next_to(b, LEFT, buff=0.35)
        head = Text("5 network calls per picture", font_size=26).to_edge(UP, buff=0.6)
        sa = Text("episode 6's model, with fresh noise (η = 1)", font_size=16, color=GREY_B).next_to(a, UP, buff=0.12)
        sb = Text("the same U-Net, trained with flow matching", font_size=16, color=GREY_B).next_to(b, UP, buff=0.12)
        self.at("five")
        self.play(FadeIn(head), run_time=0.4)
        self.at("diffusion")
        self.play(FadeIn(a), FadeIn(sa), run_time=0.5)
        self.at("7")
        self.play(FadeIn(la, scale=1.2), run_time=0.3)
        self.at("same")
        self.play(FadeIn(b), FadeIn(sb), run_time=0.5)
        self.at("76")
        self.play(FadeIn(lb, scale=1.2), run_time=0.3)
        self.end_section()

    # ------------------------------------------------------------------ 2. the idea
    def idea(self):
        self.section(2)
        self.clear_stage()
        pic = np.asarray(Image.open(f"{OUT}/flow_euler50_step20000.png").convert("RGB"))[:128:4, 130 * 3:130 * 3 + 128:4] / 255.0
        noise = np.clip(0.5 + 0.5 * np.random.default_rng(5).standard_normal(pic.shape), 0, 1)
        ts = [0, 0.25, 0.5, 0.75, 1.0]
        h = 1.3
        tiles = Group(*[img((1 - t) * pic + t * noise, h) for t in ts]).arrange(LEFT, buff=0.55).move_to([0, 1.0, 0])
        labs = VGroup(*[Text(f"t = {t:g}", font=MONO, font_size=16, color=GREY_B).next_to(tiles[i], DOWN, buff=0.12)
                        for i, t in enumerate(ts)])
        line = Line(tiles[4].get_center(), tiles[0].get_center(), color=GREY_D, stroke_width=3)
        self.add(line)
        line.set_opacity(0)
        self.at("sliding")
        self.play(line.animate.set_opacity(1), FadeIn(tiles[0]), FadeIn(labs[0]), FadeIn(tiles[4]), FadeIn(labs[4]), run_time=0.6)
        n_l = Text("pure noise", font_size=18, color=GREY_B).next_to(tiles[4], UP, buff=0.12)
        p_l = Text("the picture", font_size=18, color=GREY_B).next_to(tiles[0], UP, buff=0.12)
        self.at("noise")
        self.play(FadeIn(n_l), run_time=0.3)
        self.at("picture")
        self.play(FadeIn(p_l), run_time=0.3)
        self.at("mix")
        self.play(*[FadeIn(tiles[i]) for i in (1, 2, 3)], *[FadeIn(labs[i]) for i in (1, 2, 3)], run_time=0.6)
        eq = Text("x_t = (1 − t)·picture + t·noise", font=MONO, font_size=22).move_to([0, -1.0, 0])
        self.play(FadeIn(eq), run_time=0.4)
        v = Arrow(tiles[1].get_center() + DOWN * 1.15, tiles[3].get_center() + DOWN * 1.15, buff=0, color=FLOW_C, stroke_width=5)
        vl = Text("the network learns this direction:  v = noise − picture", font_size=20, color=FLOW_C).move_to([0, -1.75, 0])
        self.at("direction")
        self.play(FadeOut(eq), GrowArrow(v), FadeIn(vl), run_time=0.6)
        g = Text("generate: start from noise, step against v̂", font_size=22, color=YELLOW).move_to([0, -2.6, 0])
        self.at("generate")
        self.play(FadeIn(g), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 3. versus diffusion
    def versus(self):
        self.section(3)
        self.clear_stage()
        ax = Axes(x_range=[0, 1, 0.25], y_range=[0, 1, 0.5], x_length=5.6, y_length=3.4, tips=False,
                  axis_config={"color": GREY_C}).move_to([-2.9, -0.2, 0])
        xl = VGroup(*[Text(f"{v:g}", font=MONO, font_size=14, color=GREY_B).next_to(ax.c2p(v, 0), DOWN, buff=0.1) for v in (0, 0.5, 1)])
        xt = Text("t  (0 = picture, 1 = noise)", font_size=16, color=GREY_B).next_to(xl, DOWN, buff=0.1)
        yt = Text("how much picture is left", font_size=18).next_to(ax, UP, buff=0.2)
        T = 1000
        s = 0.008
        f = lambda t: math.cos((t + s) / (1 + s) * math.pi / 2) ** 2 / math.cos(s / (1 + s) * math.pi / 2) ** 2
        diff = ax.plot(lambda u: math.sqrt(max(f(u), 0)), x_range=[0, 1, 0.01], color=DIFF_C, stroke_width=4)
        flow = ax.plot(lambda u: 1 - u, x_range=[0, 1, 0.01], color=FLOW_C, stroke_width=4)
        ld = Text("diffusion: a curved schedule", font_size=16, color=DIFF_C).move_to(ax.c2p(0.62, 0.95))
        lf = Text("flow: a straight line", font_size=16, color=FLOW_C).move_to(ax.c2p(0.3, 0.25))
        self.at("curved")
        self.play(FadeIn(ax), FadeIn(xl), FadeIn(xt), FadeIn(yt), Create(diff), FadeIn(ld), run_time=0.8)
        tgt = VGroup(Text("what the network predicts", font_size=20),
                     Text("diffusion:  the noise ε", font=MONO, font_size=20, color=DIFF_C),
                     Text("flow:       ε − picture", font=MONO, font_size=20, color=FLOW_C),
                     Text("(the whole direction of travel)", font_size=16, color=GREY_B)).arrange(DOWN, aligned_edge=LEFT, buff=0.25).move_to([3.6, 0.4, 0])
        self.at("predicts")
        self.play(FadeIn(tgt[0]), FadeIn(tgt[1]), run_time=0.4)
        self.at("whole")
        self.play(Create(flow), FadeIn(lf), FadeIn(tgt[2]), FadeIn(tgt[3]), run_time=0.7)
        st = Text("a straight line: a few long steps can cross it", font_size=20, color=YELLOW).move_to([0, -2.9, 0])
        self.at("crossed")
        self.play(FadeIn(st), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 4. toy
    def toy(self):
        self.section(4)
        self.clear_stage()
        specs = [(-4.5, "diffusion (DDIM), 4 steps", DIFF_C, BY[4]["diffusion"]["points"]),
                 (0.0, "flow matching, 4 steps", FLOW_C, BY[4]["flow"]["points"]),
                 (4.5, "flow matching, 1 step", FLOW_C, BY[1]["flow"]["points"])]
        panels = []
        for x, title, col, pts in specs:
            box, ax, t = self.toy_plane([x, -0.2, 0], title, col)
            data = dots(ax, DATA, DATA_C, r=0.018)
            pts = np.clip(np.array(pts), -2.35, 2.35)
            panels.append((Group(box, t, data), dots(ax, pts, col, r=0.025)))
        self.at("spiral")
        self.play(*[FadeIn(p[0]) for p in panels[:2]], run_time=0.5)
        self.at("four")
        self.play(FadeIn(panels[1][1]), run_time=0.5)
        self.at("diffusion")
        self.play(FadeIn(panels[0][1]), run_time=0.5)
        self.at("single")
        self.play(FadeIn(panels[2][0]), FadeIn(panels[2][1]), run_time=0.6)
        c = Text("all 600 points in one spot", font_size=18, color=RED_B).next_to(panels[2][0], DOWN, buff=0.15)
        self.at("same")
        self.play(FadeIn(c), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 5. not straight
    def not_straight(self):
        self.section(5)
        self.clear_stage()
        box1, ax1, t1 = self.toy_plane([-3.4, -0.3, 0], "training lines: noise → data point", GREY_B, size=4.4, lo=-3.2, hi=3.2)
        box2, ax2, t2 = self.toy_plane([3.4, -0.3, 0], "learned flow paths (50 steps)", FLOW_C, size=4.4, lo=-3.2, hi=3.2)
        rng = np.random.default_rng(3)
        eps = rng.standard_normal((200, 2))
        eps = eps[np.abs(eps).max(1) < 2.9][:24]
        tgt = DATA[rng.choice(len(DATA), 24, replace=False)]
        lines = VGroup(*[Line(ax1.c2p(*e), ax1.c2p(*d), stroke_width=2, color=BLUE_B, stroke_opacity=0.8) for e, d in zip(eps, tgt)])
        self.at("paths")
        self.play(FadeIn(box1), FadeIn(t1), FadeIn(dots(ax1, DATA, DATA_C, r=0.015)), run_time=0.4)
        self.at("cross")
        self.play(LaggedStart(*[Create(l) for l in lines], lag_ratio=0.05), run_time=1.2)
        av = Text("where they cross: only the average\ndirection can be learned", font_size=16, color=GREY_B, line_spacing=0.9).next_to(box1, DOWN, buff=0.15)
        self.at("average")
        self.play(FadeIn(av), run_time=0.4)
        P = np.array(TOY["paths"]["flow"]["points"])
        P = P[:, np.abs(P).max(axis=(0, 2)) < 3.1]
        paths = VGroup(*[VMobject(stroke_width=2, color=FLOW_C).set_points_smoothly([ax2.c2p(*p) for p in P[:, i]]) for i in range(P.shape[1])])
        st = Text(f"median straightness {TOY['paths']['flow']['straightness_median']:.2f}  (1 = straight)", font_size=16,
                  color=FLOW_C).next_to(box2, DOWN, buff=0.15)
        self.at("bend")
        self.play(FadeIn(box2), FadeIn(t2), FadeIn(dots(ax2, DATA, DATA_C, r=0.015)), Create(paths), run_time=1.0)
        self.play(FadeIn(st), run_time=0.3)
        self.end_section()

    # ------------------------------------------------------------------ 6. reflow
    def reflow(self):
        self.section(6)
        self.clear_stage()
        box1, ax1, t1 = self.toy_plane([-3.4, -0.3, 0], "reflow paths (50 steps)", REFLOW_C, size=4.4, lo=-3.2, hi=3.2)
        box2, ax2, t2 = self.toy_plane([3.4, -0.3, 0], "reflow, 1 step", REFLOW_C, size=4.4, lo=-2.4, hi=2.4)
        head = Text("reflow: pair each noise with where the trained flow takes it, train a new flow on those pairs",
                    font_size=18).to_edge(UP, buff=0.35)
        self.at("reflow")
        self.play(FadeIn(head), run_time=0.4)
        self.at("pair")
        self.play(FadeIn(box1), FadeIn(t1), FadeIn(dots(ax1, DATA, DATA_C, r=0.015)), run_time=0.4)
        P = np.array(TOY["paths"]["reflow"]["points"])
        P = P[:, np.abs(P).max(axis=(0, 2)) < 3.1]
        paths = VGroup(*[VMobject(stroke_width=2, color=REFLOW_C).set_points_smoothly([ax1.c2p(*p) for p in P[:, i]]) for i in range(P.shape[1])])
        st = Text(f"median straightness {TOY['paths']['reflow']['straightness_median']:.2f}", font_size=16,
                  color=REFLOW_C).next_to(box1, DOWN, buff=0.15)
        self.at("straight")
        self.play(Create(paths), run_time=0.8)
        self.play(FadeIn(st), run_time=0.3)
        pts = np.clip(np.array(BY[1]["reflow"]["points"]), -2.35, 2.35)
        self.at("single")
        self.play(FadeIn(box2), FadeIn(t2), FadeIn(dots(ax2, DATA, DATA_C, r=0.015)), run_time=0.4)
        self.play(FadeIn(dots(ax2, pts, REFLOW_C, r=0.025)), run_time=0.5)
        self.end_section()

    # ------------------------------------------------------------------ 7. shapes
    def shapes(self):
        self.section(7)
        self.clear_stage()
        steps = [5, 10, 20, 50]
        ax = Axes(x_range=[0, 4, 1], y_range=[0, 1, 0.5], x_length=8.0, y_length=4.0, tips=False,
                  axis_config={"color": GREY_C, "include_ticks": False}).move_to([0, -0.4, 0])
        yl = VGroup(*[Text(f"{int(v * 100)}%", font=MONO, font_size=14, color=GREY_B).next_to(ax.c2p(0, v), LEFT, buff=0.1) for v in (0, 0.5, 1)])
        title = Text("recognized by the judge (500 samples, same starting noise)", font_size=20).to_edge(UP, buff=0.4)
        leg = VGroup(Text("flow matching (Euler)", font_size=18, color=FLOW_C),
                     Text("diffusion, fresh noise (η = 1)", font_size=18, color=DIFF_C)).arrange(RIGHT, buff=0.8).next_to(title, DOWN, buff=0.2)
        bars, labels, xls = [], [], VGroup()
        for i, k in enumerate(steps):
            pair = []
            for j, (val, col) in enumerate(((FLOW[k], FLOW_C), (DIFF[k], DIFF_C))):
                x0 = i + 0.2 + j * 0.32
                r = Polygon(ax.c2p(x0, 0), ax.c2p(x0 + 0.28, 0), ax.c2p(x0 + 0.28, val), ax.c2p(x0, val),
                            color=col, fill_color=col, fill_opacity=0.8, stroke_width=0)
                lab = Text(f"{val:.0%}", font=MONO, font_size=14, color=col).next_to(r, UP, buff=0.08)
                pair.append((r, lab))
            bars.append(pair)
            xls.add(Text(f"{k} steps", font=MONO, font_size=16, color=GREY_B).next_to(ax.c2p(i + 0.5, 0), DOWN, buff=0.15))
        self.at("shapes")
        self.play(FadeIn(title), FadeIn(ax), FadeIn(yl), FadeIn(xls), FadeIn(leg), run_time=0.6)
        for i, cue in enumerate(("five", "10", "20")):
            self.at(cue)
            items = [bars[i][0][0], bars[i][0][1], bars[i][1][0], bars[i][1][1]]
            if i == 2:
                items += [bars[3][j][k] for j in range(2) for k in range(2)]
            self.play(*[FadeIn(m) for m in items], run_time=0.6)
        self.end_section()

    # ------------------------------------------------------------------ 8. in practice
    def practice(self):
        self.section(8)
        self.clear_stage()
        t = Text("good pictures in far fewer steps, for the same training", font_size=26, color=YELLOW).move_to([0, 1.2, 0])
        m = VGroup(Text("trained with flow matching (rectified flow):", font_size=20, color=GREY_B),
                   Text("Stable Diffusion 3  ·  Flux", font_size=30, color=FLOW_C)).arrange(DOWN, buff=0.3).move_to([0, -0.6, 0])
        self.at("trade")
        self.play(FadeIn(t), run_time=0.5)
        self.at("recent")
        self.play(FadeIn(m), run_time=0.5)
        self.end_section()

    # ------------------------------------------------------------------ 9. code
    def code(self):
        self.section(9)
        self.clear_stage()
        code, hl = code_panel(CODE, font_size=18)
        code.move_to([0, 0.3, 0])
        self.play(FadeIn(code, shift=0.2 * UP), run_time=0.4)
        self.at("mix")
        self.play(Create(hl), run_time=0.2)
        self.play(highlight(hl, code, 1), run_time=0.3)
        self.at("predict")
        self.play(highlight(hl, code, 2), run_time=0.3)
        self.at("step")
        self.play(highlight(hl, code, 4), run_time=0.3)
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
