"""How Generative Models Work: Images and Sound, episode 2 — Autoencoders.

Render from the repo root:  ./render.sh generative g02
Numbers and pictures come from code/g02_autoencoders (train_autoencoder.py → results.json and out/*.png).
"""
import json

import numpy as np
from manim import *

from common import MONO, code_panel, finish, highlight, next_up_card
from gen_common import dots, frame, pic, plane, tile
from intro import play_token_intro
from g02_script import NEXT, TAGLINE, TITLE
from voiced_scene import VoicedScene

SERIES_LABEL = "HOW GENERATIVE MODELS WORK  ·  EPISODE 2"
CODE_DIR = "code/g02_autoencoders"
OUT = f"{CODE_DIR}/out"
R = json.load(open(f"{CODE_DIR}/results.json"))
RUNS = R["runs"]
ENC_C, DEC_C, CODE_C, GEN_C = BLUE_C, GREEN_C, YELLOW, ORANGE
POS_NAMES = ["top left", "top", "top right", "left", "center", "right", "bottom left", "bottom", "bottom right"]
POS_COLORS = [RED_C, ORANGE, YELLOW, GREEN_C, TEAL_C, BLUE_C, PURPLE_B, PINK, GREY_B]
CODE = """code = encoder(x)                    # 3,072 numbers → 8
x_hat = decoder(code)                # 8 → 3,072
loss = ((x_hat - x) ** 2).mean()     # match the input
opt.zero_grad(); loss.backward(); opt.step()"""


def funnel(w_big, w_small, h, color, label, narrowing=True):
    a, b = (w_big, w_small) if narrowing else (w_small, w_big)
    poly = Polygon([-h / 2, a / 2, 0], [h / 2, b / 2, 0], [h / 2, -b / 2, 0], [-h / 2, -a / 2, 0],
                   color=color, fill_opacity=0.18, stroke_width=3)
    return VGroup(poly, Text(label, font_size=22, color=color).move_to(poly))


def code_bar(values, color=CODE_C, w=0.32, scale=0.09):
    bars = VGroup()
    for v in values:
        h = max(abs(v) * scale, 0.03)
        r = Rectangle(width=w * 0.8, height=h, stroke_width=0, fill_color=color, fill_opacity=0.9)
        bars.add(r.move_to([0, h / 2 if v >= 0 else -h / 2, 0]))
    bars.arrange(RIGHT, buff=w * 0.2, aligned_edge=ORIGIN)
    for r, v in zip(bars, values):
        r.shift(UP * (r.height / 2 if v >= 0 else -r.height / 2) - UP * r.get_center()[1])
    base = Line(bars.get_left() + LEFT * 0.08, bars.get_right() + RIGHT * 0.08, color=GREY_C, stroke_width=1.5)
    return VGroup(base, bars)


class AutoencoderVideo(VoicedScene):
    VIDEO = "g02"

    def construct(self):
        play_token_intro(self, TITLE, 2, TAGLINE, label=SERIES_LABEL)
        self.hook()        # 1
        self.idea()        # 2
        self.network()     # 3
        self.sizes()       # 4
        self.why()         # 5
        self.random()      # 6
        self.halfway()     # 7
        self.islands()     # 8
        self.code()        # 9
        self.outro()       # 10

    def clear_stage(self, *keep, run_time=0.6):
        self.play(*[FadeOut(m) for m in self.mobjects if m not in keep], run_time=run_time)

    # ------------------------------------------------------------------ 1. hook
    def hook(self):
        self.section(1)
        mush = Group(*[tile("code/g01_what_it_means_to_generate/out/pixel_samples.png", i, 1.6) for i in range(3)]).arrange(RIGHT, buff=0.2)
        mush.move_to([0, 1.4, 0])
        lab = Text("episode 1: pixels sampled one by one", font_size=22, color=GREY_B).next_to(mush, UP, buff=0.25)
        self.at("pixels")
        self.play(FadeIn(mush), FadeIn(VGroup(*[frame(m) for m in mush])), FadeIn(lab), run_time=0.6)
        x = Text("never a shape", font_size=24, color=RED_B).next_to(mush, DOWN, buff=0.25)
        self.at("never")
        self.play(FadeIn(x), run_time=0.3)
        need = Text("learn how pixels depend on each other", font_size=26, color=YELLOW).move_to([0, -1.2, 0])
        self.at("learn")
        self.play(FadeIn(need), run_time=0.4)
        neck = VGroup(funnel(1.6, 0.4, 1.6, ENC_C, ""), funnel(1.6, 0.4, 1.6, DEC_C, "", narrowing=False)).arrange(RIGHT, buff=0).move_to([0, -2.6, 0])
        nl = Text("a bottleneck", font_size=22, color=CODE_C).next_to(neck, RIGHT, buff=0.3)
        self.at("bottleneck")
        self.play(FadeIn(neck), FadeIn(nl), run_time=0.6)
        self.end_section()

    # ------------------------------------------------------------------ 2. the idea
    def idea(self):
        self.section(2)
        self.clear_stage()
        inp = tile(f"{OUT}/originals.png", 0, 2.0).move_to([-5.3, 0.6, 0])
        enc = funnel(2.2, 0.8, 2.0, ENC_C, "encoder").move_to([-2.7, 0.6, 0])
        code = code_bar([round(v, 1) for v in RUNS["8"]["first_codes"][0]]).move_to([0, 0.6, 0])
        dec = funnel(2.2, 0.8, 2.0, DEC_C, "decoder", narrowing=False).move_to([2.7, 0.6, 0])
        out = tile(f"{OUT}/recon_8.png", 0, 2.0).move_to([5.3, 0.6, 0])
        li = Text("3,072 numbers", font_size=20, color=GREY_B).next_to(inp, DOWN, buff=0.2)
        lc = Text("code: 8 numbers", font_size=20, color=CODE_C).next_to(code, DOWN, buff=0.35)
        lo = Text("3,072 numbers", font_size=20, color=GREY_B).next_to(out, DOWN, buff=0.2)
        self.at("autoencoder")
        self.play(FadeIn(inp), FadeIn(frame(inp)), FadeIn(li), run_time=0.4)
        self.at("encoder")
        self.play(FadeIn(enc), run_time=0.4)
        self.at("short", "code")
        self.play(FadeIn(code, lag_ratio=0.1), FadeIn(lc), run_time=0.6)
        self.at("decoder")
        self.play(FadeIn(dec), run_time=0.4)
        self.at("picture")
        self.play(FadeIn(out), FadeIn(frame(out)), FadeIn(lo), run_time=0.4)
        arc = CurvedArrow(out.get_bottom() + DOWN * 0.7, inp.get_bottom() + DOWN * 0.7, angle=-PI / 6, color=GEN_C)
        al = Text("loss: output should match input", font_size=22, color=GEN_C).next_to(arc, DOWN, buff=0.15)
        self.at("output")
        self.play(Create(arc), FadeIn(al), run_time=0.7)
        nl = Text("no labels, no captions", font_size=22, color=GREY_B).next_to(al, DOWN, buff=0.2)
        self.at("labels")
        self.play(FadeIn(nl), run_time=0.3)
        self.end_section()

    # ------------------------------------------------------------------ 3. the network
    def network(self):
        self.section(3)
        self.clear_stage()
        shapes = [("3×32×32", 2.0), ("32×16×16", 1.5), ("64×8×8", 1.1), ("128×4×4", 0.75)]
        blocks = VGroup()
        for name, s in shapes:
            sq = Square(side_length=s, color=ENC_C, fill_opacity=0.15, stroke_width=2)
            blocks.add(VGroup(sq, Text(name, font=MONO, font_size=15).next_to(sq, DOWN, buff=0.15)))
        code = VGroup(Rectangle(width=0.3, height=1.3, color=CODE_C, fill_opacity=0.4), Text("8", font=MONO, font_size=18, color=CODE_C))
        code[1].next_to(code[0], DOWN, buff=0.15)
        row = VGroup(*blocks, code).arrange(RIGHT, buff=0.75, aligned_edge=ORIGIN).move_to([0, 0.9, 0])
        for b in row:
            b.shift(UP * (row.get_center()[1] - b[0].get_center()[1]))
        arrows = VGroup(*[Arrow(row[i][0].get_right(), row[i + 1][0].get_left(), buff=0.1, color=GREY_B) for i in range(4)])
        al = VGroup(*[Text(t, font_size=14, color=GREY_B).next_to(a, UP, buff=0.05) for t, a in zip(["conv", "conv", "conv", "linear"], arrows)])
        self.at("three")
        self.play(FadeIn(row[0]), run_time=0.3)
        for k, cue in enumerate(["16", "8", "4"]):
            self.at(cue)
            self.play(GrowArrow(arrows[k]), FadeIn(al[k]), FadeIn(row[k + 1]), run_time=0.4)
        self.at("linear")
        self.play(GrowArrow(arrows[3]), FadeIn(al[3]), FadeIn(row[4]), run_time=0.4)
        mirror = Text("decoder: the same, in reverse (transposed convolutions)", font_size=22, color=DEC_C).move_to([0, -1.5, 0])
        self.at("mirrors")
        self.play(FadeIn(mirror), run_time=0.4)
        facts = Text(f"{RUNS['8']['parameters']:,} parameters  ·  {RUNS['8']['seconds']:.0f} s of training (15 passes over 20,000 pictures, CPU)",
                     font_size=20, color=GREY_B).move_to([0, -2.4, 0])
        self.at("370", "parameters")
        self.play(FadeIn(facts), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 4. code size
    def sizes(self):
        self.section(4)
        self.clear_stage()
        rows = [("original", f"{OUT}/originals.png", None), ("2 numbers", f"{OUT}/recon_2.png", "2"),
                ("8 numbers", f"{OUT}/recon_8.png", "8"), ("32 numbers", f"{OUT}/recon_32.png", "32")]
        g, labels, stats = [], VGroup(), VGroup()
        y = 2.5
        for name, path, key in rows:
            r = Group(*[tile(path, i, 1.05) for i in range(6)]).arrange(RIGHT, buff=0.1).move_to([0.3, y, 0])
            g.append(r)
            labels.add(Text(name, font_size=20, color=CODE_C if key else GREY_B).next_to(r, LEFT, buff=0.3))
            if key:
                stats.add(Text(f"{RUNS[key]['wrong_pixel_share']:.1%} wrong", font=MONO, font_size=18, color=GREY_A).next_to(r, RIGHT, buff=0.3))
            y -= 1.35
        self.play(FadeIn(g[0]), FadeIn(labels[0]), run_time=0.4)
        cues = ["two", "eight", "32"]
        for k, cue in enumerate(cues):
            self.at(cue)
            self.play(FadeIn(g[k + 1]), FadeIn(labels[k + 1]), run_time=0.4)
            if k == 0:
                self.at("four")
            self.play(FadeIn(stats[k]), run_time=0.3)
        n = Text("3,072 / 32 = 96 times smaller", font_size=22, color=YELLOW).to_edge(DOWN, buff=0.35)
        self.at("96")
        self.play(FadeIn(n), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 5. why it works
    def why(self):
        self.section(5)
        self.clear_stage()
        h = Text("Shapes pictures vary in only a few ways", font_size=28).to_edge(UP, buff=0.6)
        self.at("vary")
        self.play(FadeIn(h), run_time=0.4)
        facts = VGroup(*[Text(t, font_size=24, color=c) for t, c in
                         [("shape (3)", BLUE_C), ("color (5)", RED_C), ("position (9)", GREEN_C), ("size", PURPLE_B)]]).arrange(RIGHT, buff=0.7).move_to([0, 1.5, 0])
        for f, cue in zip(facts, ["shape", "color", "position", "size"]):
            self.at(cue)
            self.play(FadeIn(f, shift=0.1 * DOWN), run_time=0.3)
        code = code_bar([round(v, 1) for v in RUNS["8"]["first_codes"][0]], w=0.5, scale=0.12).move_to([1.2, -0.4, 0])
        im = tile(f"{OUT}/originals.png", 0, 1.8).move_to([-3.6, -0.4, 0])
        arr = Arrow(im.get_right(), code.get_left(), buff=0.3, color=ENC_C)
        cl = Text("its 8-number code", font_size=20, color=CODE_C).next_to(code, DOWN, buff=0.3)
        self.at("eight")
        self.play(FadeIn(im), FadeIn(frame(im)), GrowArrow(arr), FadeIn(code), FadeIn(cl), run_time=0.7)
        note = Text("found by training, not told", font_size=24, color=YELLOW).to_edge(DOWN, buff=0.3)
        self.at("nobody")
        self.play(FadeIn(note), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 6. random codes
    def random(self):
        self.section(6)
        self.clear_stage()
        h = Text("decode random codes (same average and spread as real codes)", font_size=24).to_edge(UP, buff=0.5)
        self.at("pick")
        self.play(FadeIn(h), run_time=0.4)
        rows = [("2 numbers", "2"), ("8 numbers", "8"), ("32 numbers", "32")]
        y = 1.75
        groups = []
        for (name, key), cue in zip(rows, ["two", "8", "32"]):
            r = Group(*[tile(f"{OUT}/random_{key}.png", i, 1.1) for i in range(6)]).arrange(RIGHT, buff=0.1).move_to([0.2, y, 0])
            lab = Text(name, font_size=20, color=CODE_C).next_to(r, LEFT, buff=0.3)
            groups.append((r, lab))
            y -= 1.45
        self.at("two")
        self.play(FadeIn(groups[0][0]), FadeIn(groups[0][1]), run_time=0.5)
        pl = Text("plausible blobs", font_size=18, color=GREEN_B).next_to(groups[0][0], RIGHT, buff=0.2)
        self.play(FadeIn(pl), run_time=0.3)
        self.at("8")
        self.play(FadeIn(groups[1][0]), FadeIn(groups[1][1]), FadeIn(groups[2][0]), FadeIn(groups[2][1]), run_time=0.6)
        ml = Text("mush", font_size=18, color=RED_B).next_to(Group(groups[1][0], groups[2][0]), RIGHT, buff=0.2)
        self.at("mush")
        self.play(FadeIn(ml), run_time=0.3)
        r8, r32 = RUNS["8"], RUNS["32"]
        t = VGroup(Text("squared error to the nearest training picture (8 numbers):", font_size=20, color=GREY_B),
                   Text(f"random code {r8['random_nearest_train_mse']:.4f}   rebuilt picture {r8['test_mse']:.4f}   "
                        f"→ {r8['random_nearest_train_mse'] / r8['test_mse']:.1f}× further", font=MONO, font_size=19, color=YELLOW)
                   ).arrange(DOWN, buff=0.12).to_edge(DOWN, buff=0.3)
        self.at("further")
        self.play(FadeIn(t), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 7. halfway
    def halfway(self):
        self.section(7)
        self.clear_stage()
        a = tile(f"{OUT}/halfway_8.png", 0, 2.4, row=0)
        m = tile(f"{OUT}/halfway_8.png", 0, 2.4, row=1)
        b = tile(f"{OUT}/halfway_8.png", 0, 2.4, row=2)
        row = Group(a, m, b).arrange(RIGHT, buff=1.2).move_to([0, 0.5, 0])
        la = Text("code A", font_size=22, color=CODE_C).next_to(a, DOWN, buff=0.25)
        lm = Text("(A + B) / 2", font_size=22, color=GEN_C).next_to(m, DOWN, buff=0.25)
        lb = Text("code B", font_size=22, color=CODE_C).next_to(b, DOWN, buff=0.25)
        self.at("yellow")
        self.play(FadeIn(a), FadeIn(frame(a)), FadeIn(la), run_time=0.4)
        self.at("purple")
        self.play(FadeIn(b), FadeIn(frame(b)), FadeIn(lb), run_time=0.4)
        self.at("decoder")
        self.play(FadeIn(m), FadeIn(frame(m, GEN_C)), FadeIn(lm), run_time=0.5)
        n = Text("faint smudges in both places, in colors neither picture has", font_size=22, color=YELLOW).to_edge(DOWN, buff=0.8)
        self.at("green")
        self.play(FadeIn(n), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 8. islands
    def islands(self):
        self.section(8)
        self.clear_stage()
        g, ax = plane([-1.6, 0.0, 0], 6.2, -14, 11)
        codes = np.array(R["codes_2d"]["codes"])
        pos = np.array(R["codes_2d"]["labels"])[:, 2]
        cap = Text("2,000 test pictures, 2-number codes", font_size=20, color=GREY_B).next_to(g, UP, buff=0.15)
        self.at("point")
        self.play(FadeIn(g), FadeIn(cap), run_time=0.4)
        pts = VGroup(*[Dot(ax.c2p(*c), radius=0.025, color=POS_COLORS[p]) for c, p in zip(codes, pos)])
        self.play(FadeIn(pts, lag_ratio=0.0005), run_time=1.2)
        legend = VGroup(*[VGroup(Dot(radius=0.09, color=c), Text(n, font_size=19)).arrange(RIGHT, buff=0.12) for n, c in zip(POS_NAMES, POS_COLORS)])
        legend.arrange(DOWN, aligned_edge=LEFT, buff=0.14).move_to([4.6, 0.6, 0])
        lt = Text("color = position", font_size=18, color=GREY_B).next_to(legend, UP, buff=0.2)
        self.at("islands")
        self.play(FadeIn(legend), FadeIn(lt), run_time=0.5)
        gx, gy = np.meshgrid(np.linspace(-8, 6, 57), np.linspace(-8, 7, 61))       # emptiest spot inside the cloud
        cand = np.stack([gx.ravel(), gy.ravel()], 1)
        far = np.sqrt(((cand[:, None] - codes[None]) ** 2).sum(-1)).min(1)
        gap = ax.c2p(*cand[far.argmax()])
        q = VGroup(Circle(radius=0.35, color=RED_B, stroke_width=3).move_to(gap), Text("?", font_size=30, color=RED_B).move_to(gap))
        self.at("between")
        self.play(Create(q[0]), FadeIn(q[1]), run_time=0.5)
        n = Text("no training picture\never landed here", font_size=20, color=RED_B, line_spacing=0.9).next_to(legend, DOWN, buff=0.4)
        self.at("nothing")
        self.play(FadeIn(n), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 9. code
    def code(self):
        self.section(9)
        self.clear_stage()
        code, hl = code_panel(CODE, font_size=22)
        code.move_to([0, 0.4, 0])
        self.play(FadeIn(code, shift=0.2 * UP), run_time=0.5)
        hl.match_y(code.line_numbers[2])
        self.at("squared")
        self.play(Create(hl), run_time=0.3)
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
