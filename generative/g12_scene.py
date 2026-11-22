"""How Generative Models Work: Images and Sound, episode 12 — CLIP: Matching Pictures and Words.

Render from the repo root:  ./render.sh generative g12
Numbers and pictures come from code/g12_clip (train_clip.py → results.json and out/*.png).
"""
import json

import numpy as np
from manim import *

from common import MONO, code_panel, finish, highlight, next_up_card
from gen_common import frame, tile
from intro import play_token_intro
from g12_script import NEXT, TAGLINE, TITLE
from voiced_scene import VoicedScene

SERIES_LABEL = "HOW GENERATIVE MODELS WORK  ·  EPISODE 12"
CODE_DIR = "code/g12_clip"
OUT = f"{CODE_DIR}/out"
R = json.load(open(f"{CODE_DIR}/results.json"))
H = {h["step"]: h for h in R["history"]}
IMG_C, TXT_C, POS_C, COL_C, SHP_C = BLUE_C, ORANGE, BLUE_C, ORANGE, GREEN_C
CODE = """img = F.normalize(image_encoder(pictures), dim=-1)     # (N, 64)
txt = F.normalize(text_encoder(captions), dim=-1)      # (N, 64)
logits = scale * img @ txt.T                           # the N × N table
target = torch.arange(N)                               # the diagonal
loss = (F.cross_entropy(logits, target)              # each picture → its caption
        + F.cross_entropy(logits.T, target)) / 2     # each caption → its picture"""


def row(path, n=8, h=1.0, r=0, buff=0.1):
    return Group(*[tile(path, i, h, row=r) for i in range(n)]).arrange(RIGHT, buff=buff)


def strip(color, n=16, seed=0, w=0.16, h=0.32):
    v = np.random.default_rng(seed).uniform(0.15, 1.0, n)
    return VGroup(*[Rectangle(width=w, height=h, stroke_width=0, fill_color=color, fill_opacity=o) for o in v]).arrange(RIGHT, buff=0.02)


class ClipVideo(VoicedScene):
    VIDEO = "g12"

    def construct(self):
        play_token_intro(self, TITLE, 12, TAGLINE, label=SERIES_LABEL)
        self.hook()          # 1
        self.encoders()      # 2
        self.table()         # 3
        self.ours()          # 4
        self.order()         # 5
        self.held_out()      # 6
        self.floor()         # 7
        self.search()        # 8
        self.real()          # 9
        self.code()          # 10
        self.outro()         # 11

    def clear_stage(self, *keep, run_time=0.6):
        self.play(*[FadeOut(m) for m in self.mobjects if m not in keep], run_time=run_time)

    # ------------------------------------------------------------------ 1. hook
    def hook(self):
        self.section(1)
        labels = VGroup(*[VGroup(RoundedRectangle(width=3.4, height=0.6, corner_radius=0.12, color=GREY_B),
                                 Text(t, font=MONO, font_size=18)) for t in ("shape = circle", "color = red", "position = top left")])
        for l in labels:
            l[1].move_to(l[0])
        labels.arrange(RIGHT, buff=0.4).move_to([0, 1.4, 0])
        sent = Text("“a little red ball up in the left corner”", font_size=28, color=YELLOW).move_to([0, -0.4, 0])
        q = Text("which words go with which pictures?", font_size=24, color=GREY_B).move_to([0, -1.9, 0])
        self.at("labels")
        self.play(LaggedStart(*[FadeIn(l) for l in labels], lag_ratio=0.2), run_time=0.8)
        self.at("sentences")
        self.play(FadeIn(sent), run_time=0.5)
        self.at("know")
        self.play(FadeIn(q), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 2. two encoders
    def encoders(self):
        self.section(2)
        self.clear_stage()
        pic = tile(f"{OUT}/table_pictures.png", 7, 1.2).move_to([-5.2, 1.3, 0])
        cap = Text(R["table_captions"][7], font_size=18).move_to([-4.9, -1.3, 0])
        enc = lambda name, col, y: VGroup(RoundedRectangle(width=2.2, height=0.8, corner_radius=0.12, color=col, fill_color=col, fill_opacity=0.15),
                                          Text(name, font_size=18, color=col)).move_to([-1.3, y, 0])
        ie, te = enc("image encoder", IMG_C, 1.3), enc("text encoder", TXT_C, -1.3)
        for e in (ie, te):
            e[1].move_to(e[0])
        si, st = strip(IMG_C, seed=1).move_to([2.6, 1.3, 0]), strip(TXT_C, seed=2).move_to([2.6, -1.3, 0])
        li = Text("64 numbers", font_size=16, color=GREY_B).next_to(si, UP, buff=0.1)
        lt = Text("64 numbers", font_size=16, color=GREY_B).next_to(st, DOWN, buff=0.1)
        self.at("picture")
        self.play(FadeIn(pic), FadeIn(frame(pic)), FadeIn(ie), GrowArrow(Arrow(pic.get_right(), ie.get_left(), buff=0.1, color=GREY_B)), run_time=0.5)
        self.at("64")
        self.play(GrowArrow(Arrow(ie.get_right(), si.get_left(), buff=0.1, color=GREY_B)), FadeIn(si), FadeIn(li), run_time=0.5)
        self.at("caption")
        self.play(FadeIn(cap), FadeIn(te), GrowArrow(Arrow(cap.get_right(), te.get_left(), buff=0.1, color=GREY_B)), run_time=0.5)
        self.play(GrowArrow(Arrow(te.get_right(), st.get_left(), buff=0.1, color=GREY_B)), FadeIn(st), FadeIn(lt), run_time=0.5)
        o = np.array([5.6, 0, 0])
        a1 = Arrow(o, o + np.array([1.2, 0.55, 0]), buff=0, color=IMG_C)
        a2 = Arrow(o, o + np.array([1.25, 0.35, 0]), buff=0, color=TXT_C)
        same = Text("point the\nsame way", font_size=18, color=YELLOW, line_spacing=0.9).next_to(VGroup(a1, a2), DOWN, buff=0.25)
        self.at("same")
        self.play(GrowArrow(a1), GrowArrow(a2), FadeIn(same), run_time=0.6)
        self.end_section()

    # ------------------------------------------------------------------ 3. the table
    def table(self):
        self.section(3)
        self.clear_stage()
        T = np.array(R["table"])                   # T[i][j]: picture i · caption j
        cell = 0.62
        pics = Group(*[tile(f"{OUT}/table_pictures.png", i, cell - 0.06) for i in range(8)]).arrange(RIGHT, buff=0.06)
        pics.move_to([2.2, 2.6, 0])
        x0 = pics[0].get_x()
        y0 = 1.9
        caps = VGroup(*[Text(R["table_captions"][j], font_size=30).scale(0.55).move_to([0, y0 - j * cell, 0], aligned_edge=RIGHT)
                        for j in range(8)])
        for c in caps:
            c.align_to([x0 - cell / 2 - 0.2, 0, 0], RIGHT)
        cells = VGroup()
        for j in range(8):
            for i in range(8):
                v = T[i][j]
                col = interpolate_color(GREY_E, YELLOW, max(0.0, v))
                sq = Square(side_length=cell - 0.04, stroke_width=0, fill_color=col, fill_opacity=1).move_to([x0 + i * cell, y0 - j * cell, 0])
                cells.add(sq)
        self.at("batch")
        self.play(FadeIn(pics), FadeIn(caps), run_time=0.6)
        self.at("table")
        self.play(LaggedStart(*[FadeIn(c) for c in cells], lag_ratio=0.01), run_time=1.0)
        diag = VGroup(*[SurroundingRectangle(cells[j * 8 + j], color=GREEN, buff=0.02, stroke_width=3) for j in range(8)])
        self.at("diagonal")
        self.play(Create(diag), run_time=0.6)
        up = Text("push up", font_size=20, color=GREEN).move_to([5.9, 0.6, 0])
        down = Text("push the\nrest down", font_size=20, color=GREY_B, line_spacing=0.9).next_to(up, DOWN, buff=0.3)
        self.at("push")
        self.play(FadeIn(up), run_time=0.3)
        self.at("everything")
        self.play(FadeIn(down), run_time=0.3)
        n = Text("each picture picks its caption (rows) · each caption picks its picture (columns)", font_size=17,
                 color=GREY_B).to_edge(DOWN, buff=0.35)
        self.at("pick")
        self.play(FadeIn(n), run_time=0.4)
        self.tab = (cells, caps, pics)
        self.end_section()

    # ------------------------------------------------------------------ 4. ours
    def ours(self):
        self.section(4)
        self.clear_stage()
        p = R["parameters"]
        a = VGroup(Text("image encoder", font_size=22, color=IMG_C), Text("small convolutional network", font_size=18),
                   Text(f"{p['image']:,} parameters", font=MONO, font_size=16, color=GREY_B)).arrange(DOWN, buff=0.15).move_to([-3.3, 1.3, 0])
        b = VGroup(Text("text encoder", font_size=22, color=TXT_C), Text("2-layer transformer over the words", font_size=18),
                   Text(f"{p['text']:,} parameters", font=MONO, font_size=16, color=GREY_B)).arrange(DOWN, buff=0.15).move_to([3.3, 1.3, 0])
        words = [w for w in R["vocabulary"] if w != "<pad>"]
        chips = VGroup(*[Text(w, font=MONO, font_size=18, color=TXT_C) for w in words]).arrange_in_grid(rows=3, buff=(0.45, 0.3))
        chips.move_to([0, -1.4, 0])
        vl = Text(f"the whole vocabulary: {len(words)} words", font_size=20, color=GREY_B).next_to(chips, UP, buff=0.35)
        self.at("tiny")
        self.play(FadeIn(a), run_time=0.4)
        self.at("transformer")
        self.play(FadeIn(b), run_time=0.4)
        self.at("vocabulary")
        self.play(FadeIn(vl), LaggedStart(*[FadeIn(c) for c in chips], lag_ratio=0.05), run_time=0.9)
        self.end_section()

    # ------------------------------------------------------------------ 5. learning order
    def order(self):
        self.section(5)
        self.clear_stage()
        steps = [h["step"] for h in R["history"] if h["step"] <= 250]
        ax = Axes(x_range=[0, 250, 50], y_range=[0, 1, 0.5], x_length=8.4, y_length=3.8, tips=False,
                  axis_config={"color": GREY_C}).move_to([-0.8, -0.4, 0])
        xl = VGroup(*[Text(f"{s}", font=MONO, font_size=14, color=GREY_B).next_to(ax.c2p(s, 0), DOWN, buff=0.1) for s in (0, 50, 100, 150, 200, 250)])
        yl = VGroup(*[Text(f"{int(v * 100)}%", font=MONO, font_size=14, color=GREY_B).next_to(ax.c2p(0, v), LEFT, buff=0.1) for v in (0, 0.5, 1)])
        xt = Text("training step", font_size=16, color=GREY_B).next_to(xl, DOWN, buff=0.1)
        title = Text("read a picture: the closest of all 135 captions", font_size=22).to_edge(UP, buff=0.45)
        line = lambda f, col, w=4: VMobject(color=col, stroke_width=w).set_points_as_corners([ax.c2p(s, f(H[s])) for s in steps])
        curves = {"position": line(lambda h: h["trained"]["position"], POS_C), "color": line(lambda h: h["trained"]["color"], COL_C),
                  "shape": line(lambda h: h["trained"]["shape"], SHP_C), "exact": line(lambda h: h["trained"]["exact"], WHITE, 6)}
        leg = VGroup(*[Text(k if k != "exact" else "whole caption", font_size=18, color=c.get_color()) for k, c in curves.items()]
                     ).arrange(DOWN, aligned_edge=LEFT, buff=0.2).move_to([5.3, -0.2, 0])
        self.at("closest")
        self.play(FadeIn(title), FadeIn(ax), FadeIn(xl), FadeIn(yl), FadeIn(xt), run_time=0.5)
        b = Text(f"{H[0]['trained']['exact']:.1%}", font=MONO, font_size=16).next_to(ax.c2p(0, 0), UR, buff=0.1)
        self.at("before")
        self.play(FadeIn(b), run_time=0.3)
        self.at("10")
        self.play(Create(curves["position"]), FadeIn(leg[0]), run_time=0.6)
        self.at("color")
        self.play(Create(curves["color"]), FadeIn(leg[1]), run_time=0.6)
        self.at("shape")
        self.play(Create(curves["shape"]), FadeIn(leg[2]), run_time=0.6)
        self.at("caption")
        self.play(Create(curves["exact"]), FadeIn(leg[3]), run_time=0.6)
        self.end_section()

    # ------------------------------------------------------------------ 6. held out
    def held_out(self):
        self.section(6)
        self.clear_stage()
        head = Text("8 combinations never seen in training, as pictures or as captions", font_size=22).to_edge(UP, buff=0.5)
        caps = VGroup(*[Text(c, font_size=18) for c in R["held_out"]]).arrange_in_grid(rows=4, cols=2, buff=(0.8, 0.3), col_alignments="ll").move_to([0, 0.2, 0])
        ticks = VGroup(*[Text("✓", font_size=20, color=GREEN).next_to(c, LEFT, buff=0.15) for c in caps])
        res = Text(f"read correctly: {H[R['steps']]['held_out']['exact']:.0%} of 200 pictures", font_size=24, color=GREEN).move_to([0, -2.2, 0])
        self.at("eight")
        self.play(FadeIn(head), run_time=0.4)
        self.at("purple")
        self.play(LaggedStart(*[FadeIn(c) for c in caps], lag_ratio=0.08), run_time=0.8)
        self.at("correctly")
        self.play(FadeIn(ticks), FadeIn(res), run_time=0.5)
        k = Text("each word learned on its own, then combined", font_size=20, color=YELLOW).next_to(res, DOWN, buff=0.3)
        self.at("combines")
        self.play(FadeIn(k), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 7. loss floor
    def floor(self):
        self.section(7)
        self.clear_stage()
        L = R["loss_curve"]                      # mean of every 50 steps
        ax = Axes(x_range=[0, R["steps"], 1000], y_range=[0, 3, 1], x_length=7.6, y_length=3.6, tips=False,
                  axis_config={"color": GREY_C}).move_to([-1.2, 0.2, 0])
        xl = VGroup(*[Text(f"{s:,}", font=MONO, font_size=14, color=GREY_B).next_to(ax.c2p(s, 0), DOWN, buff=0.1) for s in (0, 2000, 4000)])
        yl = VGroup(*[Text(f"{v}", font=MONO, font_size=14, color=GREY_B).next_to(ax.c2p(0, v), LEFT, buff=0.1) for v in (0, 1, 2, 3)])
        title = Text("training loss", font_size=20).next_to(ax, UP, buff=0.2)
        curve = VMobject(color=WHITE, stroke_width=3).set_points_as_corners([ax.c2p(i * 50 + 25, min(v, 3)) for i, v in enumerate(L)])
        self.at("loss")
        self.play(FadeIn(ax), FadeIn(xl), FadeIn(yl), FadeIn(title), Create(curve), run_time=0.9)
        best = 0.574
        dl = DashedLine(ax.c2p(0, best), ax.c2p(R["steps"], best), color=YELLOW)
        lab = Text(f"{L[-1]:.2f}", font=MONO, font_size=18).next_to(ax.c2p(R["steps"], L[-1]), RIGHT, buff=0.15)
        self.at("57")
        self.play(FadeIn(lab), run_time=0.3)
        ex = VGroup(Text("batch of 128 from 127 captions:", font_size=18, color=GREY_B),
                    Text("“a yellow square at the top” ×2", font_size=18),
                    Text("→ two identical columns", font_size=18, color=GREY_B)).arrange(DOWN, aligned_edge=LEFT, buff=0.15).move_to([4.6, 1.4, 0])
        self.at("twice")
        self.play(FadeIn(ex), run_time=0.5)
        bl = Text("best possible: 0.574", font=MONO, font_size=18, color=YELLOW).next_to(dl, UP, buff=0.1).shift(LEFT * 1.5)
        self.at("best")
        self.play(Create(dl), FadeIn(bl), run_time=0.6)
        self.end_section()

    # ------------------------------------------------------------------ 8. search
    def search(self):
        self.section(8)
        self.clear_stage()
        a = row(f"{OUT}/search_trained.png", h=1.1).move_to([0, 1.0, 0])
        b = row(f"{OUT}/search_held_out.png", h=1.1).move_to([0, -1.6, 0])
        qa = Text("search: “a green square at the bottom left”", font_size=20, color=TXT_C).next_to(a, UP, buff=0.2)
        qb = Text("search: “a purple triangle at the top left”  (never trained on)", font_size=20, color=TXT_C).next_to(b, UP, buff=0.2)
        self.at("caption")
        self.play(FadeIn(qa), run_time=0.4)
        self.at("rank")
        self.play(FadeIn(a), run_time=0.5)
        self.at("never")
        self.play(FadeIn(qb), FadeIn(b), run_time=0.5)
        self.end_section()

    # ------------------------------------------------------------------ 9. real CLIP
    def real(self):
        self.section(9)
        self.clear_stage()
        a = VGroup(Text("OpenAI's CLIP (Radford et al., 2021)", font_size=24, color=TXT_C),
                   Text("400 million pictures and captions from the web", font_size=20)).arrange(DOWN, buff=0.2).move_to([0, 1.3, 0])
        b = VGroup(Text("its text encoder guides Stable Diffusion", font_size=24, color=YELLOW),
                   Text("words → numbers that already know what things look like", font_size=20, color=GREY_B)).arrange(DOWN, buff=0.2).move_to([0, -1.0, 0])
        self.at("openais", "openai")
        self.play(FadeIn(a[0]), run_time=0.4)
        self.at("400")
        self.play(FadeIn(a[1]), run_time=0.4)
        self.at("text")
        self.play(FadeIn(b[1]), run_time=0.4)
        self.at("stable")
        self.play(FadeIn(b[0]), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 10. code
    def code(self):
        self.section(10)
        self.clear_stage()
        code, hl = code_panel(CODE, font_size=18)
        code.move_to([0, 0.3, 0])
        self.play(FadeIn(code, shift=0.2 * UP), run_time=0.4)
        self.at("normalize")
        self.play(Create(hl), run_time=0.2)
        self.play(highlight(hl, code, 0), run_time=0.3)
        self.at("multiply")
        self.play(highlight(hl, code, 2), run_time=0.3)
        self.at("cross")
        self.play(highlight(hl, code, 4), run_time=0.3)
        self.play(highlight(hl, code, 5), run_time=0.3)
        self.end_section()

    # ------------------------------------------------------------------ 11. outro
    def outro(self):
        self.section(11)
        self.clear_stage(run_time=0.4)
        card_ = next_up_card(NEXT)
        self.at("next")
        self.play(FadeIn(card_, shift=0.2 * UP), run_time=0.6)
        self.end_section()
        finish(self, hold=1.5)
