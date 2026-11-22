"""How Generative Models Work: Images and Sound, episode 10 — Conditioning: Telling the Model What to Draw.

Render from the repo root:  ./render.sh generative g10
Numbers and pictures come from code/g10_conditioning (evaluate_cond.py → results_cond.json and out/*.png).
"""
import json

import numpy as np
from manim import *

from common import MONO, code_panel, finish, highlight, next_up_card
from gen_common import frame, tile
from intro import play_token_intro
from g10_script import NEXT, TAGLINE, TITLE
from voiced_scene import VoicedScene

SERIES_LABEL = "HOW GENERATIVE MODELS WORK  ·  EPISODE 10"
CODE_DIR = "code/g10_conditioning"
OUT = f"{CODE_DIR}/out"
R = json.load(open(f"{CODE_DIR}/results_cond.json"))
SHP_C, COL_C, POS_C, T_C = GREEN_C, ORANGE, BLUE_C, PURPLE_B
CODE = """# one table of vectors per label (+1 entry for "not said")
labels = [nn.Embedding(10, t_dim) for _ in range(3)]

temb = t_mlp(timestep_embedding(t))              # the time, as before
temb = temb + sum(e(y[:, k]) for k, e in enumerate(labels))

# training: hide labels at random
y[torch.rand(len(y)) < 0.1] = NULL               # 10%: nothing said
y[torch.rand(y.shape) < 0.1] = NULL              # each label: 10%"""


def row(path, n=8, h=1.0, r=0, buff=0.1):
    return Group(*[tile(path, i, h, row=r) for i in range(n)]).arrange(RIGHT, buff=buff)


def strip(color, n=12, seed=0, w=0.16, h=0.32):
    v = np.random.default_rng(seed).uniform(0.15, 1.0, n)
    return VGroup(*[Rectangle(width=w, height=h, stroke_width=0, fill_color=color, fill_opacity=o) for o in v]).arrange(RIGHT, buff=0.02)


def chip(text, color, w=2.6):
    box = RoundedRectangle(width=w, height=0.55, corner_radius=0.12, color=color, fill_color=color, fill_opacity=0.12)
    return VGroup(box, Text(text, font=MONO, font_size=18, color=color).move_to(box))


def small(text, color=GREY_B, size=30, scale=0.55):
    return Text(text, font_size=size, color=color).scale(scale)


def pct(v):
    return f"{v * 100:.1f}%"


class ConditioningVideo(VoicedScene):
    VIDEO = "g10"

    def construct(self):
        play_token_intro(self, TITLE, 10, TAGLINE, label=SERIES_LABEL)
        self.hook()            # 1
        self.extra_input()     # 2
        self.training()        # 3
        self.dropout()         # 4
        self.results()         # 5
        self.held_out()        # 6
        self.partial()         # 7
        self.unconditional()   # 8
        self.real()            # 9
        self.code()            # 10
        self.outro()           # 11

    def clear_stage(self, *keep, run_time=0.6):
        self.play(*[FadeOut(m) for m in self.mobjects if m not in keep], run_time=run_time)

    # ------------------------------------------------------------------ 1. hook
    def hook(self):
        self.section(1)
        r = row(f"{OUT}/unconditional.png", h=1.25).move_to([0, 1.2, 0])
        cap = Text("no say in what comes out", font_size=22, color=GREY_B).next_to(r, DOWN, buff=0.3)
        want = Text("“a red triangle in the center”", font_size=30, color=YELLOW).move_to([0, -1.7, 0])
        self.at("random")
        self.play(LaggedStart(*[FadeIn(t) for t in r], lag_ratio=0.1), run_time=1.0)
        self.at("choose")
        self.play(FadeIn(cap), run_time=0.4)
        self.at("red")
        self.play(FadeIn(want, shift=0.2 * UP), run_time=0.5)
        self.end_section()

    # ------------------------------------------------------------------ 2. the extra input
    def extra_input(self):
        self.section(2)
        self.clear_stage()
        chips = VGroup(chip("shape = triangle", SHP_C), chip("color = red", COL_C), chip("position = center", POS_C))
        chips.arrange(DOWN, buff=0.35).move_to([-5.0, 0.6, 0])
        strips = VGroup(*[strip(c, seed=k) for k, c in enumerate((SHP_C, COL_C, POS_C))]).arrange(DOWN, buff=0.58).move_to([-1.4, 0.6, 0])
        arrows = VGroup(*[Arrow(c.get_right(), s.get_left(), buff=0.12, color=GREY_B, stroke_width=3) for c, s in zip(chips, strips)])
        tchip = chip("time t", T_C, w=1.8).move_to([-5.0, -1.9, 0])
        tstrip = strip(T_C, seed=7).move_to([-1.4, -1.9, 0])
        tarrow = Arrow(tchip.get_right(), tstrip.get_left(), buff=0.12, color=GREY_B, stroke_width=3)
        plus = Text("+", font_size=40).move_to([0.8, -0.2, 0])
        brace = Brace(VGroup(strips, tstrip), RIGHT, color=GREY_B).shift(0.15 * RIGHT)
        unet = VGroup(*[Rectangle(width=0.5, height=h, color=WHITE, fill_color=GREY_E, fill_opacity=1)
                        for h in (2.6, 1.8, 1.1, 1.8, 2.6)]).arrange(RIGHT, buff=0.25).move_to([4.4, -0.2, 0])
        ul = Text("U-Net", font_size=20).next_to(unet, UP, buff=0.25)
        feeds = VGroup(*[Arrow([b.get_x(), -2.5, 0], b.get_bottom(), buff=0.05, color=YELLOW, stroke_width=3,
                               max_tip_length_to_length_ratio=0.35) for b in unet])
        bus = Line([plus.get_x() + 0.4, -2.5, 0], [unet[-1].get_x(), -2.5, 0], color=YELLOW, stroke_width=3)
        lead = Line(plus.get_bottom() + 0.05 * DOWN, [plus.get_x(), -2.5, 0], color=YELLOW, stroke_width=3)
        self.at("labels", "label")
        self.play(LaggedStart(*[FadeIn(c) for c in chips], lag_ratio=0.2), run_time=0.7)
        self.at("vector")
        self.play(*[GrowArrow(a) for a in arrows], *[FadeIn(s) for s in strips], run_time=0.6)
        self.at("added")
        self.play(FadeIn(tchip), GrowArrow(tarrow), FadeIn(tstrip), run_time=0.5)
        self.play(GrowFromCenter(brace), FadeIn(plus), run_time=0.4)
        self.at("block")
        self.play(FadeIn(unet), FadeIn(ul), run_time=0.4)
        self.play(Create(lead), Create(bus), *[GrowArrow(f) for f in feeds], run_time=0.7)
        req = Text("every block reads the time and the request", font_size=20, color=YELLOW).to_edge(DOWN, buff=0.3)
        self.at("request")
        self.play(FadeIn(req), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 3. training
    def training(self):
        self.section(3)
        self.clear_stage()
        noisy = tile(f"{OUT}/requests.png", 0, 1.6).move_to([-5.0, 0.4, 0])
        rng = np.random.default_rng(3)
        nz = ImageMobject((np.clip(rng.normal(0.5, 0.25, (32, 32, 3)), 0, 1) * 255).astype(np.uint8))
        nz.set_resampling_algorithm(RESAMPLING_ALGORITHMS["nearest"]).scale_to_fit_height(1.6).move_to(noisy)
        nz.set_opacity(0.75)
        nl = small("noisy picture").next_to(noisy, DOWN, buff=0.2)
        labs = VGroup(chip("triangle", SHP_C, 1.9), chip("red", COL_C, 1.9), chip("center", POS_C, 1.9)).arrange(DOWN, buff=0.15)
        labs.move_to([-5.0, -2.2, 0]).scale(0.8)
        net = VGroup(RoundedRectangle(width=2.4, height=1.4, corner_radius=0.15, color=WHITE), Text("network", font_size=22)).move_to([-0.6, 0.4, 0])
        net[1].move_to(net[0])
        guess = VGroup(Text("noise guess", font_size=22, color=YELLOW), small("same loss as episode 6")).arrange(DOWN, buff=0.15).move_to([3.6, 0.4, 0])
        a1 = Arrow(noisy.get_right(), net.get_left(), buff=0.15, color=GREY_B)
        a2 = Arrow(labs.get_right(), net.get_corner(DL) + 0.3 * RIGHT, buff=0.15, color=GREY_B)
        a3 = Arrow(net.get_right(), guess.get_left(), buff=0.15, color=GREY_B)
        self.at("noisy")
        self.play(FadeIn(noisy), FadeIn(nz), FadeIn(nl), run_time=0.5)
        self.play(FadeIn(labs), run_time=0.4)
        self.play(FadeIn(net), GrowArrow(a1), GrowArrow(a2), run_time=0.5)
        self.at("guesses")
        self.play(GrowArrow(a3), FadeIn(guess), run_time=0.5)
        hint = Text("the labels say what the clean picture should be", font_size=20, color=GREY_B).move_to([2.6, -2.2, 0])
        self.at("now")
        self.play(FadeIn(hint), run_time=0.4)
        loss = small(f"final loss {0.0026:.4f}   (unconditional, episode 8: 0.0028)", WHITE).next_to(hint, DOWN, buff=0.3)
        self.at("easier")
        self.play(FadeIn(loss), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 4. label dropout
    def dropout(self):
        self.section(4)
        self.clear_stage()
        head = Text("labels the network sees during training", font_size=24).to_edge(UP, buff=0.6)
        examples = [("triangle", "red", "center", ""), ("—", "—", "—", "10%: nothing said"),
                    ("circle", "—", "top left", "each label hidden"), ("square", "blue", "—", "10% of the time"),
                    ("—", "yellow", "bottom", "")]
        rows = VGroup()
        for j, (s, c, p, note) in enumerate(examples):
            y = 1.3 - 0.75 * j
            cells = VGroup(*[Text(v, font=MONO, font_size=22, color=(GREY_B if v == "—" else col)) for v, col in ((s, SHP_C), (c, COL_C), (p, POS_C))])
            for k, cell in enumerate(cells):
                cell.move_to([-4.6 + 2.6 * k, y, 0])
            n = (Text(note, font_size=20, color=YELLOW) if note else VectorizedPoint()).move_to([2.9, y, 0], aligned_edge=LEFT)
            rows.add(VGroup(cells, n))
        hdr = VGroup(*[Text(h, font_size=18, color=GREY_B) for h in ("shape", "color", "position")])
        for k, h in enumerate(hdr):
            h.move_to([rows[0][0][k].get_x(), rows[0].get_y() + 0.7, 0])
        self.play(FadeIn(head), FadeIn(hdr), FadeIn(rows[0][0]), run_time=0.5)
        self.at("hide")
        self.play(FadeIn(rows[1]), run_time=0.4)
        self.at("each")
        self.play(FadeIn(rows[2][0]), FadeIn(rows[3][0]), FadeIn(rows[4][0]), FadeIn(rows[2][1]), FadeIn(rows[3][1]), run_time=0.6)
        tail = Text("→ it can draw from part of a description, or from none", font_size=22, color=GREY_A).to_edge(DOWN, buff=0.5)
        self.at("part")
        self.play(FadeIn(tail), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 5. results
    def results(self):
        self.section(5)
        self.clear_stage()
        names = R["requests_png"]
        grid = Group()
        for r_ in range(3):
            rr = row(f"{OUT}/requests.png", h=1.0, r=r_, buff=0.08)
            lab = Text(names[r_], font_size=30, color=YELLOW).scale(0.6).next_to(rr, LEFT, buff=0.35)
            grid.add(Group(lab, rr))
        grid.arrange(DOWN, buff=0.25, aligned_edge=RIGHT).move_to([0.2, 0.7, 0])
        T = R["trained"]
        res = Text(f"all {T['combinations']} trained combinations: {pct(T['follows'])} exactly as asked",
                   font_size=24, color=GREEN).to_edge(DOWN, buff=0.7)
        sub = small(f"shape {pct(T['per_attribute']['shape'])} · color {pct(T['per_attribute']['color'])} · "
                    f"position {pct(T['per_attribute']['position'])}", size=28).next_to(res, DOWN, buff=0.15)
        for k, cue in enumerate(("red", "yellow", "blue")):
            self.at(cue)
            self.play(FadeIn(grid[k][0]), LaggedStart(*[FadeIn(t) for t in grid[k][1]], lag_ratio=0.06), run_time=0.7)
        self.at("97")
        self.play(FadeIn(res), FadeIn(sub), run_time=0.5)
        self.end_section()

    # ------------------------------------------------------------------ 6. held out
    def held_out(self):
        self.section(6)
        self.clear_stage()
        Hd = R["held_out"]
        head = Text("8 combinations never seen in training", font_size=24).to_edge(UP, buff=0.5)
        grid = Group()
        for r_ in range(4):
            rr = row(f"{OUT}/held_out.png", h=0.95, r=r_, buff=0.08)
            lab = Text(Hd["by_combination"][r_]["request"], font_size=30, color=YELLOW).scale(0.55).next_to(rr, LEFT, buff=0.35)
            grid.add(Group(lab, rr))
        grid.arrange(DOWN, buff=0.18, aligned_edge=RIGHT).move_to([0.3, 0.0, 0])
        res = Text(f"{Hd['samples']} pictures: {pct(Hd['follows'])} exactly as asked", font_size=24, color=GREEN).to_edge(DOWN, buff=0.45)
        self.at("eight")
        self.play(FadeIn(head), run_time=0.4)
        self.at("purple")
        self.play(FadeIn(grid[0]), run_time=0.5)
        self.play(LaggedStart(*[FadeIn(g) for g in grid[1:]], lag_ratio=0.25), run_time=1.0)
        self.at("97")
        self.play(FadeIn(res), run_time=0.4)
        k = small("shape, color and position learned separately, then combined", YELLOW, 32).next_to(head, DOWN, buff=0.12)
        self.at("separately")
        self.play(FadeIn(k), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 7. part of a description
    def partial(self):
        self.section(7)
        self.clear_stage()
        P = R["partial"]["red"]
        r = row(f"{OUT}/partial_red.png", h=1.1).move_to([0, 2.1, 0])
        said = Text("said: “red”", font_size=24, color=COL_C).next_to(r, UP, buff=0.2)
        self.at("red")
        self.play(FadeIn(said), LaggedStart(*[FadeIn(t) for t in r], lag_ratio=0.06), run_time=0.8)
        sp = P["spread_of_unsaid"]
        mx = max(sp["shape"].values())
        bars = VGroup()
        for k, (name, v) in enumerate(sp["shape"].items()):
            b = Rectangle(width=0.7, height=2.0 * v / mx, stroke_width=0, fill_color=SHP_C, fill_opacity=0.85)
            b.move_to([-4.6 + 1.2 * k, -2.2, 0], aligned_edge=DOWN)
            bars.add(VGroup(b, Text(str(v), font=MONO, font_size=16).next_to(b, UP, buff=0.08),
                            small(name, GREY_B, 28).next_to(b, DOWN, buff=0.12)))
        bl = Text("shape (not said)", font_size=20, color=SHP_C).move_to([-3.4, 0.6, 0])
        pv = list(sp["position"].values())
        pm = max(pv)
        cells = VGroup()
        for k, v in enumerate(pv):
            sq = Square(side_length=0.78, stroke_color=GREY_D, stroke_width=2, fill_color=POS_C, fill_opacity=0.15 + 0.75 * v / pm)
            cells.add(VGroup(sq, Text(str(v), font=MONO, font_size=16).move_to(sq)))
        cells.arrange_in_grid(rows=3, cols=3, buff=0.06).move_to([3.4, -1.2, 0])
        pl = Text("position (not said)", font_size=20, color=POS_C).next_to(cells, UP, buff=0.25)
        self.at("shape")
        self.play(FadeIn(bl), *[FadeIn(b) for b in bars], run_time=0.6)
        self.at("place")
        self.play(FadeIn(pl), FadeIn(cells), run_time=0.6)
        self.at("random")
        foot = small(f"{pct(P['follows'])} red and recognizable  ·  counts of recognizable pictures out of 200", GREY_B, 28).to_edge(DOWN, buff=0.15)
        self.play(FadeIn(foot), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 8. unconditional
    def unconditional(self):
        self.section(8)
        self.clear_stage()
        U = R["unconditional"]
        g = Group(row(f"{OUT}/unconditional.png", h=1.1, r=0), row(f"{OUT}/unconditional.png", h=1.1, r=1)).arrange(DOWN, buff=0.1).move_to([0, 1.0, 0])
        said = Text("said: nothing", font_size=24, color=GREY_B).next_to(g, UP, buff=0.2)
        self.at("nothing")
        self.play(FadeIn(said), FadeIn(g), run_time=0.7)
        res = Text(f"{pct(U['recognizable'])} recognizable · {U['combinations']} of 135 combinations", font_size=24, color=GREEN).move_to([0, -1.4, 0])
        self.at("90")
        self.play(FadeIn(res), run_time=0.4)
        two = Text("one network, two models: conditional and unconditional", font_size=24, color=YELLOW).move_to([0, -2.4, 0])
        self.at("network")
        self.play(FadeIn(two), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 9. real models
    def real(self):
        self.section(9)
        self.clear_stage()
        a = VGroup(Text("class-conditional models", font_size=26, color=SHP_C),
                   Text("one of 1,000 ImageNet classes: “goldfish”, “volcano”, …", font_size=20)).arrange(DOWN, buff=0.2).move_to([0, 1.3, 0])
        b = VGroup(Text("text-to-image models", font_size=26, color=YELLOW),
                   Text("a whole sentence, turned into numbers by a text encoder", font_size=20)).arrange(DOWN, buff=0.2).move_to([0, -1.0, 0])
        self.at("same")
        self.play(FadeIn(a[0]), run_time=0.4)
        self.at("1000", "thousand")
        self.play(FadeIn(a[1]), run_time=0.4)
        self.at("sentence")
        self.play(FadeIn(b[0]), run_time=0.4)
        self.at("text")
        self.play(FadeIn(b[1]), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 10. code
    def code(self):
        self.section(10)
        self.clear_stage()
        code, hl = code_panel(CODE, font_size=17)
        code.move_to([0, 0.2, 0])
        self.play(FadeIn(code, shift=0.2 * UP), run_time=0.4)
        self.at("look")
        self.play(Create(hl), run_time=0.2)
        self.play(highlight(hl, code, 1), run_time=0.3)
        self.at("add")
        self.play(highlight(hl, code, 4), run_time=0.3)
        self.at("drop")
        self.play(highlight(hl, code, 7), run_time=0.3)
        self.play(highlight(hl, code, 8), run_time=0.3)
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
