"""How Generative Models Work: Images and Sound, episode 1 — What It Means to Generate.

Render from the repo root:  ./render.sh generative g01
Numbers and pictures come from code/g01_what_it_means_to_generate (generate_basics.py → results.json and out/*.png).
"""
import json

import numpy as np
from PIL import Image
from manim import *

from common import MONO, code_panel, finish, highlight, next_up_card, token
from intro import play_token_intro
from g01_script import NEXT, TAGLINE, TITLE
from voiced_scene import VoicedScene

SERIES_LABEL = "HOW GENERATIVE MODELS WORK  ·  EPISODE 1"
CODE_DIR = "code/g01_what_it_means_to_generate"
R = json.load(open(f"{CODE_DIR}/results.json"))
DATA_C, GEN_C = GREY_B, ORANGE
CODE = """counts = np.histogram2d(x, y, bins=30)[0]
p = counts.ravel() / counts.sum()
cells = rng.choice(900, size=2000, p=p)"""


def pic(path, height):
    m = ImageMobject(path)
    m.set_resampling_algorithm(RESAMPLING_ALGORITHMS["nearest"])
    return m.scale_to_fit_height(height)


def tile(path, i, height, size=128, pad=2):
    """Image i of a grid PNG written by shapes.save_grid."""
    arr = np.asarray(Image.open(path).convert("RGB"))
    x = i * (size + pad)
    m = ImageMobject(arr[:size, x:x + size])
    m.set_resampling_algorithm(RESAMPLING_ALGORITHMS["nearest"])
    return m.scale_to_fit_height(height)


def power(base, exp, font_size=24, color=WHITE, font=MONO):
    b = Text(base, font=font, font_size=font_size, color=color)
    e = Text(exp, font=font, font_size=font_size * 0.62, color=color).next_to(b, UR, buff=0.02).shift(DOWN * font_size * 0.006)
    return VGroup(b, e)


def frame(m, color=GREY_C):
    return SurroundingRectangle(m, buff=0, color=color, stroke_width=2)


def plane(center, size=4.4):
    ax = Axes(x_range=[-1.15, 1.15, 1], y_range=[-1.15, 1.15, 1], x_length=size, y_length=size,
              tips=False, axis_config={"stroke_opacity": 0.0})
    box = Square(side_length=size, color=GREY_D, stroke_width=2)
    return VGroup(box, ax).move_to(center), ax


def dots(ax, pts, color, r=0.022):
    return VGroup(*[Dot(ax.c2p(x, y), radius=r, color=color) for x, y in pts])


class GenerateVideo(VoicedScene):
    VIDEO = "g01"

    def construct(self):
        play_token_intro(self, TITLE, 1, TAGLINE, label=SERIES_LABEL)
        self.hook()        # 1
        self.sampling()    # 2
        self.shapes()      # 3
        self.counting()    # 4
        self.explode()     # 5
        self.pixels()      # 6
        self.extremes()    # 7
        self.series_map()  # 8
        self.code()        # 9
        self.outro()       # 10

    def clear_stage(self, *keep, run_time=0.6):
        self.play(*[FadeOut(m) for m in self.mobjects if m not in keep], run_time=run_time)

    # ------------------------------------------------------------------ 1. hook
    def hook(self):
        self.section(1)
        words = ["The", "cat", "sat", "on", "the"]
        row = VGroup(*[token(w, font_size=26) for w in words]).arrange(RIGHT, buff=0.12).move_to([-3.4, 1.0, 0])
        self.at("language")
        self.play(FadeIn(row[0]), run_time=0.3)
        self.at("one")
        self.play(LaggedStart(*[FadeIn(t, shift=0.15 * RIGHT) for t in row[1:]], lag_ratio=0.35), run_time=1.4)
        lab = Text("one token at a time", font_size=22, color=GREY_B).next_to(row, DOWN, buff=0.35)
        self.play(FadeIn(lab), run_time=0.3)
        img = tile(f"{CODE_DIR}/out/real.png", 3, 4.2).move_to([3.2, 0.5, 0])
        self.at("pictures")
        self.play(FadeIn(img), FadeIn(frame(img)), run_time=0.6)
        s = img.height / 32
        grid = VGroup(*[Line(img.get_corner(UL) + RIGHT * s * k, img.get_corner(DL) + RIGHT * s * k, stroke_width=0.6, color=GREY_D) for k in range(1, 32)],
                      *[Line(img.get_corner(UL) + DOWN * s * k, img.get_corner(UR) + DOWN * s * k, stroke_width=0.6, color=GREY_D) for k in range(1, 32)])
        self.at("32")
        self.play(Create(grid, lag_ratio=0.01), run_time=1.0)
        n = Text("32 × 32 pixels × 3 colors = 3,072 numbers", font_size=24, color=YELLOW).next_to(img, DOWN, buff=0.35)
        n.shift(RIGHT * min(0, 6.9 - n.get_right()[0]))
        self.at("072", "numbers")
        self.play(FadeIn(n), run_time=0.5)
        t = Text("…and they all have to fit together", font_size=24, color=GEN_C).move_to([-3.4, -1.2, 0])
        self.at("fit")
        self.play(FadeIn(t), run_time=0.5)
        self.end_section()

    # ------------------------------------------------------------------ 2. generating = sampling
    def sampling(self):
        self.section(2)
        self.clear_stage()
        rng = np.random.default_rng(3)
        g, ax = plane([-3.3, 0.2, 0], 4.0)
        cloud = [(0.55 * np.cos(a) + rng.normal(0, 0.08), 0.35 * np.sin(a) + rng.normal(0, 0.08)) for a in rng.uniform(0, 2 * PI, 120)]
        d = dots(ax, cloud, DATA_C, 0.03)
        cap1 = Text("training data", font_size=20, color=DATA_C).next_to(g, DOWN, buff=0.2)
        self.at("generate")
        self.play(FadeIn(g), FadeIn(d, lag_ratio=0.02), FadeIn(cap1), run_time=1.0)
        new = Dot(ax.c2p(-0.55, 0.05), radius=0.07, color=GEN_C)
        nl = Text("new sample", font_size=20, color=GEN_C).next_to(g, UP, buff=0.2).align_to(g, LEFT)
        nl = VGroup(nl, Arrow(nl.get_bottom(), new.get_center(), buff=0.1, color=GEN_C, stroke_width=3))
        self.at("draw")
        self.play(FadeIn(new, scale=2), FadeIn(nl), run_time=0.5)
        steps = VGroup(Text("1. learn the distribution", font_size=26), Text("2. sample from it", font_size=26, color=GEN_C)).arrange(DOWN, aligned_edge=LEFT, buff=0.3)
        steps.move_to([3.0, 2.2, 0])
        self.at("learn")
        self.play(FadeIn(steps[0]), run_time=0.4)
        self.at("then")
        self.play(FadeIn(steps[1]), run_time=0.4)
        cands = ["mat", "floor", "bed", "sofa", "roof"]
        hs = [1.6, 0.9, 0.6, 0.45, 0.3]
        bars = VGroup(*[Rectangle(width=0.55, height=h, stroke_width=0, fill_color=BLUE_C if i else GEN_C, fill_opacity=0.85) for i, h in enumerate(hs)]).arrange(RIGHT, buff=0.18, aligned_edge=DOWN)
        bars.move_to([3.0, -0.6, 0])
        labs = VGroup(*[Text(c, font_size=16).next_to(b, DOWN, buff=0.1) for c, b in zip(cands, bars)])
        cap2 = Text("a language model: p(next token | “The cat sat on the”)", font_size=18, color=GREY_B).next_to(bars, DOWN, buff=0.55)
        self.at("language")
        self.play(FadeIn(bars, lag_ratio=0.1), FadeIn(labs), FadeIn(cap2), run_time=0.8)
        self.at("next")
        self.play(Indicate(bars[0], color=GEN_C), run_time=0.6)
        self.end_section()

    # ------------------------------------------------------------------ 3. our data
    def shapes(self):
        self.section(3)
        self.clear_stage()
        h = Text("Shapes: our training data", font_size=30).to_edge(UP, buff=0.6)
        self.at("shapes")
        self.play(FadeIn(h), run_time=0.4)
        tiles = Group(*[tile(f"{CODE_DIR}/out/real.png", i, 1.35) for i in range(8)]).arrange(RIGHT, buff=0.18).move_to([0, 0.9, 0])
        frames = VGroup(*[frame(t) for t in tiles])
        self.at("each")
        self.play(LaggedStart(*[FadeIn(Group(t, f)) for t, f in zip(tiles, frames)], lag_ratio=0.12), run_time=1.2)
        caps = R["pixels"]["real_captions"]
        sel = [1, 3, 5]
        cl = VGroup()
        for i in sel:
            c = Text(f"“{caps[i]}”", font_size=17, color=YELLOW)
            cl.add(c)
        cl.arrange(DOWN, buff=0.18).move_to([0, -1.15, 0])
        arrows = VGroup(*[Line(tiles[i].get_bottom(), c.get_top(), color=GREY_D, stroke_width=1.5, buff=0.08) for i, c in zip(sel, cl)])
        self.at("caption")
        self.play(FadeIn(cl, lag_ratio=0.3), Create(arrows), run_time=1.0)
        meta = Text("32 × 32 · 3 shapes × 5 colors × 9 positions · made by shapes.py", font_size=20, color=GREY_B).move_to([0, -2.6, 0])
        self.at("code")
        self.play(FadeIn(meta), run_time=0.4)
        n = Text("20,000 pictures in a few seconds", font_size=24, color=GEN_C).next_to(meta, DOWN, buff=0.3)
        self.at("20")
        self.play(FadeIn(n), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 4. counting in 2D
    def counting(self):
        self.section(4)
        self.clear_stage()
        g1, ax1 = plane([-3.3, 0.1, 0])
        g2, ax2 = plane([3.3, 0.1, 0])
        t1 = Text("10,000 points (600 shown)", font_size=20, color=DATA_C).next_to(g1, UP, buff=0.15)
        self.at("10")
        self.play(FadeIn(g1), FadeIn(t1), FadeIn(dots(ax1, R["toy"]["data"], DATA_C), lag_ratio=0.005), run_time=1.2)
        size = g1[0].width
        c = size / 30
        corner = g1[0].get_corner(DL)
        grid = VGroup(*[Line(corner + RIGHT * c * k, corner + RIGHT * c * k + UP * size, stroke_width=0.5, color=GREY_D) for k in range(1, 30)],
                      *[Line(corner + UP * c * k, corner + UP * c * k + RIGHT * size, stroke_width=0.5, color=GREY_D) for k in range(1, 30)])
        tg = Text("900 cells", font_size=20, color=GREY_B).next_to(g1, DOWN, buff=0.15)
        self.at("split")
        self.play(Create(grid, lag_ratio=0.01), FadeIn(tg), run_time=1.0)
        counts = np.array(R["toy"]["counts"])
        mx = counts.max()
        cells = VGroup()
        for i in range(30):          # i = x bin, j = y bin
            for j in range(30):
                if counts[i, j]:
                    cells.add(Square(side_length=c, stroke_width=0, fill_color=BLUE_C, fill_opacity=0.15 + 0.75 * counts[i, j] / mx)
                              .move_to(corner + RIGHT * c * (i + 0.5) + UP * c * (j + 0.5)))
        self.at("count")
        self.play(FadeIn(cells, lag_ratio=0.003), run_time=1.2)
        tc = Text(f"{R['toy']['occupied_cells']} of 900 cells are used", font_size=20, color=BLUE_C).next_to(tg, DOWN, buff=0.1)
        self.play(FadeIn(tc), run_time=0.3)
        arrow = Arrow(g1.get_right(), g2.get_left(), buff=0.2, color=GEN_C)
        al = Text("sample", font_size=20, color=GEN_C).next_to(arrow, UP, buff=0.1)
        self.at("pick")
        self.play(FadeIn(g2), GrowArrow(arrow), FadeIn(al), run_time=0.6)
        t2 = Text("2,000 samples (600 shown)", font_size=20, color=GEN_C).next_to(g2, UP, buff=0.15)
        self.at("samples")
        self.play(FadeIn(t2), FadeIn(dots(ax2, R["toy"]["samples"], GEN_C), lag_ratio=0.005), run_time=1.2)
        self.end_section()

    # ------------------------------------------------------------------ 5. why counting fails
    def explode(self):
        self.section(5)
        self.clear_stage()
        h = Text("cells with 30 bins per number", font_size=28).to_edge(UP, buff=0.6)
        self.at("pictures")
        self.play(FadeIn(h), run_time=0.4)
        rows = [("1 number", ("30", "")), ("2 numbers", ("900", "")), ("3 numbers", ("27,000", "")), ("10 numbers", ("about 10", "15")), ("100 numbers", ("about 10", "148"))]
        tab = VGroup(*[VGroup(Text(a, font_size=24), power(*b, font_size=24, color=BLUE_C)).arrange(RIGHT, buff=0.6) for a, b in rows])
        tab.arrange(DOWN, aligned_edge=LEFT, buff=0.25).move_to([-3.2, 0.4, 0])
        for r in tab:
            r[1].align_to(np.array([-2.2, 0, 0]), LEFT)
        self.at("multiplies")
        self.play(LaggedStart(*[FadeIn(r) for r in tab], lag_ratio=0.3), run_time=1.6)
        big = VGroup(Text("3,072 numbers, each only dark or bright:", font_size=24),
                     VGroup(power("2", "3072", 34, RED_B), Text("≈", font=MONO, font_size=34, color=RED_B), power("10", "924", 34, RED_B),
                            Text("cells", font=MONO, font_size=34, color=RED_B)).arrange(RIGHT, buff=0.25)).arrange(DOWN, buff=0.25).move_to([3.4, 0.6, 0])
        self.at("dark")
        self.play(FadeIn(big[0]), run_time=0.4)
        self.at("924", "power")
        self.play(FadeIn(big[1], scale=1.2), run_time=0.6)
        have = Text("our data: 20,000 pictures", font_size=24, color=GREY_B).next_to(big, DOWN, buff=0.6)
        self.at("20")
        self.play(FadeIn(have), run_time=0.4)
        x = Text("almost every cell is empty forever", font_size=24, color=YELLOW).to_edge(DOWN, buff=0.7)
        self.at("fill")
        self.play(FadeIn(x), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 6. pixels one by one
    def pixels(self):
        self.section(6)
        self.clear_stage()
        heat = pic(f"{CODE_DIR}/out/shape_probability.png", 2.6).move_to([-4.6, 1.5, 0])
        hl = Text("chance that each pixel\nis part of a shape\n(brightest: 11%)", font_size=19, color=GREY_B, line_spacing=0.9).next_to(heat, DOWN, buff=0.15)
        self.at("pixel")
        self.play(FadeIn(heat), FadeIn(frame(heat)), FadeIn(hl), run_time=0.6)
        top = Group(*[tile(f"{CODE_DIR}/out/real.png", i, 1.15) for i in range(4)]).arrange(RIGHT, buff=0.15)
        bot = Group(*[tile(f"{CODE_DIR}/out/pixel_samples.png", i, 1.15) for i in range(4)]).arrange(RIGHT, buff=0.15)
        top.move_to([2.2, 1.9, 0])
        bot.move_to([2.2, -0.15, 0])
        lt = Text("real", font_size=20, color=DATA_C).next_to(top, LEFT, buff=0.25)
        lb = Text("sampled\npixel by pixel", font_size=20, color=GEN_C, line_spacing=0.9).next_to(bot, LEFT, buff=0.25)
        self.play(FadeIn(top), FadeIn(VGroup(*[frame(t) for t in top])), FadeIn(lt), run_time=0.4)
        self.at("independently")
        self.play(FadeIn(bot), FadeIn(VGroup(*[frame(t) for t in bot])), FadeIn(lb), run_time=0.6)
        p = R["pixels"]
        stats = VGroup(
            VGroup(Text("x", font_size=20).set_opacity(0), Text("real", font_size=20, color=DATA_C), Text("sampled", font_size=20, color=GEN_C)),
            VGroup(Text("shape pixels", font_size=20), Text(f"{p['real_shape_pixels_mean']:.0f}", font=MONO, font_size=20), Text(f"{p['sample_shape_pixels_mean']:.0f}", font=MONO, font_size=20, color=GEN_C)),
            VGroup(Text("biggest connected piece", font_size=20), Text(f"{p['real_largest_blob_mean']:.0f}", font=MONO, font_size=20), Text(f"{p['sample_largest_blob_mean']:.1f}", font=MONO, font_size=20, color=RED_B)))
        stats.arrange(DOWN, buff=0.22, aligned_edge=LEFT).move_to([1.0, -2.35, 0])
        for r in stats:
            r[0].align_to(np.array([-1.4, 0, 0]), LEFT)
            r[1].set_x(3.5)
            r[2].set_x(5.0)
        self.at("amount")
        self.play(FadeIn(stats[0]), FadeIn(stats[1]), run_time=0.5)
        self.at("biggest")
        self.play(FadeIn(stats[2]), run_time=0.5)
        self.at("real")
        self.play(Indicate(stats[2][1], color=YELLOW), run_time=0.8)
        self.end_section()

    # ------------------------------------------------------------------ 7. average and copy
    def extremes(self):
        self.section(7)
        self.clear_stage()
        avg = pic(f"{CODE_DIR}/out/average_x4.png", 2.4).move_to([-4.3, 1.0, 0])
        la = Text("the average of 20,000\npictures (contrast × 4)", font_size=20, color=GREY_B, line_spacing=0.9).next_to(avg, DOWN, buff=0.2)
        self.at("average")
        self.play(FadeIn(avg), FadeIn(frame(avg)), FadeIn(la), run_time=0.6)
        cp = Group(*[tile(f"{CODE_DIR}/out/copies.png", i, 1.1) for i in range(4)]).arrange(RIGHT, buff=0.12).move_to([3.2, 1.0, 0])
        lc = Text("a random training picture:\nperfect, but nothing new", font_size=20, color=GREY_B, line_spacing=0.9).next_to(cp, DOWN, buff=0.2)
        self.at("picking")
        self.play(FadeIn(cp), FadeIn(VGroup(*[frame(t) for t in cp])), FadeIn(lc), run_time=0.6)
        bar = Line([-5.5, -2.0, 0], [5.5, -2.0, 0], color=GREY_C)
        e1 = Text("blur", font_size=20, color=GREY_B).next_to(bar.get_start(), DOWN, buff=0.15)
        e2 = Text("copying", font_size=20, color=GREY_B).next_to(bar.get_end(), DOWN, buff=0.15)
        mid = Dot(bar.get_center(), radius=0.1, color=GEN_C)
        ml = Text("a real generator: learns how pixels depend\non each other, makes new combinations", font_size=22, color=GEN_C, line_spacing=0.9).next_to(mid, DOWN, buff=0.25)
        self.at("between")
        self.play(Create(bar), FadeIn(e1), FadeIn(e2), FadeIn(mid, scale=2), run_time=0.6)
        self.at("learns")
        self.play(FadeIn(ml), run_time=0.5)
        self.end_section()

    # ------------------------------------------------------------------ 8. the map
    def series_map(self):
        self.section(8)
        self.clear_stage()
        parts = [("Part 1", "codes", "autoencoders · VAEs · GANs", BLUE_C, "auto"),
                 ("Part 2", "diffusion", "noise → picture, step by step", GEN_C, "diffusion"),
                 ("Part 3", "text", "conditioning · guidance · CLIP", GREEN_C, "text"),
                 ("Part 4", "sound", "spectrograms · codecs · speech", TEAL_C, "sound")]
        cards = VGroup()
        for a, b, c, col, _ in parts:
            r = RoundedRectangle(width=2.85, height=2.4, corner_radius=0.15, color=col, fill_opacity=0.1)
            g = VGroup(Text(a, font_size=18, color=GREY_B), Text(b, font_size=30, color=col), Text(c, font_size=15, line_spacing=0.9)).arrange(DOWN, buff=0.25)
            if g[2].width > 2.6:
                g[2].scale_to_fit_width(2.6)
            cards.add(VGroup(r, g.move_to(r)))
        cards.arrange(RIGHT, buff=0.25).move_to([0, 0.2, 0])
        for card, (_, _, _, _, cue) in zip(cards, parts):
            self.at(cue)
            self.play(FadeIn(card, shift=0.2 * UP), run_time=0.5)
        self.end_section()

    # ------------------------------------------------------------------ 9. code
    def code(self):
        self.section(9)
        self.clear_stage()
        code, hl = code_panel(CODE, font_size=24)
        code.move_to([0, 0.4, 0])
        self.play(FadeIn(code, shift=0.2 * UP), run_time=0.5)
        self.at("count")
        self.play(Create(hl), run_time=0.3)
        self.at("probabilities")
        self.play(highlight(hl, code, 1), run_time=0.3)
        self.at("sample")
        self.play(highlight(hl, code, 2), run_time=0.3)
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
