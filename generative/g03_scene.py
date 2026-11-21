"""How Generative Models Work: Images and Sound, episode 3 — Variational Autoencoders.

Render from the repo root:  ./render.sh generative g03
Numbers and pictures come from code/g03_variational_autoencoders (train_vae.py → results.json and out/*.png) and
episode 2's 2-number autoencoder codes (code/g02_autoencoders/results.json).
"""
import json

import numpy as np
from PIL import Image
from manim import *

from common import MONO, code_panel, finish, highlight, next_up_card
from gen_common import frame, pic, plane, tile
from intro import play_token_intro
from g03_script import NEXT, TAGLINE, TITLE
from voiced_scene import VoicedScene

SERIES_LABEL = "HOW GENERATIVE MODELS WORK  ·  EPISODE 3"
CODE_DIR = "code/g03_variational_autoencoders"
OUT = f"{CODE_DIR}/out"
R = json.load(open(f"{CODE_DIR}/results.json"))
AE2 = json.load(open("code/g02_autoencoders/results.json"))["codes_2d"]
M = R["models"]
ENC_C, DEC_C, CODE_C, KL_C, GEN_C = BLUE_C, GREEN_C, YELLOW, PINK, ORANGE
POS_COLORS = [RED_C, ORANGE, YELLOW, GREEN_C, TEAL_C, BLUE_C, PURPLE_B, PINK, GREY_B]
CODE = """h = body(x)
mean, logvar = to_mean(h), to_logvar(h)
code = mean + torch.exp(0.5 * logvar) * torch.randn_like(mean)
x_hat = decoder(code)
recon = F.binary_cross_entropy(x_hat, x, reduction="sum")
kl = 0.5 * (mean**2 + logvar.exp() - 1 - logvar).sum()
loss = recon + kl"""


def scatter(codes, labels, center, size, lo, hi, r=0.022):
    g, ax = plane(center, size, lo, hi)
    pts = VGroup(*[Dot(ax.c2p(*c), radius=r, color=POS_COLORS[p]) for c, p in zip(codes, labels)])
    return g, ax, pts


def cloud(ax, mean, spread, color, n_rings=3):
    rings = VGroup()
    for k in range(n_rings, 0, -1):
        f = 0.6 * k                                  # rings at 0.6, 1.2, 1.8 spreads
        e = Ellipse(width=abs(ax.c2p(spread[0] * f, 0)[0] - ax.c2p(0, 0)[0]) * 2,
                    height=abs(ax.c2p(0, spread[1] * f)[1] - ax.c2p(0, 0)[1]) * 2,
                    stroke_width=0, fill_color=color, fill_opacity=0.18)
        rings.add(e.move_to(ax.c2p(*mean)))
    rings.add(Dot(ax.c2p(*mean), radius=0.04, color=color))
    return rings


def average_tile(path, i, j, height):
    arr = np.asarray(Image.open(path).convert("RGB")).astype(float)
    s, p = 128, 2
    a, b = arr[:s, i * (s + p):i * (s + p) + s], arr[:s, j * (s + p):j * (s + p) + s]
    m = ImageMobject(((a + b) / 2).round().astype(np.uint8))
    m.set_resampling_algorithm(RESAMPLING_ALGORITHMS["nearest"])
    return m.scale_to_fit_height(height)


class VAEVideo(VoicedScene):
    VIDEO = "g03"

    def construct(self):
        play_token_intro(self, TITLE, 3, TAGLINE, label=SERIES_LABEL)
        self.hook()        # 1
        self.clouds()      # 2
        self.kl()          # 3
        self.maps()        # 4
        self.sampling()    # 5
        self.blur()        # 6
        self.why_blur()    # 7
        self.code()        # 8
        self.outro()       # 9

    def clear_stage(self, *keep, run_time=0.6):
        self.play(*[FadeOut(m) for m in self.mobjects if m not in keep], run_time=run_time)

    # ------------------------------------------------------------------ 1. hook
    def hook(self):
        self.section(1)
        g, ax, pts = scatter(AE2["codes"], np.array(AE2["labels"])[:, 2], [-3.0, 0, 0], 5.0, -14, 11)
        lab = Text("episode 2: autoencoder codes (2 numbers)", font_size=20, color=GREY_B).next_to(g, UP, buff=0.15)
        self.at("islands")
        self.play(FadeIn(g), FadeIn(lab), FadeIn(pts, lag_ratio=0.0005), run_time=1.0)
        codes = np.array(AE2["codes"])
        gx, gy = np.meshgrid(np.linspace(-8, 6, 57), np.linspace(-8, 7, 61))      # emptiest spot inside the cloud
        cand = np.stack([gx.ravel(), gy.ravel()], 1)
        gap = cand[np.sqrt(((cand[:, None] - codes[None]) ** 2).sum(-1)).min(1).argmax()]
        q = VGroup(Circle(radius=0.3, color=RED_B, stroke_width=3), Text("?", font_size=26, color=RED_B)).move_to(ax.c2p(*gap))
        self.at("gaps")
        self.play(FadeIn(q), run_time=0.4)
        t = VGroup(Text("a variational autoencoder", font_size=28, color=CODE_C),
                   Text("changes two things", font_size=24), Text("→ no gaps", font_size=24, color=GEN_C)).arrange(DOWN, buff=0.25).move_to([3.6, 0.3, 0])
        self.at("variational")
        self.play(FadeIn(t[0]), run_time=0.4)
        self.at("two")
        self.play(FadeIn(t[1]), run_time=0.3)
        self.at("gaps")
        self.play(FadeIn(t[2]), run_time=0.3)
        self.end_section()

    # ------------------------------------------------------------------ 2. change one: clouds
    def clouds(self):
        self.section(2)
        self.clear_stage()
        h = Text("change 1: a cloud instead of a point", font_size=28).to_edge(UP, buff=0.5)
        self.play(FadeIn(h), run_time=0.4)
        g1, ax1 = plane([-3.4, -0.2, 0], 4.2, -3, 3)
        g2, ax2 = plane([3.4, -0.2, 0], 4.2, -3, 3)
        l1 = Text("autoencoder: one code", font_size=20, color=GREY_B).next_to(g1, DOWN, buff=0.15)
        l2 = Text("VAE: mean and spread", font_size=20, color=CODE_C).next_to(g2, DOWN, buff=0.15)
        p1 = Dot(ax1.c2p(0.8, 0.6), radius=0.08, color=CODE_C)
        self.at("point")
        self.play(FadeIn(g1), FadeIn(l1), FadeIn(p1, scale=2), run_time=0.6)
        c2 = cloud(ax2, (0.8, 0.6), (0.45, 0.3), CODE_C)
        self.at("mean")
        self.play(FadeIn(g2), FadeIn(l2), FadeIn(c2[-1]), run_time=0.5)
        self.at("cloud")
        self.play(FadeIn(c2[:-1]), run_time=0.6)
        f = Text("code = mean + spread × noise", font=MONO, font_size=24, color=GEN_C).move_to([0, -3.15, 0])
        rng = np.random.default_rng(4)
        samples = VGroup(*[Dot(ax2.c2p(0.8 + 0.45 * a, 0.6 + 0.3 * b), radius=0.05, color=GEN_C) for a, b in rng.normal(0, 1, (6, 2))])
        self.at("random")
        self.play(LaggedStart(*[FadeIn(s, scale=2) for s in samples], lag_ratio=0.2), run_time=1.0)
        self.at("plus")
        self.play(FadeIn(f), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 3. change two: KL
    def kl(self):
        self.section(3)
        self.clear_stage()
        h = Text("change 2: the KL penalty", font_size=28, color=KL_C).to_edge(UP, buff=0.5)
        self.at("penalty")
        self.play(FadeIn(h), run_time=0.4)
        g, ax = plane([-2.2, -0.4, 0], 5.6, -4, 4)
        target = DashedVMobject(Circle(radius=abs(ax.c2p(1, 0)[0] - ax.c2p(0, 0)[0]), color=KL_C, stroke_width=3).move_to(ax.c2p(0, 0)), num_dashes=30)
        tl = Text("standard normal", font_size=18, color=KL_C).next_to(target, DOWN, buff=0.08)
        starts = [((-2.6, 2.4), (0.25, 0.2), RED_C), ((2.4, 2.2), (0.2, 0.3), BLUE_C), ((-2.2, -2.5), (0.3, 0.2), GREEN_C), ((2.5, -2.4), (0.25, 0.25), PURPLE_B)]
        clouds = VGroup(*[cloud(ax, m, s, c) for m, s, c in starts])
        self.play(FadeIn(g), FadeIn(clouds), run_time=0.6)
        self.at("standard")
        self.play(Create(target), FadeIn(tl), run_time=0.6)
        ends = [((-0.5, 0.4), (0.8, 0.75)), ((0.5, 0.4), (0.75, 0.8)), ((-0.4, -0.5), (0.8, 0.8)), ((0.5, -0.4), (0.8, 0.75))]
        forces = VGroup(Text("rebuilding wants:\nsmall clouds, far apart", font_size=22, color=DEC_C, line_spacing=0.9),
                        Text("the penalty wants:\nwide clouds, on top of each other", font_size=22, color=KL_C, line_spacing=0.9)).arrange(DOWN, aligned_edge=LEFT, buff=0.4).move_to([3.9, 1.2, 0])
        self.at("rebuilding")
        self.play(FadeIn(forces[0]), run_time=0.4)
        self.at("penalty")
        self.play(FadeIn(forces[1]), run_time=0.4)
        new = VGroup(*[cloud(ax, m, s, c) for (m, s), (_, _, c) in zip(ends, starts)])
        self.at("overlap")
        self.play(Transform(clouds, new), run_time=1.2)
        done = Text("the clouds overlap → no gaps", font_size=24, color=GEN_C).move_to([3.7, -1.2, 0])
        self.at("close")
        self.play(FadeIn(done), run_time=0.3)
        self.end_section()

    # ------------------------------------------------------------------ 4. the map
    def maps(self):
        self.section(4)
        self.clear_stage()
        g1, ax1, p1 = scatter(AE2["codes"], np.array(AE2["labels"])[:, 2], [-3.4, -0.1, 0], 4.6, -14, 11, r=0.018)
        l1 = Text("autoencoder: −13 … +10", font_size=20, color=GREY_B).next_to(g1, UP, buff=0.15)
        self.at("autoencoders", "auto")
        self.play(FadeIn(g1), FadeIn(l1), FadeIn(p1, lag_ratio=0.0005), run_time=0.8)
        v = R["vae2"]
        g2, ax2, p2 = scatter(v["codes"], np.array(v["labels"])[:, 2], [3.4, -0.1, 0], 4.6, -3.6, 3.6, r=0.018)
        l2 = Text("VAE: −3.4 … +3.6, round, no gaps", font_size=20, color=CODE_C).next_to(g2, UP, buff=0.15)
        self.at("vaes", "fill")
        self.play(FadeIn(g2), FadeIn(l2), FadeIn(p2, lag_ratio=0.0005), run_time=0.8)
        mp = pic(f"{OUT}/map_vae2.png", 4.6).move_to(g2)
        ml = Text("decoded grid of codes, −2 … +2", font_size=20, color=GEN_C).move_to(l2)
        self.at("decode")
        self.play(FadeOut(p2), FadeOut(g2), FadeIn(mp), Transform(l2, ml), run_time=0.8)
        self.at("smoothly")
        self.play(Indicate(mp, scale_factor=1.03, color=GEN_C), run_time=0.8)
        self.end_section()

    # ------------------------------------------------------------------ 5. sampling
    def sampling(self):
        self.section(5)
        self.clear_stage()
        h = Text("draw a code from a standard normal, decode it (8-number code)", font_size=24).to_edge(UP, buff=0.5)
        self.at("draw")
        self.play(FadeIn(h), run_time=0.4)
        vae = Group(*[tile(f"{OUT}/random_vae.png", i, 1.2) for i in range(8)]).arrange(RIGHT, buff=0.1).move_to([0.6, 1.3, 0])
        ae = Group(*[tile(f"{OUT}/random_autoencoder.png", i, 1.2) for i in range(8)]).arrange(RIGHT, buff=0.1).move_to([0.6, -0.4, 0])
        lv = Text("VAE", font_size=22, color=CODE_C).next_to(vae, LEFT, buff=0.3)
        la = Text("auto-\nencoder", font_size=20, color=GREY_B, line_spacing=0.9).next_to(ae, LEFT, buff=0.3)
        self.at("decode")
        self.play(FadeIn(vae), FadeIn(lv), run_time=0.5)
        self.at("samples")
        self.play(FadeIn(ae), FadeIn(la), run_time=0.5)
        n = VGroup(Text("squared error to the nearest training picture", font_size=20, color=GREY_B),
                   Text(f"VAE {M['vae']['random_nearest_train_mse']:.4f}   autoencoder {M['autoencoder']['random_nearest_train_mse']:.4f}",
                        font=MONO, font_size=22, color=YELLOW)).arrange(DOWN, buff=0.15).move_to([0, -2.3, 0])
        self.at("half")
        self.play(FadeIn(n), run_time=0.4)
        b = Text("mostly one blob, in one place", font_size=22, color=GEN_C).next_to(n, DOWN, buff=0.3)
        self.at("blob")
        self.play(FadeIn(b), run_time=0.3)
        self.end_section()

    # ------------------------------------------------------------------ 6. the blur and the judge
    def blur(self):
        self.section(6)
        self.clear_stage()
        zoom = Group(*[tile(f"{OUT}/random_vae.png", i, 2.0) for i in (0, 1, 3)]).arrange(RIGHT, buff=0.25).move_to([-3.4, 1.4, 0])
        self.at("closer")
        self.play(FadeIn(zoom), FadeIn(VGroup(*[frame(z) for z in zoom])), run_time=0.5)
        zl = Text("blurry, sometimes a faint second shape", font_size=20, color=RED_B).next_to(zoom, DOWN, buff=0.2)
        self.at("faint")
        self.play(FadeIn(zl), run_time=0.3)
        jb = VGroup(RoundedRectangle(width=2.2, height=1.0, corner_radius=0.15, color=TEAL_C, fill_opacity=0.15), Text("judge", font_size=24, color=TEAL_C))
        jb[1].move_to(jb[0])
        jb.move_to([3.4, 2.2, 0])
        outs = Text("shape · color · position · junk?", font_size=18, color=TEAL_C).next_to(jb, DOWN, buff=0.15)
        self.at("judge")
        self.play(FadeIn(jb), run_time=0.4)
        self.at("names")
        self.play(FadeIn(outs), run_time=0.4)
        rows = [("real pictures", R["real_recognizable"], GREEN_C), ("VAE samples", M["vae"]["random_recognizable"], CODE_C),
                ("autoencoder samples", M["autoencoder"]["random_recognizable"], GREY_B)]
        bars = VGroup()
        for k, (name, v, col) in enumerate(rows):
            lab = Text(name, font_size=20).move_to([-0.4, -1.0 - 0.6 * k, 0], aligned_edge=RIGHT)
            bar = Rectangle(width=max(5.0 * v, 0.03), height=0.38, stroke_width=0, fill_color=col, fill_opacity=0.85)
            bar.move_to([0.0, lab.get_center()[1], 0], aligned_edge=LEFT)
            val = Text(f"{v:.1%}", font=MONO, font_size=18).next_to(bar, RIGHT, buff=0.15)
            bars.add(VGroup(lab, bar, val))
        cap = Text("recognized by the judge", font_size=20, color=TEAL_C).next_to(bars, UP, buff=0.25).align_to(bars[0][1], LEFT)
        self.at("passes")
        self.play(FadeIn(cap), FadeIn(bars[0]), run_time=0.5)
        self.at("one")
        self.play(FadeIn(bars[1]), FadeIn(bars[2]), run_time=0.6)
        self.end_section()

    # ------------------------------------------------------------------ 7. why blur
    def why_blur(self):
        self.section(7)
        self.clear_stage()
        g, ax = plane([-4.0, 0.6, 0], 3.4, -2, 2)
        c1 = cloud(ax, (-0.4, 0.1), (0.6, 0.5), YELLOW)
        c2 = cloud(ax, (0.45, -0.1), (0.6, 0.5), PURPLE_B)
        self.at("noise")
        self.play(FadeIn(g), FadeIn(c1), FadeIn(c2), run_time=0.6)
        dot = Dot(ax.c2p(0.05, 0.0), radius=0.07, color=GEN_C)
        dl = Text("this code: stands\nfor both pictures", font_size=18, color=GEN_C, line_spacing=0.9).next_to(g, DOWN, buff=0.15)
        self.at("nearby")
        self.play(FadeIn(dot, scale=2), FadeIn(dl), run_time=0.5)
        a = tile(f"{OUT}/originals.png", 0, 1.6)
        b = tile(f"{OUT}/originals.png", 5, 1.6)
        avg = average_tile(f"{OUT}/originals.png", 0, 5, 1.6)
        row = Group(a, Text("+", font_size=36), b, Text("→", font_size=36), avg).arrange(RIGHT, buff=0.3).move_to([2.2, 1.2, 0])
        self.at("average")
        self.play(FadeIn(row), FadeIn(frame(a)), FadeIn(frame(b)), FadeIn(frame(avg, GEN_C)), run_time=0.6)
        al = Text("the decoder's safest answer: their average", font_size=20, color=GEN_C).next_to(row, DOWN, buff=0.25)
        self.play(FadeIn(al), run_time=0.3)
        today = VGroup(Text("today: a VAE is rarely the generator", font_size=24),
                       Text("it is the compressor inside Stable Diffusion (episode 13)", font_size=22, color=CODE_C)).arrange(DOWN, buff=0.2).move_to([1.0, -2.4, 0])
        self.at("today")
        self.play(FadeIn(today[0]), run_time=0.4)
        self.at("compressor")
        self.play(FadeIn(today[1]), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 8. code
    def code(self):
        self.section(8)
        self.clear_stage()
        code, hl = code_panel(CODE, font_size=21)
        code.move_to([0, 0.3, 0])
        self.play(FadeIn(code, shift=0.2 * UP), run_time=0.5)
        hl.match_y(code.line_numbers[1])
        self.at("mean")
        self.play(Create(hl), run_time=0.3)
        self.at("code")
        self.play(highlight(hl, code, 2), run_time=0.3)
        self.at("loss")
        self.play(highlight(hl, code, 5), run_time=0.3)
        self.at("rebuilding")
        self.play(highlight(hl, code, 6), run_time=0.3)
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
