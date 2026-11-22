"""How Generative Models Work: Images and Sound, episode 14 — Diffusion Transformers.

Render from the repo root:  ./render.sh generative g14
Numbers and pictures come from code/g14_dit (results_dit_*.json, out/*.png) and code/g13_latent_diffusion.
"""
import json

import numpy as np
from PIL import Image
from manim import *

from common import MONO, code_panel, finish, highlight, next_up_card
from gen_common import tile
from intro import play_token_intro
from g14_script import NEXT, TAGLINE, TITLE
from voiced_scene import VoicedScene

SERIES_LABEL = "HOW GENERATIVE MODELS WORK  ·  EPISODE 14"
CODE_DIR = "code/g14_dit"
OUT = f"{CODE_DIR}/out"
G13 = "code/g13_latent_diffusion"
DIT = json.load(open(f"{CODE_DIR}/results_dit_p2.json"))
P2 = json.load(open(f"{CODE_DIR}/results_dit_p2_10k.json"))
P1 = json.load(open(f"{CODE_DIR}/results_dit_p1_10k.json"))
UNET = json.load(open(f"{G13}/results_latent_words.json"))
LAT_C, TOK_C, TXT_C, COND_C, UNET_C = PURPLE_B, TEAL_C, YELLOW, ORANGE, BLUE_C
CODE = """tokens = embed(patchify(z_noisy)) + position     # 16 tokens
c = time_mlp(t) + text(caption_vector)
for block in blocks:                             # 8 DiT blocks
    tokens = block(tokens, c, word_vectors)      # adaLN-Zero inside
noise_guess = unpatchify(final(tokens, c))
loss = mse(noise_guess, eps)                     # as in episode 6"""


def small(text, color=GREY_B, size=30, scale=0.55):
    return Text(text, font_size=size, color=color).scale(scale)


def pct(v, d=1):
    return f"{v * 100:.{d}f}%"


def box(label, color, w=1.8, h=1.0, size=22):
    b = RoundedRectangle(width=w, height=h, corner_radius=0.15, color=color, fill_color=color, fill_opacity=0.12)
    return VGroup(b, Text(label, font_size=size, color=color).move_to(b))


def latent_array():
    """Channel 1 of the first picture's latent, as drawn in episode 13's out/latents.png (128 × 128 pixels)."""
    arr = np.asarray(Image.open(f"{G13}/out/latents.png").convert("RGB"))
    return arr[130:258, 0:128]


def img(arr, h):
    m = ImageMobject(arr)
    m.set_resampling_algorithm(RESAMPLING_ALGORITHMS["nearest"])
    return m.scale_to_fit_height(h)


def grid_lines(m, n, color=WHITE, width=2):
    g = VGroup()
    w, h = m.width, m.height
    for k in range(1, n):
        g.add(Line(m.get_corner(UL) + RIGHT * w * k / n, m.get_corner(DL) + RIGHT * w * k / n, color=color, stroke_width=width))
        g.add(Line(m.get_corner(UL) + DOWN * h * k / n, m.get_corner(UR) + DOWN * h * k / n, color=color, stroke_width=width))
    return g


def token_bar(color=TOK_C, n=8, seed=0, w=0.1, h=0.22):
    rng = np.random.default_rng(seed)
    return VGroup(*[Rectangle(width=w, height=h, stroke_width=0, fill_color=color, fill_opacity=o)
                    for o in rng.uniform(0.2, 1, n)]).arrange(DOWN, buff=0.01)


class DiTVideo(VoicedScene):
    VIDEO = "g14"

    def construct(self):
        play_token_intro(self, TITLE, 14, TAGLINE, label=SERIES_LABEL)
        self.hook()           # 1
        self.patchify()       # 2
        self.block()          # 3
        self.adaln()          # 4
        self.unpatchify()     # 5
        self.results()        # 6
        self.why()            # 7
        self.patch_size()     # 8
        self.video()          # 9
        self.code()           # 10
        self.outro()          # 11

    def clear_stage(self, *keep, run_time=0.6):
        self.play(*[FadeOut(m) for m in self.mobjects if m not in keep], run_time=run_time)

    # ------------------------------------------------------------------ 1. hook
    def hook(self):
        self.section(1)
        d = Polygon([-1.2, 1.1, 0], [0, 0.3, 0], [0, -0.3, 0], [-1.2, -1.1, 0], color=UNET_C, fill_color=UNET_C, fill_opacity=0.15)
        u = d.copy().flip(RIGHT).next_to(d, RIGHT, buff=0)
        unet = VGroup(d, u).move_to([-3.5, 0.3, 0])
        ul = Text("U-Net (episode 13)", font_size=22, color=UNET_C).next_to(unet, UP, buff=0.3)
        cl = small("convolutions, made for pictures", UNET_C, 28).next_to(unet, DOWN, buff=0.3)
        self.at("unit", "unet")
        self.play(FadeIn(unet), FadeIn(ul), run_time=0.6)
        self.at("convolutions")
        self.play(FadeIn(cl), run_time=0.4)
        stack = VGroup(*[RoundedRectangle(width=2.4, height=0.32, corner_radius=0.08, color=TOK_C, fill_color=TOK_C,
                                          fill_opacity=0.18) for _ in range(6)]).arrange(DOWN, buff=0.08).move_to([3.5, 0.3, 0])
        tl = Text("transformer", font_size=22, color=TOK_C).next_to(stack, UP, buff=0.3)
        ex = small("Stable Diffusion 3, FLUX, Sora", TOK_C, 28).next_to(stack, DOWN, buff=0.3)
        self.at("largest")
        self.play(LaggedStart(*[FadeIn(s) for s in stack], lag_ratio=0.1), FadeIn(tl), run_time=0.7)
        self.at("transformer")
        self.play(FadeIn(ex), run_time=0.4)
        ar = CurvedArrow(unet.get_top() + 0.9 * UP, stack.get_top() + 0.9 * UP, angle=-0.6, color=YELLOW)
        self.at("swap")
        self.play(Create(ar), run_time=0.6)
        self.end_section()

    # ------------------------------------------------------------------ 2. patchify
    def patchify(self):
        self.section(2)
        self.clear_stage()
        arr = latent_array()
        lat = img(arr, 3.2).move_to([-4.4, 0.2, 0])
        ll = small("latent: 8 × 8 × 4 (one channel shown)", LAT_C, 28).next_to(lat, DOWN, buff=0.25)
        self.at("transformer")
        self.play(FadeIn(lat), FadeIn(ll), run_time=0.5)
        grid = grid_lines(lat, 4, YELLOW, 3)
        self.at("cut")
        self.play(Create(grid), run_time=0.6)
        patches = Group(*[img(arr[32 * r:32 * r + 32, 32 * c:32 * c + 32], 0.4) for r in range(4) for c in range(4)])
        for k, p in enumerate(patches):
            p.move_to(lat.get_corner(UL) + (0.4 + 0.8 * (k % 4)) * RIGHT + (0.4 + 0.8 * (k // 4)) * DOWN)
        target = Group(*[p.copy() for p in patches]).arrange(RIGHT, buff=0.08).move_to([2.1, 1.9, 0])
        pl = small("16 patches of 2 × 2 × 4 = 16 numbers", WHITE, 28).next_to(target, UP, buff=0.2)
        self.at("16")
        self.play(*[p.animate.move_to(t) for p, t in zip(patches, target)], FadeIn(pl), run_time=1.0)
        toks = VGroup(*[token_bar(seed=k) for k in range(16)])
        for t, p in zip(toks, patches):
            t.move_to(p.get_center() + 1.6 * DOWN)
        arrows = VGroup(*[Arrow(p.get_bottom(), t.get_top(), buff=0.08, color=GREY_B, stroke_width=2, tip_length=0.12,
                                max_stroke_width_to_length_ratio=10) for p, t in zip(patches, toks)])
        lin = small("linear layer → a token of 128 numbers", TOK_C, 28).next_to(toks, RIGHT, buff=0.3).shift(0.2 * UP)
        lin.next_to(toks, DOWN, buff=0.3)
        self.at("linear")
        self.play(LaggedStart(*[GrowArrow(a) for a in arrows], lag_ratio=0.04), FadeIn(toks), FadeIn(lin), run_time=0.8)
        pos = VGroup(*[Text(str(k + 1), font=MONO, font_size=16, color=COND_C).next_to(t, DOWN, buff=0.1) for k, t in enumerate(toks)])
        lin2 = small("+ a position embedding: where the patch came from", COND_C, 28).next_to(lin, DOWN, buff=0.15)
        self.at("position")
        self.play(FadeIn(pos), lin.animate.shift(0.3 * DOWN), run_time=0.4)
        lin2.next_to(lin, DOWN, buff=0.15)
        self.play(FadeIn(lin2), run_time=0.3)
        self.end_section()

    # ------------------------------------------------------------------ 3. the block
    def block(self):
        self.section(3)
        self.clear_stage()
        toks = VGroup(*[token_bar(seed=k, h=0.16) for k in range(16)]).arrange(RIGHT, buff=0.06).move_to([-1.6, -2.7, 0])
        sa = box("self-attention", TOK_C, 4.2, 0.75).move_to([-1.6, -1.3, 0])
        ca = box("cross-attention", TXT_C, 4.2, 0.75).move_to([-1.6, 0.0, 0])
        mlp = box("MLP on each token", GREEN_C, 4.2, 0.75).move_to([-1.6, 1.3, 0])
        frame_ = SurroundingRectangle(VGroup(sa, ca, mlp), buff=0.25, color=GREY_B, corner_radius=0.15)
        x8 = Text("× 8 blocks", font_size=26, color=GREY_B).next_to(frame_, LEFT, buff=0.3)
        self.at("blocks")
        self.play(FadeIn(toks), Create(frame_), FadeIn(x8), run_time=0.6)
        self.at("self-attention", "self")
        self.play(FadeIn(sa), run_time=0.4)
        lines = VGroup(*[ArcBetweenPoints(toks[i].get_top(), toks[j].get_top(), angle=-PI / 2, color=TOK_C, stroke_width=1.5,
                                          stroke_opacity=0.6) for i, j in ((0, 5), (3, 12), (7, 15), (2, 9), (10, 14))])
        self.at("every")
        self.play(LaggedStart(*[Create(l) for l in lines], lag_ratio=0.1), run_time=0.7)
        words = VGroup(*[Text(w, font=MONO, font_size=20, color=TXT_C) for w in "a red triangle in the center".split()]
                       ).arrange(DOWN, buff=0.18, aligned_edge=LEFT).move_to([4.6, 0.0, 0])
        wl = small("CLIP word vectors", TXT_C, 28).next_to(words, UP, buff=0.25)
        self.at("cross-attention", "cross")
        self.play(FadeIn(ca), run_time=0.4)
        self.at("words")
        self.play(FadeIn(words), FadeIn(wl), GrowArrow(Arrow(words.get_left(), ca.get_right(), buff=0.25, color=TXT_C)), run_time=0.6)
        self.at("network")
        self.play(FadeIn(mlp), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 4. adaLN-Zero
    def adaln(self):
        self.section(4)
        self.clear_stage()
        t = box("time t", COND_C, 1.6, 0.6, 20).move_to([-5.2, 1.0, 0])
        cap = box("caption", TXT_C, 1.6, 0.6, 20).move_to([-5.2, -0.2, 0])
        c = box("c", COND_C, 0.8, 0.8, 26).move_to([-3.0, 0.4, 0])
        self.at("time")
        self.play(FadeIn(t), FadeIn(cap), run_time=0.4)
        self.at("steer")
        self.play(FadeIn(c), GrowArrow(Arrow(t.get_right(), c.get_left(), buff=0.1, color=GREY_B)),
                  GrowArrow(Arrow(cap.get_right(), c.get_left(), buff=0.1, color=GREY_B)), run_time=0.5)
        rows_ = []
        for k, (name, col) in enumerate((("self-attention", TOK_C), ("MLP", GREEN_C))):
            y = 1.2 - 1.8 * k
            norm = box("norm", GREY_B, 1.2, 0.6, 20).move_to([-0.6, y, 0])
            ss = small("× scale + shift", COND_C, 26).next_to(norm, DOWN, buff=0.12)
            br = box(name, col, 2.4, 0.6, 20).move_to([2.0, y, 0])
            gate = box("gate", COND_C, 1.0, 0.6, 20).move_to([4.2, y, 0])
            gv = Text("0", font=MONO, font_size=22, color=RED_B).move_to([5.15, y, 0])
            plus = Text("+", font_size=32).move_to([5.95, y, 0])
            a1 = Arrow(norm.get_right(), br.get_left(), buff=0.08, color=GREY_B)
            a2 = Arrow(br.get_right(), gate.get_left(), buff=0.08, color=GREY_B)
            feed = DashedLine(c.get_right(), norm.get_left(), color=COND_C, stroke_width=2)
            feed2 = DashedLine(c.get_right(), gate.get_top() + 0.05 * UP, color=COND_C, stroke_width=2, stroke_opacity=0.5)
            rows_.append((norm, ss, br, gate, gv, plus, a1, a2, feed, feed2))
        self.at("scale")
        self.play(*[FadeIn(m) for r in rows_ for m in (r[0], r[1], r[2], r[6], r[8])], run_time=0.6)
        self.at("gate")
        self.play(*[FadeIn(m) for r in rows_ for m in (r[3], r[7], r[9], r[5])], run_time=0.5)
        self.at("zero")
        self.play(*[FadeIn(r[4]) for r in rows_], run_time=0.3)
        note = small("each block starts as “do nothing”: output = input", WHITE, 30).to_edge(DOWN, buff=0.5)
        self.at("nothing")
        self.play(FadeIn(note), run_time=0.4)
        self.at("opens")
        self.play(*[Transform(r[4], Text("0.8" if k == 0 else "0.6", font=MONO, font_size=22, color=GREEN).move_to(r[4]))
                    for k, r in enumerate(rows_)], run_time=0.6)
        self.end_section()

    # ------------------------------------------------------------------ 5. unpatchify
    def unpatchify(self):
        self.section(5)
        self.clear_stage()
        toks = VGroup(*[token_bar(seed=k + 30) for k in range(16)]).arrange(RIGHT, buff=0.14).move_to([-1.2, 2.2, 0])
        self.at("end")
        self.play(FadeIn(toks), run_time=0.4)
        rng = np.random.default_rng(3)
        noise = (np.clip(rng.normal(0.5, 0.22, (8, 8, 3)), 0, 1) * 255).astype(np.uint8)
        big = np.kron(noise, np.ones((16, 16, 1), dtype=np.uint8))
        pieces = Group(*[img(big[32 * r:32 * r + 32, 32 * c:32 * c + 32], 0.55) for r in range(4) for c in range(4)])
        for p, t in zip(pieces, toks):
            p.move_to(t.get_center() + 1.1 * DOWN)
        self.at("patch")
        self.play(LaggedStart(*[FadeIn(p, shift=0.3 * DOWN) for p in pieces], lag_ratio=0.04), run_time=0.7)
        target = [np.array([-1.2 - 0.825 + 0.55 * (k % 4), -0.2 - 0.55 * (k // 4), 0]) for k in range(16)]
        lab = small("predicted noise, 8 × 8 × 4", WHITE, 28).move_to([-1.2, -2.7, 0])
        self.at("noise")
        self.play(*[p.animate.move_to(tg) for p, tg in zip(pieces, target)], FadeIn(lab), run_time=0.8)
        same = VGroup(*[small(s, GREY_B, 30) for s in ("same loss", "same sampler", "same 40,000 training steps")]
                      ).arrange(DOWN, aligned_edge=LEFT, buff=0.25).move_to([4.4, -0.8, 0])
        self.at("loss")
        self.play(FadeIn(same[0]), run_time=0.3)
        self.at("sampler")
        self.play(FadeIn(same[1]), run_time=0.3)
        self.at("40000", "40")
        self.play(FadeIn(same[2]), run_time=0.3)
        self.end_section()

    # ------------------------------------------------------------------ 6. results
    def results(self):
        self.section(6)
        self.clear_stage()
        U, Dt = UNET["eval"], DIT["eval"]
        hdr = VGroup(Text("captions", font_size=18, color=GREY_B), Text("U-Net", font_size=22, color=UNET_C), Text("transformer", font_size=22, color=TOK_C))
        rows_ = [hdr]
        for name, key in (("trained, with guidance (w = 3)", ("w3", "trained")), ("never seen, with guidance", ("w3", "held_out")),
                          ("trained, no guidance (w = 1)", ("w1", "trained")), ("never seen, no guidance", ("w1", "held_out"))):
            a, b = U[key[0]][key[1]]["follows"], Dt[key[0]][key[1]]["follows"]
            rows_.append(VGroup(Text(name, font_size=20), Text(pct(a), font=MONO, font_size=22, color=GREEN if a > b + 0.005 else WHITE),
                                Text(pct(b), font=MONO, font_size=22, color=GREEN if b > a + 0.005 else WHITE)))
        for j, r_ in enumerate(rows_):
            for k, c in enumerate(r_):
                c.move_to([-3.0 + 3.6 * k if k else -3.4, 2.7 - 0.55 * j, 0])
            r_[0].move_to([-5.8, r_[0].get_y(), 0], aligned_edge=LEFT)
        hl = small("share of captions followed, 40,000 steps each", GREY_B, 26).to_edge(UP, buff=0.15)
        self.at("guidance")
        self.play(FadeIn(hl), FadeIn(rows_[0]), FadeIn(rows_[1]), FadeIn(rows_[2]), run_time=0.6)
        self.at("99")
        self.play(Indicate(VGroup(rows_[1][1:], rows_[2][1:]), color=YELLOW, scale_factor=1.05), run_time=0.6)
        ru = Group(*[tile(f"{G13}/out/words_held_out_w1.png", i, 0.75, row=0) for i in range(8)]).arrange(RIGHT, buff=0.06)
        rd = Group(*[tile(f"{OUT}/dit_p2_held_out_w1.png", i, 0.75, row=0) for i in range(8)]).arrange(RIGHT, buff=0.06)
        Group(ru, rd).arrange(DOWN, buff=0.2).move_to([1.0, -2.3, 0])
        lu = Text("U-Net", font_size=18, color=UNET_C).next_to(ru, LEFT, buff=0.3)
        ld = Text("transformer", font_size=18, color=TOK_C).next_to(rd, LEFT, buff=0.3)
        self.at("without")
        self.play(FadeIn(rows_[3]), FadeIn(rows_[4][0]), run_time=0.5)
        self.at("98")
        self.play(FadeIn(rows_[4][1]), FadeIn(ru), FadeIn(lu), run_time=0.5)
        self.at("96")
        self.play(FadeIn(rows_[4][2]), FadeIn(rd), FadeIn(ld), run_time=0.5)
        self.end_section()

    # ------------------------------------------------------------------ 7. why
    def why(self):
        self.section(7)
        self.clear_stage()
        cells = VGroup(*[Square(side_length=0.32, stroke_color=GREY_D, stroke_width=1.5, fill_color=UNET_C, fill_opacity=0.08)
                         for _ in range(64)]).arrange_in_grid(rows=8, cols=8, buff=0).move_to([-4.6, 1.3, 0])
        k3 = VGroup(*[cells[r * 8 + c] for r in (2, 3, 4) for c in (3, 4, 5)])
        cl = small("a convolution sees only neighbors", UNET_C, 28).next_to(cells, DOWN, buff=0.2)
        self.at("neighboring")
        self.play(FadeIn(cells), k3.animate.set_fill(UNET_C, 0.7), FadeIn(cl), run_time=0.6)
        toks = VGroup(*[Square(side_length=0.24, stroke_color=GREY_D, stroke_width=1.5, fill_color=TOK_C, fill_opacity=0.2)
                        for _ in range(16)]).arrange(RIGHT, buff=0.05).move_to([-4.6, -1.0, 0])
        tl = small("tokens: layout learned from positions", TOK_C, 28).next_to(toks, DOWN, buff=0.2)
        self.at("layout")
        self.play(FadeIn(toks), FadeIn(tl), run_time=0.5)
        ax = Axes(x_range=[0, 40, 10], y_range=[0, 0.08, 0.02], x_length=6.0, y_length=4.0, tips=False,
                  axis_config={"color": GREY_C, "include_numbers": False}).move_to([2.6, 0.2, 0])
        xl = VGroup(*[Text(f"{v}k", font=MONO, font_size=16).next_to(ax.c2p(v, 0), DOWN, buff=0.12) for v in (0, 10, 20, 30, 40)])
        yl = VGroup(*[Text(f"{v:.2f}", font=MONO, font_size=16).next_to(ax.c2p(0, v), LEFT, buff=0.12) for v in (0.02, 0.04, 0.06, 0.08)])
        cap = small("training loss (thousands of steps)", GREY_B, 26).next_to(ax, UP, buff=0.2)

        def curve(lc, col):
            pts = [ax.c2p(0.2 * (i + 1), min(v, 0.08)) for i, v in enumerate(lc) if i >= 4]
            return VMobject(color=col, stroke_width=3).set_points_smoothly(pts[::2])
        cu, cd = curve(UNET["loss_curve"], UNET_C), curve(DIT["loss_curve"], TOK_C)
        lu = Text(f"U-Net {UNET['loss_curve'][-1]:.4f}", font=MONO, font_size=16, color=UNET_C).move_to(ax.c2p(30, 0.066))
        ld = Text(f"transformer {DIT['loss_curve'][-1]:.4f}", font=MONO, font_size=16, color=TOK_C).move_to(ax.c2p(30, 0.074))
        self.play(Create(ax), FadeIn(xl), FadeIn(yl), FadeIn(cap), Create(cu), FadeIn(lu), run_time=0.8)
        self.at("falling")
        self.play(Create(cd), FadeIn(ld), run_time=0.8)
        sc = Text("strength: scale", font_size=26, color=YELLOW).to_edge(DOWN, buff=0.35)
        self.at("scale")
        self.play(FadeIn(sc), run_time=0.4)
        self.at("grew")
        self.play(sc.animate.become(Text("strength: scale. DiT paper: quality kept improving as the model grew",
                                         font_size=24, color=YELLOW).to_edge(DOWN, buff=0.35)), run_time=0.5)
        self.end_section()

    # ------------------------------------------------------------------ 8. patch size
    def patch_size(self):
        self.section(8)
        self.clear_stage()
        arr = latent_array()
        a = img(arr, 2.0).move_to([-5.0, 1.3, 0])
        b = img(arr, 2.0).move_to([-2.2, 1.3, 0])
        ga, gb = grid_lines(a, 4, YELLOW, 2), grid_lines(b, 8, YELLOW, 1.5)
        la = Text("2 × 2 patches\n16 tokens", font_size=18, color=TOK_C).next_to(a, DOWN, buff=0.2)
        lb = Text("1 × 1 patches\n64 tokens", font_size=18, color=TOK_C).next_to(b, DOWN, buff=0.2)
        self.at("smaller")
        self.play(FadeIn(a), Create(ga), FadeIn(la), run_time=0.6)
        self.at("64")
        self.play(FadeIn(b), Create(gb), FadeIn(lb), run_time=0.6)
        r2, r1 = P2["eval"]["w1"], P1["eval"]["w1"]
        tbl = VGroup(
            VGroup(Text("after 10,000 steps", font_size=18, color=GREY_B), Text("16 tokens", font_size=18, color=TOK_C), Text("64 tokens", font_size=18, color=TOK_C)),
            VGroup(Text("time per step", font_size=18), Text("1×", font=MONO, font_size=20), Text("~3×", font=MONO, font_size=20, color=RED_B)),
            VGroup(Text("trained captions", font_size=18), Text(pct(r2["trained"]["follows"]), font=MONO, font_size=20),
                   Text(pct(r1["trained"]["follows"]), font=MONO, font_size=20)),
            VGroup(Text("never seen", font_size=18), Text(pct(r2["held_out"]["follows"]), font=MONO, font_size=20),
                   Text(pct(r1["held_out"]["follows"]), font=MONO, font_size=20)))
        for j, r_ in enumerate(tbl):
            for k, c in enumerate(r_):
                c.move_to([2.8 + 1.6 * k if k else 2.0, 2.2 - 0.55 * j, 0])
            r_[0].move_to([0.6, r_[0].get_y(), 0], aligned_edge=LEFT)
        self.at("longer")
        self.play(FadeIn(tbl[0]), FadeIn(tbl[1]), run_time=0.5)
        self.at("better")
        self.play(FadeIn(tbl[2]), FadeIn(tbl[3]), run_time=0.5)
        n = 8
        dots = VGroup(*[Dot(radius=0.06, color=TOK_C) for _ in range(n)]).arrange(RIGHT, buff=0.35).move_to([-2.6, -2.0, 0])
        lines = VGroup(*[Line(dots[i].get_center(), dots[j].get_center(), stroke_width=1, color=TOK_C, stroke_opacity=0.5).set_points_smoothly(
            [dots[i].get_center(), (dots[i].get_center() + dots[j].get_center()) / 2 + 0.12 * (j - i) * UP, dots[j].get_center()])
                         for i in range(n) for j in range(i + 1, n)])
        sq = Text("attention: every token × every token → cost grows with tokens²", font_size=20, color=WHITE).move_to([1.6, -3.0, 0])
        self.at("compares")
        self.play(FadeIn(dots), LaggedStart(*[Create(l) for l in lines], lag_ratio=0.02), run_time=0.9)
        self.at("square")
        self.play(FadeIn(sq), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 9. beyond pictures
    def video(self):
        self.section(9)
        self.clear_stage()
        frames = Group(*[tile(f"{OUT}/dit_p2_prompts.png", i, 1.6, row=1) for i in range(4)])
        for k, f in enumerate(frames):
            f.move_to([-4.4 + 0.45 * k, 0.6 - 0.35 * k, 0])
        fl = small("frames of a video", WHITE, 28).next_to(frames, DOWN, buff=0.3)
        self.at("tokens")
        self.play(LaggedStart(*[FadeIn(f) for f in frames], lag_ratio=0.15), run_time=0.6)
        self.at("beyond")
        self.play(FadeIn(fl), run_time=0.3)
        cube = VGroup()
        for t in range(3):
            for r in range(3):
                for c in range(3):
                    s = Square(side_length=0.6, stroke_color=TOK_C, stroke_width=2, fill_color=TOK_C, fill_opacity=0.12)
                    s.move_to([1.0 + 0.62 * c + 0.3 * t, 1.0 - 0.62 * r + 0.3 * t, 0])
                    cube.add(s)
        cl = small("spacetime patches: blocks across space and time", TOK_C, 28).next_to(cube, DOWN, buff=0.35)
        src = small("OpenAI, Sora technical report (2024)", GREY_C, 24).next_to(cl, DOWN, buff=0.15)
        self.at("sora")
        self.play(FadeIn(src), run_time=0.3)
        self.at("spacetime")
        self.play(LaggedStart(*[FadeIn(s) for s in cube], lag_ratio=0.02), FadeIn(cl), run_time=0.8)
        tk = VGroup(*[token_bar(seed=k + 60, h=0.14, n=6) for k in range(14)]).arrange(RIGHT, buff=0.05).next_to(cube, RIGHT, buff=0.5)
        self.at("time")
        self.play(FadeIn(tk), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 10. code
    def code(self):
        self.section(10)
        self.clear_stage()
        code, hl = code_panel(CODE, font_size=18)
        code.move_to([0, 0.2, 0])
        self.play(FadeIn(code, shift=0.2 * UP), run_time=0.4)
        self.play(Create(hl), run_time=0.2)
        self.at("patchify")
        self.play(highlight(hl, code, 0), run_time=0.3)
        self.at("blocks")
        self.play(highlight(hl, code, 3), run_time=0.3)
        self.at("caption")
        self.play(highlight(hl, code, 1), run_time=0.3)
        self.at("un", "unpatchify")
        self.play(highlight(hl, code, 4), run_time=0.3)
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
