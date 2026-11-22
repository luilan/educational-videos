"""How Generative Models Work: Images and Sound, episode 13 — Latent Diffusion.

Render from the repo root:  ./render.sh generative g13
Numbers and pictures come from code/g13_latent_diffusion (results_autoencoder.json, results_latent_*.json, out/*.png).
"""
import json

import numpy as np
from manim import *

from common import MONO, code_panel, finish, highlight, next_up_card
from gen_common import tile
from intro import play_token_intro
from g13_script import NEXT, TAGLINE, TITLE
from voiced_scene import VoicedScene

SERIES_LABEL = "HOW GENERATIVE MODELS WORK  ·  EPISODE 13"
CODE_DIR = "code/g13_latent_diffusion"
OUT = f"{CODE_DIR}/out"
AE = json.load(open(f"{CODE_DIR}/results_autoencoder.json"))
WORDS = json.load(open(f"{CODE_DIR}/results_latent_words.json"))
POOL = json.load(open(f"{CODE_DIR}/results_latent_pooled.json"))
CLIP4 = json.load(open(f"{CODE_DIR}/results_latent_words_clip4.json"))
ENC_C, LAT_C, DEC_C, TXT_C = BLUE_C, PURPLE_B, GREEN_C, YELLOW
CODE = """z = autoencoder.encode(pictures)            # once, before training
# training: episode 6's loop, on z instead of pixels
loss = mse(unet(noise(z, t, eps), t, caption), eps)

# drawing: denoise a random latent, decode once
z = sample(unet, caption, torch.randn(n, 4, 8, 8))
pictures = autoencoder.decode(z)"""


def row(path, n=8, h=1.0, r=0, buff=0.1, start=0):
    return Group(*[tile(path, start + i, h, row=r) for i in range(n)]).arrange(RIGHT, buff=buff)


def small(text, color=GREY_B, size=30, scale=0.55):
    return Text(text, font_size=size, color=color).scale(scale)


def pct(v, d=1):
    return f"{v * 100:.{d}f}%"


def box(label, color, w=1.8, h=1.0, size=22):
    b = RoundedRectangle(width=w, height=h, corner_radius=0.15, color=color, fill_color=color, fill_opacity=0.12)
    return VGroup(b, Text(label, font_size=size, color=color).move_to(b))


def trapezoid(color, wide_left=True, w=1.4, h1=1.6, h2=0.8):
    a, b = (h1, h2) if wide_left else (h2, h1)
    p = Polygon([-w / 2, a / 2, 0], [w / 2, b / 2, 0], [w / 2, -b / 2, 0], [-w / 2, -a / 2, 0],
                color=color, fill_color=color, fill_opacity=0.15)
    return p


class LatentDiffusionVideo(VoicedScene):
    VIDEO = "g13"

    def construct(self):
        play_token_intro(self, TITLE, 13, TAGLINE, label=SERIES_LABEL)
        self.hook()           # 1
        self.autoencoder()    # 2
        self.latents()        # 3
        self.penalty()        # 4
        self.diffuse()        # 5
        self.text_in()        # 6
        self.cross()          # 7
        self.results()        # 8
        self.trap()           # 9
        self.real()           # 10
        self.code()           # 11
        self.outro()          # 12

    def clear_stage(self, *keep, run_time=0.6):
        self.play(*[FadeOut(m) for m in self.mobjects if m not in keep], run_time=run_time)

    # ------------------------------------------------------------------ 1. hook
    def hook(self):
        self.section(1)
        p = tile(f"{OUT}/recon.png", 0, 0.5).move_to([-3.8, 1.6, 0])
        pl = Text("32 × 32 × 3 = 3,072 numbers", font=MONO, font_size=20).next_to(p, DOWN, buff=0.25)
        self.at("32")
        self.play(FadeIn(p), FadeIn(pl), run_time=0.5)
        big = Square(side_length=0.5 * 16, color=GREY_B, stroke_width=2).scale(0.5)
        big.move_to([2.2, 0.2, 0])
        bl = Text("512 × 512 × 3 = 786,432 numbers", font=MONO, font_size=20).next_to(big, DOWN, buff=0.25)
        self.at("512")
        self.play(Create(big), FadeIn(bl), run_time=0.8)
        cost = small("a U-Net over every number,\n50 times per picture", WHITE, 32).move_to([-3.8, -0.6, 0])
        self.at("running")
        self.play(FadeIn(cost), run_time=0.4)
        no = Text("so don't diffuse the pixels", font_size=28, color=YELLOW).to_edge(DOWN, buff=0.4)
        self.at("diffuse")
        self.play(FadeIn(no), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 2. autoencoder
    def autoencoder(self):
        self.section(2)
        self.clear_stage()
        orig = Group(*[tile(f"{OUT}/recon.png", i, 0.75, row=0) for i in range(4)]).arrange(DOWN, buff=0.08).move_to([-5.6, 0, 0])
        rec = Group(*[tile(f"{OUT}/recon.png", i, 0.75, row=1) for i in range(4)]).arrange(DOWN, buff=0.08).move_to([5.6, 0, 0])
        enc = trapezoid(ENC_C, True).move_to([-3.4, 0, 0])
        dec = trapezoid(DEC_C, False).move_to([3.4, 0, 0])
        el = Text("encoder", font_size=20, color=ENC_C).next_to(enc, UP, buff=0.2)
        dl = Text("decoder", font_size=20, color=DEC_C).next_to(dec, UP, buff=0.2)
        lat = Group(*[tile(f"{OUT}/latents.png", 0, 0.7, row=1 + k) for k in range(4)])
        for k, m in enumerate(lat):
            m.move_to([0.2 * k - 0.3, 0.2 * k - 0.3, 0])
        latl = Text("8 × 8 × 4 = 256 numbers", font=MONO, font_size=20, color=LAT_C).move_to([0, -1.3, 0])
        latl2 = small("12× fewer than 3,072", LAT_C, 30).next_to(latl, DOWN, buff=0.12)
        a = [Arrow(orig.get_right(), enc.get_left(), buff=0.15, color=GREY_B), Arrow(enc.get_right(), lat.get_left(), buff=0.2, color=GREY_B),
             Arrow(lat.get_right(), dec.get_left(), buff=0.2, color=GREY_B), Arrow(dec.get_right(), rec.get_left(), buff=0.15, color=GREY_B)]
        self.at("autoencoder")
        self.play(FadeIn(orig), run_time=0.4)
        self.at("encoder")
        self.play(GrowArrow(a[0]), FadeIn(enc), FadeIn(el), run_time=0.5)
        self.at("8x8", "grid")
        self.play(GrowArrow(a[1]), LaggedStart(*[FadeIn(m) for m in lat], lag_ratio=0.15), run_time=0.7)
        self.at("256")
        self.play(FadeIn(latl), run_time=0.4)
        self.at("12")
        self.play(FadeIn(latl2), run_time=0.3)
        self.at("decoder")
        self.play(GrowArrow(a[2]), FadeIn(dec), FadeIn(dl), GrowArrow(a[3]), FadeIn(rec), run_time=0.7)
        res = Text(f"{pct(AE['same_reading_as_original'])} of rebuilt test pictures read like the originals",
                   font_size=22, color=GREEN).to_edge(DOWN, buff=0.4)
        self.at("99")
        self.play(FadeIn(res), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 3. latents
    def latents(self):
        self.section(3)
        self.clear_stage()
        g = Group()
        for r_ in range(5):
            rr = Group(*[tile(f"{OUT}/latents.png", i, 0.95, row=r_) for i in range(4)]).arrange(RIGHT, buff=0.12)
            lab = Text("picture" if r_ == 0 else f"channel {r_}", font_size=20, color=WHITE if r_ == 0 else LAT_C).next_to(rr, LEFT, buff=0.35)
            g.add(Group(lab, rr))
        g.arrange(DOWN, buff=0.1, aligned_edge=RIGHT).move_to([-0.6, -0.1, 0])
        key = VGroup(small("red: positive", RED_C, 28), small("blue: negative", BLUE_C, 28)).arrange(DOWN, aligned_edge=LEFT).move_to([4.6, 1.5, 0])
        self.at("latent")
        self.play(FadeIn(g[0]), run_time=0.4)
        self.at("channel")
        self.play(LaggedStart(*[FadeIn(x) for x in g[1:]], lag_ratio=0.15), FadeIn(key), run_time=0.9)
        note = Text("still a map:\nconvolutions\nstill make sense", font_size=22, color=YELLOW, line_spacing=0.8).move_to([4.6, -1.0, 0])
        self.at("unit")
        self.play(FadeIn(note), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 4. penalty
    def penalty(self):
        self.section(4)
        self.clear_stage()
        rows_ = [("original", f"{OUT}/recon.png", 0, WHITE), ("penalty too strong (1e-4)", f"{OUT}/recon_kl1e-4.png", 1, RED_B),
                 ("very light penalty (1e-6)", f"{OUT}/recon.png", 1, GREEN)]
        g = Group()
        for txt, path, r_, col in rows_:
            rr = row(path, h=1.0, r=r_, buff=0.08)
            g.add(Group(Text(txt, font_size=20, color=col).next_to(rr, LEFT, buff=0.3), rr))
        g.arrange(DOWN, buff=0.3, aligned_edge=RIGHT).move_to([0.6, 0.2, 0])
        self.at("vae")
        self.play(FadeIn(g[0]), run_time=0.5)
        self.at("strong")
        self.play(FadeIn(g[1]), run_time=0.5)
        self.at("light")
        self.play(FadeIn(g[2]), run_time=0.5)
        self.end_section()

    # ------------------------------------------------------------------ 5. diffusion on latents
    def diffuse(self):
        self.section(5)
        self.clear_stage()
        rng = np.random.default_rng(5)
        nz = ImageMobject((np.clip(rng.normal(0.5, 0.25, (8, 8, 3)), 0, 1) * 255).astype(np.uint8))
        nz.set_resampling_algorithm(RESAMPLING_ALGORITHMS["nearest"]).scale_to_fit_height(1.1).move_to([-5.4, 1.2, 0])
        nl = small("random 8 × 8 × 4 latent", GREY_B, 26).next_to(nz, DOWN, buff=0.15)
        unet = box("U-Net\n× 50 steps", LAT_C, 2.2, 1.2).move_to([-2.4, 1.2, 0])
        lat = tile(f"{OUT}/latents.png", 0, 1.1, row=3).move_to([0.6, 1.2, 0])
        dec = box("decoder\nonce", DEC_C, 1.8, 1.2).move_to([3.0, 1.2, 0])
        pic = tile(f"{OUT}/words_prompts.png", 0, 1.1, row=0).move_to([5.4, 1.2, 0])
        ar = [Arrow(a.get_right(), b.get_left(), buff=0.12, color=GREY_B) for a, b in ((nz, unet), (unet, lat), (lat, dec), (dec, pic))]
        loss = small("training: add noise, guess the noise (episode 6's loss), on latents", WHITE, 32).move_to([0, 2.8, 0])
        self.at("diffusion")
        self.play(FadeIn(loss), run_time=0.4)
        self.at("draw")
        self.play(FadeIn(nz), FadeIn(nl), GrowArrow(ar[0]), FadeIn(unet), GrowArrow(ar[1]), FadeIn(lat), run_time=0.8)
        self.at("decode")
        self.play(GrowArrow(ar[2]), FadeIn(dec), GrowArrow(ar[3]), FadeIn(pic), run_time=0.6)
        C = WORDS["cost"]
        mx = C["pixel_ms_per_step"]
        bars = VGroup()
        for k, (name, ms, params, col) in enumerate((("pixel U-Net", C["pixel_ms_per_step"], C["pixel_unet_params"], GREY_B),
                                                     ("latent U-Net", C["latent_ms_per_step"], C["latent_unet_params"], LAT_C))):
            b = Rectangle(width=6.0 * ms / mx, height=0.4, stroke_width=0, fill_color=col, fill_opacity=0.85)
            b.move_to([-2.6, -1.3 - 0.75 * k, 0], aligned_edge=LEFT)
            lab = Text(f"{name}  ({params / 1e6:.1f}M parameters)", font_size=18, color=col).next_to(b, LEFT, buff=0.2)
            val = Text(f"{ms:.0f} ms", font=MONO, font_size=18).next_to(b, RIGHT, buff=0.15)
            bars.add(VGroup(b, lab, val))
        bars.shift(1.3 * RIGHT)
        bh = small("one training step, batch of 128, CPU", GREY_B, 26).next_to(bars, UP, buff=0.2)
        self.at("twice")
        self.play(FadeIn(bh), FadeIn(bars), run_time=0.6)
        self.end_section()

    # ------------------------------------------------------------------ 6. text in
    def text_in(self):
        self.section(6)
        self.clear_stage()
        cap = Text("“a red triangle in the center”", font_size=26, color=TXT_C).move_to([-3.6, 1.6, 0])
        enc = box("CLIP text encoder\n(episode 12, frozen)", TXT_C, 3.4, 1.1, 20).move_to([-3.6, 0, 0])
        rng = np.random.default_rng(2)
        vec = VGroup(*[Rectangle(width=0.16, height=0.32, stroke_width=0, fill_color=TXT_C, fill_opacity=o)
                       for o in rng.uniform(0.15, 1, 12)]).arrange(RIGHT, buff=0.02).move_to([-3.6, -1.5, 0])
        vl = small("one caption vector (64 numbers)", TXT_C, 28).next_to(vec, DOWN, buff=0.15)
        self.at("sentence")
        self.play(FadeIn(cap), run_time=0.4)
        self.at("clip")
        self.play(FadeIn(enc), GrowArrow(Arrow(cap.get_bottom(), enc.get_top(), buff=0.1, color=GREY_B)), run_time=0.5)
        self.at("simple")
        a2 = Arrow(enc.get_bottom(), vec.get_top(), buff=0.1, color=GREY_B)
        self.play(GrowArrow(a2), FadeIn(vec), FadeIn(vl), run_time=0.5)
        tchip = box("time embedding", PURPLE_B, 2.6, 0.6, 20).move_to([2.6, -1.5, 0])
        plus = Text("+", font_size=40).move_to([0.2, -1.5, 0])
        note = small("like episode 10's labels", GREY_B, 30).next_to(tchip, DOWN, buff=0.25)
        self.at("time")
        self.play(FadeIn(plus), FadeIn(tchip), run_time=0.4)
        self.at("labels")
        self.play(FadeIn(note), run_time=0.3)
        self.end_section()

    # ------------------------------------------------------------------ 7. cross-attention
    def cross(self):
        self.section(7)
        self.clear_stage()
        cells = VGroup(*[Square(side_length=0.42, stroke_color=GREY_D, stroke_width=1.5, fill_color=LAT_C, fill_opacity=0.12)
                         for _ in range(64)]).arrange_in_grid(rows=8, cols=8, buff=0).move_to([-3.6, 0, 0])
        gl = small("latent positions (8 × 8)", LAT_C, 28).next_to(cells, DOWN, buff=0.25)
        words = "a red triangle in the center".split()
        wv = VGroup(*[Text(w, font=MONO, font_size=22, color=TXT_C) for w in words]).arrange(DOWN, buff=0.28, aligned_edge=LEFT).move_to([3.6, 0, 0])
        wl = small("one CLIP vector per word", TXT_C, 28).next_to(wv, DOWN, buff=0.3)
        self.at("inside")
        self.play(FadeIn(cells), FadeIn(gl), run_time=0.5)
        q = cells[3 * 8 + 4]
        self.at("looks")
        self.play(q.animate.set_fill(YELLOW, 0.8), run_time=0.3)
        self.at("vectors")
        self.play(LaggedStart(*[FadeIn(w) for w in wv], lag_ratio=0.1), FadeIn(wl), run_time=0.6)
        weights = [0.05, 0.35, 0.3, 0.05, 0.05, 0.2]
        lines = VGroup(*[Line(q.get_center(), w.get_left() + 0.1 * LEFT, color=YELLOW, stroke_width=1 + 14 * a, stroke_opacity=0.35 + a)
                         for w, a in zip(wv, weights)])
        self.play(LaggedStart(*[Create(l) for l in lines], lag_ratio=0.08), run_time=0.7)
        lab = Text("cross-attention", font_size=30, color=YELLOW).to_edge(UP, buff=0.5)
        ill = small("weights drawn for illustration", GREY_C, 26).to_edge(DOWN, buff=0.3)
        self.at("cross")
        self.play(FadeIn(lab), FadeIn(ill), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 8. results
    def results(self):
        self.section(8)
        self.clear_stage()
        Pw, Ww = POOL["eval"]["w1"], WORDS["eval"]["w1"]
        tbl = VGroup()
        hdr = VGroup(Text("", font_size=20), Text("one caption vector", font_size=20, color=TXT_C),
                     Text("cross-attention", font_size=20, color=YELLOW))
        r1 = VGroup(Text("127 trained combinations", font_size=20), Text(pct(Pw["trained"]["follows"]), font=MONO, font_size=22),
                    Text(pct(Ww["trained"]["follows"]), font=MONO, font_size=22))
        r2 = VGroup(Text("8 never-seen combinations", font_size=20), Text(pct(Pw["held_out"]["follows"]), font=MONO, font_size=22),
                    Text(pct(Ww["held_out"]["follows"]), font=MONO, font_size=22, color=GREEN))
        for j, r_ in enumerate((hdr, r1, r2)):
            for k, c in enumerate(r_):
                c.move_to([-3.4 + 3.6 * k, 2.6 - 0.6 * j, 0])
            tbl.add(r_)
        self.at("both")
        self.play(FadeIn(hdr), FadeIn(r1), run_time=0.6)
        rp = row(f"{OUT}/pooled_held_out_w1.png", h=0.9, r=0, buff=0.07)
        rw = row(f"{OUT}/words_held_out_w1.png", h=0.9, r=0, buff=0.07)
        Group(rp, rw).arrange(DOWN, buff=0.35).move_to([1.0, -1.3, 0])
        lp = Text("one vector", font_size=20, color=TXT_C).next_to(rp, LEFT, buff=0.3)
        lw = Text("cross-attention", font_size=20, color=YELLOW).next_to(rw, LEFT, buff=0.3)
        ask = small("never seen: “a purple triangle at the top left”", WHITE, 30).next_to(rp, UP, buff=0.2)
        self.at("eight")
        self.play(FadeIn(r2[0]), FadeIn(ask), FadeIn(lp), FadeIn(rp), run_time=0.6)
        self.at("94")
        self.play(FadeIn(r2[1]), run_time=0.3)
        self.at("cross")
        self.play(FadeIn(r2[2]), FadeIn(lw), FadeIn(rw), run_time=0.5)
        self.end_section()

    # ------------------------------------------------------------------ 9. the trap
    def trap(self):
        self.section(9)
        self.clear_stage()
        nl = NumberLine(x_range=[-13, 13, 1], length=11, include_numbers=False, color=GREY_C).move_to([0, 1.8, 0])
        nums = VGroup(*[Text(str(v), font=MONO, font_size=16).next_to(nl.n2p(v), DOWN, buff=0.15) for v in (-12, -4, 0, 4, 12)])
        pix = Rectangle(width=nl.n2p(1)[0] - nl.n2p(-1)[0], height=0.5, stroke_width=0, fill_color=GREEN, fill_opacity=0.4).move_to(nl.n2p(0))
        pl = small("pixels: −1 … 1", GREEN, 28).next_to(pix, UP, buff=0.25)
        self.at("pixels")
        self.play(Create(nl), FadeIn(nums), FadeIn(pix), FadeIn(pl), run_time=0.7)
        clip = Rectangle(width=nl.n2p(4)[0] - nl.n2p(-4)[0], height=0.5, stroke_color=RED_B, stroke_width=3).move_to(nl.n2p(0))
        cl = small("our clip: −4 … 4", RED_B, 28).next_to(clip, UP, buff=0.25).shift(2.2 * RIGHT)
        self.at("four")
        self.play(Create(clip), FadeIn(cl), run_time=0.5)
        rng = np.random.default_rng(1)
        vals = np.concatenate([rng.normal(0, 1, 40), rng.uniform(5, 12, 6), -rng.uniform(5, 9, 3)])
        dots = VGroup(*[Dot(nl.n2p(v) + 0.0 * UP, radius=0.05, color=LAT_C if abs(v) < 4 else YELLOW) for v in vals])
        dl = small("latent values: the shape's own reach ±12", YELLOW, 28).next_to(nl, DOWN, buff=0.6)
        self.at("range")
        self.play(FadeIn(dots), FadeIn(dl), run_time=0.6)
        bad = row(f"{OUT}/words_prompts_clip4.png", h=0.85, r=0, buff=0.06)
        good = row(f"{OUT}/words_prompts.png", h=0.85, r=0, buff=0.06)
        Group(bad, good).arrange(DOWN, buff=0.3).move_to([0.9, -1.8, 0])
        bl = Text(f"clipped at 4: {pct(CLIP4['eval']['w1']['trained']['follows'], 0)}", font_size=20, color=RED_B).next_to(bad, LEFT, buff=0.3)
        gl = Text(f"no clip: {pct(WORDS['eval']['w1']['trained']['follows'], 0)}", font_size=20, color=GREEN).next_to(good, LEFT, buff=0.3)
        self.at("44")
        self.play(FadeIn(bad), FadeIn(bl), run_time=0.5)
        self.at("99")
        self.play(FadeIn(good), FadeIn(gl), run_time=0.5)
        self.end_section()

    # ------------------------------------------------------------------ 10. real models
    def real(self):
        self.section(10)
        self.clear_stage()
        head = VGroup(Text("", font_size=22), Text("ours", font_size=24, color=LAT_C), Text("Stable Diffusion (v1)", font_size=24, color=YELLOW))
        rows_ = [("picture", "32 × 32 × 3", "512 × 512 × 3"), ("latent", "8 × 8 × 4", "64 × 64 × 4"),
                 ("fewer numbers", "12×", "48×"), ("U-Net", f"{WORDS['parameters'] / 1e6:.1f}M parameters", "~860M parameters"),
                 ("text encoder", "tiny CLIP (episode 12)", "CLIP ViT-L/14"), ("text in via", "cross-attention", "cross-attention")]
        tbl = VGroup(head)
        for name, a, b in rows_:
            tbl.add(VGroup(Text(name, font_size=20, color=GREY_B), Text(a, font=MONO, font_size=20), Text(b, font=MONO, font_size=20)))
        for j, r_ in enumerate(tbl):
            for k, c in enumerate(r_):
                c.move_to([-4.2 + 4.0 * k, 2.4 - 0.7 * j, 0])
        self.at("stable")
        self.play(FadeIn(tbl[0]), run_time=0.4)
        for j, cue in ((1, "512"), (2, "64"), (3, "48"), (4, "860"), (5, "clip"), (6, "attended")):
            self.at(cue)
            self.play(FadeIn(tbl[j]), run_time=0.35)
        self.end_section()

    # ------------------------------------------------------------------ 11. code
    def code(self):
        self.section(11)
        self.clear_stage()
        code, hl = code_panel(CODE, font_size=18)
        code.move_to([0, 0.2, 0])
        self.play(FadeIn(code, shift=0.2 * UP), run_time=0.4)
        self.play(Create(hl), run_time=0.2)
        self.at("once")
        self.play(highlight(hl, code, 0), run_time=0.3)
        self.at("train")
        self.play(highlight(hl, code, 2), run_time=0.3)
        self.at("decode")
        self.play(highlight(hl, code, 6), run_time=0.3)
        self.end_section()

    # ------------------------------------------------------------------ 12. outro
    def outro(self):
        self.section(12)
        self.clear_stage(run_time=0.4)
        card_ = next_up_card(NEXT)
        self.at("next")
        self.play(FadeIn(card_, shift=0.2 * UP), run_time=0.6)
        self.end_section()
        finish(self, hold=1.5)
