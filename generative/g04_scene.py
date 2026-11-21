"""How Generative Models Work: Images and Sound, episode 4 — GANs: a Forger and a Judge.

Render from the repo root:  ./render.sh generative g04
Numbers and pictures come from code/g04_gans (train_gan.py → results_stable.json, results_collapse.json, out/*.png),
with episode 3's VAE numbers (code/g03_variational_autoencoders/results.json).
"""
import json

import numpy as np
from manim import *

from common import MONO, code_panel, finish, highlight, next_up_card
from gen_common import frame, tile
from intro import play_token_intro
from g04_script import NEXT, TAGLINE, TITLE
from voiced_scene import VoicedScene

SERIES_LABEL = "HOW GENERATIVE MODELS WORK  ·  EPISODE 4"
CODE_DIR = "code/g04_gans"
OUT = f"{CODE_DIR}/out"
S = json.load(open(f"{CODE_DIR}/results_stable.json"))
C = json.load(open(f"{CODE_DIR}/results_collapse.json"))
V = json.load(open("code/g03_variational_autoencoders/results.json"))
G_C, D_C, REAL_C, FAKE_C = ORANGE, TEAL_C, GREEN_C, RED_C
CODE = """fake = G(torch.randn(64, 64))
d_loss = bce(D(real), ones) + bce(D(fake.detach()), zeros)
opt_d.zero_grad(); d_loss.backward(); opt_d.step()
g_loss = bce(D(fake), ones)          # fool the discriminator
opt_g.zero_grad(); g_loss.backward(); opt_g.step()"""


def box(label, color, w=2.4, h=1.2, fs=26):
    r = RoundedRectangle(width=w, height=h, corner_radius=0.18, color=color, fill_opacity=0.15)
    return VGroup(r, Text(label, font_size=fs, color=color).move_to(r))


def row(path, n, height, buff=0.1, row_i=0):
    return Group(*[tile(path, i, height, row=row_i) for i in range(n)]).arrange(RIGHT, buff=buff)


class GANVideo(VoicedScene):
    VIDEO = "g04"

    def construct(self):
        play_token_intro(self, TITLE, 4, TAGLINE, label=SERIES_LABEL)
        self.hook()          # 1
        self.players()       # 2
        self.loop()          # 3
        self.first_tries()   # 4
        self.results()       # 5
        self.instability()   # 6
        self.collapse()      # 7
        self.today()         # 8
        self.code()          # 9
        self.outro()         # 10

    def clear_stage(self, *keep, run_time=0.6):
        self.play(*[FadeOut(m) for m in self.mobjects if m not in keep], run_time=run_time)

    # ------------------------------------------------------------------ 1. hook
    def hook(self):
        self.section(1)
        vae = tile("code/g03_variational_autoencoders/out/random_vae.png", 1, 2.0).move_to([-3.6, 0.8, 0])
        lv = Text("VAE: pixel error → blur", font_size=22, color=GREY_B).next_to(vae, DOWN, buff=0.2)
        self.at("vae")
        self.play(FadeIn(vae), FadeIn(frame(vae)), FadeIn(lv), run_time=0.5)
        g = Text("GAN: generative adversarial network", font_size=26, color=G_C).move_to([2.8, 1.6, 0])
        self.at("generative")
        self.play(FadeIn(g), run_time=0.4)
        x = Text("no pixel error", font_size=22, color=GREY_B).next_to(g, DOWN, buff=0.4)
        self.at("drops")
        self.play(FadeIn(x), run_time=0.3)
        d = box("a second network:\nreal or fake?", D_C, w=4.0, h=1.4, fs=22).next_to(x, DOWN, buff=0.4)
        self.at("second")
        self.play(FadeIn(d), run_time=0.5)
        self.end_section()

    # ------------------------------------------------------------------ 2. two players
    def players(self):
        self.section(2)
        self.clear_stage()
        z = VGroup(*[Rectangle(width=0.07, height=0.15 + 0.5 * abs(v), stroke_width=0, fill_color=GREY_B, fill_opacity=0.9)
                     for v in np.random.default_rng(2).normal(0, 1, 24)]).arrange(RIGHT, buff=0.03, aligned_edge=DOWN).move_to([-5.6, 1.2, 0])
        zl = Text("64 random\nnumbers", font_size=18, color=GREY_B, line_spacing=0.9).next_to(z, DOWN, buff=0.15)
        gen = box("generator", G_C).move_to([-2.9, 1.2, 0])
        fake = tile(f"{OUT}/stable_epoch30.png", 4, 1.3).move_to([-0.3, 1.2, 0])
        self.at("generator")
        self.play(FadeIn(z), FadeIn(zl), FadeIn(gen), run_time=0.5)
        a1 = Arrow(z.get_right(), gen.get_left(), buff=0.15, color=GREY_B)
        a2 = Arrow(gen.get_right(), fake.get_left(), buff=0.15, color=G_C)
        self.at("picture")
        self.play(GrowArrow(a1), GrowArrow(a2), FadeIn(fake), FadeIn(frame(fake, FAKE_C)), run_time=0.6)
        real = tile("code/g02_autoencoders/out/originals.png", 2, 1.3).move_to([-0.3, -1.4, 0])
        rl = Text("real", font_size=18, color=REAL_C).next_to(real, LEFT, buff=0.2)
        fl = Text("fake", font_size=18, color=FAKE_C).next_to(fake, UP, buff=0.12)
        dis = box("discriminator", D_C, w=2.9).move_to([3.0, -0.1, 0])
        a3 = Arrow(fake.get_right(), dis.get_left() + UP * 0.25, buff=0.15, color=GREY_B)
        a4 = Arrow(real.get_right(), dis.get_left() + DOWN * 0.25, buff=0.15, color=GREY_B)
        out = Text("one number:\nreal or fake?", font_size=20, color=D_C, line_spacing=0.9).next_to(dis, RIGHT, buff=0.3)
        self.at("discriminator")
        self.play(FadeIn(real), FadeIn(frame(real, REAL_C)), FadeIn(rl), FadeIn(fl), FadeIn(dis), GrowArrow(a3), GrowArrow(a4), run_time=0.7)
        self.at("outputs")
        self.play(FadeIn(out), run_time=0.4)
        goals = VGroup(Text("discriminator: tell them apart", font_size=20, color=D_C), Text("generator: fool it", font_size=20, color=G_C)).arrange(DOWN, aligned_edge=LEFT, buff=0.15).move_to([3.4, -2.4, 0])
        self.at("apart")
        self.play(FadeIn(goals[0]), run_time=0.3)
        self.at("fool")
        self.play(FadeIn(goals[1]), run_time=0.3)
        fd = Text("a forger and a detective", font_size=24, color=YELLOW).to_edge(UP, buff=0.4)
        self.at("forger")
        self.play(FadeIn(fd), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 3. the loop
    def loop(self):
        self.section(3)
        self.clear_stage()
        s1 = VGroup(Text("step 1 · discriminator", font_size=24, color=D_C),
                    Text("real batch → score up,  fake batch → score down", font_size=20)).arrange(DOWN, aligned_edge=LEFT, buff=0.15)
        s2 = VGroup(Text("step 2 · generator", font_size=24, color=G_C),
                    Text("change the generator so its fakes score as more real", font_size=20)).arrange(DOWN, aligned_edge=LEFT, buff=0.15)
        VGroup(s1, s2).arrange(DOWN, aligned_edge=LEFT, buff=0.6).move_to([-0.8, 0.8, 0])
        self.at("show")
        self.play(FadeIn(s1), run_time=0.5)
        self.at("generator")
        self.play(FadeIn(s2), run_time=0.5)
        cyc = CurvedArrow(s2.get_right() + RIGHT * 0.3, s1.get_right() + RIGHT * 0.3, angle=PI / 2, color=GREY_B).shift(RIGHT * 0.2)
        self.play(Create(cyc), run_time=0.5)
        n = Text("no fixed target: each one's goal moves when the other improves", font_size=22, color=YELLOW).move_to([0, -2.0, 0])
        self.at("neither")
        self.play(FadeIn(n), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 4. first tries
    def first_tries(self):
        self.section(4)
        self.clear_stage()
        early = row(f"{OUT}/stable_raw_epoch03.png", 6, 1.6).move_to([0, 1.8, 0])
        el = Text("early in training: nearly every shape blue", font_size=20, color=RED_B).next_to(early, DOWN, buff=0.2)
        self.at("purple")
        self.play(FadeIn(early), FadeIn(el), run_time=0.5)
        fixes = VGroup(Text("fix 1 · R1 penalty: keep the discriminator smooth around real pictures", font_size=21, color=D_C),
                       Text("fix 2 · sample from a running average (EMA) of the generator's weights", font_size=21, color=G_C)).arrange(DOWN, aligned_edge=LEFT, buff=0.3).move_to([0, -0.5, 0])
        self.at("penalty")
        self.play(FadeIn(fixes[0]), run_time=0.4)
        self.at("sampling")
        self.play(FadeIn(fixes[1]), run_time=0.4)
        p = S["parameters"]
        t = Text(f"generator {p['generator']:,} and discriminator {p['discriminator']:,} parameters · 30 passes · "
                 f"{S['history'][-1]['seconds'] / 60:.0f} min on a CPU", font_size=18, color=GREY_B).move_to([0, -2.3, 0])
        self.at("30")
        self.play(FadeIn(t), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 5. results
    def results(self):
        self.section(5)
        self.clear_stage()
        g = Group(row(f"{OUT}/stable_epoch30.png", 8, 1.15), row(f"{OUT}/stable_epoch30.png", 8, 1.15, row_i=1)).arrange(DOWN, buff=0.1).move_to([0, 1.7, 0])
        self.at("sharp")
        self.play(FadeIn(g), run_time=0.6)
        h = S["history"][-1]
        rows = [("real pictures", S["real_recognizable"], REAL_C), ("GAN samples", h["recognizable"], G_C),
                ("VAE samples (ep. 3)", V["models"]["vae"]["random_recognizable"], YELLOW)]
        bars = VGroup()
        for k, (name, v, col) in enumerate(rows):
            lab = Text(name, font_size=19).move_to([-2.4, -0.6 - 0.5 * k, 0], aligned_edge=RIGHT)
            bar = Rectangle(width=max(3.6 * v, 0.03), height=0.32, stroke_width=0, fill_color=col, fill_opacity=0.85).move_to([-2.1, lab.get_center()[1], 0], aligned_edge=LEFT)
            bars.add(VGroup(lab, bar, Text(f"{v:.1%}", font=MONO, font_size=16).next_to(bar, RIGHT, buff=0.12)))
        cap = Text("recognized by the judge", font_size=18, color=TEAL_C).next_to(bars, UP, buff=0.15).align_to(bars[0][1], LEFT)
        self.at("38")
        self.play(FadeIn(cap), FadeIn(bars[0]), FadeIn(bars[1]), run_time=0.5)
        self.at("9")
        self.play(FadeIn(bars[2]), run_time=0.3)
        side = VGroup(Text(f"{h['combinations']} / 135", font=MONO, font_size=30, color=G_C),
                      Text("combinations covered", font_size=18, color=GREY_B),
                      Text(f"nearest training picture: {h['nearest_train_mse']:.4f}", font=MONO, font_size=14, color=GREY_A),
                      Text(f"(VAE {V['models']['vae']['random_nearest_train_mse']:.4f})", font=MONO, font_size=16, color=GREY_B)).arrange(DOWN, buff=0.12).move_to([4.6, -1.1, 0])
        self.at("97")
        self.play(FadeIn(side[:2]), run_time=0.4)
        self.at("closest")
        self.play(FadeIn(side[2:]), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 6. instability
    def instability(self):
        self.section(6)
        self.clear_stage()
        ax = Axes(x_range=[0, 30, 5], y_range=[0, 0.5, 0.1], x_length=9, y_length=4.2, tips=False, axis_config={"color": GREY_C}).move_to([-0.4, -0.2, 0])
        xl = VGroup(*[Text(str(e), font=MONO, font_size=14, color=GREY_B).next_to(ax.c2p(e, 0), DOWN, buff=0.12) for e in (0, 10, 20, 30)])
        yl = VGroup(*[Text(f"{int(v * 100)}%", font=MONO, font_size=14, color=GREY_B).next_to(ax.c2p(0, v), LEFT, buff=0.12) for v in (0, 0.25, 0.5)])
        ttl = Text("judge-recognized samples, pass by pass", font_size=24).to_edge(UP, buff=0.4)
        hist = S["history"]
        raw = VMobject(color=G_C, stroke_width=3).set_points_as_corners([ax.c2p(h["epoch"], h["raw_recognizable"]) for h in hist])
        avg = VMobject(color=YELLOW, stroke_width=4).set_points_as_corners([ax.c2p(h["epoch"], h["recognizable"]) for h in hist])
        lr = Text("the generator itself", font_size=18, color=G_C).move_to(ax.c2p(5, 0.47), aligned_edge=LEFT)
        la = Text("averaged weights", font_size=18, color=YELLOW).move_to(ax.c2p(5, 0.42), aligned_edge=LEFT)
        self.play(FadeIn(ttl), FadeIn(ax), FadeIn(xl), FadeIn(yl), run_time=0.4)
        self.at("generator")
        self.play(Create(raw), FadeIn(lr), run_time=1.6)
        self.at("27")
        self.play(Indicate(raw, color=RED_B), run_time=1.0)
        self.at("averaged")
        self.play(Create(avg), FadeIn(la), run_time=1.2)
        self.end_section()

    # ------------------------------------------------------------------ 7. mode collapse
    def collapse(self):
        self.section(7)
        self.clear_stage()
        h = Text("generator 10× faster, 5 steps per discriminator step", font_size=24, color=RED_B).to_edge(UP, buff=0.5)
        self.at("break")
        self.play(FadeIn(h), run_time=0.4)
        r1 = row(f"{OUT}/collapse_raw_epoch03.png", 8, 1.15).move_to([0.6, 1.4, 0])
        r2 = row(f"{OUT}/collapse_raw_epoch10.png", 8, 1.15).move_to([0.6, -0.1, 0])
        l1 = Text("pass 3", font_size=18, color=GREY_B).next_to(r1, LEFT, buff=0.25)
        l2 = Text("pass 10", font_size=18, color=GREY_B).next_to(r2, LEFT, buff=0.25)
        self.at("pink")
        self.play(FadeIn(r1), FadeIn(l1), run_time=0.5)
        self.at("green")
        self.play(FadeIn(r2), FadeIn(l2), run_time=0.5)
        mx = max(x["raw_combinations"] for x in C["history"])
        n = Text(f"at most {mx} of 135 combinations in any pass", font_size=22, color=YELLOW).move_to([0, -1.6, 0])
        self.at("never")
        self.play(FadeIn(n), run_time=0.4)
        mc = Text("mode collapse: the forger found one trick", font_size=26, color=RED_B).move_to([0, -2.5, 0])
        self.at("mode")
        self.play(FadeIn(mc), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 8. today
    def today(self):
        self.section(8)
        self.clear_stage()
        a = Text("pictures: diffusion has largely replaced GANs", font_size=26).move_to([0, 1.2, 0])
        self.at("diffusion")
        self.play(FadeIn(a), run_time=0.4)
        b = Text("but learned judges live on", font_size=26, color=D_C).next_to(a, DOWN, buff=0.5)
        self.at("idea")
        self.play(FadeIn(b), run_time=0.4)
        c = Text("our narrator's voice model: the part that makes the sound\nwas trained against discriminators (episode 17)", font_size=22, color=YELLOW, line_spacing=0.9).next_to(b, DOWN, buff=0.5)
        self.at("narrators")
        self.play(FadeIn(c), run_time=0.5)
        self.end_section()

    # ------------------------------------------------------------------ 9. code
    def code(self):
        self.section(9)
        self.clear_stage()
        code, hl = code_panel(CODE, font_size=20)
        code.move_to([0, 0.3, 0])
        self.play(FadeIn(code, shift=0.2 * UP), run_time=0.5)
        hl.match_y(code.line_numbers[1])
        self.at("push")
        self.play(Create(hl), run_time=0.3)
        self.at("then")
        self.play(highlight(hl, code, 3), run_time=0.3)
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
