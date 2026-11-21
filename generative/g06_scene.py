"""How Generative Models Work: Images and Sound, episode 6 — Diffusion II: Learning to Denoise.

Render from the repo root:  ./render.sh generative g06
Numbers and pictures come from code/g06_learning_to_denoise (train_diffusion.py → results_toy.json,
results_shapes.json and out/*.png).
"""
import json

from manim import *

from common import MONO, code_panel, finish, highlight, next_up_card
from gen_common import dots, frame, plane, tile
from intro import play_token_intro
from g06_script import NEXT, TAGLINE, TITLE
from voiced_scene import VoicedScene

SERIES_LABEL = "HOW GENERATIVE MODELS WORK  ·  EPISODE 6"
CODE_DIR = "code/g06_learning_to_denoise"
OUT = f"{CODE_DIR}/out"
TOY = json.load(open(f"{CODE_DIR}/results_toy.json"))
SH = json.load(open(f"{CODE_DIR}/results_shapes.json"))
HIST = {h["step"]: h for h in SH["history"]}
DEMO = f"{OUT}/denoise_step20000.png"          # rows: x_t, ε, ε̂, x_t minus ε̂; columns t = 50, 200, 400, 600, 900
DATA_C, NOISE_C, GUESS_C, NET_C = BLUE_C, GREY_B, ORANGE, TEAL_C
CODE = """t = torch.randint(1, 1001, (len(x0),))          # random steps
eps = torch.randn_like(x0)                      # random noise
x_t = ab[t-1].sqrt() * x0 + (1 - ab[t-1]).sqrt() * eps   # the jump
loss = F.mse_loss(model(x_t, t), eps)           # predict the noise"""


def net_box(label="network", w=2.0, h=1.0):
    r = RoundedRectangle(width=w, height=h, corner_radius=0.15, color=NET_C, fill_color=NET_C, fill_opacity=0.15)
    return VGroup(r, Text(label, font_size=20, color=NET_C).move_to(r))


class LearningToDenoiseVideo(VoicedScene):
    VIDEO = "g06"

    def construct(self):
        play_token_intro(self, TITLE, 6, TAGLINE, label=SERIES_LABEL)
        self.hook()        # 1
        self.recipe()      # 2
        self.toy()         # 3
        self.walk()        # 4
        self.shapes()      # 5
        self.sees()        # 6
        self.progress()    # 7
        self.compare()     # 8
        self.code()        # 9
        self.outro()       # 10

    def clear_stage(self, *keep, run_time=0.6):
        self.play(*[FadeOut(m) for m in self.mobjects if m not in keep], run_time=run_time)

    # ------------------------------------------------------------------ 1. hook
    def hook(self):
        self.section(1)
        clean = tile(f"{OUT}/demo_clean.png", 0, 1.6)
        steps = Group(*[tile(DEMO, i, 1.6) for i in (1, 2, 4)])
        row = Group(clean, *steps).arrange(RIGHT, buff=0.5).move_to([0, 1.7, 0])
        arrows = VGroup(*[Arrow(row[i].get_right(), row[i + 1].get_left(), buff=0.08, color=GREY_B, stroke_width=3)
                          for i in range(3)])
        self.at("destroyed")
        self.play(FadeIn(row), *[FadeIn(frame(m)) for m in row], LaggedStart(*[GrowArrow(a) for a in arrows], lag_ratio=0.3), run_time=1.0)
        noisy = tile(DEMO, 2, 1.5).move_to([-4.6, -1.5, 0])
        box = net_box().move_to([-1.6, -1.5, 0])
        a1 = Arrow(noisy.get_right(), box.get_left(), buff=0.1, color=GREY_B)
        self.at("undo")
        self.play(FadeIn(noisy), FadeIn(frame(noisy)), GrowArrow(a1), FadeIn(box), run_time=0.6)
        no = Text("not the clean picture", font_size=22, color=GREY_C).move_to([2.6, -0.9, 0])
        cross = Line(no.get_left(), no.get_right(), color=RED, stroke_width=3)
        yes = Text("but the noise in it", font_size=26, color=GUESS_C).move_to([2.6, -1.8, 0])
        a2 = Arrow(box.get_right(), yes.get_left(), buff=0.15, color=GUESS_C)
        self.at("not")
        self.play(FadeIn(no), run_time=0.3)
        self.play(Create(cross), run_time=0.3)
        self.at("noise")
        self.play(GrowArrow(a2), FadeIn(yes), run_time=0.5)
        self.end_section()

    # ------------------------------------------------------------------ 2. one training step
    def recipe(self):
        self.section(2)
        self.clear_stage()
        clean = tile(f"{OUT}/demo_clean.png", 0, 1.5).move_to([-5.3, 1.4, 0])
        lc = Text("clean picture x₀", font_size=18, color=GREY_B).next_to(clean, DOWN, buff=0.15)
        self.at("clean")
        self.play(FadeIn(clean), FadeIn(frame(clean)), FadeIn(lc), run_time=0.4)
        t = Text("random step t\n(1 to 1,000)", font_size=20, color=YELLOW, line_spacing=0.9).move_to([-2.8, 2.3, 0])
        self.at("random")
        self.play(FadeIn(t), run_time=0.4)
        noisy = tile(DEMO, 2, 1.5).move_to([-0.3, 1.4, 0])
        a = Arrow(clean.get_right(), noisy.get_left(), buff=0.15, color=GREY_B)
        lj = Text("the jump", font_size=18, color=GREY_B).next_to(a, DOWN, buff=0.1)
        eps = tile(DEMO, 2, 1.5, row=1).move_to([-2.8, -1.6, 0])
        le = Text("the noise ε (remembered)", font_size=18, color=NOISE_C).next_to(eps, DOWN, buff=0.15)
        self.at("jump")
        self.play(GrowArrow(a), FadeIn(lj), FadeIn(noisy), FadeIn(frame(noisy)), run_time=0.5)
        self.at("remembering")
        self.play(FadeIn(eps), FadeIn(frame(eps)), FadeIn(le), run_time=0.4)
        box = net_box("network(xₜ, t)", w=2.6).move_to([2.6, 1.4, 0])
        a2 = Arrow(noisy.get_right(), box.get_left(), buff=0.15, color=GREY_B)
        self.at("show")
        self.play(GrowArrow(a2), FadeIn(box), run_time=0.5)
        guess = tile(DEMO, 2, 1.5, row=2).move_to([5.4, 1.4, 0])
        lg = Text("its guess ε̂", font_size=18, color=GUESS_C).next_to(guess, DOWN, buff=0.15)
        a3 = Arrow(box.get_right(), guess.get_left(), buff=0.1, color=GUESS_C)
        self.at("predict")
        self.play(GrowArrow(a3), FadeIn(guess), FadeIn(frame(guess, GUESS_C)), FadeIn(lg), run_time=0.5)
        loss = Text("loss = (guess − ε)², averaged over pixels", font_size=26, color=YELLOW).move_to([2.6, -1.6, 0])
        self.at("loss")
        self.play(FadeIn(loss), run_time=0.4)
        w = Text("that's the whole recipe", font_size=22, color=GREY_B).next_to(loss, DOWN, buff=0.3)
        self.at("whole")
        self.play(FadeIn(w), run_time=0.3)
        self.end_section()

    # ------------------------------------------------------------------ 3. toy: a spiral
    def toy(self):
        self.section(3)
        self.clear_stage()
        box, ax = plane([-3.6, -0.2, 0], size=4.8, lo=-2.3, hi=2.3)
        data = dots(ax, TOY["data"], DATA_C, r=0.02)
        ld = Text("10,000 points on a spiral", font_size=18, color=DATA_C).next_to(box, UP, buff=0.15)
        self.at("spiral")
        self.play(FadeIn(box), FadeIn(data), FadeIn(ld), run_time=0.6)
        mlp = Text("tiny network: 3 layers of 256", font_size=18, color=NET_C).next_to(box, DOWN, buff=0.15)
        self.at("tiny")
        self.play(FadeIn(mlp), run_time=0.3)
        curve = TOY["loss_curve"]
        lax = Axes(x_range=[0, 6000, 2000], y_range=[0.3, 0.7, 0.1], x_length=5.0, y_length=3.2, tips=False,
                   axis_config={"color": GREY_C}).move_to([3.4, 0.1, 0])
        xl = VGroup(*[Text(f"{v:,}", font=MONO, font_size=14, color=GREY_B).next_to(lax.c2p(v, 0.3), DOWN, buff=0.1) for v in (0, 2000, 4000, 6000)])
        yl = VGroup(*[Text(f"{v:.1f}", font=MONO, font_size=14, color=GREY_B).next_to(lax.c2p(0, v), LEFT, buff=0.1) for v in (0.3, 0.5, 0.7)])
        lt = Text("loss (training steps)", font_size=18, color=GREY_B).next_to(lax, UP, buff=0.2)
        pts = [lax.c2p(i * 100 + 50, min(v, 0.7)) for i, v in enumerate(curve)]
        line = VMobject(color=GUESS_C, stroke_width=3).set_points_as_corners(pts)
        self.at("6")
        self.play(FadeIn(lax), FadeIn(xl), FadeIn(yl), FadeIn(lt), Create(line), run_time=1.5)
        v0 = Text(f"{TOY['loss_first_100']:.2f}", font=MONO, font_size=18, color=GUESS_C).next_to(pts[0], RIGHT, buff=0.1).shift(UP * 0.15)
        self.at("55")
        self.play(FadeIn(v0), run_time=0.3)
        v1 = Text(f"{TOY['loss_last_500']:.2f}", font=MONO, font_size=18, color=GUESS_C).next_to(pts[-1], UP, buff=0.2)
        self.at("39")
        self.play(FadeIn(v1), run_time=0.3)
        note = Text("never zero: tiny noise looks like\nthe data's own jitter", font_size=18, color=GREY_B, line_spacing=0.9).next_to(lax, DOWN, buff=0.55)
        self.at("zero")
        self.play(FadeIn(note), run_time=0.4)
        self.box, self.ax, self.data = box, ax, data
        self.end_section()

    # ------------------------------------------------------------------ 4. walking back
    def walk(self):
        self.section(4)
        box, ax = self.box, self.ax
        self.play(*[FadeOut(m) for m in self.mobjects if m not in (box,)], run_time=0.5)
        snaps = TOY["snapshots"]
        order = ["999", "750", "500", "250", "100", "0"]
        cloud = dots(ax, snaps["999"], GREY_B, r=0.02)
        lab = Text("t = 1,000: pure noise", font=MONO, font_size=18, color=GREY_B).next_to(box, UP, buff=0.15)
        self.at("generate")
        self.play(FadeIn(cloud), FadeIn(lab), run_time=0.6)
        steps = VGroup(Text("at each step, from t = 1,000 down to 1:", font_size=20, color=GREY_B),
                       Text("1. ask the network for the noise ε̂", font_size=20, color=GUESS_C),
                       Text("2. remove a little of it", font_size=20, color=WHITE),
                       Text("3. add back a smaller amount of fresh noise", font_size=20, color=NOISE_C)
                       ).arrange(DOWN, aligned_edge=LEFT, buff=0.22).move_to([3.3, 1.2, 0])
        self.at("each")
        self.play(FadeIn(steps[0]), run_time=0.3)
        self.at("ask")
        self.play(FadeIn(steps[1]), run_time=0.3)
        self.at("remove")
        self.play(FadeIn(steps[2]), run_time=0.3)
        self.at("add")
        self.play(FadeIn(steps[3]), run_time=0.3)
        self.at("thousand")
        for key in order[1:]:
            new = dots(ax, snaps[key], DATA_C if key == "0" else interpolate_color(ManimColor(GREY_B), ManimColor(DATA_C), 1 - int(key) / 1000), r=0.02)
            nl = Text(f"t = {int(key):,}" + (": a spiral" if key == "0" else ""), font=MONO, font_size=18, color=DATA_C if key == "0" else GREY_B).move_to(lab)
            self.play(Transform(cloud, new), Transform(lab, nl), run_time=0.7)
        r = VGroup(Text("median distance to the nearest real point", font_size=18, color=GREY_B),
                   Text(f"generated points   {TOY['sample_nearest_median']:.3f}", font=MONO, font_size=20, color=DATA_C),
                   Text(f"pure noise         {TOY['noise_nearest_median']:.2f}", font=MONO, font_size=20, color=NOISE_C)
                   ).arrange(DOWN, aligned_edge=LEFT, buff=0.18).move_to([3.3, -1.6, 0])
        self.at("median")
        self.play(FadeIn(r[0]), FadeIn(r[1]), run_time=0.4)
        self.at("19")
        self.play(FadeIn(r[2]), run_time=0.3)
        self.end_section()

    # ------------------------------------------------------------------ 5. the Shapes network
    def shapes(self):
        self.section(5)
        self.clear_stage()
        # a sketch of a U-Net: resolution goes down, then back up, with skip connections across
        widths = [(32, 32), (64, 16), (128, 8)]
        down, up = VGroup(), VGroup()
        for k, (c, res) in enumerate(widths):
            h = 2.4 * res / 32 + 0.4
            w = 0.25 + 0.25 * k
            down.add(Rectangle(width=w, height=h, color=NET_C, fill_color=NET_C, fill_opacity=0.25).move_to([-3.6 + 1.2 * k, 0.6 - 0.6 * k, 0]))
            up.add(Rectangle(width=w, height=h, color=NET_C, fill_color=NET_C, fill_opacity=0.25).move_to([3.6 - 1.2 * k, 0.6 - 0.6 * k, 0]))
        mid = Rectangle(width=1.0, height=0.6, color=NET_C, fill_color=NET_C, fill_opacity=0.4).move_to([0, -1.2, 0])
        skips = VGroup(*[DashedLine(down[k].get_right(), up[k].get_left(), color=GREY_C) for k in range(3)])
        labels = VGroup(*[Text(f"{res}×{res}", font=MONO, font_size=14, color=GREY_B).next_to(down[k], DOWN, buff=0.1) for k, (c, res) in enumerate(widths)])
        inp = tile(DEMO, 2, 1.1).next_to(down[0], LEFT, buff=0.4)
        outp = tile(DEMO, 2, 1.1, row=2).next_to(up[0], RIGHT, buff=0.4)
        unet = Group(down, up, mid, skips, labels, inp, outp)
        title = Text("a U-Net", font_size=30, color=NET_C).to_edge(UP, buff=0.5)
        self.at("pictures")
        self.play(FadeIn(inp), FadeIn(frame(inp)), run_time=0.4)
        self.at("unit")
        self.play(FadeIn(title), LaggedStart(*[FadeIn(m) for m in [*down, mid, *reversed(list(up))]], lag_ratio=0.15),
                  FadeIn(labels), Create(skips), FadeIn(outp), FadeIn(frame(outp, GUESS_C)), run_time=1.4)
        p = Text(f"{SH['parameters']:,} parameters", font=MONO, font_size=22).next_to(title, DOWN, buff=0.2)
        self.play(FadeIn(p), run_time=0.3)
        e8 = Text("episode 8 opens it up", font_size=18, color=GREY_B).move_to([0, -2.0, 0])
        self.at("8")
        self.play(FadeIn(e8), run_time=0.3)
        tr = Text(f"{SH['steps']:,} steps · batch {SH['batch']} · 20,000 pictures · 2 h 18 min on a CPU", font_size=20, color=YELLOW).move_to([0, -2.9, 0])
        self.at("20")
        self.play(FadeIn(tr), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 6. what it learned
    def sees(self):
        self.section(6)
        self.clear_stage()
        h = 1.05
        grid = Group(*[Group(*[tile(DEMO, i, h, row=r) for i in range(5)]).arrange(RIGHT, buff=0.3) for r in range(4)]).arrange(DOWN, buff=0.15).move_to([1.6, -0.35, 0])
        names = ["noisy picture xₜ", "noise added ε", "network's guess ε̂", "picture minus guess"]
        cols = [WHITE, NOISE_C, GUESS_C, DATA_C]
        rl = VGroup(*[Text(n, font_size=18, color=c).next_to(grid[r], LEFT, buff=0.3) for r, (n, c) in enumerate(zip(names, cols))])
        tl = VGroup(*[Text(f"t = {t}", font=MONO, font_size=16, color=GREY_B).next_to(grid[0][i], UP, buff=0.12) for i, t in enumerate(SH["demo_steps"])])
        self.at("triangle")
        self.play(FadeIn(grid[0]), FadeIn(rl[0]), FadeIn(tl), run_time=0.6)
        self.at("below")
        self.play(FadeIn(grid[1]), FadeIn(rl[1]), run_time=0.4)
        self.at("guess")
        self.play(FadeIn(grid[2]), FadeIn(rl[2]), run_time=0.4)
        self.at("subtract")
        self.play(FadeIn(grid[3]), FadeIn(rl[3]), run_time=0.5)
        good = SurroundingRectangle(Group(*grid[3][:4]), color=GREEN, buff=0.06)
        self.at("600")
        self.play(Create(good), run_time=0.4)
        bad = SurroundingRectangle(grid[3][4], color=RED, buff=0.06)
        self.at("900")
        self.play(Create(bad), run_time=0.4)
        avg = Text("an average of\neverything it\nmight have been", font_size=16, color=RED_B, line_spacing=0.9).next_to(bad, RIGHT, buff=0.15)
        self.at("average")
        self.play(FadeIn(avg), run_time=0.3)
        self.end_section()

    # ------------------------------------------------------------------ 7. samples during training
    def progress(self):
        self.section(7)
        self.clear_stage()
        rows = Group()
        for step in (2000, 5000, 10_000, 20_000):
            pics = Group(*[tile(f"{OUT}/samples_step{step:05d}.png", i, 1.0, row=i // 8) for i in range(6)]).arrange(RIGHT, buff=0.12)
            rows.add(pics)
        rows.arrange(DOWN, buff=0.28).move_to([0.2, -0.2, 0])
        left = VGroup(*[Text(f"{s:,} steps", font=MONO, font_size=18, color=GREY_B).next_to(rows[k], LEFT, buff=0.35)
                        for k, s in enumerate((2000, 5000, 10_000, 20_000))])
        right = VGroup(*[Text(f"{HIST[s]['recognizable']:.0%}", font=MONO, font_size=22, color=YELLOW if s == 20_000 else GREY_B).next_to(rows[k], RIGHT, buff=0.35)
                         for k, s in enumerate((2000, 5000, 10_000, 20_000))])
        head = Text("judge", font_size=16, color=GREY_B).next_to(right[0], UP, buff=0.3)
        cues = [("2", None), ("5", "none"), ("10", "15"), ("20", "70")]
        for k, (c1, c2) in enumerate(cues):
            self.at(c1)
            self.play(FadeIn(rows[k]), FadeIn(left[k]), run_time=0.4)
            if c2:
                self.at(c2)
                anims = [FadeIn(right[k])]
                if k == 1:
                    anims += [FadeIn(right[0]), FadeIn(head)]
                self.play(*anims, run_time=0.3)
        self.end_section()

    # ------------------------------------------------------------------ 8. against the earlier models
    def compare(self):
        self.section(8)
        self.clear_stage()
        h20 = HIST[20_000]
        vals = [("VAE (ep. 3)", 9.4, GREY_B), ("GAN (ep. 4)", 38.4, PURPLE_B), ("diffusion", h20["recognizable"] * 100, DATA_C)]
        title = Text("recognized by the judge (500 samples)", font_size=22, color=GREY_B).move_to([-3.3, 2.9, 0])
        bars = VGroup()
        for k, (n, v, c) in enumerate(vals):
            y = 1.9 - 0.75 * k
            name = Text(n, font_size=20, color=c).move_to([-5.6, y, 0])
            bar = Rectangle(width=max(0.05, 4.0 * v / 100), height=0.45, stroke_width=0, fill_color=c, fill_opacity=0.85)
            bar.move_to([-4.4, y, 0], aligned_edge=LEFT)
            num = Text(f"{v:.1f}%", font=MONO, font_size=18, color=c).next_to(bar, RIGHT, buff=0.12)
            bars.add(VGroup(name, bar, num))
        self.at("beats")
        self.play(FadeIn(title), run_time=0.3)
        self.at("70")
        self.play(FadeIn(bars[2]), run_time=0.4)
        self.at("38")
        self.play(FadeIn(bars[1]), run_time=0.3)
        self.at("9")
        self.play(FadeIn(bars[0]), run_time=0.3)
        tab = VGroup(Text("                     GAN     diffusion", font=MONO, font_size=18, color=GREY_B),
                     Text(f"combinations (135)    97     {h20['combinations']}", font=MONO, font_size=18),
                     Text(f"nearest real picture  0.0038  {h20['nearest_train_mse']:.4f}", font=MONO, font_size=18),
                     ).arrange(DOWN, aligned_edge=LEFT, buff=0.2).move_to([3.4, 1.4, 0])
        self.at("127")
        self.play(FadeIn(tab[0]), FadeIn(tab[1]), run_time=0.4)
        self.at("closest")
        self.play(FadeIn(tab[2]), run_time=0.4)
        samples = Group(*[tile(f"{OUT}/samples_step20000.png", i % 8, 0.95, row=1) for i in range(8, 16)]).arrange(RIGHT, buff=0.1).move_to([0, -0.9, 0])
        self.play(FadeIn(samples), run_time=0.4)
        cost = VGroup(Text("the price: network calls per sample", font_size=20, color=GREY_B),
                      Text("GAN 1    diffusion 50 (here)", font=MONO, font_size=22, color=YELLOW)).arrange(DOWN, buff=0.15).move_to([0, -2.6, 0])
        self.at("price")
        self.play(FadeIn(cost[0]), run_time=0.3)
        self.at("50")
        self.play(FadeIn(cost[1]), run_time=0.3)
        self.end_section()

    # ------------------------------------------------------------------ 9. code
    def code(self):
        self.section(9)
        self.clear_stage()
        code, hl = code_panel(CODE, font_size=17)
        code.move_to([0, 0.3, 0])
        self.at("code")
        self.play(FadeIn(code, shift=0.2 * UP), run_time=0.5)
        self.at("random")
        self.play(Create(hl), run_time=0.3)
        self.at("noise")
        self.play(highlight(hl, code, 1), run_time=0.3)
        self.at("jump")
        self.play(highlight(hl, code, 2), run_time=0.3)
        self.at("squared")
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
