"""How LLMs Work: Deep Dive, episode 25 — Data Parallelism.

Render from the repo root:  ./render.sh deep-dive d25
Every number on screen comes from code/d25_data_parallel/data_parallel.py (4 worker processes with torch.distributed
"gloo" and DistributedDataParallel vs one process; an 818,241-parameter tiny GPT; 200 steps).
"""
import math

from manim import *

from common import MODEL_COLOR, MONO, TOKEN_COLOR, code_panel, finish, highlight, next_up_card, token
from intro import play_token_intro
from d25_script import NEXT, TAGLINE, TITLE
from voiced_scene import VoicedScene

SERIES_LABEL = "HOW LLMs WORK: DEEP DIVE  ·  EPISODE 25"
CODE = """dist.init_process_group("gloo", rank=rank, world_size=4)
model = DDP(TinyGPT())             # same weights everywhere
for step in range(steps):
    x, y = my_quarter_of(global_batch(step))
    loss = lm_loss(model, x, y)
    loss.backward()                # gradients averaged here
    opt.step()"""


def worker_box(k, colour=BLUE_C):
    r = RoundedRectangle(width=2.4, height=1.6, corner_radius=0.15, color=colour, fill_opacity=0.15)
    m = RoundedRectangle(width=1.6, height=0.5, corner_radius=0.08, color=MODEL_COLOR, fill_opacity=0.5).move_to(r).shift(0.3 * UP)
    ml = Text("full model", font_size=16).move_to(m)
    s = VGroup(*[Rectangle(width=0.35, height=0.12, stroke_width=0.6, color=GREEN_C, fill_opacity=0.5) for _ in range(4)])
    s.arrange(DOWN, buff=0.03).move_to(r).shift(0.45 * DOWN)
    t = Text(f"worker {k}", font_size=16, color=GREY_B).next_to(r, UP, buff=0.08)
    return VGroup(r, m, ml, s, t)


class DataParallelVideo(VoicedScene):
    VIDEO = "d25"

    def construct(self):
        play_token_intro(self, TITLE, 25, TAGLINE, label=SERIES_LABEL)
        self.hook()          # 1
        self.idea()          # 2
        self.real()          # 3
        self.result()        # 4
        self.exact()         # 5
        self.cost()          # 6
        self.ring()          # 7
        self.limit()         # 8
        self.code()          # 9
        self.outro()         # 10

    def clear_stage(self, *keep, run_time=0.6):
        self.play(*[FadeOut(m) for m in self.mobjects if m not in keep], run_time=run_time)

    # ------------------------------------------------------------------ 1. hook
    def hook(self):
        self.section(1)
        gpus = VGroup(*[Square(0.35, color=GREEN_C, fill_opacity=0.4, stroke_width=1) for _ in range(96)])
        gpus.arrange_in_grid(rows=6, cols=16, buff=0.12).move_to([0, 0.6, 0])
        self.at("thousands")
        self.play(FadeIn(gpus, lag_ratio=0.01), run_time=1.0)
        p = Text("part 6: training at scale", font_size=28, color=YELLOW).move_to([0, -1.6, 0])
        self.at("six")
        self.play(FadeIn(p), run_time=0.4)
        d = Text("the simplest way: data parallelism", font_size=26).move_to([0, -2.4, 0])
        self.at("data")
        self.play(FadeIn(d), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 2. idea
    def idea(self):
        self.section(2)
        self.clear_stage()
        workers = VGroup(*[worker_box(k) for k in range(4)]).arrange(RIGHT, buff=0.5).move_to([0, 1.0, 0])
        self.at("copy")
        self.play(FadeIn(workers, lag_ratio=0.15), run_time=0.8)
        sl = Text("each: its own slice of the batch (green)", font_size=20, color=GREEN_B).move_to([0, -0.4, 0])
        self.at("slice")
        self.play(FadeIn(sl), *[Indicate(w[3], color=GREEN_B) for w in workers], run_time=0.6)
        ar = RoundedRectangle(width=6.2, height=0.8, corner_radius=0.15, color=YELLOW, fill_opacity=0.15).move_to([0, -1.8, 0])
        al = Text("all-reduce: average the gradients", font_size=22, color=YELLOW).move_to(ar)
        arrows = VGroup(*[Arrow(w[0].get_bottom(), ar.get_top(), buff=0.1, stroke_width=2, color=GREY_B) for w in workers])
        self.at("average")
        self.play(FadeOut(sl), FadeIn(ar), FadeIn(al),
                  LaggedStart(*[GrowArrow(a) for a in arrows], lag_ratio=0.1), run_time=0.8)
        same = Text("every copy takes the same step and stays identical", font_size=22).move_to([0, -2.9, 0])
        self.at("identical")
        self.play(FadeIn(same), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 3. real run
    def real(self):
        self.section(3)
        self.clear_stage()
        head = Text("a real run on this machine (torch.distributed, gloo)", font_size=26).to_edge(UP, buff=0.6)
        self.at("real")
        self.play(FadeIn(head), run_time=0.4)
        workers = VGroup(*[worker_box(k) for k in range(4)]).arrange(RIGHT, buff=0.4).scale(0.8).move_to([-2.2, 0.3, 0])
        wl = Text("4 processes × 8 sequences", font_size=22, color=BLUE_B).next_to(workers, DOWN, buff=0.3)
        self.at("eight")
        self.play(FadeIn(workers), FadeIn(wl), run_time=0.6)
        single = worker_box("—", colour=GOLD).scale(0.8).move_to([4.4, 0.3, 0])
        single[4].become(Text("one process", font_size=16, color=GREY_B).next_to(single[0], UP, buff=0.08))
        sl = Text("1 process × 32 sequences", font_size=22, color=GOLD).next_to(single, DOWN, buff=0.3)
        vs = Text("vs", font_size=26).move_to([2.6, 0.3, 0])
        self.at("32")
        self.play(FadeIn(vs), FadeIn(single), FadeIn(sl), run_time=0.6)
        self.end_section()

    # ------------------------------------------------------------------ 4. result
    def result(self):
        self.section(4)
        self.clear_stage()
        rows = VGroup(Text("first gradient: largest difference 1.5 × 10⁻⁸", font=MONO, font_size=24),
                      Text("losses over 200 steps: largest difference 4.8 × 10⁻⁷", font=MONO, font_size=24),
                      Text("step 1:   4.3335  vs  4.3335", font=MONO, font_size=26, color=YELLOW),
                      Text("step 200: 2.3006  vs  2.3006", font=MONO, font_size=26, color=YELLOW))
        rows.arrange(DOWN, buff=0.4, aligned_edge=LEFT).move_to([0, 0.2, 0])
        for k, cue in enumerate(("gradient", "200", "3335", "3006")):
            self.at(cue)
            self.play(FadeIn(rows[k]), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 5. why exact
    def exact(self):
        self.section(5)
        self.clear_stage()
        eq = Text("mean(mean of 8, mean of 8, mean of 8, mean of 8) = mean of 32", font=MONO, font_size=24)
        eq.move_to([0, 0.6, 0])
        self.at("average")
        self.play(FadeIn(eq), run_time=0.5)
        n = Text("splitting the work changes nothing in the math", font_size=28, color=YELLOW).move_to([0, -0.8, 0])
        self.at("nothing")
        self.play(FadeIn(n), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 6. cost
    def cost(self):
        self.section(6)
        self.clear_stage()
        head = Text("gradients every worker exchanges, every step (float32)", font_size=26).to_edge(UP, buff=0.6)
        self.at("communication")
        self.play(FadeIn(head), run_time=0.4)
        rows = [("our tiny model (818,241 parameters)", "3.1 MiB", "three"), ("GPT-2 small", "0.5 GiB", "half"),
                ("7 billion parameters", "26.1 GiB", "26")]
        out = VGroup()
        for k, (name, v, cue) in enumerate(rows):
            y = 1.0 - 1.0 * k
            out.add(VGroup(Text(name, font_size=26).move_to([1.4, y, 0], aligned_edge=RIGHT),
                           Text(v, font=MONO, font_size=28, color=YELLOW).move_to([2.0, y, 0], aligned_edge=LEFT)))
            self.at(cue)
            self.play(FadeIn(out[k]), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 7. ring
    def ring(self):
        self.section(7)
        self.clear_stage()
        pts = [[2.0 * math.cos(math.pi / 2 - k * math.pi / 2) - 3.6, 2.0 * math.sin(math.pi / 2 - k * math.pi / 2) + 0.2, 0]
               for k in range(4)]
        nodes = VGroup(*[VGroup(Circle(radius=0.45, color=BLUE_C, fill_opacity=0.3), Text(str(k), font_size=24)).move_to(p)
                         for k, p in enumerate(pts)])
        arrows = VGroup(*[CurvedArrow(nodes[k].get_center(), nodes[(k + 1) % 4].get_center(), angle=-0.5, color=YELLOW,
                                      stroke_width=3).scale(0.7) for k in range(4)])
        self.at("ring")
        self.play(FadeIn(nodes), *[Create(a) for a in arrows], run_time=0.8)
        t = Text("each worker passes chunks to its neighbor;\nit sends about 2 × (N−1)/N × the gradient size\n"
                 "(4 workers, tiny model: 4.7 MiB)", font_size=20, line_spacing=0.9).move_to([2.9, 0.8, 0])
        self.at("twice")
        self.play(FadeIn(t), run_time=0.5)
        e = Text("start sending during the backward pass,\nto hide the wait", font_size=20, color=GREY_A,
                 line_spacing=0.9).move_to([2.9, -1.3, 0])
        self.at("early")
        self.play(FadeIn(e), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 8. limit
    def limit(self):
        self.section(8)
        self.clear_stage()
        head = Text("the catch: every worker holds the whole model", font_size=28).to_edge(UP, buff=0.6)
        self.at("catch")
        self.play(FadeIn(head), run_time=0.4)
        parts = [("weights", BLUE_C), ("gradients", ORANGE), ("Adam m", GOLD), ("Adam v", GOLD)]
        bar = VGroup(*[VGroup(Rectangle(width=2.2, height=0.7, stroke_width=1, color=WHITE, fill_color=c, fill_opacity=0.6),
                              Text(f"{n}\n4 bytes", font_size=18, line_spacing=0.8)) for n, c in parts])
        for b in bar:
            b[1].move_to(b[0])
        bar.arrange(RIGHT, buff=0).move_to([0, 1.0, 0])
        self.at("16")
        self.play(FadeIn(bar, lag_ratio=0.2), run_time=0.7)
        s = Text("16 bytes per parameter", font_size=24, color=YELLOW).next_to(bar, DOWN, buff=0.3)
        self.play(FadeIn(s), run_time=0.3)
        b = Text("7 billion parameters → 112 GB: more than one GPU holds", font_size=26, color=RED_B).move_to([0, -1.2, 0])
        self.at("112")
        self.play(FadeIn(b), run_time=0.4)
        n = Text("next: split the model itself", font_size=24, color=GREY_A).move_to([0, -2.2, 0])
        self.at("split")
        self.play(FadeIn(n), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 9. code
    def code(self):
        self.section(9)
        self.clear_stage()
        code, hl = code_panel(CODE, font_size=24)
        code.move_to([0, 0.3, 0])
        self.at("code")
        self.play(FadeIn(code, shift=0.2 * UP), run_time=0.5)
        hl.match_y(code.line_numbers[0])
        self.at("process")
        self.play(Create(hl), run_time=0.3)
        self.at("wrap")
        self.play(highlight(hl, code, 1), run_time=0.3)
        self.at("loop")
        self.play(highlight(hl, code, 5), run_time=0.3)
        self.end_section()

    # ------------------------------------------------------------------ 10. outro
    def outro(self):
        self.section(10)
        self.clear_stage(run_time=0.4)
        card = next_up_card(NEXT)
        self.at("next")
        self.play(FadeIn(card, shift=0.2 * UP), run_time=0.6)
        self.end_section()
        finish(self, hold=1.5)
