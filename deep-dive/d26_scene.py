"""How LLMs Work: Deep Dive, episode 26 — Tensor and Pipeline Parallelism.

Render from the repo root:  ./render.sh deep-dive d26
Every number on screen comes from code/d26_tensor_pipeline/tensor_pipeline.py (2 processes with torch.distributed
"gloo": a tensor-parallel MLP 128 -> 512 -> 128 and a 4-layer pipeline split 2 + 2; the bubble formula).
"""
from manim import *

from common import MODEL_COLOR, MONO, TOKEN_COLOR, code_panel, finish, highlight, next_up_card, token
from intro import play_token_intro
from d26_script import NEXT, TAGLINE, TITLE
from voiced_scene import VoicedScene

SERIES_LABEL = "HOW LLMs WORK: DEEP DIVE  ·  EPISODE 26"
W0, W1 = BLUE_C, GOLD
CODE = """# tensor parallel, worker k of 2
half = slice(k * 256, (k + 1) * 256)
partial = F.gelu(x @ w1[:, half] + b1[half]) @ w2[half, :]
dist.all_reduce(partial)          # sum of halves = full output
out = partial + b2

# pipeline parallel
dist.send(h, dst=1)      # worker 0 → worker 1
dist.recv(h, src=0)"""


class TensorPipelineVideo(VoicedScene):
    VIDEO = "d26"

    def construct(self):
        play_token_intro(self, TITLE, 26, TAGLINE, label=SERIES_LABEL)
        self.hook()          # 1
        self.tensor()        # 2
        self.check()         # 3
        self.tp_cost()       # 4
        self.pipeline()      # 5
        self.bubble()        # 6
        self.combine()       # 7
        self.code()          # 8
        self.outro()         # 9

    def clear_stage(self, *keep, run_time=0.6):
        self.play(*[FadeOut(m) for m in self.mobjects if m not in keep], run_time=run_time)

    # ------------------------------------------------------------------ 1. hook
    def hook(self):
        self.section(1)
        need = Rectangle(width=11.2 * 0.6, height=0.7, stroke_width=0, fill_color=RED_C, fill_opacity=0.8)
        have = Rectangle(width=8.0 * 0.6, height=0.7, stroke_width=0, fill_color=GREEN_C, fill_opacity=0.8)
        need.move_to([-3.0, 1.0, 0], aligned_edge=LEFT)
        have.move_to([-3.0, 0.0, 0], aligned_edge=LEFT)
        nl = Text("7B model with Adam: 112 GB per worker", font_size=22).next_to(need, LEFT, buff=0.2)
        hl = Text("one 80 GB GPU", font_size=22).next_to(have, LEFT, buff=0.2)
        VGroup(need, have, nl, hl).move_to([0, 0.6, 0])
        self.at("112")
        self.play(FadeIn(need), FadeIn(nl), FadeIn(have), FadeIn(hl), run_time=0.6)
        s = Text("split the model itself: two classic ways", font_size=28, color=YELLOW).move_to([0, -1.6, 0])
        self.at("split")
        self.play(FadeIn(s), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 2. tensor parallelism
    def tensor(self):
        self.section(2)
        self.clear_stage()
        head = Text("tensor parallelism: split each matrix", font_size=30).to_edge(UP, buff=0.5)
        self.at("tensor")
        self.play(FadeIn(head), run_time=0.4)
        a0 = Rectangle(width=1.6, height=1.2, stroke_width=1, color=WHITE, fill_color=W0, fill_opacity=0.6)
        a1 = Rectangle(width=1.6, height=1.2, stroke_width=1, color=WHITE, fill_color=W1, fill_opacity=0.25)
        w1 = VGroup(a0, a1).arrange(RIGHT, buff=0).move_to([-3.2, 0.6, 0])
        w1l = Text("first matrix  128 × 512", font_size=20).next_to(w1, UP, buff=0.15)
        b0 = Rectangle(width=1.2, height=1.6, stroke_width=1, color=WHITE, fill_color=W0, fill_opacity=0.6)
        b1 = Rectangle(width=1.2, height=1.6, stroke_width=1, color=WHITE, fill_color=W1, fill_opacity=0.25)
        w2 = VGroup(b0, b1).arrange(DOWN, buff=0).move_to([3.2, 0.8, 0])
        w2l = Text("second matrix  512 × 128", font_size=20).next_to(w2, UP, buff=0.15)
        self.at("128")
        self.play(FadeIn(w1), FadeIn(w1l), FadeIn(w2), FadeIn(w2l), run_time=0.6)
        c = Text("worker 0: half the columns\n→ half the hidden units", font_size=20, color=W0, line_spacing=0.85)
        c.next_to(w1, DOWN, buff=0.3)
        self.at("columns")
        self.play(Indicate(a0, color=W0), FadeIn(c), run_time=0.6)
        g = Text("GELU acts on each unit separately ✓", font_size=20, color=GREY_A).move_to([-3.2, -2.4, 0])
        self.at("jell", "gelu")
        self.play(FadeIn(g), run_time=0.4)
        r = Text("matching half of the rows\n→ a partial output", font_size=20, color=W0, line_spacing=0.85)
        r.next_to(w2, DOWN, buff=0.3)
        self.at("rows")
        self.play(Indicate(b0, color=W0), FadeIn(r), run_time=0.6)
        self.mats = VGroup(w1, w2)
        self.end_section()

    # ------------------------------------------------------------------ 3. check
    def check(self):
        self.section(3)
        self.clear_stage()
        p0 = RoundedRectangle(width=3.6, height=0.8, corner_radius=0.12, color=W0, fill_opacity=0.35)
        p1 = RoundedRectangle(width=3.6, height=0.8, corner_radius=0.12, color=W1, fill_opacity=0.35)
        VGroup(p0, p1).arrange(RIGHT, buff=2.0).move_to([0, 1.6, 0])
        l0 = Text("partial output, worker 0", font_size=18).move_to(p0)
        l1 = Text("partial output, worker 1", font_size=18).move_to(p1)
        self.at("worker")
        self.play(FadeIn(p0), FadeIn(l0), FadeIn(p1), FadeIn(l1), run_time=0.5)
        s = Circle(radius=0.35, color=WHITE).move_to([0, 0.2, 0])
        sl = Text("+", font_size=32).move_to(s)
        arr = VGroup(Arrow(p0.get_bottom(), s.get_left(), buff=0.1), Arrow(p1.get_bottom(), s.get_right(), buff=0.1))
        al = Text("all-reduce (sum)", font_size=20, color=YELLOW).next_to(s, RIGHT, buff=0.5)
        self.at("all")
        self.play(FadeIn(s), FadeIn(sl), GrowArrow(arr[0]), GrowArrow(arr[1]), FadeIn(al), run_time=0.6)
        ok = Text("vs the full MLP: largest difference 1.2 × 10⁻⁶", font=MONO, font_size=24, color=GREEN_B).move_to([0, -1.2, 0])
        self.at("millionth")
        self.play(FadeIn(ok), run_time=0.4)
        h = Text("each worker holds 65,792 of the 131,584 parameters", font_size=24).move_to([0, -2.1, 0])
        self.at("half")
        self.play(FadeIn(h), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 4. cost
    def tp_cost(self):
        self.section(4)
        self.clear_stage()
        layers = VGroup(*[RoundedRectangle(width=1.4, height=0.6, corner_radius=0.1, color=GREEN_C, fill_opacity=0.3)
                          for _ in range(6)]).arrange(RIGHT, buff=0.55).move_to([0, 1.0, 0])
        stars = VGroup(*[Text("⇄", font_size=26, color=YELLOW).next_to(l, RIGHT, buff=0.08) for l in layers[:-1]])
        self.at("price")
        self.play(FadeIn(layers), FadeIn(stars), run_time=0.6)
        a = Text("an all-reduce in every layer (256 KiB here), attention split by heads too", font_size=22)
        a.move_to([0, -0.2, 0])
        self.at("heads")
        self.play(FadeIn(a), run_time=0.4)
        m = Text("→ usually inside one machine, where GPU links are fastest", font_size=24, color=YELLOW).move_to([0, -1.2, 0])
        self.at("machine")
        self.play(FadeIn(m), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 5. pipeline
    def pipeline(self):
        self.section(5)
        self.clear_stage()
        head = Text("pipeline parallelism: split the layers", font_size=30).to_edge(UP, buff=0.5)
        self.at("pipeline")
        self.play(FadeIn(head), run_time=0.4)
        g0 = VGroup(*[RoundedRectangle(width=1.4, height=0.6, corner_radius=0.1, color=W0, fill_opacity=0.4) for _ in range(2)])
        g1 = VGroup(*[RoundedRectangle(width=1.4, height=0.6, corner_radius=0.1, color=W1, fill_opacity=0.4) for _ in range(2)])
        for g, names in ((g0, ("layer 0", "layer 1")), (g1, ("layer 2", "layer 3"))):
            g.arrange(DOWN, buff=0.2)
        lab0 = VGroup(*[Text(n, font_size=18).move_to(b) for n, b in zip(("layer 0", "layer 1"), g0)])
        lab1 = VGroup(*[Text(n, font_size=18).move_to(b) for n, b in zip(("layer 2", "layer 3"), g1)])
        w0 = VGroup(g0, lab0).move_to([-3.0, 0.2, 0])
        w1 = VGroup(g1, lab1).move_to([3.0, 0.2, 0])
        t0 = Text("worker 0", font_size=20, color=W0).next_to(w0, UP, buff=0.2)
        t1 = Text("worker 1", font_size=20, color=W1).next_to(w1, UP, buff=0.2)
        self.play(FadeIn(w0), FadeIn(w1), FadeIn(t0), FadeIn(t1), run_time=0.5)
        arr = Arrow(w0.get_right(), w1.get_left(), buff=0.3, color=YELLOW)
        al = Text("activations (64 KiB\nper micro-batch)", font_size=18, color=YELLOW, line_spacing=0.8).next_to(arr, UP, 0.1)
        self.at("sends")
        self.play(GrowArrow(arr), FadeIn(al), run_time=0.5)
        ok = Text("vs the full model: difference 0.0", font=MONO, font_size=24, color=GREEN_B).move_to([0, -2.0, 0])
        self.at("exactly")
        self.play(FadeIn(ok), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 6. bubble
    def gantt(self, m, y0, label):
        unit = 4.8 / (m + 1)
        rows = VGroup()
        for stage, col in ((0, W0), (1, W1)):
            y = y0 - 0.5 * stage
            for k in range(m):
                start = stage + k
                rows.add(Rectangle(width=unit * 0.95, height=0.4, stroke_width=0, fill_color=col, fill_opacity=0.8)
                         .move_to([-2.0 + (start + 0.5) * unit, y, 0]))
        frame = VGroup(*[Rectangle(width=4.8, height=0.4, stroke_width=1, color=GREY_D).move_to([0.4, y0 - 0.5 * s, 0])
                         for s in range(2)])
        names = VGroup(Text("worker 0", font_size=16, color=W0).next_to(frame[0], LEFT, buff=0.15),
                       Text("worker 1", font_size=16, color=W1).next_to(frame[1], LEFT, buff=0.15))
        cap = Text(label, font_size=20, color=YELLOW).next_to(frame, RIGHT, buff=0.3)
        return VGroup(frame, rows, names, cap)

    def bubble(self):
        self.section(6)
        self.clear_stage()
        head = Text("the bubble: idle time at the start and end (time →)", font_size=26).to_edge(UP, buff=0.5)
        self.at("bubble")
        self.play(FadeIn(head), run_time=0.4)
        g1 = self.gantt(1, 1.6, "1 batch: 50% idle")
        self.at("half")
        self.play(FadeIn(g1), run_time=0.6)
        g4 = self.gantt(4, 0.0, "4 micro-batches: 20% idle")
        self.at("four")
        self.play(FadeIn(g4), run_time=0.6)
        tab = Text("idle = (stages − 1) / (micro-batches + stages − 1)\n"
                   "2 stages: 16 micro-batches → 6%     8 stages: 8 → 47%, 32 → 18%", font=MONO, font_size=20,
                   line_spacing=0.9).move_to([0, -2.1, 0])
        self.at("16")
        self.play(FadeIn(tab), run_time=0.5)
        self.end_section()

    # ------------------------------------------------------------------ 7. combine
    def combine(self):
        self.section(7)
        self.clear_stage()
        head = Text("real runs combine all three", font_size=30).to_edge(UP, buff=0.6)
        self.at("three")
        self.play(FadeIn(head), run_time=0.4)
        items = VGroup(Text("tensor parallel: inside a machine", font_size=26, color=W0),
                       Text("pipeline parallel: across machines", font_size=26, color=W1),
                       Text("data parallel: many copies of the whole arrangement", font_size=26, color=GREEN_C))
        items.arrange(DOWN, buff=0.45, aligned_edge=LEFT).move_to([0, 0.0, 0])
        for k, cue in enumerate(("tensor", "pipeline", "data")):
            self.at(cue)
            self.play(FadeIn(items[k]), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 8. code
    def code(self):
        self.section(8)
        self.clear_stage()
        code, hl = code_panel(CODE, font_size=24)
        code.move_to([0, 0.2, 0])
        self.at("code")
        self.play(FadeIn(code, shift=0.2 * UP), run_time=0.5)
        hl.match_y(code.line_numbers[2])
        self.at("slice")
        self.play(Create(hl), run_time=0.3)
        self.at("reduce")
        self.play(highlight(hl, code, 3), run_time=0.3)
        self.at("send")
        self.play(highlight(hl, code, 7), run_time=0.3)
        self.end_section()

    # ------------------------------------------------------------------ 9. outro
    def outro(self):
        self.section(9)
        self.clear_stage(run_time=0.4)
        card = next_up_card(NEXT)
        self.at("next")
        self.play(FadeIn(card, shift=0.2 * UP), run_time=0.6)
        self.end_section()
        finish(self, hold=1.5)
