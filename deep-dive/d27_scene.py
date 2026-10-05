"""How LLMs Work: Deep Dive, episode 27 — Sharding the Optimizer State.

Render from the repo root:  ./render.sh deep-dive d27
Every number on screen comes from code/d27_sharding/sharding.py (4 processes, torch.distributed "gloo": DDP + AdamW vs
DDP + ZeroRedundancyOptimizer(AdamW), 100 steps; the per-worker arithmetic for 7 billion parameters on 64 workers).
"""
from manim import *

from common import MODEL_COLOR, MONO, TOKEN_COLOR, code_panel, finish, highlight, next_up_card, token
from intro import play_token_intro
from d27_script import NEXT, TAGLINE, TITLE
from voiced_scene import VoicedScene

SERIES_LABEL = "HOW LLMs WORK: DEEP DIVE  ·  EPISODE 27"
ADAM = ("adam", "adams", "atom", "atoms")
COLS = {"weights": BLUE_C, "grads": ORANGE, "m": GOLD, "v": YELLOW_D}
CODE = """from torch.distributed.optim import ZeroRedundancyOptimizer

model = DDP(TinyGPT())
opt = ZeroRedundancyOptimizer(model.parameters(),
                              optimizer_class=torch.optim.AdamW, lr=1e-3)
# the loop is unchanged: opt.step() updates my slice, then all-gathers"""


def stack(sharded_from=None, k=0):
    """One worker's memory: four bars (weights, gradients, Adam m, Adam v); optionally only a quarter of some."""
    rows = VGroup()
    for name, col in COLS.items():
        full = Rectangle(width=2.0, height=0.28, stroke_width=1, color=GREY_D)
        if sharded_from and name in sharded_from:
            part = Rectangle(width=0.5, height=0.28, stroke_width=0, fill_color=col, fill_opacity=0.9)
            part.move_to(full.get_left() + (0.25 + 0.5 * k) * RIGHT)
        else:
            part = Rectangle(width=2.0, height=0.28, stroke_width=0, fill_color=col, fill_opacity=0.9).move_to(full)
        rows.add(VGroup(full, part))
    return rows.arrange(DOWN, buff=0.08)


class ShardingVideo(VoicedScene):
    VIDEO = "d27"

    def construct(self):
        play_token_intro(self, TITLE, 27, TAGLINE, label=SERIES_LABEL)
        self.hook()          # 1
        self.how()           # 2
        self.run()           # 3
        self.memory()        # 4
        self.scale()         # 5
        self.cost()          # 6
        self.names()         # 7
        self.code()          # 8
        self.outro()         # 9

    def clear_stage(self, *keep, run_time=0.6):
        self.play(*[FadeOut(m) for m in self.mobjects if m not in keep], run_time=run_time)

    def workers(self, sharded_from=None, y=0.4):
        ws = VGroup()
        for k in range(4):
            s = stack(sharded_from, k)
            t = Text(f"worker {k}", font_size=18, color=GREY_B).next_to(s, UP, buff=0.12)
            ws.add(VGroup(s, t))
        ws.arrange(RIGHT, buff=0.6).move_to([0.6, y, 0])
        legend = VGroup(*[Text(n, font_size=18, color=c) for n, c in
                          zip(("weights", "gradients", "Adam m", "Adam v"), COLS.values())])
        for lab, row in zip(legend, ws[0][0]):
            lab.next_to(row, LEFT, buff=0.25)
        return ws, legend

    # ------------------------------------------------------------------ 1. hook
    def hook(self):
        self.section(1)
        ws, legend = self.workers()
        self.at("wasteful")
        self.play(FadeIn(ws, lag_ratio=0.1), FadeIn(legend), run_time=0.8)
        dup = VGroup(*[SurroundingRectangle(VGroup(w[0][2], w[0][3]), color=RED_C, buff=0.04) for w in ws])
        self.at("same")
        self.play(Create(dup), run_time=0.5)
        t = Text("the same Adam state, copied on every worker", font_size=24, color=RED_B).move_to([0, -1.6, 0])
        self.play(FadeIn(t), run_time=0.3)
        s = Text("sharding: each worker keeps only its own slice", font_size=26, color=YELLOW).move_to([0, -2.5, 0])
        self.at("slice")
        self.play(FadeIn(s), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 2. how
    def how(self):
        self.section(2)
        self.clear_stage()
        ws, legend = self.workers(sharded_from=("m", "v"), y=0.8)
        self.at("averaged")
        self.play(FadeIn(ws), FadeIn(legend), run_time=0.6)
        upd = Text("each updates only its quarter of the weights, with its quarter of Adam's state", font_size=22)
        upd.move_to([0, -1.0, 0])
        self.at("slice")
        self.play(FadeIn(upd), run_time=0.4)
        g = Text("then all-gather: everyone ends the step with the full, identical model", font_size=22, color=YELLOW)
        g.move_to([0, -1.9, 0])
        self.at("gather")
        self.play(FadeIn(g), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 3. run
    def run(self):
        self.section(3)
        self.clear_stage()
        head = Text("4 processes, 100 steps: plain AdamW vs ZeroRedundancyOptimizer", font_size=26).to_edge(UP, buff=0.8)
        self.at("run")
        self.play(FadeIn(head), run_time=0.4)
        a = Text("largest loss difference: 0.0", font=MONO, font_size=30, color=GREEN_B).move_to([0, 0.6, 0])
        b = Text("step 100:  2.4467  vs  2.4467", font=MONO, font_size=30, color=YELLOW).move_to([0, -0.4, 0])
        self.at("identical")
        self.play(FadeIn(a), run_time=0.4)
        self.at("4467")
        self.play(FadeIn(b), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 4. memory
    def memory(self):
        self.section(4)
        self.clear_stage()
        head = Text("Adam state held by each worker (818,241 parameters)", font_size=26).to_edge(UP, buff=0.6)
        self.at("memory")
        self.play(FadeIn(head), run_time=0.4)
        rows = VGroup()
        for k, (name, v, col) in enumerate((("plain", 6.24, RED_C), ("sharded", 1.56, GREEN_C))):
            y = 0.8 - 1.2 * k
            lab = Text(name, font_size=26, color=col).move_to([-2.4, y, 0], aligned_edge=RIGHT)
            b = Rectangle(width=v * 0.9, height=0.6, stroke_width=0, fill_color=col, fill_opacity=0.85)
            b.move_to([-2.1, y, 0], aligned_edge=LEFT)
            rows.add(VGroup(lab, b, Text(f"{v:.2f} MiB", font=MONO, font_size=24).next_to(b, RIGHT, buff=0.15)))
        self.at("6")
        self.play(FadeIn(rows[0]), run_time=0.4)
        self.at("1")
        self.play(FadeIn(rows[1]), run_time=0.4)
        q = Text("a quarter, with four workers", font_size=26, color=YELLOW).move_to([0, -2.0, 0])
        self.at("quarter")
        self.play(FadeIn(q), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 5. scale
    def scale(self):
        self.section(5)
        self.clear_stage()
        head = Text("7 billion parameters, Adam in float32, 64 workers: memory per worker", font_size=26)
        head.to_edge(UP, buff=0.6)
        self.at("7", "seven")
        self.play(FadeIn(head), run_time=0.4)
        rows = [("everything copied", 112.0, RED_C, "112"), ("Adam state sharded", 56.9, GOLD, "57"),
                ("weights, gradients and state sharded", 1.8, GREEN_C, "two")]
        out = VGroup()
        for k, (name, v, col, cue) in enumerate(rows):
            y = 1.2 - 1.1 * k
            lab = Text(name, font_size=22, color=col).move_to([-1.2, y, 0], aligned_edge=RIGHT)
            b = Rectangle(width=max(0.05, v * 0.05), height=0.55, stroke_width=0, fill_color=col, fill_opacity=0.85)
            b.move_to([-0.9, y, 0], aligned_edge=LEFT)
            out.add(VGroup(lab, b, Text(f"{v:.1f} GB", font=MONO, font_size=24).next_to(b, RIGHT, buff=0.15)))
            self.at(cue)
            self.play(FadeIn(out[k]), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 6. cost
    def cost(self):
        self.section(6)
        self.clear_stage()
        layers = VGroup(*[RoundedRectangle(width=1.6, height=0.6, corner_radius=0.1, color=GREEN_C, fill_opacity=0.3)
                          for _ in range(5)]).arrange(RIGHT, buff=0.5).move_to([0, 0.8, 0])
        self.at("nothing")
        self.play(FadeIn(layers), run_time=0.4)
        gathers = VGroup(*[Text("gather ↓", font_size=18, color=YELLOW).next_to(l, UP, buff=0.15) for l in layers])
        frees = VGroup(*[Text("free ↑", font_size=18, color=GREY_B).next_to(l, DOWN, buff=0.15) for l in layers])
        self.at("gathered")
        self.play(LaggedStart(*[FadeIn(g) for g in gathers], lag_ratio=0.15), run_time=0.7)
        self.play(LaggedStart(*[FadeIn(f) for f in frees], lag_ratio=0.15), run_time=0.6)
        t = Text("more communication, traded for memory", font_size=26, color=YELLOW).move_to([0, -1.4, 0])
        self.at("communication")
        self.play(FadeIn(t), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 7. names
    def names(self):
        self.section(7)
        self.clear_stage()
        a = Text("ZeRO  (Microsoft DeepSpeed)", font_size=30).move_to([0, 1.0, 0])
        b = Text("FSDP: fully sharded data parallel  (PyTorch)", font_size=30).move_to([0, 0.0, 0])
        self.at("zro", "zero", "deep")
        self.play(FadeIn(a), run_time=0.4)
        self.at("fsdp")
        self.play(FadeIn(b), run_time=0.4)
        c = Text("no worker stores what another already holds", font_size=26, color=YELLOW).move_to([0, -1.3, 0])
        self.at("principle")
        self.play(FadeIn(c), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 8. code
    def code(self):
        self.section(8)
        self.clear_stage()
        code, hl = code_panel(CODE, font_size=22)
        code.move_to([0, 0.3, 0])
        self.at("code")
        self.play(FadeIn(code, shift=0.2 * UP), run_time=0.5)
        hl.match_y(code.line_numbers[3])
        self.at("wrap")
        self.play(Create(hl), run_time=0.3)
        self.at("loop")
        self.play(highlight(hl, code, 5), run_time=0.3)
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
