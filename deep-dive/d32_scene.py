"""How LLMs Work: Deep Dive, episode 32 — Batching and Paged Attention.

Render from the repo root:  ./render.sh deep-dive d32
Every number on screen comes from code/d32_batching_paging/batching_paging.py (GPT-2 small generating 64 tokens for 1,
4 and 16 requests on this CPU; padding waste for 16 uneven requests; a KV-cache allocation count for 100 requests).
"""
import random

from manim import *

from common import MODEL_COLOR, MONO, TOKEN_COLOR, code_panel, finish, highlight, next_up_card, token
from intro import play_token_intro
from d32_script import NEXT, TAGLINE, TITLE
from voiced_scene import VoicedScene

SERIES_LABEL = "HOW LLMs WORK: DEEP DIVE  ·  EPISODE 32"
LENGTHS = [20, 50, 50, 50, 100, 100, 100, 100, 400, 400, 400, 400, 400, 1000, 1000, 1000]
CODE = """PAGE = 16
pages = -(-n_tokens // PAGE)          # round up to whole pages
while len(table[req]) < pages:
    table[req].append(free.pop())     # grab a free page
# attention reads keys and values page by page"""


class BatchingVideo(VoicedScene):
    VIDEO = "d32"

    def construct(self):
        play_token_intro(self, TITLE, 32, TAGLINE, label=SERIES_LABEL)
        self.hook()          # 1
        self.batching()      # 2
        self.why()           # 3
        self.padding()       # 4
        self.continuous()    # 5
        self.memory()        # 6
        self.paging()        # 7
        self.code()          # 8
        self.outro()         # 9

    def clear_stage(self, *keep, run_time=0.6):
        self.play(*[FadeOut(m) for m in self.mobjects if m not in keep], run_time=run_time)

    # ------------------------------------------------------------------ 1. hook
    def hook(self):
        self.section(1)
        model = RoundedRectangle(width=2.2, height=1.4, corner_radius=0.15, color=MODEL_COLOR, fill_opacity=0.35).move_to([2.6, 0.3, 0])
        ml = Text("one model", font_size=24).move_to(model)
        users = VGroup(*[Circle(radius=0.25, color=TOKEN_COLOR, fill_opacity=0.4) for _ in range(8)])
        users.arrange_in_grid(rows=4, cols=2, buff=0.35).move_to([-3.4, 0.3, 0])
        self.at("user")
        self.play(FadeIn(model), FadeIn(ml), FadeIn(users[0]), run_time=0.5)
        arrows = VGroup(*[Arrow(u.get_right(), model.get_left(), buff=0.1, stroke_width=2, color=GREY_B) for u in users])
        self.at("many")
        self.play(FadeIn(users[1:]), LaggedStart(*[GrowArrow(a) for a in arrows], lag_ratio=0.05), run_time=0.7)
        ideas = Text("batching · continuous batching · paged attention", font_size=26, color=YELLOW).move_to([0, -2.2, 0])
        self.at("three")
        self.play(FadeIn(ideas), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 2. batching
    def batching(self):
        self.section(2)
        self.clear_stage()
        head = Text("GPT-2, 64 new tokens per request (this CPU)", font_size=26).to_edge(UP, buff=0.6)
        self.at("batching")
        self.play(FadeIn(head), run_time=0.3)
        rows = [("1 request", 26, "26"), ("4 at once", 153, "153"), ("16 at once", 404, "400")]
        out = VGroup()
        for k, (name, v, cue) in enumerate(rows):
            y = 1.0 - 1.0 * k
            lab = Text(name, font_size=24).move_to([-2.6, y, 0], aligned_edge=RIGHT)
            b = Rectangle(width=v * 0.016, height=0.55, stroke_width=0, fill_color=GREEN_C, fill_opacity=0.85)
            b.move_to([-2.3, y, 0], aligned_edge=LEFT)
            out.add(VGroup(lab, b, Text(f"{v} tokens/s", font=MONO, font_size=22).next_to(b, RIGHT, buff=0.15)))
            self.at(cue)
            self.play(FadeIn(out[k]), run_time=0.4)
        x = Text("15x the throughput", font_size=28, color=YELLOW).move_to([0, -2.2, 0])
        self.at("15")
        self.play(FadeIn(x), run_time=0.3)
        self.end_section()

    # ------------------------------------------------------------------ 3. why
    def why(self):
        self.section(3)
        self.clear_stage()
        w = Rectangle(width=2.4, height=2.4, color=MODEL_COLOR, fill_opacity=0.3).move_to([-2.6, 0.3, 0])
        wl = Text("weights", font_size=24).move_to(w)
        one = Rectangle(width=2.4, height=0.25, color=TOKEN_COLOR, fill_opacity=0.5).next_to(w, LEFT, buff=0.6).rotate(PI / 2)
        self.at("reads")
        self.play(FadeIn(w), FadeIn(wl), run_time=0.4)
        a = Text("1 request: read all the weights\nfor one row of math", font_size=22, line_spacing=0.85).move_to([2.6, 1.2, 0])
        self.play(FadeIn(a), run_time=0.4)
        b = Text("16 requests: the same read\nserves 16 rows", font_size=22, color=GREEN_B, line_spacing=0.85).move_to([2.6, 0.0, 0])
        self.at("pass")
        self.play(FadeIn(b), run_time=0.4)
        c = Text("here, 4 requests (1.68 s) even finished\nsooner than 1 request (2.42 s)", font_size=20, color=YELLOW,
                 line_spacing=0.85).move_to([2.6, -1.4, 0])
        self.at("sooner")
        self.play(FadeIn(c), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 4. padding
    def padding(self):
        self.section(4)
        self.clear_stage()
        head = Text("16 requests, 20 to 1,000 tokens, in one fixed batch", font_size=26).to_edge(UP, buff=0.5)
        self.at("lengths")
        self.play(FadeIn(head), run_time=0.4)
        scale = 9.0 / 1000
        rows = VGroup()
        for k, n in enumerate(LENGTHS):
            y = 2.2 - 0.27 * k
            full = Rectangle(width=1000 * scale, height=0.2, stroke_width=0, fill_color=GREY_E, fill_opacity=0.9)
            used = Rectangle(width=n * scale, height=0.2, stroke_width=0, fill_color=GREEN_C, fill_opacity=0.85)
            full.move_to([-4.5, y, 0], aligned_edge=LEFT)
            used.move_to([-4.5, y, 0], aligned_edge=LEFT)
            rows.add(VGroup(full, used))
        self.play(FadeIn(VGroup(*[r[1] for r in rows]), lag_ratio=0.03), run_time=0.6)
        self.at("longest")
        self.play(FadeIn(VGroup(*[r[0] for r in rows])), *[r[1].animate.set_z_index(1) for r in rows], run_time=0.5)
        w = Text("16,000 token slots, 5,570 useful: 65% wasted (grey)", font_size=24, color=RED_B).move_to([0, -2.6, 0])
        self.at("65")
        self.play(FadeIn(w), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 5. continuous
    def continuous(self):
        self.section(5)
        self.clear_stage()
        head = Text("continuous batching", font_size=30, color=GREEN_B).to_edge(UP, buff=0.6)
        self.at("continuous")
        self.play(FadeIn(head), run_time=0.3)
        random.seed(3)
        lanes = VGroup()
        cols = [GREEN_C, BLUE_C, GOLD, PURPLE_B, TEAL_C, MAROON_C]
        for lane in range(4):
            x, k, segs = -5.0, lane, VGroup()
            while x < 5.0:
                w = random.uniform(0.8, 3.0)
                w = min(w, 5.0 - x)
                segs.add(Rectangle(width=w - 0.05, height=0.45, stroke_width=0, fill_color=cols[k % len(cols)],
                                   fill_opacity=0.8).move_to([x + w / 2, 1.2 - 0.7 * lane, 0]))
                x += w
                k += 1
            lanes.add(segs)
        self.at("leaves")
        self.play(LaggedStart(*[FadeIn(s, lag_ratio=0.2) for s in lanes], lag_ratio=0.1), run_time=1.2)
        f = Text("a finished request leaves; a waiting one joins at the next step", font_size=22, color=GREY_A)
        f.move_to([0, -1.9, 0])
        self.play(FadeIn(f), run_time=0.3)
        t = Text("the batch stays full of useful work", font_size=24, color=YELLOW).move_to([0, -2.6, 0])
        self.at("full")
        self.play(FadeIn(t), run_time=0.3)
        self.end_section()

    # ------------------------------------------------------------------ 6. memory
    def kv_rows(self, paged):
        random.seed(1)
        reqs = [random.randint(50, 1500) for _ in range(100)][:12]
        scale = 4.5 / 2048
        rows = VGroup()
        for k, n in enumerate(reqs):
            y = 1.6 - 0.3 * k
            if paged:
                pages = -(-n // 16) * 16
                block = Rectangle(width=pages * scale, height=0.22, stroke_width=0, fill_color=GREEN_C, fill_opacity=0.85)
                block.move_to([-6.0 if not paged else 1.0, y, 0], aligned_edge=LEFT)
                rows.add(block)
            else:
                full = Rectangle(width=2048 * scale, height=0.22, stroke_width=0, fill_color=GREY_E, fill_opacity=0.9)
                used = Rectangle(width=n * scale, height=0.22, stroke_width=0, fill_color=BLUE_C, fill_opacity=0.85)
                full.move_to([-6.0, y, 0], aligned_edge=LEFT)
                used.move_to([-6.0, y, 0], aligned_edge=LEFT)
                rows.add(VGroup(full, used))
        return rows

    def memory(self):
        self.section(6)
        self.clear_stage()
        head = Text("KV-cache memory, 100 requests of 50 to 1,500 tokens (first 12 shown)", font_size=24).to_edge(UP, buff=0.5)
        self.at("memory")
        self.play(FadeIn(head), run_time=0.3)
        self.cont = self.kv_rows(paged=False)
        cl = Text("reserve 2,048 tokens each", font_size=22, color=BLUE_B).move_to([-3.75, 2.2, 0])
        self.at("2")
        self.play(FadeIn(self.cont, lag_ratio=0.03), FadeIn(cl), run_time=0.7)
        a = Text("7.03 GiB, 40% used", font=MONO, font_size=24, color=RED_B).move_to([-3.75, -2.3, 0])
        self.at("40")
        self.play(FadeIn(a), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 7. paging
    def paging(self):
        self.section(7)
        paged = self.kv_rows(paged=True)
        pl = Text("pages of 16 tokens, as needed", font_size=22, color=GREEN_B).move_to([3.4, 2.2, 0])
        self.at("pages")
        self.play(FadeIn(paged, lag_ratio=0.03), FadeIn(pl), run_time=0.7)
        b = Text("2.81 GiB, 99.0% used", font=MONO, font_size=24, color=GREEN_B).move_to([3.4, -2.3, 0])
        self.at("99")
        self.play(FadeIn(b), run_time=0.4)
        c = Text("the same memory holds 2.5x more requests", font_size=24, color=YELLOW).move_to([0, -3.1, 0])
        self.at("half")
        self.play(FadeIn(c), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 8. code
    def code(self):
        self.section(8)
        self.clear_stage()
        code, hl = code_panel(CODE, font_size=26)
        code.move_to([0, 0.3, 0])
        self.at("code")
        self.play(FadeIn(code, shift=0.2 * UP), run_time=0.5)
        hl.match_y(code.line_numbers[1])
        self.at("round")
        self.play(Create(hl), run_time=0.3)
        self.at("table")
        self.play(highlight(hl, code, 3), run_time=0.3)
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
