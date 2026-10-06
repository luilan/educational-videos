"""How LLMs Work: Deep Dive, episode 45 — Circuits.

Render from the repo root:  ./render.sh deep-dive d45
Every number on screen comes from code/d45_circuits/circuits.py (GPT-2 small, 90 indirect-object prompts; activation
patching from clean into corrupted runs; mean-ablation knockouts). assets/d45/residual.json is its residual-stream
patching grid (template 1).
"""
import json
from pathlib import Path

from manim import *

from common import MODEL_COLOR, MONO, TOKEN_COLOR, code_panel, finish, highlight
from intro import play_token_intro
from d45_script import TAGLINE, TITLE
from voiced_scene import VoicedScene

SERIES_LABEL = "HOW LLMs WORK: DEEP DIVE  ·  EPISODE 45"
GRID = json.loads((Path(__file__).parent / "assets/d45/residual.json").read_text())
HEADS = [("8.10", 0.29), ("8.6", 0.29), ("7.9", 0.23), ("9.9", 0.22), ("7.3", 0.14), ("10.0", 0.08), ("10.7", -0.35)]
CODE = """def hook(module, inp, out):
    h = out[0].clone()
    h[:, pos] = clean_acts[layer][:, pos]
    return (h,) + out[1:]"""


def sentence(s_name=" Peter", font_size=26):
    parts = [("When", WHITE), (" Kate", GREEN_B), (" and", WHITE), (" Peter", BLUE_B), (" went to the store,", WHITE),
             (s_name, BLUE_B if s_name == " Peter" else GREEN_B), (" gave a drink to", WHITE)]
    return VGroup(*[Text(t.strip(), font_size=font_size, color=c) for t, c in parts]).arrange(RIGHT, buff=0.15)


class CircuitsVideo(VoicedScene):
    VIDEO = "d45"

    def construct(self):
        play_token_intro(self, TITLE, 45, TAGLINE, label=SERIES_LABEL)
        self.hook()          # 1
        self.task()          # 2
        self.patching()      # 3
        self.residual()      # 4
        self.heads()         # 5
        self.circuit()       # 6
        self.knockout()      # 7
        self.code()          # 8
        self.outro()         # 9

    def clear_stage(self, *keep, run_time=0.6):
        self.play(*[FadeOut(m) for m in self.mobjects if m not in keep], run_time=run_time)

    # ------------------------------------------------------------------ 1. hook
    def hook(self):
        self.section(1)
        a = Text("which parts do what, and in what order", font_size=28).move_to([0, 0.8, 0])
        self.at("algorithm")
        self.play(FadeIn(a), run_time=0.4)
        b = Text("indirect object identification (IOI)", font_size=32, color=YELLOW).move_to([0, -0.6, 0])
        self.at("indirect")
        self.play(FadeIn(b), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 2. task
    def task(self):
        self.section(2)
        self.clear_stage()
        s = sentence().move_to([0, 1.6, 0])
        self.at("kate")
        self.play(FadeIn(s), run_time=0.6)
        ans = Text("→ Kate", font_size=30, color=GREEN_B).move_to([0, 0.5, 0])
        self.at("answer")
        self.play(FadeIn(ans), run_time=0.3)
        n = Text("the name that wasn't repeated", font_size=22, color=GREY_A).next_to(ans, DOWN, buff=0.2)
        self.at("repeated")
        self.play(FadeIn(n), run_time=0.3)
        st = Text("90 prompts: right name 100% · logit(Kate) − logit(Peter) = 2.68", font_size=24, color=YELLOW).move_to([0, -1.6, 0])
        self.at("90")
        self.play(FadeIn(st), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 3. patching
    def patching(self):
        self.section(3)
        self.clear_stage()
        c = sentence(font_size=20).move_to([0, 2.2, 0])
        x = sentence(" Kate", font_size=20).move_to([0, 0.2, 0])
        cl = Text("clean: answer Kate (+2.68)", font_size=20, color=GREEN_B).next_to(c, DOWN, buff=0.15)
        xl = Text("corrupted, second name swapped: answer flips (−3.30)", font_size=20, color=RED_B).next_to(x, DOWN, buff=0.15)
        self.at("corrupted")
        self.play(FadeIn(c), FadeIn(cl), FadeIn(x), FadeIn(xl), run_time=0.6)
        arr = CurvedArrow(c[5].get_bottom() + DOWN * 0.45, x[5].get_top() + UP * 0.05, angle=PI / 4, color=YELLOW)
        al = Text("copy one activation", font_size=20, color=YELLOW).next_to(arr, RIGHT, buff=0.1)
        self.at("copy")
        self.play(Create(arr), FadeIn(al), run_time=0.6)
        m = Text("recovered = (patched − corrupted) / (clean − corrupted)", font=MONO, font_size=22).move_to([0, -1.8, 0])
        self.at("measure")
        self.play(FadeIn(m), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 4. residual
    def residual(self):
        self.section(4)
        self.clear_stage()
        head = Text("patch the residual stream: share recovered", font_size=26).to_edge(UP, buff=0.4)
        self.play(FadeIn(head), run_time=0.3)
        cw, ch = 1.4, 0.38
        cells = VGroup()
        for l in range(12):
            for j in range(4):
                v = GRID[l][j]
                col = interpolate_color(BLACK, GREEN_C, max(0, min(v, 1)))
                r = Rectangle(width=cw, height=ch, stroke_width=0.5, stroke_color=GREY_D, fill_color=col, fill_opacity=1)
                r.move_to([(j - 1.5) * cw - 2.2, (5.5 - l) * ch - 0.5, 0])
                t = Text(f"{v:.2f}", font=MONO, font_size=14, color=WHITE if v < 0.6 else BLACK).move_to(r)
                cells.add(VGroup(r, t))
        cols = VGroup(*[Text(n, font_size=18).next_to(cells[j], UP, buff=0.12) for j, n in
                        enumerate(["Kate", "Peter", "Peter (2nd)", "to (last)"])])
        rows = VGroup(*[Text(str(l), font_size=16).next_to(cells[l * 4], LEFT, buff=0.12) for l in range(12)])
        yl = Text("layer", font_size=18).next_to(rows, LEFT, buff=0.2)
        self.at("patch")
        self.play(FadeIn(cells, lag_ratio=0.01), FadeIn(cols), FadeIn(rows), FadeIn(yl), run_time=1.0)
        b1 = SurroundingRectangle(VGroup(*[cells[l * 4 + 2] for l in range(7)]), color=BLUE_B, buff=0.03)
        t1 = Text("through layer 6:\nat the repeated name", font_size=22, color=BLUE_B, line_spacing=0.85).move_to([3.9, 1.2, 0])
        self.at("six")
        self.play(Create(b1), FadeIn(t1), run_time=0.5)
        b2 = SurroundingRectangle(VGroup(cells[7 * 4 + 2], cells[8 * 4 + 3]), color=YELLOW, buff=0.03)
        t2 = Text("layers 7–8:\nmoves to the last position", font_size=22, color=YELLOW, line_spacing=0.85).move_to([3.9, -0.6, 0])
        self.at("seven")
        self.play(Create(b2), FadeIn(t2), run_time=0.5)
        self.end_section()

    # ------------------------------------------------------------------ 5. heads
    def heads(self):
        self.section(5)
        self.clear_stage()
        head = Text("patch one head's output at the last position", font_size=26).to_edge(UP, buff=0.5)
        self.at("single")
        self.play(FadeIn(head), run_time=0.3)
        g = VGroup()
        for k, (n, v) in enumerate(HEADS):
            y = 1.8 - 0.62 * k
            b = Rectangle(width=abs(v) * 8, height=0.42, stroke_width=0, fill_color=GREEN_C if v > 0 else RED_C, fill_opacity=0.85)
            b.move_to([0.0, y, 0], aligned_edge=LEFT if v > 0 else RIGHT)
            g.add(VGroup(Text(n, font=MONO, font_size=22).move_to([-3.8, y, 0]), b,
                         Text(f"{v:+.2f}", font=MONO, font_size=20).move_to([3.6, y, 0])))
        axis = Line([0, 2.2, 0], [0, -2.2, 0], color=GREY_B)
        self.at("handful")
        self.play(Create(axis), FadeIn(g[:6], lag_ratio=0.15), run_time=1.0)
        self.at("wrong")
        self.play(FadeIn(g[6]), run_time=0.4)
        t = Text("argues against the answer", font_size=22, color=RED_B).move_to([-2.4, -2.6, 0])
        self.at("argues")
        self.play(FadeIn(t), run_time=0.3)
        self.end_section()

    # ------------------------------------------------------------------ 6. circuit
    def circuit(self):
        self.section(6)
        self.clear_stage()
        def stage(t1, t2, col):
            r = RoundedRectangle(width=4.0, height=1.6, corner_radius=0.12, color=col, fill_opacity=0.2)
            return VGroup(r, VGroup(Text(t1, font_size=22), Text(t2, font_size=18, color=GREY_A)).arrange(DOWN, buff=0.12).move_to(r))
        s1 = stage("early heads", "“Peter appears twice”", BLUE_C)
        s2 = stage("middle heads\n7.3, 7.9, 8.6, 8.10", "“not the repeated name”", GOLD)
        s3 = stage("name movers\n9.9, 10.0", "copy the other name: Kate", GREEN_C)
        for s in (s1, s2, s3):
            s[1].scale_to_fit_width(min(s[1].width, 3.7))
        g = VGroup(s1, s2, s3).arrange(RIGHT, buff=0.5).move_to([0, 0.4, 0])
        arrows = VGroup(Arrow(s1.get_right(), s2.get_left(), buff=0.08), Arrow(s2.get_right(), s3.get_left(), buff=0.08))
        src = Text("the published analysis: Wang et al., 2022", font_size=20, color=GREY_B).move_to([0, -1.6, 0])
        self.at("published")
        self.play(FadeIn(src), run_time=0.3)
        self.at("twice")
        self.play(FadeIn(s1), run_time=0.4)
        self.at("middle")
        self.play(GrowArrow(arrows[0]), FadeIn(s2), run_time=0.5)
        self.at("movers")
        self.play(GrowArrow(arrows[1]), FadeIn(s3), run_time=0.5)
        self.end_section()

    # ------------------------------------------------------------------ 7. knockout
    def knockout(self):
        self.section(7)
        self.clear_stage()
        head = Text("mean-ablate heads at the last position (90 prompts)", font_size=26).to_edge(UP, buff=0.5)
        self.at("knock")
        self.play(FadeIn(head), run_time=0.3)
        rows = [("intact", 2.68, GREY_B), ("top 3 heads removed", 2.08, GOLD), ("top 6 heads removed", 1.64, RED_C),
                ("6 random heads removed", 2.63, GREY_B)]
        g = VGroup()
        for k, (n, v, col) in enumerate(rows):
            y = 1.4 - 0.8 * k
            b = Rectangle(width=v * 2.2, height=0.5, stroke_width=0, fill_color=col, fill_opacity=0.85).move_to([-0.6, y, 0], aligned_edge=LEFT)
            g.add(VGroup(Text(n, font_size=22).move_to([-0.8, y, 0], aligned_edge=RIGHT), b,
                         Text(f"{v:.2f}", font=MONO, font_size=20).next_to(b, RIGHT, buff=0.15)))
        lab = Text("logit difference", font_size=18, color=GREY_B).move_to([1.8, 2.0, 0])
        self.play(FadeIn(lab), FadeIn(g[0]), run_time=0.3)
        self.at("drops")
        self.play(FadeIn(g[2]), FadeIn(g[1]), run_time=0.4)
        self.at("random")
        self.play(FadeIn(g[3]), run_time=0.3)
        t = Text("still right 93% of the time: backup heads take over", font_size=24, color=YELLOW).move_to([0, -2.3, 0])
        self.at("backup")
        self.play(FadeIn(t), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 8. code
    def code(self):
        self.section(8)
        self.clear_stage()
        code, hl = code_panel(CODE, font_size=26)
        code.move_to([0, 0.3, 0])
        self.at("code")
        self.play(FadeIn(code, shift=0.2 * UP), run_time=0.5)
        hl.match_y(code.line_numbers[0])
        self.at("hook")
        self.play(Create(hl), run_time=0.3)
        self.at("overwrite")
        self.play(highlight(hl, code, 2), run_time=0.3)
        self.end_section()

    # ------------------------------------------------------------------ 9. outro
    def outro(self):
        self.section(9)
        self.clear_stage(run_time=0.4)
        e = Text("How LLMs Work: Deep Dive · 45 episodes", font_size=30).move_to([0, 1.6, 0])
        self.at("end")
        self.play(FadeIn(e), run_time=0.4)
        parts = VGroup(*[Text(t, font_size=22, color=c) for t, c in (("bytes & tokens", TOKEN_COLOR), ("attention", BLUE_B),
                                                                       ("training", GOLD), ("scaling", GREEN_B),
                                                                       ("alignment", PURPLE_B), ("interpretability", RED_B))])
        parts.arrange(RIGHT, buff=0.45).move_to([0, 0.5, 0])
        self.at("bytes")
        self.play(LaggedStart(*[FadeIn(p) for p in parts], lag_ratio=0.3), run_time=1.6)
        c = Text("every number from code you can run", font_size=24, color=GREY_A).move_to([0, -0.6, 0])
        self.at("number")
        self.play(FadeIn(c), run_time=0.4)
        t = Text("Thanks for watching", font_size=40).move_to([0, -1.9, 0])
        self.at("thanks")
        self.play(FadeIn(t), run_time=0.5)
        self.end_section()
        finish(self, hold=2.0)
