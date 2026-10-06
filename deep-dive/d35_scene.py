"""How LLMs Work: Deep Dive, episode 35 — Decoding Strategies Compared.

Render from the repo root:  ./render.sh deep-dive d35
Every number on screen comes from code/d35_decoding/decoding.py (GPT-2 small continues 4 prompts for 120 tokens; repetition
= share of repeated 4-grams; judge = Qwen2.5-1.5B's mean cross-entropy on the continuation). assets/d35/dist.json is
GPT-2's next-token distribution after prompt 1.
"""
import json
from pathlib import Path

from manim import *

from common import MODEL_COLOR, MONO, TOKEN_COLOR, code_panel, finish, highlight, next_up_card
from intro import play_token_intro
from d35_script import NEXT, TAGLINE, TITLE
from voiced_scene import VoicedScene

SERIES_LABEL = "HOW LLMs WORK: DEEP DIVE  ·  EPISODE 35"
D = json.loads((Path(__file__).parent / "assets/d35/dist.json").read_text())
RESULTS = [("greedy", 69, 0.67), ("beam, 4", 76, 0.57), ("T 0.7", 4, 2.50), ("T 1.0", 0, 4.91), ("T 1.5", 0, 9.10),
           ("top-k 40", 2, 3.10), ("top-p 0.9", 0, 3.87), ("min-p 0.1", 10, 2.28)]
CODE = """logits = logits / temperature
p = logits.softmax(-1)
logits[p < 0.1 * p.max()] = -inf     # min-p
token = torch.multinomial(logits.softmax(-1), 1)"""


def dist_bars(ps, tail, labels=True, height=3.2, top=0.17, tail_font=18):
    g = VGroup()
    for v in ps:
        g.add(Rectangle(width=0.55, height=max(v / top * height, 0.02), stroke_width=0, fill_color=TOKEN_COLOR, fill_opacity=0.85))
    tb = Rectangle(width=2.2, height=0.04, stroke_width=0, fill_color=GREY_B, fill_opacity=0.85)
    g.add(tb)
    g.arrange(RIGHT, buff=0.1, aligned_edge=DOWN)
    tb.shift(RIGHT * 0.3)
    tl = Text(f"the other 50,245\ntokens: {tail * 100:.0f}%", font_size=tail_font, color=GREY_B, line_spacing=0.8)
    tl.next_to(tb, UP, buff=0.15)
    out = VGroup(g, tl)
    if labels:
        out.add(VGroup(*[Text(t.strip() or t, font=MONO, font_size=16).rotate(PI / 4).next_to(g[k], DOWN, buff=0.15)
                         for k, t in enumerate(D["tokens"])]))
    return out


class DecodingVideo(VoicedScene):
    VIDEO = "d35"

    def construct(self):
        play_token_intro(self, TITLE, 35, TAGLINE, label=SERIES_LABEL)
        self.hook()          # 1
        self.measures()      # 2
        self.greedy()        # 3
        self.beam()          # 4
        self.temperature()   # 5
        self.truncation()    # 6
        self.results()       # 7
        self.code()          # 8
        self.outro()         # 9

    def clear_stage(self, *keep, run_time=0.6):
        self.play(*[FadeOut(m) for m in self.mobjects if m not in keep], run_time=run_time)

    # ------------------------------------------------------------------ 1. hook
    def hook(self):
        self.section(1)
        pr = Text("“The old lighthouse keeper opened the door and …”", font_size=26).to_edge(UP, buff=0.7)
        bars = dist_bars(D["p"], D["tail_after_12"]).move_to([0, -0.4, 0])
        self.at("probabilities")
        self.play(FadeIn(pr), FadeIn(bars), run_time=0.6)
        outs = VGroup(Text("a broken record", font_size=24, color=RED_B), Text("a poet", font_size=24, color=GREEN_B),
                      Text("a random word generator", font_size=24, color=GOLD)).arrange(RIGHT, buff=0.8).move_to([0, -3.0, 0])
        for k, cue in enumerate(("broken", "poet", "random")):
            self.at(cue)
            self.play(FadeIn(outs[k]), run_time=0.3)
        self.end_section()

    # ------------------------------------------------------------------ 2. setup
    def measures(self):
        self.section(2)
        self.clear_stage()
        a = Text("GPT-2 · 4 story openings · 120 new tokens", font_size=28).move_to([0, 1.8, 0])
        self.at("gpt")
        self.play(FadeIn(a), run_time=0.4)
        m1 = VGroup(Text("repetition", font_size=26, color=RED_B),
                    Text("share of 4-token phrases\nalready used", font_size=20, line_spacing=0.8)).arrange(DOWN)
        m2 = VGroup(Text("judge surprise", font_size=26, color=BLUE_B),
                    Text("Qwen2.5-1.5B's loss on the text\n(lower = more predictable)", font_size=20, line_spacing=0.8)).arrange(DOWN)
        m1.move_to([-3.2, -0.4, 0])
        m2.move_to([3.2, -0.4, 0])
        self.at("share")
        self.play(FadeIn(m1), run_time=0.4)
        self.at("surprising")
        self.play(FadeIn(m2), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 3. greedy
    def greedy(self):
        self.section(3)
        self.clear_stage()
        h = Text("greedy: always the most likely token", font_size=28).to_edge(UP, buff=0.6)
        self.at("greedy")
        self.play(FadeIn(h), run_time=0.3)
        lines = ["…saw the man standing there. He was wearing a black suit and a black hat.",
                 "He was wearing a black hat with a black belt.", "He was wearing a black hat with a black belt.",
                 "He was wearing a black hat with a black belt.", "He was wearing a black hat with a black belt. …"]
        t = VGroup(*[Text(l, font_size=22, color=GREY_A if k == 0 else RED_B) for k, l in enumerate(lines)])
        t.arrange(DOWN, aligned_edge=LEFT, buff=0.18).move_to([0, 0.2, 0])
        self.at("starts")
        self.play(FadeIn(t[0]), run_time=0.4)
        self.at("stuck")
        self.play(LaggedStart(*[FadeIn(x) for x in t[1:]], lag_ratio=0.4), run_time=1.6)
        r = Text("69% of its 4-token phrases are repeats", font_size=26, color=YELLOW).move_to([0, -2.5, 0])
        self.at("69")
        self.play(FadeIn(r), run_time=0.3)
        self.end_section()

    # ------------------------------------------------------------------ 4. beam
    def beam(self):
        self.section(4)
        self.clear_stage()
        h = Text("beam search: keep the 4 most likely partial texts", font_size=28).to_edge(UP, buff=0.6)
        self.at("beam")
        self.play(FadeIn(h), run_time=0.3)
        t = Text("“What are you doing here?” / “I don't know,” she said. /\n“What are you doing here?” / “I don't know,” he said. / …",
                 font_size=22, color=RED_B, line_spacing=0.9).move_to([0, 1.4, 0])
        self.play(FadeIn(t), run_time=0.4)
        rows = [("greedy", 69, 0.67), ("beam", 76, 0.57)]
        tbl = VGroup(Text("repetition     judge surprise", font_size=22, color=GREY_B),
                     *[Text(f"{n:<8}{r:>4}%{l:>14.2f}", font=MONO, font_size=24) for n, r, l in rows])
        tbl.arrange(DOWN, aligned_edge=RIGHT, buff=0.25).move_to([0, -0.6, 0])
        self.at("76")
        self.play(FadeIn(tbl), run_time=0.4)
        self.at("least")
        self.play(Indicate(tbl[1:], color=BLUE_B), run_time=0.8)
        m = Text("the most likely text is not the best text", font_size=28, color=YELLOW).move_to([0, -2.6, 0])
        self.at("best")
        self.play(FadeIn(m), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 5. temperature
    def temperature(self):
        self.section(5)
        self.clear_stage()
        h = Text("sampling with temperature T: probabilities ∝ exp(logit / T)", font_size=26).to_edge(UP, buff=0.5)
        self.at("temperature")
        self.play(FadeIn(h), run_time=0.4)
        cols = VGroup()
        for T, ps, tail, loss in ((0.7, D["T0.7"], D["T0.7_tail"], 2.50), (1.0, D["p"], D["tail_after_12"], 4.91),
                                  (1.5, D["T1.5"], D["T1.5_tail"], 9.10)):
            b = VGroup(*[Rectangle(width=0.22, height=max(v / 0.17 * 2.6, 0.02), stroke_width=0, fill_color=TOKEN_COLOR,
                                   fill_opacity=0.85) for v in ps]).arrange(RIGHT, buff=0.06, aligned_edge=DOWN)
            b = VGroup(b, Text(f"top 12 shown; the rest: {tail * 100:.0f}%", font_size=18, color=GREY_B).next_to(b, DOWN, buff=0.15))
            lab = Text(f"T = {T}", font_size=24).next_to(b, UP, buff=0.2)
            ls = Text(f"judge surprise {loss:.2f}", font=MONO, font_size=20, color=RED_B if loss > 5 else GREEN_B).next_to(b, DOWN, buff=0.25)
            cols.add(VGroup(lab, b, ls))
        cols.arrange(RIGHT, buff=0.5, aligned_edge=DOWN).move_to([0, 0.2, 0])
        for k, cue in enumerate(("7", "wanders", "salad")):
            self.at(cue)
            self.play(FadeIn(cols[k]), run_time=0.4)
        s = Text("“…Skooter bipartisan Zionist Liberty Innovators > CathP wrestler…”", font_size=20, color=GOLD).move_to([0, -2.6, 0])
        self.play(FadeIn(s), run_time=0.3)
        self.end_section()

    # ------------------------------------------------------------------ 6. truncation
    def truncation(self):
        self.section(6)
        self.clear_stage()
        h = Text("cut the long tail, then sample", font_size=28).to_edge(UP, buff=0.5)
        self.at("tail")
        self.play(FadeIn(h), run_time=0.3)
        c = D["confident"]
        rows = [("top-k 40", 40, 40, "top"), ("top-p 0.9", D["topp90"], c["topp90"], "covers"),
                ("min-p 0.1", D["minp10"], c["minp10"], "least")]
        sub = Text(f"top token: {D['p'][0]:.3f} (“saw”) vs {c['p']:.3f} (“hat”)", font_size=20, color=GREY_B)
        xs, y0 = (-3.6, 0.2, 3.8), 1.2
        hdr = VGroup(Text("after “…door and”", font_size=22, color=GREY_B).move_to([xs[1], y0, 0]),
                     Text("after “…a black”", font_size=22, color=GREY_B).move_to([xs[2], y0, 0]))
        lines = VGroup(*[VGroup(Text(n, font=MONO, font_size=24).move_to([xs[0], y0 - 0.7 * (k + 1), 0]),
                                Text(str(a), font=MONO, font_size=24).move_to([xs[1], y0 - 0.7 * (k + 1), 0]),
                                Text(str(b), font=MONO, font_size=24).move_to([xs[2], y0 - 0.7 * (k + 1), 0]))
                         for k, (n, a, b, _) in enumerate(rows)])
        sub.move_to([0, -1.9, 0])
        lab = Text("tokens kept", font_size=24, color=YELLOW).move_to([0, 2.2, 0])
        self.at("cut")
        self.play(FadeIn(hdr), FadeIn(lab), FadeIn(sub), run_time=0.4)
        for k, (_, _, _, cue) in enumerate(rows):
            self.at(cue)
            self.play(FadeIn(lines[k]), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 7. results
    def results(self):
        self.section(7)
        self.clear_stage()
        ax = Axes(x_range=[0, 80, 20], y_range=[0, 10, 2], x_length=9, y_length=4.8, tips=False,
                  axis_config={"color": GREY_B}).move_to([0.3, -0.1, 0])
        xl = Text("repetition (%)", font_size=20).next_to(ax.x_axis, DOWN, buff=0.3)
        yl = Text("judge surprise", font_size=20).rotate(PI / 2).next_to(ax.y_axis, LEFT, buff=0.3)
        ticks = VGroup(*[Text(str(v), font_size=16).next_to(ax.c2p(v, 0), DOWN, buff=0.1) for v in (0, 20, 40, 60, 80)],
                       *[Text(str(v), font_size=16).next_to(ax.c2p(0, v), LEFT, buff=0.1) for v in (0, 2, 4, 6, 8, 10)])
        self.at("three")
        self.play(Create(ax), FadeIn(xl), FadeIn(yl), FadeIn(ticks), run_time=0.6)
        cols = {"greedy": RED_C, "beam, 4": RED_C, "T 0.7": GOLD, "T 1.0": GOLD, "T 1.5": GOLD, "top-k 40": GREEN_C,
                "top-p 0.9": GREEN_C, "min-p 0.1": GREEN_B}
        offs = {"T 1.0": RIGHT, "top-p 0.9": RIGHT, "top-k 40": RIGHT, "T 0.7": DOWN, "min-p 0.1": RIGHT, "T 1.5": RIGHT,
                "greedy": UL, "beam, 4": UR}
        dots = VGroup()
        for n, r, l in RESULTS:
            d = Dot(ax.c2p(r, l), radius=0.09, color=cols[n])
            t = Text(n, font_size=18, color=cols[n]).next_to(d, offs[n], buff=0.1)
            dots.add(VGroup(d, t))
        self.play(FadeIn(dots, lag_ratio=0.1), run_time=1.0)
        self.at("lowest")
        self.play(Indicate(dots[7], color=YELLOW, scale_factor=1.5), run_time=0.8)
        a = Text("min-p: strict when confident, loose when not", font_size=22, color=YELLOW).move_to([2.2, 2.6, 0])
        self.at("adapts")
        self.play(FadeIn(a), run_time=0.4)
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
        self.at("edit")
        self.play(Create(hl), run_time=0.3)
        self.at("random")
        self.play(highlight(hl, code, 3), run_time=0.3)
        self.at("drop")
        self.play(highlight(hl, code, 2), run_time=0.3)
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
