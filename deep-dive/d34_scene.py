"""How LLMs Work: Deep Dive, episode 34 — Speculative Decoding.

Render from the repo root:  ./render.sh deep-dive d34
Every number on screen comes from code/d34_speculative/speculative.py (draft Qwen2.5-0.5B, target Qwen2.5-1.5B, greedy,
128 new tokens, CPU float32). The guesses in section 3 are a real round from the prose prompt with k = 4.
"""
from manim import *

from common import MODEL_COLOR, MONO, TOKEN_COLOR, code_panel, finish, highlight, next_up_card, token
from intro import play_token_intro
from d34_script import NEXT, TAGLINE, TITLE
from voiced_scene import VoicedScene

SERIES_LABEL = "HOW LLMs WORK: DEEP DIVE  ·  EPISODE 34"
DRAFT = ["ates", " from", " the", " land"]
TARGET = ["ates", " from", " the", " Earth", " and"]
CODE = """preds = target(ids + guesses).argmax(-1)   # one pass
n = 0
while n < k and guesses[n] == preds[n]:
    n += 1                                  # count matches
ids += guesses[:n] + [preds[n]]             # + its own token"""


def chip(word, color=TOKEN_COLOR):
    t = Text(word.replace(" ", "·"), font=MONO, font_size=24)
    box = RoundedRectangle(width=max(t.width + 0.3, 0.9), height=0.6, corner_radius=0.1, color=color, fill_opacity=0.25)
    return VGroup(box, t.move_to(box))


def hbars(rows, unit, x0=-1.0, y0=1.2, dy=0.8, fmt="{}", min_x=None):
    g = VGroup()
    for k, (name, v, col) in enumerate(rows):
        y = y0 - dy * k
        lab = Text(name, font_size=22).move_to([x0 - 0.3, y, 0], aligned_edge=RIGHT)
        b = Rectangle(width=v * unit, height=0.5, stroke_width=0, fill_color=col, fill_opacity=0.85).move_to([x0, y, 0], aligned_edge=LEFT)
        num = Text(fmt.format(v), font=MONO, font_size=20).next_to(b, RIGHT, buff=0.15)
        if min_x is not None and num.get_left()[0] < min_x:
            num.align_to([min_x, 0, 0], LEFT)
        g.add(VGroup(lab, b, num))
    return g


class SpeculativeVideo(VoicedScene):
    VIDEO = "d34"

    def construct(self):
        play_token_intro(self, TITLE, 34, TAGLINE, label=SERIES_LABEL)
        self.hook()          # 1
        self.cheap()         # 2
        self.algorithm()     # 3
        self.code_results()  # 4
        self.prose()         # 5
        self.tradeoff()      # 6
        self.sampling()      # 7
        self.code()          # 8
        self.outro()         # 9

    def clear_stage(self, *keep, run_time=0.6):
        self.play(*[FadeOut(m) for m in self.mobjects if m not in keep], run_time=run_time)

    # ------------------------------------------------------------------ 1. hook
    def hook(self):
        self.section(1)
        big = RoundedRectangle(width=3.0, height=2.0, corner_radius=0.15, color=MODEL_COLOR, fill_opacity=0.35).move_to([2.6, 0.2, 0])
        bl = Text("big model\n(target)", font_size=24, line_spacing=0.8).move_to(big)
        small = RoundedRectangle(width=2.2, height=1.1, corner_radius=0.12, color=GREEN_C, fill_opacity=0.35).move_to([-3.6, 0.2, 0])
        sl = Text("small model\n(draft)", font_size=20, line_spacing=0.8).move_to(small)
        self.at("big")
        self.play(FadeIn(big), FadeIn(bl), run_time=0.5)
        self.at("small")
        self.play(FadeIn(small), FadeIn(sl), run_time=0.5)
        gs = VGroup(*[chip(w, GREEN_C).scale(0.7) for w in ("guess", "guess", "guess")]).arrange(RIGHT, buff=0.1)
        gs.move_to([-0.4, 0.2, 0])
        self.at("guess")
        self.play(LaggedStart(*[FadeIn(g, shift=0.4 * RIGHT) for g in gs], lag_ratio=0.3), run_time=0.8)
        ck = Text("check", font_size=24, color=YELLOW).next_to(big, UP)
        self.at("check")
        self.play(FadeIn(ck), run_time=0.3)
        same = Text("exactly the same output, faster", font_size=26, color=YELLOW).move_to([0, -2.4, 0])
        self.at("exactly")
        self.play(FadeIn(same), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 2. checking is cheap
    def cheap(self):
        self.section(2)
        self.clear_stage()
        head = Text("Qwen2.5-1.5B, one pass after 100 tokens (this CPU)", font_size=26).to_edge(UP, buff=0.6)
        self.at("target")
        self.play(FadeIn(head), run_time=0.4)
        g = hbars([("1 new token", 332, RED_C), ("5 tokens", 199, GREEN_C), ("9 tokens", 242, GREEN_C)], unit=0.016, fmt="{} ms")
        for k, cue in enumerate(("330", "200", "240")):
            self.at(cue)
            self.play(FadeIn(g[k]), run_time=0.4)
        t = Text("checking several tokens costs no more than writing one", font_size=24, color=YELLOW).move_to([0, -2.2, 0])
        self.at("handful")
        self.play(FadeIn(t), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 3. algorithm
    def algorithm(self):
        self.section(3)
        self.clear_stage()
        ctx = Text("…It begins when water evapor", font_size=26, color=GREY_A).move_to([0, 2.7, 0])
        self.play(FadeIn(ctx), run_time=0.3)
        d = VGroup(*[chip(w, GREEN_C) for w in DRAFT]).arrange(RIGHT, buff=0.15).move_to([0.4, 1.3, 0])
        dl = Text("draft guesses", font_size=22, color=GREEN_B).next_to(d, LEFT, buff=0.4)
        self.at("guesses")
        self.play(FadeIn(dl), run_time=0.2)
        self.play(LaggedStart(*[FadeIn(c, shift=0.2 * RIGHT) for c in d], lag_ratio=0.35), run_time=1.2)
        tgt = VGroup(*[chip(w, MODEL_COLOR) for w in TARGET]).arrange(RIGHT, buff=0.15)
        tgt.move_to([0.4, -0.2, 0]).align_to(d, LEFT)
        tl = Text("target would pick", font_size=22, color=BLUE_B).next_to(tgt, LEFT, buff=0.4)
        self.at("single")
        self.play(FadeIn(tl), FadeIn(tgt, lag_ratio=0.1), run_time=0.7)
        marks = VGroup(*[Text("✓" if k < 3 else "✗", font_size=30, color=GREEN_B if k < 3 else RED_B).next_to(d[k], UP, buff=0.1)
                         for k in range(4)])
        self.at("disagreement")
        self.play(FadeIn(marks, lag_ratio=0.2), d[3].animate.set_opacity(0.35), run_time=0.6)
        res = VGroup(*[chip(w, GREEN_C) for w in DRAFT[:3]], chip(TARGET[3], MODEL_COLOR)).arrange(RIGHT, buff=0.15)
        res.move_to([0.4, -1.7, 0]).align_to(d, LEFT)
        rl = Text("kept: 3 + 1", font_size=22, color=YELLOW).next_to(res, LEFT, buff=0.4)
        self.at("own")
        self.play(FadeIn(res, lag_ratio=0.1), FadeIn(rl), run_time=0.6)
        n = Text("every pass: at least 1 token, at most k + 1", font_size=24, color=YELLOW).move_to([0, -2.9, 0])
        self.at("least")
        self.play(FadeIn(n), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 4. code results
    def code_results(self):
        self.section(4)
        self.clear_stage()
        head = Text("Python code, 128 new tokens", font_size=28).to_edge(UP, buff=0.6)
        self.at("python")
        self.play(FadeIn(head), run_time=0.3)
        g = hbars([("plain decoding", 44.0, RED_C), ("4 guesses", 19.0, GREEN_C)], unit=0.13, y0=1.0, dy=1.0, fmt="{} s")
        self.at("44")
        self.play(FadeIn(g[0]), run_time=0.4)
        s = Text("96% of guesses accepted · 4.7 tokens per pass", font_size=24, color=GREEN_B).move_to([0, -1.0, 0])
        self.at("96")
        self.play(FadeIn(s), run_time=0.4)
        self.at("19")
        self.play(FadeIn(g[1]), run_time=0.4)
        x = Text("2.3x faster · identical output, token for token", font_size=26, color=YELLOW).move_to([0, -2.2, 0])
        self.at("identical")
        self.play(FadeIn(x), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 5. prose
    def prose(self):
        self.section(5)
        self.clear_stage()
        head = Text("prose: speed-up vs plain decoding", font_size=28).to_edge(UP, buff=0.6)
        self.at("pros")
        self.play(FadeIn(head), run_time=0.3)
        rows = [("k=2 · 63% accepted", 1.46, GREEN_C), ("k=4 · 54%", 1.37, GREEN_C), ("k=6 · 37%", 0.93, RED_C),
                ("k=8 · 32%", 0.91, RED_C)]
        g = hbars(rows, unit=3.0, x0=-0.6, y0=1.4, dy=0.8, fmt="{}x", min_x=2.55)
        line = DashedLine([-0.6 + 3.0, 1.9, 0], [-0.6 + 3.0, -1.5, 0], color=GREY_B)
        ll = Text("1x", font_size=18, color=GREY_B).next_to(line, UP, buff=0.05)
        self.at("63")
        self.play(FadeIn(g[0]), Create(line), FadeIn(ll), run_time=0.4)
        self.play(FadeIn(g[1]), run_time=0.3)
        self.at("37")
        self.play(FadeIn(g[2]), FadeIn(g[3]), run_time=0.4)
        t = Text("slower than plain decoding", font_size=24, color=RED_B).move_to([0, -2.4, 0])
        self.at("slower")
        self.play(FadeIn(t), run_time=0.3)
        self.end_section()

    # ------------------------------------------------------------------ 6. trade-off
    def tradeoff(self):
        self.section(6)
        self.clear_stage()
        head = Text("one round with 4 guesses: where the time goes", font_size=26).to_edge(UP, buff=0.6)
        self.play(FadeIn(head), run_time=0.3)
        unit = 0.012
        segs = VGroup(*[Rectangle(width=131 * unit, height=0.6, stroke_width=1, color=BLACK, fill_color=GREEN_C, fill_opacity=0.85)
                        for _ in range(4)], Rectangle(width=199 * unit, height=0.6, stroke_width=1, color=BLACK,
                                                      fill_color=MODEL_COLOR, fill_opacity=0.85)).arrange(RIGHT, buff=0)
        segs.move_to([0, 0.6, 0])
        labs = VGroup(*[Text("draft 131 ms", font_size=16).move_to(s) for s in segs[:4]], Text("check", font_size=18).move_to(segs[4]))
        self.at("130")
        self.play(FadeIn(segs[:4], lag_ratio=0.2), FadeIn(labs[:4]), run_time=0.6)
        p = Text("each guess: 40% of a 332 ms target step", font_size=24, color=GREEN_B).move_to([0, -0.6, 0])
        self.at("40")
        self.play(FadeIn(segs[4]), FadeIn(labs[4]), FadeIn(p), run_time=0.4)
        w = Text("rejected guesses are wasted time", font_size=24, color=RED_B).move_to([0, -1.5, 0])
        self.at("rejected")
        self.play(FadeIn(w), run_time=0.3)
        b = Text("best k: depends on how predictable the text is, and how cheap the draft is", font_size=22, color=YELLOW)
        b.move_to([0, -2.4, 0])
        self.at("predictable")
        self.play(FadeIn(b), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 7. sampling
    def sampling(self):
        self.section(7)
        self.clear_stage()
        rule = Text("accept the guess x with probability min(1, p(x) / q(x));\notherwise sample from max(0, p − q)",
                    font_size=24, line_spacing=0.9).to_edge(UP, buff=0.5)
        self.at("probability")
        self.play(FadeIn(rule), run_time=0.5)
        p = [0.50, 0.30, 0.15, 0.05]
        q = [0.25, 0.50, 0.20, 0.05]
        r = [0.501, 0.299, 0.150, 0.050]
        groups = VGroup()
        for k in range(4):
            bars = VGroup(*[Rectangle(width=0.4, height=v * 5, stroke_width=0, fill_color=c, fill_opacity=0.85)
                            for v, c in ((p[k], MODEL_COLOR), (q[k], GREEN_C), (r[k], GOLD))]).arrange(RIGHT, buff=0.05, aligned_edge=DOWN)
            groups.add(bars)
        groups.arrange(RIGHT, buff=0.9, aligned_edge=DOWN).move_to([-2.0, -2.6, 0], aligned_edge=DOWN)
        names = VGroup(*[Text(f"token {k + 1}", font_size=18).next_to(groups[k], DOWN, buff=0.1) for k in range(4)])
        key = VGroup(Text("target p", font_size=20, color=BLUE_B), Text("draft q", font_size=20, color=GREEN_B),
                     Text("result, 1M draws", font_size=20, color=GOLD)).arrange(DOWN, aligned_edge=LEFT).move_to([4.4, -0.4, 0])
        self.play(FadeIn(VGroup(*[g[0] for g in groups])), FadeIn(VGroup(*[g[1] for g in groups])), FadeIn(names),
                  FadeIn(key[:2]), run_time=0.5)
        self.at("million")
        self.play(FadeIn(VGroup(*[g[2] for g in groups]), lag_ratio=0.2), FadeIn(key[2]), run_time=0.6)
        a = Text("75% accepted = sum of min(p, q)", font_size=22, color=YELLOW).move_to([4.2, -1.9, 0])
        self.at("75")
        self.play(FadeIn(a), run_time=0.3)
        self.end_section()

    # ------------------------------------------------------------------ 8. code
    def code(self):
        self.section(8)
        self.clear_stage()
        code, hl = code_panel(CODE, font_size=26)
        code.move_to([0, 0.3, 0])
        self.at("encode")
        self.play(FadeIn(code, shift=0.2 * UP), run_time=0.5)
        hl.match_y(code.line_numbers[0])
        self.at("once")
        self.play(Create(hl), run_time=0.3)
        self.at("count")
        self.play(highlight(hl, code, 2), run_time=0.3)
        self.at("keep")
        self.play(highlight(hl, code, 4), run_time=0.3)
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
