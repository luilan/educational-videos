"""LLMs in Practice, episode 3 — Sampling: Temperature, Top-p, and Why Answers Vary.

Render from the repo root:  ./render.sh llms-in-practice p03
Every number and continuation on screen comes from code/p03_sampling/sampling.py
(Qwen2.5-0.5B-Instruct, prompt "The cat sat on the").
"""
import math

from manim import *

from common import MONO, TOKEN_COLOR, code_panel, finish, highlight, next_up_card, token
from intro import play_token_intro
from p03_script import NEXT, TAGLINE, TITLE
from voiced_scene import VoicedScene

SERIES_LABEL = "LLMs IN PRACTICE  ·  EPISODE 3"
PROMPT = "The cat sat on the"
WORDS = ["couch", "bed", "window", "fence", "sofa", "windows", "top", "chair", "mat"]
PROBS = {0.5: [0.2924, 0.1442, 0.1221, 0.1043, 0.1043, 0.0342, 0.0333, 0.0285, 0.0178],
         1.0: [0.0808, 0.0567, 0.0522, 0.0483, 0.0483, 0.0276, 0.0273, 0.0252, 0.0199],
         2.0: [0.0061, 0.0051, 0.0049, 0.0047, 0.0047, 0.0036, 0.0035, 0.0034, 0.0030]}
CUM = [(1, 0.0808), (2, 0.1375), (5, 0.2862), (10, 0.4042), (19, 0.5066), (50, 0.6584), (100, 0.7603),
       (200, 0.8438), (378, 0.9001), (1000, 0.9571), (2000, 0.9787), (5000, 0.9929), (10000, 0.9973),
       (50000, 0.9999), (151936, 1.0)]
GREEDY = " couch, and the dog was sitting next"
SAMPLES = [" bed of how many mice?", " fence at night. It felt very calm", " train. Which of the following options"]
VOCAB = "151,936"
CODE = """probs = softmax(logits / temperature)

order = argsort(probs, descending=True)
mass  = cumsum(probs[order])
keep  = order[mass - probs[order] < top_p]      # nucleus

token = random.choices(keep, weights=probs[keep])[0]"""
SCALE = 14.0     # bar height per unit of probability


def prob_bars(probs, scale=SCALE, width=0.9):
    bars = VGroup()
    for p in probs:
        bars.add(Rectangle(width=width, height=max(p * scale, 0.02), stroke_width=0, fill_color=TOKEN_COLOR,
                           fill_opacity=0.85))
    bars.arrange(RIGHT, buff=0.35, aligned_edge=DOWN)
    return bars


def value_labels(bars, probs):
    return VGroup(*[Text(f"{p * 100:.1f}%", font_size=18, color=GREY_A).next_to(b, UP, buff=0.1)
                    for b, p in zip(bars, probs)])


def line_box(text, color=GREY_D, font_size=22):
    label = Text(text, font=MONO, font_size=font_size)
    box = RoundedRectangle(corner_radius=0.12, width=label.width + 0.5, height=label.height + 0.35, stroke_width=0,
                           fill_color=color, fill_opacity=0.9)
    return VGroup(box, label.move_to(box))


class SamplingVideo(VoicedScene):
    VIDEO = "p03"

    def construct(self):
        play_token_intro(self, TITLE, 3, TAGLINE, label=SERIES_LABEL)
        self.hook()          # 1
        self.probabilities() # 2
        self.greedy()        # 3
        self.sampling()      # 4
        self.temperature()   # 5
        self.top_p()         # 6
        self.code()          # 7
        self.settings()      # 8
        self.outro()         # 9

    def clear_stage(self, *keep, run_time=0.6):
        self.play(*[FadeOut(m) for m in self.mobjects if m not in keep], run_time=run_time)

    def chart(self, t=1.0, y=-2.4):
        """Bar chart of the 9 top tokens at temperature t, with word labels under the bars."""
        bars = prob_bars(PROBS[t]).move_to([0, y, 0], aligned_edge=DOWN)
        words = VGroup(*[Text(w, font_size=22).next_to(b, DOWN, buff=0.18) for w, b in zip(WORDS, bars)])
        base = Line(bars.get_corner(DL) + 0.3 * LEFT, bars.get_corner(DR) + 0.3 * RIGHT, color=GREY_C)
        return bars, words, base

    # ------------------------------------------------------------------ 1. hook
    def hook(self):
        self.section(1)
        prompt = line_box(PROMPT, BLUE_E, 26).move_to([0, 2.2, 0])
        a = line_box(PROMPT + GREEDY, GREY_D).move_to([0, 0.5, 0])
        b = line_box(PROMPT + SAMPLES[1], GREY_D).move_to([0, -0.6, 0])
        self.at("twice")
        self.play(FadeIn(prompt, shift=0.2 * DOWN), run_time=0.5)
        self.at("different")
        self.play(LaggedStart(FadeIn(a, shift=0.2 * UP), FadeIn(b, shift=0.2 * UP), lag_ratio=0.4), run_time=0.9)
        note = Text("same model, same prompt", font_size=24, color=GREY_B).next_to(b, DOWN, buff=0.4)
        self.at("bug")
        self.play(FadeIn(note), run_time=0.4)
        name = Text("sampling", font_size=48, color=YELLOW).move_to([0, -2.6, 0])
        self.at("sampling")
        self.play(FadeIn(name, scale=1.2), run_time=0.5)
        self.end_section()

    # ------------------------------------------------------------------ 2. real probabilities
    def probabilities(self):
        self.section(2)
        self.clear_stage()
        vocab = Text(f"{VOCAB} tokens in the vocabulary", font_size=30).move_to([0, 2.9, 0])
        self.at("vocabulary")
        self.play(FadeIn(vocab, shift=0.2 * DOWN), run_time=0.5)
        prompt = Text(f"“{PROMPT} …”", font_size=30, color=BLUE_B).move_to([0, 2.1, 0])
        self.bars, self.words, self.base = self.chart()
        self.labels = value_labels(self.bars, PROBS[1.0])
        self.at("softmax")
        self.play(Create(self.base), run_time=0.4)
        self.at("real")
        self.play(FadeIn(prompt), FadeOut(vocab), run_time=0.5)
        self.prompt = prompt
        cues = ["couch", "bed", "window"]
        for i, cue in enumerate(cues):
            self.at(cue)
            self.play(GrowFromEdge(self.bars[i], DOWN), FadeIn(self.words[i]), FadeIn(self.labels[i]), run_time=0.4)
        self.play(*[GrowFromEdge(self.bars[i], DOWN) for i in range(3, 8)],
                  *[FadeIn(self.words[i]) for i in range(3, 8)], *[FadeIn(self.labels[i]) for i in range(3, 8)],
                  run_time=0.6)
        self.at("mat")
        self.play(GrowFromEdge(self.bars[8], DOWN), FadeIn(self.words[8]), FadeIn(self.labels[8]), run_time=0.5)
        ninth = Text("9th place", font_size=24, color=YELLOW).next_to(self.labels[8], UP, buff=0.2)
        self.at("ninth")
        self.play(self.bars[8].animate.set_fill(YELLOW), FadeIn(ninth, shift=0.2 * DOWN), run_time=0.5)
        unsure = Text("real models are rarely this sure of one word", font_size=24, color=GREY_B)
        unsure.next_to(prompt, DOWN, buff=0.3)
        self.at("rarely")
        self.play(FadeIn(unsure), run_time=0.5)
        self.extra = VGroup(ninth, unsure)
        self.end_section()

    # ------------------------------------------------------------------ 3. greedy
    def greedy(self):
        self.section(3)
        self.play(FadeOut(self.extra), self.bars[8].animate.set_fill(TOKEN_COLOR), run_time=0.4)
        chart = VGroup(self.bars, self.words, self.base, self.labels)
        self.at("top")
        self.play(chart.animate.scale(0.8).move_to([0, -2.3, 0]), self.prompt.animate.move_to([0, 3.1, 0]),
                  run_time=0.6)
        pick = SurroundingRectangle(self.bars[0], buff=0.05, color=YELLOW)
        name = Text("greedy: always the top token", font_size=26, color=YELLOW).move_to([0, 2.45, 0])
        self.at("greedy")
        self.play(Create(pick), FadeIn(name), run_time=0.5)
        runs = VGroup(*[Text(PROMPT + GREEDY, font=MONO, font_size=20) for _ in range(3)])
        runs.arrange(DOWN, buff=0.38).move_to([0, 1.0, 0])
        tags = VGroup(*[Text(f"run {i}", font_size=18, color=GREY_B).next_to(r, UP, buff=0.08).align_to(r, LEFT)
                        for i, r in enumerate(runs, 1)])
        self.at("same")
        self.play(LaggedStart(*[FadeIn(VGroup(r, t), shift=0.2 * LEFT) for r, t in zip(runs, tags)],
                              lag_ratio=0.3), run_time=1.0)
        dull = Text("identical, often dull, can loop", font_size=24, color=GREY_B).next_to(runs, RIGHT, buff=0.4)
        dull.set_x(0).next_to(runs, DOWN, buff=0.2)
        self.at("dull")
        self.play(FadeIn(dull), run_time=0.4)
        self.greedy_parts = VGroup(pick, name, runs, tags, dull)
        self.end_section()

    # ------------------------------------------------------------------ 4. sampling
    def sampling(self):
        self.section(4)
        self.play(FadeOut(self.greedy_parts), run_time=0.4)
        name = Text("sampling: draw by probability", font_size=26, color=GREEN_B).move_to([0, 2.45, 0])
        self.at("random")
        self.play(FadeIn(name), run_time=0.4)
        self.at("couch")
        self.play(Indicate(self.bars[0], color=GREEN), run_time=0.5)
        self.at("bed")
        self.play(Indicate(self.bars[1], color=GREEN), run_time=0.5)
        runs = VGroup(*[Text(PROMPT + s, font=MONO, font_size=20) for s in SAMPLES])
        runs.arrange(DOWN, buff=0.38, aligned_edge=LEFT).move_to([0, 1.0, 0])
        for r in runs:
            r[len(PROMPT.replace(" ", "")):].set_color(GREEN_B)
        tags = VGroup(*[Text(f"run {i}", font_size=18, color=GREY_B).next_to(r, UP, buff=0.08).align_to(r, LEFT)
                        for i, r in enumerate(runs, 1)])
        self.at("three")
        self.play(LaggedStart(*[FadeIn(VGroup(r, t), shift=0.2 * LEFT) for r, t in zip(runs, tags)],
                              lag_ratio=0.35), run_time=1.2)
        self.sample_parts = VGroup(name, runs, tags)
        self.end_section()

    # ------------------------------------------------------------------ 5. temperature
    def temperature(self):
        self.section(5)
        self.clear_stage()
        self.bars, self.words, self.base = self.chart()
        self.labels = value_labels(self.bars, PROBS[1.0])
        t_label = Text("temperature = 1", font_size=34).move_to([0, 2.9, 0])
        formula = Text("probabilities = softmax(scores / temperature)", font=MONO, font_size=22, color=GREY_A)
        formula.next_to(t_label, DOWN, buff=0.25)
        self.at("reshapes")
        self.play(FadeIn(VGroup(self.bars, self.words, self.base, self.labels)), FadeIn(t_label), run_time=0.6)
        self.at("divide")
        self.play(FadeIn(formula, shift=0.2 * DOWN), run_time=0.5)
        for t, cue_word, value_cue in [(0.5, "5", "29"), (2.0, "two", "1")]:
            new_bars = prob_bars(PROBS[t]).move_to(self.bars, aligned_edge=DOWN)
            new_bars.align_to(self.bars, DOWN)
            for nb, ob in zip(new_bars, self.bars):
                nb.set_x(ob.get_x())
            new_labels = value_labels(new_bars, PROBS[t])
            new_t = Text(f"temperature = {t:g}", font_size=34).move_to(t_label)
            self.at(cue_word)
            self.play(Transform(self.bars, new_bars), Transform(self.labels, new_labels),
                      Transform(t_label, new_t), run_time=0.9)
            self.at(value_cue)
            self.play(Indicate(self.labels[0], color=YELLOW, scale_factor=1.4), run_time=0.5)
        scale = VGroup(Text("low: focused, predictable", font_size=24, color=BLUE_B),
                       Text("high: creative → chaotic", font_size=24, color=RED_B)).arrange(RIGHT, buff=1.0)
        scale.move_to([0, 0.9, 0])
        self.at("low")
        self.play(FadeIn(scale[0], shift=0.2 * UP), run_time=0.4)
        self.at("creative")
        self.play(FadeIn(scale[1], shift=0.2 * UP), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 6. top-p
    def top_p(self):
        self.section(6)
        self.clear_stage()
        ax = Axes(x_range=[0, 5.3, 1], y_range=[0, 1, 0.5], x_length=9.5, y_length=4.2, tips=False,
                  axis_config={"color": GREY_B})
        ax.move_to([0.3, -0.3, 0])
        ynums = VGroup(*[Text(s, font_size=20, color=GREY_B).next_to(ax.c2p(0, v), LEFT, buff=0.15)
                         for s, v in [("0", 0), ("0.5", 0.5), ("1", 1)]])
        xnums = VGroup(*[Text(s, font_size=20, color=GREY_B).next_to(ax.c2p(math.log10(v), 0), DOWN, buff=0.15)
                         for s, v in [("1", 1), ("10", 10), ("100", 100), ("1,000", 1e3), ("10,000", 1e4),
                                      ("151,936", 151936)]])
        xl = Text("top tokens kept (log scale)", font_size=22, color=GREY_B).next_to(xnums, DOWN, buff=0.2)
        yl = Text("probability covered", font_size=22, color=GREY_B).rotate(PI / 2).next_to(ax, LEFT, buff=0.55)
        pts = [ax.c2p(math.log10(r), c) for r, c in CUM]
        curve = VMobject(color=YELLOW, stroke_width=5).set_points_smoothly(pts)
        self.at("tail")
        self.play(Create(ax), FadeIn(xnums), FadeIn(ynums), FadeIn(xl), FadeIn(yl), run_time=0.8)
        self.at("add")
        self.play(Create(curve), run_time=1.6)
        tail = Text("long tail: thousands of tiny probabilities", font_size=22, color=RED_B)
        tail.next_to(ax.c2p(4.2, 1.0), UP, buff=0.2)
        self.at("nonsense")
        self.play(FadeIn(tail), run_time=0.4)
        title = Text("top-p: keep the top tokens until they add up to p", font_size=28).to_edge(UP, buff=0.35)
        self.at("smallest")
        self.play(FadeIn(title, shift=0.2 * DOWN), run_time=0.5)
        for p, rank, cue in [(0.9, 378, "378"), (0.5, 19, "19")]:
            x = math.log10(rank)
            h = DashedLine(ax.c2p(0, p), ax.c2p(x, p), color=GREEN_B)
            v = DashedLine(ax.c2p(x, p), ax.c2p(x, 0), color=GREEN_B)
            lab = Text(f"p = {p}: {rank} tokens", font_size=24, color=GREEN_B).next_to(ax.c2p(x, p), RIGHT, buff=0.2)
            if p == 0.5:
                lab.shift(0.25 * DOWN)
            self.at(cue)
            self.play(Create(h), Create(v), FadeIn(lab), run_time=0.6)
        cut = Rectangle(width=ax.c2p(5.18, 0)[0] - ax.c2p(math.log10(378), 0)[0], height=4.2, stroke_width=0,
                        fill_color=RED, fill_opacity=0.18)
        cut.move_to(ax.c2p(math.log10(378), 0.5), aligned_edge=LEFT)
        cut_l = Text("cut (p = 0.9)", font_size=22, color=RED_B).move_to(cut).shift(0.6 * DOWN)
        self.at("cut")
        self.play(FadeIn(cut), FadeIn(cut_l), run_time=0.5)
        self.end_section()

    # ------------------------------------------------------------------ 7. code
    def code(self):
        self.section(7)
        self.clear_stage()
        code, hl = code_panel(CODE, font_size=22)
        code.move_to([0, 0.3, 0])
        self.at("code")
        self.play(FadeIn(code, shift=0.2 * UP), run_time=0.5)
        hl.match_y(code.line_numbers[0])
        self.at("divide")
        self.play(Create(hl), run_time=0.4)
        self.at("sort")
        self.play(highlight(hl, code, 2), run_time=0.4)
        self.at("add")
        self.play(highlight(hl, code, 3), run_time=0.4)
        self.at("reach")
        self.play(highlight(hl, code, 4), run_time=0.4)
        self.at("draw")
        self.play(highlight(hl, code, 6), run_time=0.4)
        note = Text("full version: code/p03_sampling", font_size=22, color=GREY_B).next_to(code, DOWN, buff=0.35)
        self.play(FadeIn(note), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 8. settings
    def settings(self):
        self.section(8)
        self.clear_stage()

        def card(title, lines, color):
            head = Text(title, font_size=28, color=color)
            body = VGroup(*[Text(l, font_size=24) for l in lines]).arrange(DOWN, aligned_edge=LEFT, buff=0.2)
            content = VGroup(head, body).arrange(DOWN, aligned_edge=LEFT, buff=0.35)
            box = RoundedRectangle(corner_radius=0.2, width=5.6, height=3.4, stroke_color=color, fill_color=color,
                                   fill_opacity=0.1)
            content.move_to(box).align_to(box, LEFT).shift(0.35 * RIGHT)
            return VGroup(box, content)
        precise = card("facts · code · extraction", ["low temperature (0 – 0.3)", "close to greedy"], BLUE_B)
        creative = card("brainstorming · stories", ["higher temperature (0.8 – 1.2)", "plus top-p ≈ 0.9"], GOLD)
        VGroup(precise, creative).arrange(RIGHT, buff=0.6).move_to([0, 0.5, 0])
        self.at("facts")
        self.play(FadeIn(precise, shift=0.2 * UP), run_time=0.5)
        self.at("brainstorming")
        self.play(FadeIn(creative, shift=0.2 * UP), run_time=0.5)
        seed = Text("repeatable tests: fix the random seed", font_size=26, color=GREEN_B).to_edge(DOWN, buff=0.9)
        typical = Text("typical starting points", font_size=20, color=GREY_B).next_to(seed, DOWN, buff=0.25)
        self.at("seed")
        self.play(FadeIn(seed, shift=0.2 * UP), FadeIn(typical), run_time=0.5)
        self.end_section()

    # ------------------------------------------------------------------ 9. outro
    def outro(self):
        self.section(9)
        self.clear_stage(run_time=0.4)
        card = next_up_card(NEXT)
        self.at("embeddings")
        self.play(FadeIn(card, shift=0.2 * UP), run_time=0.6)
        self.end_section()
        finish(self, hold=1.5)
