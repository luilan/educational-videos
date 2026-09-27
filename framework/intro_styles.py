"""The five intro styles tried for episode 1 (the series uses IntroB_Tokens, see intro.py).

Render one from the repo root:  manim -ql framework/intro_styles.py IntroB_Tokens
"""
import random
import string

from manim import *

TITLE = "What is an LLM?"
SERIES = "HOW LLMs WORK  ·  EPISODE 1"
TAGLINE = "How a language model writes, one word at a time"
MONO = "DejaVu Sans Mono"
# character slices of each word in Text(TITLE) (Text drops spaces)
WORD_SLICES = [(0, 4), (4, 6), (6, 8), (8, 11), (11, 12)]


def series_label():
    return Text(SERIES, font_size=22, color=GREY_B).to_edge(UP, buff=1.6)


class IntroA_Typewriter(Scene):
    def construct(self):
        title = Text(TITLE, font=MONO, font_size=64)
        cursor = Rectangle(width=0.35, height=0.08, stroke_width=0, fill_color=YELLOW, fill_opacity=1)
        cursor.next_to(title[0], DOWN, buff=0.08).align_to(title, LEFT)
        tag = Text("> " + TAGLINE.lower(), font=MONO, font_size=24, color=GREY_B).next_to(title, DOWN, buff=0.6)
        self.add(cursor)
        for ch in title:
            self.add(ch)
            cursor.next_to(ch, RIGHT, buff=0.06).align_to(title, DOWN).shift(0.1 * DOWN)
            self.wait(0.07)
        for _ in range(2):
            self.play(cursor.animate.set_opacity(0), rate_func=there_and_back, run_time=0.35)
        self.play(FadeIn(tag, shift=0.1 * RIGHT), FadeIn(series_label()), run_time=0.6)
        self.wait(1.2)


class IntroB_Tokens(Scene):
    def construct(self):
        title = Text(TITLE, font_size=72)
        colors = [BLUE_C, TEAL_C, GREEN_C, GOLD_C, RED_C]
        tokens = VGroup()
        for (a, b), color in zip(WORD_SLICES, colors):
            word = title[a:b]
            box = SurroundingRectangle(word, buff=0.06, corner_radius=0.1, stroke_color=color,
                                       fill_color=color, fill_opacity=0.3)
            tokens.add(VGroup(box, word))
        random.seed(3)
        self.play(LaggedStart(*[FadeIn(t, shift=random.choice([UP, DOWN]) * 1.2 + random.uniform(-1, 1) * RIGHT)
                                for t in tokens], lag_ratio=0.15, run_time=1.2))
        tag = Text(TAGLINE, font_size=28, color=GREY_B).next_to(title, DOWN, buff=0.6)
        self.play(*[FadeOut(t[0], scale=1.15) for t in tokens], FadeIn(tag, shift=0.2 * UP),
                  FadeIn(series_label()), run_time=0.8)
        self.wait(1.2)


class IntroC_Decode(Scene):
    def construct(self):
        random.seed(5)
        chars = list(TITLE)
        order = [i for i, c in enumerate(chars) if c != " "]
        random.shuffle(order)
        lock_at = {i: 0.15 + 0.75 * k / len(order) for k, i in enumerate(order)}
        pool = string.ascii_letters + string.digits + "#$%&*+<>?"
        progress = ValueTracker(0)

        def frame_text():
            p = progress.get_value()
            shown = "".join(c if c == " " or p >= lock_at[i] else random.choice(pool) for i, c in enumerate(chars))
            text = Text(shown, font=MONO, font_size=64, color=GREEN_B if p < 1 else WHITE)
            return text

        title = always_redraw(frame_text)
        self.add(title)
        self.play(progress.animate.set_value(1), run_time=1.6, rate_func=linear)
        title.clear_updaters()
        final = Text(TITLE, font=MONO, font_size=64)
        self.remove(title)
        self.add(final)
        self.play(Indicate(final, color=GREEN_B, scale_factor=1.04), run_time=0.5)
        tag = Text(TAGLINE, font=MONO, font_size=22, color=GREY_B).next_to(final, DOWN, buff=0.6)
        self.play(AddTextLetterByLetter(tag, time_per_char=0.015), FadeIn(series_label()))
        self.wait(1.2)


class IntroD_Prediction(Scene):
    CANDIDATES = [
        [("What", 0.62), ("How", 0.21), ("Why", 0.09)],
        [("is", 0.71), ("are", 0.12), ("does", 0.08)],
        [("an", 0.84), ("the", 0.07), ("a", 0.05)],
        [("LLM", 0.58), ("AI", 0.25), ("API", 0.04)],
        [("?", 0.91), (".", 0.05), ("!", 0.02)],
    ]

    def construct(self):
        title = Text(TITLE, font_size=72).shift(0.5 * UP)
        for (a, b), cands in zip(WORD_SLICES, self.CANDIDATES):
            slot = title[a:b]
            rows = VGroup()
            for k, (word, p) in enumerate(cands):
                bar = Rectangle(width=1.4 * p, height=0.16, stroke_width=0,
                                fill_color=YELLOW if k == 0 else GREY_C, fill_opacity=0.9)
                label = Text(word, font_size=26, color=YELLOW if k == 0 else GREY_B)
                pct = Text(f"{p:.0%}", font_size=18, color=GREY_B)
                rows.add(VGroup(label, bar, pct).arrange(RIGHT, buff=0.15))
            rows.arrange(DOWN, buff=0.12, aligned_edge=LEFT).next_to(slot, DOWN, buff=0.5)
            self.play(FadeIn(rows, shift=0.15 * UP), run_time=0.18)
            self.play(ReplacementTransform(rows[0][0], slot), FadeOut(VGroup(rows[0][1:], rows[1:])),
                      run_time=0.3)
        tag = Text(TAGLINE, font_size=28, color=GREY_B).next_to(title, DOWN, buff=0.6)
        self.play(FadeIn(tag, shift=0.2 * UP), FadeIn(series_label().shift(0.5 * UP)), run_time=0.6)
        self.wait(1.2)


class IntroE_Classic(Scene):
    def construct(self):
        title = Text(TITLE, font_size=80, weight=BOLD).set_color_by_gradient(BLUE_B, TEAL_B)
        underline = Line(LEFT, RIGHT, color=TEAL_B, stroke_width=4).set_width(title.width * 0.9)
        underline.next_to(title, DOWN, buff=0.3)
        tag = Text(TAGLINE, font_size=28, color=GREY_B).next_to(underline, DOWN, buff=0.4)
        self.play(Write(title), run_time=1.3)
        self.play(GrowFromCenter(underline), FadeIn(series_label(), shift=0.2 * DOWN), run_time=0.5)
        self.play(FadeIn(tag, shift=0.2 * UP), run_time=0.5)
        self.wait(1.2)
