"""Series intro: the title flies in as coloured tokens, then dissolves into plain text."""
import random
import re

from manim import *

SERIES = "HOW LLMs WORK"
TOKEN_COLORS = [BLUE_C, TEAL_C, GREEN_C, GOLD_C, RED_C, PURPLE_B, MAROON_C]
MAX_TITLE_WIDTH = 12.8


def play_token_intro(scene, title_text, episode, tagline, seed=3, label=None):
    """label overrides the small series line, e.g. "HOW LLMs WORK  ·  FOUNDATIONS F3"."""
    title = Text(title_text, font_size=72)
    if title.width > MAX_TITLE_WIDTH:  # long titles would run off the 14.2-unit frame
        title.scale_to_fit_width(MAX_TITLE_WIDTH)
    tokens = VGroup()
    start = 0  # Text() has no submobjects for spaces, so count only visible characters
    for i, piece in enumerate(re.findall(r"\w+|[^\w\s]", title_text)):
        word = title[start:start + len(piece)]
        start += len(piece)
        color = TOKEN_COLORS[i % len(TOKEN_COLORS)]
        box = SurroundingRectangle(word, buff=0.06, corner_radius=0.1, stroke_color=color,
                                   fill_color=color, fill_opacity=0.3)
        tokens.add(VGroup(box, word))
    series = Text(label or f"{SERIES}  ·  EPISODE {episode}", font_size=22, color=GREY_B).to_edge(UP, buff=1.6)
    tag = Text(tagline, font_size=28, color=GREY_B).next_to(title, DOWN, buff=0.6)

    rng = random.Random(seed)
    scene.play(LaggedStart(*[FadeIn(t, shift=rng.choice([UP, DOWN]) * 1.2 + rng.uniform(-1, 1) * RIGHT)
                             for t in tokens], lag_ratio=0.3, run_time=2.4))
    scene.wait(0.5)
    scene.play(*[FadeOut(t[0], scale=1.15) for t in tokens], run_time=1.0)
    scene.play(FadeIn(series, shift=0.2 * DOWN), FadeIn(tag, shift=0.2 * UP), run_time=1.0)
    scene.wait(2.0)
    scene.play(FadeOut(VGroup(title, series, tag)), run_time=0.8)
