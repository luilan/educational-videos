"""Shared visuals for Reinforcement Learning for LLMs scenes."""
from manim import (DOWN, GREEN_C, GREY_B, BLUE_C, RED_C, RIGHT, YELLOW, Circle, Text, VGroup)

from common import MONO, token

GOOD, BAD = GREEN_C, RED_C          # right / wrong answers
MODEL_C = BLUE_C                    # the model / policy
REWARD_C = YELLOW                   # rewards and grades


def char_row(s, font_size=28, buff=0.06):
    """A string as one token box per character (every character is a token in this series)."""
    return VGroup(*[token(c, font_size=font_size) for c in s]).arrange(RIGHT, buff=buff)


def grade_badge(r, font_size=26):
    col = GOOD if r else BAD
    c = Circle(radius=0.3, color=col, fill_opacity=0.15, stroke_width=3)
    return VGroup(c, Text(str(r), font=MONO, font_size=font_size, color=col).move_to(c))


def study_footer():
    return Text("Study guide (PDF) and code: github.com/luilan/educational-videos  ·  "
                "one concept at a time in the StepStone app", font_size=18, color=GREY_B).scale_to_fit_width(12.4).to_edge(DOWN, buff=0.45)
