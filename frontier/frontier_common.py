"""Shared visuals for The Frontier scenes."""
from manim import (DOWN, RIGHT, GREEN_C, GREY_B, GREY_C, LEFT, PURPLE_B, RED_C, YELLOW, RoundedRectangle, Text, VGroup)

GOOD, BAD = GREEN_C, RED_C
MODEL_C = PURPLE_B          # the model
EDIT_C = YELLOW             # edits to weights


def paper_card(title, meta, lines, width=9.6):
    """A paper as shown on screen: title, arXiv id and date, and a few lines of what it claims."""
    t = Text(title, font_size=28)
    if t.width > width - 0.6:
        t.scale_to_fit_width(width - 0.6)
    m = Text(meta, font_size=18, color=GREY_B)
    body = VGroup(*[Text(l, font_size=20, color=GREY_C) for l in lines]).arrange(DOWN, aligned_edge=LEFT, buff=0.1)
    inner = VGroup(t, m, body).arrange(DOWN, aligned_edge=LEFT, buff=0.18)
    box = RoundedRectangle(width=width, height=inner.height + 0.5, corner_radius=0.15, color=GREY_C, stroke_width=2)
    inner.move_to(box).align_to(box, LEFT).shift(0.3 * RIGHT)
    return VGroup(box, t, m, body)


def study_footer():
    return Text("Study guide (PDF) and code: github.com/luilan/educational-videos  ·  "
                "one concept at a time in the StepStone app", font_size=18, color=GREY_B).scale_to_fit_width(12.4).to_edge(DOWN, buff=0.45)
