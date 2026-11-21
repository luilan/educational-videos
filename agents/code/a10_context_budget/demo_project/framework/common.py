"""Shared visual helpers for the series (token boxes, code panel, Next up card)."""
from manim import *

TOKEN_COLOR = BLUE_D
MODEL_COLOR = PURPLE_B
MONO = "DejaVu Sans Mono"
MAX_WIDTH = 12.8  # usable frame width


def token(word, color=TOKEN_COLOR, font_size=28):
    """A rounded token box with its label. Recolour the box via result[0], never the whole group."""
    label = Text(word, font_size=font_size)
    box = RoundedRectangle(corner_radius=0.12, width=max(label.width + 0.35, 0.6), height=0.65,
                           stroke_color=color, fill_color=color, fill_opacity=0.3)
    return VGroup(box, label.move_to(box))


def token_row(words, buff=0.12, **kwargs):
    return VGroup(*[token(w, **kwargs) for w in words]).arrange(RIGHT, buff=buff)


def code_panel(code_string, font_size=22, max_width=MAX_WIDTH):
    """Monokai code window plus a yellow line highlight drawn behind the text.

    Returns (code, hl). Move the highlight with `highlight(hl, code, i)`, where i is the
    0-based line index (blank lines count).
    """
    code = Code(code_string=code_string, language="python", formatter_style="monokai",
                background="window", paragraph_config={"font_size": font_size})
    if code.width > max_width:
        code.scale_to_fit_width(max_width)
    row_h = abs(code.line_numbers[0].get_y() - code.line_numbers[1].get_y())
    hl = Rectangle(width=code.code_lines.width + 0.25, height=row_h, color=YELLOW, stroke_width=2)
    hl.align_to(code.code_lines, LEFT).shift(0.12 * LEFT).match_y(code.line_numbers[0])
    code.code_lines.set_z_index(2)  # keep underscores visible above the highlight edge
    hl.set_z_index(1)
    return code, hl


def highlight(hl, code, i):
    return hl.animate.match_y(code.line_numbers[i])


def next_up_card(title):
    card = VGroup(Text("Next up", font_size=30, color=GREY_B),
                  Text(title, font_size=44)).arrange(DOWN, buff=0.35)
    if card.width > MAX_WIDTH:
        card.scale_to_fit_width(MAX_WIDTH)
    return card


def used_in_card(items):
    """'Where you'll see this' card. items: list of (episode_number, title) from the main series."""
    head = Text("Where you'll see this", font_size=30, color=GREY_B)
    rows = VGroup(*[
        VGroup(Text(f"Ep {n}", font_size=26, color=YELLOW), Text(title, font_size=26)).arrange(RIGHT, buff=0.3)
        for n, title in items
    ]).arrange(DOWN, aligned_edge=LEFT, buff=0.25)
    card = VGroup(head, rows).arrange(DOWN, buff=0.5)
    if card.width > MAX_WIDTH:
        card.scale_to_fit_width(MAX_WIDTH)
    if card.height > 6.6:
        card.scale_to_fit_height(6.6)
    return card


def finish(scene, *mobjects, hold=1.0):
    """Call after the last end_section(): hold the final card, fade everything, short pause."""
    scene.wait(hold)
    scene.play(*[FadeOut(m) for m in (mobjects or scene.mobjects)], run_time=0.8)
    scene.wait(0.5)
