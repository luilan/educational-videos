"""Shared visuals for How Generative Models Work scenes (pictures from the episode code, scatter planes, powers)."""
import numpy as np
from PIL import Image
from manim import (Axes, DOWN, Dot, GREY_C, GREY_D, ImageMobject, RESAMPLING_ALGORITHMS, Square,
                   SurroundingRectangle, Text, UR, VGroup, WHITE)

from common import MONO

TILE, PAD = 128, 2          # shapes.save_grid: 32×32 pictures scaled 4×, 2-pixel gaps


def pic(path, height):
    """A whole PNG, with sharp pixels."""
    m = ImageMobject(path)
    m.set_resampling_algorithm(RESAMPLING_ALGORITHMS["nearest"])
    return m.scale_to_fit_height(height)


def tile(path, i, height, row=0, size=TILE, pad=PAD):
    """Picture i (column) of row `row` of a grid PNG written by shapes.save_grid."""
    arr = np.asarray(Image.open(path).convert("RGB"))
    x, y = i * (size + pad), row * (size + pad)
    m = ImageMobject(arr[y:y + size, x:x + size])
    m.set_resampling_algorithm(RESAMPLING_ALGORITHMS["nearest"])
    return m.scale_to_fit_height(height)


def frame(m, color=GREY_C):
    return SurroundingRectangle(m, buff=0, color=color, stroke_width=2)


def power(base, exp, font_size=24, color=WHITE, font=MONO):
    b = Text(base, font=font, font_size=font_size, color=color)
    e = Text(exp, font=font, font_size=font_size * 0.62, color=color).next_to(b, UR, buff=0.02).shift(DOWN * font_size * 0.006)
    return VGroup(b, e)


def plane(center, size=4.4, lo=-1.15, hi=1.15):
    ax = Axes(x_range=[lo, hi, 1], y_range=[lo, hi, 1], x_length=size, y_length=size,
              tips=False, axis_config={"stroke_opacity": 0.0})
    box = Square(side_length=size, color=GREY_D, stroke_width=2)
    return VGroup(box, ax).move_to(center), ax


def dots(ax, pts, color, r=0.022):
    return VGroup(*[Dot(ax.c2p(x, y), radius=r, color=color) for x, y in pts])
