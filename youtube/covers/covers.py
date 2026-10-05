"""Square playlist covers (2048x2048) in the series style.
    manim -s --resolution 2048,2048 youtube/covers/covers.py LLMCover FoundationsCover
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "framework"))
from manim import *
from common import token, token_row, TOKEN_COLOR, MODEL_COLOR

config.frame_width = config.frame_height = 8


def fit(*parts):
    g = VGroup(*parts).arrange(DOWN, buff=0.45)
    g.scale(min(6.8 / g.width, 6.8 / g.height))
    return g.move_to(ORIGIN)


def title_block(main, sub):
    return VGroup(Text(main, font_size=64, weight=BOLD), Text(sub, font_size=30, color=GREY_B)).arrange(DOWN, buff=0.3)


class LLMCover(Scene):
    def construct(self):
        row = token_row(["The", "cat", "sat", "on", "the"], font_size=26).scale(0.9)
        box = RoundedRectangle(corner_radius=0.2, width=2.6, height=1.3, color=MODEL_COLOR,
                               fill_color=MODEL_COLOR, fill_opacity=0.25)
        model = VGroup(box, Text("LLM", font_size=40, weight=BOLD).move_to(box))
        nxt = token("mat", color=YELLOW, font_size=30)
        nxt[0].set_fill(YELLOW, 0.3)
        flow = VGroup(row, Arrow(UP, DOWN, buff=0, color=GREY_B).scale(0.6), model,
                      Arrow(UP, DOWN, buff=0, color=GREY_B).scale(0.6), nxt).arrange(DOWN, buff=0.25)
        tb = title_block("How LLMs Work", "Build a language model from scratch")
        self.add(fit(tb, flow, Text("14 episodes · study guides included", font_size=22, color=GREY_C)))


class FoundationsCover(Scene):
    def construct(self):
        plane = NumberPlane(x_range=[-3, 3], y_range=[-2, 2], x_length=4.8, y_length=3.2,
                            background_line_style={"stroke_opacity": 0.35})
        a = Arrow(plane.c2p(0, 0), plane.c2p(2, 1), buff=0, color=TOKEN_COLOR)
        b = Arrow(plane.c2p(0, 0), plane.c2p(0.5, 1.8), buff=0, color=YELLOW)
        curve = plane.plot(lambda x: 1.6 / (1 + 2.718 ** (-2.2 * x)) - 0.8, x_range=[-3, 3], color=MODEL_COLOR)
        formula = Text("softmax:  pᵢ = eᶻⁱ / Σ eᶻʲ", font_size=30, font="DejaVu Sans")
        art = VGroup(VGroup(plane, curve, a, b), formula).arrange(DOWN, buff=0.4)
        tb = title_block("How LLMs Work: Foundations", "The math and code behind LLMs")
        tb = VGroup(Text("How LLMs Work", font_size=64, weight=BOLD), Text("Foundations", font_size=56, weight=BOLD, color=YELLOW),
                    Text("The math and code behind LLMs", font_size=30, color=GREY_B)).arrange(DOWN, buff=0.25)
        self.add(fit(tb, art, Text("14 short lessons · study guides included", font_size=22, color=GREY_C)))


class PracticeCover(Scene):
    def construct(self):
        def bubble(text, color):
            label = Text(text, font_size=24)
            box = RoundedRectangle(corner_radius=0.2, width=label.width + 0.5, height=label.height + 0.4,
                                   stroke_width=0, fill_color=color, fill_opacity=0.9)
            return VGroup(box, label.move_to(box))
        chat = VGroup(bubble("How long should I boil an egg?", BLUE_E), bubble("About 7 minutes.", GREY_D))
        chat.arrange(DOWN, buff=0.25)
        chat[0].shift(0.6 * RIGHT)
        chat[1].shift(0.6 * LEFT)
        tools = VGroup(*[token(w, color=c, font_size=24) for w, c in
                         [("prompts", TOKEN_COLOR), ("RAG", TEAL_C), ("tools", GOLD), ("agents", MODEL_COLOR)]])
        tools.arrange(RIGHT, buff=0.18)
        tb = VGroup(Text("LLMs in Practice", font_size=64, weight=BOLD),
                    Text("Building real products on language models", font_size=30, color=GREY_B))
        tb.arrange(DOWN, buff=0.3)
        self.add(fit(tb, chat, tools, Text("12 episodes · study guides and code included", font_size=22,
                                            color=GREY_C)))
