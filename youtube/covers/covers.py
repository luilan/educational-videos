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


class DeepDiveCover(Scene):
    def construct(self):
        # a transformer block, opened up: the parts the series explains
        chips = VGroup(*[token(h, color=c, font_size=24) for h, c in
                         [("BPE", TOKEN_COLOR), ("RoPE", TEAL_C), ("attention", GOLD), ("RMSNorm", GREEN_C),
                          ("SwiGLU", RED_C), ("MoE", MODEL_COLOR), ("KV cache", BLUE_C), ("RLHF", MAROON_C)]])
        chips.arrange_in_grid(2, 4, buff=(0.25, 0.3))
        block = SurroundingRectangle(chips, buff=0.35, corner_radius=0.25, color=YELLOW, stroke_width=4)
        label = Text("inside the transformer", font_size=22, color=YELLOW).next_to(block, UP, buff=0.12)
        art = VGroup(label, block, chips)
        tb = VGroup(Text("How LLMs Work", font_size=60, weight=BOLD),
                    Text("Deep Dive", font_size=60, weight=BOLD, color=YELLOW),
                    Text("Inside a modern LLM, piece by piece", font_size=30, color=GREY_B)).arrange(DOWN, buff=0.25)
        self.add(fit(tb, art, Text("real code · study guides", font_size=22, color=GREY_C)))


class AgentsCover(Scene):
    def construct(self):
        # a model inside its harness: the parts the series explains around it
        box = RoundedRectangle(corner_radius=0.2, width=2.2, height=1.1, color=MODEL_COLOR,
                               fill_color=MODEL_COLOR, fill_opacity=0.25)
        model = VGroup(box, Text("model", font_size=34, weight=BOLD).move_to(box))
        chips = VGroup(*[token(h, color=c, font_size=22) for h, c in
                         [("tools", GOLD), ("loop", TEAL_C), ("sandbox", RED_C), ("approvals", MAROON_C),
                          ("AGENTS.md", BLUE_C), ("context", GREEN_C), ("MCP", TOKEN_COLOR), ("sub-agents", PURPLE_B)]])
        top, bottom = chips[:4].arrange(RIGHT, buff=0.25), chips[4:].arrange(RIGHT, buff=0.25)
        inner = VGroup(top, model, bottom).arrange(DOWN, buff=0.35)
        harness = SurroundingRectangle(inner, buff=0.35, corner_radius=0.3, color=TEAL_C, stroke_width=4)
        label = Text("the harness", font_size=22, color=TEAL_C).next_to(harness, UP, buff=0.12)
        art = VGroup(label, harness, inner)
        tb = VGroup(Text("How AI Agents Work", font_size=60, weight=BOLD),
                    Text("Inside the Harness", font_size=56, weight=BOLD, color=TEAL_C),
                    Text("How coding agents really run", font_size=30, color=GREY_B)).arrange(DOWN, buff=0.25)
        self.add(fit(tb, art, Text("real traces · open-source code · study guides", font_size=22, color=GREY_C)))


class GenerativeCover(Scene):
    def construct(self):
        # noise → picture, and a waveform: the series turns noise into images and sound
        rng = np.random.default_rng(7)
        n, s = 12, 0.16

        def grid(f):
            g = VGroup()
            for i in range(n):
                for j in range(n):
                    x, y = (j - n / 2 + 0.5) / n, (n / 2 - i - 0.5) / n
                    inside = x * x + y * y < 0.13
                    clean = np.array([0.95, 0.45, 0.25]) if inside else np.array([0.12, 0.14, 0.2])
                    v = np.clip((1 - f) * clean + f * rng.uniform(0, 1, 3), 0, 1)
                    g.add(Square(side_length=s, stroke_width=0, fill_color=rgb_to_color(v), fill_opacity=1)
                          .move_to([(j - n / 2 + 0.5) * s, (n / 2 - i - 0.5) * s, 0]))
            return g
        steps = VGroup(*[grid(f) for f in (1.0, 0.55, 0.0)]).arrange(RIGHT, buff=0.55)
        arrows = VGroup(*[Arrow(steps[k].get_right(), steps[k + 1].get_left(), buff=0.08, color=GREY_B,
                                max_tip_length_to_length_ratio=0.35) for k in range(2)])
        xs = np.linspace(-2.9, 2.9, 300)
        wave = VMobject(color=TEAL_C, stroke_width=4).set_points_smoothly(
            [[x, 0.45 * np.sin(5 * x) * np.exp(-0.25 * x * x) + 0.15 * np.sin(17 * x), 0] for x in xs])
        art = VGroup(VGroup(steps, arrows), wave).arrange(DOWN, buff=0.5)
        tb = VGroup(Text("How Generative Models Work", font_size=56, weight=BOLD),
                    Text("Images and Sound", font_size=56, weight=BOLD, color=ORANGE),
                    Text("From noise to pictures and voices", font_size=30, color=GREY_B)).arrange(DOWN, buff=0.25)
        self.add(fit(tb, art, Text("small models · runnable code · study guides", font_size=22, color=GREY_C)))


class RLCover(Scene):
    def construct(self):
        # a model tries, a checker grades, and the reward curve climbs
        box = RoundedRectangle(corner_radius=0.2, width=2.0, height=1.0, color=BLUE_C, fill_color=BLUE_C, fill_opacity=0.25)
        model = VGroup(box, Text("model", font_size=32, weight=BOLD).move_to(box))
        tries = VGroup()
        for ans, r in [("85.", 1), ("75.", 0), ("85.", 1), ("8.", 0)]:
            col = GREEN_C if r else RED_C
            badge = VGroup(Circle(radius=0.2, color=col, fill_opacity=0.15, stroke_width=3),
                           Text(str(r), font_size=20, color=col))
            badge[1].move_to(badge[0])
            tries.add(VGroup(token_row(list(ans), font_size=22, buff=0.04), badge).arrange(RIGHT, buff=0.18))
        tries.arrange(DOWN, buff=0.16, aligned_edge=LEFT)
        flow = VGroup(model, tries).arrange(RIGHT, buff=0.6)
        arrow = Arrow(model.get_right(), tries.get_left(), buff=0.1, color=GREY_B)
        ax = Axes(x_range=[0, 10, 10], y_range=[0, 1, 1], x_length=4.6, y_length=1.5, tips=False,
                  axis_config={"color": GREY_C, "stroke_width": 2})
        curve = ax.plot(lambda x: 0.14 + 0.8 / (1 + np.exp(-(x - 4.5) * 1.1)), color=YELLOW, stroke_width=5)
        reward = VGroup(ax, curve, Text("reward", font_size=20, color=YELLOW).next_to(ax, LEFT, buff=0.15))
        art = VGroup(VGroup(flow, arrow), reward).arrange(DOWN, buff=0.45)
        tb = VGroup(Text("Reinforcement Learning", font_size=60, weight=BOLD),
                    Text("for LLMs", font_size=60, weight=BOLD, color=YELLOW),
                    Text("Learning from a grade, built from scratch", font_size=30, color=GREY_B)).arrange(DOWN, buff=0.25)
        self.add(fit(tb, art, Text("REINFORCE · PPO · GRPO · reasoning · runnable code", font_size=22, color=GREY_C)))


class FrontierCover(Scene):
    def construct(self):
        # a network editing its own weights (a loop back onto itself) and a memory beside it
        rng = np.random.default_rng(11)
        n, s = 8, 0.3
        cells = VGroup()
        for i in range(n):
            for j in range(n):
                v = rng.normal()
                col = interpolate_color(BLUE_D, ORANGE, (np.tanh(v) + 1) / 2)
                cells.add(Square(side_length=s, stroke_width=0.5, stroke_color=BLACK, fill_color=col, fill_opacity=1)
                          .move_to([(j - n / 2 + 0.5) * s, (n / 2 - i - 0.5) * s, 0]))
        edited = cells[19]
        ring = SurroundingRectangle(edited, buff=0.02, color=YELLOW, stroke_width=4)
        weights = VGroup(cells, ring, Text("its own weights", font_size=24, color=GREY_B).next_to(cells, DOWN, buff=0.15))
        box = RoundedRectangle(corner_radius=0.2, width=2.3, height=1.2, color=PURPLE_B, fill_color=PURPLE_B, fill_opacity=0.25)
        model = VGroup(box, Text("model", font_size=38, weight=BOLD).move_to(box))
        notes = VGroup(*[RoundedRectangle(corner_radius=0.05, width=1.8, height=0.38, color=TEAL_C, fill_opacity=0.2)
                         for _ in range(4)]).arrange(DOWN, buff=0.08)
        memory = VGroup(notes, Text("memory", font_size=24, color=TEAL_C).next_to(notes, UP, buff=0.15))
        row = VGroup(weights, model, memory).arrange(RIGHT, buff=0.7)
        edit = CurvedArrow(model.get_top() + 0.05 * UP, ring.get_top() + 0.05 * UP, angle=TAU / 5, color=YELLOW)
        rem = CurvedArrow(model.get_bottom() + 0.05 * DOWN, notes.get_bottom() + 0.05 * DOWN, angle=TAU / 6, color=TEAL_C)
        art = VGroup(row, edit, rem)
        tb = VGroup(Text("The Frontier", font_size=64, weight=BOLD, color=YELLOW),
                    Text("Self-Improvement and Memory", font_size=48, weight=BOLD),
                    Text("Models that edit themselves, and remember", font_size=30, color=GREY_B)).arrange(DOWN, buff=0.25)
        self.add(fit(tb, art, Text("current papers · rebuilt small · runnable code", font_size=22, color=GREY_C)))
