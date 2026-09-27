"""Video 1 — What is an LLM?

Render from the repo root:  ./render.sh how-llms-work v01
"""
import random

from manim import *

from intro import play_token_intro
from v01_script import TAGLINE, TITLE
from voiced_scene import VoicedScene

TOKEN_COLOR = BLUE_D
MODEL_COLOR = PURPLE_B
MONO = "DejaVu Sans Mono"

PROBS = [("mat", 0.41), ("floor", 0.22), ("sofa", 0.12), ("bed", 0.09), ("roof", 0.05), ("banana", 0.0001)]

CODE = """def generate(model, tokens, max_new=50):
    for _ in range(max_new):
        probs = model(tokens)      # a probability per word
        next_tok = sample(probs)   # pick one
        tokens.append(next_tok)    # add it to the text
        if next_tok == END:        # model says it's done
            break
    return tokens"""


def token(word, color=TOKEN_COLOR, font_size=28):
    label = Text(word, font_size=font_size)
    box = RoundedRectangle(corner_radius=0.12, width=max(label.width + 0.35, 0.6), height=0.65,
                           stroke_color=color, fill_color=color, fill_opacity=0.3)
    return VGroup(box, label.move_to(box))


def model_box(width=2.4, height=1.6):
    box = RoundedRectangle(corner_radius=0.2, width=width, height=height,
                           stroke_color=MODEL_COLOR, fill_color=MODEL_COLOR, fill_opacity=0.25)
    return VGroup(box, Text("LLM", font_size=40, weight=BOLD).move_to(box))


def chat_bubble(text, color, align):
    label = Text(text, font_size=26, line_spacing=0.8)
    box = RoundedRectangle(corner_radius=0.25, width=label.width + 0.6, height=label.height + 0.5,
                           stroke_width=0, fill_color=color, fill_opacity=0.9)
    return VGroup(box, label.move_to(box).align_to(box, align).shift(-0.3 * align))


class WhatAnLLMDoes(VoicedScene):
    VIDEO = "v01"

    def construct(self):
        play_token_intro(self, TITLE, 1, TAGLINE)

        # 1. Hook — a chat exchange ------------------------------------------
        self.section(1)
        question = chat_bubble("Why do cats love boxes?", BLUE_E, RIGHT).to_edge(RIGHT, buff=1).shift(1.8 * UP)
        answer = chat_bubble("Boxes feel safe and warm,\nso a cat can relax and\nwatch the world from cover.",
                             GREY_D, LEFT).to_edge(LEFT, buff=1).shift(0.1 * DOWN)
        self.at("question")
        self.play(FadeIn(question, shift=0.3 * UP))
        self.at("writes")
        self.play(FadeIn(answer[0]), AddTextWordByWord(answer[1], run_time=2.2))
        self.at("magic")
        cursor = Rectangle(width=0.08, height=0.4, stroke_width=0, fill_color=WHITE, fill_opacity=1)
        cursor.next_to(answer[1], RIGHT, buff=0.1).align_to(answer[1], DOWN)
        self.play(FadeIn(cursor))
        self.play(cursor.animate.set_opacity(0), rate_func=there_and_back, run_time=0.6)
        self.play(cursor.animate.set_opacity(0), rate_func=there_and_back, run_time=0.6)
        self.at("core")
        title = Text("One simple thing, over and over.", font_size=44)
        self.play(FadeOut(question), FadeOut(answer), FadeOut(cursor))
        self.at("simple")
        self.play(Write(title))
        self.end_section()

        # 2. Next-word prediction ----------------------------------------------
        self.section(2)
        sentence = VGroup(*[token(w) for w in "The cat sat on the".split()]).arrange(RIGHT, buff=0.12)
        sentence.to_corner(UL, buff=0.7).shift(0.6 * DOWN)
        llm = model_box().move_to([-4.2, -0.6, 0])
        feed = Arrow(sentence.get_bottom() + 1.6 * LEFT, llm.get_top(), buff=0.15, color=GREY_B)
        out_slot = DashedVMobject(RoundedRectangle(corner_radius=0.12, width=0.9, height=0.65, color=YELLOW))
        out_slot.next_to(llm, RIGHT, buff=1.2)
        out_arrow = Arrow(llm.get_right(), out_slot.get_left(), buff=0.15, color=GREY_B)
        qmark = Text("?", font_size=36, color=YELLOW).move_to(out_slot)
        self.play(FadeOut(title, shift=UP))
        self.play(FadeIn(llm, scale=0.8))
        self.at("words")
        self.play(LaggedStart(*[FadeIn(t, shift=0.2 * DOWN) for t in sentence], lag_ratio=0.15))
        self.play(GrowArrow(feed))
        self.at("question")
        self.play(GrowArrow(out_arrow), Create(out_slot))
        self.at("next")
        self.play(Write(qmark))
        self.end_section()

        # 3. A probability for every word -----------------------------------------
        self.section(3)
        header = Text("probability of the next word", font_size=24, color=GREY_B)
        rows = VGroup()
        for i, (word, p) in enumerate(PROBS):
            y = -0.6 * i
            bar = Rectangle(width=max(p * 13, 0.02), height=0.42, stroke_width=0,
                            fill_color=TEAL, fill_opacity=0.85).move_to([0, y, 0], aligned_edge=LEFT)
            label = Text(word, font_size=26).move_to([-0.25, y, 0], aligned_edge=RIGHT)
            value = Text(f"{p:.0%}" if p >= 0.01 else "0.01%", font_size=22, color=GREY_A)
            rows.add(VGroup(label, bar, value.next_to(bar, RIGHT, buff=0.2)))
        chart = VGroup(header, rows).arrange(DOWN, buff=0.3, aligned_edge=LEFT)
        chart.move_to([2.3, -1.0, 0])
        more = Text("… and ~50,000 more words", font_size=22, color=GREY_B).next_to(rows, DOWN, buff=0.25, aligned_edge=LEFT)
        self.at("probability")
        self.play(FadeOut(qmark), FadeOut(out_slot), Write(header))
        self.at("every")
        self.play(LaggedStart(*[AnimationGroup(FadeIn(r[0]), GrowFromEdge(r[1], LEFT), FadeIn(r[2]))
                                for r in rows], lag_ratio=0.12), FadeIn(more))
        for word, i in [("mat", 0), ("floor", 1), ("sofa", 2), ("banana", 5)]:
            self.at(word)
            self.play(Indicate(rows[i], color=YELLOW, scale_factor=1.08), run_time=0.8)
        self.end_section()

        # 4. Pick one and append it -------------------------------------------------
        self.section(4)
        pick = SurroundingRectangle(rows[0], color=YELLOW, buff=0.1)
        self.at("pick")
        self.play(Create(pick))
        self.at("add")
        mat = token("mat", color=YELLOW).move_to(rows[0][0])
        self.play(FadeIn(mat), run_time=0.3)
        self.play(mat.animate.next_to(sentence, RIGHT, buff=0.12),
                  FadeOut(VGroup(chart, more, pick)), run_time=0.9)
        mat[0].set_stroke(TOKEN_COLOR).set_fill(TOKEN_COLOR, 0.3)
        sentence.add(mat)
        self.end_section()

        # 5. The loop -----------------------------------------------------------------
        self.section(5)
        self.at("feed")
        self.play(ShowPassingFlash(feed.copy().set_color(YELLOW), time_width=0.6),
                  Indicate(sentence, color=YELLOW, scale_factor=1.03))
        self.at("predicts")
        self.play(Indicate(llm, color=YELLOW, scale_factor=1.08))
        cycle_words = ["predict", "pick", "append", "repeat"]
        cycle = VGroup(*[Text(w, font_size=30, color=GREY_B) for w in cycle_words])
        cycle.arrange(RIGHT, buff=0.9).move_to([2.0, -2.6, 0])
        cycle_arrows = VGroup(*[Arrow(a.get_right(), b.get_left(), buff=0.12, color=GREY_D, stroke_width=3)
                                for a, b in zip(cycle[:-1], cycle[1:])])
        for cue, word in [("period", "."), ("another", "It"), ("another", "purred")]:
            self.at(cue)
            new = token(word, color=YELLOW).move_to(out_slot)
            self.play(FadeIn(new, scale=0.6), run_time=0.25)
            self.play(new.animate.next_to(sentence, RIGHT, buff=0.12), run_time=0.5)
            new[0].set_stroke(TOKEN_COLOR).set_fill(TOKEN_COLOR, 0.3)
            sentence.add(new)
        self.at("predict")
        self.play(FadeIn(cycle[0]))
        for i, cue in enumerate(cycle_words[1:], 1):
            self.at(cue)
            self.play(GrowArrow(cycle_arrows[i - 1]), FadeIn(cycle[i]),
                      cycle[i - 1].animate.set_color(GREY_B), cycle[i].animate.set_color(YELLOW), run_time=0.5)
        self.end_section()

        # 6. The loop in code ---------------------------------------------------------
        self.section(6)
        code = Code(code_string=CODE, language="python", formatter_style="monokai",
                    background="window", paragraph_config={"font_size": 24}).move_to(ORIGIN)
        row_h = abs(code.line_numbers[0].get_y() - code.line_numbers[1].get_y())
        hl = Rectangle(width=code.code_lines.width + 0.25, height=row_h, color=YELLOW, stroke_width=2)
        hl.align_to(code.code_lines, LEFT).shift(0.12 * LEFT).match_y(code.line_numbers[0])

        def move_hl(i):
            return hl.animate.match_y(code.line_numbers[i])

        self.at("code")
        self.play(FadeOut(VGroup(sentence, llm, feed, out_arrow, cycle, cycle_arrows)), FadeIn(code, shift=0.3 * UP))
        self.at("tokens")
        self.play(Create(hl))
        self.at("probabilities")
        self.play(move_hl(2))
        self.at("sample")
        self.play(move_hl(3))
        self.at("append")
        self.play(move_hl(4))
        self.at("around")
        self.play(move_hl(1))
        self.at("done")
        self.play(move_hl(5))
        self.end_section()

        # 7. What is inside `model`? -------------------------------------------------
        self.section(7)
        self.at("model")
        self.play(move_hl(2))
        self.play(Circumscribe(code.code_lines[2], color=YELLOW))
        random.seed(7)
        numbers = VGroup(*[
            Text("  ".join(f"{random.gauss(0, 0.6):+.2f}" for _ in range(9)), font=MONO, font_size=16, color=GREY_B)
            for _ in range(11)
        ]).arrange(DOWN, buff=0.12)
        big = RoundedRectangle(corner_radius=0.3, width=numbers.width + 0.8, height=numbers.height + 0.8,
                               stroke_color=MODEL_COLOR, fill_color=MODEL_COLOR, fill_opacity=0.12)
        big_label = Text("model", font=MONO, font_size=28, color=MODEL_COLOR).next_to(big, UP, buff=0.15)
        brain = VGroup(big, numbers, big_label).move_to(0.2 * DOWN)
        self.at("underneath")
        self.play(FadeOut(code), FadeOut(hl), FadeIn(big), FadeIn(big_label))
        self.at("pile")
        self.play(LaggedStart(*[FadeIn(r) for r in numbers], lag_ratio=0.08, run_time=1.2))
        self.play(brain.animate.scale(0.65).shift(0.6 * RIGHT))
        prompt = Text("the cat sat on the", font_size=26).next_to(big, LEFT, buff=0.25)
        guess = Text("mat?", font_size=34, color=YELLOW).next_to(big, RIGHT, buff=0.3)
        self.at("read")
        self.play(FadeIn(prompt, shift=0.3 * RIGHT))
        self.at("likely")
        self.play(FadeIn(guess, shift=0.3 * RIGHT), Indicate(numbers, color=WHITE, scale_factor=1.02))
        self.at("series")
        self.play(Circumscribe(big, color=YELLOW, time_width=0.8))
        self.end_section()

        # 8. Map of the series --------------------------------------------------------
        self.section(8)
        heading = Text("The road ahead", font_size=40).to_edge(UP, buff=0.5)
        steps = [("2", "Tokens"), ("3", "Embeddings"), ("4", "Position"),
                 ("5–7", "Attention"), ("8", "MLP"), ("10", "Probabilities")]
        boxes = VGroup()
        for num, name in steps:
            label = Text(name, font_size=20)
            box = RoundedRectangle(corner_radius=0.15, width=2.0, height=0.9,
                                   stroke_color=GREY_B, fill_color=GREY_E, fill_opacity=0.6)
            ep = Text(num, font_size=18, color=GREY_B).next_to(box, UP, buff=0.1)
            boxes.add(VGroup(box, label.move_to(box), ep))
        boxes.arrange(RIGHT, buff=0.3).move_to(0.9 * UP)
        links = VGroup(*[Arrow(a[0].get_right(), b[0].get_left(), buff=0.04, color=GREY_C,
                               stroke_width=3, max_tip_length_to_length_ratio=0.4)
                         for a, b in zip(boxes[:-1], boxes[1:])])
        block = DashedVMobject(SurroundingRectangle(VGroup(boxes[3], boxes[4]), buff=0.18, color=ORANGE,
                                                    corner_radius=0.15), num_dashes=60)
        block_label = Text("9 · Transformer block  × N", font_size=22, color=ORANGE).next_to(block, DOWN, buff=0.15)
        extras = VGroup()
        for num, name in [("11", "Training"), ("12", "Build a tiny GPT")]:
            label = Text(name, font_size=22)
            box = RoundedRectangle(corner_radius=0.15, width=label.width + 0.6, height=0.9,
                                   stroke_color=GREY_B, fill_color=GREY_E, fill_opacity=0.6)
            ep = Text(num, font_size=18, color=GREY_B).next_to(box, UP, buff=0.1)
            extras.add(VGroup(box, label.move_to(box), ep))
        extras.arrange(RIGHT, buff=1.2).move_to(2.1 * DOWN)

        def light(group, color=TOKEN_COLOR):
            return group[0].animate.set_stroke(color).set_fill(color, 0.35)

        self.play(FadeOut(VGroup(brain, prompt, guess)), Write(heading))
        cues = [("tokens",), ("embedding",), ("position",), ("attention", "tension"), ("mlp",)]
        for i, cue in enumerate(cues):
            self.at(*cue)
            anims = [FadeIn(boxes[i], shift=0.2 * UP), light(boxes[i])]
            if i > 0:
                anims.append(GrowArrow(links[i - 1]))
                anims.append(boxes[i - 1][0].animate.set_stroke(GREY_B).set_fill(GREY_E, 0.6))
            self.play(*anims, run_time=0.7)
        self.at("transformer")
        self.play(boxes[4][0].animate.set_stroke(GREY_B).set_fill(GREY_E, 0.6),
                  Create(block), FadeIn(block_label))
        self.at("probabilities")
        self.play(GrowArrow(links[4]), FadeIn(boxes[5], shift=0.2 * UP), light(boxes[5]), run_time=0.7)
        self.at("training")
        self.play(boxes[5][0].animate.set_stroke(GREY_B).set_fill(GREY_E, 0.6),
                  FadeIn(extras[0], shift=0.2 * UP), light(extras[0], GREEN), run_time=0.7)
        self.at("tiny", "gpt")
        self.play(extras[0][0].animate.set_stroke(GREY_B).set_fill(GREY_E, 0.6),
                  FadeIn(extras[1], shift=0.2 * UP), light(extras[1], GOLD), run_time=0.7)
        self.end_section()

        # 9. Next up: tokens --------------------------------------------------------
        self.section(9)
        next_label = Text("Next up", font_size=30, color=GREY_B)
        next_title = Text("Tokens: Chopping Text into Pieces", font_size=44)
        card = VGroup(next_label, next_title).arrange(DOWN, buff=0.35)
        self.play(FadeOut(VGroup(heading, links, block, block_label, extras, *boxes[1:])),
                  light(boxes[0]))
        self.at("tokens")
        self.play(ReplacementTransform(boxes[0], card))
        self.end_section()
        self.play(FadeOut(card))
        self.wait(0.5)
