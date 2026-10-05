"""How LLMs Work: Deep Dive, episode 3 — Tokenizer Quirks That Shape Model Behavior.

Render from the repo root:  ./render.sh deep-dive d03
Every token id, split, probability and rank on screen comes from code/d03_tokenizer_quirks/quirks.py
(GPT-2 and Qwen2.5 tokenizers, Qwen2.5-0.5B-Instruct, GPT-2 small's embedding table).
"""
from manim import *

from common import MODEL_COLOR, MONO, TOKEN_COLOR, code_panel, finish, highlight, next_up_card, token
from intro import play_token_intro
from d03_script import NEXT, TAGLINE, TITLE
from voiced_scene import VoicedScene

SERIES_LABEL = "HOW LLMs WORK: DEEP DIVE  ·  EPISODE 3"
HELLOS = [("hello", 31373), ("␣hello", 23748), ("Hello", 15496), ("␣Hello", 18435)]
NUMBERS = [("1234567", ["123", "45", "67"], list("1234567")), ("2024", ["20", "24"], list("2024")),
           ("3.14159", ["3", ".", "14", "159"], list("3.14159"))]
NO_SPACE = [("␣Paris", 0.302), ("␣______", 0.123), (":↵", 0.066), (":↵↵", 0.058), ("␣__", 0.048)]
SPACE = [("1", 0.078), ("2", 0.060), ("3", 0.035), ("␣.↵", 0.032), ("．", 0.032)]
GLITCH = [("␣SolidGoldMagikarp", 43453), ("␣TheNitromeFan", 42090), ("␣davidjl", 23282)]
CLOSEST = ["␣externalToEVA", "control chars", "broken bytes (�)", "quickShip"]
CODE = """gpt2 = AutoTokenizer.from_pretrained("openai-community/gpt2")

gpt2("Hello")["input_ids"]          # [15496]
gpt2(" Hello")["input_ids"]         # [18435]  a different token
ids = gpt2("1234567")["input_ids"]
[gpt2.decode([i]) for i in ids]     # ['123', '45', '67']"""


def probs_panel(title, rows, color):
    head = Text(title, font=MONO, font_size=22, color=BLUE_B)
    toks = [token(t, GREY_B, 18) for t, _ in rows]
    tw = max(t.width for t in toks)
    lines = VGroup()
    for k, ((_, p), tk) in enumerate(zip(rows, toks)):
        y = -k * 0.62
        tk.move_to([0, y, 0], aligned_edge=LEFT)
        bar = Rectangle(width=max(p * 9, 0.04), height=0.3, stroke_width=0, fill_color=color, fill_opacity=0.85)
        bar.move_to([tw + 0.2, y, 0], aligned_edge=LEFT)
        num = Text(f"{p:.3f}", font=MONO, font_size=18).next_to(bar, RIGHT, buff=0.15)
        lines.add(VGroup(tk, bar, num))
    return VGroup(head, lines).arrange(DOWN, aligned_edge=LEFT, buff=0.25)


class TokenizerQuirksVideo(VoicedScene):
    VIDEO = "d03"

    def construct(self):
        play_token_intro(self, TITLE, 3, TAGLINE, label=SERIES_LABEL)
        self.hook()          # 1
        self.case()          # 2
        self.numbers()       # 3
        self.trailing()      # 4
        self.glitch()        # 5
        self.finding()       # 6
        self.lessons()       # 7
        self.code()          # 8
        self.outro()         # 9

    def clear_stage(self, *keep, run_time=0.6):
        self.play(*[FadeOut(m) for m in self.mobjects if m not in keep], run_time=run_time)

    # ------------------------------------------------------------------ 1. hook
    def hook(self):
        self.section(1)
        flow = VGroup(Text("text", font_size=30), Arrow(LEFT, RIGHT, color=GREY_B),
                      VGroup(RoundedRectangle(corner_radius=0.2, width=2.6, height=1.0, stroke_color=YELLOW,
                                              fill_color=YELLOW, fill_opacity=0.15), Text("tokenizer", font_size=28)),
                      Arrow(LEFT, RIGHT, color=GREY_B), Text("what the model sees", font_size=30)).arrange(RIGHT, buff=0.3)
        flow[2][1].move_to(flow[2][0])
        flow.move_to([0, 1.0, 0])
        self.at("tokenizer")
        self.play(FadeIn(flow), run_time=0.6)
        q = Text("small quirks → strange behaviour", font_size=32, color=YELLOW).move_to([0, -0.6, 0])
        self.at("quirks")
        self.play(FadeIn(q), run_time=0.4)
        four = Text("four quirks, each tested for real", font_size=26, color=GREY_B).next_to(q, DOWN, buff=0.4)
        self.at("four")
        self.play(FadeIn(four), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 2. case and spaces
    def case(self):
        self.section(2)
        self.clear_stage()
        head = Text("quirk 1 · the same word, many tokens (GPT-2)", font_size=30).to_edge(UP, buff=0.5)
        self.at("same")
        self.play(FadeIn(head), run_time=0.4)
        rows = VGroup(*[VGroup(token(w, TOKEN_COLOR, 30), Text(f"id {i:,}", font=MONO, font_size=26, color=YELLOW))
                        .arrange(RIGHT, buff=0.5) for w, i in HELLOS]).arrange(DOWN, aligned_edge=LEFT, buff=0.3)
        rows.move_to([-1.5, 0.6, 0])
        for k, cue in [(0, "hello"), (1, "front"), (2, "capital")]:
            self.at(cue)
            self.play(FadeIn(rows[k], shift=0.2 * RIGHT), run_time=0.4)
        self.play(FadeIn(rows[3], shift=0.2 * RIGHT), run_time=0.3)
        brace = Brace(rows, RIGHT, color=GREY_B)
        four = Text("4 unrelated numbers\nfor one word", font_size=26, color=YELLOW, line_spacing=0.8).next_to(brace, RIGHT)
        self.at("four")
        self.play(GrowFromCenter(brace), FadeIn(four), run_time=0.5)
        sep = Text("the model learns separately that they mean the same", font_size=24, color=GREY_A).move_to([0, -1.9, 0])
        self.at("separately")
        self.play(FadeIn(sep), run_time=0.4)
        caps = VGroup(Text("HELLO →", font_size=28), *[token(t, RED_C, 28) for t in ["HE", "LL", "O"]]).arrange(RIGHT, buff=0.15)
        caps.move_to([0, -2.8, 0])
        self.at("capitals")
        self.play(FadeIn(caps), run_time=0.5)
        self.end_section()

    # ------------------------------------------------------------------ 3. numbers
    def numbers(self):
        self.section(3)
        self.clear_stage()
        head = Text("quirk 2 · numbers", font_size=30).to_edge(UP, buff=0.5)
        self.at("numbers")
        self.play(FadeIn(head), run_time=0.4)
        lab_g = Text("GPT-2", font_size=24, color=RED_B).move_to([0.2, 2.4, 0])
        lab_q = Text("Qwen2.5", font_size=24, color=GREEN_B).move_to([4.4, 2.4, 0])
        rows = VGroup()
        for k, (n, g, q) in enumerate(NUMBERS):
            y = 1.5 - k * 1.0
            num = Text(n, font=MONO, font_size=30).move_to([-4.6, y, 0])
            gt = VGroup(*[token(t, RED_C, 26) for t in g]).arrange(RIGHT, buff=0.08).move_to([0.2, y, 0])
            qt = VGroup(*[token(t, GREEN_C, 22) for t in q]).arrange(RIGHT, buff=0.05).move_to([4.4, y, 0])
            rows.add(VGroup(num, gt, qt))
        self.at("cuts")
        self.play(FadeIn(lab_g), FadeIn(rows[0][:2]), run_time=0.6)
        self.at("2024")
        self.play(FadeIn(rows[1][:2]), FadeIn(rows[2][:2]), run_time=0.5)
        val = Text("pieces don't line up with place value", font_size=26, color=RED_B).move_to([0, -1.9, 0])
        self.at("value")
        self.play(FadeIn(val), run_time=0.4)
        self.at("digits")
        self.play(FadeIn(lab_q), *[FadeIn(r[2]) for r in rows], run_time=0.7)
        newer = Text("one token per digit: each digit keeps its place", font_size=26, color=GREEN_B).next_to(val, DOWN, 0.25)
        self.at("newer")
        self.play(FadeIn(newer), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 4. trailing space
    def trailing(self):
        self.section(4)
        self.clear_stage()
        head = Text("quirk 3 · a trailing space (Qwen2.5-0.5B)", font_size=30).to_edge(UP, buff=0.45)
        self.at("trailing")
        self.play(FadeIn(head), run_time=0.4)
        left = probs_panel("“The capital of France is”", NO_SPACE, GREEN_C).move_to([-3.5, 0.4, 0])
        right = probs_panel("“The capital of France is␣”", SPACE, RED_C).move_to([3.5, 0.4, 0])
        self.at("capital")
        self.play(FadeIn(left[0]), run_time=0.4)
        self.at("paris")
        self.play(FadeIn(left[1]), run_time=0.5)
        self.at("end")
        self.play(FadeIn(right[0]), run_time=0.4)
        paris = Text("␣Paris: rank 28, 0.008", font=MONO, font_size=22, color=YELLOW).next_to(right, DOWN, buff=0.3)
        self.at("28th")
        self.play(FadeIn(right[1]), FadeIn(paris), run_time=0.6)
        why = Text("words carry their space at the start: a lone trailing space is unusual text", font_size=24,
                   color=GREY_A).to_edge(DOWN, buff=0.5)
        self.at("start")
        self.play(FadeIn(why), run_time=0.5)
        self.end_section()

    # ------------------------------------------------------------------ 5. glitch tokens
    def glitch(self):
        self.section(5)
        self.clear_stage()
        head = Text("quirk 4 · glitch tokens", font_size=30).to_edge(UP, buff=0.5)
        self.at("glitch")
        self.play(FadeIn(head), run_time=0.4)
        rows = VGroup(*[VGroup(token(w, GOLD, 28), Text(f"one token, id {i:,}", font=MONO, font_size=22, color=GREY_A))
                        .arrange(RIGHT, buff=0.4) for w, i in GLITCH]).arrange(DOWN, aligned_edge=LEFT, buff=0.3)
        rows.move_to([0, 1.3, 0])
        self.at("magikarp")
        self.play(FadeIn(rows[0]), run_time=0.4)
        user = Text("(SolidGoldMagikarp: a username from a Reddit forum)", font_size=22, color=GREY_B)
        user.next_to(rows, DOWN, buff=0.25)
        self.at("username")
        self.play(FadeIn(user), FadeIn(rows[1:]), run_time=0.5)
        two = VGroup(Text("vocabulary built from text A", font_size=24, color=TEAL_B),
                     Text("model trained on text B, where they almost never appear", font_size=24, color=RED_B))
        two.arrange(DOWN, buff=0.2).move_to([0, -1.5, 0])
        self.at("built")
        self.play(FadeIn(two[0]), run_time=0.4)
        self.at("trained")
        self.play(FadeIn(two[1]), run_time=0.4)
        biz = Text("asked about them, early models of that family produced bizarre answers", font_size=22,
                   color=YELLOW).to_edge(DOWN, buff=0.4)
        self.at("bizarre")
        self.play(FadeIn(biz), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 6. finding them
    def finding(self):
        self.section(6)
        self.clear_stage()
        head = Text("never seen in training → almost no updates → a generic embedding", font_size=28).to_edge(UP, buff=0.5)
        self.at("updates")
        self.play(FadeIn(head), run_time=0.5)
        center = Dot([-3.0, 0.2, 0], radius=0.12, color=YELLOW)
        avg = Text("average embedding", font_size=20, color=YELLOW).next_to(center, DOWN, buff=0.15)
        import random
        rng = random.Random(3)
        cloud = VGroup(*[Dot([-3.0 + rng.gauss(0, 1.1), 0.2 + rng.gauss(0, 0.9), 0], radius=0.04, color=GREY_C)
                         for _ in range(160)])
        self.at("average")
        self.play(FadeIn(cloud), FadeIn(center), FadeIn(avg), run_time=0.6)
        near = VGroup(*[token(t, GREEN_C, 22) for t in CLOSEST]).arrange(DOWN, aligned_edge=LEFT, buff=0.2)
        near.move_to([3.2, 0.6, 0])
        near_l = Text("closest to the average (GPT-2):", font_size=22, color=GREY_B).next_to(near, UP, buff=0.2)
        self.at("control")
        self.play(FadeIn(near_l), FadeIn(near[1]), FadeIn(near[2]), run_time=0.5)
        self.at("eva")
        self.play(FadeIn(near[0]), run_time=0.3)
        self.at("quickship")
        self.play(FadeIn(near[3]), run_time=0.3)
        miss = Text("␣SolidGoldMagikarp: rank 14,357 of 50,257, not flagged", font=MONO, font_size=20, color=RED_B)
        miss.move_to([0, -2.2, 0])
        self.at("flagged")
        self.play(FadeIn(miss), run_time=0.4)
        alln = Text("no single check finds them all", font_size=26, color=YELLOW).next_to(miss, DOWN, buff=0.3)
        self.at("all")
        self.play(FadeIn(alln), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 7. lessons
    def lessons(self):
        self.section(7)
        self.clear_stage()
        items = VGroup(*[Text(t, font_size=28) for t in ["never end a prompt with a trailing space",
                                                          "keep spelling and capitalization consistent",
                                                          "for math, prefer digit-splitting tokenizers",
                                                          "look at how your own data is tokenized"]])
        items.arrange(DOWN, aligned_edge=LEFT, buff=0.45).move_to([0.3, 0, 0])
        ticks = VGroup(*[Text("✓", font_size=28, color=GREEN).next_to(t, LEFT, buff=0.3) for t in items])
        for i, cue in enumerate(["trailing", "spelling", "math", "data"]):
            self.at(cue)
            self.play(FadeIn(items[i], shift=0.2 * RIGHT), FadeIn(ticks[i]), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 8. code
    def code(self):
        self.section(8)
        self.clear_stage()
        code, hl = code_panel(CODE, font_size=22)
        code.move_to([0, 0.3, 0])
        self.play(FadeIn(code, shift=0.2 * UP), run_time=0.5)
        hl.match_y(code.line_numbers[2])
        self.at("tokenize")
        self.play(Create(hl), run_time=0.3)
        self.play(highlight(hl, code, 3), run_time=0.4)
        self.at("pieces")
        self.play(highlight(hl, code, 5), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 9. outro
    def outro(self):
        self.section(9)
        self.clear_stage(run_time=0.4)
        card = next_up_card(NEXT)
        self.at("next")
        self.play(FadeIn(card, shift=0.2 * UP), run_time=0.6)
        self.end_section()
        finish(self, hold=1.5)
