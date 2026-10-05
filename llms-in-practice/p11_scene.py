"""LLMs in Practice, episode 11 — Evaluating LLMs: How We Know It's Better.

Render from the repo root:  ./render.sh llms-in-practice p11
Every score and answer on screen comes from code/p11_evaluation/evaluate.py (Qwen2.5 0.5B/1.5B/3B-Instruct,
greedy, the made-up bakery handbook in the prompt). "By hand" scores: each hard-set answer read and judged for a
correct answer with correct reasoning (see the study guide).
"""
from manim import *

from common import MODEL_COLOR, MONO, code_panel, finish, highlight, next_up_card
from intro import play_token_intro
from p11_script import NEXT, TAGLINE, TITLE
from voiced_scene import VoicedScene

SERIES_LABEL = "LLMs IN PRACTICE  ·  EPISODE 11"
MODELS = ["0.5B", "1.5B", "3B"]
COLORS = [BLUE_C, TEAL_C, GOLD]
HARD_AUTO, HARD_HAND = [3, 3, 6], [2, 1, 5]
CODE = """score = 0
for question, expected in TESTS:
    answer = ask(model, question)
    if check(answer, expected):
        score += 1
    else:
        print("FAIL", question, "->", answer)     # read these!
print(score, "/", len(TESTS))"""


def labeled_box(text, color=MODEL_COLOR, width=2.4, height=1.0, font_size=26):
    label = Text(text, font_size=font_size, line_spacing=0.8)
    box = RoundedRectangle(corner_radius=0.2, width=max(width, label.width + 0.5), height=height, stroke_color=color,
                           fill_color=color, fill_opacity=0.25)
    return VGroup(box, label.move_to(box))


def answer_box(text, color, font_size=20):
    label = Text(text, font_size=font_size, line_spacing=0.8)
    box = RoundedRectangle(corner_radius=0.15, width=label.width + 0.5, height=label.height + 0.35, stroke_width=0,
                           fill_color=color, fill_opacity=0.85)
    return VGroup(box, label.move_to(box))


def score_bars(values, total, y, width=6.0, labels=MODELS):
    group = VGroup()
    for k, (v, name, color) in enumerate(zip(values, labels, COLORS)):
        frame = Rectangle(width=width, height=0.45, stroke_color=GREY_D, stroke_width=1.5)
        fill = Rectangle(width=max(width * v / total, 0.02), height=0.45, stroke_width=0, fill_color=color,
                         fill_opacity=0.85)
        row_y = y - k * 0.65
        frame.move_to([0.6, row_y, 0])
        fill.align_to(frame, LEFT).match_y(frame)
        name_t = Text(name, font=MONO, font_size=24, color=color).next_to(frame, LEFT, buff=0.3)
        val = Text(f"{v} / {total}", font=MONO, font_size=24).next_to(frame, RIGHT, buff=0.3)
        group.add(VGroup(name_t, frame, fill, val))
    return group


class EvaluationVideo(VoicedScene):
    VIDEO = "p11"

    def construct(self):
        play_token_intro(self, TITLE, 11, TAGLINE, label=SERIES_LABEL)
        self.hook()          # 1
        self.test_set()      # 2
        self.exact()         # 3
        self.contains()      # 4
        self.harder()        # 5
        self.read()          # 6
        self.better()        # 7
        self.noise()         # 8
        self.code()          # 9
        self.keep()          # 10
        self.outro()         # 11

    def clear_stage(self, *keep, run_time=0.6):
        self.play(*[FadeOut(m) for m in self.mobjects if m not in keep], run_time=run_time)

    # ------------------------------------------------------------------ 1. hook
    def hook(self):
        self.section(1)
        changes = VGroup(*[labeled_box(t, c, width=3.0, height=0.9, font_size=24) for t, c in
                           [("new prompt", BLUE_C), ("new model", TEAL_C), ("quantized", GOLD)]])
        changes.arrange(RIGHT, buff=0.5).move_to([0, 2.0, 0])
        for cue, k in [("prompt", 0), ("model", 1), ("quantized", 2)]:
            self.at(cue)
            self.play(FadeIn(changes[k], shift=0.2 * DOWN), run_time=0.35)
        q = Text("better?", font_size=48, color=YELLOW).move_to([0, 0.4, 0])
        self.at("better")
        self.play(FadeIn(q, scale=1.3), run_time=0.4)
        tries = Text("“it looked fine on a few tries”", font_size=28, color=RED_B).move_to([0, -0.8, 0])
        cross = Line(tries.get_left(), tries.get_right(), color=RED, stroke_width=4)
        self.at("tries")
        self.play(FadeIn(tries), run_time=0.4)
        self.play(Create(cross), run_time=0.3)
        ev = Text("an evaluation: a test you can run again and again", font_size=30, color=GREEN_B).move_to([0, -2.2, 0])
        self.at("evaluation")
        self.play(FadeIn(ev, shift=0.2 * UP), run_time=0.5)
        self.end_section()

    # ------------------------------------------------------------------ 2. test set
    def test_set(self):
        self.section(2)
        self.clear_stage()
        qs = VGroup(*[Text(t, font_size=22) for t in ["What time does the bakery open on weekdays?   → 7:30",
                                                      "On which day is the bakery closed?   → Monday",
                                                      "How much is a cake for 12 people?   → 39", "… 20 questions"]])
        qs.arrange(DOWN, aligned_edge=LEFT, buff=0.22).move_to([-1.8, 1.4, 0])
        head = Text("known answers", font_size=26, color=YELLOW).next_to(qs, UP, buff=0.3).align_to(qs, LEFT)
        self.at("known")
        self.play(FadeIn(head), LaggedStart(*[FadeIn(q, shift=0.2 * RIGHT) for q in qs[:3]], lag_ratio=0.2),
                  run_time=0.8)
        self.at("20")
        self.play(FadeIn(qs[3]), run_time=0.3)
        hb = labeled_box("handbook\nin the prompt", TEAL_C, width=2.6, height=1.2, font_size=22).move_to([5.3, 1.4, 0])
        self.at("handbook")
        self.play(FadeIn(hb), run_time=0.4)
        chips = VGroup(*[labeled_box(m, c, width=2.0, height=0.8, font_size=26) for m, c in zip(MODELS, COLORS)])
        chips.arrange(RIGHT, buff=0.5).move_to([0, -1.6, 0])
        cmp = Text("three models (Qwen2.5-Instruct)", font_size=22, color=GREY_B).next_to(chips, UP, buff=0.25)
        self.at("compare")
        self.play(FadeIn(cmp), LaggedStart(*[FadeIn(c, shift=0.2 * UP) for c in chips], lag_ratio=0.2), run_time=0.8)
        self.end_section()

    # ------------------------------------------------------------------ 3. exact match
    def exact(self):
        self.section(3)
        self.clear_stage()
        head = Text("check 1 · exact match", font_size=30).to_edge(UP, buff=0.5)
        self.at("simplest")
        self.play(FadeIn(head), run_time=0.4)
        score = Text("1.5B: 1 / 20", font=MONO, font_size=44, color=RED_B).move_to([0, 1.6, 0])
        self.at("scores")
        self.play(FadeIn(score, scale=1.2), run_time=0.5)
        exp = Text("expected:  7:30", font=MONO, font_size=24, color=YELLOW).move_to([-2.5, 0.2, 0])
        ans = answer_box("The bakery opens at 7:30 AM on weekdays.", TEAL_E, 24).move_to([0, -0.8, 0])
        self.at("look")
        self.play(FadeIn(exp), FadeIn(ans, shift=0.2 * UP), run_time=0.5)
        mark = Text("✗ marked wrong", font_size=28, color=RED).next_to(ans, DOWN, buff=0.25)
        self.at("correct")
        self.play(FadeIn(mark), run_time=0.4)
        lesson = Text("the test was wrong, not the model", font_size=30, color=YELLOW).move_to([0, -2.7, 0])
        self.at("wrong")
        self.play(FadeIn(lesson, shift=0.2 * UP), run_time=0.5)
        self.end_section()

    # ------------------------------------------------------------------ 4. contains
    def contains(self):
        self.section(4)
        self.clear_stage()
        head = Text("check 2 · does the answer contain the fact?", font_size=30).to_edge(UP, buff=0.5)
        self.at("contains")
        self.play(FadeIn(head), run_time=0.4)
        bars = score_bars([20, 20, 20], 20, 1.2)
        self.at("every")
        self.play(LaggedStart(*[FadeIn(b) for b in bars], lag_ratio=0.2), run_time=0.8)
        sat = Text("everyone passes: the test can't tell them apart", font_size=28, color=RED_B).move_to([0, -1.4, 0])
        self.at("problem")
        self.play(FadeIn(sat), run_time=0.4)
        easy = Text("too easy", font_size=40, color=YELLOW).move_to([0, -2.4, 0])
        self.at("easy")
        self.play(FadeIn(easy, scale=1.2), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 5. harder set
    def harder(self):
        self.section(5)
        self.clear_stage()
        head = Text("10 harder questions: two facts each", font_size=30).to_edge(UP, buff=0.45)
        self.at("10")
        self.play(FadeIn(head), run_time=0.4)
        qs = VGroup(*[Text(t, font_size=22) for t in ["A cake for 12, delivered on Sunday: total?   → 39",
                                                      "Order a birthday cake on Thursday for Saturday?   → no",
                                                      "Open at 8:30 on a Saturday?   → no"]])
        qs.arrange(DOWN, aligned_edge=LEFT, buff=0.22).move_to([0, 1.7, 0])
        for cue, k in [("total", 0), ("thursday", 1), ("saturday", 2)]:
            self.at(cue)
            self.play(FadeIn(qs[k], shift=0.2 * RIGHT), run_time=0.35)
        bars = score_bars(HARD_AUTO, 10, -0.3)
        self.at("separate")
        self.play(LaggedStart(*[FadeIn(b) for b in bars], lag_ratio=0.25), run_time=0.9)
        self.at("ahead")
        self.play(Indicate(bars[2], color=YELLOW, scale_factor=1.05), run_time=0.6)
        self.end_section()

    # ------------------------------------------------------------------ 6. read the outputs
    def read(self):
        self.section(6)
        self.clear_stage()
        head = Text("read the answers", font_size=32, color=YELLOW).to_edge(UP, buff=0.4)
        self.at("read")
        self.play(FadeIn(head), run_time=0.4)
        cases = [("1.5B: “… delivery fee of 3 euros, making the total cost 42 euros.”", "scored ✓ (contains 39)", RED_B),
                 ("0.5B: “No, Bella's Bakery is closed on Saturdays.”", "scored ✓ (right word, wrong reason)", RED_B),
                 ("1.5B: “Yes, … For an order of 25 euros, delivery would not qualify.”", "scored ✗ (right facts)",
                  ORANGE)]
        rows = VGroup()
        for text, verdict, color in cases:
            rows.add(VGroup(Text(text, font_size=20), Text(verdict, font_size=20, color=color)).arrange(
                DOWN, aligned_edge=LEFT, buff=0.1))
        rows.arrange(DOWN, aligned_edge=LEFT, buff=0.35).move_to([0, 1.0, 0])
        for cue, k in [("42", 0), ("closed", 1), ("yes", 2)]:
            self.at(cue)
            self.play(FadeIn(rows[k], shift=0.2 * RIGHT), run_time=0.45)
        table = VGroup()
        hdr = VGroup(Text("model", font_size=22, color=GREY_B), Text("automatic", font_size=22, color=GREY_B),
                     Text("by hand", font_size=22, color=GREEN_B))
        table.add(hdr)
        for m, c, a, h in zip(MODELS, COLORS, HARD_AUTO, HARD_HAND):
            table.add(VGroup(Text(m, font=MONO, font_size=24, color=c), Text(f"{a} / 10", font=MONO, font_size=24),
                             Text(f"{h} / 10", font=MONO, font_size=24, color=GREEN_B)))
        for i, row in enumerate(table):
            for j, cell in enumerate(row):
                cell.move_to([-2.0 + j * 2.4, -1.4 - i * 0.45, 0])
        self.at("hand")
        self.play(FadeIn(table), run_time=0.6)
        self.end_section()

    # ------------------------------------------------------------------ 7. better checks
    def better(self):
        self.section(7)
        self.clear_stage()
        items = VGroup(*[Text(t, font_size=28) for t in ["ask for a structured answer: a number, yes or no",
                                                          "use several checks",
                                                          "a stronger model grades with a rubric (check the grader!)",
                                                          "always read a sample by hand"]])
        items.arrange(DOWN, aligned_edge=LEFT, buff=0.45).move_to([0.3, 0.2, 0])
        ticks = VGroup(*[Text("✓", font_size=28, color=GREEN).next_to(t, LEFT, buff=0.3) for t in items])
        for i, cue in enumerate(["structured", "several", "rubric", "sample"]):
            self.at(cue)
            self.play(FadeIn(items[i], shift=0.2 * RIGHT), FadeIn(ticks[i]), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 8. noise
    def noise(self):
        self.section(8)
        self.clear_stage()
        head = Text("mind the noise", font_size=34).to_edge(UP, buff=0.6)
        self.at("noise")
        self.play(FadeIn(head), run_time=0.4)
        squares = VGroup(*[Square(0.7, stroke_color=GREY_B, fill_color=GREEN_C, fill_opacity=0.6) for _ in range(10)])
        squares.arrange(RIGHT, buff=0.15).move_to([0, 1.0, 0])
        one = Text("1 question = 10%", font_size=30, color=YELLOW).next_to(squares, DOWN, buff=0.35)
        self.at("ten")
        self.play(FadeIn(squares), run_time=0.4)
        self.play(squares[-1].animate.set_fill(RED, 0.8), FadeIn(one), run_time=0.4)
        runs = Text("with sampling: run each question several times", font_size=28).move_to([0, -1.2, 0])
        self.at("sampling")
        self.play(FadeIn(runs), run_time=0.4)
        small = Text("small differences on small test sets often mean nothing", font_size=28, color=GREY_A)
        small.move_to([0, -2.2, 0])
        self.at("nothing")
        self.play(FadeIn(small), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 9. code
    def code(self):
        self.section(9)
        self.clear_stage()
        code, hl = code_panel(CODE, font_size=24)
        code.move_to([0, 0.3, 0])
        self.at("loop")
        self.play(FadeIn(code, shift=0.2 * UP), run_time=0.5)
        hl.match_y(code.line_numbers[1])
        self.play(Create(hl), run_time=0.3)
        self.at("ask")
        self.play(highlight(hl, code, 2), run_time=0.4)
        self.at("check")
        self.play(highlight(hl, code, 3), run_time=0.4)
        self.at("count")
        self.play(highlight(hl, code, 4), run_time=0.4)
        self.at("failures")
        self.play(highlight(hl, code, 6), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 10. keep it
    def keep(self):
        self.section(10)
        self.clear_stage()
        center = labeled_box("your test set", YELLOW, width=3.0, height=1.0).move_to([0, 0.2, 0])
        around = VGroup(*[labeled_box(t, c, width=2.6, height=0.8, font_size=22) for t, c in
                          [("new prompt", BLUE_C), ("new model", TEAL_C), ("quantized", GOLD)]])
        for box, ang in zip(around, [PI / 2 + 0.6, PI / 2 - 0.6, -PI / 2]):
            box.move_to([3.8 * np.cos(ang), 0.2 + 2.2 * np.sin(ang), 0])
        arrows = VGroup(*[Arrow(b.get_center(), center.get_center(), buff=0.75, color=GREY_B) for b in around])
        self.at("keep")
        self.play(FadeIn(center), run_time=0.4)
        self.at("change")
        self.play(LaggedStart(*[AnimationGroup(FadeIn(b), GrowArrow(a)) for b, a in zip(around, arrows)],
                              lag_ratio=0.3), run_time=1.0)
        line = Text("better, not just different", font_size=32, color=GREEN_B).to_edge(DOWN, buff=0.5)
        self.at("different")
        self.play(FadeIn(line, shift=0.2 * UP), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 11. outro
    def outro(self):
        self.section(11)
        self.clear_stage(run_time=0.4)
        card = next_up_card(NEXT)
        self.at("next")
        self.play(FadeIn(card, shift=0.2 * UP), run_time=0.6)
        self.end_section()
        finish(self, hold=1.5)
