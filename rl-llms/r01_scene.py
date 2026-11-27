"""Reinforcement Learning for LLMs, episode 1 — Learning From a Grade.

Render from the repo root:  ./render.sh rl-llms r01
Numbers come from code/r01_learning_from_a_grade (grade.py → results.json).
"""
import json

from manim import *

from common import MONO, code_panel, finish, highlight, next_up_card, token
from intro import play_token_intro
from r01_script import NEXT, TAGLINE, TITLE
from rl_common import GOOD, BAD, MODEL_C, REWARD_C, char_row, grade_badge, study_footer
from voiced_scene import VoicedScene

SERIES_LABEL = "REINFORCEMENT LEARNING FOR LLMs  ·  EPISODE 1"
CODE_DIR = "code/r01_learning_from_a_grade"
R = json.load(open(f"{CODE_DIR}/results.json"))
CODE = """def reward(a, b, answer):
    return 1.0 if answer == f"{a + b}." else 0.0

for t in range(4):                    # up to 4 characters
    probs = softmax(model(idx)[:, -1])
    next_char = multinomial(probs, 1)
    idx = cat([idx, next_char], dim=1)"""


def pct(x):
    return f"{100 * x:.0f}%"


class GradeVideo(VoicedScene):
    VIDEO = "r01"

    def construct(self):
        play_token_intro(self, TITLE, 1, TAGLINE, label=SERIES_LABEL)
        self.hook()        # 1
        self.why()         # 2
        self.task()        # 3
        self.pretrain()    # 4
        self.policy()      # 5
        self.tries()       # 6
        self.spread()      # 7
        self.idea()        # 8
        self.code()        # 9
        self.outro()       # 10

    def clear_stage(self, *keep, run_time=0.6):
        self.play(*[FadeOut(m) for m in self.mobjects if m not in keep], run_time=run_time)

    # ------------------------------------------------------------------ 1. hook
    def hook(self):
        self.section(1)
        left = VGroup(Text("copy the answer", font_size=30, color=GREY_A),
                      char_row("47+38=85.", font_size=30),
                      Text("the model is shown it, and imitates", font_size=20, color=GREY_B)).arrange(DOWN, buff=0.4)
        left.move_to([-3.4, 0.4, 0])
        self.at("show")
        self.play(FadeIn(left[0]), FadeIn(left[1], lag_ratio=0.1), run_time=0.8)
        self.at("copy")
        self.play(FadeIn(left[2]), run_time=0.4)
        try1 = VGroup(char_row("47+38=75.", font_size=30), grade_badge(0)).arrange(RIGHT, buff=0.3)
        try2 = VGroup(char_row("47+38=85.", font_size=30), grade_badge(1)).arrange(RIGHT, buff=0.3)
        right = VGroup(Text("try, and get a grade", font_size=30, color=GREY_A), try1, try2,
                       Text("only the score comes back", font_size=20, color=GREY_B)).arrange(DOWN, buff=0.4)
        right.move_to([3.3, 0.4, 0])
        divider = DashedLine([0, 2.4, 0], [0, -1.8, 0], color=GREY_D)
        self.at("try")
        self.play(FadeIn(right[0]), Create(divider), FadeIn(try1), run_time=0.7)
        self.at("did")
        self.play(FadeIn(try2), FadeIn(right[3]), run_time=0.6)
        name = Text("reinforcement learning", font_size=34, color=REWARD_C).move_to([3.3, -2.6, 0])
        self.at("reinforcement")
        self.play(FadeIn(name, shift=0.2 * UP), run_time=0.5)
        sub = Text("built from scratch, the way language models use it", font_size=22, color=GREY_B).to_edge(DOWN, buff=0.35)
        self.at("scratch")
        self.play(FadeIn(sub), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 2. why a grade
    def why(self):
        self.section(2)
        self.clear_stage()
        h = Text("checking is often easier than answering", font_size=32).to_edge(UP, buff=0.7)
        self.at("checking")
        self.play(FadeIn(h), run_time=0.5)
        cards = [("code", "run the tests:\npass or fail", "program"),
                 ("math", "compare the\nfinal number", "grader"),
                 ("new skills", "reward what\nnobody wrote down", "nobody")]
        group = VGroup()
        for title, body, _ in cards:
            box = RoundedRectangle(width=3.7, height=2.3, corner_radius=0.15, color=REWARD_C, fill_opacity=0.08)
            txt = VGroup(Text(title, font_size=28, color=REWARD_C),
                         Text(body, font_size=22, line_spacing=0.9)).arrange(DOWN, buff=0.35).move_to(box)
            group.add(VGroup(box, txt))
        group.arrange(RIGHT, buff=0.35).move_to([0, -0.3, 0])
        for card, (_, _, cue) in zip(group, cards):
            self.at(cue)
            self.play(FadeIn(card, shift=0.2 * UP), run_time=0.5)
        self.end_section()

    # ------------------------------------------------------------------ 3. task and checker
    def task(self):
        self.section(3)
        self.clear_stage()
        h = Text("the task for this series", font_size=30, color=GREY_A).to_edge(UP, buff=0.6)
        row = char_row("47+38=85.", font_size=40).move_to([0, 1.6, 0])
        self.at("adding")
        self.play(FadeIn(h), FadeIn(row[:6], lag_ratio=0.1), run_time=0.7)
        self.at("character")
        self.play(LaggedStart(*[FadeIn(t, shift=0.15 * RIGHT) for t in row[6:]], lag_ratio=0.35), run_time=1.0)
        self.at("period")
        self.play(Indicate(row[-1], color=REWARD_C), run_time=0.6)
        chk = VGroup(Text("checker:", font_size=26, color=REWARD_C),
                     Text("reward = 1 if the answer is exactly right, else 0", font=MONO, font_size=22)).arrange(RIGHT, buff=0.3)
        chk.move_to([0, 0.1, 0])
        self.at("checker")
        self.play(FadeIn(chk), run_time=0.5)
        ex = VGroup(VGroup(char_row("85.", font_size=26), grade_badge(1)).arrange(RIGHT, buff=0.25),
                    VGroup(char_row("75.", font_size=26), grade_badge(0)).arrange(RIGHT, buff=0.25),
                    VGroup(char_row("8.", font_size=26), grade_badge(0)).arrange(RIGHT, buff=0.25)).arrange(RIGHT, buff=0.9)
        ex.move_to([0, -1.1, 0])
        self.at("one")
        self.play(FadeIn(ex[0]), run_time=0.4)
        self.at("zero")
        self.play(FadeIn(ex[1:]), run_time=0.4)
        m = Text(f"the model: a tiny GPT · {R['params']:,} parameters", font_size=24, color=MODEL_C).to_edge(DOWN, buff=0.6)
        self.at("tiny")
        self.play(FadeIn(m), run_time=0.5)
        self.end_section()

    # ------------------------------------------------------------------ 4. pretraining
    def pretrain(self):
        self.section(4)
        self.clear_stage()
        lines = ["12+45=57.", "83+19=102.", "36+36=72.", "70+28=98.", "…"]
        stack = VGroup(*[Text(s, font=MONO, font_size=24, color=GREY_A) for s in lines]).arrange(DOWN, aligned_edge=LEFT, buff=0.18)
        lab = Text(f"{R['worked_examples']} worked examples", font_size=24, color=GREY_A).next_to(stack, UP, buff=0.3)
        g = VGroup(lab, stack).move_to([-4.2, 0.3, 0])
        self.at("pre")
        self.play(FadeIn(g, lag_ratio=0.1), run_time=0.8)
        axis_y = -2.2
        bars = VGroup()
        specs = [("the 500 it trained on", R["acc_worked_greedy"], GOOD, "heart"),
                 ("1,000 new questions", R["acc_test_greedy"], BAD, "seen")]
        for i, (name, v, col, _) in enumerate(specs):
            x = 0.9 + 3.6 * i
            bar = Rectangle(width=1.3, height=max(0.04, 4.0 * v), stroke_width=0, fill_color=col, fill_opacity=0.85)
            bar.move_to([x, axis_y, 0], aligned_edge=DOWN)
            val = Text(pct(v), font_size=30, color=col).next_to(bar, UP, buff=0.15)
            nm = Text(name, font_size=20, color=GREY_B).next_to([x, axis_y, 0], DOWN, buff=0.2)
            bars.add(VGroup(bar, val, nm))
        base = Line([-0.4, axis_y, 0], [6.2, axis_y, 0], color=GREY_C)
        acc = Text("right answers (first choice)", font_size=22, color=GREY_B).move_to([2.2, 2.6, 0])
        self.play(Create(base), FadeIn(acc), run_time=0.4)
        for b, (_, _, _, cue) in zip(bars, specs):
            self.at(cue)
            self.play(GrowFromEdge(b[0], DOWN), FadeIn(b[1:]), run_time=0.6)
        self.at("14")
        self.play(Indicate(bars[1][1], color=BAD), run_time=0.6)
        self.end_section()

    # ------------------------------------------------------------------ 5. the policy
    def policy(self):
        self.section(5)
        self.clear_stage()
        terms = [("policy", "the model", "policy"), ("action", "one character", "action"),
                 ("episode", "one whole answer", "episode"), ("reward", "1 or 0, at the end", "reward")]
        tab = VGroup(*[VGroup(Text(a, font_size=26, color=REWARD_C), Text("= " + b, font_size=24))
                       .arrange(RIGHT, buff=0.25) for a, b, _ in terms]).arrange(DOWN, aligned_edge=LEFT, buff=0.3)
        tab.move_to([-3.9, 1.2, 0])
        for row, (_, _, cue) in zip(tab, terms):
            self.at(cue)
            self.play(FadeIn(row, shift=0.15 * RIGHT), run_time=0.4)
        ex = R["policy_example"]
        q = char_row(ex["question"], font_size=34).move_to([2.6, 2.3, 0])
        self.at("47")
        self.play(FadeIn(q), run_time=0.5)
        first = ex["steps"][0]
        bars = VGroup()
        for i, (ch, p) in enumerate(first[:2]):
            y = 1.0 - 0.75 * i
            lab = Text(f"'{ch}'", font=MONO, font_size=26).move_to([0.9, y, 0])
            bar = Rectangle(width=max(0.05, 4.2 * p), height=0.42, stroke_width=0,
                            fill_color=BAD if ch == "7" else GOOD, fill_opacity=0.85)
            bar.move_to([1.5, y, 0], aligned_edge=LEFT)
            val = Text(pct(p), font_size=22).next_to(bar, RIGHT, buff=0.15)
            bars.add(VGroup(lab, bar, val))
        cap = Text("first character of the answer", font_size=20, color=GREY_B).next_to(bars, UP, buff=0.25).align_to(bars, LEFT)
        self.at("83")
        self.play(FadeIn(cap), LaggedStart(*[GrowFromEdge(b[1], LEFT) for b in bars], lag_ratio=0.3),
                  FadeIn(VGroup(*[VGroup(b[0], b[2]) for b in bars])), run_time=0.9)
        note = Text("47 + 38 = 85: the 8 needs the carried 1", font_size=22, color=GREY_B).move_to([2.6, -1.4, 0])
        self.at("carry")
        self.play(FadeIn(note), Indicate(bars[0], color=BAD), run_time=0.8)
        self.end_section()

    # ------------------------------------------------------------------ 6. sample and grade
    def tries(self):
        self.section(6)
        self.clear_stage()
        d = R["detail"]
        h = VGroup(Text("64 tries at", font_size=30), char_row(d["question"], font_size=30)).arrange(RIGHT, buff=0.3).to_edge(UP, buff=0.7)
        self.at("64")
        self.play(FadeIn(h), run_time=0.5)
        rows = VGroup()
        cues = []
        for i, (ans, n, r) in enumerate(d["tally"]):
            y = 1.2 - 1.05 * i
            a = char_row(ans, font_size=28).move_to([-4.0, y, 0])
            bar = Rectangle(width=0.16 * n, height=0.5, stroke_width=0, fill_color=GOOD if r else GREY_C, fill_opacity=0.85)
            bar.move_to([-2.6, y, 0], aligned_edge=LEFT)
            cnt = Text(f"× {n}", font_size=24).next_to(bar, RIGHT, buff=0.2)
            badge = grade_badge(int(r)).move_to([4.6, y, 0])
            rows.add(VGroup(a, bar, cnt, badge))
            cues.append(ans.rstrip(".")[:2])
        for row, cue in zip(rows, cues):
            self.at(cue)
            self.play(FadeIn(row[0]), GrowFromEdge(row[1], LEFT), FadeIn(row[2]), run_time=0.5)
        self.at("grade")
        self.play(LaggedStart(*[FadeIn(r[3], scale=1.3) for r in rows], lag_ratio=0.25), run_time=0.8)
        self.end_section()

    # ------------------------------------------------------------------ 7. the spread
    def spread(self):
        self.section(7)
        self.clear_stage()
        sp = R["spread"]
        labels = list(sp)
        h = Text("1,000 new questions × 64 tries each: how often is each question right?", font_size=24).to_edge(UP, buff=0.5)
        self.at("thousand")
        self.play(FadeIn(h), run_time=0.5)
        axis_y, scale = -2.3, 4.2 / max(sp.values())
        bars = VGroup()
        for i, lab in enumerate(labels):
            x = -5.0 + 1.75 * i
            n = sp[lab]
            col = BAD if lab == "0" else (GOOD if lab == "100%" else REWARD_C)
            bar = Rectangle(width=1.2, height=max(0.04, n * scale), stroke_width=0, fill_color=col, fill_opacity=0.85)
            bar.move_to([x, axis_y, 0], aligned_edge=DOWN)
            val = Text(str(n), font_size=22).next_to(bar, UP, buff=0.12)
            nm = Text(lab if lab != "0" else "never", font_size=18, color=GREY_B).next_to([x, axis_y, 0], DOWN, buff=0.2)
            bars.add(VGroup(bar, val, nm))
        base = Line([-5.9, axis_y, 0], [4.6, axis_y, 0], color=GREY_C)
        xl = Text("share of the 64 tries that are right", font_size=18, color=GREY_B).next_to(base, DOWN, buff=0.6)
        self.play(Create(base), FadeIn(xl), LaggedStart(*[GrowFromEdge(b[0], DOWN) for b in bars], lag_ratio=0.1),
                  FadeIn(VGroup(*[VGroup(b[1], b[2]) for b in bars])), run_time=1.2)
        ov = Text(f"{pct(R['pass1_sampled'])} of all 64,000 answers are right", font_size=24, color=REWARD_C).move_to([3.0, 1.8, 0])
        self.at("overall")
        self.play(FadeIn(ov), run_time=0.4)
        self.at("644")
        self.play(Indicate(bars[0], color=BAD), run_time=0.7)
        k = Text(f"{R['greedy_wrong_but_some_sample_right']} questions: first choice wrong,\nbut some tries right",
                 font_size=24, color=GOOD, line_spacing=0.9).move_to([3.0, 0.8, 0])
        self.at("217")
        self.play(FadeIn(k), Indicate(VGroup(*bars[1:5]), color=GOOD), run_time=0.9)
        self.end_section()

    # ------------------------------------------------------------------ 8. the idea
    def idea(self):
        self.section(8)
        self.clear_stage()
        d = R["detail"]
        q = char_row(d["question"], font_size=30).move_to([-3.8, 2.3, 0])
        counts = [n for _, n, _ in d["tally"]]
        rows = VGroup()
        for i, (ans, n, r) in enumerate(d["tally"]):
            y = 1.1 - 0.9 * i
            a = char_row(ans, font_size=26).move_to([-5.0, y, 0])
            bar = Rectangle(width=0.12 * n, height=0.45, stroke_width=0, fill_color=GOOD if r else GREY_C, fill_opacity=0.85)
            bar.move_to([-3.9, y, 0], aligned_edge=LEFT)
            rows.add(VGroup(a, bar))
        self.at("works")
        self.play(FadeIn(q), FadeIn(rows), run_time=0.6)
        rule = VGroup(Text("scored 1  →  make it more likely", font_size=26, color=GOOD),
                      Text("scored 0  →  make it less likely", font_size=26, color=BAD),
                      Text("…and repeat", font_size=26, color=GREY_A)).arrange(DOWN, aligned_edge=LEFT, buff=0.35)
        rule.move_to([3.2, 1.0, 0])
        self.at("more")
        self.play(FadeIn(rule[0]), rows[1][1].animate.stretch_to_fit_width(0.12 * 48).align_to(rows[1][1], LEFT), run_time=0.8)
        self.at("less")
        self.play(FadeIn(rule[1]), rows[0][1].animate.stretch_to_fit_width(0.12 * 12).align_to(rows[0][1], LEFT),
                  rows[2][1].animate.stretch_to_fit_width(0.12 * 3).align_to(rows[2][1], LEFT), run_time=0.8)
        self.at("repeat")
        self.play(FadeIn(rule[2]), run_time=0.4)
        illus = Text("(illustration: the real training run is in episode 2)", font_size=18, color=GREY_B).move_to([-3.2, -1.7, 0])
        self.play(FadeIn(illus), run_time=0.3)
        key = Text("no answer key, only grades", font_size=32, color=REWARD_C).to_edge(DOWN, buff=0.6)
        self.at("key")
        self.play(FadeIn(key, shift=0.2 * UP), run_time=0.5)
        self.end_section()

    # ------------------------------------------------------------------ 9. code
    def code(self):
        self.section(9)
        self.clear_stage()
        code, hl = code_panel(CODE, font_size=22)
        code.move_to([0, 0.2, 0])
        self.play(FadeIn(code, shift=0.2 * UP), run_time=0.5)
        self.at("grade")
        self.play(Create(hl), highlight(hl, code, 1), run_time=0.4)
        self.at("sampling")
        self.play(highlight(hl, code, 3), run_time=0.3)
        self.at("characters")
        self.play(highlight(hl, code, 5), run_time=0.3)
        self.at("time")
        self.play(highlight(hl, code, 6), run_time=0.3)
        self.end_section()

    # ------------------------------------------------------------------ 10. outro
    def outro(self):
        self.section(10)
        self.clear_stage(run_time=0.4)
        card_ = next_up_card(NEXT)
        self.at("next")
        self.play(FadeIn(card_, shift=0.2 * UP), FadeIn(study_footer()), run_time=0.6)
        self.end_section()
        finish(self, hold=1.5)
