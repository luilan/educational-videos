"""Reinforcement Learning for LLMs, episode 7 — Rewards From Preferences.

Render from the repo root:  ./render.sh rl-llms r07
Numbers come from code/r07_rewards_from_preferences (answers.json, results.json).
"""
import json
import math

from manim import *

from common import MONO, code_panel, finish, highlight, next_up_card
from intro import play_token_intro
from r07_script import NEXT, TAGLINE, TITLE
from rl_common import BAD, GOOD, MODEL_C, REWARD_C, study_footer
from voiced_scene import VoicedScene

SERIES_LABEL = "REINFORCEMENT LEARNING FOR LLMs  ·  EPISODE 7"
CODE_DIR = "code/r07_rewards_from_preferences"
R = json.load(open(f"{CODE_DIR}/results.json"))
A = json.load(open(f"{CODE_DIR}/answers.json"))
RM_C = TEAL_C
CODE = """r_good = reward_model(preferred)
r_bad = reward_model(other)
loss = -F.logsigmoid(r_good - r_bad).mean()     # Bradley-Terry"""


def pct(x):
    return f"{100 * x:.0f}%"


def answer_card(text, width=5.6, font_size=18, color=GREY_A, title=None, tcolor=GREY_B):
    t = Text(text, font_size=font_size, color=color, line_spacing=0.85)
    if t.width > width - 0.4:
        t.scale_to_fit_width(width - 0.4)
    box = RoundedRectangle(corner_radius=0.12, width=width, height=t.height + 0.45, stroke_color=GREY_C)
    g = VGroup(box, t.move_to(box))
    if title:
        g.add(Text(title, font_size=18, color=tcolor).next_to(box, UP, buff=0.08).align_to(box, LEFT))
    return g


def wrap(s, n):
    words, lines, cur = s.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > n:
            lines.append(cur)
            cur = w
        else:
            cur = (cur + " " + w).strip()
    return "\n".join(lines + [cur])


class PreferencesVideo(VoicedScene):
    VIDEO = "r07"

    def construct(self):
        play_token_intro(self, TITLE, 7, TAGLINE, label=SERIES_LABEL)
        self.hook()        # 1
        self.data()        # 2
        self.people()      # 3
        self.bt()          # 4
        self.model()       # 5
        self.results()     # 6
        self.learned()     # 7
        self.blind()       # 8
        self.lesson()      # 9
        self.code()        # 10
        self.outro()       # 11

    def clear_stage(self, *keep, run_time=0.6):
        self.play(*[FadeOut(m) for m in self.mobjects if m not in keep], run_time=run_time)

    # ------------------------------------------------------------------ 1. hook
    def hook(self):
        self.section(1)
        p = Text("a program grades: 47+38 = 85. ✓", font=MONO, font_size=24, color=GOOD).move_to([0, 2.5, 0])
        self.at("program")
        self.play(FadeIn(p), run_time=0.4)
        q = Text("“Explain why the sky is blue to a child.”  → no program can grade this", font_size=24, color=GREY_A).move_to([0, 1.5, 0])
        self.at("chatbot")
        self.play(FadeIn(q), run_time=0.5)
        a = answer_card("answer A", width=3.0).move_to([-3.8, -0.2, 0])
        b = answer_card("answer B", width=3.0).move_to([3.8, -0.2, 0])
        w = Text("which is better?", font_size=28, color=REWARD_C).move_to([0, -0.2, 0])
        self.at("people")
        self.play(FadeIn(a), FadeIn(b), run_time=0.4)
        self.at("better")
        self.play(FadeIn(w), run_time=0.3)
        rm = VGroup(RoundedRectangle(corner_radius=0.15, width=7.2, height=0.9, stroke_color=RM_C, fill_color=RM_C, fill_opacity=0.12),
                    Text("reward model: learns to predict the choices", font_size=22, color=RM_C))
        rm[1].move_to(rm[0])
        rm.move_to([0, -2.2, 0])
        self.at("reward")
        self.play(FadeIn(rm), run_time=0.5)
        self.end_section()

    # ------------------------------------------------------------------ 2. our data
    def data(self):
        self.section(2)
        self.clear_stage()
        h = Text("Qwen2.5-0.5B-Instruct · 400 word problems · 4 answers each", font_size=26).to_edge(UP, buff=0.6)
        self.at("compare")
        self.play(FadeIn(h), run_time=0.4)
        q = A[0]
        qt = Text(q["question"], font_size=20, color=REWARD_C).next_to(h, DOWN, buff=0.35)
        self.at("400")
        self.play(FadeIn(qt), run_time=0.4)
        flat = [s for o in A for s in o["samples"]]
        longs = [s for s in flat if s["style"] == "long"]
        shorts = [s for s in flat if s["style"] == "short"]
        ls = [s for s in q["samples"] if s["style"] == "long"][0]
        ss = [s for s in q["samples"] if s["style"] == "short"][0]
        lc = answer_card(wrap(ls["text"][:230] + " …", 52), width=6.2, font_size=16, title=f"full explanation: {pct(sum(s['correct'] for s in longs) / len(longs))} right, {sum(s['words'] for s in longs) / len(longs):.0f} words on average", tcolor=GOOD).move_to([-3.3, -1.0, 0])
        sc = answer_card(wrap(ss["text"], 40), width=5.6, font_size=16, title=f"one sentence: {pct(sum(s['correct'] for s in shorts) / len(shorts))} right, {sum(s['words'] for s in shorts) / len(shorts):.0f} words", tcolor=BAD).move_to([3.5, -0.4, 0])
        self.at("explanation")
        self.play(FadeIn(lc), run_time=0.5)
        self.at("sentence")
        self.play(FadeIn(sc), run_time=0.5)
        n = Text(f"{len(flat):,} answers", font_size=30, color=REWARD_C).move_to([3.5, -2.4, 0])
        self.at("1600")
        self.play(FadeIn(n), run_time=0.3)
        self.end_section()

    # ------------------------------------------------------------------ 3. our people
    def people(self):
        self.section(3)
        self.clear_stage()
        h = Text("our “people” are a script: we know their rule", font_size=30).to_edge(UP, buff=0.8)
        self.at("script")
        self.play(FadeIn(h), run_time=0.4)
        rules = VGroup(Text("1.  the right answer beats a wrong one", font_size=28, color=GOOD),
                       Text("2.  both right, or both wrong → the shorter one", font_size=28, color=REWARD_C),
                       Text(f"3.  {int(R['noise'] * 100)}% of labels flipped, like real disagreement", font_size=28, color=GREY_A)
                       ).arrange(DOWN, aligned_edge=LEFT, buff=0.45).move_to([0, -0.2, 0])
        for r, cue in zip(rules, ("right", "shorter", "10")):
            self.at(cue)
            self.play(FadeIn(r, shift=0.1 * RIGHT), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 4. Bradley-Terry
    def bt(self):
        self.section(4)
        self.clear_stage()
        s = Text("each answer gets a score r", font_size=28, color=RM_C).to_edge(UP, buff=0.8)
        self.at("score")
        self.play(FadeIn(s), run_time=0.4)
        f = Text("P(A preferred over B) = sigmoid( r(A) − r(B) )", font=MONO, font_size=26, color=WHITE).move_to([0, 1.6, 0])
        ax = Axes(x_range=[-4, 4, 2], y_range=[0, 1, 0.5], x_length=6, y_length=2.6, tips=False, axis_config={"color": GREY_C}).move_to([0, -0.9, 0])
        curve = ax.plot(lambda x: 1 / (1 + math.exp(-x)), x_range=[-4, 4], color=REWARD_C, stroke_width=4)
        xl = Text("r(A) − r(B)", font_size=18, color=GREY_B).next_to(ax, DOWN, buff=0.15)
        yl = VGroup(*[Text(t, font_size=16, color=GREY_B).next_to(ax.c2p(-4, v), LEFT, buff=0.1) for t, v in (("0", 0), ("½", 0.5), ("1", 1))])
        self.at("sigmoid")
        self.play(FadeIn(f), FadeIn(ax), FadeIn(xl), FadeIn(yl), Create(curve), run_time=0.8)
        d = Text("only differences matter (like a baseline, episode 3)", font_size=22, color=GREY_A).move_to([0, -3.0, 0])
        self.at("differences")
        self.play(FadeIn(d), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 5. the model
    def model(self):
        self.section(5)
        self.clear_stage()
        inp = Text("question + answer", font_size=22, color=GREY_A).move_to([-5.4, 1.2, 0])
        q = VGroup(RoundedRectangle(corner_radius=0.15, width=3.4, height=1.3, stroke_color=MODEL_C, fill_color=MODEL_C, fill_opacity=0.1),
                   VGroup(Text("Qwen2.5-0.5B", font_size=24, color=MODEL_C), Text("frozen", font_size=18, color=GREY_B)).arrange(DOWN, buff=0.1))
        q[1].move_to(q[0])
        q.move_to([-1.6, 1.2, 0])
        lin = VGroup(RoundedRectangle(corner_radius=0.12, width=2.6, height=1.0, stroke_color=RM_C, fill_color=RM_C, fill_opacity=0.15),
                     VGroup(Text("one linear layer", font_size=20, color=RM_C), Text("897 numbers, trained", font_size=16, color=GREY_B)).arrange(DOWN, buff=0.08))
        lin[1].move_to(lin[0])
        lin.move_to([2.2, 1.2, 0])
        out = Text("score r", font_size=26, color=REWARD_C).move_to([5.2, 1.2, 0])
        arr = VGroup(Arrow(inp.get_right(), q.get_left(), buff=0.1), Arrow(q.get_right(), lin.get_left(), buff=0.1), Arrow(lin.get_right(), out.get_left(), buff=0.1))
        self.at("frozen")
        self.play(FadeIn(inp), FadeIn(q), GrowArrow(arr[0]), run_time=0.5)
        self.at("897")
        self.play(GrowArrow(arr[1]), FadeIn(lin), GrowArrow(arr[2]), FadeIn(out), run_time=0.5)
        tr = Text(f"trained on {R['train_pairs']:,} pairs from 300 problems", font_size=24, color=GREY_A).move_to([0, -0.9, 0])
        te = Text(f"tested on {R['test_pairs']} pairs from 100 problems it never saw", font_size=24, color=GOOD).move_to([0, -1.7, 0])
        self.at("300")
        self.play(FadeIn(tr), run_time=0.4)
        self.at("100")
        self.play(FadeIn(te), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 6. results
    def results(self):
        self.section(6)
        self.clear_stage()
        h = Text("held-out pairs: how often it agrees", font_size=28).to_edge(UP, buff=0.6)
        self.play(FadeIn(h), run_time=0.3)
        te = R["test"]
        items = [("with the labels", te["vs_labels"], RM_C, "73"),
                 ("one right, one wrong", te["correctness_differs"]["vs_true_rule"], GOOD, "wrong"),
                 ("length should decide", te["same_correctness"]["vs_true_rule"], BAD, "55")]
        base_y, unit = -2.2, 4.2
        bars = VGroup()
        for i, (lab, v, col, _) in enumerate(items):
            x = -4.0 + 4.0 * i
            b = Rectangle(width=1.5, height=v * unit, stroke_width=0, fill_color=col, fill_opacity=0.8).move_to([x, base_y, 0], aligned_edge=DOWN)
            bars.add(VGroup(b, Text(pct(v), font_size=30, color=col).next_to(b, UP, buff=0.1), Text(lab, font_size=20, color=col).next_to([x, base_y, 0], DOWN, buff=0.2)))
        base = Line([-6, base_y, 0], [6, base_y, 0], color=GREY_C)
        ceil = DashedLine([-5.2, base_y + 0.9 * unit, 0], [-2.8, base_y + 0.9 * unit, 0], color=GREY_B)
        cl = Text("best possible ≈ 90%\n(10% of labels flipped)", font_size=16, color=GREY_B, line_spacing=0.85).next_to(ceil, UP, buff=0.08)
        coin = DashedLine([2.8, base_y + 0.5 * unit, 0], [5.2, base_y + 0.5 * unit, 0], color=GREY_B)
        col_ = Text("coin flip 50%", font_size=16, color=GREY_B).next_to(coin, RIGHT, buff=0.1)
        self.play(Create(base), run_time=0.2)
        self.at("73")
        self.play(GrowFromEdge(bars[0][0], DOWN), FadeIn(bars[0][1:]), run_time=0.5)
        self.at("possible")
        self.play(Create(ceil), FadeIn(cl), run_time=0.4)
        self.at("wrong")
        self.play(GrowFromEdge(bars[1][0], DOWN), FadeIn(bars[1][1:]), run_time=0.5)
        self.at("55")
        self.play(GrowFromEdge(bars[2][0], DOWN), FadeIn(bars[2][1:]), run_time=0.5)
        self.at("coin")
        self.play(Create(coin), FadeIn(col_), run_time=0.3)
        self.end_section()

    # ------------------------------------------------------------------ 7. what it learned
    def learned(self):
        self.section(7)
        self.clear_stage()
        h = Text("average score on held-out answers", font_size=28).to_edge(UP, buff=0.7)
        self.at("average")
        self.play(FadeIn(h), run_time=0.4)
        m = R["mean_reward"]
        cols = ("short (≤ 25 words)", "long")
        rows = (("right", "right_short", "right_long", GOOD), ("wrong", "wrong_short", "wrong_long", BAD))
        xs, ys = (-0.6, 3.0), (1.0, -0.4)
        head = VGroup(*[Text(c, font_size=22, color=GREY_B).move_to([x, 2.0, 0]) for c, x in zip(cols, xs)])
        cells = VGroup()
        for (lab, k1, k2, col), y in zip(rows, ys):
            cells.add(VGroup(Text(lab, font_size=26, color=col).move_to([-4.0, y, 0]),
                             Text(f"{m[k1]['reward']:+.2f}", font_size=30, color=col).move_to([xs[0], y, 0]),
                             Text(f"{m[k2]['reward']:+.2f}", font_size=30, color=col).move_to([xs[1], y, 0])))
        self.play(FadeIn(head), run_time=0.3)
        self.at("right")
        self.play(FadeIn(cells), run_time=0.5)
        box = SurroundingRectangle(cells[1][1:], color=REWARD_C, buff=0.2)
        n = Text("among wrong answers it prefers the long ones: the opposite of the rule", font_size=22, color=REWARD_C).move_to([0, -1.6, 0])
        self.at("long")
        self.play(Create(box), FadeIn(n), run_time=0.5)
        flat = [s for o in A for s in o["samples"]]
        lr = [s for s in flat if s["style"] == "long"]
        why = Text(f"in the data, long answers were {pct(sum(s['correct'] for s in lr) / len(lr))} right → it learned “long looks right”", font_size=22, color=GREY_A).move_to([0, -2.5, 0])
        self.at("nearly")
        self.play(FadeIn(why), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 8. a blind spot
    def blind(self):
        self.section(8)
        self.clear_stage()
        e = [x for x in R["examples"] if "61 books" in x["wrong"]][0]
        q = Text(e["question"], font_size=22, color=GREY_A).to_edge(UP, buff=0.7)
        self.at("spot")
        self.play(FadeIn(q), run_time=0.4)
        w = answer_card(wrap(e["wrong"], 44), width=5.8, font_size=18, title=f"wrong (right answer: {e['answer']})", tcolor=BAD).move_to([-3.2, 0.2, 0])
        r = answer_card(wrap(e["right"][:200] + " …", 46), width=5.8, font_size=16, title=f"correct, {e['right_words']} words", tcolor=GOOD).move_to([3.2, 0.2, 0])
        self.at("61")
        self.play(FadeIn(w), run_time=0.4)
        self.at("63")
        self.play(FadeIn(r), run_time=0.4)
        sw = Text(f"score {e['wrong_reward']:+.2f}", font_size=30, color=BAD).next_to(w, DOWN, buff=0.4)
        sr = Text(f"score {e['right_reward']:+.2f}", font_size=30, color=GOOD).next_to(r, DOWN, buff=0.4)
        self.at("higher")
        self.play(FadeIn(sw), FadeIn(sr), run_time=0.5)
        self.end_section()

    # ------------------------------------------------------------------ 9. lesson
    def lesson(self):
        self.section(9)
        self.clear_stage()
        a = Text("a reward model = a learned guess of what people prefer", font_size=28, color=RM_C).move_to([0, 1.6, 0])
        self.at("guess")
        self.play(FadeIn(a), run_time=0.4)
        b = Text("with their noise, and the shortcuts in the data", font_size=26, color=GREY_A).next_to(a, DOWN, buff=0.35)
        self.at("noise")
        self.play(FadeIn(b), run_time=0.4)
        c = Text("fine for ranking answers…", font_size=26, color=GOOD).move_to([0, -0.6, 0])
        self.at("ranking")
        self.play(FadeIn(c), run_time=0.3)
        d = Text("…but RL will push against it as hard as it can", font_size=28, color=BAD).next_to(c, DOWN, buff=0.35)
        self.at("push")
        self.play(FadeIn(d), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 10. code
    def code(self):
        self.section(10)
        self.clear_stage()
        code, hl = code_panel(CODE, font_size=22)
        code.move_to([0, 0.3, 0])
        self.play(FadeIn(code, shift=0.2 * UP), run_time=0.4)
        self.at("loss")
        self.play(Create(hl), run_time=0.3)
        self.play(highlight(hl, code, 2), run_time=0.3)
        self.at("preferred")
        self.play(highlight(hl, code, 0), run_time=0.3)
        self.at("other")
        self.play(highlight(hl, code, 1), run_time=0.3)
        self.end_section()

    # ------------------------------------------------------------------ 11. outro
    def outro(self):
        self.section(11)
        self.clear_stage(run_time=0.4)
        card_ = next_up_card(NEXT)
        self.at("next")
        self.play(FadeIn(card_, shift=0.2 * UP), FadeIn(study_footer()), run_time=0.6)
        self.end_section()
        finish(self, hold=1.5)
