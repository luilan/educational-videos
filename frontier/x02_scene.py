"""The Frontier: Self-Improvement and Memory, episode 2 — Weight-Space Programming.

Render from the repo root:  ./render.sh frontier x02
Numbers come from code/x02_weight_space_programming (wsp.py → results.json, runs/*.json; export_view.py → view.json).
"""
import json

import numpy as np
from manim import *

from common import MONO, code_panel, finish, highlight, next_up_card, token
from intro import play_token_intro
from x02_script import NEXT, TAGLINE, TITLE
from frontier_common import BAD, EDIT_C, GOOD, MODEL_C, study_footer
from voiced_scene import VoicedScene

SERIES_LABEL = "THE FRONTIER  ·  EPISODE 2"
CODE_DIR = "code/x02_weight_space_programming"
R = json.load(open(f"{CODE_DIR}/results.json"))
V = json.load(open(f"{CODE_DIR}/view.json"))
CODE = """for round in range(4):
    reply = ask(describe(weights) + graded(practice))
    edited = apply(weights, reply["edits"])
    if score(edited, practice) > score(weights, practice):
        weights = edited                   # keep only what helps"""
PAPERS = [("Can a Frontier LLM Repair a Quantized Network by Tool-Free Weight-Space Programming?", "2611.33031"),
          ("WeightLoop: Weight-Space Programming with Experimental Feedback", "2611.07319"),
          ("Weight-space programming for auditing clinical decision networks", "2611.05341"),
          ("Can an LLM Actually Reason and Program about Neural Network's Weights…?", "2611.08070"),
          ("Sparks of Weight-Space Programming for Robot Foundation Models", "2611.31715")]


def pct(x, d=0):
    return f"{100 * x:.{d}f}%"


def strip(values, width=8.0, height=0.45, flip=False):
    """A weight row as a strip of colored cells (blue negative, orange positive)."""
    v = np.array(values) * (-1 if flip else 1)
    m = np.abs(np.array(values)).max()
    cells = VGroup()
    for x in v:
        t = float(np.clip(x / m, -1, 1))
        col = interpolate_color(ManimColor("#14161C"), ManimColor("#F58C32") if t > 0 else ManimColor("#286EC8"), abs(t))
        cells.add(Rectangle(width=width / len(v), height=height, stroke_width=0, fill_color=col, fill_opacity=1))
    return cells.arrange(RIGHT, buff=0)


def reply_box(text, width=7.6, font_size=20):
    t = Text(text, font_size=font_size, color=GREY_A, line_spacing=0.9)
    if t.width > width - 0.5:
        t.scale_to_fit_width(width - 0.5)
    box = RoundedRectangle(width=width, height=t.height + 0.55, corner_radius=0.12, color=MODEL_C, fill_opacity=0.08)
    lab = Text("Claude", font_size=16, color=MODEL_C).next_to(box, UP, buff=0.08).align_to(box, LEFT)
    return VGroup(box, t.move_to(box), lab)


def result_line(parts, font_size=24):
    return VGroup(*[Text(p, font_size=font_size, color=c) for p, c in parts]).arrange(RIGHT, buff=0.45)


class WSPVideo(VoicedScene):
    VIDEO = "x02"

    def construct(self):
        play_token_intro(self, TITLE, 2, TAGLINE, label=SERIES_LABEL)
        self.hook()        # 1
        self.loop()        # 2
        self.case_zero()   # 3
        self.case_swap()   # 4
        self.case_flip()   # 5
        self.flip_again()  # 6
        self.summary()     # 7
        self.code()        # 8
        self.outro()       # 9

    def clear_stage(self, *keep, run_time=0.6):
        self.play(*[FadeOut(m) for m in self.mobjects if m not in keep], run_time=run_time)

    # ------------------------------------------------------------------ 1. hook
    def hook(self):
        self.section(1)
        rows = VGroup(*[VGroup(Text(t, font_size=19, color=GREY_A), Text(f"arXiv {i} · Nov 2026", font_size=15, color=GREY_B))
                        .arrange(DOWN, aligned_edge=LEFT, buff=0.06) for t, i in PAPERS]).arrange(DOWN, aligned_edge=LEFT, buff=0.22)
        rows.move_to([0, 1.0, 0])
        self.at("papers")
        self.play(LaggedStart(*[FadeIn(r, shift=0.1 * UP) for r in rows], lag_ratio=0.25), run_time=1.8)
        name = Text("weight-space programming", font_size=40, color=EDIT_C).move_to([0, -2.0, 0])
        self.at("programming")
        self.play(FadeIn(name, scale=1.1), run_time=0.5)
        sub = Text("today: Claude repairs our tiny addition model", font_size=24, color=MODEL_C).next_to(name, DOWN, buff=0.3)
        self.at("claude")
        self.play(FadeIn(sub), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 2. the loop
    def loop(self):
        self.section(2)
        self.clear_stage()
        specs = [("break the model\n(in secret)", BAD, "break"),
                 ("Claude reads: architecture,\n300 graded answers,\nweights as text", MODEL_C, "architecture"),
                 ("it proposes edits", MODEL_C, "proposes"),
                 ("apply, test on\npractice questions", GREY_A, "apply"),
                 ("keep only if\nthe score goes up", GOOD, "keep")]
        boxes = VGroup()
        for text, col, _ in specs:
            t = Text(text, font_size=24, color=col, line_spacing=0.9)
            b = RoundedRectangle(width=max(2.8, t.width + 0.5), height=1.55, corner_radius=0.12, color=col, fill_opacity=0.06)
            boxes.add(VGroup(b, t.move_to(b)))
        VGroup(*boxes[:3]).arrange(RIGHT, buff=0.6).move_to([0, 1.9, 0])
        boxes[3].move_to([boxes[2].get_x(), -0.3, 0])
        boxes[4].next_to(boxes[3], LEFT, buff=0.9)
        arrows = VGroup(Arrow(boxes[0].get_right(), boxes[1].get_left(), buff=0.08, color=GREY_B),
                        Arrow(boxes[1].get_right(), boxes[2].get_left(), buff=0.08, color=GREY_B),
                        Arrow(boxes[2].get_bottom(), boxes[3].get_top(), buff=0.08, color=GREY_B),
                        Arrow(boxes[3].get_left(), boxes[4].get_right(), buff=0.08, color=GREY_B))
        back = Arrow(boxes[4].get_corner(UR), boxes[2].get_corner(DL), buff=0.12, color=GREY_B)
        rl = Text("up to 4 rounds", font_size=20, color=GREY_B).next_to(boxes[4], LEFT, buff=0.3)
        for i, (_, _, cue) in enumerate(specs):
            self.at(cue)
            anims = [FadeIn(boxes[i], shift=0.1 * RIGHT)]
            if i:
                anims.append(GrowArrow(arrows[i - 1]))
            self.play(*anims, run_time=0.45)
        self.play(GrowArrow(back), FadeIn(rl), run_time=0.5)
        notes = VGroup(Text("no gradients", font_size=26, color=EDIT_C),
                       Text("the 1,000 test questions are never shown", font_size=26, color=GREY_A)).arrange(RIGHT, buff=0.8)
        notes.move_to([0, -2.6, 0])
        self.at("gradients")
        self.play(FadeIn(notes[0]), run_time=0.4)
        self.at("hidden")
        self.play(FadeIn(notes[1]), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 3. one zero
    def case_zero(self):
        self.section(3)
        self.clear_stage()
        z = R["zero"]
        h = Text("damage 1: one number set to zero", font_size=28, color=BAD).to_edge(UP, buff=0.5)
        g = V["ln1_gain"]
        bars = VGroup(*[Rectangle(width=0.12, height=max(0.02, 1.6 * (0 if i == 13 else v)), stroke_width=0,
                                  fill_color=BAD if i == 13 else GREY_B, fill_opacity=0.9) for i, v in enumerate(g)])
        bars.arrange(RIGHT, buff=0.04, aligned_edge=DOWN).move_to([0, 1.3, 0])
        cap = Text("blocks.0.ln1.weight: 64 gains", font=MONO, font_size=18, color=GREY_B).next_to(bars, DOWN, buff=0.15)
        self.at("zero")
        self.play(FadeIn(h), FadeIn(bars), FadeIn(cap), run_time=0.6)
        acc = Text(f"test: {pct(z['test_damaged'])}", font_size=26, color=BAD).to_corner(UR, buff=0.6).shift(DOWN * 0.5)
        self.at("78")
        self.play(FadeIn(acc), run_time=0.3)
        rb = reply_box("“blocks.0.ln1.weight[13] is exactly 0.000. Every other gain in that\n"
                       "LayerNorm sits between about 0.71 and 0.88, so this one was almost\n"
                       "certainly zeroed by the damage.”").move_to([0, -0.75, 0])
        self.at("reply")
        self.play(FadeIn(rb), run_time=0.6)
        fix = bars[13].copy().stretch_to_fit_height(1.6 * 0.79).set_fill(GOOD).align_to(bars[13], DOWN)
        e = Text("[13]: 0.000 → 0.79   (it was 0.81)", font=MONO, font_size=22, color=EDIT_C).move_to([0, -2.1, 0])
        self.at("sets")
        self.play(Transform(bars[13], fix), FadeIn(e), run_time=0.6)
        res = result_line([("1 number", GREY_A), (f"{z['rounds'][0]['seconds']:.0f} seconds", GREY_A),
                           (f"test {pct(z['test_damaged'])} → {pct(z['test_final'], 1)}", GOOD)]).move_to([0, -2.95, 0])
        self.at("eight")
        self.play(FadeIn(res[:2]), run_time=0.4)
        self.at("99")
        self.play(FadeIn(res[2]), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 4. swapped rows
    def case_swap(self):
        self.section(4)
        self.clear_stage()
        s = R["swap"]
        h = Text("damage 2: the embeddings of 3 and 8 swapped", font_size=28, color=BAD).to_edge(UP, buff=0.5)
        r3 = strip(V["tok"]["8"])        # after the damage, row 3 holds 8's vector and vice versa
        r8 = strip(V["tok"]["3"])
        l3 = Text("tok.weight[3]", font=MONO, font_size=20).next_to(r3, LEFT, buff=0.25)
        l8 = Text("tok.weight[8]", font=MONO, font_size=20).next_to(r8, LEFT, buff=0.25)
        g3, g8 = VGroup(l3, r3), VGroup(l8, r8)
        VGroup(g3, g8).arrange(DOWN, buff=0.3, aligned_edge=RIGHT).move_to([0, 1.5, 0])
        self.at("swap")
        self.play(FadeIn(h), FadeIn(g3), FadeIn(g8), run_time=0.6)
        acc = Text(f"test: {pct(s['test_damaged'])}", font_size=26, color=BAD).to_corner(UR, buff=0.6).shift(DOWN * 0.5)
        self.at("39")
        self.play(FadeIn(acc), run_time=0.3)
        rb = reply_box("“The embedding rows for tokens '3' and '8' have been swapped, so the model\n"
                       "reads every input 3 as an 8 and every 8 as a 3.”", width=11.0, font_size=22).move_to([0, -0.3, 0])
        self.at("reads")
        self.play(FadeIn(rb), run_time=0.6)
        self.at("swaps")
        p3, p8 = r3.get_center(), r8.get_center()
        self.play(r3.animate.move_to(p8), r8.animate.move_to(p3), run_time=0.8)
        res = result_line([("2 rows, 128 numbers", GREY_A), (f"test {pct(s['test_damaged'])} → {pct(s['test_final'], 1)}", GOOD)]).move_to([0, -2.0, 0])
        self.play(FadeIn(res), run_time=0.4)
        same = Text("every weight identical to the original", font_size=26, color=GOOD).move_to([0, -2.8, 0])
        self.at("original")
        self.play(FadeIn(same), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 5. flip, first run
    def case_flip(self):
        self.section(5)
        self.clear_stage()
        f = R["flip"]
        h = Text("damage 3: the sign of 7's embedding flipped", font_size=28, color=BAD).to_edge(UP, buff=0.5)
        row = strip(V["tok"]["7"], flip=True, width=7.0).move_to([0.6, 2.1, 0])
        lab = Text("tok.weight[7] × −1", font=MONO, font_size=20).next_to(row, LEFT, buff=0.25)
        self.at("flip")
        self.play(FadeIn(h), FadeIn(row), FadeIn(lab), run_time=0.6)
        acc = Text(f"practice: {pct(f['practice_damaged'])}", font_size=24, color=BAD).to_corner(UR, buff=0.6).shift(DOWN * 0.5)
        self.at("59")
        self.play(FadeIn(acc), run_time=0.3)
        rb = reply_box("“Most of the wrong answers are what you would get if an input digit 7\n"
                       "were read as 2: 70+26=46, 71+39=60, 78+95=123…”", font_size=19).move_to([0, 0.75, 0])
        self.at("diagnosis")
        self.play(FadeIn(rb), run_time=0.6)
        tries = [("round 1", "scale row 7 by 0.6", "no change", "shrinks"),
                 ("round 2", "set one number of row 7 to 0", "+0.3 points, kept", "single"),
                 ("round 3", "set that number to +2.516", "no change", "single"),
                 ("round 4", "halve neuron 55 in block 2's MLP", "no change: a red herring", "herring")]
        tab = VGroup()
        for i, (r, what, out, _) in enumerate(tries):
            y = -0.75 - 0.5 * i
            tab.add(VGroup(Text(r, font_size=20, color=GREY_B).move_to([-5.2, y, 0], aligned_edge=LEFT),
                           Text(what, font_size=20).move_to([-3.6, y, 0], aligned_edge=LEFT),
                           Text(out, font_size=20, color=BAD if "herring" in out else GREY_A).move_to([1.6, y, 0], aligned_edge=LEFT)))
        self.at("none")
        self.play(FadeIn(tab[0]), run_time=0.4)
        self.at("single")
        self.play(FadeIn(tab[1:3]), run_time=0.5)
        self.at("herring")
        self.play(FadeIn(tab[3]), run_time=0.4)
        end = Text(f"after 4 rounds: practice {pct(f['practice_damaged'], 1)} → {pct(f['practice_final'], 1)}",
                   font_size=24, color=BAD).move_to([0, -3.1, 0])
        self.at("barely")
        self.play(FadeIn(end), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 6. flip, runs 2 and 3
    def flip_again(self):
        self.section(6)
        self.clear_stage()
        runs = VGroup()
        for name in ("flip#2", "flip#3"):
            r = R[name]
            runs.add(VGroup(Text(f"run {name[-1]}", font_size=22, color=GREY_B),
                            Text("tok.weight[7] × −1", font=MONO, font_size=22, color=EDIT_C),
                            Text("1 round", font_size=22),
                            Text(f"test {pct(r['test_damaged'])} → {pct(r['test_final'], 1)}", font_size=22, color=GOOD)).arrange(RIGHT, buff=0.5))
        runs.arrange(DOWN, buff=0.35, aligned_edge=LEFT).move_to([0, 2.2, 0])
        self.at("fresh")
        self.play(FadeIn(runs[0][0]), FadeIn(runs[1][0]), run_time=0.4)
        self.at("minus")
        self.play(FadeIn(VGroup(*[r[1:3] for r in runs])), run_time=0.5)
        # the circle of digits: a sign flip is a half turn, i.e. +5 (mod 10)
        center, rad = np.array([-3.2, -1.2, 0]), 1.45
        circle = Circle(radius=rad, color=GREY_C).move_to(center)
        digits = VGroup(*[Text(str(d), font_size=26, color=EDIT_C if d in (2, 7) else GREY_A)
                          .move_to(center + rad * 1.22 * np.array([np.sin(TAU * d / 10), np.cos(TAU * d / 10), 0])) for d in range(10)])
        p7 = center + rad * np.array([np.sin(TAU * 0.7), np.cos(TAU * 0.7), 0])
        p2 = center + rad * np.array([np.sin(TAU * 0.2), np.cos(TAU * 0.2), 0])
        arrow = Arrow(p7, p2, buff=0.15, color=BAD)
        quote = reply_box("“If the digit embeddings encode digits periodically\n(mod 10), flipping the sign acts like adding 5.”",
                          width=6.6, font_size=19).move_to([3.0, -0.6, 0])
        self.at("reason")
        self.play(FadeIn(quote), run_time=0.5)
        self.at("circle")
        self.play(Create(circle), FadeIn(digits), run_time=0.7)
        self.at("adds")
        self.play(GrowArrow(arrow), run_time=0.5)
        lab = Text("7 + 5 → 2", font_size=26, color=BAD).move_to([3.0, -2.2, 0])
        self.at("seven")
        self.play(FadeIn(lab), run_time=0.4)
        self.at("99")
        self.play(FadeIn(VGroup(*[r[3] for r in runs])), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 7. summary
    def summary(self):
        self.section(7)
        self.clear_stage()
        head = VGroup(*[Text(t, font_size=20, color=GREY_B) for t in ("damage", "fixed", "numbers edited", "test after")])
        data = [("one zero", "1 of 1", "1", pct(R["zero"]["test_final"], 1)),
                ("3 and 8 swapped", "1 of 1", "128 (2 rows)", pct(R["swap"]["test_final"], 1)),
                ("7 sign-flipped", "2 of 3", "64 (1 row)", f"{pct(R['flip#2']['test_final'], 1)} / {pct(R['flip']['test_final'], 1)}")]
        rows = VGroup(head, *[VGroup(*[Text(t, font_size=22, color=GOOD if j == 1 and t.startswith("1") else (EDIT_C if j == 1 else WHITE))
                                       for j, t in enumerate(r)]) for r in data])
        xs = [-5.4, -1.6, 0.6, 3.6]
        for i, r in enumerate(rows):
            for j, t in enumerate(r):
                t.move_to([xs[j], 2.3 - 0.6 * i, 0], aligned_edge=LEFT)
        self.at("structural")
        self.play(FadeIn(rows), run_time=0.8)
        pts = VGroup(Text("finds and undoes structural damage, often with tiny edits", font_size=24, color=GOOD),
                     Text("the same question can get a different answer on another run", font_size=24, color=EDIT_C),
                     Text("repairing is not teaching it something new", font_size=24, color=GREY_A)).arrange(DOWN, aligned_edge=LEFT, buff=0.3)
        pts.move_to([0, -1.3, 0])
        self.play(FadeIn(pts[0]), run_time=0.4)
        cite = Text("papers this month report the same (e.g. arXiv 2611.04268)", font_size=16, color=GREY_B).next_to(pts[0], DOWN, buff=0.08).align_to(pts[0], LEFT)
        self.at("papers")
        self.play(FadeIn(cite), run_time=0.3)
        self.at("same")
        self.play(FadeIn(pts[1]), run_time=0.4)
        self.at("repairing")
        self.play(FadeIn(pts[2]), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 8. code
    def code(self):
        self.section(8)
        self.clear_stage()
        code, hl = code_panel(CODE, font_size=22)
        code.move_to([0, 0.3, 0])
        self.play(FadeIn(code, shift=0.2 * UP), run_time=0.5)
        self.at("describe")
        self.play(Create(hl), highlight(hl, code, 1), run_time=0.3)
        self.at("apply")
        self.play(highlight(hl, code, 2), run_time=0.3)
        self.at("test")
        self.play(highlight(hl, code, 3), run_time=0.3)
        self.at("keep")
        self.play(highlight(hl, code, 4), run_time=0.3)
        self.end_section()

    # ------------------------------------------------------------------ 9. outro
    def outro(self):
        self.section(9)
        self.clear_stage(run_time=0.4)
        card_ = next_up_card(NEXT)
        self.at("next")
        self.play(FadeIn(card_, shift=0.2 * UP), FadeIn(study_footer()), run_time=0.6)
        self.end_section()
        finish(self, hold=1.5)
