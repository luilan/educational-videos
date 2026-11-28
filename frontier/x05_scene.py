"""The Frontier: Self-Improvement and Memory, episode 5 — Diagnosing Its Own Mistakes.

Render from the repo root:  ./render.sh frontier x05
Numbers come from code/x05_diagnosing_its_own_mistakes (run_diagnose.json, run_probe.json, confirm.json).
"""
import json

from manim import *

from common import MONO, code_panel, finish, highlight, next_up_card
from intro import play_token_intro
from x05_script import NEXT, TAGLINE, TITLE
from frontier_common import BAD, EDIT_C, GOOD, MODEL_C, paper_card, study_footer
from voiced_scene import VoicedScene

SERIES_LABEL = "THE FRONTIER  ·  EPISODE 5"
CODE_DIR = "code/x05_diagnosing_its_own_mistakes"
D = json.load(open(f"{CODE_DIR}/run_diagnose.json"))
C = json.load(open(f"{CODE_DIR}/confirm.json"))["results"]
RAND_C, PROBE_C = ORANGE, TEAL_C
CODE = """effects = probe(model, wrong)                  # run itself: which vectors matter
diagnosis = model.generate(wrong_answers)
edit = parse(model.generate(diagnosis + table(effects)))
undo = apply(model, edit)
if score(model, practice) <= best: undo()      # same test and keep"""


def pct(x, d=0):
    return f"{100 * x:.{d}f}%"


def box(text, color, w=3.9, h=1.6, fs=17):
    r = RoundedRectangle(corner_radius=0.12, width=w, height=h, stroke_color=color, fill_color=color, fill_opacity=0.08)
    return VGroup(r, Text(text, font_size=fs, color=color, line_spacing=0.85).move_to(r))


class DiagnoseVideo(VoicedScene):
    VIDEO = "x05"

    def construct(self):
        play_token_intro(self, TITLE, 5, TAGLINE, label=SERIES_LABEL)
        self.hook()          # 1
        self.steps()         # 2
        self.diagnosis()     # 3
        self.result()        # 4
        self.confirm()       # 5
        self.baselines()     # 6
        self.understand()    # 7
        self.changed()       # 8
        self.shows()         # 9
        self.code()          # 10
        self.outro()         # 11

    def clear_stage(self, *keep, run_time=0.6):
        self.play(*[FadeOut(m) for m in self.mobjects if m not in keep], run_time=run_time)

    # ------------------------------------------------------------------ 1. hook
    def hook(self):
        self.section(1)
        card = paper_card("Language Models Can Rewrite Their Own Weights", "arXiv 2611.21885 · 16 November 2026 · preprint",
                          ["an open 27B model, given its weights, its graded answers",
                           "and the ability to run itself, diagnoses its own failure",
                           "and rewrites the weights behind it"], width=11.0).move_to([0, 1.4, 0])
        self.at("november")
        self.play(FadeIn(card), run_time=0.6)
        a = Text("+6.6 points on problems it never saw", font_size=30, color=GOOD).move_to([0, -1.0, 0])
        b = Text("smaller models of the same family: no edit, or broke themselves", font_size=24, color=BAD).move_to([0, -2.0, 0])
        self.at("6")
        self.play(FadeIn(a), run_time=0.4)
        self.at("smaller")
        self.play(FadeIn(b), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 2. two new steps
    def steps(self):
        self.section(2)
        self.clear_stage()
        h = Text("Qwen3.5-0.8B, same task, loop and budget as episode 4, plus two steps", font_size=24, color=GREY_B).to_edge(UP, buff=0.6)
        self.play(FadeIn(h), run_time=0.3)
        b1 = box("1. diagnose\nread your wrong answers,\nsay what goes wrong", MODEL_C).move_to([-4.75, 0.6, 0])
        b2 = box("2. probe: run yourself\nscale each of 49 norm vectors\nby 1.1: effect on right answers", GOOD, w=4.5).move_to([0, 0.6, 0])
        b3 = box("3. edit\n4 proposals, keep the best\nif the practice score rises", EDIT_C).move_to([4.75, 0.6, 0])
        ar = VGroup(Arrow(b1.get_right(), b2.get_left(), buff=0.08), Arrow(b2.get_right(), b3.get_left(), buff=0.08))
        self.at("diagnose")
        self.play(FadeIn(b1), run_time=0.4)
        self.at("probe")
        self.play(GrowArrow(ar[0]), FadeIn(b2), run_time=0.4)
        t = Text("the 10 biggest effects are shown to the model", font_size=20, color=GREY_B).next_to(b2, DOWN, buff=0.25)
        self.at("49")
        self.play(FadeIn(t), run_time=0.3)
        self.at("edit")
        self.play(GrowArrow(ar[1]), FadeIn(b3), run_time=0.4)
        n = Text("25 rounds · 100 proposals · 48 practice and 200 held-out problems", font_size=22, color=GREY_A).move_to([0, -2.2, 0])
        self.at("budget")
        self.play(FadeIn(n), run_time=0.3)
        self.end_section()

    # ------------------------------------------------------------------ 3. the diagnosis
    def diagnosis(self):
        self.section(3)
        self.clear_stage()
        diags = [r["diagnosis"][:200] for r in D["rounds"]]          # "almost the same": first 200 characters
        common = max(set(diags), key=diags.count)
        common = [r["diagnosis"] for r in D["rounds"] if r["diagnosis"][:200] == common][0]
        n_common = sum(1 for d in diags if d == common[:200])
        txt = common.replace("**", "")
        words, lines, cur = txt.split(), [], ""
        for w in words:
            if len(cur) + len(w) > 70:
                lines.append(cur); cur = w
            else:
                cur = (cur + " " + w).strip()
        lines.append(cur)
        card = VGroup(RoundedRectangle(corner_radius=0.12, width=12.0, height=0.6 + 0.45 * len(lines), stroke_color=MODEL_C, fill_color=MODEL_C, fill_opacity=0.06))
        t = Text("\n".join(lines), font_size=22, color=GREY_A, line_spacing=0.9).move_to(card[0])
        lab = Text("its diagnosis", font_size=18, color=MODEL_C).next_to(card[0], UP, buff=0.1).align_to(card[0], LEFT)
        VGroup(card, t, lab).move_to([0, 0.8, 0])
        self.at("diagnosis")
        self.play(FadeIn(card), FadeIn(t), FadeIn(lab), run_time=0.6)
        v = Text("vague", font_size=30, color=BAD).move_to([0, -1.3, 0])
        self.at("vague")
        self.play(FadeIn(v), run_time=0.3)
        s = Text(f"almost the same sentence in {n_common} of {len(diags)} rounds", font_size=26, color=GREY_A).move_to([0, -2.2, 0])
        self.at("19")
        self.play(FadeIn(s), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 4. result
    def result(self):
        self.section(4)
        self.clear_stage()
        h = Text(f"{len(D['kept'])} edits kept", font_size=30).to_edge(UP, buff=0.7)
        self.at("six")
        self.play(FadeIn(h), run_time=0.4)
        rows = VGroup()
        for i, (lab, a, b, col) in enumerate((("practice (48)", D["practice_start"], D["practice_final"], EDIT_C),
                                              ("held-out (200)", D["test_start"], D["test_final"], GOOD))):
            y = 0.8 - 2.0 * i
            l = Text(lab, font_size=26, color=col).move_to([-4.6, y, 0])
            b0 = Rectangle(width=a * 14, height=0.55, stroke_width=0, fill_color=GREY_D, fill_opacity=0.9).move_to([-2.6, y + 0.35, 0], aligned_edge=LEFT)
            b1 = Rectangle(width=b * 14, height=0.55, stroke_width=0, fill_color=col, fill_opacity=0.85).move_to([-2.6, y - 0.35, 0], aligned_edge=LEFT)
            t0 = Text(f"before {pct(a, 1)}", font_size=20, color=GREY_B).next_to(b0, RIGHT, buff=0.2)
            t1 = Text(f"after {pct(b, 1)}", font_size=24, color=col).next_to(b1, RIGHT, buff=0.2)
            rows.add(VGroup(l, b0, t0, b1, t1))
        self.at("practice")
        self.play(FadeIn(rows[0][:3]), run_time=0.3)
        self.at("35")
        self.play(GrowFromEdge(rows[0][3], LEFT), FadeIn(rows[0][4]), run_time=0.5)
        self.at("held")
        self.play(FadeIn(rows[1][:3]), run_time=0.3)
        self.at("32")
        self.play(GrowFromEdge(rows[1][3], LEFT), FadeIn(rows[1][4]), run_time=0.5)
        self.end_section()

    # ------------------------------------------------------------------ 5. confirm on 1,000
    def confirm(self):
        self.section(5)
        self.clear_stage()
        h = Text("checked on 1,000 fresh problems", font_size=30).to_edge(UP, buff=0.7)
        self.at("thousand")
        self.play(FadeIn(h), run_time=0.4)
        o, d = C["original"]["accuracy"], C["x05 diagnose + probe"]["accuracy"]
        big = VGroup(Text(pct(o, 1), font_size=56, color=GREY_B), Text("→", font_size=56), Text(pct(d, 1), font_size=56, color=GOOD)).arrange(RIGHT, buff=0.5).move_to([0, 1.0, 0])
        self.at("forty")
        self.play(FadeIn(big), run_time=0.5)
        g = VGroup(Text(f"{C['x05 diagnose + probe']['gained']}", font_size=48, color=GOOD), Text("missed before, right now", font_size=22, color=GOOD)).arrange(DOWN, buff=0.1).move_to([-3.0, -1.2, 0])
        l = VGroup(Text(f"{C['x05 diagnose + probe']['lost']}", font_size=48, color=BAD), Text("right before, missed now", font_size=22, color=BAD)).arrange(DOWN, buff=0.1).move_to([3.0, -1.2, 0])
        self.at("eighty")
        self.play(FadeIn(g), run_time=0.4)
        self.at("eleven")
        self.play(FadeIn(l), run_time=0.4)
        r = Text("real, not noise", font_size=28, color=EDIT_C).move_to([0, -2.8, 0])
        self.at("real")
        self.play(FadeIn(r), run_time=0.3)
        self.end_section()

    # ------------------------------------------------------------------ 6. baselines
    def baselines(self):
        self.section(6)
        self.clear_stage()
        h = Text("the same 1,000 fresh problems", font_size=26, color=GREY_B).to_edge(UP, buff=0.6)
        self.play(FadeIn(h), run_time=0.3)
        items = [("original", C["original"]["accuracy"], GREY_B, None),
                 ("episode 4\nblind self-edits", C["x04 self-edits"]["accuracy"], MODEL_C, "blind"),
                 ("episode 4\nrandom edits", C["x04 random edits"]["accuracy"], RAND_C, "random"),
                 ("probe alone,\nno model", C["x05 probe only"]["accuracy"], PROBE_C, "probe"),
                 ("diagnose + probe\n+ self-edit", C["x05 diagnose + probe"]["accuracy"], GOOD, "only")]
        base_y, unit = -2.2, 9.0
        line = Line([-6.4, base_y, 0], [6.4, base_y, 0], color=GREY_C)
        self.play(Create(line), run_time=0.2)
        for i, (lab, v, col, cue) in enumerate(items):
            x = -5.2 + 2.6 * i
            b = Rectangle(width=1.3, height=v * unit, stroke_width=0, fill_color=col, fill_opacity=0.8).move_to([x, base_y, 0], aligned_edge=DOWN)
            t = Text(pct(v, 1), font_size=24, color=col).next_to(b, UP, buff=0.1)
            n = Text(lab, font_size=17, color=col, line_spacing=0.8).next_to([x, base_y, 0], DOWN, buff=0.15)
            if cue:
                self.at(cue)
            self.play(GrowFromEdge(b, DOWN), FadeIn(t), FadeIn(n), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 7. understanding?
    def understand(self):
        self.section(7)
        self.clear_stage()
        h = Text("did it understand?", font_size=32, color=EDIT_C).to_edge(UP, buff=0.8)
        self.at("understand")
        self.play(FadeIn(h), run_time=0.4)
        valid = [(r, p) for r in D["rounds"] for p in r["proposals"] if p["edit"]]
        top = sum(1 for r, p in valid if p["edit"]["tensor"] == r["probe_top"][0][0])
        sc = [(p["edit"]["factor"] > 1) == (p["probe_effect"] > 0) for r, p in valid if p["edit"]["op"] == "scale" and p.get("probe_effect") is not None]
        a = Text(f"{top} of {len(valid)} valid edits picked the probe's top suggestion", font_size=28).move_to([0, 0.9, 0])
        b = Text(f"scaled in the direction its own probe pointed: {sum(sc)} of {len(sc)}", font_size=28).move_to([0, -0.2, 0])
        c = Text("a coin flip", font_size=32, color=BAD).move_to([0, -1.4, 0])
        self.at("three")
        self.play(FadeIn(a), run_time=0.4)
        self.at("half")
        self.play(FadeIn(b), run_time=0.4)
        self.at("coin")
        self.play(FadeIn(c), run_time=0.3)
        self.end_section()

    # ------------------------------------------------------------------ 8. what it changed
    def changed(self):
        self.section(8)
        self.clear_stage()
        h = Text("the kept edits (stored vector w × factor)", font_size=26).to_edge(UP, buff=0.6)
        self.at("kept")
        self.play(FadeIn(h), run_time=0.3)
        net = {}
        for k in D["kept"]:
            t = k["edit"]["tensor"].replace("model.", "").replace(".weight", "").replace("layers.", "layer ")
            net[t] = net.get(t, 1.0) * k["edit"]["factor"]
        rows = VGroup()
        for t, f in net.items():
            col = BAD if f < 0.5 else GOOD if f > 1.2 else GREY_A
            rows.add(VGroup(Text(t, font=MONO, font_size=22, color=GREY_A), Text(f"× {f:.2f}", font=MONO, font_size=26, color=col)).arrange(RIGHT, buff=0.6))
        rows.arrange(DOWN, aligned_edge=RIGHT, buff=0.3).move_to([0, 0.0, 0])
        cues = ["quarter", "quarter", "half", "nudged"]
        order = sorted(range(len(rows)), key=lambda i: (abs(list(net.values())[i] - 1) < 0.2, list(net.values())[i] > 1))
        self.at("quarter")
        self.play(*[FadeIn(rows[i]) for i in order[:2]], run_time=0.4)
        self.at("half")
        self.play(FadeIn(rows[order[2]]), run_time=0.3)
        self.at("nudged")
        self.play(FadeIn(rows[order[3]]), run_time=0.3)
        f = Text("4 vectors · 4,096 numbers of 752 million", font_size=26, color=EDIT_C).move_to([0, -2.4, 0])
        self.at("four")
        self.play(FadeIn(f), run_time=0.3)
        self.end_section()

    # ------------------------------------------------------------------ 9. what it shows
    def shows(self):
        self.section(9)
        self.clear_stage()
        a = Text("with a diagnosis and a self-probe in front of it,\nthe loop found edits that generalize", font_size=28, color=GOOD, line_spacing=0.9).move_to([0, 1.4, 0])
        self.at("generalize")
        self.play(FadeIn(a), run_time=0.5)
        b = Text("one run · one small model · one task", font_size=26, color=GREY_A).move_to([0, -0.2, 0])
        self.at("one")
        self.play(FadeIn(b), run_time=0.3)
        c = Text("does it know why they work? → episode 7", font_size=28, color=EDIT_C).move_to([0, -1.6, 0])
        self.at("seven")
        self.play(FadeIn(c), run_time=0.3)
        self.end_section()

    # ------------------------------------------------------------------ 10. code
    def code(self):
        self.section(10)
        self.clear_stage()
        code, hl = code_panel(CODE, font_size=20)
        code.move_to([0, 0.3, 0])
        self.play(FadeIn(code, shift=0.2 * UP), run_time=0.3)
        self.at("probe")
        self.play(Create(hl), run_time=0.2)
        self.at("diagnose")
        self.play(highlight(hl, code, 1), run_time=0.2)
        self.at("propose")
        self.play(highlight(hl, code, 2), run_time=0.2)
        self.at("test")
        self.play(highlight(hl, code, 4), run_time=0.2)
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
