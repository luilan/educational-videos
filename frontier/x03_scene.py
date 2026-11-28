"""The Frontier: Self-Improvement and Memory, episode 3 — The Random Baseline.

Render from the repo root:  ./render.sh frontier x03
Numbers come from code/x03_the_random_baseline (random_search.py → results.json) and, for Claude, episode 2's
code/x02_weight_space_programming/results.json.
"""
import json
import math

from manim import *

from common import MONO, code_panel, finish, highlight, next_up_card
from intro import play_token_intro
from x03_script import NEXT, TAGLINE, TITLE
from frontier_common import BAD, EDIT_C, GOOD, MODEL_C, paper_card, study_footer
from voiced_scene import VoicedScene

SERIES_LABEL = "THE FRONTIER  ·  EPISODE 3"
CODE_DIR = "code/x03_the_random_baseline"
R = json.load(open(f"{CODE_DIR}/results.json"))
C = json.load(open("code/x02_weight_space_programming/results.json"))
RAND_C = ORANGE
NUDGE_C = GREY_B
CASE_L = {"zero": "one number zeroed", "swap": "digits 3 and 8 swapped", "flip": "sign of 7 flipped"}
CODE = """def random_edit(rng, weights):
    name = rng.choice(tensor_names)              # which weights
    op = rng.choice(["set", "scale_row", "swap_rows", "set_row"])
    edit = random_numbers_for(op, weights[name], rng)
    return edit      # same loop: apply, test, keep only if better"""


def runs(case, method="edits"):
    return [r for r in R["runs"] if r["case"] == case and r["method"] == method]


def pct(x, d=0):
    return f"{100 * x:.{d}f}%"


def die(size=0.9, color=RAND_C):
    sq = RoundedRectangle(corner_radius=0.12, width=size, height=size, stroke_color=color, fill_color=color, fill_opacity=0.12)
    pts = [(-1, 1), (1, 1), (0, 0), (-1, -1), (1, -1)]
    dots = VGroup(*[Dot(sq.get_center() + size * 0.27 * (x * RIGHT + y * UP), radius=0.06, color=color) for x, y in pts])
    return VGroup(sq, dots)


class RandomBaselineVideo(VoicedScene):
    VIDEO = "x03"

    def construct(self):
        play_token_intro(self, TITLE, 3, TAGLINE, label=SERIES_LABEL)
        self.hook()        # 1
        self.editors()     # 2
        self.fair()        # 3
        self.easy()        # 4
        self.overfit()     # 5
        self.hard()        # 6
        self.lesson()      # 7
        self.code()        # 8
        self.outro()       # 9

    def clear_stage(self, *keep, run_time=0.6):
        self.play(*[FadeOut(m) for m in self.mobjects if m not in keep], run_time=run_time)

    # ------------------------------------------------------------------ 1. hook
    def hook(self):
        self.section(1)
        steps = VGroup(*[RoundedRectangle(corner_radius=0.12, width=3.1, height=0.9, stroke_color=GREY_B) for _ in range(3)]).arrange(RIGHT, buff=0.8).move_to([1.2, 0.3, 0])
        labs = ["propose an edit", "test on practice", "keep if better"]
        for b, t in zip(steps, labs):
            b.add(Text(t, font_size=22).move_to(b))
        arrows = VGroup(Arrow(steps[0].get_right(), steps[1].get_left(), buff=0.1), Arrow(steps[1].get_right(), steps[2].get_left(), buff=0.1))
        back = CurvedArrow(steps[2].get_bottom() + 0.1 * DOWN, steps[0].get_bottom() + 0.1 * DOWN, angle=-TAU / 6, color=GREY_B)
        claude = VGroup(Text("Claude", font_size=30, color=MODEL_C)).next_to(steps[0], LEFT, buff=0.7)
        self.at("claude")
        self.play(FadeIn(claude), FadeIn(steps[0]), run_time=0.5)
        self.at("harness")
        self.play(FadeIn(steps[1:]), Create(arrows), Create(back), run_time=0.8)
        self.at("raised")
        self.play(Indicate(steps[2], color=GOOD), run_time=0.8)
        q = Text("the reasoning, or just the loop?", font_size=30, color=EDIT_C).move_to([0, 2.4, 0])
        self.at("loop")
        self.play(FadeIn(q), run_time=0.5)
        d = die().move_to(claude)
        self.at("dice")
        self.play(FadeOut(claude), FadeIn(d, scale=0.6), run_time=0.6)
        self.end_section()

    # ------------------------------------------------------------------ 2. the random editors
    def editors(self):
        self.section(2)
        self.clear_stage()
        h = Text("random editor 1: Claude's edits, every choice random", font_size=28, color=RAND_C).to_edge(UP, buff=0.6)
        d = die(0.7).next_to(h, LEFT, buff=0.3)
        self.at("random")
        self.play(FadeIn(h), FadeIn(d), run_time=0.5)
        ops = [("set a number", "set", "entry, value"), ("scale a row", "scale", "row, factor −2…2"),
               ("swap two rows", "swap", "two rows"), ("replace a row", "replace", "row, values")]
        cards = VGroup()
        for name, _, sub in ops:
            b = RoundedRectangle(corner_radius=0.1, width=2.9, height=1.2, stroke_color=EDIT_C, fill_color=EDIT_C, fill_opacity=0.06)
            t = VGroup(Text(name, font_size=22, color=EDIT_C), Text(sub, font_size=15, color=GREY_B)).arrange(DOWN, buff=0.12).move_to(b)
            cards.add(VGroup(b, t))
        cards.arrange(RIGHT, buff=0.25).move_to([0, 1.2, 0])
        for c, cue in zip(cards, ("set", "scale", "swap", "replace")):
            self.at(cue)
            self.play(FadeIn(c, shift=0.1 * UP), run_time=0.3)
        pick = Text("which weights?  which row?  which value?  →  dice", font_size=24, color=GREY_A).move_to([0, -0.2, 0])
        self.at("picks")
        self.play(FadeIn(pick), run_time=0.4)
        n = Text("random editor 2: nudge every weight by a little noise", font_size=26, color=NUDGE_C).move_to([0, -1.2, 0])
        self.at("nudges")
        self.play(FadeIn(n), run_time=0.4)
        rule = Text("same rule: keep an edit only if the practice score goes up · up to 10,000 tries", font_size=22, color=GOOD).move_to([0, -2.3, 0])
        self.at("rule")
        self.play(FadeIn(rule), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 3. equal tries
    def fair(self):
        self.section(3)
        self.clear_stage()
        h = Text("a fair fight: the same number of tries", font_size=30).to_edge(UP, buff=0.7)
        self.at("fight")
        self.play(FadeIn(h), run_time=0.4)
        cols = VGroup(Text("damage", font_size=22, color=GREY_B), Text("Claude, 1 try", font_size=22, color=MODEL_C),
                      Text("random, 4 tries", font_size=22, color=RAND_C))
        xs = (-4.2, 0.6, 4.2)
        for c, x in zip(cols, xs):
            c.move_to([x, 1.6, 0])
        claude_test = {"zero": C["zero"]["test_final"], "swap": C["swap"]["test_final"], "flip": C["flip#2"]["test_final"]}
        rows = []
        for i, case in enumerate(("zero", "swap", "flip")):
            y = 0.6 - 1.0 * i
            dmg = runs(case)[0]["test_damaged"]
            r4 = [r["checks"]["4"]["test"] for r in runs(case)]
            rows.append(VGroup(Text(f"{CASE_L[case]} ({pct(dmg)})", font_size=22).move_to([xs[0], y, 0]),
                               Text(f"{pct(dmg)} → {pct(claude_test[case])}", font_size=24, color=GOOD).move_to([xs[1], y, 0]),
                               Text(f"{pct(dmg)} → {' / '.join(pct(v) for v in r4)}", font_size=22, color=BAD).move_to([xs[2], y, 0])))
        self.play(FadeIn(cols[0]), *[FadeIn(r[0]) for r in rows], run_time=0.5)
        self.at("claudes")
        self.play(FadeIn(cols[1]), *[FadeIn(r[1]) for r in rows], run_time=0.5)
        self.at("four")
        self.play(FadeIn(cols[2]), run_time=0.3)
        self.at("nothing")
        self.play(*[FadeIn(r[2]) for r in rows], run_time=0.5)
        note = Text("(the sign flip: 2 of 3 Claude runs; test accuracy, 1,000 hidden questions)", font_size=18, color=GREY_B).move_to([0, -2.6, 0])
        self.at("three")
        self.play(FadeIn(note), run_time=0.3)
        self.end_section()

    # ------------------------------------------------------------------ helpers for the try-count plots
    def log_axes(self, pos, x_length=7.0, y_length=3.6, y0=0.3):
        ax = Axes(x_range=[0, 4, 1], y_range=[y0, 1.0, 0.1], x_length=x_length, y_length=y_length, tips=False,
                  axis_config={"color": GREY_C})
        ax.move_to(pos)
        xl = VGroup(*[Text(f"{10 ** k:,}", font_size=16, color=GREY_B).next_to(ax.c2p(k, y0), DOWN, buff=0.12) for k in range(5)])
        yl = VGroup(*[Text(f"{int(round(v * 100))}%", font_size=16, color=GREY_B).next_to(ax.c2p(0, v), LEFT, buff=0.12)
                      for v in [y0 + 0.1 * i for i in range(int(round((1.0 - y0) / 0.1)) + 1)] if int(round(v * 100)) % 20 == 0 or y0 > 0.5])
        cap = Text("tries (log scale) · practice score", font_size=16, color=GREY_B).next_to(xl, DOWN, buff=0.08)
        return ax, VGroup(ax, xl, yl, cap)

    def steps_curve(self, ax, run, color, width=3):
        pts, acc = [ax.c2p(0, run["practice_damaged"])], run["practice_damaged"]
        for k in run["kept"]:
            x = math.log10(k["trial"])
            pts += [ax.c2p(x, acc), ax.c2p(x, k["practice"])]
            acc = k["practice"]
        end = math.log10(run["trials"])
        pts.append(ax.c2p(end, acc))
        return VMobject(color=color, stroke_width=width).set_points_as_corners(pts)

    # ------------------------------------------------------------------ 4. the easy case
    def easy(self):
        self.section(4)
        self.clear_stage()
        ax, frame = self.log_axes([2.4, 0.2, 0], y0=0.8)
        h = Text("one number zeroed", font_size=26).move_to([-6.6, 2.7, 0], aligned_edge=LEFT)
        cheap = Text("random try: ~45 ms", font_size=22, color=RAND_C).move_to([-6.6, 2.1, 0], aligned_edge=LEFT)
        self.at("cheap")
        self.play(FadeIn(h), FadeIn(frame), run_time=0.4)
        self.at("45")
        self.play(FadeIn(cheap), run_time=0.3)
        rs = runs("zero")
        curves = [self.steps_curve(ax, r, RAND_C) for r in rs]
        target = DashedLine(ax.c2p(0, 0.99), ax.c2p(4, 0.99), color=GOOD, stroke_width=2)
        self.at("zeroed")
        self.play(Create(target), *[Create(c) for c in curves], run_time=1.6)
        solved = [r for r in rs if r["solved_at"]]
        marks = VGroup(*[VGroup(Dot(ax.c2p(math.log10(r["solved_at"]), 0.99), color=GOOD),
                                Text(f"{r['solved_at']:,}", font_size=18, color=GOOD).next_to(ax.c2p(math.log10(r["solved_at"]), 0.99), UP if j == 0 else DOWN, buff=0.12).shift((LEFT if j == 0 else RIGHT) * 0.25))
                         for j, r in enumerate(solved)])
        self.at("774")
        self.play(FadeIn(marks), run_time=0.4)
        cl = VGroup(Dot(ax.c2p(0, C["zero"]["practice_final"]), color=MODEL_C, radius=0.1),
                    Text("Claude, try 1", font_size=18, color=MODEL_C).next_to(ax.c2p(0, C["zero"]["practice_final"]), UR, buff=0.1))
        nc = VGroup(Text("numbers changed:", font_size=22, color=GREY_A),
                    Text(f"Claude  {C['zero']['numbers_changed_vs_intact']}", font_size=24, color=MODEL_C),
                    Text("random  " + "  ·  ".join(f"{r['numbers_changed_vs_intact']:,}" for r in solved), font_size=24, color=RAND_C)
                    ).arrange(DOWN, aligned_edge=LEFT, buff=0.15).move_to([-6.6, -0.6, 0], aligned_edge=LEFT)
        self.at("claude")
        self.play(FadeIn(cl), FadeIn(nc[:2]), run_time=0.4)
        self.at("331")
        self.play(FadeIn(nc[2]), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 5. overfitting
    def overfit(self):
        self.section(5)
        self.clear_stage()
        h = Text("practice score vs hidden test · one number zeroed", font_size=26).to_edge(UP, buff=0.6)
        self.play(FadeIn(h), run_time=0.4)
        solved = [r for r in runs("zero") if r["solved_at"]]
        nudge = runs("zero", "nudges")[0]
        items = [("Claude", C["zero"]["practice_final"], C["zero"]["test_final"], MODEL_C)] + \
                [(f"random, run {r['seed']}", r["practice_final"], r["test_final"], RAND_C) for r in solved] + \
                [("nudges", nudge["practice_final"], nudge["test_final"], NUDGE_C)]
        base_y, unit = -2.2, 3.6 / 0.3          # bars from 70% to 100%
        groups = VGroup()
        for i, (lab, p, t, col) in enumerate(items):
            x = -4.5 + 3.0 * i
            bp = Rectangle(width=0.8, height=(p - 0.7) * unit, stroke_width=0, fill_color=col, fill_opacity=0.45).move_to([x - 0.45, base_y, 0], aligned_edge=DOWN)
            bt = Rectangle(width=0.8, height=(t - 0.7) * unit, stroke_width=0, fill_color=col, fill_opacity=0.95).move_to([x + 0.45, base_y, 0], aligned_edge=DOWN)
            vp = Text(pct(p, 1), font_size=17, color=GREY_A).next_to(bp, UP, buff=0.08)
            vt = Text(pct(t, 1), font_size=17, color=col).next_to(bt, UP, buff=0.08)
            name = Text(lab, font_size=20, color=col).move_to([x, base_y - 0.35, 0])
            groups.add(VGroup(bp, vp, name, bt, vt))
        base = Line([-6.4, base_y, 0], [6.4, base_y, 0], color=GREY_C)
        key = Text("pale: practice (300 questions it was tuned on)   solid: test (1,000 hidden)", font_size=18, color=GREY_B).next_to(h, DOWN, buff=0.2)
        self.at("practice")
        self.play(Create(base), FadeIn(key), *[FadeIn(g[:3]) for g in groups], run_time=0.6)
        self.at("hidden")
        self.play(*[AnimationGroup(GrowFromEdge(g[3], DOWN), FadeIn(g[4])) for g in groups[:3]], run_time=0.6)
        self.at("nudges")
        self.play(Indicate(groups[3][0], color=NUDGE_C), run_time=0.5)
        self.at("92")
        self.play(GrowFromEdge(groups[3][3], DOWN), FadeIn(groups[3][4]), run_time=0.4)
        axis_note = Text("bars start at 70%", font_size=16, color=GREY_B).next_to(base, DOWN, buff=0.6).to_edge(RIGHT, buff=0.6)
        self.play(FadeIn(axis_note), run_time=0.2)
        self.end_section()

    # ------------------------------------------------------------------ 6. the hard cases
    def hard(self):
        self.section(6)
        self.clear_stage()
        panels = VGroup()
        objs = []
        for i, case in enumerate(("swap", "flip")):
            ax, frame = self.log_axes([-3.4 + 6.8 * i, 0.4, 0], x_length=5.2, y_length=3.0)
            t = Text(CASE_L[case], font_size=24).next_to(frame, UP, buff=0.25)
            curves = VGroup(*[self.steps_curve(ax, r, RAND_C) for r in runs(case)])
            nud = self.steps_curve(ax, runs(case, "nudges")[0], NUDGE_C, width=4)
            fin = [r["test_final"] for r in runs(case)]
            ft = Text(f"test after 10,000: {pct(min(fin))}–{pct(max(fin))}", font_size=20, color=RAND_C).next_to(frame, DOWN, buff=0.2)
            cl = Text(f"Claude: 1 try → {pct(C[case if case == 'swap' else 'flip#2']['test_final'])}", font_size=20, color=MODEL_C).next_to(ft, DOWN, buff=0.12)
            objs.append((frame, t, curves, nud, ft, cl))
        self.at("swap")
        self.play(*[FadeIn(o[0]) for o in objs], *[FadeIn(o[1]) for o in objs], run_time=0.5)
        self.at("10")
        self.play(*[Create(c) for c in objs[0][2]], *[Create(c) for c in objs[1][2]], run_time=1.6)
        self.at("60")
        self.play(FadeIn(objs[0][4]), run_time=0.3)
        self.at("66")
        self.play(FadeIn(objs[1][4]), run_time=0.3)
        never = Text("never the real fix: piles of patches", font_size=26, color=BAD).move_to([0, -2.9, 0])
        self.at("never")
        self.play(FadeIn(never), FadeIn(objs[0][5]), FadeIn(objs[1][5]), run_time=0.5)
        odds = Text("the exact swap: 1/30 tensors × 1/4 operations × 2/13 × 1/12 rows ≈ 1 in 9,400 per try",
                    font_size=19, color=GREY_A).to_edge(UP, buff=0.25)
        self.at("chance")
        self.play(FadeIn(odds), run_time=0.4)
        nl = Text("grey: nudges, never kept an edit", font_size=18, color=NUDGE_C).next_to(odds, DOWN, buff=0.12)
        self.at("nudges")
        self.play(Create(objs[0][3]), Create(objs[1][3]), FadeIn(nl), run_time=0.6)
        self.end_section()

    # ------------------------------------------------------------------ 7. lesson
    def lesson(self):
        self.section(7)
        self.clear_stage()
        items = VGroup(
            Text("the loop alone fixes small damage:", font_size=26, color=GOOD),
            Text("thousands of tries, a messier model", font_size=22, color=GREY_A),
            Text("structural damage needs the right edit in a huge space:", font_size=26, color=MODEL_C),
            Text("reading the weights pays", font_size=22, color=GREY_A),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.18).move_to([0, 1.6, 0])
        self.at("loop")
        self.play(FadeIn(items[0]), run_time=0.4)
        self.at("messier")
        self.play(FadeIn(items[1]), run_time=0.3)
        self.at("structural")
        self.play(FadeIn(items[2]), run_time=0.4)
        self.at("reading")
        self.play(FadeIn(items[3]), run_time=0.3)
        card = paper_card("Neural Thickets (Gan & Isola)", "arXiv 2603.12228 · March 2026",
                          ["random weight perturbations plus selection", "rival gradient methods near pretrained weights"],
                          width=8.4).move_to([0, -1.1, 0])
        self.at("thickets")
        self.play(FadeIn(card), run_time=0.5)
        must = Text("any self-improvement claim must beat random search", font_size=26, color=EDIT_C).move_to([0, -2.8, 0])
        self.at("beat")
        self.play(FadeIn(must), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 8. code
    def code(self):
        self.section(8)
        self.clear_stage()
        code, hl = code_panel(CODE, font_size=20)
        code.move_to([0, 0.3, 0])
        self.play(FadeIn(code, shift=0.2 * UP), run_time=0.5)
        self.at("tensor")
        self.play(Create(hl), run_time=0.3)
        self.play(highlight(hl, code, 1), run_time=0.2)
        self.at("operation")
        self.play(highlight(hl, code, 2), run_time=0.3)
        self.at("numbers")
        self.play(highlight(hl, code, 3), run_time=0.3)
        self.at("loop")
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
