"""The Frontier: Self-Improvement and Memory, episode 4 — A Model That Edits Itself.

Render from the repo root:  ./render.sh frontier x04
Numbers come from code/x04_a_model_that_edits_itself (self_edit.py → run_self.json, run_random.json,
pilot_run_self_misdescribed.json).
"""
import collections
import json

from manim import *

from common import MONO, code_panel, finish, highlight, next_up_card
from intro import play_token_intro
from x04_script import NEXT, TAGLINE, TITLE
from frontier_common import BAD, EDIT_C, GOOD, MODEL_C, paper_card, study_footer
from voiced_scene import VoicedScene

SERIES_LABEL = "THE FRONTIER  ·  EPISODE 4"
CODE_DIR = "code/x04_a_model_that_edits_itself"
S = json.load(open(f"{CODE_DIR}/run_self.json"))
RND = json.load(open(f"{CODE_DIR}/run_random.json"))
PILOT = json.load(open(f"{CODE_DIR}/pilot_run_self_misdescribed.json"))
RAND_C = ORANGE
CODE = """edit = parse(model.generate(describe(model) + wrong_answers))   # the model edits itself
undo = apply(model, edit)
if score(model, practice) > best:
    best = score(model, practice)          # keep: the new model is the next editor
else:
    undo()"""


def pct(x):
    return f"{100 * x:.0f}%"


def pct1(x):
    return f"{100 * x:.1f}%"


def proposals(run):
    return [(r, p) for r in run["rounds"] for p in r["proposals"]]


def helped(run):
    return sum(1 for r, p in proposals(run) if p["edit"] and p["practice"] > r["practice_before"])


class SelfEditVideo(VoicedScene):
    VIDEO = "x04"

    def construct(self):
        play_token_intro(self, TITLE, 4, TAGLINE, label=SERIES_LABEL)
        self.hook()        # 1
        self.setup_()      # 2
        self.loop()        # 3
        self.writes()      # 4
        self.result()      # 5
        self.strange()     # 6
        self.baseline()    # 7
        self.mistake()     # 8
        self.shows()       # 9
        self.code()        # 10
        self.outro()       # 11

    def clear_stage(self, *keep, run_time=0.6):
        self.play(*[FadeOut(m) for m in self.mobjects if m not in keep], run_time=run_time)

    # ------------------------------------------------------------------ 1. hook
    def hook(self):
        self.section(1)
        card = paper_card("Self-WSP: Can Language Models Improve Themselves Through Weight-Space Programming?",
                          "arXiv 2611.28833 · 21 November 2026 · preprint",
                          ["Qwen3.5-0.8B reads its own weights, proposes an edit,",
                           "and continues with the edited weights as solver and editor"], width=11.0).move_to([0, 1.3, 0])
        self.at("november")
        self.play(FadeIn(card), run_time=0.6)
        nums = VGroup(Text("school maths (GSM8K): 11.8% → 62.2%", font_size=30, color=GOOD),
                      Text("sometimes by changing only 16 parameters", font_size=26, color=EDIT_C)).arrange(DOWN, buff=0.3).move_to([0, -1.6, 0])
        self.at("12")
        self.play(FadeIn(nums[0]), run_time=0.5)
        self.at("16")
        self.play(FadeIn(nums[1]), run_time=0.5)
        self.end_section()

    # ------------------------------------------------------------------ 2. setup
    def setup_(self):
        self.section(2)
        self.clear_stage()
        m = VGroup(RoundedRectangle(corner_radius=0.15, width=6.2, height=1.4, stroke_color=MODEL_C, fill_color=MODEL_C, fill_opacity=0.12),
                   VGroup(Text("Qwen3.5-0.8B", font_size=30, color=MODEL_C), Text("752 million parameters · on our CPU", font_size=20, color=GREY_B)).arrange(DOWN, buff=0.12))
        m[1].move_to(m[0])
        m.move_to([0, 2.0, 0])
        self.at("tried")
        self.play(FadeIn(m), run_time=0.5)
        task = Text("What is 412 + 958 + 136? Answer with just the number.", font=MONO, font_size=22).move_to([0, 0.2, 0])
        self.at("task")
        self.play(FadeIn(task), run_time=0.4)
        sc = VGroup(VGroup(Text("practice: 48 problems", font_size=24, color=GREY_A), Text(pct(S["practice_start"]), font_size=40, color=EDIT_C)).arrange(DOWN, buff=0.15),
                    VGroup(Text("held-out: 200 problems", font_size=24, color=GREY_A), Text(pct(S["test_start"]), font_size=40, color=GOOD)).arrange(DOWN, buff=0.15)
                    ).arrange(RIGHT, buff=2.0).move_to([0, -1.8, 0])
        self.at("23")
        self.play(FadeIn(sc[0]), run_time=0.4)
        self.at("26")
        self.play(FadeIn(sc[1]), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 3. the loop
    def loop(self):
        self.section(3)
        self.clear_stage()
        pos = [(-4.5, 1.4), (0, 1.4), (4.5, 1.4), (4.5, -1.4), (0, -1.4), (-4.5, -1.4)]
        labels = [("sees its own norm weights\n+ its wrong answers", GREY_A, "sees"), ("writes 4 edits\n(JSON)", EDIT_C, "four"),
                  ("test each on\nthe practice problems", GREY_A, "test"), ("keep the best,\nif it helps", GOOD, "keep"),
                  ("the edited model\nis the next editor", MODEL_C, "next")]
        boxes = VGroup()
        for (x, y), (t, c, _) in zip(pos, labels):
            b = RoundedRectangle(corner_radius=0.12, width=3.9, height=1.3, stroke_color=c)
            boxes.add(VGroup(b, Text(t, font_size=20, color=c, line_spacing=0.85).move_to(b)).move_to([x, y, 0]))
        arrows = VGroup(*[Arrow(boxes[i].get_right() if i < 2 else boxes[i].get_bottom() if i == 2 else boxes[i].get_left(),
                                boxes[i + 1].get_left() if i < 2 else boxes[i + 1].get_top() if i == 2 else boxes[i + 1].get_right(),
                                buff=0.1, color=GREY_B) for i in range(4)])
        back = Arrow(boxes[4].get_left(), boxes[0].get_bottom() + 0.0 * LEFT, buff=0.1, color=MODEL_C, path_arc=-PI / 3)
        for i, (_, _, cue) in enumerate(labels):
            self.at(cue)
            anims = [FadeIn(boxes[i])] + ([GrowArrow(arrows[i - 1])] if i > 0 else [])
            self.play(*anims, run_time=0.4)
        self.play(Create(back), run_time=0.4)
        n = Text("25 rounds · 100 edits", font_size=28, color=EDIT_C).move_to([-4.5, -2.9, 0])
        self.at("twenty")
        self.play(FadeIn(n), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 4. what it writes
    def writes(self):
        self.section(4)
        self.clear_stage()
        props = [p for _, p in proposals(S)]
        valid = [p for p in props if p["edit"]]
        v = Text(f"{len(valid)} of {len(props)} replies were valid edits", font_size=28, color=GOOD).to_edge(UP, buff=0.6)
        ex = Text(props[0]["raw"].strip(), font=MONO, font_size=18, color=EDIT_C).next_to(v, DOWN, buff=0.35)
        self.at("95")
        self.play(FadeIn(v), FadeIn(ex), run_time=0.5)
        factors = [p["edit"]["factor"] for p in valid if p["edit"]["op"] == "scale"]
        bins = [("< 1", lambda f: f < 1), ("1.0", lambda f: f == 1), ("1.01–1.09", lambda f: 1 < f < 1.1),
                ("1.1–1.5", lambda f: 1.1 <= f <= 1.5), ("> 1.5", lambda f: f > 1.5)]
        counts = [sum(1 for f in factors if fn(f)) for _, fn in bins]
        base_y = -2.2
        bars = VGroup()
        for i, ((lab, _), c) in enumerate(zip(bins, counts)):
            x = -5.0 + 1.6 * i
            col = EDIT_C if lab == "1.1–1.5" else GREY_B
            b = Rectangle(width=1.0, height=max(0.04, c * 0.065), stroke_width=0, fill_color=col, fill_opacity=0.8).move_to([x, base_y, 0], aligned_edge=DOWN)
            bars.add(VGroup(b, Text(str(c), font_size=22, color=col).next_to(b, UP, buff=0.08), Text(lab, font_size=16, color=GREY_B).next_to([x, base_y, 0], DOWN, buff=0.15)))
        cap = Text("scale factors it chose (76 'scale' edits)", font_size=18, color=GREY_B).next_to(bars, UP, buff=0.35)
        self.at("alike")
        self.play(FadeIn(cap), *[FadeIn(b[2]) for b in bars], run_time=0.4)
        self.at("scaling")
        self.play(*[AnimationGroup(GrowFromEdge(b[0], DOWN), FadeIn(b[1])) for b in bars], run_time=0.6)
        tens = collections.Counter(p["edit"]["tensor"] for p in valid)
        q = Text(f"{tens['model.norm.weight']} of {len(valid)} on the very last layer's norm", font_size=24, color=MODEL_C).move_to([3.9, 0.3, 0])
        self.at("quarter")
        self.play(FadeIn(q), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 5. result
    def result(self):
        self.section(5)
        self.clear_stage()
        h = Text(f"{len(S['kept'])} edits kept", font_size=30).to_edge(UP, buff=0.7)
        self.at("three")
        self.play(FadeIn(h), run_time=0.4)
        rows = VGroup()
        for i, (lab, a, b, col, cue) in enumerate((("practice (48)", S["practice_start"], S["practice_final"], EDIT_C, "practice"),
                                                   ("held-out (200)", S["test_start"], S["test_final"], GOOD, "held"))):
            y = 0.8 - 2.0 * i
            l = Text(lab, font_size=26, color=col).move_to([-4.6, y, 0])
            b0 = Rectangle(width=a * 14, height=0.55, stroke_width=0, fill_color=GREY_D, fill_opacity=0.9).move_to([-2.6, y + 0.35, 0], aligned_edge=LEFT)
            b1 = Rectangle(width=b * 14, height=0.55, stroke_width=0, fill_color=col, fill_opacity=0.85).move_to([-2.6, y - 0.35, 0], aligned_edge=LEFT)
            t0 = Text(f"before {pct1(a)}", font_size=20, color=GREY_B).next_to(b0, RIGHT, buff=0.2)
            t1 = Text(f"after {pct1(b)}", font_size=24, color=col).next_to(b1, RIGHT, buff=0.2)
            rows.add(VGroup(l, b0, t0, b1, t1))
        self.at("practice")
        self.play(FadeIn(rows[0][:3]), run_time=0.4)
        self.at("35")
        self.play(GrowFromEdge(rows[0][3], LEFT), FadeIn(rows[0][4]), run_time=0.5)
        self.at("held")
        self.play(FadeIn(rows[1][:3]), run_time=0.4)
        self.at("26")
        self.play(GrowFromEdge(rows[1][3], LEFT), FadeIn(rows[1][4]), run_time=0.5)
        self.end_section()

    # ------------------------------------------------------------------ 6. the strange edit
    def strange(self):
        self.section(6)
        self.clear_stage()
        e = [k for k in S["kept"] if k["edit"]["op"] == "set"][0]["edit"]
        h = Text(json.dumps(e), font=MONO, font_size=20, color=EDIT_C).to_edge(UP, buff=0.8)
        self.at("single")
        self.play(FadeIn(h), run_time=0.5)
        before, after = 3.484, e["value"]
        rows = VGroup(
            VGroup(Text("stored w[114]", font_size=24, color=GREY_B), Text(f"{before:.2f}", font_size=30), Text("→", font_size=30), Text(f"{after}", font_size=30, color=BAD)).arrange(RIGHT, buff=0.4),
            VGroup(Text("channel multiplier 1 + w", font_size=24, color=GREY_B), Text(f"{1 + before:.2f}", font_size=30), Text("→", font_size=30), Text(f"{1 + after:.1f}", font_size=30, color=BAD)).arrange(RIGHT, buff=0.4),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.5).move_to([0, 0.4, 0])
        self.at("minus")
        self.play(FadeIn(rows[0]), run_time=0.5)
        self.at("flips")
        self.play(FadeIn(rows[1]), run_time=0.5)
        x = Text(f"sign flipped, {abs(1 + after) / (1 + before):.1f}× bigger", font_size=28, color=BAD).move_to([0, -1.4, 0])
        self.at("seven")
        self.play(FadeIn(x), run_time=0.4)
        why = Text("kept because one more practice problem came out right (+2 points)", font_size=24, color=GREY_A).move_to([0, -2.5, 0])
        self.at("kept")
        self.play(FadeIn(why), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 7. random baseline
    def baseline(self):
        self.section(7)
        self.clear_stage()
        h = Text("the same loop, 100 random edits", font_size=30, color=RAND_C).to_edge(UP, buff=0.7)
        self.at("random")
        self.play(FadeIn(h), run_time=0.4)
        cols = ("", "practice", "held-out", "edits that beat\nthe current model")
        xs = (-4.4, -1.2, 1.6, 4.6)
        head = VGroup(*[Text(c, font_size=20, color=GREY_B, line_spacing=0.85).move_to([x, 1.4, 0]) for c, x in zip(cols, xs)])
        def row(lab, run, col, y):
            vals = (lab, pct1(run["practice_final"]), pct1(run["test_final"]), f"{helped(run)} of 100")
            return VGroup(*[Text(v, font_size=26, color=col).move_to([x, y, 0]) for v, x in zip(vals, xs)])
        r1, r2 = row("self-edits", S, MODEL_C, 0.4), row("random", RND, RAND_C, -0.6)
        self.play(FadeIn(head), FadeIn(r1[:3]), FadeIn(r2[0]), run_time=0.5)
        self.at("31")
        self.play(FadeIn(r2[1]), run_time=0.3)
        self.at("27")
        self.play(FadeIn(r2[2]), run_time=0.3)
        self.at("three")
        self.play(FadeIn(r1[3]), run_time=0.3)
        self.at("five")
        self.play(FadeIn(r2[3]), run_time=0.3)
        n = Text("one practice problem = 2 points · held-out noise ≈ ±3 points", font_size=20, color=GREY_B).move_to([0, -2.2, 0])
        self.play(FadeIn(n), run_time=0.3)
        self.end_section()

    # ------------------------------------------------------------------ 8. our mistake
    def mistake(self):
        self.section(8)
        self.clear_stage()
        h = Text("our mistake", font_size=32, color=BAD).to_edge(UP, buff=0.8)
        self.at("mistake")
        self.play(FadeIn(h), run_time=0.4)
        a = Text("first run: we called the stored numbers 'gains'", font_size=26).move_to([0, 1.0, 0])
        b = Text("Qwen3.5's norm multiplies by 1 + w, not by w", font_size=26, color=GREY_A).next_to(a, DOWN, buff=0.25)
        self.at("wrongly")
        self.play(FadeIn(a), FadeIn(b), run_time=0.5)
        p = PILOT
        c = Text(f"stopped after {len(p['rounds'])} rounds: practice {pct(p['kept'][-1]['practice'])}, held-out {pct(p['kept'][-1]['test'])}", font_size=24, color=GREY_B).move_to([0, -0.8, 0])
        d = Text(f"re-run, correct description: practice {pct(S['practice_final'])}, held-out {pct(S['test_final'])}", font_size=24, color=MODEL_C).next_to(c, DOWN, buff=0.25)
        self.at("reran")
        self.play(FadeIn(c), FadeIn(d), run_time=0.5)
        e = Text("same picture", font_size=28, color=EDIT_C).move_to([0, -2.6, 0])
        self.at("change")
        self.play(FadeIn(e), run_time=0.3)
        self.end_section()

    # ------------------------------------------------------------------ 9. what it shows
    def shows(self):
        self.section(9)
        self.clear_stage()
        a = Text("at our scale: a search the practice set could fool,\nno better than dice", font_size=28, line_spacing=0.9, color=EDIT_C).move_to([0, 2.0, 0])
        self.at("search")
        self.play(FadeIn(a), run_time=0.5)
        q = VGroup(Text("the Self-WSP abstract:", font_size=20, color=GREY_B),
                   Text("“WSP driven by random search sometimes performs comparably”", font_size=24, color=GREY_A),
                   Text("“The model shows little understanding of the edits it makes”", font_size=24, color=GREY_A)
                   ).arrange(DOWN, aligned_edge=LEFT, buff=0.25).move_to([0, -0.6, 0])
        self.at("abstract")
        self.play(FadeIn(q[0]), FadeIn(q[1]), run_time=0.5)
        self.at("understanding")
        self.play(FadeIn(q[2]), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 10. code
    def code(self):
        self.section(10)
        self.clear_stage()
        code, hl = code_panel(CODE, font_size=20)
        code.move_to([0, 0.3, 0])
        self.play(FadeIn(code, shift=0.2 * UP), run_time=0.4)
        self.at("writes")
        self.play(Create(hl), run_time=0.3)
        self.at("apply")
        self.play(highlight(hl, code, 1), run_time=0.3)
        self.at("score")
        self.play(highlight(hl, code, 2), run_time=0.3)
        self.at("keep")
        self.play(highlight(hl, code, 3), run_time=0.3)
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
