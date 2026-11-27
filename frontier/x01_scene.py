"""The Frontier: Self-Improvement and Memory, episode 1 — The Frozen Model.

Render from the repo root:  ./render.sh frontier x01
Numbers come from code/x01_the_frozen_model (frozen.py + recheck.py → results.json; picture.py → out/weights.png).
"""
import json
import math

from manim import *

from common import MONO, code_panel, finish, highlight, next_up_card, token
from intro import play_token_intro
from x01_script import NEXT, TAGLINE, TITLE
from frontier_common import EDIT_C, GOOD, BAD, MODEL_C, paper_card, study_footer
from voiced_scene import VoicedScene

SERIES_LABEL = "THE FRONTIER  ·  EPISODE 1"
CODE_DIR = "code/x01_the_frozen_model"
R = json.load(open(f"{CODE_DIR}/results.json"))
EDIT = R["worst"][2]                                  # blocks.0.ln1.weight[13]: 0.81 → 0
CODE = """state = torch.load("model.pt")
state["blocks.0.ln1.weight"][13] = 0.0
model.load_state_dict(state)
accuracy(model, test)          # 0.992 → 0.777"""


def pct(x, d=0):
    return f"{100 * x:.{d}f}%"


def groups():
    """Tensor groups: (label, numbers, numbers whose removal costs ≥ 1 point)."""
    pt = R["per_tensor"]
    def tot(pred):
        ks = [k for k in pt if pred(k)]
        return sum(pt[k]["numbers"] for k in ks), sum(pt[k]["matter"] for k in ks)
    return [("embeddings", *tot(lambda k: k in ("tok.weight", "pos.weight"))),
            ("normalization scales", *tot(lambda k: k.endswith(("ln1.weight", "ln2.weight")) or k == "ln.weight")),
            ("normalization shifts", *tot(lambda k: k.endswith(("ln1.bias", "ln2.bias")) or k == "ln.bias")),
            ("attention tables", *tot(lambda k: ".attn." in k)),
            ("MLP tables", *tot(lambda k: ".mlp." in k)),
            ("output layer", *tot(lambda k: k.startswith("head.")))]


class FrozenVideo(VoicedScene):
    VIDEO = "x01"

    def construct(self):
        play_token_intro(self, TITLE, 1, TAGLINE, label=SERIES_LABEL)
        self.hook()          # 1
        self.four_ways()     # 2
        self.the_file()      # 3
        self.one_edit()      # 4
        self.every_number()  # 5
        self.where()         # 6
        self.fake_gains()    # 7
        self.code()          # 8
        self.outro()         # 9

    def clear_stage(self, *keep, run_time=0.6):
        self.play(*[FadeOut(m) for m in self.mobjects if m not in keep], run_time=run_time)

    # ------------------------------------------------------------------ 1. hook
    def hook(self):
        self.section(1)
        c1 = paper_card("Language Models Can Rewrite Their Own Weights", "arXiv 2611.21885 · 16 Nov 2026",
                        ["an open 27B model reads its own weights,", "diagnoses its failures, rewrites them",
                         "no gradients, no training run"]).move_to([0, 1.3, 0])
        self.at("november")
        self.play(FadeIn(c1[:3]), run_time=0.6)
        self.at("rewrote")
        self.play(FadeIn(c1[3]), run_time=0.6)
        gain = Text("+6.6 points on problems it never saw", font_size=28, color=GOOD).next_to(c1, DOWN, buff=0.3)
        self.at("6")
        self.play(FadeIn(gain, shift=0.1 * UP), run_time=0.5)
        c2 = paper_card("Self-WSP", "arXiv 2611.28833 · 21 Nov 2026",
                        ["Qwen3.5-0.8B edits its own weights"], width=7.0).next_to(gain, DOWN, buff=0.35)
        self.at("five")
        self.play(FadeIn(c2), run_time=0.6)
        self.end_section()

    # ------------------------------------------------------------------ 2. four ways
    def four_ways(self):
        self.section(2)
        self.clear_stage()
        h = Text("a trained model is frozen", font_size=32).to_edge(UP, buff=0.6)
        self.at("frozen")
        self.play(FadeIn(h), run_time=0.4)
        specs = [("train again", "a whole new run", GREY_A, "train"),
                 ("better prompt\nor harness", "gone when the\ntask ends", GREY_A, "prompt"),
                 ("more time\nto think", "gone when the\ntask ends", GREY_A, "think"),
                 ("edit the weights", "by reasoning\nabout them", EDIT_C, "fourth")]
        cards = VGroup()
        for title, sub, col, _ in specs:
            box = RoundedRectangle(width=2.9, height=2.5, corner_radius=0.15, color=col, fill_opacity=0.08)
            t = VGroup(Text(title, font_size=24, color=col, line_spacing=0.9),
                       Text(sub, font_size=19, color=GREY_B, line_spacing=0.9)).arrange(DOWN, buff=0.35).move_to(box)
            cards.add(VGroup(box, t))
        cards.arrange(RIGHT, buff=0.3).move_to([0, -0.3, 0])
        for card, (_, _, _, cue) in list(zip(cards, specs))[:3]:
            self.at(cue)
            self.play(FadeIn(card, shift=0.2 * UP), run_time=0.5)
        self.at("gone")
        self.play(Indicate(VGroup(cards[1][1][1], cards[2][1][1]), color=BAD), run_time=0.8)
        self.at("fourth")
        self.play(FadeIn(cards[3], shift=0.2 * UP), run_time=0.5)
        new = Text("new", font_size=22, color=EDIT_C).next_to(cards[3], UP, buff=0.15)
        self.at("directly")
        self.play(FadeIn(new), cards[3][0].animate.set_fill(EDIT_C, opacity=0.2), run_time=0.5)
        self.end_section()

    # ------------------------------------------------------------------ 3. the file
    def the_file(self):
        self.section(3)
        self.clear_stage()
        q = VGroup(*[token(c, font_size=28) for c in "47+38=85."]).arrange(RIGHT, buff=0.06).move_to([-3.6, 2.6, 0])
        acc = Text(f"{pct(R['acc_test'], 1)} of 1,000 new questions right", font_size=24, color=GOOD).next_to(q, DOWN, buff=0.3)
        self.at("tiny")
        self.play(FadeIn(q, lag_ratio=0.1), run_time=0.6)
        self.at("99")
        self.play(FadeIn(acc), run_time=0.4)
        img = ImageMobject(f"{CODE_DIR}/out/weights.png")
        img.set_resampling_algorithm(RESAMPLING_ALGORITHMS["nearest"])
        img.scale_to_fit_width(6.4).move_to([3.2, 1.6, 0])
        cap = Text("one of its tables: 64 × 256 numbers", font_size=18, color=GREY_B).next_to(img, DOWN, buff=0.15)
        f = VGroup(Text("model.pt", font=MONO, font_size=30, color=MODEL_C),
                   Text(f"{R['file_bytes']:,} bytes", font=MONO, font_size=24)).arrange(DOWN, buff=0.2).move_to([-3.6, 0.2, 0])
        self.at("file")
        self.play(FadeIn(f), FadeIn(img), FadeIn(cap), run_time=0.7)
        nums = Text("  ".join(f"{v:+.3f}" for v in R["example_weights"][:6]) + "  …", font=MONO, font_size=20, color=GREY_A)
        nums.move_to([0, -1.6, 0])
        n = Text(f"{R['params']:,} numbers", font_size=34, color=EDIT_C).move_to([-3.6, -2.6, 0])
        self.at("102")
        self.play(FadeIn(nums), FadeIn(n), run_time=0.6)
        t = Text(f"in {len(R['tensors'])} tables", font_size=28, color=GREY_A).next_to(n, RIGHT, buff=0.4)
        self.at("30")
        self.play(FadeIn(t), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 4. one edit
    def one_edit(self):
        self.section(4)
        self.clear_stage()
        name = f"{EDIT['tensor']}[{EDIT['index'][0]}]"
        line = VGroup(Text(name, font=MONO, font_size=28, color=GREY_A), Text("=", font=MONO, font_size=28),
                      Text(f"{EDIT['value']:.3f}", font=MONO, font_size=28, color=MODEL_C)).arrange(RIGHT, buff=0.25)
        line.move_to([0, 2.3, 0])
        self.at("edit")
        self.play(FadeIn(line[:2]), run_time=0.4)
        self.at("81")
        self.play(FadeIn(line[2]), run_time=0.4)
        zero = Text("0.000", font=MONO, font_size=28, color=EDIT_C).move_to(line[2])
        self.at("zero")
        self.play(Transform(line[2], zero), Flash(line[2], color=EDIT_C), run_time=0.6)
        axis_y = -2.0
        bars = VGroup()
        for i, (lab, v, col) in enumerate([("before", R["acc_test"], GOOD), ("after", EDIT["acc_test"], BAD)]):
            x = -4.8 + 2.2 * i
            bar = Rectangle(width=1.4, height=3.0 * v, stroke_width=0, fill_color=col, fill_opacity=0.85)
            bar.move_to([x, axis_y, 0], aligned_edge=DOWN)
            bars.add(VGroup(bar, Text(pct(v), font_size=28, color=col).next_to(bar, UP, buff=0.12),
                            Text(lab, font_size=20, color=GREY_B).next_to([x, axis_y, 0], DOWN, buff=0.2)))
        self.add(bars[0])
        self.at("drops")
        self.play(FadeIn(bars[0]), run_time=0.3)
        self.at("78")
        self.play(GrowFromEdge(bars[1][0], DOWN), FadeIn(bars[1][1:]), run_time=0.6)
        before = VGroup(*[token(c, font_size=30) for c in "27+15=42."]).arrange(RIGHT, buff=0.06)
        after = VGroup(*[token(c, font_size=30) for c in "27+15=" + EDIT["answers"]["27+15="]]).arrange(RIGHT, buff=0.06)
        ans = VGroup(VGroup(Text("before", font_size=20, color=GREY_B), before).arrange(RIGHT, buff=0.3),
                     VGroup(Text("after", font_size=20, color=GREY_B), after).arrange(RIGHT, buff=0.3)).arrange(DOWN, buff=0.4, aligned_edge=RIGHT)
        ans.move_to([2.6, -0.6, 0])
        self.at("27")
        self.play(FadeIn(ans[0]), run_time=0.4)
        self.at("41")
        self.play(FadeIn(ans[1]), run_time=0.4)
        self.play(Indicate(after[-2], color=BAD), run_time=0.6)
        self.end_section()

    # ------------------------------------------------------------------ 5. every number
    def every_number(self):
        self.section(5)
        self.clear_stage()
        h = Text("zero each of the 102,413 numbers in turn, test on 250 questions", font_size=26).to_edge(UP, buff=0.6)
        self.at("every")
        self.play(FadeIn(h), run_time=0.5)
        hist = R["drop_hist"]
        rows = [("no change", hist["no change"], GREY_B), ("raised the score", R["improved"], GREY_B),
                ("cost under 1 point", hist["under 1 point"], GREY_B), ("cost 1–10 points", hist["1–10 points"], BAD),
                ("cost 10–50 points", hist["10–50 points"], BAD), ("cost over 50 points", hist["over 50 points"], BAD)]
        mx = math.log10(max(n for _, n, _ in rows) + 1)
        tab = VGroup()
        for i, (lab, n, col) in enumerate(rows):
            y = 1.6 - 0.72 * i
            l = Text(lab, font_size=22).move_to([-4.0, y, 0], aligned_edge=RIGHT).shift(RIGHT * 1.3)
            w = max(0.04, 5.6 * math.log10(n + 1) / mx)
            b = Rectangle(width=w, height=0.42, stroke_width=0, fill_color=col, fill_opacity=0.8).move_to([-2.4, y, 0], aligned_edge=LEFT)
            v = Text(f"{n:,}", font_size=22, color=col).next_to(b, RIGHT, buff=0.15)
            tab.add(VGroup(l, b, v))
        note = Text("bar length on a log scale", font_size=16, color=GREY_C).to_edge(DOWN, buff=0.5)
        self.at("250")
        self.play(LaggedStart(*[FadeIn(r) for r in tab], lag_ratio=0.15), FadeIn(note), run_time=1.4)
        self.at("95")
        self.play(Indicate(tab[0], color=GREY_A), run_time=0.7)
        s = Text(f"{hist['1–10 points'] + hist['10–50 points']} cost a point or more", font_size=26, color=BAD).move_to([3.6, -2.2, 0])
        self.at("163")
        self.play(FadeIn(s), Indicate(VGroup(tab[3], tab[4]), color=BAD), run_time=0.8)
        self.at("none")
        self.play(Indicate(tab[5], color=YELLOW), run_time=0.6)
        self.end_section()

    # ------------------------------------------------------------------ 6. where they live
    def where(self):
        self.section(6)
        self.clear_stage()
        head = VGroup(Text("table", font_size=22, color=GREY_B), Text("numbers", font_size=22, color=GREY_B),
                      Text("cost ≥ 1 point", font_size=22, color=GREY_B))
        rows = VGroup(head)
        for lab, n, m in groups():
            rows.add(VGroup(Text(lab, font_size=24), Text(f"{n:,}", font=MONO, font_size=24),
                            Text(f"{m}", font=MONO, font_size=24, color=BAD if m else GREY_C)))
        for i, r in enumerate(rows):
            y = 2.2 - 0.62 * i
            r[0].move_to([-4.6, y, 0], aligned_edge=LEFT)
            r[1].move_to([1.4, y, 0], aligned_edge=RIGHT)
            r[2].move_to([4.4, y, 0], aligned_edge=RIGHT)
        self.play(FadeIn(rows[0]), run_time=0.3)
        self.at("embeddings")
        self.play(FadeIn(rows[1]), run_time=0.4)
        self.at("normalization")
        self.play(FadeIn(rows[2:4]), run_time=0.5)
        self.at("attention")
        self.play(FadeIn(rows[4:]), run_time=0.6)
        box = SurroundingRectangle(VGroup(rows[4], rows[5]), color=GOOD, buff=0.12)
        self.at("barely")
        self.play(Create(box), run_time=0.5)
        k = Text("what it knows is spread across thousands of numbers", font_size=24, color=GOOD).to_edge(DOWN, buff=0.6)
        self.at("spread")
        self.play(FadeIn(k), run_time=0.5)
        self.end_section()

    # ------------------------------------------------------------------ 7. fake improvements
    def fake_gains(self):
        self.section(7)
        self.clear_stage()
        imp = R["improvements"]
        h = Text(f"{imp['count']:,} single edits raised the score", font_size=30, color=GOOD).to_edge(UP, buff=0.6)
        self.at("help")
        self.play(FadeIn(h), run_time=0.5)
        by = imp["by_size"]
        left = VGroup(Text("by how much (of 250 questions)", font_size=20, color=GREY_B),
                      *[VGroup(Text(k, font_size=24), Text(f"{v:,}", font=MONO, font_size=24)).arrange(RIGHT, buff=0.5)
                        for k, v in by.items() if v]).arrange(DOWN, aligned_edge=LEFT, buff=0.3).move_to([-3.6, 0.2, 0])
        self.at("exactly")
        self.play(FadeIn(left, lag_ratio=0.2), run_time=0.8)
        better, worse = imp["top20_better_on_rest"], imp["top20_worse_on_rest"]
        same = len(imp["top20"]) - better - worse
        right = VGroup(Text("the best 20, on the other 750 questions", font_size=20, color=GREY_B))
        for lab, n, col in [("better", better, GOOD), ("same", same, GREY_B), ("worse", worse, BAD)]:
            dots = VGroup(*[Dot(radius=0.11, color=col) for _ in range(n)]).arrange(RIGHT, buff=0.08)
            lt = Text(lab, font_size=24, color=col)
            dots.next_to(lt, RIGHT, buff=1.25 - lt.width)
            right.add(VGroup(lt, dots))
        right.arrange(DOWN, aligned_edge=LEFT, buff=0.3).move_to([3.3, 0.2, 0])
        self.at("best")
        self.play(FadeIn(right[0]), run_time=0.4)
        self.at("four")
        self.play(FadeIn(right[1]), FadeIn(right[2]), run_time=0.5)
        self.at("seven")
        self.play(FadeIn(right[3]), run_time=0.4)
        noise = Text("noise", font_size=36, color=YELLOW).move_to([0, -2.0, 0])
        self.at("noise")
        self.play(FadeIn(noise, scale=1.2), run_time=0.4)
        rule = Text("a gain on the questions you used to pick the edit isn't a gain", font_size=24, color=GREY_A).to_edge(DOWN, buff=0.5)
        self.at("gain")
        self.play(FadeIn(rule), run_time=0.5)
        self.end_section()

    # ------------------------------------------------------------------ 8. code
    def code(self):
        self.section(8)
        self.clear_stage()
        code, hl = code_panel(CODE, font_size=24)
        code.move_to([0, 0.3, 0])
        self.play(FadeIn(code, shift=0.2 * UP), run_time=0.5)
        self.at("load")
        self.play(Create(hl), run_time=0.3)
        self.at("change")
        self.play(highlight(hl, code, 1), run_time=0.3)
        self.at("test")
        self.play(highlight(hl, code, 3), run_time=0.3)
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
