"""LLMs in Practice, episode 10 — Quantization: Shrinking a Model to Fit a Laptop.

Render from the repo root:  ./render.sh llms-in-practice p10
Every size, weight, loss and answer on screen comes from code/p10_quantization/quantize.py
(Qwen2.5-0.5B-Instruct, round-to-nearest per output row, simulated; loss on 4,096 tokens of How LLMs Work narration).
"""
import random

from manim import *

from common import MODEL_COLOR, MONO, code_panel, finish, highlight, next_up_card
from intro import play_token_intro
from p10_script import NEXT, TAGLINE, TITLE
from voiced_scene import VoicedScene

SERIES_LABEL = "LLMs IN PRACTICE  ·  EPISODE 10"
BEFORE = [-0.0019, -0.0052, 0.0188, 0.0125, 0.0040, -0.0092]
AFTER = [-0.0, -0.0082, 0.0165, 0.0165, 0.0, -0.0082]
ROWS = [("fp32", 32, 1365, "0", "2.836", WHITE), ("int8", 8, 341, "1.1%", "2.837", GREEN_B),
        ("int4", 4, 171, "19.2%", "3.582", YELLOW), ("int3", 3, 128, "43.3%", "12.491", RED_B),
        ("int2", 2, 85, "93.1%", "16.610", RED)]
FP32_ANSWER = "A context window refers to the portion of a document or message that contains\ninformation about the current conversation or interaction between two parties."
INT8_ANSWER = "A context window refers to the portion of a video or audio clip that remains\nvisible after the main content has been cut off."
INT4_ANSWER = "A \"context\" window refers to the full or comprehensive view of information and\nmaterials that provide insight into various aspects, themes, topics, …"
INT3_ANSWER = "r,采；+ s  收 有的\\ng  f onClosea、张常 ning同旧， a善製n::之算了一个是 …"
CODE = """def quantize(w, bits):
    levels = 2 ** (bits - 1) - 1                    # 8 bits: -127..127
    scale = w.abs().amax(dim=1, keepdim=True) / levels    # per row
    q = torch.clamp(torch.round(w / scale), -levels, levels)
    return q * scale"""


def labeled_box(text, color=MODEL_COLOR, width=2.4, height=1.0, font_size=26):
    label = Text(text, font_size=font_size, line_spacing=0.8)
    box = RoundedRectangle(corner_radius=0.2, width=max(width, label.width + 0.5), height=height, stroke_color=color,
                           fill_color=color, fill_opacity=0.25)
    return VGroup(box, label.move_to(box))


def answer_box(text, color, font_size=18):
    label = Text(text, font_size=font_size, line_spacing=0.8)
    box = RoundedRectangle(corner_radius=0.15, width=label.width + 0.5, height=label.height + 0.35, stroke_width=0,
                           fill_color=color, fill_opacity=0.85)
    return VGroup(box, label.move_to(box))


class QuantizationVideo(VoicedScene):
    VIDEO = "p10"

    def construct(self):
        play_token_intro(self, TITLE, 10, TAGLINE, label=SERIES_LABEL)
        self.hook()          # 1
        self.bits()          # 2
        self.recipe()        # 3
        self.real()          # 4
        self.results()       # 5
        self.lower()         # 6
        self.subtle()        # 7
        self.better()        # 8
        self.code()          # 9
        self.outro()         # 10

    def clear_stage(self, *keep, run_time=0.6):
        self.play(*[FadeOut(m) for m in self.mobjects if m not in keep], run_time=run_time)

    # ------------------------------------------------------------------ 1. hook
    def hook(self):
        self.section(1)
        n = Text("494,032,768 weights", font_size=52, color=YELLOW).move_to([0, 1.8, 0])
        model = Text("Qwen2.5-0.5B-Instruct", font=MONO, font_size=24, color=GREY_B).next_to(n, UP, buff=0.3)
        self.at("494")
        self.play(FadeIn(n, scale=1.2), FadeIn(model), run_time=0.6)
        size = Text("× 4 bytes ≈ 1.98 GB", font_size=36).next_to(n, DOWN, buff=0.4)
        self.at("two")
        self.play(FadeIn(size, shift=0.2 * UP), run_time=0.5)
        big = Text("big models: hundreds of GB", font_size=30, color=RED_B).next_to(size, DOWN, buff=0.5)
        self.at("hundreds")
        self.play(FadeIn(big), run_time=0.5)
        laptop = Text("laptop?  phone?", font_size=34, color=GREEN_B).to_edge(DOWN, buff=0.7)
        self.at("laptop")
        self.play(FadeIn(laptop, shift=0.2 * UP), run_time=0.5)
        self.end_section()

    # ------------------------------------------------------------------ 2. bits per weight
    def bits(self):
        self.section(2)
        self.clear_stage()
        rows = VGroup()
        for bits, label, color in [(32, "32 bits = 4 bytes", GREY_B), (16, "16 bits = 2 bytes", BLUE_C),
                                   (8, "8 bits = 1 byte", GREEN_C), (4, "4 bits = ½ byte", YELLOW)]:
            cells = VGroup(*[Square(0.3, stroke_width=1, stroke_color=BLACK, fill_color=color, fill_opacity=0.85)
                             for _ in range(bits)]).arrange(RIGHT, buff=0)
            rows.add(VGroup(Text(label, font_size=26), cells))
        for k, r in enumerate(rows):
            y = 2.2 - k * 0.75
            r[0].move_to([-6.2, y, 0], aligned_edge=LEFT)
            r[1].move_to([-2.6, y, 0], aligned_edge=LEFT)
        self.at("32")
        self.play(FadeIn(rows[0]), run_time=0.5)
        self.at("16")
        self.play(FadeIn(rows[1]), run_time=0.5)
        q = Text("does a weight need that much precision?", font_size=28, color=GREY_A).move_to([0, -1.6, 0])
        self.at("precision")
        self.play(FadeIn(q), run_time=0.5)
        self.at("fewer")
        self.play(FadeIn(rows[2]), FadeIn(rows[3]), run_time=0.6)
        name = Text("quantization: fewer bits per weight", font_size=32, color=YELLOW).move_to([0, -2.5, 0])
        self.play(FadeIn(name, shift=0.2 * UP), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 3. how it works
    def recipe(self):
        self.section(3)
        self.clear_stage()
        rng = random.Random(10)
        weights = [rng.gauss(0, 0.35) for _ in range(14)]
        m = max(abs(w) for w in weights)
        line = NumberLine(x_range=[-m, m, m], length=11, include_numbers=False, color=GREY_B).move_to([0, 0.6, 0])
        row_l = Text("one row of a weight matrix", font_size=26).to_edge(UP, buff=0.6)
        dots = VGroup(*[Dot(line.n2p(w) + 0.45 * UP, radius=0.08, color=BLUE_C) for w in weights])
        self.at("row")
        self.play(FadeIn(row_l), Create(line), LaggedStart(*[FadeIn(d, shift=0.2 * DOWN) for d in dots],
                                                            lag_ratio=0.05), run_time=0.9)
        ends = VGroup(Text("−max", font_size=22, color=YELLOW).next_to(line.n2p(-m), DOWN, buff=0.25),
                      Text("+max", font_size=22, color=YELLOW).next_to(line.n2p(m), DOWN, buff=0.25))
        self.at("largest")
        self.play(FadeIn(ends), Indicate(dots[max(range(14), key=lambda i: abs(weights[i]))], color=YELLOW,
                                         scale_factor=2), run_time=0.6)
        levels = 7
        ticks = VGroup(*[Line(UP * 0.15, DOWN * 0.15, color=YELLOW).move_to(line.n2p(m * k / levels))
                         for k in range(-levels, levels + 1)])
        self.at("round")
        self.play(Create(ticks), run_time=0.6)
        snapped = [round(w / m * levels) * m / levels for w in weights]
        self.at("grid")
        self.play(*[d.animate.move_to(line.n2p(s) + 0.45 * UP).set_color(YELLOW) for d, s in zip(dots, snapped)],
                  run_time=0.8)
        counts = VGroup(Text("8 bits: 255 steps", font_size=28, color=GREEN_B),
                        Text("4 bits: 15 steps (shown)", font_size=28, color=YELLOW)).arrange(RIGHT, buff=1.2)
        counts.move_to([0, -1.2, 0])
        self.at("eight")
        self.play(FadeIn(counts[0]), run_time=0.4)
        self.at("15")
        self.play(FadeIn(counts[1]), run_time=0.4)
        store = Text("store: small integers + one scale per row", font_size=28).move_to([0, -2.4, 0])
        self.at("integers")
        self.play(FadeIn(store, shift=0.2 * UP), run_time=0.5)
        self.end_section()

    # ------------------------------------------------------------------ 4. real weights
    def real(self):
        self.section(4)
        self.clear_stage()
        head = Text("real weights: layer 1, first row, first six", font_size=28).to_edge(UP, buff=0.6)
        self.at("real")
        self.play(FadeIn(head), run_time=0.4)
        cols = VGroup()
        for b, a in zip(BEFORE, AFTER):
            cols.add(VGroup(Text(f"{b:+.4f}", font=MONO, font_size=26),
                            Text(f"{a:+.4f}".replace("+0.0000", " 0.0000").replace("-0.0000", " 0.0000"),
                                 font=MONO, font_size=26, color=YELLOW)).arrange(DOWN, buff=0.5))
        cols.arrange(RIGHT, buff=0.45).move_to([0.8, 0.3, 0])
        labels = VGroup(Text("32-bit", font_size=24, color=GREY_B), Text("4-bit", font_size=24, color=YELLOW))
        labels[0].next_to(cols[0][0], LEFT, buff=0.5)
        labels[1].next_to(cols[0][1], LEFT, buff=0.5)
        self.at("before")
        self.play(FadeIn(labels[0]), FadeIn(VGroup(*[c[0] for c in cols])), run_time=0.5)
        self.play(FadeIn(labels[1]), *[TransformFromCopy(c[0], c[1]) for c in cols], run_time=0.8)
        self.at("close")
        self.play(Indicate(VGroup(*[c[1] for c in cols]), color=YELLOW, scale_factor=1.05), run_time=0.6)
        zeros = VGroup(*[SurroundingRectangle(cols[i][1], color=RED, buff=0.08) for i in (0, 4)])
        self.at("zero")
        self.play(Create(zeros), run_time=0.5)
        self.end_section()

    # ------------------------------------------------------------------ 5. results
    def table_row(self, row, y):
        name, bits, size, change, loss, color = row
        bar = Rectangle(width=size / 1365 * 4.5, height=0.42, stroke_width=0, fill_color=color, fill_opacity=0.8)
        cells = VGroup(Text(name, font=MONO, font_size=26, color=color), bar,
                       Text(f"{size:,} MB", font=MONO, font_size=22),
                       Text(change, font=MONO, font_size=22), Text(loss, font=MONO, font_size=26, color=color))
        cells[0].move_to([-6.0, y, 0], aligned_edge=LEFT)
        bar.move_to([-4.6, y, 0], aligned_edge=LEFT)
        cells[2].move_to([1.2, y, 0], aligned_edge=LEFT)
        cells[3].move_to([3.6, y, 0])
        cells[4].move_to([5.5, y, 0])
        return cells

    def results(self):
        self.section(5)
        self.clear_stage()
        head = VGroup(Text("size of weight matrices", font_size=20, color=GREY_B).move_to([-2.4, 2.6, 0]),
                      Text("weight change", font_size=20, color=GREY_B).move_to([3.6, 2.6, 0]),
                      Text("loss", font_size=20, color=GREY_B).move_to([5.5, 2.6, 0]))
        note = Text("weight matrices hold 72% of the parameters · loss on 4,096 tokens of How LLMs Work narration",
                    font_size=18, color=GREY_B).to_edge(DOWN, buff=0.35)
        self.at("measure")
        self.play(FadeIn(head), run_time=0.4)
        self.at("72")
        self.play(FadeIn(note), run_time=0.4)
        self.rows = VGroup(*[self.table_row(r, 1.9 - k * 0.8) for k, r in enumerate(ROWS)])
        self.at("1365")
        self.play(FadeIn(self.rows[0][:3]), run_time=0.5)
        self.at("836")
        self.play(FadeIn(self.rows[0][3:]), run_time=0.4)
        self.at("quarter")
        self.play(FadeIn(self.rows[1][:3], shift=0.2 * RIGHT), run_time=0.5)
        self.at("1")
        self.play(FadeIn(self.rows[1][3]), run_time=0.3)
        self.at("837")
        self.play(FadeIn(self.rows[1][4]), run_time=0.3)
        same = Text("practically identical", font_size=24, color=GREEN_B).next_to(self.rows[1][4], DOWN, buff=0.1)
        same.align_to(self.rows[1][4], RIGHT)
        self.at("identical")
        self.play(FadeIn(same), run_time=0.4)
        self.same = same
        self.end_section()

    # ------------------------------------------------------------------ 6. going lower
    def lower(self):
        self.section(6)
        self.play(FadeOut(self.same), run_time=0.3)
        self.at("eighth")
        self.play(FadeIn(self.rows[2][:3], shift=0.2 * RIGHT), run_time=0.5)
        self.at("19")
        self.play(FadeIn(self.rows[2][3]), run_time=0.3)
        self.at("58")
        self.play(FadeIn(self.rows[2][4]), run_time=0.3)
        vague = answer_box(INT4_ANSWER, GREY_D, 16).move_to([0, -2.3, 0])
        self.at("sense")
        self.play(FadeIn(vague, shift=0.2 * UP), run_time=0.5)
        self.at("three")
        self.play(FadeIn(self.rows[3]), FadeOut(vague), run_time=0.5)
        self.at("12")
        self.play(Indicate(self.rows[3][4], color=RED), run_time=0.5)
        self.at("two")
        self.play(FadeIn(self.rows[4]), run_time=0.5)
        junk = answer_box(INT3_ANSWER, RED_E, 18).move_to([0, -2.3, 0])
        self.at("gibberish")
        self.play(FadeIn(junk, shift=0.2 * UP), run_time=0.5)
        self.end_section()

    # ------------------------------------------------------------------ 7. a subtle point
    def subtle(self):
        self.section(7)
        self.clear_stage()
        q = Text("“In one sentence, what is a context window?”", font_size=26, color=BLUE_B).to_edge(UP, buff=0.6)
        a32 = answer_box(FP32_ANSWER, GREY_D, 18)
        a8 = answer_box(INT8_ANSWER, TEAL_E, 18)
        VGroup(a32, a8).arrange(DOWN, buff=0.6).move_to([0.6, 0.6, 0])
        l32 = Text("fp32", font=MONO, font_size=22).next_to(a32, LEFT, buff=0.3)
        l8 = Text("int8", font=MONO, font_size=22, color=GREEN_B).next_to(a8, LEFT, buff=0.3)
        self.at("subtle")
        self.play(FadeIn(q), FadeIn(a32), FadeIn(l32), run_time=0.6)
        self.at("changed")
        self.play(FadeIn(a8, shift=0.2 * UP), FadeIn(l8), run_time=0.5)
        tie = Text("near-tied tokens: a tiny change flips the choice, and everything after it", font_size=24,
                   color=YELLOW).move_to([0, -2.2, 0])
        self.at("tied")
        self.play(FadeIn(tie), run_time=0.5)
        note = Text("(neither answer is great: it's a 0.5B model)", font_size=20, color=GREY_B).next_to(tie, DOWN, 0.2)
        self.at("flip")
        self.play(FadeIn(note), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 8. better methods
    def better(self):
        self.section(8)
        self.clear_stage()
        head = Text("real tools do better than simple rounding", font_size=30).to_edge(UP, buff=0.6)
        self.play(FadeIn(head), run_time=0.4)
        items = VGroup(*[Text(t, font_size=28) for t in ["small groups of weights, each with its own scale",
                                                          "sensitive weights kept at higher precision",
                                                          "calibration on sample text"]])
        items.arrange(DOWN, aligned_edge=LEFT, buff=0.45).move_to([0.3, 0.3, 0])
        ticks = VGroup(*[Text("✓", font_size=28, color=GREEN).next_to(t, LEFT, buff=0.3) for t in items])
        for i, cue in enumerate(["groups", "sensitive", "calibrate"]):
            self.at(cue)
            self.play(FadeIn(items[i], shift=0.2 * RIGHT), FadeIn(ticks[i]), run_time=0.4)
        pop = Text("4-bit versions of big models: a popular trade-off for laptops", font_size=26, color=YELLOW)
        pop.move_to([0, -2.3, 0])
        self.at("popular")
        self.play(FadeIn(pop, shift=0.2 * UP), run_time=0.5)
        self.end_section()

    # ------------------------------------------------------------------ 9. code
    def code(self):
        self.section(9)
        self.clear_stage()
        code, hl = code_panel(CODE, font_size=22)
        code.move_to([0, 0.3, 0])
        self.at("code")
        self.play(FadeIn(code, shift=0.2 * UP), run_time=0.5)
        hl.match_y(code.line_numbers[2])
        self.at("scale")
        self.play(Create(hl), run_time=0.3)
        self.at("round")
        self.play(highlight(hl, code, 3), run_time=0.4)
        self.at("multiply")
        self.play(highlight(hl, code, 4), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 10. outro
    def outro(self):
        self.section(10)
        self.clear_stage(run_time=0.4)
        card = next_up_card(NEXT)
        self.at("next")
        self.play(FadeIn(card, shift=0.2 * UP), run_time=0.6)
        self.end_section()
        finish(self, hold=1.5)
