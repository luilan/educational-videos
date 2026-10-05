"""How LLMs Work: Deep Dive, episode 12 — Attention Sinks.

Render from the repo root:  ./render.sh deep-dive d12
Every number on screen comes from code/d12_attention_sinks/attention_sinks.py (GPT-2 small, real weights, forward pass
written by hand; 256 and 1,024 tokens of Tiny Shakespeare).
"""
from manim import *

from common import MODEL_COLOR, MONO, TOKEN_COLOR, code_panel, finish, highlight, next_up_card, token
from intro import play_token_intro
from d12_script import NEXT, TAGLINE, TITLE
from voiced_scene import VoicedScene

SERIES_LABEL = "HOW LLMs WORK: DEEP DIVE  ·  EPISODE 12"
SINK = [[0.02, 0.0, 0.02, 0.0, 0.0, 0.0, 0.02, 0.0, 0.01, 0.01, 0.01, 0.03],
        [0.01, 0.01, 0.0, 0.29, 0.07, 0.04, 0.03, 0.03, 0.07, 0.08, 0.0, 0.0],
        [0.12, 0.4, 0.05, 0.06, 0.26, 0.06, 0.03, 0.31, 0.11, 0.06, 0.0, 0.45],
        [0.43, 0.24, 0.11, 0.47, 0.39, 0.45, 0.14, 0.15, 0.18, 0.38, 0.54, 0.23],
        [0.43, 0.31, 0.45, 0.33, 0.49, 0.54, 0.4, 0.15, 0.43, 0.42, 0.5, 0.0],
        [0.52, 0.81, 0.48, 0.3, 0.56, 0.58, 0.67, 0.76, 0.71, 0.52, 0.37, 0.56],
        [0.36, 0.73, 0.49, 0.61, 0.36, 0.6, 0.5, 0.4, 0.4, 0.79, 0.38, 0.45],
        [0.51, 0.56, 0.83, 0.58, 0.78, 0.49, 0.54, 0.61, 0.54, 0.53, 0.77, 0.54],
        [0.47, 0.69, 0.27, 0.58, 0.67, 0.46, 0.58, 0.57, 0.53, 0.67, 0.59, 0.7],
        [0.61, 0.64, 0.49, 0.41, 0.6, 0.61, 0.67, 0.6, 0.48, 0.71, 0.68, 0.45],
        [0.58, 0.57, 0.45, 0.45, 0.39, 0.69, 0.61, 0.44, 0.51, 0.6, 0.64, 0.57],
        [0.12, 0.43, 0.58, 0.54, 0.4, 0.56, 0.63, 0.53, 0.0, 0.61, 0.5, 0.38]]
CODE = """window = (j <= i) & (i - j < W)          # the last W tokens
sinks = (j < 4) & (j <= i)               # ... plus the first four, always
allowed = window | sinks"""


class AttentionSinksVideo(VoicedScene):
    VIDEO = "d12"

    def construct(self):
        play_token_intro(self, TITLE, 12, TAGLINE, label=SERIES_LABEL)
        self.hook()          # 1
        self.measure()       # 2
        self.content()       # 3
        self.why()           # 4
        self.noop()          # 5
        self.streaming()     # 6
        self.fix()           # 7
        self.design()        # 8
        self.code()          # 9
        self.outro()         # 10

    def clear_stage(self, *keep, run_time=0.6):
        self.play(*[FadeOut(m) for m in self.mobjects if m not in keep], run_time=run_time)

    # ------------------------------------------------------------------ 1. hook
    def hook(self):
        self.section(1)
        n, s = 10, 0.42
        g = VGroup()
        for i in range(n):
            for j in range(n):
                val = 0.0 if j > i else (0.7 if j == 0 else 0.3 / max(1, i))
                g.add(Square(s, stroke_width=0.8, stroke_color=GREY_D,
                             fill_color=BLACK if j > i else interpolate_color(GREY_E, GOLD, min(1, val * 1.3)),
                             fill_opacity=0.95).move_to([j * s, -i * s, 0]))
        g.move_to([-2.8, -0.3, 0])
        cap = Text("schematic", font_size=16, color=GREY_B).next_to(g, DOWN, buff=0.15)
        self.at("stare")
        self.play(FadeIn(g, lag_ratio=0.005), FadeIn(cap), run_time=0.8)
        col = SurroundingRectangle(VGroup(*[g[n * i] for i in range(n)]), color=YELLOW, buff=0.03)
        self.play(Create(col), run_time=0.4)
        qs = VGroup(Text("how much?", font_size=28), Text("why?", font_size=28),
                    Text("why it matters for long streams", font_size=28)).arrange(DOWN, buff=0.4, aligned_edge=LEFT)
        qs.move_to([3.0, 0.0, 0])
        for k, cue in enumerate(("much", "why", "matters")):
            self.at(cue)
            self.play(FadeIn(qs[k]), run_time=0.3)
        self.end_section()

    # ------------------------------------------------------------------ 2. measure
    def measure(self):
        self.section(2)
        self.clear_stage()
        s = 0.38
        cells = VGroup()
        for l in range(12):
            for h in range(12):
                v = SINK[l][h]
                cells.add(Square(s, stroke_width=0.6, stroke_color=GREY_D,
                                 fill_color=interpolate_color(GREY_E, GOLD, v), fill_opacity=0.95)
                          .move_to([h * s, -l * s, 0]))
        cells.move_to([-2.6, -0.4, 0])
        yl = VGroup(*[Text(str(l), font_size=14, color=GREY_B).next_to(cells[12 * l], LEFT, buff=0.1) for l in range(12)])
        ylt = Text("layer", font_size=18, color=GREY_B).next_to(yl, LEFT, buff=0.15)
        xlt = Text("head →", font_size=18, color=GREY_B).next_to(cells, UP, buff=0.12)
        head = Text("attention on the first token, all 144 heads (256 tokens)", font_size=24).to_edge(UP, buff=0.4)
        self.at("144")
        self.play(FadeIn(head), FadeIn(cells, lag_ratio=0.003), FadeIn(yl), FadeIn(ylt), FadeIn(xlt), run_time=1.0)
        a = Text("average: 39%", font_size=30, color=GOLD).move_to([3.4, 1.4, 0])
        self.at("39")
        self.play(FadeIn(a), run_time=0.4)
        b = Text("59 heads: more than half", font_size=26).move_to([3.4, 0.5, 0])
        self.at("59")
        self.play(FadeIn(b), run_time=0.4)
        early = SurroundingRectangle(VGroup(*cells[:36]), color=GREY_B, buff=0.03)
        el = Text("layers 0–2: barely", font_size=22, color=GREY_B).move_to([3.4, -0.6, 0])
        self.at("barely")
        self.play(Create(early), FadeIn(el), run_time=0.4)
        late = SurroundingRectangle(VGroup(*cells[60:132]), color=GOLD, buff=0.03)
        ll = Text("layers 5–10: more than half\n(0.51 to 0.61 per layer)", font_size=22, color=GOLD,
                  line_spacing=0.85).move_to([3.4, -1.6, 0])
        self.at("sixth")
        self.play(Create(late), FadeIn(ll), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 3. content
    def content(self):
        self.section(3)
        self.clear_stage()
        head = Text("change the first token", font_size=32).to_edge(UP, buff=0.7)
        self.at("special")
        self.play(FadeIn(head), run_time=0.4)
        rows = [("line break", "0.394"), ("zebra", "0.392"), ("The", "0.394"), ("“,” (a comma)", "0.393"),
                ("text cut mid-sentence", "0.395")]
        table = VGroup()
        for k, (name, val) in enumerate(rows):
            r = VGroup(Text(name, font_size=26), Text(val, font=MONO, font_size=26, color=GOLD))
            r[0].move_to([-1.2, 1.5 - 0.65 * k, 0], aligned_edge=RIGHT)
            r[1].move_to([0.8, 1.5 - 0.65 * k, 0], aligned_edge=LEFT)
            table.add(r)
        for k, cue in enumerate(("break", "zebra", "they", "comma")):
            self.at(cue, "the") if cue == "they" else self.at(cue)
            self.play(FadeIn(table[k]), run_time=0.25)
        self.at("cut")
        self.play(FadeIn(table[4]), run_time=0.3)
        pos = Text("not the word: the position", font_size=32, color=YELLOW).move_to([0, -2.4, 0])
        self.at("position")
        self.play(FadeIn(pos), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 4. why
    def why(self):
        self.section(4)
        self.clear_stage()
        s = Text("softmax: the weights must add up to 1", font_size=30).move_to([0, 2.4, 0])
        self.at("softmax")
        self.play(FadeIn(s), run_time=0.4)
        toks = VGroup(*[token(w, font_size=22) for w in ["First", "of", "all", ",", "the", "king"]]).arrange(RIGHT, buff=0.2)
        toks.move_to([0, 0.3, 0])
        q = Text("a head with nothing useful to look at…", font_size=24, color=GREY_A).move_to([0, -0.4, 0])
        self.at("nothing")
        self.play(FadeIn(toks), FadeIn(q), run_time=0.5)
        arrows = VGroup(*[CurvedArrow(toks[k].get_top() + 0.05 * UP, toks[0].get_top() + 0.05 * UP, angle=1.5,
                                      color=GOLD, stroke_width=3) for k in (2, 4, 5)])
        self.at("first")
        self.play(*[Create(a) for a in arrows], toks[0][0].animate.set_fill(GOLD, 0.6).set_stroke(GOLD), run_time=0.7)
        see = Text("…parks it on the one token every query can see", font_size=24, color=GOLD).move_to([0, -1.2, 0])
        self.play(FadeIn(see), run_time=0.4)
        sink = Text("an attention sink", font_size=34, color=YELLOW).move_to([0, -2.4, 0])
        self.at("sink")
        self.play(FadeIn(sink), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 5. no-op
    def noop(self):
        self.section(5)
        self.clear_stage()
        head = Text("size of value vectors (layers 2–11)", font_size=28).to_edge(UP, buff=0.8)
        self.at("free")
        self.play(FadeIn(head), run_time=0.4)
        b1 = Rectangle(width=0.9, height=1.42 * 0.6, stroke_width=0, fill_color=GOLD, fill_opacity=0.85)
        b2 = Rectangle(width=0.9, height=6.42 * 0.6, stroke_width=0, fill_color=BLUE_C, fill_opacity=0.85)
        b1.move_to([-1.5, -2.2 + b1.height / 2, 0])
        b2.move_to([1.5, -2.2 + b2.height / 2, 0])
        l1 = VGroup(Text("1.42", font=MONO, font_size=26).next_to(b1, UP, buff=0.1),
                    Text("first token", font_size=22).next_to(b1, DOWN, buff=0.15))
        l2 = VGroup(Text("6.42", font=MONO, font_size=26).next_to(b2, UP, buff=0.1),
                    Text("other tokens", font_size=22).next_to(b2, DOWN, buff=0.15))
        self.at("small")
        self.play(GrowFromEdge(b1, DOWN), FadeIn(l1), run_time=0.5)
        self.at("6")
        self.play(GrowFromEdge(b2, DOWN), FadeIn(l2), run_time=0.5)
        cap = Text("attending to the sink adds almost nothing", font_size=24, color=YELLOW).move_to([0, -3.3, 0])
        self.at("nothing")
        self.play(FadeIn(cap), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 6. streaming
    def streaming(self):
        self.section(6)
        self.clear_stage()
        n = 20
        boxes = VGroup(*[Square(0.42, stroke_width=1, color=GREY_B) for _ in range(n)]).arrange(RIGHT, buff=0.05)
        boxes.move_to([0, 2.0, 0])
        self.at("stream")
        self.play(FadeIn(boxes, lag_ratio=0.02), run_time=0.6)
        win = SurroundingRectangle(VGroup(*boxes[12:]), color=BLUE_C, buff=0.05)
        wl = Text("window: last 256 tokens", font_size=20, color=BLUE_C).next_to(win, UP, buff=0.1)
        self.at("window")
        self.play(Create(win), FadeIn(wl), *[boxes[k].animate.set_fill(BLUE_C, 0.5) for k in range(12, n)], run_time=0.5)
        drop = Text("dropped, including the first", font_size=20, color=RED_B).next_to(VGroup(*boxes[:12]), DOWN, 0.15)
        self.at("dropping")
        self.play(*[boxes[k].animate.set_opacity(0.25) for k in range(12)], FadeIn(drop), run_time=0.5)
        rows = [("full attention", 48, GREY_B), ("window 256", 2133, RED_C)]
        bars = VGroup()
        for k, (name, ppl, col) in enumerate(rows):
            y = -0.2 - 1.0 * k
            lab = Text(name, font_size=24).move_to([-2.6, y, 0], aligned_edge=RIGHT)
            w = min(8.0, 0.02 * ppl) if ppl < 100 else 8.0
            b = Rectangle(width=max(w, 0.9), height=0.55, stroke_width=0, fill_color=col, fill_opacity=0.85)
            b.move_to([-2.3, y, 0], aligned_edge=LEFT)
            num = Text(f"perplexity {ppl:,}", font=MONO, font_size=22).next_to(b, RIGHT, buff=0.15)
            if num.get_right()[0] > 6.9:
                num.move_to(b).set_color(BLACK)
            bars.add(VGroup(lab, b, num))
        cap = Text("GPT-2, positions 512–1,022 of 1,024 tokens of Shakespeare", font_size=18, color=GREY_B)
        cap.move_to([0, -2.4, 0])
        self.at("48")
        self.play(FadeIn(bars[0]), FadeIn(cap), run_time=0.4)
        self.at("2")
        self.play(FadeIn(bars[1]), run_time=0.5)
        col = Text("the model collapses", font_size=28, color=RED_B).move_to([0, -3.1, 0])
        self.at("collapses")
        self.play(FadeIn(col), run_time=0.4)
        self.boxes, self.win, self.bars = boxes, win, bars
        self.end_section()

    # ------------------------------------------------------------------ 7. fix
    def fix(self):
        self.section(7)
        keep = [self.boxes, self.win, *self.bars]
        self.play(*[FadeOut(m) for m in self.mobjects if m not in keep], run_time=0.4)
        sinks = VGroup(*self.boxes[:4])
        self.at("four")
        self.play(*[b.animate.set_opacity(1).set_fill(GOLD, 0.7).set_stroke(GOLD) for b in sinks], run_time=0.5)
        sl = Text("+ 4 sink tokens", font_size=20, color=GOLD).next_to(sinks, UP, buff=0.12)
        self.play(FadeIn(sl), run_time=0.3)
        y = -0.2 - 2.0
        lab = Text("window 256 + 4 sinks", font_size=24).move_to([-2.6, y, 0], aligned_edge=RIGHT)
        b = Rectangle(width=0.9, height=0.55, stroke_width=0, fill_color=GOLD, fill_opacity=0.85)
        b.move_to([-2.3, y, 0], aligned_edge=LEFT)
        num = Text("perplexity 43", font=MONO, font_size=22).next_to(b, RIGHT, buff=0.15)
        self.at("43")
        self.play(FadeIn(lab), GrowFromEdge(b, LEFT), FadeIn(num), run_time=0.5)
        name = Text("“streaming with attention sinks”", font_size=24, color=YELLOW).move_to([0, -3.2, 0])
        self.at("streaming")
        self.play(FadeIn(name), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 8. design
    def design(self):
        self.section(8)
        self.clear_stage()
        a = Text("newer designs: an explicit way to attend to nothing", font_size=28).move_to([0, 1.0, 0])
        self.at("explicit")
        self.play(FadeIn(a), run_time=0.4)
        b = Text("e.g. a learned sink score per head", font_size=26, color=GOLD).move_to([0, 0.0, 0])
        self.at("learned")
        self.play(FadeIn(b), run_time=0.4)
        c = Text("no need to borrow the first token", font_size=24, color=GREY_A).move_to([0, -1.0, 0])
        self.at("borrow")
        self.play(FadeIn(c), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 9. code
    def code(self):
        self.section(9)
        self.clear_stage()
        code, hl = code_panel(CODE, font_size=24)
        code.move_to([0, 0.3, 0])
        self.at("code")
        self.play(FadeIn(code, shift=0.2 * UP), run_time=0.5)
        hl.match_y(code.line_numbers[0])
        self.at("window")
        self.play(Create(hl), run_time=0.3)
        self.at("first")
        self.play(highlight(hl, code, 1), run_time=0.3)
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
