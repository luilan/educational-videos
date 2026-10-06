"""How LLMs Work: Deep Dive, episode 41 — The Logit Lens.

Render from the repo root:  ./render.sh deep-dive d41
Every number on screen comes from code/d41_logit_lens/logit_lens.py (GPT-2 small and Qwen2.5-0.5B; final norm and
unembedding applied to the hidden state after each layer; layer 0 = embeddings; agreement over 1,024 tokens of Tiny
Shakespeare).
"""
from manim import *

from common import MODEL_COLOR, MONO, TOKEN_COLOR, code_panel, finish, highlight, next_up_card
from intro import play_token_intro
from d41_script import NEXT, TAGLINE, TITLE
from voiced_scene import VoicedScene

SERIES_LABEL = "HOW LLMs WORK: DEEP DIVE  ·  EPISODE 41"
PARIS = [("destro", 0.00), ("the", 0.00), ("the", 0.00), ("the", 0.00), ("the", 0.00), ("the", 0.00), ("the", 0.00),
         ("East", 0.00), ("Ing", 0.00), ("Rome", 0.03), ("London", 0.25), ("Paris", 0.18), ("Paris", 0.07)]
SHAKE = [("William", 0.00), ("William", 0.00), ("William", 0.00), ("William", 0.00), ("William", 0.02), ("William", 0.01),
         ("William", 0.07), ("William", 0.15), ("Shakespeare", 0.36), ("Shakespeare", 0.91), ("Shakespeare", 1.00),
         ("Shakespeare", 0.96), ("Shakespeare", 0.21)]
GPT2_AGREE = [0.0, 13.4, 10.2, 8.7, 6.6, 6.9, 11.4, 15.3, 30.0, 35.1, 44.5, 56.3, 100.0]
QWEN_AGREE = [0.2, 0.2, 0.1, 0.1, 0.3, 0.2, 0.5, 1.5, 1.2, 0.9, 1.1, 1.3, 0.8, 1.9, 1.6, 1.3, 2.9, 4.3, 10.0, 8.4, 14.9,
              19.9, 28.3, 45.7, 100.0]
CODE = """h = out.hidden_states[layer][0, -1]        # after any layer
logits = model.lm_head(model.transformer.ln_f(h))
print(tok.decode(logits.argmax()))"""


def lens_table(rows, target, title):
    g = VGroup()
    hdr = VGroup(Text("layer", font_size=18, color=GREY_B), Text("top guess", font_size=18, color=GREY_B),
                 Text(f"p({target})", font_size=18, color=GREY_B))
    for k, (x, m) in enumerate(zip((-4.6, -2.6, 0.4), hdr)):
        m.move_to([x, 2.3, 0])
    for i, (w, p) in enumerate(rows):
        y = 1.9 - 0.36 * i
        name = "out" if i == len(rows) - 1 else str(i)
        col = GREEN_B if w.strip() == target else WHITE
        b = Rectangle(width=max(p * 4.0, 0.02), height=0.24, stroke_width=0, fill_color=GREEN_C, fill_opacity=0.85)
        b.move_to([-0.6, y, 0], aligned_edge=LEFT)
        g.add(VGroup(Text(name, font=MONO, font_size=18).move_to([-4.6, y, 0]),
                     Text(w, font=MONO, font_size=18, color=col).move_to([-2.6, y, 0]), b,
                     Text(f"{p:.2f}", font=MONO, font_size=16).next_to(b, RIGHT, buff=0.1)))
    head = Text(title, font_size=24).to_edge(UP, buff=0.3)
    return head, hdr, g


def agree_chart(values, n_layers, color, x_len=8.5):
    ax = Axes(x_range=[0, n_layers, 4 if n_layers > 12 else 2], y_range=[0, 100, 25], x_length=x_len, y_length=3.8,
              tips=False, axis_config={"color": GREY_B}).move_to([0.3, -0.4, 0])
    ticks = VGroup(*[Text(str(v), font_size=16).next_to(ax.c2p(v, 0), DOWN, buff=0.1) for v in range(0, n_layers + 1, 4 if n_layers > 12 else 2)],
                   *[Text(f"{v}%", font_size=16).next_to(ax.c2p(0, v), LEFT, buff=0.1) for v in (0, 25, 50, 75, 100)])
    xl = Text("layer", font_size=20).next_to(ax.x_axis, DOWN, buff=0.4)
    line = VMobject(color=color).set_points_as_corners([ax.c2p(i, v) for i, v in enumerate(values)])
    dots = VGroup(*[Dot(ax.c2p(i, v), radius=0.05, color=color) for i, v in enumerate(values)])
    return ax, VGroup(ticks, xl), line, dots


class LogitLensVideo(VoicedScene):
    VIDEO = "d41"

    def construct(self):
        play_token_intro(self, TITLE, 41, TAGLINE, label=SERIES_LABEL)
        self.hook()          # 1
        self.method()        # 2
        self.paris()         # 3
        self.shakespeare()   # 4
        self.over_text()     # 5
        self.qwen()          # 6
        self.caveat()        # 7
        self.code()          # 8
        self.outro()         # 9

    def clear_stage(self, *keep, run_time=0.6):
        self.play(*[FadeOut(m) for m in self.mobjects if m not in keep], run_time=run_time)

    def stack(self, n=6):
        blocks = VGroup(*[RoundedRectangle(width=2.6, height=0.5, corner_radius=0.08, color=MODEL_COLOR, fill_opacity=0.3)
                          for _ in range(n)]).arrange(UP, buff=0.12)
        labels = VGroup(*[Text(f"layer {i + 1}", font_size=18).move_to(b) for i, b in enumerate(blocks)])
        return blocks, labels

    # ------------------------------------------------------------------ 1. hook
    def hook(self):
        self.section(1)
        blocks, labels = self.stack()
        g = VGroup(blocks, labels).move_to([-2.5, 0, 0])
        self.at("layer")
        self.play(FadeIn(g, lag_ratio=0.1), run_time=0.8)
        q = Text("?", font_size=60, color=YELLOW).next_to(blocks[2], RIGHT, buff=1.2)
        self.at("halfway")
        self.play(FadeIn(q), Indicate(blocks[2], color=YELLOW), run_time=0.6)
        eye = Text("peek at every layer", font_size=26, color=YELLOW).move_to([2.5, -2.6, 0])
        self.at("read")
        self.play(FadeIn(eye), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 2. method
    def method(self):
        self.section(2)
        self.clear_stage()
        blocks, labels = self.stack()
        g = VGroup(blocks, labels).move_to([-4.0, 0, 0])
        self.at("residual")
        self.play(FadeIn(g), run_time=0.4)
        box = RoundedRectangle(width=4.4, height=0.9, corner_radius=0.1, color=GREEN_C, fill_opacity=0.25).move_to([0.6, 2.6, 0])
        bl = Text("final norm + unembedding", font_size=22).move_to(box)
        out = Text("→ words", font_size=24, color=GREEN_B).next_to(box, RIGHT, buff=0.2)
        self.at("words")
        self.play(GrowArrow(Arrow(blocks[-1].get_right(), box.get_left(), buff=0.1)), FadeIn(box), FadeIn(bl), FadeIn(out), run_time=0.6)
        arrows = VGroup(*[DashedLine(b.get_right(), [1.0, b.get_y(), 0], color=YELLOW) for b in blocks[:-1]])
        mini = VGroup(*[Text("norm + unembed → word", font_size=16, color=YELLOW).next_to(a, RIGHT, buff=0.1) for a in arrows])
        self.at("intermediate")
        self.play(Create(arrows, lag_ratio=0.1), FadeIn(mini, lag_ratio=0.1), run_time=0.9)
        n = Text("no training, just a peek", font_size=24, color=YELLOW).move_to([1.5, -2.9, 0])
        self.at("training")
        self.play(FadeIn(n), run_time=0.3)
        self.end_section()

    def reveal_table(self, rows, target, title, cues):
        self.clear_stage()
        head, hdr, g = lens_table(rows, target, title)
        self.play(FadeIn(head), FadeIn(hdr), run_time=0.4)
        shown = 0
        for upto, cue in cues:
            self.at(cue)
            self.play(FadeIn(g[shown:upto], lag_ratio=0.15), run_time=min(0.3 + 0.08 * (upto - shown), 1.2))
            shown = upto
        return g

    # ------------------------------------------------------------------ 3. Paris
    def paris(self):
        self.section(3)
        g = self.reveal_table(PARIS, "Paris", "GPT-2: “The Eiffel Tower is in the city of”",
                              [(9, "filler"), (10, "rome"), (11, "london"), (13, "paris")])
        n = Text("a city → a European capital → the right one", font_size=22, color=YELLOW).move_to([4.2, 0.2, 0]).scale_to_fit_width(5.0)
        self.at("narrows")
        self.play(FadeIn(n), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 4. Shakespeare
    def shakespeare(self):
        self.section(4)
        g = self.reveal_table(SHAKE, "Shakespeare", "GPT-2: “Romeo and Juliet was written by William”",
                              [(8, "echo"), (10, "shakespeare"), (12, "100"), (13, "back")])
        n = Text("the final layer hedges", font_size=24, color=YELLOW).move_to([4.6, -2.4, 0])
        self.at("hedges")
        self.play(FadeIn(n), run_time=0.3)
        self.end_section()

    # ------------------------------------------------------------------ 5. over text
    def over_text(self):
        self.section(5)
        self.clear_stage()
        head = Text("GPT-2: lens guess = final prediction (1,024 tokens)", font_size=26).to_edge(UP, buff=0.5)
        ax, deco, line, dots = agree_chart(GPT2_AGREE, 12, GREEN_B)
        self.at("thousand")
        self.play(FadeIn(head), Create(ax), FadeIn(deco), run_time=0.6)
        self.play(Create(line), FadeIn(dots, lag_ratio=0.1), run_time=1.2)
        labs = VGroup()
        for layer, cue in ((5, "7"), (8, "30"), (11, "56")):
            t = Text(f"{GPT2_AGREE[layer]:.0f}%", font=MONO, font_size=20, color=YELLOW).next_to(dots[layer], UP, buff=0.15)
            self.at(cue)
            self.play(FadeIn(t), Indicate(dots[layer], color=YELLOW), run_time=0.5)
        m = Text("most of the decision is made late", font_size=24, color=YELLOW).move_to([0, 2.3, 0])
        self.at("late")
        self.play(FadeIn(m), run_time=0.3)
        self.end_section()

    # ------------------------------------------------------------------ 6. Qwen
    def qwen(self):
        self.section(6)
        self.clear_stage()
        head = Text("Qwen2.5-0.5B: lens guess = final prediction", font_size=26).to_edge(UP, buff=0.5)
        ax, deco, line, dots = agree_chart(QWEN_AGREE, 24, BLUE_B)
        self.at("different")
        self.play(FadeIn(head), Create(ax), FadeIn(deco), Create(line), FadeIn(dots), run_time=1.0)
        junk = Text("layer 10 on “…the city of”:  ().'/      layer 15:  PushMatrix", font=MONO, font_size=20, color=GREY_B)
        junk.move_to([0, 2.3, 0])
        self.at("junk")
        self.play(FadeIn(junk), run_time=0.4)
        lt = Text("< 3% until layer 16", font_size=22, color=RED_B).move_to(ax.c2p(8, 30))
        self.at("less")
        self.play(FadeIn(lt), run_time=0.3)
        p = Text("“Paris” first on top: layer 22 of 24", font_size=22, color=GREEN_B).move_to(ax.c2p(12, 60))
        self.at("paris")
        self.play(FadeIn(p), run_time=0.3)
        self.end_section()

    # ------------------------------------------------------------------ 7. caveat
    def caveat(self):
        self.section(7)
        self.clear_stage()
        t1 = Text("not “knows nothing”: its middle layers use directions\nthe unembedding can't read", font_size=26,
                  line_spacing=0.9).move_to([0, 1.8, 0])
        self.at("nothing")
        self.play(FadeIn(t1), run_time=0.5)
        blocks = VGroup(*[RoundedRectangle(width=1.5, height=0.5, corner_radius=0.08, color=MODEL_COLOR, fill_opacity=0.3)
                          for _ in range(4)]).arrange(RIGHT, buff=0.6).move_to([0, 0, 0])
        tr = VGroup(*[Square(0.4, color=GOLD, fill_opacity=0.4).next_to(b, DOWN, buff=0.35) for b in blocks])
        tl = Text("tuned lens: a small learned translator per layer", font_size=22, color=GOLD).move_to([0, -1.3, 0])
        self.at("tuned")
        self.play(FadeIn(blocks), FadeIn(tr, lag_ratio=0.1), FadeIn(tl), run_time=0.6)
        w = Text("the plain lens: a quick, imperfect window", font_size=24, color=YELLOW).move_to([0, -2.5, 0])
        self.at("window")
        self.play(FadeIn(w), run_time=0.3)
        self.end_section()

    # ------------------------------------------------------------------ 8. code
    def code(self):
        self.section(8)
        self.clear_stage()
        code, hl = code_panel(CODE, font_size=26)
        code.move_to([0, 0.3, 0])
        self.at("code")
        self.play(FadeIn(code, shift=0.2 * UP), run_time=0.5)
        hl.match_y(code.line_numbers[0])
        self.at("hidden")
        self.play(Create(hl), run_time=0.3)
        self.at("norm")
        self.play(highlight(hl, code, 1), run_time=0.3)
        self.end_section()

    # ------------------------------------------------------------------ 9. outro
    def outro(self):
        self.section(9)
        self.clear_stage(run_time=0.4)
        card = next_up_card(NEXT)
        self.at("next")
        self.play(FadeIn(card, shift=0.2 * UP), run_time=0.6)
        self.end_section()
        finish(self, hold=1.5)
