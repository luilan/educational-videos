"""How LLMs Work: Deep Dive, episode 29 — State-Space Models.

Render from the repo root:  ./render.sh deep-dive d29
Every number on screen comes from code/d29_state_space/state_space.py (4-layer tiny language models with attention, a
fixed-decay recurrence and a selective recurrence, 2,000 steps; per-layer memory and time per token on this CPU).
"""
from manim import *

from common import MODEL_COLOR, MONO, TOKEN_COLOR, code_panel, finish, highlight, next_up_card, token
from intro import play_token_intro
from d29_script import NEXT, TAGLINE, TITLE
from voiced_scene import VoicedScene

SERIES_LABEL = "HOW LLMs WORK: DEEP DIVE  ·  EPISODE 29"
ATT_C, FIX_C, SEL_C = BLUE_C, GREY_B, GREEN_C
CODE = """a = torch.sigmoid(self.decay(x))      # selective: a decay for every token
h = torch.zeros(B, D)                 # the whole memory: D numbers
for i in range(T):
    h = a[:, i] * h + (1 - a[:, i]) * u[:, i]    # one update per token
    outs.append(h)"""


class StateSpaceVideo(VoicedScene):
    VIDEO = "d29"

    def construct(self):
        play_token_intro(self, TITLE, 29, TAGLINE, label=SERIES_LABEL)
        self.hook()          # 1
        self.recurrence()    # 2
        self.selective()     # 3
        self.experiment()    # 4
        self.caveat()        # 5
        self.memory()        # 6
        self.timing()        # 7
        self.hybrids()       # 8
        self.code()          # 9
        self.outro()         # 10

    def clear_stage(self, *keep, run_time=0.6):
        self.play(*[FadeOut(m) for m in self.mobjects if m not in keep], run_time=run_time)

    # ------------------------------------------------------------------ 1. hook
    def hook(self):
        self.section(1)
        n = 8
        tri = VGroup(*[Square(0.3, stroke_width=0.6, color=GREY_D, fill_color=ATT_C if j <= i else BLACK, fill_opacity=0.8)
                       .move_to([j * 0.3, -i * 0.3, 0]) for i in range(n) for j in range(n)]).move_to([-3.6, 0.2, 0])
        tl = Text("attention: every pair,\na cache that grows", font_size=22, color=ATT_C, line_spacing=0.85)
        tl.next_to(tri, DOWN, buff=0.3)
        self.at("attention")
        self.play(FadeIn(tri, lag_ratio=0.01), FadeIn(tl), run_time=0.7)
        toks = VGroup(*[Square(0.4, color=TOKEN_COLOR, fill_opacity=0.3) for _ in range(6)]).arrange(RIGHT, buff=0.25)
        toks.move_to([3.2, -0.4, 0])
        state = RoundedRectangle(width=0.9, height=0.6, corner_radius=0.1, color=SEL_C, fill_opacity=0.4).move_to([0.8, 0.7, 0])
        sl = Text("state", font_size=18).move_to(state)
        sm = Text("state space: a fixed-size memory,\nupdated once per token", font_size=22, color=SEL_C,
                  line_spacing=0.85).move_to([3.2, -1.6, 0])
        self.at("fixed")
        self.play(FadeIn(toks), FadeIn(state), FadeIn(sl), FadeIn(sm), run_time=0.5)
        for k in range(6):
            self.play(state.animate.move_to(toks[k].get_top() + 0.6 * UP), sl.animate.move_to(toks[k].get_top() + 0.6 * UP),
                      run_time=0.2)
        self.end_section()

    # ------------------------------------------------------------------ 2. recurrence
    def recurrence(self):
        self.section(2)
        self.clear_stage()
        eq = Text("h  ←  a · h  +  (1 − a) · u", font=MONO, font_size=36).to_edge(UP, buff=0.7)
        self.at("channel")
        self.play(FadeIn(eq), run_time=0.5)
        ax = Axes(x_range=[0, 30, 10], y_range=[0, 1.05, 0.5], x_length=7, y_length=3.2, tips=False,
                  axis_config={"color": GREY_B}).move_to([0.0, -0.8, 0])
        xl = Text("steps ago", font_size=18, color=GREY_B).next_to(ax, DOWN, buff=0.15)
        yl = Text("how much an input\nstill counts", font_size=18, color=GREY_B, line_spacing=0.8).next_to(ax, LEFT, buff=0.15)
        self.at("decaying")
        self.play(Create(ax), FadeIn(xl), FadeIn(yl), run_time=0.5)
        curves = VGroup()
        for a, col in ((0.5, RED_C), (0.9, GOLD), (0.99, GREEN_C)):
            c = ax.plot(lambda s, a=a: a ** s, x_range=[0, 30], color=col, stroke_width=4)
            lab = Text(f"a = {a}", font=MONO, font_size=20, color=col).move_to([5.2, 0.6 - 0.7 * len(curves) / 2, 0])
            curves.add(VGroup(c, lab))
        for k, g in enumerate(curves):
            g[1].move_to([5.2, 0.2 - 0.6 * k, 0])
        self.at("fixed")
        self.play(Create(curves[0][0]), FadeIn(curves[0][1]), run_time=0.5)
        self.at("long")
        self.play(*[AnimationGroup(Create(g[0]), FadeIn(g[1])) for g in curves[1:]], run_time=0.7)
        self.end_section()

    # ------------------------------------------------------------------ 3. selective
    def selective(self):
        self.section(3)
        self.clear_stage()
        head = Text("selective: a depends on the token itself", font_size=30, color=SEL_C).to_edge(UP, buff=0.7)
        self.at("mamba")
        self.play(FadeIn(head), run_time=0.4)
        words = ["The", "king", ",", "who", "was", "tired", ",", "slept"]
        avals = [0.6, 0.97, 0.3, 0.8, 0.5, 0.95, 0.3, 0.9]
        toks = VGroup(*[token(w, font_size=22) for w in words]).arrange(RIGHT, buff=0.18).move_to([0, 0.6, 0])
        bars = VGroup(*[Rectangle(width=0.5, height=a * 1.6, stroke_width=0, fill_color=SEL_C, fill_opacity=0.8)
                        .next_to(t, DOWN, buff=0.3).align_to(toks, DOWN).shift(0.3 * DOWN) for t, a in zip(toks, avals)])
        for b, t in zip(bars, toks):
            b.move_to([t.get_x(), -0.7 - b.height / 2 + 0.8, 0]).align_to(Point([0, -0.2, 0]), UP)
        bl = Text("a for each token (illustration): high = keep the past, low = make room", font_size=20, color=GREY_A)
        bl.move_to([0, -2.6, 0])
        self.at("depend")
        self.play(FadeIn(toks), run_time=0.4)
        self.at("choose")
        self.play(LaggedStart(*[GrowFromEdge(b, UP) for b in bars], lag_ratio=0.08), FadeIn(bl), run_time=0.8)
        self.end_section()

    # ------------------------------------------------------------------ 4. experiment
    def experiment(self):
        self.section(4)
        self.clear_stage()
        head = Text("tiny language model, 4 layers, 2,000 steps: validation loss", font_size=26).to_edge(UP, buff=0.6)
        self.at("replace")
        self.play(FadeIn(head), run_time=0.4)
        n = Text("no position embedding needed: a recurrence knows the order", font_size=22, color=GREY_A).move_to([0, 2.0, 0])
        self.at("position")
        self.play(FadeIn(n), run_time=0.3)
        rows = [("attention", 1.644, ATT_C, "644"), ("recurrence, fixed decay", 1.648, FIX_C, "648"),
                ("recurrence, selective decay", 1.591, SEL_C, "591")]
        out = VGroup()
        for k, (name, v, col, cue) in enumerate(rows):
            y = 0.9 - 1.0 * k
            lab = Text(name, font_size=24, color=col).move_to([-0.4, y, 0], aligned_edge=RIGHT)
            b = Rectangle(width=(v - 1.5) * 25, height=0.55, stroke_width=0, fill_color=col, fill_opacity=0.85)
            b.move_to([-0.1, y, 0], aligned_edge=LEFT)
            out.add(VGroup(lab, b, Text(f"{v:.3f}", font=MONO, font_size=24).next_to(b, RIGHT, buff=0.15)))
            self.at(cue)
            self.play(FadeIn(out[k]), run_time=0.4)
        cap = Text("bars start at 1.5", font_size=16, color=GREY_B).next_to(out, DOWN, buff=0.3)
        self.play(FadeIn(cap), run_time=0.2)
        self.end_section()

    # ------------------------------------------------------------------ 5. caveat
    def caveat(self):
        self.section(5)
        self.clear_stage()
        items = VGroup(Text("here = 64-character texts: nearby characters matter most", font_size=26),
                       Text("a fixed-size state must squeeze the whole past", font_size=26),
                       Text("exact recall from far back: where attention keeps its edge", font_size=26, color=YELLOW))
        items.arrange(DOWN, buff=0.45).move_to([0, 0.3, 0])
        for k, cue in enumerate(("64", "squeeze", "recall")):
            self.at(cue)
            self.play(FadeIn(items[k]), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 6. memory
    def memory(self):
        self.section(6)
        self.clear_stage()
        head = Text("memory per layer while generating (128 numbers per token, float32)", font_size=24).to_edge(UP, buff=0.6)
        self.at("inference")
        self.play(FadeIn(head), run_time=0.4)
        xs = [-3.6, 0.4, 4.0]
        hdr = VGroup(*[Text(t, font_size=22, color=GREY_B).move_to([x, 1.4, 0]) for t, x in
                       zip(("after", "attention KV cache", "recurrent state"), xs)])
        rows = [("1,000 tokens", "0.98 MiB", "0.50 KiB", "thousand"), ("100,000 tokens", "97.66 MiB", "0.50 KiB", "98")]
        self.play(FadeIn(hdr), run_time=0.3)
        for k, (a, b, c, cue) in enumerate(rows):
            row = VGroup(Text(a, font_size=24).move_to([xs[0], 0.5 - 0.8 * k, 0]),
                         Text(b, font=MONO, font_size=26, color=ATT_C).move_to([xs[1], 0.5 - 0.8 * k, 0]),
                         Text(c, font=MONO, font_size=26, color=SEL_C).move_to([xs[2], 0.5 - 0.8 * k, 0]))
            self.at(cue)
            self.play(FadeIn(row), run_time=0.4)
        f = Text("the state never grows", font_size=26, color=YELLOW).move_to([0, -1.8, 0])
        self.at("forever")
        self.play(FadeIn(f), run_time=0.3)
        self.end_section()

    # ------------------------------------------------------------------ 7. time
    def timing(self):
        self.section(7)
        self.clear_stage()
        head = Text("time to process one new token in one layer (this CPU)", font_size=26).to_edge(UP, buff=0.6)
        self.at("time")
        self.play(FadeIn(head), run_time=0.3)
        xs = [-3.6, 0.4, 4.0]
        hdr = VGroup(*[Text(t, font_size=22, color=GREY_B).move_to([x, 1.4, 0]) for t, x in
                       zip(("context", "attention", "recurrence"), xs)])
        self.play(FadeIn(hdr), run_time=0.3)
        rows = [("1,000 tokens", "37.9 µs", "6.3 µs", "38"), ("10,000 tokens", "312.6 µs", "6.4 µs", "313"),
                ("100,000 tokens", "11,334 µs", "6.4 µs", "11")]
        for k, (a, b, c, cue) in enumerate(rows):
            row = VGroup(Text(a, font_size=24).move_to([xs[0], 0.5 - 0.8 * k, 0]),
                         Text(b, font=MONO, font_size=26, color=ATT_C).move_to([xs[1], 0.5 - 0.8 * k, 0]),
                         Text(c, font=MONO, font_size=26, color=SEL_C).move_to([xs[2], 0.5 - 0.8 * k, 0]))
            self.at(cue)
            self.play(FadeIn(row[:2]), run_time=0.3)
            self.add(row[2].set_opacity(0))
            setattr(self, f"rec{k}", row[2])
        self.at("six")
        self.play(*[getattr(self, f"rec{k}").animate.set_opacity(1) for k in range(3)], run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 8. hybrids
    def hybrids(self):
        self.section(8)
        self.clear_stage()
        layers = VGroup()
        for k in range(8):
            att = k % 4 == 3
            r = Rectangle(width=3.2, height=0.42, stroke_width=1, color=ATT_C if att else SEL_C, fill_opacity=0.35)
            layers.add(VGroup(r, Text("attention" if att else "Mamba-style", font_size=18).move_to(r)))
        layers.arrange(UP, buff=0.08).move_to([-2.4, 0.0, 0])
        self.at("mix")
        self.play(FadeIn(layers, lag_ratio=0.08), run_time=0.8)
        t = Text("AI21's Jamba: Mamba layers\ninterleaved with a few attention layers", font_size=24,
                 line_spacing=0.85).move_to([3.2, 0.8, 0])
        self.at("jamba")
        self.play(FadeIn(t), run_time=0.4)
        n = Text("cheap memory for most of the work,\nexact lookup where it's needed", font_size=22, color=YELLOW,
                 line_spacing=0.85).move_to([3.2, -0.8, 0])
        self.at("exact")
        self.play(FadeIn(n), run_time=0.4)
        cap = Text("(layer pattern illustrative)", font_size=16, color=GREY_B).next_to(layers, DOWN, buff=0.15)
        self.play(FadeIn(cap), run_time=0.2)
        self.end_section()

    # ------------------------------------------------------------------ 9. code
    def code(self):
        self.section(9)
        self.clear_stage()
        code, hl = code_panel(CODE, font_size=24)
        code.move_to([0, 0.3, 0])
        self.at("code")
        self.play(FadeIn(code, shift=0.2 * UP), run_time=0.5)
        hl.match_y(code.line_numbers[3])
        self.at("loop")
        self.play(Create(hl), run_time=0.3)
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
