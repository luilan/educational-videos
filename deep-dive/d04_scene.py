"""How LLMs Work: Deep Dive, episode 4 — Why Attention Needs Position.

Render from the repo root:  ./render.sh deep-dive d04
Every similarity and difference on screen comes from code/d04_position/position.py (GPT-2 small, real weights);
the curve is the cosine similarity between GPT-2's learned position vector 100 and positions 0–400.
"""
from manim import *

from common import MODEL_COLOR, MONO, TOKEN_COLOR, code_panel, finish, highlight, next_up_card, token
from intro import play_token_intro
from d04_script import NEXT, TAGLINE, TITLE
from voiced_scene import VoicedScene

SERIES_LABEL = "HOW LLMs WORK: DEEP DIVE  ·  EPISODE 4"
CURVE = [0.299, 0.252, 0.257, 0.364, 0.495, 0.638, 0.771, 0.869, 0.938, 0.983, 1.0, 0.984, 0.949, 0.897, 0.829,
         0.757, 0.68, 0.599, 0.515, 0.43, 0.35, 0.27, 0.187, 0.11, 0.038, -0.026, -0.082, -0.129, -0.17, -0.203,
         -0.225, -0.239, -0.246, -0.247, -0.241, -0.227, -0.21, -0.183, -0.154, -0.122, -0.087]
CODE = """def run(ids, use_position):
    x = model.wte(ids)                                   # token embeddings
    if use_position:
        x = x + model.wpe(torch.arange(ids.shape[1]))    # + learned position vectors
    for block in model.h:                                # 12 transformer blocks
        x = block(x, attention_mask=no_mask)[0]
    return model.ln_f(x)[0]

cosine_similarity(run(a, False)[0], run(b, False)[2])   # dog vs dog: 1.000000"""


def labeled_box(text, color=MODEL_COLOR, width=2.4, height=1.0, font_size=26):
    label = Text(text, font_size=font_size, line_spacing=0.8)
    box = RoundedRectangle(corner_radius=0.2, width=max(width, label.width + 0.5), height=height, stroke_color=color,
                           fill_color=color, fill_opacity=0.25)
    return VGroup(box, label.move_to(box))


def sentence(words, color=TOKEN_COLOR, font_size=28):
    return VGroup(*[token(w, color, font_size) for w in words]).arrange(RIGHT, buff=0.12)


def heat(values, cell=0.6):
    g = VGroup()
    for i, row in enumerate(values):
        for j, v in enumerate(row):
            sq = Square(cell, stroke_width=1, stroke_color=BLACK, fill_color=YELLOW, fill_opacity=0.15 + 0.8 * v)
            sq.move_to([j * cell, -i * cell, 0])
            g.add(sq)
    return g


class PositionVideo(VoicedScene):
    VIDEO = "d04"

    def construct(self):
        play_token_intro(self, TITLE, 4, TAGLINE, label=SERIES_LABEL)
        self.hook()          # 1
        self.why()           # 2
        self.experiment()    # 3
        self.on()            # 4
        self.causal()        # 5
        self.learned()       # 6
        self.limit()         # 7
        self.code()          # 8
        self.outro()         # 9

    def clear_stage(self, *keep, run_time=0.6):
        self.play(*[FadeOut(m) for m in self.mobjects if m not in keep], run_time=run_time)

    # ------------------------------------------------------------------ 1. hook
    def hook(self):
        self.section(1)
        s1 = sentence(["dog", "bites", "man"]).move_to([0, 1.6, 0])
        s2 = sentence(["man", "bites", "dog"]).move_to([0, 0.5, 0])
        self.at("dog")
        self.play(FadeIn(s1), run_time=0.5)
        self.play(FadeIn(s2), run_time=0.5)
        news = Text("same three words, opposite news", font_size=30, color=YELLOW).move_to([0, -0.8, 0])
        self.at("news")
        self.play(FadeIn(news), run_time=0.4)
        blind = Text("attention has no idea of order", font_size=30, color=RED_B).move_to([0, -1.8, 0])
        self.at("attention")
        self.play(FadeIn(blind), run_time=0.4)
        proof = Text("proof, with real GPT-2 weights", font_size=24, color=GREY_B).move_to([0, -2.6, 0])
        self.at("prove")
        self.play(FadeIn(proof), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 2. why attention is order-blind
    def why(self):
        self.section(2)
        self.clear_stage()
        words = ["dog", "bites", "man"]
        w = [[0.9, 0.3, 0.6], [0.4, 0.8, 0.5], [0.6, 0.2, 0.9]]
        grid = heat(w, cell=1.0).move_to([-3.4, 0.0, 0])
        rows = VGroup(*[Text(t, font_size=22).next_to(grid[3 * i], LEFT, buff=0.2) for i, t in enumerate(words)])
        cols = VGroup(*[Text(t, font_size=22).next_to(grid[j], UP, buff=0.2) for j, t in enumerate(words)])
        cap = Text("attention weights: query · key", font_size=22, color=GREY_B).next_to(grid, DOWN, buff=0.3)
        out = Text("output = weighted average of values", font_size=26).move_to([2.6, 1.6, 0])
        self.at("output")
        self.play(FadeIn(out), run_time=0.4)
        self.at("dot")
        self.play(FadeIn(grid), FadeIn(rows), FadeIn(cols), FadeIn(cap), run_time=0.7)
        perm = [2, 1, 0]
        w2 = [[w[perm[i]][perm[j]] for j in range(3)] for i in range(3)]
        grid2 = heat(w2, cell=1.0).move_to(grid)
        rows2 = VGroup(*[Text(words[perm[i]], font_size=22).move_to(rows[i]) for i in range(3)])
        cols2 = VGroup(*[Text(words[perm[j]], font_size=22).move_to(cols[j]) for j in range(3)])
        self.at("shuffle")
        self.play(Transform(grid, grid2), Transform(rows, rows2), Transform(cols, cols2), run_time=1.0)
        same = Text("same numbers, just shuffled", font_size=26, color=YELLOW).move_to([2.6, 0.4, 0])
        self.at("same")
        self.play(FadeIn(same), run_time=0.4)
        sum_ = Text("a weighted sum doesn't care about order", font_size=26).move_to([2.6, -0.6, 0])
        self.at("order")
        self.play(FadeIn(sum_), run_time=0.4)
        name = Text("permutation equivariance", font_size=32, color=YELLOW).move_to([2.6, -1.8, 0])
        self.at("equivariance")
        self.play(FadeIn(name, scale=1.1), run_time=0.5)
        self.end_section()

    # ------------------------------------------------------------------ 3. the experiment
    def experiment(self):
        self.section(3)
        self.clear_stage()
        stack = VGroup(*[Rectangle(width=2.6, height=0.18, stroke_color=MODEL_COLOR, fill_color=MODEL_COLOR,
                                   fill_opacity=0.35) for _ in range(12)]).arrange(UP, buff=0.05).move_to([-4.4, 0.4, 0])
        lab = Text("GPT-2 small\n12 layers", font_size=22, line_spacing=0.8).next_to(stack, DOWN, buff=0.2)
        self.at("experiment")
        self.play(FadeIn(stack), FadeIn(lab), run_time=0.5)
        off = VGroup(Text("position embeddings: OFF", font_size=24, color=RED_B),
                     Text("every token sees every other", font_size=24, color=GREY_B)).arrange(DOWN, aligned_edge=LEFT)
        off.next_to(stack, UP, buff=0.3).align_to(stack, LEFT)
        self.at("switch")
        self.play(FadeIn(off), run_time=0.5)
        s1 = sentence(["dog", "bites", "man"], font_size=24).move_to([1.6, 2.2, 0])
        s2 = sentence(["man", "bites", "dog"], font_size=24).move_to([1.6, 1.2, 0])
        self.at("feed")
        self.play(FadeIn(s1), FadeIn(s2), run_time=0.5)
        link = CurvedArrow(s1[0].get_bottom(), s2[2].get_top(), angle=0.6, color=YELLOW)
        self.at("compare")
        self.play(Create(link), run_time=0.5)
        res = VGroup(Text("dog vs dog: 1.000000", font=MONO, font_size=28, color=YELLOW),
                     Text("man vs man: 1.000000", font=MONO, font_size=28, color=YELLOW),
                     Text("largest difference: 0.0002 (rounding)", font=MONO, font_size=24, color=GREY_A))
        res.arrange(DOWN, aligned_edge=LEFT, buff=0.25).move_to([1.6, -0.6, 0])
        self.at("identical")
        self.play(FadeIn(res[0], scale=1.1), run_time=0.4)
        self.at("man")
        self.play(FadeIn(res[1]), run_time=0.3)
        self.at("rounding")
        self.play(FadeIn(res[2]), run_time=0.3)
        who = Text("it cannot tell who bit whom", font_size=30, color=RED_B).move_to([1.6, -2.4, 0])
        self.at("whom")
        self.play(FadeIn(who), run_time=0.4)
        self.res, self.off, self.stack, self.who = res, off, stack, who
        self.end_section()

    # ------------------------------------------------------------------ 4. positions back on
    def on(self):
        self.section(4)
        on = Text("position embeddings: ON", font_size=24, color=GREEN_B).move_to(self.off[0], aligned_edge=LEFT)
        self.at("back")
        self.play(Transform(self.off[0], on), FadeOut(self.who), run_time=0.5)
        new = Text("dog vs dog: 0.96", font=MONO, font_size=28, color=GREEN_B).move_to(self.res[0], aligned_edge=LEFT)
        self.at("96")
        self.play(Transform(self.res[0], new), self.res[1].animate.set_opacity(0.3), run_time=0.6)
        matter = Text("now order can matter", font_size=28, color=GREEN_B).next_to(self.res, DOWN, buff=0.3)
        self.at("matter")
        self.play(FadeIn(matter), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 5. the causal twist
    def causal(self):
        self.section(5)
        self.clear_stage()
        head = Text("a twist: the causal mask", font_size=30).to_edge(UP, buff=0.5)
        self.at("twist")
        self.play(FadeIn(head), run_time=0.4)
        n = 4
        cells = VGroup()
        for i in range(n):
            for j in range(n):
                vis = j <= i
                sq = Square(0.6, stroke_width=1, stroke_color=BLACK, fill_color=BLUE_C if vis else GREY_E,
                            fill_opacity=0.8 if vis else 0.3)
                sq.move_to([j * 0.6, -i * 0.6, 0])
                cells.add(sq)
        cells.move_to([-3.2, 0.0, 0])
        cap = Text("each token sees only earlier tokens", font_size=22, color=GREY_B).next_to(cells, DOWN, buff=0.3)
        self.at("causal")
        self.play(FadeIn(cells), FadeIn(cap), run_time=0.6)
        first = Text("first token: sees itself", font_size=24).move_to([2.4, 1.2, 0])
        last = Text("last token: sees everything", font_size=24).move_to([2.4, 0.5, 0])
        self.at("first")
        self.play(FadeIn(first), Indicate(cells[0], color=YELLOW), run_time=0.5)
        self.at("last")
        self.play(FadeIn(last), Indicate(VGroup(*cells[12:]), color=YELLOW), run_time=0.5)
        res = Text("no positions + causal mask:\ndog vs dog = 0.987", font=MONO, font_size=26, color=YELLOW,
                   line_spacing=0.8).move_to([2.4, -0.8, 0])
        self.at("987")
        self.play(FadeIn(res), run_time=0.5)
        ind = Text("order leaks in, indirectly", font_size=26, color=GREY_A).move_to([2.4, -2.0, 0])
        self.at("indirect")
        self.play(FadeIn(ind), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 6. what GPT-2 learned
    def learned(self):
        self.section(6)
        self.clear_stage()
        table = VGroup(*[Rectangle(width=2.2, height=0.16, stroke_width=0, fill_color=TEAL_C,
                                   fill_opacity=0.25 + 0.5 * ((k * 37) % 10) / 10) for k in range(16)])
        table.arrange(DOWN, buff=0.04).move_to([-5.0, 0.2, 0])
        tl = Text("1,024 positions\n× 768 numbers", font_size=22, line_spacing=0.8).next_to(table, DOWN, buff=0.2)
        self.at("learn")
        self.play(FadeIn(table), run_time=0.4)
        self.at("1024")
        self.play(FadeIn(tl), run_time=0.4)
        ax = Axes(x_range=[0, 400, 100], y_range=[-0.4, 1, 0.5], x_length=7.5, y_length=4.2, tips=False,
                  axis_config={"color": GREY_B}).move_to([1.8, 0.2, 0])
        xl = VGroup(*[Text(str(v), font_size=18, color=GREY_B).next_to(ax.c2p(v, -0.4), DOWN, buff=0.1)
                      for v in (0, 100, 200, 300, 400)])
        yl = VGroup(*[Text(s, font_size=18, color=GREY_B).next_to(ax.c2p(0, v), LEFT, buff=0.1)
                      for s, v in [("1", 1), ("0", 0)]])
        zero = DashedLine(ax.c2p(0, 0), ax.c2p(400, 0), color=GREY_D)
        title = Text("similarity of position 100 to every position", font_size=22, color=GREY_B).next_to(ax, UP, 0.1)
        curve = VMobject(color=YELLOW, stroke_width=5).set_points_smoothly(
            [ax.c2p(10 * i, v) for i, v in enumerate(CURVE)])
        self.at("smooth")
        self.play(Create(ax), FadeIn(xl), FadeIn(yl), FadeIn(zero), FadeIn(title), run_time=0.6)
        self.play(Create(curve), run_time=1.5)
        for cue, x, label in [("101", 101, "101: 0.999"), ("110", 110, "110: 0.98"), ("150", 150, "150: 0.76"),
                              ("300", 300, "300: −0.23")]:
            v = {101: 0.999, 110: 0.984, 150: 0.757, 300: -0.225}[x]
            d = Dot(ax.c2p(x, v), color=RED, radius=0.07)
            where = {101: RIGHT, 110: DR, 150: RIGHT, 300: DOWN}[x]
            t = Text(label, font=MONO, font_size=18, color=RED_B).next_to(d, where, buff=0.12)
            self.at(cue)
            self.play(FadeIn(d, scale=1.5), FadeIn(t), run_time=0.35)
        self.end_section()

    # ------------------------------------------------------------------ 7. the limit
    def limit(self):
        self.section(7)
        self.clear_stage()
        rows = VGroup(*[Text(t, font=MONO, font_size=24, color=c) for t, c in
                        [("position 1     → vector", GREY_A), ("position 2     → vector", GREY_A), ("…", GREY_A),
                         ("position 1,024 → vector", GREY_A), ("position 1,025 → ?", RED)]]).arrange(DOWN, aligned_edge=LEFT, buff=0.2)
        rows.move_to([-2.8, 0.4, 0])
        self.at("limit")
        self.play(FadeIn(rows[:4]), run_time=0.5)
        self.at("025")
        self.play(FadeIn(rows[4], scale=1.2), run_time=0.4)
        longer = Text("can't read anything longer", font_size=24, color=RED_B).next_to(rows, DOWN, buff=0.3)
        self.at("longer")
        self.play(FadeIn(longer), run_time=0.4)
        c = Circle(radius=1.3, color=GREY_C).move_to([3.6, 0.4, 0])
        v1 = Arrow(c.get_center(), c.point_at_angle(0.3), buff=0, color=BLUE_C, stroke_width=5)
        v2 = Arrow(c.get_center(), c.point_at_angle(1.2), buff=0, color=GOLD, stroke_width=5)
        arc = Arc(radius=0.7, start_angle=0.3, angle=0.9, arc_center=c.get_center(), color=YELLOW)
        rl = Text("rotate queries and keys:\nattention sees relative distance", font_size=22, color=YELLOW,
                  line_spacing=0.8).next_to(c, DOWN, buff=0.25)
        self.at("rotating")
        self.play(Create(c), GrowArrow(v1), run_time=0.5)
        self.play(TransformFromCopy(v1, v2), Create(arc), run_time=0.7)
        self.at("relative")
        self.play(FadeIn(rl), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 8. code
    def code(self):
        self.section(8)
        self.clear_stage()
        code, hl = code_panel(CODE, font_size=20)
        code.move_to([0, 0.3, 0])
        self.play(FadeIn(code, shift=0.2 * UP), run_time=0.5)
        hl.match_y(code.line_numbers[2])
        self.at("switch")
        self.play(Create(hl), run_time=0.3)
        self.at("add")
        self.play(highlight(hl, code, 3), run_time=0.4)
        self.at("run")
        self.play(highlight(hl, code, 5), run_time=0.4)
        self.at("compare")
        self.play(highlight(hl, code, 8), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 9. outro
    def outro(self):
        self.section(9)
        self.clear_stage(run_time=0.4)
        card = next_up_card(NEXT)
        self.at("rope")
        self.play(FadeIn(card, shift=0.2 * UP), run_time=0.6)
        self.end_section()
        finish(self, hold=1.5)
