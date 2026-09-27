"""Video 4 — Where Am I? Position.

Render from the repo root:  ./render.sh how-llms-work v04
"""
import numpy as np
from manim import *

from intro import play_token_intro
from v04_script import TAGLINE, TITLE
from voiced_scene import VoicedScene

TOKEN_COLOR = BLUE_D
POS_COLOR = ORANGE
SIN_COLOR = TEAL
COS_COLOR = PURPLE_B
MONO = "DejaVu Sans Mono"

CODE = """def positional_encoding(n_pos, d_model):
    pos = np.arange(n_pos)[:, None]           # 0, 1, 2, ...
    i = np.arange(0, d_model, 2)[None, :]     # even dimensions
    angle = pos / 10000 ** (i / d_model)
    pe = np.zeros((n_pos, d_model))
    pe[:, 0::2] = np.sin(angle)               # even dims: sine
    pe[:, 1::2] = np.cos(angle)               # odd dims: cosine
    return pe

x = E[ids] + positional_encoding(len(ids), d_model)"""


def sinusoidal_pe(n_pos, d_model):
    pos = np.arange(n_pos)[:, None]
    i = np.arange(0, d_model, 2)[None, :]
    angle = pos / 10000 ** (i / d_model)
    pe = np.zeros((n_pos, d_model))
    pe[:, 0::2] = np.sin(angle)
    pe[:, 1::2] = np.cos(angle)
    return pe


PE4 = sinusoidal_pe(8, 4)                       # real values, d = 4
EMB_CAT = np.array([0.35, -0.12, 0.61, 0.08])   # illustrative embedding of "cat"


def fmt(v, nd=2):
    s = f"{v:.{nd}f}"
    if s.startswith("-") and float(s) == 0:
        s = s[1:]
    return s.replace("-", "−")


def token(word, color=TOKEN_COLOR, font_size=28):
    label = Text(word, font_size=font_size)
    box = RoundedRectangle(corner_radius=0.12, width=max(label.width + 0.35, 0.6), height=0.65,
                           stroke_color=color, fill_color=color, fill_opacity=0.3)
    return VGroup(box, label.move_to(box))


def vec(entries, color=WHITE, font_size=22, buff=0.16, colors=None, bracket_color=GREY_B):
    """A column vector: VGroup(brackets, rows)."""
    colors = colors or [color] * len(entries)
    rows = VGroup(*[Text(e, font=MONO, font_size=font_size, color=c) for e, c in zip(entries, colors)])
    rows.arrange(DOWN, buff=buff)
    for r in rows:
        r.align_to(rows, RIGHT)
    h, w = rows.height + 0.3, rows.width + 0.36

    def bracket(side):
        x = side * w / 2
        d = -side * 0.12
        pts = [[x + d, h / 2, 0], [x, h / 2, 0], [x, -h / 2, 0], [x + d, -h / 2, 0]]
        return VMobject(stroke_color=bracket_color, stroke_width=2.5).set_points_as_corners(pts)

    brackets = VGroup(bracket(-1), bracket(1)).move_to(rows)
    return VGroup(brackets, rows)


def label(text, color=GREY_B, font_size=22, **kw):
    return Text(text, font_size=font_size, color=color, **kw)


def cat_icon(color=GOLD_C):
    body = Ellipse(width=0.75, height=0.38, stroke_width=0, fill_color=color, fill_opacity=1)
    head = Circle(radius=0.19, stroke_width=0, fill_color=color, fill_opacity=1)
    head.move_to(body.get_right() + 0.02 * LEFT + 0.2 * UP)
    ears = VGroup(*[Triangle(stroke_width=0, fill_color=color, fill_opacity=1).scale(0.09)
                    .move_to(head.get_top() + dx * RIGHT + 0.01 * DOWN) for dx in (-0.1, 0.1)])
    tail = Arc(radius=0.2, start_angle=-PI / 2, angle=-PI * 0.8, stroke_color=color, stroke_width=5)
    tail.next_to(body, LEFT, buff=-0.08).shift(0.12 * UP)
    return VGroup(tail, body, head, ears)


def mat_icon(width=1.3):
    return RoundedRectangle(corner_radius=0.05, width=width, height=0.14, stroke_width=0,
                            fill_color=RED_D, fill_opacity=1)


class PositionVideo(VoicedScene):
    VIDEO = "v04"

    def clear_anims(self):
        """FadeOut everything on stage (and drop value trackers)."""
        anims = []
        for m in list(self.mobjects):
            if isinstance(m, ValueTracker):
                m.clear_updaters()
                self.remove(m)
            else:
                anims.append(FadeOut(m))
        return anims

    def construct(self):
        play_token_intro(self, TITLE, 4, TAGLINE)
        self.s1_same_tokens()
        self.s2_why_it_matters()
        self.s3_raw_numbers()
        self.s4_learned()
        self.s5_waves()
        self.s6_clock()
        self.s7_adding()
        self.s8_code()
        self.s9_rope()
        self.s10_outro()

    # 1. Same tokens, different meaning -------------------------------------------
    def s1_same_tokens(self):
        self.section(1)
        words1 = "The cat sat on the mat".split()
        words2 = "The mat sat on the cat".split()
        row1 = VGroup(*[token(w, font_size=26) for w in words1]).arrange(RIGHT, buff=0.08).move_to([-3.5, 2.75, 0])
        row2 = VGroup(*[token(w, font_size=26) for w in words2]).arrange(RIGHT, buff=0.08).move_to([3.5, 2.75, 0])
        self.at("cat")
        self.play(LaggedStart(*[FadeIn(t, shift=0.2 * DOWN) for t in row1], lag_ratio=0.12), run_time=1.0)
        self.at("mat")
        self.at("the")
        self.play(LaggedStart(*[FadeIn(t, shift=0.2 * DOWN) for t in row2], lag_ratio=0.12), run_time=1.0)

        def bag_of(row, words, x):
            order = sorted(range(len(words)), key=lambda k: words[k])
            movers = [row[k].copy() for k in order]
            grid = VGroup(*[m.copy() for m in movers]).arrange_in_grid(rows=2, cols=3, buff=0.14)
            grid.move_to([x, 0.95, 0])
            bag = RoundedRectangle(corner_radius=0.3, width=grid.width + 0.5, height=grid.height + 0.5,
                                   stroke_color=GREY_B, stroke_width=2, fill_color=GREY_E, fill_opacity=0.35)
            bag.move_to(grid)
            return movers, grid, bag

        m1, g1, bag1 = bag_of(row1, words1, -3.5)
        m2, g2, bag2 = bag_of(row2, words2, 3.5)
        same = label("same tokens", font_size=24).move_to([0, 1.55, 0])
        self.at("same")
        self.play(FadeIn(bag1), FadeIn(bag2),
                  *[m.animate.move_to(g) for m, g in zip(m1 + m2, list(g1) + list(g2))],
                  FadeIn(same), run_time=1.1)

        # different meanings: little pictures
        cat_a, mat_a = cat_icon(), mat_icon()
        pic1 = VGroup(mat_a, cat_a.next_to(mat_a, UP, buff=0.0)).move_to([-3.5, -1.55, 0])
        cat_b, mat_b = cat_icon(), mat_icon(1.1)
        pic2 = VGroup(cat_b, mat_b.next_to(cat_b, UP, buff=0.0)).move_to([3.5, -1.55, 0])
        cap1 = Text("cat on mat", font_size=26, color=GOLD_A).next_to(pic1, DOWN, buff=0.35)
        cap2 = Text("mat on cat", font_size=26, color=GOLD_A).move_to([3.5, cap1.get_y(), 0])
        neq = Text("≠", font_size=64, color=RED).move_to([0, -1.6, 0])
        diff = label("different meaning", font_size=24).next_to(neq, DOWN, buff=0.25)
        self.at("meaning")
        self.play(FadeIn(pic1, shift=0.2 * UP), FadeIn(pic2, shift=0.2 * UP), FadeIn(cap1), FadeIn(cap2),
                  FadeIn(neq), FadeIn(diff), run_time=1.0)

        eq = Text("=", font_size=64, color=YELLOW).move_to([0, 1.0, 0])
        ident = Text("vectors:\nidentical", font_size=24, color=YELLOW, line_spacing=0.8).move_to([0, 0.25, 0])
        self.at("apart")
        self.play(FadeIn(eq, scale=1.4), FadeIn(ident, shift=0.2 * UP), run_time=0.7)
        self.end_section()

    # 2. Why it matters: attention has no notion of order --------------------------
    def s2_why_it_matters(self):
        self.section(2)
        self.play(*self.clear_anims(), run_time=0.6)
        words = "The cat sat on the mat".split()
        toks = VGroup(*[token(w) for w in words]).arrange(RIGHT, buff=0.55).move_to([0, -1.0, 0])
        hl = ValueTracker(0)
        pairs = [(i, j) for i in range(6) for j in range(i + 1, 6) if (i, j) != (1, 5)] + [(1, 5)]

        def web():
            g = VGroup()
            for i, j in pairs:
                p, q = toks[i].get_top(), toks[j].get_top()
                if p[0] > q[0]:
                    p, q = q, p
                h = 0.42 * abs(q[0] - p[0]) + 0.2
                c = CubicBezier(p, p + h * UP, q + h * UP, q)
                if (i, j) == (1, 5):
                    a = hl.get_value()
                    c.set_stroke(interpolate_color(GREY, YELLOW, a), 1.6 + 2.6 * a, opacity=0.85 + 0.15 * a)
                else:
                    c.set_stroke(GREY, 1.6, opacity=0.8)
                g.add(c)
            return g

        heading = Text("attention", font_size=38).move_to([0, 3.1, 0])
        self.at("attention")
        self.play(FadeIn(heading, shift=0.2 * DOWN),
                  LaggedStart(*[FadeIn(t, shift=0.2 * UP) for t in toks], lag_ratio=0.1), run_time=0.9)
        arcs = web()
        self.play(LaggedStart(*[Create(c) for c in arcs], lag_ratio=0.06), run_time=1.3)
        self.remove(arcs)
        arcs = always_redraw(web)
        self.add(arcs, toks)

        sub = label("compares every token with every other token", font_size=26).move_to([0, 2.45, 0])
        self.at("compares")
        flash = web().set_stroke(YELLOW, 4, opacity=1)
        self.play(FadeIn(sub), LaggedStart(*[ShowPassingFlash(c, time_width=0.5) for c in flash],
                                           lag_ratio=0.04), run_time=1.4)

        self.at("nothing")
        self.play(hl.animate.set_value(1), run_time=0.6)

        perm = [5, 3, 0, 4, 1, 2]
        slots = VGroup(*[toks[k].copy() for k in perm]).arrange(RIGHT, buff=0.55).move_to(toks)
        order_q = Text("order?", font_size=34, color=RED_B).move_to([0, -2.15, 0])
        self.at("first")
        self.play(*[toks[k].animate(path_arc=0.9).move_to(slots[s]) for s, k in enumerate(perm)],
                  run_time=1.2)
        self.play(FadeIn(order_q, scale=1.2), run_time=0.4)

        final = Text("put the order into the vectors", font_size=30, color=POS_COLOR).move_to([0, -3.1, 0])
        self.at("order")
        self.play(FadeIn(final, shift=0.2 * UP), run_time=0.8)
        self.end_section()

    # 3. Idea 1: add the raw position number --------------------------------------
    def s3_raw_numbers(self):
        self.section(3)
        self.play(*self.clear_anims(), run_time=0.6)
        heading = Text("idea 1: add the position number", font_size=32).move_to([0, 3.2, 0])
        self.at("idea")
        self.play(FadeIn(heading, shift=0.2 * DOWN), run_time=0.7)

        items = VGroup(*[token(w) for w in ["The", "cat", "sat", "on"]], Text("…", font_size=36), token("mat"))
        items.arrange(RIGHT, buff=0.4).move_to([0, 2.0, 0])
        nums = [Text(f"+{k}", font=MONO, font_size=26, color=POS_COLOR).next_to(items[k - 1], DOWN, buff=0.22)
                for k in range(1, 5)]
        counter = ValueTracker(5)
        last_num = always_redraw(lambda: Text(f"+{int(round(counter.get_value()))}", font=MONO, font_size=26,
                                              color=POS_COLOR).next_to(items[5], DOWN, buff=0.22))
        self.at("position")
        self.play(LaggedStart(*[FadeIn(t, shift=0.2 * DOWN) for t in items[:4]], lag_ratio=0.15), run_time=0.8)
        for k, cue in enumerate(["one", "two", "three"]):
            self.at(cue)
            self.play(FadeIn(nums[k], shift=0.2 * UP), run_time=0.25)
        self.at("on")
        self.play(FadeIn(nums[3], shift=0.2 * UP), FadeIn(items[4]), FadeIn(items[5]), FadeIn(last_num),
                  run_time=0.6)
        self.at("limit")
        self.play(counter.animate.set_value(5000), rate_func=rate_functions.ease_in_quad, run_time=1.0)
        self.play(Indicate(last_num, color=RED, scale_factor=1.3), run_time=0.6)

        meaning = vec(["0.21", "−0.47", "0.83"], color=BLUE_B)
        plus = Text("+", font_size=40)
        five = Text("5000", font=MONO, font_size=28, color=POS_COLOR)
        eq = Text("=", font_size=40)
        total = vec(["5000.21", "4999.53", "5000.83"])
        eqn = VGroup(meaning, plus, five, eq, total).arrange(RIGHT, buff=0.35).move_to([-3.1, -1.45, 0])
        m_lab = label("meaning", BLUE_B).next_to(meaning, UP, buff=0.2)
        p_lab = label("position", POS_COLOR).next_to(five, UP, buff=0.2).match_y(m_lab)
        t_lab = label("token 5000's input", WHITE).next_to(total, UP, buff=0.2).match_y(m_lab)
        illus = label("illustrative values", font_size=20).next_to(eqn, DOWN, buff=0.35)
        self.at("5000")
        self.play(FadeIn(meaning), FadeIn(m_lab), run_time=0.4)
        self.play(FadeIn(plus), FadeIn(five), FadeIn(p_lab), run_time=0.4)
        self.play(FadeIn(eq), TransformFromCopy(VGroup(meaning, five), total), FadeIn(t_lab), FadeIn(illus),
                  run_time=0.7)

        base_y = -2.85
        axis = Line([1.6, base_y, 0], [6.2, base_y, 0], color=GREY_B, stroke_width=2)
        m_bar = Rectangle(width=0.9, height=0.035, stroke_width=0, fill_color=BLUE_B, fill_opacity=1)
        m_bar.move_to([2.8, base_y, 0], aligned_edge=DOWN)
        p_bar = Rectangle(width=0.9, height=2.5, stroke_width=0, fill_color=POS_COLOR, fill_opacity=0.9)
        p_bar.move_to([5.0, base_y, 0], aligned_edge=DOWN)
        m_val = Text("0.83", font=MONO, font_size=22, color=BLUE_B).next_to(m_bar, UP, buff=0.15)
        p_val = Text("5000", font=MONO, font_size=22, color=POS_COLOR).next_to(p_bar, UP, buff=0.15)
        m_name = label("meaning", BLUE_B).next_to([2.8, base_y, 0], DOWN, buff=0.2)
        p_name = label("position", POS_COLOR).next_to([5.0, base_y, 0], DOWN, buff=0.2)
        self.at("drown")
        self.play(Create(axis), GrowFromEdge(m_bar, DOWN), GrowFromEdge(p_bar, DOWN),
                  FadeIn(m_val), FadeIn(p_val), FadeIn(m_name), FadeIn(p_name), run_time=0.7)
        lost = Text("meaning lost", font_size=28, color=RED).move_to([2.8, -1.75, 0])
        self.at("meaning")
        self.play(FadeIn(lost, shift=0.2 * DOWN), run_time=0.45)
        self.end_section()

    # 4. Idea 2: learned position embeddings (GPT-2) ---------------------------------
    def s4_learned(self):
        self.section(4)
        self.play(*self.clear_anims(), run_time=0.6)
        heading = Text("idea 2: learn them (GPT-2)", font_size=32).move_to([0, 3.25, 0])
        self.at("gpt")
        self.play(FadeIn(heading, shift=0.2 * DOWN), run_time=0.7)

        rng = np.random.default_rng(4)
        xs = [0.0, 1.35, 2.55, 3.75, 4.7]
        row_labels = ["0", "1", "2", "⋮", "1023"]
        table = VGroup()
        header = VGroup(label("pos", font_size=20).move_to([xs[0], 0, 0]),
                        label("vector", font_size=20).move_to([xs[2], 0, 0]))
        rows, row_vals = VGroup(), []
        for r, lab in enumerate(row_labels):
            y = -0.5 * (r + 1)
            cells = VGroup(Text(lab, font=MONO, font_size=22, color=POS_COLOR).move_to([xs[0], y, 0]))
            vals = []
            for c in range(3):
                if lab == "⋮":
                    t = Text("⋮", font_size=22, color=GREY_A)
                else:
                    v = rng.uniform(-0.9, 0.9)
                    vals.append(v)
                    t = Text(fmt(v), font=MONO, font_size=22, color=GREY_A)
                cells.add(t.move_to([xs[c + 1], y, 0]))
            cells.add(Text("…", font_size=22, color=GREY_B).move_to([xs[4], y, 0]))
            rows.add(cells)
            row_vals.append(vals)
        table.add(header, rows)
        frame = RoundedRectangle(corner_radius=0.15, width=table.width + 0.6, height=table.height + 0.4,
                                 stroke_color=GREY_B, stroke_width=2).move_to(table)
        sep_x = (xs[0] + xs[1]) / 2 + 0.05
        sep = Line([sep_x, table.get_top()[1] + 0.1, 0], [sep_x, table.get_bottom()[1] - 0.1, 0], color=GREY_D)
        title = Text("position embeddings", font_size=26, color=POS_COLOR).next_to(frame, UP, buff=0.2)
        tbl = VGroup(title, frame, sep, table)
        tbl.move_to([-3.5, 0.75, 0])
        self.at("learn")
        self.play(FadeIn(title), Create(frame), FadeIn(sep), FadeIn(header),
                  LaggedStart(*[FadeIn(r, shift=0.1 * DOWN) for r in rows], lag_ratio=0.12), run_time=1.2)

        self.at("position")
        self.play(LaggedStart(*[Indicate(r[0], color=YELLOW, scale_factor=1.3) for r in rows], lag_ratio=0.1),
                  run_time=0.8)

        pos2 = row_vals[2]
        emb = vec([fmt(v) for v in EMB_CAT[:3]] + ["⋮"], color=BLUE_B)
        pe = vec([fmt(v) for v in pos2] + ["⋮"], color=POS_COLOR)
        inp = vec([fmt(round(a, 2) + round(b, 2)) for a, b in zip(EMB_CAT[:3], pos2)] + ["⋮"])
        plus, eq = Text("+", font_size=40), Text("=", font_size=40)
        eqn = VGroup(emb, plus, pe, eq, inp).arrange(RIGHT, buff=0.45).move_to([3.3, 0.3, 0])
        h_emb = Text("embedding\n(cat)", font_size=22, color=BLUE_B, line_spacing=0.8)
        h_pe = Text("position 2", font_size=22, color=POS_COLOR)
        h_inp = Text("input\nvector", font_size=22, line_spacing=0.8)
        heads = VGroup(h_emb, h_pe, h_inp)
        for h, col in zip(heads, [emb, pe, inp]):
            h.next_to(col, UP, buff=0.25)
        heads.align_to(h_emb, DOWN)
        pick = SurroundingRectangle(rows[2], color=POS_COLOR, buff=0.08, corner_radius=0.08)
        illus = label("illustrative values", font_size=20).next_to(eqn, DOWN, buff=0.4)
        self.at("add")
        self.play(FadeIn(emb), FadeIn(h_emb), FadeIn(plus), Create(pick), run_time=0.6)
        self.play(TransformFromCopy(VGroup(*rows[2][1:4]), pe[1][:3]), FadeIn(pe[0]), FadeIn(pe[1][3]),
                  FadeIn(h_pe), run_time=0.8)
        self.play(FadeIn(eq), TransformFromCopy(VGroup(emb[1], pe[1]), inp[1]), FadeIn(inp[0]),
                  FadeIn(h_inp), FadeIn(illus), run_time=0.7)

        extra = VGroup(Text("1024", font=MONO, font_size=22, color=POS_COLOR).move_to([xs[0], 0, 0]),
                       *[Text("?", font_size=24, color=GREY_B).move_to([xs[c + 1], 0, 0]) for c in range(3)])
        for c in range(4):
            extra[c].match_x(rows[4][c])
        extra.set_y(frame.get_bottom()[1] - 0.45)
        extra_box = DashedVMobject(RoundedRectangle(corner_radius=0.1, width=frame.width, height=0.55,
                                                    stroke_color=RED_B, stroke_width=2), num_dashes=40)
        extra_box.move_to([frame.get_x(), extra.get_y(), 0])
        cross = Text("✗", font_size=44, color=RED).next_to(extra_box, RIGHT, buff=0.2)
        limit = Text("GPT-2: 1,024 positions (0–1023)", font_size=26).next_to(extra_box, DOWN, buff=0.45)
        limit.match_x(frame)
        self.at("positions")
        self.play(Create(extra_box), FadeIn(extra), run_time=0.7)
        self.at("training")
        self.play(FadeIn(cross, scale=1.5), FadeIn(limit, shift=0.2 * UP), run_time=0.6)
        self.end_section()

    # 5. Sinusoidal waves ------------------------------------------------------------
    def s5_waves(self):
        self.section(5)
        self.play(*self.clear_anims(), run_time=0.6)
        heading = Text("sinusoidal position encoding", font_size=32).move_to([0, 3.35, 0])
        self.at("paper")
        self.play(FadeIn(heading, shift=0.2 * DOWN), run_time=0.7)

        X0, X1, N = -4.9, 2.1, 40
        SX = (X1 - X0) / N
        ROW_Y = [2.0, 0.7, -0.6, -1.9]
        AMP = 0.45
        OMEGA = [1, 1 / 3, 1 / 9, 1 / 27]

        def xp(p):
            return X0 + p * SX

        axes_lines, sins, coss, names = VGroup(), VGroup(), VGroup(), VGroup()
        for r, (y, w) in enumerate(zip(ROW_Y, OMEGA)):
            axes_lines.add(Line([X0, y, 0], [X1 + 0.1, y, 0], color=GREY_D, stroke_width=1.5))
            sins.add(ParametricFunction(lambda p, y=y, w=w: np.array([xp(p), y + AMP * np.sin(w * p), 0]),
                                        t_range=[0, N, 0.05], color=SIN_COLOR, stroke_width=3))
            coss.add(ParametricFunction(lambda p, y=y, w=w: np.array([xp(p), y + AMP * np.cos(w * p), 0]),
                                        t_range=[0, N, 0.05], color=COS_COLOR, stroke_width=3))
            names.add(label(f"pair {r + 1}", font_size=22).next_to([X0, y, 0], LEFT, buff=0.3))
        pos_axis = Arrow([X0, -2.65, 0], [X1 + 0.35, -2.65, 0], buff=0, color=GREY_B, stroke_width=2,
                         max_tip_length_to_length_ratio=0.03)
        ticks = VGroup(*[label(str(p), font_size=20).move_to([xp(p), -2.95, 0]) for p in (0, 10, 20, 30, 40)])
        pos_name = label("position", font_size=22).next_to(pos_axis, RIGHT, buff=0.15)
        illus = label("illustrative frequencies", font_size=20).move_to([xp(20), -3.45, 0])
        self.at("waves")
        self.play(FadeIn(axes_lines), FadeIn(names), FadeIn(pos_axis), FadeIn(ticks), FadeIn(pos_name),
                  run_time=0.5)
        self.play(LaggedStart(*[AnimationGroup(Create(s), Create(c)) for s, c in zip(sins, coss)],
                              lag_ratio=0.2), FadeIn(illus), run_time=1.5)

        leg_sin = VGroup(Line(ORIGIN, 0.45 * RIGHT, color=SIN_COLOR, stroke_width=4),
                         Text("sine", font_size=22, color=SIN_COLOR)).arrange(RIGHT, buff=0.15)
        leg_cos = VGroup(Line(ORIGIN, 0.45 * RIGHT, color=COS_COLOR, stroke_width=4),
                         Text("cosine", font_size=22, color=COS_COLOR)).arrange(RIGHT, buff=0.15)
        legend = VGroup(leg_sin, leg_cos).arrange(RIGHT, buff=0.4).move_to([4.7, 2.75, 0])
        self.at("sine")
        self.play(FadeIn(leg_sin), *[ShowPassingFlash(s.copy().set_stroke(YELLOW, 6), time_width=0.5)
                                     for s in sins], run_time=0.65)
        self.at("cosine")
        self.play(FadeIn(leg_cos), *[ShowPassingFlash(c.copy().set_stroke(YELLOW, 6), time_width=0.5)
                                     for c in coss], run_time=0.65)

        self.at("frequency")
        self.play(LaggedStart(*[AnimationGroup(Indicate(names[r], color=YELLOW),
                                               ShowPassingFlash(VGroup(sins[r], coss[r]).copy()
                                                                .set_stroke(YELLOW, 6), time_width=0.6))
                                for r in range(4)], lag_ratio=0.3), run_time=1.2)

        pos = ValueTracker(2)
        line = always_redraw(lambda: Line([xp(pos.get_value()), 2.62, 0], [xp(pos.get_value()), -2.45, 0],
                                          color=POS_COLOR, stroke_width=3))
        line_lab = always_redraw(lambda: Text(f"pos = {pos.get_value():.0f}", font_size=22, color=POS_COLOR)
                                 .move_to([xp(pos.get_value()), 2.9, 0]))

        def values(p):
            out = []
            for w in OMEGA:
                out += [np.sin(w * p), np.cos(w * p)]
            return out

        def dots():
            p = pos.get_value()
            g = VGroup()
            for r, (y, w) in enumerate(zip(ROW_Y, OMEGA)):
                g.add(Dot([xp(p), y + AMP * np.sin(w * p), 0], radius=0.075, color=SIN_COLOR))
                g.add(Dot([xp(p), y + AMP * np.cos(w * p), 0], radius=0.075, color=COS_COLOR))
            return g.set_stroke(WHITE, 1.5, background=False)

        crossings = always_redraw(dots)
        self.at("position")
        self.play(FadeIn(line), FadeIn(line_lab), FadeIn(crossings), run_time=0.6)

        def pe_vec():
            p = pos.get_value()
            v = vec([fmt(x) for x in values(p)], colors=[SIN_COLOR, COS_COLOR] * 4, font_size=22)
            v.move_to([4.7, -0.05, 0])
            head = Text(f"PE(pos = {p:.0f})", font_size=24, color=POS_COLOR).next_to(v, UP, buff=0.25)
            return VGroup(v, head)

        column = always_redraw(pe_vec)
        self.at("read")
        self.play(FadeIn(column, shift=0.3 * LEFT), run_time=0.6)
        self.play(pos.animate.set_value(6), run_time=1.3)
        self.end_section()

    # 6. Clock analogy -------------------------------------------------------------
    def s6_clock(self):
        self.section(6)
        self.play(*self.clear_anims(), run_time=0.6)
        C = np.array([-3.4, 0.25, 0])
        R = 2.0
        face = Circle(radius=R, stroke_color=GREY_B, stroke_width=3).move_to(C)
        ticks = VGroup()
        for k in range(60):
            a = k / 60 * TAU
            d = np.array([np.sin(a), np.cos(a), 0])
            long = k % 5 == 0
            ticks.add(Line(C + (R - (0.25 if long else 0.1)) * d, C + (R - 0.02) * d,
                           color=GREY_B if long else GREY_D, stroke_width=4 if long else 1.5))
        t0 = 10 * 3600 + 8 * 60
        clock_t = ValueTracker(t0)
        emph = {k: ValueTracker(0) for k in ("s", "m", "h")}
        allv = ValueTracker(1)
        HANDS = {"s": (60, 1.8, 2.5, RED_C), "m": (3600, 1.5, 5, TEAL), "h": (43200, 1.0, 8, PURPLE_B)}

        def hand(key):
            period, length, width, color = HANDS[key]

            def make():
                a = (clock_t.get_value() % period) / period * TAU
                d = np.array([np.sin(a), np.cos(a), 0])
                e = emph[key].get_value()
                op = 0.35 + 0.65 * max(e, allv.get_value())
                start = C - (0.3 * d if key == "s" else 0)
                return Line(start, C + length * d, color=color, stroke_width=width * (1 + 0.7 * e),
                            stroke_opacity=op)
            return always_redraw(make)

        hands = VGroup(hand("h"), hand("m"), hand("s"))
        pin = Dot(C, radius=0.08, color=WHITE)
        title = Text("like a clock", font_size=32).move_to([0, 3.35, 0])
        self.at("clock")
        self.play(FadeIn(title, shift=0.2 * DOWN), Create(face), FadeIn(ticks), FadeIn(hands), FadeIn(pin),
                  run_time=0.8)
        clock_t.add_updater(lambda m, dt: m.increment_value(40 * dt))
        self.add(clock_t)

        rows_y = [1.75, 0.35, -1.05]
        specs = [("s", "second hand · fast"), ("m", "minute hand · slower"), ("h", "hour hand · slowest")]
        rows = VGroup()
        for (key, text), y in zip(specs, rows_y):
            color = HANDS[key][3]
            sw = Line(ORIGIN, 0.5 * RIGHT, color=color, stroke_width=6).move_to([-0.55, y, 0])
            rows.add(VGroup(sw, Text(text, font_size=26).next_to(sw, RIGHT, buff=0.2)))

        def focus(key):
            anims = [allv.animate.set_value(0)]
            for k, tr in emph.items():
                anims.append(tr.animate.set_value(1 if k == key else 0))
            return anims

        for key, cue, row in [("s", "second", rows[0]), ("m", "minute", rows[1]), ("h", "hour", rows[2])]:
            self.at(cue)
            self.play(FadeIn(row, shift=0.2 * LEFT), *focus(key), run_time=0.6)

        amb = label("each hand alone: ambiguous", font_size=24).move_to([1.6, -2.35, 0])
        self.at("ambiguous")
        self.play(FadeIn(amb), run_time=0.5)

        def readout_text():
            t = int(clock_t.get_value())
            return Text(f"{t // 3600:02d}:{t // 60 % 60:02d}:{t % 60:02d}", font=MONO, font_size=34,
                        color=YELLOW).move_to(C + (R + 0.55) * DOWN)
        readout = always_redraw(readout_text)
        self.at("together")
        self.play(allv.animate.set_value(1), *[tr.animate.set_value(0) for tr in emph.values()],
                  FadeIn(readout), amb.animate.set_opacity(0.5), run_time=0.6)

        wave_x0 = max(rows[0][1].get_right()[0], rows[2][1].get_right()[0]) + 0.85
        fast_w = FunctionGraph(lambda x: 0.22 * np.sin(8 * x), x_range=[0, 1.9, 0.01], color=RED_C,
                               stroke_width=3).move_to([wave_x0, rows_y[0], 0], aligned_edge=LEFT)
        slow_w = FunctionGraph(lambda x: 0.22 * np.sin(0.8 * x - 0.8), x_range=[0, 1.9, 0.01], color=PURPLE_B,
                               stroke_width=3).move_to([wave_x0, rows_y[2], 0], aligned_edge=LEFT)
        arrow_f = Text("→", font_size=30, color=GREY_B).move_to([wave_x0 - 0.4, rows_y[0], 0])
        arrow_s = Text("→", font_size=30, color=GREY_B).move_to([wave_x0 - 0.4, rows_y[2], 0])
        cap_f = label("fast wave: tells neighbours apart", font_size=20).next_to(rows[0][1], DOWN, buff=0.2,
                                                                                aligned_edge=LEFT)
        cap_s = label("slow wave: tracks the long run", font_size=20).next_to(rows[2][1], DOWN, buff=0.2,
                                                                             aligned_edge=LEFT)
        self.at("fast")
        self.play(FadeIn(arrow_f), Create(fast_w), FadeIn(cap_f), *focus("s"), run_time=0.7)
        self.at("slow")
        self.play(FadeIn(arrow_s), Create(slow_w), FadeIn(cap_s), *focus("h"), run_time=0.7)
        self.end_section()

    # 7. Adding position to meaning ----------------------------------------------
    def s7_adding(self):
        self.section(7)
        self.play(*self.clear_anims(), run_time=0.6)
        xs = [-5.3, -4.15, -3.0, -1.85, -0.7]

        def equation(p, y):
            emb = vec([fmt(v, 3) for v in EMB_CAT], color=BLUE_B).move_to([xs[0], y, 0])
            pe = vec([fmt(v, 3) for v in PE4[p]], color=POS_COLOR).move_to([xs[2], y, 0])
            tot = vec([fmt(v, 3) for v in EMB_CAT + PE4[p]]).move_to([xs[4], y, 0])
            plus = Text("+", font_size=40).move_to([xs[1], y, 0])
            eq = Text("=", font_size=40).move_to([xs[3], y, 0])
            top = emb.get_top()[1] + 0.3
            h_emb = Text("embedding(cat)", font_size=22, color=BLUE_B).move_to([xs[0], top, 0])
            h_pe = Text(f"PE({p})", font_size=22, color=POS_COLOR).move_to([xs[2], top, 0])
            h_tot = Text(f"cat at {p}", font_size=22).move_to([xs[4], top, 0])
            return VGroup(emb, plus, pe, eq, tot), VGroup(h_emb, h_pe, h_tot)

        eq_a, heads_a = equation(2, 1.55)
        eq_b, heads_b = equation(6, -1.45)
        self.at("added")
        self.play(FadeIn(eq_a[0]), FadeIn(heads_a[0]), run_time=0.4)
        self.play(FadeIn(eq_a[1]), FadeIn(eq_a[2], shift=0.2 * LEFT), FadeIn(heads_a[1]), run_time=0.5)
        self.at("two", "2")
        self.play(FadeIn(eq_a[3]), TransformFromCopy(VGroup(eq_a[0][1], eq_a[2][1]), eq_a[4][1]),
                  FadeIn(eq_a[4][0]), FadeIn(heads_a[2]), run_time=0.8)
        note = label("cat: illustrative values  ·  PE: real values (d = 4)", font_size=20)
        note.next_to(eq_b, DOWN, buff=0.35)
        self.at("six", "6")
        self.play(FadeIn(eq_b), FadeIn(heads_b), FadeIn(note), run_time=0.9)

        O = np.array([4.1, 0.25, 0])
        axes = VGroup(Arrow(O + [-1.8, -1.6, 0], O + [1.9, -1.6, 0], buff=0, color=GREY_D, stroke_width=2,
                            max_tip_length_to_length_ratio=0.05),
                      Arrow(O + [-1.8, -1.6, 0], O + [-1.8, 1.9, 0], buff=0, color=GREY_D, stroke_width=2,
                            max_tip_length_to_length_ratio=0.05))
        sketch = label("2-D sketch (illustrative)", font_size=20).move_to(O + [0, 2.3, 0])
        others = VGroup()
        for name, off in [("mat", [1.2, -0.9, 0]), ("sat", [-1.0, -1.05, 0]), ("the", [1.3, 1.3, 0])]:
            d = Dot(O + off, radius=0.07, color=GREY_B)
            others.add(VGroup(d, label(name, font_size=20).next_to(d, RIGHT, buff=0.12)))
        c2 = Dot(O + [-0.2, 0.6, 0], radius=0.09, color=BLUE_B)
        c6 = Dot(O + [0.2, 0.3, 0], radius=0.09, color=BLUE_B)
        ring = Circle(radius=0.42, color=YELLOW, stroke_width=2).move_to((c2.get_center() + c6.get_center()) / 2)
        l2 = Text("cat at 2", font_size=20).next_to(ring, LEFT, buff=0.1).match_y(c2)
        l6 = Text("cat at 6", font_size=20).next_to(ring, RIGHT, buff=0.1).match_y(c6)
        self.at("different")
        self.play(FadeIn(axes), FadeIn(sketch), FadeIn(others), FadeIn(c2, scale=1.5), FadeIn(c6, scale=1.5),
                  FadeIn(l2), FadeIn(l6), run_time=0.8)
        self.play(Create(ring), run_time=0.5)

        self.at("same")
        self.play(Indicate(eq_a[0], color=YELLOW, scale_factor=1.08),
                  Indicate(eq_b[0], color=YELLOW, scale_factor=1.08), run_time=0.7)
        caption = Text("same meaning, different place", font_size=24, color=YELLOW).move_to(O + [-0.15, -2.25, 0])
        self.at("place")
        self.play(FadeIn(caption, shift=0.2 * UP), Indicate(eq_a[2], color=YELLOW, scale_factor=1.08),
                  Indicate(eq_b[2], color=YELLOW, scale_factor=1.08), run_time=0.7)
        self.end_section()

    # 8. Code ------------------------------------------------------------------------
    def s8_code(self):
        self.section(8)
        code = Code(code_string=CODE, language="python", formatter_style="monokai",
                    background="window", paragraph_config={"font_size": 20})
        code.scale_to_fit_width(12.4).move_to([0, 0.2, 0])
        row_h = abs(code.line_numbers[0].get_y() - code.line_numbers[1].get_y())
        hl = Rectangle(width=code.code_lines.width + 0.25, height=row_h, color=YELLOW, stroke_width=2)
        hl.align_to(code.code_lines, LEFT).shift(0.12 * LEFT).match_y(code.line_numbers[1])

        def move_hl(i):
            return hl.animate.match_y(code.line_numbers[i])

        self.at("code")
        self.play(*self.clear_anims(), FadeIn(code, shift=0.3 * UP), run_time=0.8)
        self.at("position")
        self.play(Create(hl), run_time=0.5)
        for cue, line in [("dimensions", 2), ("angle", 3), ("sign", 5), ("cosine", 6), ("add", 9)]:
            self.at(cue)
            self.play(move_hl(line), run_time=0.5)
        self.end_section()

    # 9. RoPE -------------------------------------------------------------------------
    def s9_rope(self):
        self.section(9)
        self.play(*self.clear_anims(), run_time=0.6)
        title = Text("RoPE — rotary position embedding", font_size=34).move_to([0, 3.2, 0])
        modern = label("used by Llama and most modern models", font_size=22).move_to([0, 2.6, 0])
        self.at("llama")
        self.play(FadeIn(modern), run_time=0.6)
        self.at("rope")
        self.play(FadeIn(title, shift=0.2 * DOWN), run_time=0.7)

        C = np.array([-3.3, -1.45, 0])
        L = 2.1
        TH = 20 * DEGREES
        grid = VGroup(Line(C + [-2.6, 0, 0], C + [2.6, 0, 0], color=GREY_D, stroke_width=2),
                      Line(C + [0, -0.6, 0], C + [0, 2.6, 0], color=GREY_D, stroke_width=2),
                      DashedVMobject(Arc(radius=L, start_angle=0, angle=PI, arc_center=C, color=GREY_D,
                                         stroke_width=1.5), num_dashes=30))
        pair_cap = label("one pair of numbers in a vector", font_size=22).move_to(C + [0, -1.05, 0])
        am, an = ValueTracker(0), ValueTracker(0)

        def direction(a):
            return np.array([np.cos(a * TH), np.sin(a * TH), 0])

        def arrow_for(tr, color, name):
            def make():
                d = direction(tr.get_value())
                arr = Arrow(C, C + L * d, buff=0, color=color, stroke_width=6,
                            max_tip_length_to_length_ratio=0.12)
                tag = Text(name, font_size=26, color=color).move_to(C + (L + 0.35) * d)
                return VGroup(arr, tag)
            return always_redraw(make)

        arrow_m = arrow_for(am, BLUE_B, "m")

        def theta_text(k):
            return "θ" if k == 1 else f"{k}θ"

        def pos_arc():
            a = am.get_value()
            arc = Arc(radius=0.6, start_angle=0, angle=max(a, 0.001) * TH, arc_center=C, color=POS_COLOR,
                      stroke_width=4)
            k = int(round(a))
            tag = Text(theta_text(k) if a > 0.4 else "", font_size=24, color=POS_COLOR)
            tag.move_to(C + 0.9 * direction(a / 2))
            return VGroup(arc, tag)

        orange_arc = always_redraw(pos_arc)
        not_add = Text("x + PE(pos)", font=MONO, font_size=34, t2c={"PE(pos)": POS_COLOR}).move_to([0, 0.7, 0])
        strike = Line(not_add.get_left() + 0.15 * LEFT, not_add.get_right() + 0.15 * RIGHT, color=RED,
                      stroke_width=5)
        no_lab = label("no added vector", font_size=24).next_to(not_add, DOWN, buff=0.3)
        self.at("adding")
        self.play(FadeIn(not_add), run_time=0.4)
        self.play(Create(strike), FadeIn(no_lab), run_time=0.4)
        self.at("rotates")
        self.play(FadeOut(VGroup(not_add, strike, no_lab)), FadeIn(grid), FadeIn(arrow_m), FadeIn(pair_cap),
                  run_time=0.6)
        self.add(orange_arc)
        self.play(am.animate.set_value(1), run_time=0.7)

        rule = Text("angle = position · θ", font_size=28, color=POS_COLOR).move_to([0.5, 1.4, 0], aligned_edge=LEFT)
        self.at("angle")
        self.play(FadeIn(rule, shift=0.2 * LEFT), run_time=0.5)

        def info(tr, name, color, y):
            def make():
                k = int(round(tr.get_value()))
                return Text(f"token {name}: position {k} → {theta_text(k)}", font_size=26, color=color) \
                    .move_to([0.5, y, 0], aligned_edge=LEFT)
            return always_redraw(make)

        info_m = info(am, "m", BLUE_B, 0.5)
        self.at("grows")
        self.play(am.animate.set_value(2), run_time=0.6)
        info_m.update()
        self.play(FadeIn(info_m), run_time=0.3)

        arrow_n = arrow_for(an, TEAL, "n")
        info_n = info(an, "n", TEAL, -0.2)
        self.at("two")
        self.play(FadeOut(orange_arc), run_time=0.2)
        self.add(arrow_n)
        self.play(an.animate.set_value(5), run_time=0.9)
        info_n.update()
        self.play(FadeIn(info_n), run_time=0.3)

        qk = label("applied to the vectors attention compares (queries & keys)", font_size=22).move_to([0, -3.4, 0])
        self.at("compared")
        self.play(FadeIn(qk), run_time=0.5)

        def gap():
            a, b = am.get_value(), an.get_value()
            arc = Arc(radius=1.15, start_angle=a * TH, angle=(b - a) * TH, arc_center=C, color=YELLOW,
                      stroke_width=5)
            k = int(round(b - a))
            tag = Text(theta_text(k), font_size=28, color=YELLOW).move_to(C + 1.5 * direction((a + b) / 2))
            return VGroup(arc, tag)

        gap_arc = always_redraw(gap)
        diff = Text("difference: n − m = 3  →  3θ", font_size=26, color=YELLOW).move_to([0.5, -1.0, 0],
                                                                                         aligned_edge=LEFT)
        self.at("difference")
        self.play(FadeIn(gap_arc), FadeIn(diff), run_time=0.7)

        still = Text("shift both by 3: the gap is still 3θ", font_size=24, color=GREY_A).move_to(
            [0.5, -1.75, 0], aligned_edge=LEFT)
        self.at("directly")
        self.play(am.animate.set_value(5), an.animate.set_value(8), run_time=1.2)
        self.at("apart")
        self.play(FadeIn(still, shift=0.2 * UP), Indicate(diff, color=WHITE, scale_factor=1.05), run_time=0.8)
        self.end_section()

    # 10. Outro + next up -------------------------------------------------------------
    def s10_outro(self):
        self.section(10)
        self.play(*self.clear_anims(), run_time=0.6)
        emb = vec([fmt(v, 3) for v in EMB_CAT], color=BLUE_B)
        pe = vec([fmt(v, 3) for v in PE4[2]], color=POS_COLOR)
        res = vec([fmt(v, 3) for v in EMB_CAT + PE4[2]])
        plus, eq = Text("+", font_size=44), Text("=", font_size=44)
        row = VGroup(emb, plus, pe, eq, res).arrange(RIGHT, buff=0.6).move_to([0, -0.1, 0])
        cat = token("cat").next_to(emb, UP, buff=0.45)
        where = token("position 2", color=POS_COLOR, font_size=26).next_to(pe, UP, buff=0.45)
        what_lab = Text("what it is", font_size=28, color=BLUE_B).next_to(emb, DOWN, buff=0.4)
        where_lab = Text("where it is", font_size=28, color=POS_COLOR).next_to(pe, DOWN, buff=0.4)
        res_lab = Text("token vector", font_size=26).next_to(res, DOWN, buff=0.4)
        illus = label("illustrative values", font_size=20).move_to([0, -3.3, 0])
        self.at("token")
        self.play(FadeIn(cat, shift=0.2 * DOWN), run_time=0.5)
        self.at("what")
        self.play(FadeIn(emb), FadeIn(what_lab, shift=0.2 * UP), FadeIn(illus), run_time=0.6)
        self.at("where")
        self.play(FadeIn(plus), FadeIn(where, shift=0.2 * DOWN), FadeIn(pe), FadeIn(where_lab, shift=0.2 * UP),
                  run_time=0.6)
        self.at("ready")
        self.play(FadeIn(eq), TransformFromCopy(VGroup(emb[1], pe[1]), res[1]), FadeIn(res[0]), FadeIn(res_lab),
                  run_time=0.8)

        next_label = Text("Next up", font_size=30, color=GREY_B)
        next_title = Text("Attention I: Tokens Talking to Each Other", font_size=44)
        card = VGroup(next_label, next_title).arrange(DOWN, buff=0.35)
        self.at("next")
        self.play(FadeOut(VGroup(cat, where, emb, plus, pe, eq, what_lab, where_lab, res_lab, illus)),
                  run_time=0.6)
        self.at("attention")
        self.play(ReplacementTransform(res, card), run_time=1.0)
        self.end_section()
        self.wait(1.0)  # silent hold so the Next up card is fully readable
        self.play(FadeOut(card))
        self.wait(0.5)
