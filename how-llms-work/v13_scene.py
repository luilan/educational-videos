"""Video 13 — Making It Fast: the KV Cache.

Render from the repo root:  ./render.sh how-llms-work v13
"""
import numpy as np
from manim import *

from common import code_panel, finish, next_up_card, token
from intro import play_token_intro
from v13_script import NEXT, TAGLINE, TITLE
from voiced_scene import VoicedScene

TOKEN_COLOR = BLUE_D
GEN_COLOR = GREEN_C              # generated tokens
Q_COLOR, K_COLOR, V_COLOR, W_COLOR = YELLOW, TEAL, ORANGE, GREEN
NEW_COLOR, OLD_COLOR = BLUE_D, RED_C   # freshly computed vs recomputed work
CACHE_FILL = GREY_E

CODE = """def attend_new_token(x_new, cache, Wq, Wk, Wv):
    q, k, v = x_new @ Wq, x_new @ Wk, x_new @ Wv
    cache.keys.append(k)                    # keep for later steps
    cache.values.append(v)
    K, V = np.stack(cache.keys), np.stack(cache.values)
    w = softmax(q @ K.T / np.sqrt(len(k)))  # vs all past tokens
    return w @ V"""

WORDS = ["The", "cat", "sat", "on", "the", "mat", "and"]

# illustrative causal attention weights (row i only uses columns 0..i)
_rng = np.random.default_rng(13)
_RAW = np.tril(_rng.uniform(0.15, 1.0, (7, 7)))
WEIGHTS = _RAW / _RAW.sum(1, keepdims=True)


def weight_opacity(i, j):
    return 0.15 + 0.7 * WEIGHTS[i, j] / WEIGHTS[i, :i + 1].max()


# ------------------------------------------------------------------ helpers
def label(text, color=GREY_B, font_size=22, **kw):
    return Text(text, font_size=font_size, color=color, **kw)


def small_token(word, color=TOKEN_COLOR):
    lab = Text(word, font_size=24).scale(20 / 24)
    box = RoundedRectangle(corner_radius=0.08, width=max(lab.width + 0.24, 0.5), height=0.4,
                           stroke_color=color, stroke_width=2, fill_color=color, fill_opacity=0.3)
    return VGroup(box, lab.move_to(box))


def strip(color, seed=0, n=4, cell=0.24, direction=RIGHT, buff=0.04):
    """A small vector drawn as a row (or column) of tinted cells."""
    ops = np.random.default_rng(seed).uniform(0.25, 0.95, n)
    return VGroup(*[Square(side_length=cell, stroke_color=color, stroke_width=1.5, fill_color=color,
                           fill_opacity=float(o)) for o in ops]).arrange(direction, buff=buff)


def gslice(mob, s, start, end):
    """Glyphs of Text `mob` (built from string s) for s[start:end]; spaces have no glyphs."""
    a = len(s[:start].replace(" ", ""))
    return mob[a:a + len(s[start:end].replace(" ", ""))]


def grid_cell(i, j, w, h):
    """One cell of the causal attention grid: green (weight) on/below the diagonal, dark above."""
    if j > i:
        return Rectangle(width=w, height=h, stroke_color=GREY_D, stroke_width=1.5, fill_color=CACHE_FILL,
                         fill_opacity=0.25)
    return Rectangle(width=w, height=h, stroke_color=GREY_D, stroke_width=1.5, fill_color=W_COLOR,
                     fill_opacity=weight_opacity(i, j))


def chat_bubble(width, height, color=GREY_B, tail="left"):
    body = RoundedRectangle(corner_radius=min(0.3, height / 3), width=width, height=height, stroke_color=color,
                            stroke_width=3, fill_color=BLACK, fill_opacity=1)
    bl = body.get_corner(DL)
    sx = 1 if tail == "left" else -1
    if tail == "right":
        bl = body.get_corner(DR)
    tip = Polygon(bl + np.array([sx * 0.35, 0.05, 0]), bl + np.array([sx * 0.8, 0.05, 0]),
                  bl + np.array([sx * 0.15, -0.35, 0]), stroke_color=color, stroke_width=3, fill_color=BLACK,
                  fill_opacity=1)
    return VGroup(tip, body)


# KV cabinets (sections 3-4)
K_CX, V_CX = 0.9, 4.0
W_X = (K_CX + V_CX) / 2   # weights column between the cabinets
SLOT_P = 0.46


def slot_y(i):
    return 1.49 - SLOT_P * i


def cabinet(cx, color, title):
    body = RoundedRectangle(corner_radius=0.12, width=2.2, height=3.38, stroke_color=GREY_B, stroke_width=2,
                            fill_color=CACHE_FILL, fill_opacity=1).move_to([cx, 0.11, 0])
    slots = VGroup(*[RoundedRectangle(corner_radius=0.06, width=1.9, height=0.38, stroke_color=GREY_D,
                                      stroke_width=1.5, fill_color=BLACK, fill_opacity=0.4)
                     .move_to([cx, slot_y(i), 0]) for i in range(7)])
    head = Text(title, font_size=26, color=color).move_to([cx, 2.3, 0])
    return VGroup(body, slots, head)


def slot_word(word, cx, i):
    return Text(word, font_size=20, color=GREY_A).move_to([cx - 0.6, slot_y(i), 0]).set_z_index(3)


def slot_strip_pos(cx, i):
    return np.array([cx + 0.35, slot_y(i), 0])


def q_head():
    return RoundedRectangle(corner_radius=0.08, width=0.45, height=0.45, stroke_color=Q_COLOR, stroke_width=2,
                            fill_color=Q_COLOR, fill_opacity=0.45)


def kv_head(w=0.45):
    k = Rectangle(width=w, height=0.22, stroke_color=K_COLOR, stroke_width=2, fill_color=K_COLOR, fill_opacity=0.6)
    v = Rectangle(width=w, height=0.22, stroke_color=V_COLOR, stroke_width=2, fill_color=V_COLOR, fill_opacity=0.6)
    return VGroup(k, v).arrange(DOWN, buff=0.03)


class KVCacheVideo(VoicedScene):
    VIDEO = "v13"

    # ------------------------------------------------------------------ stage helpers
    def on_stage(self, keep=()):
        """Top-level mobjects on stage that are not part of another on-stage mobject."""
        top = [m for m in self.mobjects if not isinstance(m, ValueTracker)]
        inner = set()
        for m in top:
            for d in m.get_family()[1:]:
                inner.add(id(d))
        keep_ids = set()
        for k in keep:
            for d in k.get_family():
                keep_ids.add(id(d))
        return [m for m in top if id(m) not in inner and id(m) not in keep_ids]

    def clear_anims(self, keep=()):
        return [FadeOut(m) for m in self.on_stage(keep)]

    def construct(self):
        play_token_intro(self, TITLE, 13, TAGLINE)
        self.s1_naive()
        self.s2_mask()
        self.s3_cache()
        self.s4_new_row()
        self.s5_fast()
        self.s6_memory()
        self.s7_phases()
        self.s8_gqa()
        self.s9_code()
        self.s10_outro()
        finish(self)

    # 1. The problem ------------------------------------------------------------------
    def s1_naive(self):
        self.section(1)
        words = ["The", "cat", "sat", "on", "the", "mat", "and", "fell", "asleep", "."]
        toks = VGroup(*[token(w, color=TOKEN_COLOR if i < 5 else GEN_COLOR, font_size=26)
                        for i, w in enumerate(words)]).arrange(RIGHT, buff=0.12)
        toks.shift([-toks.width / 2 + 0.45 - toks.get_left()[0], 2.3 - toks.get_y(), 0])
        title = Text("one model run per new token", font_size=30).move_to([0, 3.35, 0])
        self.at("generating")
        self.play(LaggedStart(*[FadeIn(t, shift=0.2 * UP) for t in toks[:5]], lag_ratio=0.12),
                  FadeIn(title, shift=0.2 * DOWN), run_time=0.9)

        naive = Text("naive: every step reprocesses the whole text", font_size=30).move_to(title)
        rows = VGroup()
        for k, cue in enumerate(["once", "naively", "whole", "first", "long"], 1):
            n = 4 + k
            y = 1.3 - 0.5 * (k - 1)
            row = VGroup(*[Square(side_length=0.34, stroke_color=BLUE_B, stroke_width=1.5, fill_color=NEW_COLOR,
                                  fill_opacity=0.8).move_to([toks[i].get_x(), y, 0]) for i in range(n)])
            lab = label(f"step {k}", font_size=24).move_to([toks.get_left()[0] - 0.25, y, 0], aligned_edge=RIGHT)
            self.at(cue)
            anims = [LaggedStart(*[Indicate(toks[i][0], color=YELLOW, scale_factor=1.1) for i in range(n)],
                                 lag_ratio=0.08),
                     LaggedStart(*[FadeIn(c, shift=0.1 * DOWN) for c in row], lag_ratio=0.08), FadeIn(lab)]
            if k == 2:
                anims.append(FadeTransform(title, naive))
            self.play(*anims, run_time=0.8)
            if k == 1:
                self.at("new")
            self.play(FadeIn(toks[n], shift=0.3 * LEFT), run_time=0.4)
            rows.add(row)

        repeated = [c for row in rows[1:] for c in row[:-1]]
        rep = Text("repeated work", font_size=34, color=OLD_COLOR).move_to([0, -1.75, 0])
        count = label("26 of 35 token computations were already done", font_size=24).move_to([0, -2.45, 0])
        self.at("repeated")
        self.play(*[c.animate.set_fill(OLD_COLOR, 0.8).set_stroke(RED_A) for c in repeated],
                  FadeIn(rep, shift=0.15 * UP), FadeIn(count, shift=0.15 * UP), run_time=0.6)
        self.end_section()

    # 2. The observation ----------------------------------------------------------------
    def s2_mask(self):
        self.section(2)
        self.play(*self.clear_anims(), run_time=0.35)

        def gx(j):
            return 1.25 + 0.82 * j

        def gy(i):
            return 1.45 - 0.62 * i

        title = Text("attention with a causal mask", font_size=32).move_to([0, 3.3, 0])
        cells = {(i, j): Rectangle(width=0.72, height=0.56, stroke_color=GREY_D, stroke_width=1.5,
                                   fill_color=CACHE_FILL, fill_opacity=0.6).move_to([gx(j), gy(i), 0])
                 for i in range(5) for j in range(5)}
        row_labs = VGroup(*[small_token(w).move_to([0.25, gy(i), 0]) for i, w in enumerate(WORDS[:6])])
        col_labs = VGroup(*[small_token(w).move_to([gx(j), 2.1, 0]) for j, w in enumerate(WORDS[:6])])
        row_labs[5][0].set_stroke(GEN_COLOR).set_fill(GEN_COLOR, 0.3)
        col_labs[5][0].set_stroke(GEN_COLOR).set_fill(GEN_COLOR, 0.3)
        self.at("attention")
        self.play(FadeIn(title, shift=0.2 * DOWN), FadeIn(VGroup(*cells.values())), FadeIn(row_labs[:5]),
                  FadeIn(col_labs[:5]), run_time=0.8)
        self.at("mask")
        self.play(*[cells[i, j].animate.become(grid_cell(i, j, 0.72, 0.56).move_to(cells[i, j]))
                    for i in range(5) for j in range(5)], run_time=0.8)

        # the same five tokens on the left: arrows only point back
        arc_toks = VGroup(*[small_token(w).move_to([-5.6 + 1.0 * k, 0.15, 0]) for k, w in enumerate(WORDS[:5])])
        back = VGroup()
        for i in range(5):
            for j in range(i):
                back.add(CurvedArrow(arc_toks[i].get_top() + 0.05 * UP + 0.08 * LEFT,
                                     arc_toks[j].get_top() + 0.05 * UP + 0.08 * RIGHT, angle=PI / 2,
                                     color=GREEN_C, stroke_width=2.5, tip_length=0.15))
        back_lab = Text("each token looks back", font_size=24, color=GREEN_C).move_to([-3.6, 1.75, 0])
        self.at("earlier")
        self.play(FadeIn(arc_toks), LaggedStart(*[Create(a) for a in back], lag_ratio=0.08), FadeIn(back_lab),
                  run_time=1.0)
        fwd = CurvedArrow(arc_toks[1].get_bottom() + 0.05 * DOWN, arc_toks[3].get_bottom() + 0.05 * DOWN,
                          angle=PI / 2, color=RED_C, stroke_width=3, tip_length=0.15)
        low = fwd.get_bottom()
        cross = Text("✗", font_size=36, color=RED_C).move_to([fwd.get_x(), low[1], 0])
        fwd_lab = Text("never looks ahead", font_size=24, color=RED_C).move_to([-3.6, low[1] - 0.55, 0])
        self.at("later")
        self.play(Create(fwd), FadeIn(cross, scale=1.4), FadeIn(fwd_lab), run_time=0.6)

        # a new token arrives: one new row and one new (masked) column
        new_cells = {(i, 5): grid_cell(i, 5, 0.72, 0.56).move_to([gx(5), gy(i), 0]) for i in range(5)}
        new_cells.update({(5, j): grid_cell(5, j, 0.72, 0.56).move_to([gx(j), gy(5), 0]) for j in range(6)})
        new_row = VGroup(*[new_cells[5, j] for j in range(6)])
        row_box = SurroundingRectangle(new_row, buff=0.04, color=YELLOW, stroke_width=3)
        new_lab = Text("new token →", font_size=24, color=YELLOW).move_to([-0.1, gy(5), 0], aligned_edge=RIGHT)
        self.at("arrives")
        self.play(FadeOut(VGroup(arc_toks, back, back_lab, fwd, cross, fwd_lab)),
                  FadeIn(row_labs[5], shift=0.2 * UP), FadeIn(col_labs[5], shift=0.2 * LEFT),
                  LaggedStart(*[FadeIn(new_cells[i, 5], shift=0.1 * LEFT) for i in range(5)], lag_ratio=0.1),
                  LaggedStart(*[FadeIn(c, shift=0.1 * UP) for c in new_row], lag_ratio=0.1),
                  Create(row_box), FadeIn(new_lab), run_time=1.0)

        checks = VGroup(*[Text("✓", font_size=30, color=GREEN_C).move_to([6.15, gy(i), 0]) for i in range(5)])
        same_lab = Text("old rows: unchanged", font_size=26, color=GREEN_C).move_to([gx(2.5), -2.4, 0])
        self.at("changes")
        self.play(LaggedStart(*[FadeIn(c, scale=1.4) for c in checks], lag_ratio=0.12), FadeIn(same_lab),
                  run_time=0.7)

        kv_labs = VGroup(*[small_token(w).move_to([-5.3, gy(i), 0]) for i, w in enumerate(WORDS[:5])])
        ks = VGroup(*[strip(K_COLOR, seed=10 + i).move_to([-3.6, gy(i), 0]) for i in range(5)])
        vs = VGroup(*[strip(V_COLOR, seed=20 + i).move_to([-2.0, gy(i), 0]) for i in range(5)])
        k_head = Text("keys", font_size=26, color=K_COLOR).move_to([-3.6, 2.1, 0])
        v_head = Text("values", font_size=26, color=V_COLOR).move_to([-2.0, 2.1, 0])
        self.at("keys")
        self.play(FadeIn(kv_labs), FadeIn(k_head), LaggedStart(*[FadeIn(s, shift=0.2 * RIGHT) for s in ks],
                                                               lag_ratio=0.1), run_time=0.5)
        self.at("values")
        self.play(FadeIn(v_head), LaggedStart(*[FadeIn(s, shift=0.2 * RIGHT) for s in vs], lag_ratio=0.1),
                  run_time=0.5)
        last = Text("same as last time ✓", font_size=26, color=WHITE).move_to([-3.3, -2.4, 0])
        last[-1].set_color(GREEN_C)
        self.at("same")
        self.play(FadeIn(last, shift=0.15 * UP), Indicate(ks, color=K_COLOR, scale_factor=1.04),
                  Indicate(vs, color=V_COLOR, scale_factor=1.04), run_time=0.6)
        self.kv = (kv_labs, ks, vs, k_head, v_head)
        self.end_section()

    # 3. The cache ------------------------------------------------------------------------
    def s3_cache(self):
        self.section(3)
        kv_labs, ks, vs, k_head, v_head = self.kv
        self.play(*self.clear_anims(keep=[kv_labs, ks, vs, k_head, v_head]), run_time=0.3)
        for m in (kv_labs, ks, vs):
            m.set_z_index(3)

        k_cab = cabinet(K_CX, K_COLOR, "K cache")
        v_cab = cabinet(V_CX, V_COLOR, "V cache")
        self.at("save")
        self.play(FadeIn(k_cab, shift=0.2 * LEFT), FadeIn(v_cab, shift=0.2 * LEFT), run_time=0.8)
        title = Text("KV cache", font_size=40).move_to([W_X, 3.25, 0])
        self.at("cache")
        self.play(FadeIn(title, shift=0.2 * DOWN), run_time=0.6)
        # "mat", the token that just arrived, gets its own key and value too
        y5 = 1.45 - 0.62 * 5   # row 5 of the section-2 layout
        mat_lab = small_token("mat", color=GEN_COLOR).move_to([-5.3, y5, 0]).set_z_index(3)
        mat_k = strip(K_COLOR, seed=15).move_to([-3.6, y5, 0]).set_z_index(3)
        mat_v = strip(V_COLOR, seed=25).move_to([-2.0, y5, 0]).set_z_index(3)
        self.at("key")
        self.play(FadeIn(mat_lab), FadeIn(mat_k, shift=0.2 * RIGHT), Indicate(ks, color=K_COLOR, scale_factor=1.06),
                  run_time=0.34)
        self.at("value")
        self.play(FadeIn(mat_v, shift=0.2 * RIGHT), Indicate(vs, color=V_COLOR, scale_factor=1.06), run_time=0.35)
        kv_labs.add(mat_lab)
        ks.add(mat_k)
        vs.add(mat_v)

        shadows = VGroup()
        for cab in (k_cab, v_cab):
            for d in (2, 1):
                shadows.add(cab[0].copy().set_fill(BLACK, 1).set_stroke(GREY_C, 1.5, opacity=0.7 - 0.2 * d)
                            .shift(0.14 * d * (UP + RIGHT)).set_z_index(-1))
        badge = Text("× every layer", font_size=28, color=GREY_A).move_to([W_X, -2.3, 0])
        self.at("layer")
        self.play(FadeIn(shadows), FadeIn(badge, shift=0.15 * UP), run_time=0.5)

        # each row slides into its drawer; the token's own label text becomes the K drawer label
        k_words = VGroup(*[slot_word(w, K_CX, i) for i, w in enumerate(WORDS[:6])])
        v_words = VGroup(*[slot_word(w, V_CX, i) for i, w in enumerate(WORDS[:6])])
        self.at("store")
        # values first (they cross the still-empty K cabinet), then keys with their labels: no text ever overlaps
        self.play(*[vs[i].animate.move_to(slot_strip_pos(V_CX, i)) for i in range(6)],
                  FadeOut(k_head), FadeOut(v_head), run_time=0.45)
        self.play(*[ks[i].animate.move_to(slot_strip_pos(K_CX, i)) for i in range(6)],
                  *[ReplacementTransform(kv_labs[i][1], k_words[i]) for i in range(6)],
                  *[FadeOut(kv_labs[i][0]) for i in range(6)], FadeIn(v_words), run_time=0.45)
        self.cab = (k_cab, v_cab, title, shadows, badge, ks, vs, k_words, v_words)
        self.end_section()

    # 4. One new row ------------------------------------------------------------------------
    def s4_new_row(self):
        self.section(4)
        k_cab, v_cab, title, shadows, badge, ks, vs, k_words, v_words = self.cab
        self.play(FadeOut(shadows), FadeOut(badge), run_time=0.2)

        new = token("and", font_size=28).move_to([-4.6, 2.0, 0])
        new_lab = label("next token", font_size=24).move_to([-4.6, 2.8, 0])
        self.at("next")
        self.play(FadeIn(new, shift=0.4 * RIGHT), FadeIn(new_lab), run_time=0.5)

        q = strip(Q_COLOR, seed=31).move_to([-4.3, 1.0, 0])
        k = strip(K_COLOR, seed=32).move_to([-4.3, 0.35, 0]).set_z_index(3)
        v = strip(V_COLOR, seed=33).move_to([-4.3, -0.3, 0]).set_z_index(3)
        letters = VGroup(*[Text(s, font_size=28, color=c).move_to([-5.4, m.get_y(), 0])
                           for s, c, m in zip("qkv", [Q_COLOR, K_COLOR, V_COLOR], [q, k, v])])
        for cue, m, lt in zip(["query", "key", "value"], [q, k, v], letters):
            self.at(cue)
            self.play(FadeIn(m, shift=0.25 * DOWN), FadeIn(lt), run_time=0.4)
        kw, vw = slot_word("and", K_CX, 6), slot_word("and", V_CX, 6)
        self.play(k.animate.move_to(slot_strip_pos(K_CX, 6)), v.animate.move_to(slot_strip_pos(V_CX, 6)),
                  FadeOut(letters[1:]), FadeIn(kw), FadeIn(vw), run_time=0.55)

        keys = VGroup(*ks, k)
        vals = VGroup(*vs, v)
        fan = VGroup(*[Line(q.get_right() + 0.05 * RIGHT, [K_CX - 0.97, slot_y(i), 0], color=Q_COLOR,
                            stroke_width=2, stroke_opacity=0.8) for i in range(7)])
        self.at("compared")
        self.play(LaggedStart(*[Create(l) for l in fan], lag_ratio=0.08), run_time=0.6)

        wsq = VGroup(*[Square(side_length=0.32, stroke_color=GREY_D, stroke_width=1.5, fill_color=W_COLOR,
                              fill_opacity=weight_opacity(6, i)).move_to([W_X, slot_y(i), 0]) for i in range(7)])
        w_lab = Text("weights", font_size=24, color=W_COLOR).move_to([W_X, -1.95, 0])
        self.at("keys")
        self.play(Indicate(keys, color=K_COLOR, scale_factor=1.04),
                  LaggedStart(*[FadeIn(s, shift=0.2 * RIGHT) for s in wsq], lag_ratio=0.08), FadeIn(w_lab),
                  run_time=0.7)
        self.play(FadeOut(fan), run_time=0.3)

        out = strip(WHITE, seed=34, direction=DOWN).move_to([6.0, 0.11, 0])
        out_lab = Text("output", font_size=24).move_to([6.0, 2.3, 0])
        mix = VGroup(*[Line([V_CX + 1.0, slot_y(i), 0], out.get_left() + 0.05 * LEFT, color=V_COLOR,
                            stroke_width=1 + 5 * weight_opacity(6, i), stroke_opacity=0.9) for i in range(7)])
        self.at("mix")
        self.play(Indicate(wsq, color=W_COLOR, scale_factor=1.1), LaggedStart(*[Create(l) for l in mix],
                                                                               lag_ratio=0.06),
                  FadeIn(out_lab), run_time=0.6)
        self.at("values")
        self.play(Indicate(vals, color=V_COLOR, scale_factor=1.04), FadeIn(out, shift=0.2 * RIGHT), run_time=0.5)

        # one new row of the attention grid, instead of the whole grid
        def mx(j):
            return -1.9 + 0.52 * j

        def my(i):
            return 2.0 - 0.52 * i

        old = VGroup(*[grid_cell(i, j, 0.46, 0.46).move_to([mx(j), my(i), 0]) for i in range(6) for j in range(7)])
        for c in old:
            c.set_fill(opacity=c.get_fill_opacity() * 0.35).set_stroke(opacity=0.4)
        row_cells = VGroup(*[grid_cell(6, j, 0.46, 0.46).move_to([mx(j), my(6), 0]) for j in range(7)])
        words = VGroup(*[Text(w, font_size=20, color=GREY_B if i < 6 else WHITE).move_to([-2.35, my(i), 0],
                                                                                         aligned_edge=RIGHT)
                         for i, w in enumerate(WORDS)])
        self.at("one")
        self.play(*self.clear_anims(keep=[wsq]), ReplacementTransform(wsq, row_cells), FadeIn(old), FadeIn(words),
                  run_time=0.45)
        box = SurroundingRectangle(row_cells, buff=0.05, color=YELLOW, stroke_width=3)
        one_lab = Text("one new row", font_size=28, color=YELLOW).next_to(box, RIGHT, buff=0.35)
        self.at("row")
        self.play(Create(box), FadeIn(one_lab, shift=0.2 * LEFT), run_time=0.3)
        instead = Text("instead of redoing the whole grid", font_size=28, color=GREY_B).move_to([0, -2.2, 0])
        self.at("instead")
        self.play(FadeIn(instead, shift=0.15 * UP), run_time=0.5)
        self.at("whole")
        self.play(Indicate(old, color=GREY_B, scale_factor=1.03), run_time=0.6)
        self.end_section()

    # 5. Why it's fast ----------------------------------------------------------------------
    def s5_fast(self):
        self.section(5)
        self.play(*self.clear_anims(), run_time=0.4)
        def column(n, x, base=-1.3):
            cells = VGroup()
            for m in range(n):
                c = NEW_COLOR if m == n - 1 else OLD_COLOR
                cells.add(Square(side_length=0.26, stroke_color=GREY_D, stroke_width=1, fill_color=c,
                                 fill_opacity=0.85).move_to([x, base + 0.13 + 0.29 * m, 0]))
            return cells

        def panel(cx, head, sub, color, sizes, total):
            h = Text(head, font_size=32, color=color).move_to([cx, 2.95, 0])
            sb = label(sub, font_size=24).move_to([cx, 2.4, 0])
            cols = VGroup(*[column(n, cx + 0.85 * (k - 2.5)) for k, n in enumerate(sizes)])
            base = Line([cx - 2.5, -1.34, 0], [cx + 2.5, -1.34, 0], color=GREY_D, stroke_width=2)
            tot = Text(total, font_size=28, color=color).move_to([cx, -1.8, 0])
            return h, sb, cols, base, tot

        a_h, a_sub, a_cols, a_base, a_tot = panel(-3.4, "with cache", "one token per step", GREEN_C, [1] * 6,
                                                  "total: 6 tokens")
        self.at("one")
        self.play(FadeIn(a_h, shift=0.15 * DOWN), FadeIn(a_sub), Create(a_base),
                  LaggedStart(*[FadeIn(c, shift=0.15 * UP) for c in a_cols], lag_ratio=0.15), FadeIn(a_tot),
                  run_time=0.9)
        b_h, b_sub, b_cols, b_base, b_tot = panel(3.4, "without cache", "the whole text every step", OLD_COLOR,
                                                  list(range(6, 12)), "total: 51 tokens")
        self.at("entire")
        self.play(FadeIn(b_h, shift=0.15 * DOWN), FadeIn(b_sub), Create(b_base),
                  LaggedStart(*[FadeIn(c, shift=0.15 * UP) for c in b_cols], lag_ratio=0.15), FadeIn(b_tot),
                  run_time=1.0)

        bubble = chat_bubble(10.2, 0.85).move_to([0.3, -2.85, 0])
        sent = "The cat sat on the mat and fell asleep in the sun."
        text = Text(sent, font_size=26).move_to(bubble[1]).align_to(bubble[1], LEFT).shift(0.35 * RIGHT)
        bot = label("chatbot", font_size=22).next_to(bubble[1], LEFT, buff=0.2)
        self.at("chatbots")
        self.play(FadeIn(bubble), FadeIn(bot), run_time=0.5)
        pieces, pos = [], 0
        for w in sent.split(" "):
            pieces.append(gslice(text, sent, pos, pos + len(w)))
            pos += len(w) + 1
        self.at("stream")
        self.play(LaggedStart(*[FadeIn(p, shift=0.1 * RIGHT) for p in pieces], lag_ratio=1.0), run_time=1.8)
        self.end_section()

    # 6. The cost: memory ---------------------------------------------------------------------
    def s6_memory(self):
        self.section(6)
        self.play(*self.clear_anims(), run_time=0.4)
        title = Text("the price: memory", font_size=34).move_to([0, 3.35, 0])
        outline = Rectangle(width=11.0, height=0.55, stroke_color=GREY_B, stroke_width=2.5).move_to([-0.5, -2.25, 0])
        x0 = outline.get_left()[0] + 0.03

        def fill(w):
            top = Rectangle(width=w, height=0.245, stroke_width=0, fill_color=K_COLOR, fill_opacity=0.85)
            bot = Rectangle(width=w, height=0.245, stroke_width=0, fill_color=V_COLOR, fill_opacity=0.85)
            g = VGroup(top, bot).arrange(DOWN, buff=0.0)
            return g.move_to([x0, -2.25, 0], aligned_edge=LEFT)

        bar = fill(0.35)
        mem_lab = Text("GPU memory", font_size=24, color=GREY_B).move_to([outline.get_left()[0], -1.7, 0],
                                                                        aligned_edge=LEFT)
        legend = VGroup(Square(0.22, stroke_width=0, fill_color=K_COLOR, fill_opacity=0.85),
                        Square(0.22, stroke_width=0, fill_color=V_COLOR, fill_opacity=0.85),
                        Text("KV cache", font_size=24, color=GREY_B)).arrange(RIGHT, buff=0.1)
        legend.move_to([outline.get_right()[0], -1.7, 0], aligned_edge=RIGHT)
        self.at("memory")
        self.play(FadeIn(title, shift=0.2 * DOWN), Create(outline), FadeIn(mem_lab), FadeIn(bar), FadeIn(legend),
                  run_time=0.8)

        words = ["The", "cat", "sat", "on", "the", "mat", "and", "fell"]
        pairs = VGroup()
        for k, w in enumerate(words):
            x = -5.3 + 0.85 * k
            tok = small_token(w).move_to([x, 2.2, 0])
            kk = strip(K_COLOR, seed=40 + k, cell=0.22, buff=0.03, direction=DOWN).move_to([x - 0.14, 1.3, 0])
            vv = strip(V_COLOR, seed=50 + k, cell=0.22, buff=0.03, direction=DOWN).move_to([x + 0.14, 1.3, 0])
            pairs.add(VGroup(tok, kk, vv))
        self.at("key")
        self.play(FadeIn(pairs[0][0]), FadeIn(pairs[0][1], shift=0.2 * DOWN), run_time=0.4)
        self.at("value")
        self.play(FadeIn(pairs[0][2], shift=0.2 * DOWN), run_time=0.4)
        plate = RoundedRectangle(corner_radius=0.12, width=pairs.width + 0.5, height=2.0, stroke_color=GREY_B,
                                 stroke_width=2, fill_color=CACHE_FILL, fill_opacity=1).move_to(
            [pairs.get_x(), 1.62, 0]).set_z_index(-1)
        tok_lab = Text("every token", font_size=26).move_to([3.9, 2.2, 0])
        self.at("every")
        self.play(FadeIn(plate), LaggedStart(*[FadeIn(p, shift=0.2 * LEFT) for p in pairs[1:]], lag_ratio=0.12),
                  FadeIn(tok_lab), run_time=0.8)
        backs = VGroup(*[plate.copy().set_fill(BLACK, 1).set_stroke(GREY_C, 1.5, opacity=0.8 - 0.2 * d)
                         .shift(0.14 * d * (UP + RIGHT)).set_z_index(-2 - d) for d in (1, 2)])
        lay_lab = Text("× every layer", font_size=26).move_to([3.9, 1.3, 0])
        self.at("layer")
        self.play(FadeIn(backs), FadeIn(lay_lab), run_time=0.5)

        fs = "2 (K, V) × 12 layers × 768 = 18,432 numbers per token"
        form = Text(fs, font_size=30, t2c={"K": K_COLOR, "V": V_COLOR, "18,432": YELLOW}).move_to([0, -0.55, 0])
        head = Text("GPT-2 small", font_size=26, color=GREY_B).move_to([0, 0.15, 0])
        i12, i768, ieq = fs.index("× 12"), fs.index("× 768"), fs.index("=")
        inum = fs.index("numbers")
        self.at("gpt")
        self.play(FadeIn(head), FadeIn(gslice(form, fs, 0, i12)), run_time=0.4)
        self.at("too")
        self.play(FadeIn(gslice(form, fs, i12, i768)), run_time=0.3)
        self.at("small")
        self.play(FadeIn(gslice(form, fs, i768, ieq)), run_time=0.3)
        self.at("18")
        self.play(FadeIn(gslice(form, fs, ieq, inum), scale=1.2), run_time=0.4)
        self.at("numbers")
        self.play(FadeIn(gslice(form, fs, inum, len(fs))), run_time=0.4)

        big = label("big models", font_size=26).move_to([-3.7, -3.1, 0])
        self.at("big")
        self.play(Transform(bar, fill(3.2)), FadeIn(big), run_time=0.6)
        long = label("long conversations", font_size=26).move_to([0.9, -3.1, 0])
        self.at("long")
        self.play(Transform(bar, fill(8.6)), FadeIn(long), run_time=0.8)
        gb = Text("gigabytes", font_size=30, color=RED_C).move_to([4.95, -3.1, 0])
        self.at("gigabytes")
        self.play(Transform(bar, fill(6.65 - x0)), outline.animate.set_stroke(RED_C), FadeIn(gb, scale=1.2),
                  run_time=0.8)
        self.end_section()

    # 7. Prefill and decode --------------------------------------------------------------------
    def s7_phases(self):
        self.section(7)
        self.play(*self.clear_anims(), run_time=0.4)
        words = ["The", "cat", "sat", "on", "the", "mat", "and"]
        toks = VGroup(*[token(w, color=TOKEN_COLOR if i < 5 else GEN_COLOR, font_size=26)
                        for i, w in enumerate(words)]).arrange(RIGHT, buff=0.12)
        toks[5:].shift(0.6 * RIGHT)
        toks.move_to([0, 1.0, 0])
        prompt, gen = toks[:5], toks[5:]

        slots = VGroup()
        for i, t in enumerate(toks):
            kq = Square(side_length=0.34, stroke_color=GREY_D, stroke_width=1.5, fill_color=BLACK, fill_opacity=0.5)
            vq = kq.copy()
            slots.add(VGroup(kq, vq).arrange(DOWN, buff=0.06).move_to([t.get_x(), -0.6, 0]))
        frame = RoundedRectangle(corner_radius=0.12, width=toks.width + 0.4, height=1.15, stroke_color=GREY_B,
                                 stroke_width=2, fill_color=CACHE_FILL, fill_opacity=1).move_to([0, -0.6, 0])
        cache_lab = Text("KV cache", font_size=24, color=GREY_B).next_to(frame, LEFT, buff=0.25)

        pre = Text("1 · prefill", font_size=30, color=GREY_D).move_to([prompt.get_x(), 2.2, 0])
        dec = Text("2 · decode", font_size=30, color=GREY_D).move_to([gen.get_x(), 2.2, 0])
        sep = DashedLine([toks[4].get_right()[0] + 0.36, 2.5, 0], [toks[4].get_right()[0] + 0.36, -1.35, 0],
                         color=GREY_D, dash_length=0.1)
        self.at("phases")
        self.play(FadeIn(pre), FadeIn(dec), Create(sep), FadeIn(frame), FadeIn(slots), FadeIn(cache_lab),
                  run_time=0.8)
        self.at("pre")
        self.play(pre.animate.set_color(WHITE), run_time=0.4)
        self.at("whole")
        self.play(*[FadeIn(t, shift=0.3 * DOWN) for t in prompt], run_time=0.5)
        once = Text("all at once", font_size=26, color=YELLOW).move_to([prompt.get_x(), -2.3, 0])
        self.at("once")
        self.play(*[Indicate(t[0], color=YELLOW, scale_factor=1.12) for t in prompt], FadeIn(once), run_time=0.5)
        self.at("fill")
        self.play(*[s[0].animate.set_fill(K_COLOR, 0.8).set_stroke(K_COLOR) for s in slots[:5]],
                  *[s[1].animate.set_fill(V_COLOR, 0.8).set_stroke(V_COLOR) for s in slots[:5]], run_time=0.5)

        step = Text("one at a time", font_size=26, color=YELLOW).move_to([gen.get_x(), -2.3, 0])
        self.at("decode")
        self.play(pre.animate.set_color(GREY_B), dec.animate.set_color(WHITE), run_time=0.4)
        for k, cue in enumerate(["one", "time"]):
            i = 5 + k
            self.at(cue)
            self.play(FadeIn(toks[i], shift=0.3 * DOWN), slots[i][0].animate.set_fill(K_COLOR, 0.8).set_stroke(K_COLOR),
                      slots[i][1].animate.set_fill(V_COLOR, 0.8).set_stroke(V_COLOR),
                      *([FadeIn(step)] if k == 0 else []), run_time=0.4)
        reuse_box = SurroundingRectangle(slots[:6], buff=0.1, color=YELLOW, stroke_width=3)
        reuse = Text("reused, not recomputed", font_size=24, color=YELLOW).move_to([reuse_box.get_x(), -1.6, 0])
        self.at("reusing")
        self.play(Create(reuse_box), FadeIn(reuse), Indicate(toks[6][0], color=YELLOW, scale_factor=1.12),
                  run_time=0.6)
        self.end_section()

    # 8. Shrinking the cache (grouped-query attention) ---------------------------------------
    def s8_gqa(self):
        self.section(8)
        self.play(*self.clear_anims(), run_time=0.4)
        title = Text("shrinking the cache", font_size=34).move_to([0, 3.35, 0])
        self.at("large")
        self.play(FadeIn(title, shift=0.2 * DOWN), run_time=0.5)

        legend = VGroup(Text("query heads", font_size=24, color=Q_COLOR),
                        Text("key / value heads", font_size=24, t2c={"key": K_COLOR, "value": V_COLOR})
                        ).arrange(RIGHT, buff=1.0).move_to([0, 2.75, 0])
        lx, rx = -3.4, 3.4
        l_title = Text("standard", font_size=28, color=GREY_B).move_to([lx, 2.05, 0])
        lq = VGroup(*[q_head().move_to([lx + 0.62 * (i - 3.5), 1.0, 0]) for i in range(8)])
        lkv = VGroup(*[kv_head().move_to([lx + 0.62 * (i - 3.5), -0.8, 0]) for i in range(8)])
        llines = VGroup(*[Line(a.get_bottom(), b.get_top(), buff=0.06, color=GREY_B, stroke_width=2)
                          for a, b in zip(lq, lkv)])
        l_cap = label("8 K/V heads cached", font_size=24).move_to([lx, -1.75, 0])
        illus = label("illustrative", font_size=20).move_to([5.9, -3.55, 0])
        self.at("shrink")
        self.play(FadeIn(legend), FadeIn(l_title), FadeIn(lq), FadeIn(lkv), Create(llines), FadeIn(l_cap),
                  FadeIn(illus), run_time=1.0)

        r_title = Text("grouped-query", font_size=28).move_to([rx, 2.05, 0])
        rq = VGroup(*[q_head().move_to([rx + 0.62 * (i - 3.5), 1.0, 0]) for i in range(8)])
        self.at("several")
        self.play(FadeIn(r_title), LaggedStart(*[FadeIn(h, shift=0.15 * DOWN) for h in rq], lag_ratio=0.08),
                  run_time=0.6)
        rkv = VGroup(*[kv_head(0.7).move_to([rx + s * 1.24, -0.8, 0]) for s in (-1, 1)])
        rlines = VGroup(*[Line(h.get_bottom(), rkv[i // 4].get_top(), buff=0.06, color=GREY_A, stroke_width=2.5)
                          for i, h in enumerate(rq)])
        self.at("share")
        self.play(FadeIn(rkv, shift=0.15 * UP), LaggedStart(*[Create(l) for l in rlines], lag_ratio=0.06),
                  run_time=0.7)
        self.at("keys")
        self.play(lkv.animate.set_opacity(0.2), llines.animate.set_stroke(opacity=0.2),
                  lq.animate.set_opacity(0.35), Indicate(rkv, scale_factor=1.12), run_time=0.5)
        r_cap = Text("2 K/V heads cached: 4× smaller", font_size=24, color=GREEN_C).move_to([rx, -1.75, 0])
        self.at("values")
        self.play(FadeIn(r_cap, shift=0.15 * UP), run_time=0.5)
        gqa = Text("grouped-query attention (GQA)", font_size=34).move_to([0, -2.8, 0])
        self.at("group")
        self.play(FadeIn(gqa, shift=0.2 * UP), run_time=0.6)
        self.end_section()

    # 9. Code ------------------------------------------------------------------------------------
    def s9_code(self):
        self.section(9)
        code, hl = code_panel(CODE, 24)   # code_panel scales it to the 12.8-unit maximum width
        VGroup(code, hl).move_to([0, 0.4, 0])
        hl.match_y(code.line_numbers[1])
        self.play(*self.clear_anims(), FadeIn(code, shift=0.3 * UP), run_time=0.6)
        self.at("compute")
        self.play(Create(hl), run_time=0.4)

        row_h = hl.height

        def span(a, b):
            ya, yb = code.line_numbers[a].get_y(), code.line_numbers[b].get_y()
            return hl.copy().stretch_to_fit_height(abs(ya - yb) + row_h).set_y((ya + yb) / 2)

        self.at("append")
        self.play(Transform(hl, span(2, 3)), run_time=0.4)
        self.at("attend")
        self.play(Transform(hl, span(4, 6)), run_time=0.4)
        cap = Text("no mask needed: the cache only holds the past", font_size=26, color=YELLOW)
        cap.next_to(code, DOWN, buff=0.4)
        self.at("mask")
        self.play(Transform(hl, span(5, 5)), FadeIn(cap, shift=0.15 * UP), run_time=0.5)
        self.end_section()

    # 10. Outro ------------------------------------------------------------------------------------
    def s10_outro(self):
        self.section(10)
        self.play(*self.clear_anims(), run_time=0.5)
        gpt = VGroup(RoundedRectangle(corner_radius=0.15, width=2.0, height=1.2, stroke_color=PURPLE_B,
                                      fill_color=PURPLE_B, fill_opacity=0.25),
                     Text("GPT", font_size=40)).move_to([-4.6, 0.9, 0])
        gpt[1].move_to(gpt[0])
        self.at("gpt")
        self.play(FadeIn(gpt, scale=0.9), run_time=0.5)
        trained = label("trained on internet text", font_size=22).move_to([-4.6, -0.05, 0])
        self.at("internet")
        self.play(FadeIn(trained, shift=0.1 * UP), run_time=0.4)

        box = RoundedRectangle(corner_radius=0.15, width=9.0, height=1.7, stroke_color=GREY_B,
                               stroke_width=2).move_to([1.95, 0.9, 0])
        box_lab = label("autocomplete", font_size=22).move_to([box.get_left()[0], 2.0, 0], aligned_edge=LEFT)
        arrow = Arrow(gpt[0].get_right(), box.get_left(), buff=0.1, color=GREY_B, stroke_width=3)
        prompt = Text("How do I boil an egg?", font_size=26).move_to([box.get_left()[0] + 0.3, 1.3, 0],
                                                                    aligned_edge=LEFT)
        cont_s = "How do I fry an egg? How do I poach an egg?"
        cont = Text(cont_s, font_size=26, color=GREY_B).move_to([box.get_left()[0] + 0.3, 0.55, 0],
                                                                aligned_edge=LEFT)
        self.at("brilliant")
        self.play(Create(box), FadeIn(box_lab), GrowArrow(arrow), FadeIn(prompt), run_time=0.4)
        pieces, pos = [], 0
        for w in cont_s.split(" "):
            pieces.append(gslice(cont, cont_s, pos, pos + len(w)))
            pos += len(w) + 1
        self.at("autocomplete")
        self.play(LaggedStart(*[FadeIn(p, shift=0.1 * RIGHT) for p in pieces], lag_ratio=1.0), run_time=1.1)

        bub = chat_bubble(1.3, 0.95, color=GREEN_C).move_to([-1.0, -1.9, 0])
        qmark = Text("?", font_size=44, color=GREEN_C).move_to(bub[1])
        helper = Text("a helpful assistant", font_size=28, color=GREY_A).next_to(bub[1], RIGHT, buff=0.4)
        self.at("assistant")
        self.play(FadeIn(VGroup(bub, qmark), scale=0.8), FadeIn(helper, shift=0.15 * LEFT), run_time=0.6)
        path = CurvedArrow(trained.get_bottom() + 0.1 * DOWN, bub[1].get_left() + 0.1 * LEFT, angle=PI / 3,
                           color=GREY_B, stroke_width=3, tip_length=0.2)
        nxt = label("next time", font_size=24).next_to(path.point_from_proportion(0.5), DL, buff=0.12)
        self.at("next")
        self.play(Create(path), FadeIn(nxt), run_time=0.7)

        card = next_up_card(NEXT)
        self.at("chatbot")
        self.play(*self.clear_anims(), FadeIn(card, shift=0.2 * UP), run_time=0.8)
        self.end_section()
