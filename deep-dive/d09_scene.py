"""How LLMs Work: Deep Dive, episode 9 — Multi-Query and Grouped-Query Attention.

Render from the repo root:  ./render.sh deep-dive d09
Every number on screen comes from code/d09_gqa/gqa.py (Qwen2.5-0.5B-Instruct's real KV cache in bfloat16; GPT-2 small
with keys and values averaged per group, no retraining; a tiny GPT trained 2,000 steps with 8, 2 and 1 K/V heads).
"""
from manim import *

from common import MODEL_COLOR, MONO, TOKEN_COLOR, code_panel, finish, highlight, next_up_card, token
from intro import play_token_intro
from d09_script import NEXT, TAGLINE, TITLE
from voiced_scene import VoicedScene

SERIES_LABEL = "HOW LLMs WORK: DEEP DIVE  ·  EPISODE 9"
Q_COL, KV_COL = YELLOW, GREEN_C
CODE = """self.q = nn.Linear(D, H * HD)                      # 8 query heads
self.k = nn.Linear(D, kv_heads * HD)               # fewer key heads
self.v = nn.Linear(D, kv_heads * HD)               # fewer value heads

k = k.repeat_interleave(H // kv_heads, dim=1)      # one copy per query in the group
v = v.repeat_interleave(H // kv_heads, dim=1)
att = F.scaled_dot_product_attention(q, k, v, is_causal=True)"""


def heads_diagram(n_q, n_kv, width=3.6, size=0.3):
    """Query heads on top, key/value heads below, each query linked to its group's K/V head."""
    qs = VGroup(*[Square(size, color=Q_COL, fill_opacity=0.7, stroke_width=1) for _ in range(n_q)])
    qs.arrange(RIGHT, buff=(width - n_q * size) / max(1, n_q - 1))
    kvs = VGroup(*[Square(size, color=KV_COL, fill_opacity=0.7, stroke_width=1) for _ in range(n_kv)])
    per = n_q // n_kv
    for g, kv in enumerate(kvs):
        kv.move_to([VGroup(*qs[g * per:(g + 1) * per]).get_x(), qs.get_y() - 1.3, 0])
    lines = VGroup(*[Line(qs[i].get_bottom(), kvs[i // per].get_top(), stroke_width=1.5, color=GREY_B)
                     for i in range(n_q)])
    return VGroup(lines, qs, kvs)


class GQAVideo(VoicedScene):
    VIDEO = "d09"

    def construct(self):
        play_token_intro(self, TITLE, 9, TAGLINE, label=SERIES_LABEL)
        self.hook()          # 1
        self.idea()          # 2
        self.qwen()          # 3
        self.at_scale()      # 4
        self.why()           # 5
        self.scratch()       # 6
        self.after()         # 7
        self.speed()         # 8
        self.code()          # 9
        self.outro()         # 10

    def clear_stage(self, *keep, run_time=0.6):
        self.play(*[FadeOut(m) for m in self.mobjects if m not in keep], run_time=run_time)

    # ------------------------------------------------------------------ 1. hook
    def hook(self):
        self.section(1)
        cols = VGroup()
        for t in range(10):
            stack = VGroup(*[Rectangle(width=0.5, height=0.16, stroke_width=0, fill_color=KV_COL if r % 2 else BLUE_C,
                                       fill_opacity=0.8) for r in range(12)]).arrange(DOWN, buff=0.03)
            cols.add(stack)
        cols.arrange(RIGHT, buff=0.15).move_to([0, 0.0, 0])
        lab = Text("a key and a value per token, per layer, per head: the KV cache", font_size=26).to_edge(UP, buff=0.8)
        self.at("writes")
        self.play(FadeIn(lab), run_time=0.4)
        self.at("token")
        self.play(LaggedStart(*[FadeIn(c, shift=0.2 * DOWN) for c in cols[:4]], lag_ratio=0.2), run_time=0.8)
        self.at("huge")
        self.play(LaggedStart(*[FadeIn(c, shift=0.2 * DOWN) for c in cols[4:]], lag_ratio=0.1), run_time=0.8)
        slow = Text("reading it slows every step", font_size=26, color=RED_B).move_to([0, -1.9, 0])
        self.at("slows")
        self.play(FadeIn(slow), run_time=0.4)
        fix = Text("fix: let heads share keys and values", font_size=30, color=YELLOW).move_to([0, -2.8, 0])
        self.at("share")
        self.play(FadeIn(fix), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 2. the idea
    def idea(self):
        self.section(2)
        self.clear_stage()
        specs = [("multi-head", 8, 8, "every query head has its own"), ("grouped-query", 8, 2, "groups share one"),
                 ("multi-query", 8, 1, "all share a single one")]
        panels = VGroup()
        for name, nq, nkv, note in specs:
            d = heads_diagram(nq, nkv, width=3.4)
            t = Text(name, font_size=26).next_to(d, UP, buff=0.35)
            nt = Text(note, font_size=18, color=GREY_B).next_to(d, DOWN, buff=0.3)
            panels.add(VGroup(t, d, nt))
        panels.arrange(RIGHT, buff=0.7).move_to([0, 0.2, 0])
        legend = VGroup(Square(0.25, color=Q_COL, fill_opacity=0.7), Text("query heads", font_size=20),
                        Square(0.25, color=KV_COL, fill_opacity=0.7), Text("key/value heads", font_size=20))
        legend.arrange(RIGHT, buff=0.2).to_edge(DOWN, buff=0.6)
        self.at("classic")
        self.play(FadeIn(panels[0]), FadeIn(legend), run_time=0.6)
        self.at("single")
        self.play(FadeIn(panels[2]), run_time=0.6)
        self.at("between")
        self.play(FadeIn(panels[1]), run_time=0.6)
        self.end_section()

    # ------------------------------------------------------------------ 3. Qwen
    def qwen(self):
        self.section(3)
        self.clear_stage()
        head = Text("Qwen2.5-0.5B, really", font_size=32).to_edge(UP, buff=0.5)
        self.at("qen", "kwen", "qwen")
        self.play(FadeIn(head), run_time=0.4)
        d = heads_diagram(14, 2, width=7.0, size=0.34).move_to([0, 1.1, 0])
        info = Text("24 layers · 14 query heads · 2 key/value heads", font_size=24, color=GREY_A).next_to(d, DOWN, 0.3)
        self.at("24")
        self.play(FadeIn(d), FadeIn(info), run_time=0.7)
        grp = Text("groups of 7", font_size=24, color=KV_COL).next_to(d, RIGHT, buff=0.3)
        self.at("seven")
        self.play(FadeIn(grp), run_time=0.3)
        shape = Text("cached keys, layer 0: (1, 2, 11, 64)", font=MONO, font_size=22).move_to([0, -1.0, 0])
        self.at("cached")
        self.play(FadeIn(shape), run_time=0.4)
        a = Text("12,288 bytes per token (measured)", font=MONO, font_size=24, color=GREEN_B).move_to([0, -1.9, 0])
        self.at("12")
        self.play(FadeIn(a), run_time=0.4)
        b = Text("one K/V per query head: 86,016  (×7)", font=MONO, font_size=24, color=RED_B).move_to([0, -2.7, 0])
        self.at("86")
        self.play(FadeIn(b), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 4. at scale
    def at_scale(self):
        self.section(4)
        self.clear_stage()
        rows = [("32,768 tokens", 384, 2688), ("131,072 tokens", 1536, 10752)]
        scale = 11.5 / 10752
        groups = VGroup()
        for k, (name, g, f) in enumerate(rows):
            y = 1.4 - 2.3 * k
            lab = Text(name, font_size=24).move_to([-6.0, y + 0.45, 0], aligned_edge=LEFT)
            bg = Rectangle(width=max(0.05, g * scale), height=0.35, stroke_width=0, fill_color=GREEN_C, fill_opacity=0.9)
            bf = Rectangle(width=f * scale, height=0.35, stroke_width=0, fill_color=RED_C, fill_opacity=0.8)
            bg.move_to([-6.0, y - 0.05, 0], aligned_edge=LEFT)
            bf.move_to([-6.0, y - 0.5, 0], aligned_edge=LEFT)
            tg = Text(f"{g:,} MiB  grouped (2 K/V heads)", font=MONO, font_size=20).next_to(bg, RIGHT, buff=0.15)
            tf = Text(f"{f:,} MiB  one per query head", font=MONO, font_size=20).next_to(bf, RIGHT, buff=0.15)
            if tf.get_right()[0] > 6.9:
                tf.move_to(bf).set_color(BLACK)
            groups.add(VGroup(lab, bg, tg, bf, tf))
        self.at("32")
        self.play(FadeIn(groups[0][0]), run_time=0.3)
        self.at("384")
        self.play(GrowFromEdge(groups[0][1], LEFT), FadeIn(groups[0][2]), run_time=0.4)
        self.at("gigabytes")
        self.play(GrowFromEdge(groups[0][3], LEFT), FadeIn(groups[0][4]), run_time=0.5)
        self.at("131")
        self.play(FadeIn(groups[1][0]), GrowFromEdge(groups[1][1], LEFT), FadeIn(groups[1][2]), run_time=0.5)
        self.at("10")
        self.play(GrowFromEdge(groups[1][3], LEFT), FadeIn(groups[1][4]), run_time=0.6)
        self.end_section()

    # ------------------------------------------------------------------ 5. why it works
    def why(self):
        self.section(5)
        self.clear_stage()
        qs = VGroup(*[VGroup(Square(0.5, color=Q_COL, fill_opacity=0.7), Text("?", font_size=24).set_z_index(1))
                      for _ in range(7)])
        for q in qs:
            q[1].move_to(q[0])
        qs.arrange(RIGHT, buff=0.35).move_to([0, 1.4, 0])
        idx = RoundedRectangle(width=7.0, height=1.0, corner_radius=0.15, color=KV_COL, fill_opacity=0.25)
        idx.move_to([0, -1.3, 0])
        il = Text("one shared index: keys + values", font_size=24, color=KV_COL).move_to(idx)
        arrows = VGroup(*[Arrow(q.get_bottom(), [q.get_x() * 0.8, idx.get_top()[1], 0], buff=0.1, stroke_width=2, color=GREY_B) for q in qs])
        self.at("question")
        self.play(FadeIn(qs, lag_ratio=0.1), run_time=0.6)
        self.at("shared")
        self.play(FadeIn(idx), FadeIn(il), LaggedStart(*[GrowArrow(a) for a in arrows], lag_ratio=0.05), run_time=0.8)
        cap = Text("7 different questions, the same index", font_size=28, color=YELLOW).to_edge(DOWN, buff=0.6)
        self.at("index")
        self.play(FadeIn(cap), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 6. from scratch
    def scratch(self):
        self.section(6)
        self.clear_stage()
        head = Text("tiny GPT, 8 query heads, trained 2,000 steps", font_size=28).to_edge(UP, buff=0.6)
        self.at("three")
        self.play(FadeIn(head), run_time=0.4)
        hdr = ["K/V heads", "val loss", "cache / token"]
        data = [("8", "1.673", "1,024"), ("2", "1.685", "256"), ("1", "1.704", "128")]
        xs = [-3.5, 0.0, 3.5]
        hrow = VGroup(*[Text(h, font_size=24, color=GREY_B).move_to([x, 1.6, 0]) for h, x in zip(hdr, xs)])
        self.play(FadeIn(hrow), run_time=0.3)
        rows = VGroup()
        for k, r in enumerate(data):
            rows.add(VGroup(*[Text(v, font=MONO, font_size=28).move_to([x, 0.7 - 0.8 * k, 0]) for v, x in zip(r, xs)]))
        self.play(*[FadeIn(r[0]) for r in rows], run_time=0.4)
        for k, cue in enumerate(["673", "685", "704"]):
            self.at(cue)
            self.play(FadeIn(rows[k][1]), run_time=0.3)
        self.at("1024")
        self.play(FadeIn(rows[0][2]), run_time=0.3)
        self.at("256")
        self.play(FadeIn(rows[1][2]), run_time=0.3)
        self.at("128")
        self.play(FadeIn(rows[2][2]), run_time=0.3)
        cap = Text("+0.03 loss for an 8× smaller cache", font_size=28, color=YELLOW).move_to([0, -2.4, 0])
        self.at("price")
        self.play(FadeIn(cap), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 7. after training
    def after(self):
        self.section(7)
        self.clear_stage()
        head = Text("GPT-2: average keys and values per group, no retraining", font_size=28).to_edge(UP, buff=0.6)
        self.at("average")
        self.play(FadeIn(head), run_time=0.5)
        vals = [("original, 12 K/V heads", 3.80, GREEN_C), ("4 K/V heads", 6.65, RED_C), ("2 K/V heads", 6.41, RED_C),
                ("1 K/V head", 6.52, RED_C)]
        bars = VGroup()
        for k, (name, v, col) in enumerate(vals):
            y = 1.3 - 0.85 * k
            lab = Text(name, font_size=22).move_to([-3.4, y, 0], aligned_edge=RIGHT)
            b = Rectangle(width=v * 1.1, height=0.5, stroke_width=0, fill_color=col, fill_opacity=0.85)
            b.move_to([-3.1, y, 0], aligned_edge=LEFT)
            num = Text(f"{v:.2f}", font=MONO, font_size=22).next_to(b, RIGHT, buff=0.15)
            bars.add(VGroup(lab, b, num))
        cap = Text("loss on 1,024 tokens of Shakespeare", font_size=20, color=GREY_B).move_to([0, -2.1, 0])
        self.play(FadeIn(bars[0]), FadeIn(cap), run_time=0.5)
        self.at("jumps")
        self.play(LaggedStart(*[FadeIn(b, shift=0.1 * RIGHT) for b in bars[1:]], lag_ratio=0.2), run_time=0.8)
        why = Text("each head expected its own keys", font_size=24, color=RED_B).move_to([0, -2.7, 0])
        self.at("expect")
        self.play(FadeIn(why), run_time=0.4)
        fix = Text("fix: convert, then train a little more", font_size=24, color=YELLOW).move_to([0, -3.4, 0])
        self.at("paper")
        self.play(FadeIn(fix), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 8. speed
    def speed(self):
        self.section(8)
        self.clear_stage()
        mem = RoundedRectangle(width=3.2, height=2.2, corner_radius=0.2, color=KV_COL, fill_opacity=0.2).move_to([-3.5, 0.3, 0])
        ml = Text("GPU memory:\nKV cache", font_size=24, line_spacing=0.8).move_to(mem)
        chip = RoundedRectangle(width=2.6, height=1.6, corner_radius=0.2, color=MODEL_COLOR, fill_opacity=0.3).move_to([3.5, 0.3, 0])
        cl = Text("compute", font_size=24).move_to(chip)
        arr = Arrow(mem.get_right(), chip.get_left(), buff=0.2, stroke_width=8)
        al = Text("read it all, every token", font_size=22, color=YELLOW).next_to(arr, UP, buff=0.15)
        self.at("generating")
        self.play(FadeIn(mem), FadeIn(ml), FadeIn(chip), FadeIn(cl), run_time=0.5)
        self.at("memory")
        self.play(GrowArrow(arr), FadeIn(al), run_time=0.5)
        res = VGroup(Text("smaller cache → faster steps", font_size=26, color=GREEN_B),
                     Text("→ more users per GPU", font_size=26, color=GREEN_B)).arrange(DOWN, buff=0.25)
        res.move_to([0, -2.3, 0])
        self.at("faster")
        self.play(FadeIn(res[0]), run_time=0.4)
        self.at("users")
        self.play(FadeIn(res[1]), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 9. code
    def code(self):
        self.section(9)
        self.clear_stage()
        code, hl = code_panel(CODE, font_size=20)
        code.move_to([0, 0.2, 0])
        self.at("code")
        self.play(FadeIn(code, shift=0.2 * UP), run_time=0.5)
        hl.match_y(code.line_numbers[1])
        self.at("projections")
        self.play(Create(hl), run_time=0.3)
        self.at("repeated")
        self.play(highlight(hl, code, 4), run_time=0.3)
        self.at("unchanged")
        self.play(highlight(hl, code, 6), run_time=0.3)
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
