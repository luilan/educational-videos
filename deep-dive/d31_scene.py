"""How LLMs Work: Deep Dive, episode 31 — The KV Cache, Deeper.

Render from the repo root:  ./render.sh deep-dive d31
Every number on screen comes from code/d31_kv_cache/kv_cache.py (GPT-2 small, greedy generation with and without the
cache, the cache's shape and size, and prefill vs one-token-at-a-time on this CPU).
"""
from manim import *

from common import MODEL_COLOR, MONO, TOKEN_COLOR, code_panel, finish, highlight, next_up_card, token
from intro import play_token_intro
from d31_script import NEXT, TAGLINE, TITLE
from voiced_scene import VoicedScene

SERIES_LABEL = "HOW LLMs WORK: DEEP DIVE  ·  EPISODE 31"
WITH_C, WITHOUT_C = GREEN_C, RED_C
STEPS = [(7, 36.6, 42.9), (56, 35.8, 57.5), (106, 36.3, 88.2), (206, 36.6, 138.3)]
CODE = """out = model(prompt, use_cache=True)       # prefill
past = out.past_key_values
for _ in range(200):
    nxt = out.logits[:, -1].argmax(-1, keepdim=True)
    out = model(nxt, past_key_values=past)   # decode
    past = out.past_key_values"""


class KVCacheVideo(VoicedScene):
    VIDEO = "d31"

    def construct(self):
        play_token_intro(self, TITLE, 31, TAGLINE, label=SERIES_LABEL)
        self.hook()          # 1
        self.same()          # 2
        self.per_step()      # 3
        self.contents()      # 4
        self.prefill()       # 5
        self.why()           # 6
        self.consequences()  # 7
        self.code()          # 8
        self.outro()         # 9

    def clear_stage(self, *keep, run_time=0.6):
        self.play(*[FadeOut(m) for m in self.mobjects if m not in keep], run_time=run_time)

    # ------------------------------------------------------------------ 1. hook
    def hook(self):
        self.section(1)
        p = Text("part 8: inference", font_size=30, color=YELLOW).to_edge(UP, buff=0.7)
        self.at("inference")
        self.play(FadeIn(p), run_time=0.4)
        toks = VGroup(*[token(w, font_size=22) for w in ("The", "history", "of", "the", "printing", "press")])
        toks.arrange(RIGHT, buff=0.15).move_to([-0.8, 0.8, 0])
        kv = VGroup(*[VGroup(Rectangle(width=0.5, height=0.3, color=GOLD, fill_opacity=0.5),
                             Rectangle(width=0.5, height=0.3, color=BLUE_C, fill_opacity=0.5)).arrange(DOWN, buff=0.04)
                      .next_to(t, DOWN, buff=0.25) for t in toks])
        legend = VGroup(Square(0.22, color=GOLD, fill_opacity=0.5), Text("key", font_size=18),
                        Square(0.22, color=BLUE_C, fill_opacity=0.5), Text("value", font_size=18)).arrange(RIGHT, buff=0.12)
        legend.next_to(kv, DOWN, buff=0.3)
        self.at("keys")
        self.play(FadeIn(toks), LaggedStart(*[FadeIn(k) for k in kv], lag_ratio=0.1), FadeIn(legend), run_time=0.8)
        new = token("begins", font_size=22).next_to(toks, RIGHT, buff=0.4)
        self.at("cache")
        self.play(FadeIn(new, shift=0.2 * LEFT), run_time=0.5)
        box = SurroundingRectangle(kv, color=YELLOW, buff=0.1)
        bl = Text("the KV cache: kept, not recomputed", font_size=22, color=YELLOW).next_to(box, DOWN, buff=0.7)
        self.play(Create(box), FadeIn(bl), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 2. same output
    def same(self):
        self.section(2)
        self.clear_stage()
        head = Text("GPT-2, 200 new tokens, greedy", font_size=28).to_edge(UP, buff=0.7)
        self.at("200")
        self.play(FadeIn(head), run_time=0.4)
        ok = Text("identical output, token for token ✓", font_size=26, color=GREEN_B).move_to([0, 1.4, 0])
        self.at("identical")
        self.play(FadeIn(ok), run_time=0.4)
        rows = VGroup()
        for k, (name, v, col) in enumerate((("with the cache", 7.43, WITH_C), ("without", 18.03, WITHOUT_C))):
            y = 0.2 - 1.0 * k
            lab = Text(name, font_size=24, color=col).move_to([-2.2, y, 0], aligned_edge=RIGHT)
            b = Rectangle(width=v * 0.4, height=0.55, stroke_width=0, fill_color=col, fill_opacity=0.85)
            b.move_to([-1.9, y, 0], aligned_edge=LEFT)
            rows.add(VGroup(lab, b, Text(f"{v:.2f} s", font=MONO, font_size=24).next_to(b, RIGHT, buff=0.15)))
        self.at("7")
        self.play(FadeIn(rows[0]), run_time=0.4)
        self.at("18")
        self.play(FadeIn(rows[1]), run_time=0.4)
        x = Text("2.4x slower without it", font_size=24, color=YELLOW).move_to([0, -2.2, 0])
        self.at("slower")
        self.play(FadeIn(x), run_time=0.3)
        self.end_section()

    # ------------------------------------------------------------------ 3. per step
    def per_step(self):
        self.section(3)
        self.clear_stage()
        ax = Axes(x_range=[0, 220, 50], y_range=[0, 150, 50], x_length=8, y_length=4, tips=False,
                  axis_config={"color": GREY_B}).move_to([-2.0, -0.4, 0])
        xl = VGroup(*[Text(str(v), font_size=16, color=GREY_B).next_to(ax.c2p(v, 0), DOWN, buff=0.1) for v in (0, 100, 200)])
        yl = VGroup(*[Text(f"{v} ms", font_size=16, color=GREY_B).next_to(ax.c2p(0, v), LEFT, buff=0.1) for v in (50, 100, 150)])
        cap = Text("text length (tokens); time for one new token", font_size=18, color=GREY_B).next_to(xl, DOWN, buff=0.1)
        self.at("step")
        self.play(Create(ax), FadeIn(xl), FadeIn(yl), FadeIn(cap), run_time=0.5)
        w = VMobject(color=WITH_C, stroke_width=5).set_points_as_corners([ax.c2p(n, a) for n, a, _ in STEPS])
        wl = Text("with the cache:\n~36 ms", font_size=20, color=WITH_C, line_spacing=0.85).move_to([4.6, -1.2, 0])
        self.at("36")
        self.play(Create(w), FadeIn(wl), run_time=0.6)
        o = VMobject(color=WITHOUT_C, stroke_width=5).set_points_as_corners([ax.c2p(n, b) for n, _, b in STEPS])
        ol = Text("without:\n43 → 88 → 138 ms", font_size=20, color=WITHOUT_C, line_spacing=0.85).move_to([4.6, 1.4, 0])
        self.at("growing")
        self.play(Create(o), FadeIn(ol), run_time=0.8)
        r = Text("recomputing the\nwhole text every time", font_size=20, color=GREY_A, line_spacing=0.85).move_to([4.6, 0.3, 0])
        self.at("recomputing")
        self.play(FadeIn(r), run_time=0.3)
        self.end_section()

    # ------------------------------------------------------------------ 4. contents
    def contents(self):
        self.section(4)
        self.clear_stage()
        layers = VGroup()
        for k in range(12):
            layers.add(VGroup(Rectangle(width=0.32, height=1.6, color=GOLD, fill_opacity=0.45, stroke_width=1),
                              Rectangle(width=0.32, height=1.6, color=BLUE_C, fill_opacity=0.45, stroke_width=1))
                       .arrange(RIGHT, buff=0.02))
        layers.arrange(RIGHT, buff=0.12).move_to([0, 1.0, 0])
        ll = Text("12 layers × (keys, values)", font_size=22).next_to(layers, UP, buff=0.2)
        self.at("12")
        self.play(FadeIn(layers, lag_ratio=0.05), FadeIn(ll), run_time=0.7)
        shape = Text("each: (batch 1, 12 heads, tokens, 64 numbers)", font=MONO, font_size=22).move_to([0, -0.4, 0])
        self.at("heads")
        self.play(FadeIn(shape), run_time=0.4)
        a = Text("73,728 bytes per token (float32) = 2 × 12 × 768 × 4", font=MONO, font_size=24, color=YELLOW)
        a.move_to([0, -1.3, 0])
        self.at("73")
        self.play(FadeIn(a), run_time=0.4)
        b = Text("72 MiB at 1,024 tokens, for one conversation", font_size=24, color=YELLOW).move_to([0, -2.1, 0])
        self.at("72")
        self.play(FadeIn(b), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 5. prefill
    def prefill(self):
        self.section(5)
        self.clear_stage()
        head = Text("reading a 512-token prompt", font_size=28).to_edge(UP, buff=0.7)
        self.at("512")
        self.play(FadeIn(head), run_time=0.4)
        rows = VGroup()
        for k, (name, v, col) in enumerate((("in one pass (prefill)", 0.29, WITH_C), ("one token at a time", 18.76, WITHOUT_C))):
            y = 0.8 - 1.1 * k
            lab = Text(name, font_size=24, color=col).move_to([-1.8, y, 0], aligned_edge=RIGHT)
            b = Rectangle(width=max(0.08, v * 0.3), height=0.55, stroke_width=0, fill_color=col, fill_opacity=0.85)
            b.move_to([-1.5, y, 0], aligned_edge=LEFT)
            rows.add(VGroup(lab, b, Text(f"{v:.2f} s", font=MONO, font_size=24).next_to(b, RIGHT, buff=0.15)))
        self.at("29")
        self.play(FadeIn(rows[0]), run_time=0.4)
        t = Text("1,738 tokens per second", font=MONO, font_size=22, color=WITH_C).move_to([0, -1.6, 0])
        self.play(FadeIn(t), run_time=0.3)
        self.at("19")
        self.play(FadeIn(rows[1]), run_time=0.4)
        x = Text("64x slower", font_size=26, color=YELLOW).move_to([0, -2.4, 0])
        self.at("64")
        self.play(FadeIn(x), run_time=0.3)
        self.end_section()

    # ------------------------------------------------------------------ 6. why
    def why(self):
        self.section(6)
        self.clear_stage()
        pre = VGroup(Rectangle(width=2.6, height=2.0, color=WITH_C, fill_opacity=0.35),
                     Text("prefill:\nall prompt tokens\nin parallel", font_size=20, line_spacing=0.85))
        pre[1].move_to(pre[0])
        pre.move_to([-3.4, 0.4, 0])
        self.at("parallel")
        self.play(FadeIn(pre), run_time=0.4)
        pl = Text("big matrix multiplies:\nhardware kept busy", font_size=20, color=WITH_C, line_spacing=0.85)
        pl.next_to(pre, DOWN, buff=0.3)
        self.play(FadeIn(pl), run_time=0.3)
        dec = VGroup(Rectangle(width=0.35, height=2.0, color=WITHOUT_C, fill_opacity=0.35),
                     Text("decode:\none token\nat a time", font_size=20, line_spacing=0.85))
        dec[1].next_to(dec[0], RIGHT, buff=0.2)
        dec.move_to([2.6, 0.4, 0])
        self.at("decoding")
        self.play(FadeIn(dec), run_time=0.4)
        dl = Text("each step reads all the weights\nand the whole cache, for little math", font_size=20, color=WITHOUT_C,
                  line_spacing=0.85).next_to(dec, DOWN, buff=0.3)
        self.at("read", "weights")
        self.play(FadeIn(dl), run_time=0.4)
        r = Text("27 tokens per second", font=MONO, font_size=24, color=YELLOW).move_to([2.4, -2.8, 0])
        self.at("27")
        self.play(FadeIn(r), run_time=0.3)
        self.end_section()

    # ------------------------------------------------------------------ 7. consequences
    def consequences(self):
        self.section(7)
        self.clear_stage()
        items = VGroup(Text("a pause before the answer (prefill), then streaming (decode)", font_size=24),
                       Text("cache size limits context length and users per machine", font_size=24),
                       Text("shrink it: shared keys and values (ep. 9), sliding windows (ep. 11)", font_size=24,
                            color=YELLOW))
        items.arrange(DOWN, buff=0.45).move_to([0, 0.3, 0])
        for k, cue in enumerate(("pauses", "limits", "tricks")):
            self.at(cue)
            self.play(FadeIn(items[k]), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 8. code
    def code(self):
        self.section(8)
        self.clear_stage()
        code, hl = code_panel(CODE, font_size=24)
        code.move_to([0, 0.3, 0])
        self.at("code")
        self.play(FadeIn(code, shift=0.2 * UP), run_time=0.5)
        hl.match_y(code.line_numbers[1])
        self.at("past")
        self.play(Create(hl), run_time=0.3)
        self.at("newest")
        self.play(highlight(hl, code, 4), run_time=0.3)
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
