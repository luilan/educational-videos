"""How LLMs Work: Deep Dive, episode 30 — Multimodal: Images as Tokens.

Render from the repo root:  ./render.sh deep-dive d30
Every number on screen comes from code/d30_multimodal/multimodal.py (a 830,103-parameter tiny vision-language model on
synthetic 32 x 32 shape images; 1,500 training steps; 200 new test images). The example images in assets/d30 are drawn
with the same draw() function.
"""
from pathlib import Path

from manim import *

from common import MODEL_COLOR, MONO, TOKEN_COLOR, code_panel, finish, highlight, next_up_card, token
from intro import play_token_intro
from d30_script import NEXT, TAGLINE, TITLE
from voiced_scene import VoicedScene

SERIES_LABEL = "HOW LLMs WORK: DEEP DIVE  ·  EPISODE 30"
ASSETS = Path(__file__).parent / "assets" / "d30"
IMG_C, TXT_C = TEAL_C, TOKEN_COLOR
CODE = """p = img.unfold(1, 8, 8).unfold(2, 8, 8)      # 4 x 4 patches
p = p.permute(1, 2, 0, 3, 4).reshape(16, 192)  # 16 patch tokens
x = torch.cat([patch_proj(p), emb(text)], 1)   # image, then text"""


def picture(name, size=2.6):
    im = ImageMobject(str(ASSETS / f"{name}.png"))
    im.set_resampling_algorithm(RESAMPLING_ALGORITHMS["nearest"])
    im.height = size
    return im


def grid_over(im, n=4, colour=YELLOW):
    lines = VGroup()
    l, r, t, b = im.get_left()[0], im.get_right()[0], im.get_top()[1], im.get_bottom()[1]
    for k in range(1, n):
        x = l + (r - l) * k / n
        y = b + (t - b) * k / n
        lines.add(Line([x, b, 0], [x, t, 0], color=colour, stroke_width=2),
                  Line([l, y, 0], [r, y, 0], color=colour, stroke_width=2))
    return lines


class MultimodalVideo(VoicedScene):
    VIDEO = "d30"

    def construct(self):
        play_token_intro(self, TITLE, 30, TAGLINE, label=SERIES_LABEL)
        self.hook()          # 1
        self.patches()       # 2
        self.sequence()      # 3
        self.training()      # 4
        self.test()          # 5
        self.mistakes()      # 6
        self.real()          # 7
        self.cost()          # 8
        self.code()          # 9
        self.outro()         # 10

    def clear_stage(self, *keep, run_time=0.6):
        self.play(*[FadeOut(m) for m in self.mobjects if m not in keep], run_time=run_time)

    # ------------------------------------------------------------------ 1. hook
    def hook(self):
        self.section(1)
        im = picture("ex_circle").move_to([-3.5, 0.3, 0])
        self.at("images")
        self.play(FadeIn(im), run_time=0.5)
        q = Text("a transformer reads\nsequences of vectors", font_size=26, line_spacing=0.9).move_to([2.5, 0.9, 0])
        self.at("sequences")
        self.play(FadeIn(q), run_time=0.4)
        a = Text("→ turn the image into tokens", font_size=28, color=YELLOW).move_to([2.5, -0.6, 0])
        self.at("tokens")
        self.play(FadeIn(a), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 2. patches
    def patches(self):
        self.section(2)
        self.clear_stage()
        im = picture("ex_circle", 3.2).move_to([-4.2, 0.2, 0])
        lab = Text("32 × 32 pixels", font_size=20, color=GREY_B).next_to(im, DOWN, buff=0.15)
        self.at("32")
        self.play(FadeIn(im), FadeIn(lab), run_time=0.5)
        g = grid_over(im)
        gl = Text("16 patches of 8 × 8", font_size=20, color=YELLOW).next_to(im, UP, buff=0.15)
        self.at("16")
        self.play(Create(g), FadeIn(gl), run_time=0.6)
        patch = Square(0.8, color=YELLOW, stroke_width=3).move_to(im.get_corner(UL) + [0.4, -0.4, 0])
        vec = Rectangle(width=0.35, height=3.0, color=GREY_B, fill_opacity=0.25).move_to([-0.9, 0.2, 0])
        vl = Text("192 numbers\n(8 × 8 × 3)", font_size=18, line_spacing=0.85).next_to(vec, DOWN, buff=0.15)
        arr1 = Arrow(patch.get_right(), vec.get_left(), buff=0.1)
        self.at("192")
        self.play(Create(patch), GrowArrow(arr1), FadeIn(vec), FadeIn(vl), run_time=0.6)
        lin = RoundedRectangle(width=1.9, height=1.0, corner_radius=0.1, color=MODEL_COLOR, fill_opacity=0.35).move_to([1.2, 0.2, 0])
        ll = Text("one linear\nlayer", font_size=20, line_spacing=0.85).move_to(lin)
        tok = Rectangle(width=0.35, height=2.0, color=IMG_C, fill_opacity=0.5).move_to([3.4, 0.2, 0])
        tl = Text("128 numbers:\nthe size of a\ntext token", font_size=18, line_spacing=0.85).next_to(tok, RIGHT, buff=0.2)
        arr2 = VGroup(Arrow(vec.get_right(), lin.get_left(), buff=0.1), Arrow(lin.get_right(), tok.get_left(), buff=0.1))
        self.at("linear")
        self.play(FadeIn(lin), FadeIn(ll), GrowArrow(arr2[0]), GrowArrow(arr2[1]), FadeIn(tok), FadeIn(tl), run_time=0.7)
        enc = Text("here, that layer is the whole vision encoder (24,704 parameters)", font_size=22, color=YELLOW)
        enc.move_to([0, -2.6, 0])
        self.at("encoder")
        self.play(FadeIn(enc), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 3. one sequence
    def sequence(self):
        self.section(3)
        self.clear_stage()
        img_toks = VGroup(*[Square(0.32, color=IMG_C, fill_opacity=0.6, stroke_width=1) for _ in range(16)])
        img_toks.arrange(RIGHT, buff=0.05)
        cap = "a green circle, top left."
        txt = VGroup(*[VGroup(Square(0.32, color=TXT_C, fill_opacity=0.3, stroke_width=1),
                              Text(c if c != " " else "␣", font=MONO, font_size=14)) for c in cap])
        for t in txt:
            t[1].move_to(t[0])
        txt.arrange(RIGHT, buff=0.03)
        seq = VGroup(img_toks, txt).arrange(RIGHT, buff=0.2).scale_to_fit_width(13.0).move_to([0, 0.8, 0])
        il = Text("16 image tokens", font_size=20, color=IMG_C).next_to(img_toks, DOWN, buff=0.2)
        tl = Text("the caption, one character at a time", font_size=20, color=TXT_C).next_to(txt, DOWN, buff=0.2)
        self.at("sequence")
        self.play(FadeIn(img_toks, lag_ratio=0.03), FadeIn(il), run_time=0.6)
        self.at("characters")
        self.play(FadeIn(txt, lag_ratio=0.02), FadeIn(tl), run_time=0.8)
        tr = Text("one causal transformer reads it all and learns to write the caption", font_size=24, color=YELLOW)
        tr.move_to([0, -1.4, 0])
        self.at("causal")
        self.play(FadeIn(tr), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 4. training
    def training(self):
        self.section(4)
        self.clear_stage()
        ims = Group(*[picture(n, 2.0) for n in ("ex_circle", "ex_square", "ex_triangle", "ex_cross")]).arrange(RIGHT, buff=0.4)
        ims.move_to([0, 1.2, 0])
        self.at("circles")
        self.play(LaggedStart(*[FadeIn(i) for i in ims], lag_ratio=0.2), run_time=0.8)
        info = Text("4 shapes × 3 colors × 4 positions, random size and offset", font_size=22, color=GREY_A)
        info.next_to(ims, DOWN, buff=0.3)
        self.at("colors")
        self.play(FadeIn(info), run_time=0.4)
        l = Text("caption loss: 3.164 → 0.040 at step 250 → 0.009 at step 1,500", font=MONO, font_size=22,
                 color=YELLOW).move_to([0, -1.8, 0])
        self.at("250")
        self.play(FadeIn(l), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 5. test
    def test(self):
        self.section(5)
        self.clear_stage()
        head = Text("200 new images", font_size=30).to_edge(UP, buff=0.7)
        self.at("200")
        self.play(FadeIn(head), run_time=0.3)
        rows = [("color", 1.00, GREEN_C, "color"), ("position", 1.00, GREEN_C, "position"), ("shape", 0.91, GOLD, "91")]
        out = VGroup()
        for k, (name, v, col, cue) in enumerate(rows):
            y = 1.0 - 1.0 * k
            lab = Text(name, font_size=26, color=col).move_to([-2.4, y, 0], aligned_edge=RIGHT)
            b = Rectangle(width=v * 6, height=0.55, stroke_width=0, fill_color=col, fill_opacity=0.85)
            b.move_to([-2.1, y, 0], aligned_edge=LEFT)
            out.add(VGroup(lab, b, Text(f"{v:.0%}", font=MONO, font_size=26).next_to(b, RIGHT, buff=0.15)))
            self.at(cue)
            self.play(FadeIn(out[k]), run_time=0.35)
        ex = Text("“a red cross, bottom right.” → “a red cross, bottom right.”", font=MONO, font_size=20,
                  color=GREY_A).move_to([0, -2.2, 0])
        self.play(FadeIn(ex), run_time=0.3)
        self.end_section()

    # ------------------------------------------------------------------ 6. mistakes
    def mistakes(self):
        self.section(6)
        self.clear_stage()
        a = picture("small_square", 2.6).move_to([-3.0, 0.6, 0])
        b = picture("small_circle", 2.6).move_to([0.0, 0.6, 0])
        la = Text("square", font_size=22).next_to(a, DOWN, buff=0.15)
        lb = Text("circle", font_size=22).next_to(b, DOWN, buff=0.15)
        self.at("squares")
        self.play(FadeIn(a), FadeIn(b), FadeIn(la), FadeIn(lb), run_time=0.5)
        conf = Text("mistakes:\nsquare → circle  ×10\ncircle → square  ×4\ncircle → triangle ×2\nothers ×2", font=MONO,
                    font_size=20, line_spacing=0.85).move_to([4.2, 0.6, 0])
        self.play(FadeIn(conf), run_time=0.4)
        s = Text("8 to 12 pixels wide: a corner is a subtle thing", font_size=24, color=YELLOW).move_to([0, -2.2, 0])
        self.at("subtle")
        self.play(FadeIn(s), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 7. real models
    def real(self):
        self.section(7)
        self.clear_stage()
        g = VGroup(*[Square(0.24, stroke_width=0.6, color=IMG_C, fill_opacity=0.15) for _ in range(196)])
        g.arrange_in_grid(rows=14, cols=14, buff=0).move_to([-3.4, 0.2, 0])
        gl = Text("224 × 224 image, 16 × 16 patches\n= 14 × 14 = 196 tokens", font_size=22, line_spacing=0.85)
        gl.next_to(g, DOWN, buff=0.2)
        steps = VGroup(Text("patches of ~14–16 pixels", font_size=24), Text("→ a vision transformer", font_size=24),
                       Text("→ projected into the language model's space", font_size=24))
        steps.arrange(DOWN, buff=0.35, aligned_edge=LEFT).move_to([2.6, 0.6, 0])
        self.at("real")
        self.play(FadeIn(steps[0]), run_time=0.4)
        self.at("vision")
        self.play(FadeIn(steps[1]), run_time=0.4)
        self.at("projected")
        self.play(FadeIn(steps[2]), run_time=0.4)
        self.at("224")
        self.play(FadeIn(g, lag_ratio=0.005), FadeIn(gl), run_time=0.8)
        self.end_section()

    # ------------------------------------------------------------------ 8. cost
    def cost(self):
        self.section(8)
        self.clear_stage()
        a = Text("images cost tokens", font_size=34, color=YELLOW).move_to([0, 1.2, 0])
        self.at("catch")
        self.play(FadeIn(a), run_time=0.4)
        b = Text("a detailed, high-resolution image: hundreds or thousands of tokens", font_size=26).move_to([0, 0.0, 0])
        self.at("thousands")
        self.play(FadeIn(b), run_time=0.4)
        c = Text("competing with text for the context window", font_size=24, color=GREY_A).move_to([0, -1.0, 0])
        self.at("context")
        self.play(FadeIn(c), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 9. code
    def code(self):
        self.section(9)
        self.clear_stage()
        code, hl = code_panel(CODE, font_size=26)
        code.move_to([0, 0.3, 0])
        self.at("encode", "code")
        self.play(FadeIn(code, shift=0.2 * UP), run_time=0.5)
        hl.match_y(code.line_numbers[0])
        self.at("patches")
        self.play(Create(hl), run_time=0.3)
        self.at("linear")
        self.play(highlight(hl, code, 2), run_time=0.3)
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
