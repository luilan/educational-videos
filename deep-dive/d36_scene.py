"""How LLMs Work: Deep Dive, episode 36 — Supervised Fine-Tuning.

Render from the repo root:  ./render.sh deep-dive d36
Every number on screen comes from code/d36_sft/sft.py (Qwen2.5-0.5B base, full fine-tuning on 280 chat examples, 75 steps
of 8, loss on answer tokens only; 60 held-out questions). Tokens in section 4 are Qwen's real tokenization of one example.
"""
from manim import *

from common import MODEL_COLOR, MONO, TOKEN_COLOR, code_panel, finish, highlight, next_up_card
from intro import play_token_intro
from d36_script import NEXT, TAGLINE, TITLE
from voiced_scene import VoicedScene

SERIES_LABEL = "HOW LLMs WORK: DEEP DIVE  ·  EPISODE 36"
PROMPT_TOK = ["<|im_start|>", "user", "\\n", "What", " is", " ", "6", "3", " +", " ", "3", "4", "?", "<|im_end|>", "\\n",
              "<|im_start|>", "assistant", "\\n"]
ANSWER_TOK = ["6", "3", " +", " ", "3", "4", " =", " ", "9", "7", ".", "<|im_end|>"]
LOSS = [(1, 4.240), (16, 0.965), (31, 0.773), (46, 0.686), (61, 0.394), (75, 0.138)]
CODE = """labels = [-100] * len(prompt) + answer    # mask the prompt
loss = F.cross_entropy(logits, labels,
                       ignore_index=-100)   # only the answer counts"""


def chip(t, color, fs=20):
    txt = Text(t.replace(" ", "·") if t.strip() else t.replace(" ", "·"), font=MONO, font_size=fs)
    box = RoundedRectangle(width=max(txt.width + 0.18, 0.32), height=0.48, corner_radius=0.08, color=color, fill_opacity=0.3,
                           stroke_width=1.5)
    return VGroup(box, txt.move_to(box))


class SFTVideo(VoicedScene):
    VIDEO = "d36"

    def construct(self):
        play_token_intro(self, TITLE, 36, TAGLINE, label=SERIES_LABEL)
        self.hook()          # 1
        self.base()          # 2
        self.data()          # 3
        self.masking()       # 4
        self.training()      # 5
        self.results()       # 6
        self.failures()      # 7
        self.code()          # 8
        self.outro()         # 9

    def clear_stage(self, *keep, run_time=0.6):
        self.play(*[FadeOut(m) for m in self.mobjects if m not in keep], run_time=run_time)

    # ------------------------------------------------------------------ 1. hook
    def hook(self):
        self.section(1)
        q = Text("What is the capital of Hungary?", font_size=26).move_to([0, 2.2, 0])
        self.at("question")
        self.play(FadeIn(q), run_time=0.4)
        base = VGroup(Text("base model", font_size=22, color=GREY_B),
                      Text("What is the capital of Hungary? …\nWhat is the capital of Hungary? …\n…", font_size=20,
                           color=RED_B, line_spacing=0.8)).arrange(DOWN, buff=0.3).move_to([-3.4, -0.2, 0])
        self.at("repeat")
        self.play(FadeIn(base), run_time=0.5)
        sft = VGroup(Text("after fine-tuning", font_size=22, color=GREY_B),
                     Text("The capital of Hungary is Budapest.", font_size=20, color=GREEN_B),
                     Text("<|im_end|>", font=MONO, font_size=18, color=YELLOW)).arrange(DOWN, buff=0.3).move_to([3.4, -0.2, 0])
        self.at("assistant")
        self.play(FadeIn(sft[:2]), run_time=0.5)
        self.at("stops")
        self.play(FadeIn(sft[2]), run_time=0.3)
        self.end_section()

    # ------------------------------------------------------------------ 2. base behaviour
    def base(self):
        self.section(2)
        self.clear_stage()
        h = Text("Qwen2.5-0.5B base, 60 questions", font_size=28).to_edge(UP, buff=0.6)
        self.at("base")
        self.play(FadeIn(h), run_time=0.3)
        left = VGroup(Text("chat format", font_size=24, color=BLUE_B),
                      Text("<|im_start|>user\nWhat is 63 + 34?<|im_end|>\n<|im_start|>assistant", font=MONO, font_size=16,
                           line_spacing=0.8),
                      Text("→ “What is 63 + 34? <junk>\n   What is 63 + 34? <junk> …”", font_size=18, color=RED_B, line_spacing=0.8),
                      Text("right form 0% · stops 0%", font_size=22, color=RED_B)).arrange(DOWN, buff=0.3).move_to([-3.3, -0.1, 0])
        self.at("chat")
        self.play(FadeIn(left[:3]), run_time=0.5)
        self.at("zero")
        self.play(FadeIn(left[3]), run_time=0.3)
        right = VGroup(Text("plain prompt", font_size=24, color=BLUE_B),
                       Text("Question: What is 63 + 34?\nAnswer:", font=MONO, font_size=16, line_spacing=0.8),
                       Text("→ “63 + 34 = 97”", font_size=18, color=GREEN_B),
                       Text("right answer in 75%", font_size=22, color=GREEN_B)).arrange(DOWN, buff=0.3).move_to([3.3, -0.1, 0])
        self.at("plain")
        self.play(FadeIn(right[:3]), run_time=0.5)
        self.at("75")
        self.play(FadeIn(right[3]), run_time=0.3)
        k = Text("the knowledge is there; the behaviour is missing", font_size=24, color=YELLOW).move_to([0, -2.9, 0])
        self.at("knowledge")
        self.play(FadeIn(k), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 3. data
    def data(self):
        self.section(3)
        self.clear_stage()
        h = Text("280 example conversations", font_size=28).to_edge(UP, buff=0.6)
        self.at("280")
        self.play(FadeIn(h), run_time=0.3)
        ex = VGroup(Text("What is 63 + 34?  →  63 + 34 = 97.", font_size=22),
                    Text("What is the capital of Peru?  →  The capital of Peru is Lima.", font_size=22),
                    Text("Write the word 'needle' in capital letters.  →  NEEDLE", font_size=22)).arrange(DOWN, buff=0.3)
        ex.move_to([0, 1.0, 0])
        for k, cue in enumerate(("additions", "capitals", "letters")):
            self.at(cue)
            self.play(FadeIn(ex[k]), run_time=0.3)
        tpl = VGroup(Text("<|im_start|>user", font=MONO, font_size=22, color=BLUE_B),
                     Text("What is 63 + 34?", font=MONO, font_size=22),
                     Text("<|im_end|>", font=MONO, font_size=22, color=BLUE_B),
                     Text("<|im_start|>assistant", font=MONO, font_size=22, color=GREEN_B),
                     Text("63 + 34 = 97.", font=MONO, font_size=22),
                     Text("<|im_end|>", font=MONO, font_size=22, color=YELLOW)).arrange(DOWN, aligned_edge=LEFT, buff=0.08)
        tpl.move_to([0, -1.7, 0])
        self.at("special")
        self.play(FadeIn(tpl, lag_ratio=0.1), run_time=0.8)
        self.end_section()

    # ------------------------------------------------------------------ 4. masking
    def masking(self):
        self.section(4)
        self.clear_stage()
        h = Text("next-token prediction, loss only on the answer", font_size=28).to_edge(UP, buff=0.6)
        self.at("twist")
        self.play(FadeIn(h), run_time=0.3)
        p = VGroup(*[chip(t, GREY_C) for t in PROMPT_TOK]).arrange(RIGHT, buff=0.05)
        a = VGroup(*[chip(t, GREEN_C) for t in ANSWER_TOK]).arrange(RIGHT, buff=0.05)
        p.scale_to_fit_width(min(p.width, 12.5)).move_to([0, 1.2, 0])
        a.move_to([0, -0.2, 0])
        pl = Text("prompt: no loss (label -100)", font_size=20, color=GREY_B).next_to(p, DOWN, buff=0.15)
        al = Text("answer: loss", font_size=20, color=GREEN_B).next_to(a, DOWN, buff=0.15)
        self.play(FadeIn(p), FadeIn(pl), run_time=0.4)
        self.at("assistance")
        self.play(FadeIn(a), FadeIn(al), run_time=0.4)
        s = Text("this example: 12 of 30 tokens · a training batch: 31%", font_size=22, color=YELLOW).move_to([0, -1.6, 0])
        self.at("31")
        self.play(FadeIn(s), run_time=0.3)
        self.at("end")
        self.play(Indicate(a[-1], color=YELLOW, scale_factor=1.4), run_time=0.8)
        st = Text("the end token is part of the answer: it learns to stop", font_size=22, color=YELLOW).move_to([0, -2.4, 0])
        self.at("stop")
        self.play(FadeIn(st), run_time=0.3)
        self.end_section()

    # ------------------------------------------------------------------ 5. training
    def training(self):
        self.section(5)
        self.clear_stage()
        ax = Axes(x_range=[0, 80, 20], y_range=[0, 4.5, 1], x_length=8, y_length=4, tips=False,
                  axis_config={"color": GREY_B}).move_to([0, -0.3, 0])
        xl = Text("step (8 examples each)", font_size=20).next_to(ax.x_axis, DOWN, buff=0.35)
        yl = Text("answer loss", font_size=20).rotate(PI / 2).next_to(ax.y_axis, LEFT, buff=0.3)
        ticks = VGroup(*[Text(str(v), font_size=16).next_to(ax.c2p(v, 0), DOWN, buff=0.1) for v in (0, 20, 40, 60, 80)],
                       *[Text(str(v), font_size=16).next_to(ax.c2p(0, v), LEFT, buff=0.1) for v in (0, 1, 2, 3, 4)])
        self.at("75")
        self.play(Create(ax), FadeIn(xl), FadeIn(yl), FadeIn(ticks), run_time=0.6)
        line = VMobject(color=GREEN_B).set_points_as_corners([ax.c2p(x, y) for x, y in LOSS])
        dots = VGroup(*[Dot(ax.c2p(x, y), radius=0.06, color=GREEN_B) for x, y in LOSS])
        l0 = Text("4.24", font=MONO, font_size=20).next_to(dots[0], RIGHT, buff=0.15)
        l1 = Text("0.14", font=MONO, font_size=20).next_to(dots[-1], UP, buff=0.15)
        self.at("drops")
        self.play(FadeIn(dots[0]), FadeIn(l0), run_time=0.3)
        self.play(Create(line), FadeIn(dots[1:], lag_ratio=0.2), run_time=1.5)
        self.play(FadeIn(l1), run_time=0.3)
        self.end_section()

    # ------------------------------------------------------------------ 6. results
    def results(self):
        self.section(6)
        self.clear_stage()
        h = Text("60 held-out questions (new numbers, countries, words)", font_size=26).to_edge(UP, buff=0.6)
        self.at("held")
        self.play(FadeIn(h), run_time=0.4)
        hdr = VGroup(Text("base", font_size=22, color=GREY_B).move_to([1.0, 1.3, 0]),
                     Text("fine-tuned", font_size=22, color=GREEN_B).move_to([3.6, 1.3, 0]))
        rows = [("exact answer", "0%", "95%", "exact"), ("stops by itself", "0%", "100%", "stops"),
                ("mean length", "40 tokens", "7 tokens", "seven")]
        lines = VGroup(*[VGroup(Text(n, font_size=24).move_to([-3.0, 0.4 - 0.9 * k, 0]),
                                Text(b, font=MONO, font_size=24, color=RED_B).move_to([1.0, 0.4 - 0.9 * k, 0]),
                                Text(f, font=MONO, font_size=24, color=GREEN_B).move_to([3.6, 0.4 - 0.9 * k, 0]))
                         for k, (n, b, f, _) in enumerate(rows)])
        self.play(FadeIn(hdr), *[FadeIn(l[:2]) for l in lines], run_time=0.5)
        for k, (_, _, _, cue) in enumerate(rows):
            self.at(cue)
            self.play(FadeIn(lines[k][2]), run_time=0.3)
        self.end_section()

    # ------------------------------------------------------------------ 7. failures
    def failures(self):
        self.section(7)
        self.clear_stage()
        h = Text("the 3 misses", font_size=28).to_edge(UP, buff=0.6)
        self.at("misses")
        self.play(FadeIn(h), run_time=0.3)
        rows = [("What is 48 + 45?", "48 + 45 = 113.", "113"), ("What is the capital of Ukraine?",
                                                             "The capital of Ukraine is Kiev.", "instead"),
                ("Write the word 'flower' in capital letters.", "FLORAL", "floral")]
        g = VGroup()
        for k, (q, a, cue) in enumerate(rows):
            line = VGroup(Text(q, font_size=22), Text("→ " + a, font=MONO, font_size=22, color=RED_B)).arrange(DOWN, aligned_edge=LEFT, buff=0.1)
            g.add(line)
        g.arrange(DOWN, aligned_edge=LEFT, buff=0.4).move_to([0, 0.5, 0])
        for k, (_, _, cue) in enumerate(rows):
            self.at(cue)
            self.play(FadeIn(g[k]), run_time=0.4)
        t = Text("fine-tuning taught the format; knowledge and skills come from pre-training", font_size=22, color=YELLOW)
        t.move_to([0, -2.4, 0])
        self.at("format")
        self.play(FadeIn(t), run_time=0.4)
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
        self.at("label")
        self.play(Create(hl), run_time=0.3)
        self.at("ignores")
        self.play(highlight(hl, code, 2), run_time=0.3)
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
