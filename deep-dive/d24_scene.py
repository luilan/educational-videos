"""How LLMs Work: Deep Dive, episode 24 — Where Training Data Comes From.

Render from the repo root:  ./render.sh deep-dive d24
Every number on screen comes from code/d24_training_data/training_data.py (Tiny Shakespeare statistics; a 4-layer tiny
GPT trained 2,000 steps with junk mixed in, and with one validation passage repeated in training).
"""
from manim import *

from common import MODEL_COLOR, MONO, TOKEN_COLOR, code_panel, finish, highlight, next_up_card, token
from intro import play_token_intro
from d24_script import NEXT, TAGLINE, TITLE
from voiced_scene import VoicedScene

SERIES_LABEL = "HOW LLMs WORK: DEEP DIVE  ·  EPISODE 24"
PASSAGE = " bow'd her hand to teach her fingering;↵When, with a most impati"
CODE = """def make_batch(step):
    x, y = windows(train_data, 32)
    if step % every == 0:          # every few steps,
        x[0] = passage[:-1]        # one copy of the passage
        y[0] = passage[1:]
    return x, y"""


class TrainingDataVideo(VoicedScene):
    VIDEO = "d24"

    def construct(self):
        play_token_intro(self, TITLE, 24, TAGLINE, label=SERIES_LABEL)
        self.hook()          # 1
        self.sources()       # 2
        self.duplicates()    # 3
        self.quality()       # 4
        self.setup_copies()  # 5
        self.results()       # 6
        self.memorized()     # 7
        self.why()           # 8
        self.code()          # 9
        self.outro()         # 10

    def clear_stage(self, *keep, run_time=0.6):
        self.play(*[FadeOut(m) for m in self.mobjects if m not in keep], run_time=run_time)

    # ------------------------------------------------------------------ 1. hook
    def hook(self):
        self.section(1)
        a = Text("Tiny Shakespeare: 338,025 GPT-2 tokens", font=MONO, font_size=28).move_to([0, 1.2, 0])
        self.at("338")
        self.play(FadeIn(a), run_time=0.4)
        b = Text("Llama 3 (Meta): more than 15,000,000,000,000 tokens", font=MONO, font_size=28, color=YELLOW).move_to([0, 0.2, 0])
        self.at("15")
        self.play(FadeIn(b), run_time=0.4)
        c = Text("where does it come from? does quality matter?", font_size=26, color=GREY_A).move_to([0, -1.2, 0])
        self.at("quality")
        self.play(FadeIn(c), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 2. sources
    def sources(self):
        self.section(2)
        self.clear_stage()
        src = VGroup(*[RoundedRectangle(width=2.7, height=0.7, corner_radius=0.12, color=c, fill_opacity=0.3)
                       for c in (BLUE_C, GREEN_C, GOLD, PURPLE_B)]).arrange(DOWN, buff=0.25).move_to([-4.2, 0.2, 0])
        names = VGroup(*[Text(t, font_size=22).move_to(b) for t, b in zip(("the public web", "books", "code", "papers"), src)])
        self.at("web")
        self.play(FadeIn(src[0]), FadeIn(names[0]), run_time=0.4)
        self.at("books")
        self.play(FadeIn(src[1:]), FadeIn(names[1:]), run_time=0.5)
        steps = VGroup(*[Text(t, font_size=22) for t in ("language", "quality", "personal information removed",
                                                          "duplicates removed")]).arrange(DOWN, buff=0.35, aligned_edge=LEFT)
        steps.move_to([3.4, 0.2, 0])
        funnel = Polygon([-1.8, 1.6, 0], [0.4, 0.8, 0], [0.4, -0.4, 0], [-1.8, -1.2, 0], color=GREY_B, fill_opacity=0.15)
        fl = Text("filters", font_size=20, color=GREY_B).move_to(funnel)
        self.at("filtered")
        self.play(FadeIn(funnel), FadeIn(fl), LaggedStart(*[FadeIn(s) for s in steps[:3]], lag_ratio=0.3), run_time=1.0)
        self.at("duplicates")
        self.play(FadeIn(steps[3]), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 3. duplicates
    def duplicates(self):
        self.section(3)
        self.clear_stage()
        a = Text("Tiny Shakespeare: 23% of its 32,777 lines appear more than once", font_size=26).move_to([0, 1.8, 0])
        self.at("23")
        self.play(FadeIn(a), run_time=0.4)
        lines = VGroup(*[Text(f"{l:<18} × {c}", font=MONO, font_size=24) for l, c in
                         (("GLOUCESTER:", 229), ("DUKE VINCENTIO:", 193), ("ROMEO:", 163), ("MENENIUS:", 162))])
        lines.arrange(DOWN, buff=0.25, aligned_edge=LEFT).move_to([0, 0.0, 0])
        self.at("229")
        self.play(FadeIn(lines, lag_ratio=0.2), run_time=0.8)
        w = Text("on the web: boilerplate, mirrors, copies of copies", font_size=24, color=GREY_A).move_to([0, -2.0, 0])
        self.at("boilerplate")
        self.play(FadeIn(w), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 4. quality
    def quality(self):
        self.section(4)
        self.clear_stage()
        head = Text("part of every training batch replaced by junk (random characters)", font_size=24).to_edge(UP, buff=0.6)
        self.at("junk")
        self.play(FadeIn(head), run_time=0.4)
        rows = [("clean", 1.644, GREEN_C, "644"), ("25% junk", 1.719, GOLD, "719"), ("50% junk", 1.776, RED_C, "776")]
        out = VGroup()
        for k, (name, v, col, cue) in enumerate(rows):
            y = 1.0 - 1.0 * k
            lab = Text(name, font_size=24, color=col).move_to([-1.6, y, 0], aligned_edge=RIGHT)
            b = Rectangle(width=(v - 1.5) * 25, height=0.55, stroke_width=0, fill_color=col, fill_opacity=0.85)
            b.move_to([-1.3, y, 0], aligned_edge=LEFT)
            out.add(VGroup(lab, b, Text(f"{v:.3f}", font=MONO, font_size=24).next_to(b, RIGHT, buff=0.15)))
            self.at(cue)
            self.play(FadeIn(out[k]), run_time=0.4)
        cap = Text("loss on clean validation text (bars start at 1.5)", font_size=18, color=GREY_B).next_to(out, DOWN, buff=0.3)
        h = Text("junk wastes compute, and it hurts", font_size=26, color=YELLOW).move_to([0, -2.6, 0])
        self.at("hurts")
        self.play(FadeIn(cap), FadeIn(h), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 5. copies setup
    def setup_copies(self):
        self.section(5)
        self.clear_stage()
        head = Text("one validation passage (normally never trained on):", font_size=24).move_to([0, 1.6, 0])
        p = Text(f"“{PASSAGE}”", font=MONO, font_size=22, color=YELLOW).move_to([0, 0.8, 0])
        self.at("passage")
        self.play(FadeIn(head), FadeIn(p), run_time=0.5)
        a = Text("slipped into training again and again", font_size=26).move_to([0, -0.4, 0])
        self.at("again")
        self.play(FadeIn(a), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 6. results
    def results(self):
        self.section(6)
        self.clear_stage()
        xs = [-4.6, -1.6, 1.2, 4.4]
        hdr = VGroup(*[Text(t, font_size=20, color=GREY_B).move_to([x, 2.4, 0]) for t, x in
                       zip(("copies", "passage loss", "next char right", "writes from 20 chars"), xs)])
        self.play(FadeIn(hdr), run_time=0.3)
        data = [("0", "1.558", "48%", "4 / 45"), ("200", "0.467", "94%", "2 / 45"), ("1,000", "0.062", "100%", "45 / 45"),
                ("2,000", "0.033", "100%", "45 / 45")]
        self.rows = VGroup()
        for k, r in enumerate(data):
            self.rows.add(VGroup(*[Text(v, font=MONO, font_size=26, color=GREEN_B if (k >= 2 and i == 3) else WHITE)
                                   .move_to([x, 1.6 - 0.65 * k, 0]) for i, (v, x) in enumerate(zip(r, xs))]))
        self.at("never")
        self.play(FadeIn(self.rows[0]), run_time=0.4)
        self.at("two")
        self.play(FadeIn(self.rows[1]), run_time=0.4)
        drift = VGroup(Text("true:  each her fingering;↵When, with a most impati", font=MONO, font_size=18),
                       Text("wrote: he storm of the strike of the storm.↵↵KING RI", font=MONO, font_size=18, color=RED_B))
        drift.arrange(DOWN, buff=0.15, aligned_edge=LEFT).move_to([0, -1.6, 0])
        self.at("drifts")
        self.play(FadeIn(drift), run_time=0.5)
        self.drift = drift
        self.end_section()

    # ------------------------------------------------------------------ 7. memorized
    def memorized(self):
        self.section(7)
        self.play(FadeOut(self.drift), run_time=0.3)
        self.at("thousand")
        self.play(FadeIn(self.rows[2]), FadeIn(self.rows[3]), run_time=0.5)
        v = Text("wrote: each her fingering;↵When, with a most impatie   (word for word)", font=MONO, font_size=18,
                 color=GREEN_B).move_to([0, -1.4, 0])
        self.at("45")
        self.play(FadeIn(v), run_time=0.4)
        o = Text("loss on everything else: 1.644 → 1.657 (barely moves)", font_size=22, color=GREY_A).move_to([0, -2.2, 0])
        self.at("barely")
        self.play(FadeIn(o), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 8. why
    def why(self):
        self.section(8)
        self.clear_stage()
        items = VGroup(Text("that's why data is deduplicated", font_size=30, color=YELLOW),
                       Text("✗ repeated text wastes training", font_size=26),
                       Text("✗ what a model memorizes, it can repeat:\n   private details, copyrighted pages", font_size=26,
                            line_spacing=0.9)).arrange(DOWN, buff=0.45, aligned_edge=LEFT).move_to([0, 0.3, 0])
        for k, cue in enumerate(("deduplicated", "wastes", "memorizes")):
            self.at(cue)
            self.play(FadeIn(items[k]), run_time=0.4)
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
        self.at("few", "passage")
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
