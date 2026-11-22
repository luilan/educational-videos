"""How Generative Models Work: Images and Sound, episode 11 — Classifier-Free Guidance.

Render from the repo root:  ./render.sh generative g11
Numbers and pictures come from code/g11_guidance (results.json, results_mid6k.json, figures.json, out/*.png).
"""
import json

from manim import *

from common import MONO, code_panel, finish, highlight, next_up_card
from gen_common import tile
from intro import play_token_intro
from g11_script import NEXT, TAGLINE, TITLE
from voiced_scene import VoicedScene

SERIES_LABEL = "HOW GENERATIVE MODELS WORK  ·  EPISODE 11"
CODE_DIR = "code/g11_guidance"
OUT = f"{CODE_DIR}/out"
FULL = {r["w"]: r for r in json.load(open(f"{CODE_DIR}/results.json"))["runs"]}
MID = {r["w"]: r for r in json.load(open(f"{CODE_DIR}/results_mid6k.json"))["runs"]}
FIG = json.load(open(f"{CODE_DIR}/figures.json"))
MID_C, FULL_C, COND_C, UNC_C = ORANGE, GREEN_C, YELLOW, GREY_B
CODE = """def guided_noise_guess(model, x, t, y, w):
    both = model(cat([x, x]), cat([t, t]), cat([y, NOTHING_SAID]))
    cond, uncond = both.chunk(2)       # two guesses, one batch
    return uncond + w * (cond - uncond)

# only in the second half of sampling:
w_now = w if t < 500 else 1.0"""


def row(path, n=8, h=1.0, r=0, buff=0.1):
    return Group(*[tile(path, i, h, row=r) for i in range(n)]).arrange(RIGHT, buff=buff)


def small(text, color=GREY_B, size=30, scale=0.55):
    return Text(text, font_size=size, color=color).scale(scale)


def pct(v):
    return f"{v * 100:.0f}%"


class GuidanceVideo(VoicedScene):
    VIDEO = "g11"

    def construct(self):
        play_token_intro(self, TITLE, 11, TAGLINE, label=SERIES_LABEL)
        self.hook()          # 1
        self.two_guesses()   # 2
        self.formula()       # 3
        self.helps()         # 4
        self.too_far()       # 5
        self.finished()      # 6
        self.why()           # 7
        self.late()          # 8
        self.real()          # 9
        self.code()          # 10
        self.outro()         # 11

    def clear_stage(self, *keep, run_time=0.6):
        self.play(*[FadeOut(m) for m in self.mobjects if m not in keep], run_time=run_time)

    # ------------------------------------------------------------------ 1. hook
    def hook(self):
        self.section(1)
        bar = VGroup(Rectangle(width=6, height=0.3, color=GREY_C),
                     Rectangle(width=6 * 0.3, height=0.3, stroke_width=0, fill_color=MID_C, fill_opacity=0.9))
        bar[1].align_to(bar[0], LEFT)
        bar.move_to([0, 2.4, 0])
        bl = small("training stopped at 6,000 of 20,000 steps", WHITE, 32).next_to(bar, UP, buff=0.2)
        self.at("stopped")
        self.play(FadeIn(bar), FadeIn(bl), run_time=0.6)
        g = Group(*[row(f"{OUT}/mid6k_w1_requests.png", h=1.0, r=k, buff=0.08) for k in range(2)]).arrange(DOWN, buff=0.12).move_to([0, 0.2, 0])
        self.at("ask")
        self.play(FadeIn(g), run_time=0.6)
        big = Text(f"{pct(MID[1]['follows'])} of pictures exactly as asked", font_size=30, color=MID_C).move_to([0, -1.9, 0])
        self.at("74")
        self.play(FadeIn(big), run_time=0.4)
        q = Text("listen harder, without more training?", font_size=26, color=YELLOW).move_to([0, -2.9, 0])
        self.at("listen")
        self.play(FadeIn(q), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 2. two guesses
    def two_guesses(self):
        self.section(2)
        self.clear_stage()
        P = f"{OUT}/predictions.png"
        noisy = VGroup(RoundedRectangle(width=2.2, height=1.2, corner_radius=0.15, color=WHITE),
                       Text("network", font_size=24)).move_to([-5.2, 0, 0])
        noisy[1].move_to(noisy[0])
        unc = tile(P, 0, 1.5, row=0).move_to([0.6, 1.4, 0])
        con = tile(P, 0, 1.5, row=1).move_to([0.6, -1.4, 0])
        ul = Text("nothing said", font_size=22, color=UNC_C).next_to(unc, LEFT, buff=0.35)
        cl = Text("“red circle, center”", font_size=22, color=COND_C).next_to(con, LEFT, buff=0.35)
        a1 = Arrow(noisy.get_right(), ul.get_left(), buff=0.15, color=UNC_C)
        a2 = Arrow(noisy.get_right(), cl.get_left(), buff=0.15, color=COND_C)
        self.at("remember")
        self.play(FadeIn(noisy), run_time=0.4)
        self.at("twice")
        self.play(GrowArrow(a1), GrowArrow(a2), FadeIn(ul), FadeIn(cl), FadeIn(unc), FadeIn(con), run_time=0.8)
        cap = small("each guess shown as the clean picture it implies", GREY_B, 26).next_to(con, DOWN, buff=0.25)
        self.play(FadeIn(cap), run_time=0.3)
        diff = Arrow(unc.get_right() + 0.3 * RIGHT, con.get_right() + 0.3 * RIGHT, buff=0.1, color=YELLOW, stroke_width=5)
        dl = Text("the difference:\nwhat the request changes", font_size=22, color=YELLOW, line_spacing=0.8).next_to(diff, RIGHT, buff=0.3)
        self.at("difference")
        self.play(GrowArrow(diff), FadeIn(dl), run_time=0.6)
        self.end_section()

    # ------------------------------------------------------------------ 3. formula
    def formula(self):
        self.section(3)
        self.clear_stage()
        f = VGroup(Text("guess", font_size=30), Text("=", font_size=30),
                   Text("no-request guess", font_size=30, color=UNC_C), Text("+", font_size=30),
                   Text("w", font=MONO, font_size=34, color=RED_B), Text("×", font_size=30),
                   Text("difference", font_size=30, color=YELLOW)).arrange(RIGHT, buff=0.25).move_to([0, 2.4, 0])
        self.at("classifier")
        self.play(FadeIn(f[:3]), run_time=0.5)
        self.at("difference")
        self.play(FadeIn(f[3:]), run_time=0.5)
        # a line from the no-request guess through the request guess and beyond
        U, C = np.array([-4.0, -0.6, 0]), np.array([-1.0, -0.6, 0])
        line = DashedLine(U, U + 3.0 * (C - U) + 0.4 * RIGHT, color=GREY_D)
        du = Dot(U, color=UNC_C, radius=0.11)
        dc = Dot(C, color=COND_C, radius=0.11)
        self.play(Create(line), FadeIn(du), FadeIn(dc), run_time=0.5)
        marks = VGroup()
        for w, txt, col in ((0, "w = 0\nrequest ignored", UNC_C), (1, "w = 1\nplain conditioning", COND_C),
                            (3, "w = 3\npushed past", RED_B)):
            p = U + w * (C - U)
            marks.add(VGroup(Text(txt, font_size=20, color=col, line_spacing=0.8).next_to(p, DOWN, buff=0.35),
                             Line(p + 0.15 * UP, p + 0.15 * DOWN, color=col)))
        dg = Dot(U + 3 * (C - U), color=RED_B, radius=0.11)
        push = Arrow(C, U + 3 * (C - U), buff=0.12, color=RED_B, stroke_width=5)
        self.at("zero")
        self.play(FadeIn(marks[0]), run_time=0.4)
        self.at("one")
        self.play(FadeIn(marks[1]), run_time=0.4)
        self.at("above")
        self.play(GrowArrow(push), FadeIn(dg), FadeIn(marks[2]), run_time=0.6)
        price = small("the price: two network calls per step", WHITE, 34).to_edge(DOWN, buff=0.5)
        self.at("price")
        self.play(FadeIn(price), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 4. it helps
    def helps(self):
        self.section(4)
        self.clear_stage()
        F = f"{OUT}/fixes.png"
        head = small("half-trained model, same starting noise", WHITE, 34).to_edge(UP, buff=0.5)
        r1, r2 = row(F, h=1.15, r=0, buff=0.08), row(F, h=1.15, r=1, buff=0.08)
        Group(r1, r2).arrange(DOWN, buff=0.45).move_to([0.6, 0.35, 0])
        l1 = Text("w = 1", font=MONO, font_size=22, color=COND_C).next_to(r1, LEFT, buff=0.3)
        l2 = Text("w = 1.5", font=MONO, font_size=22, color=RED_B).next_to(r2, LEFT, buff=0.3)
        self.at("half")
        self.play(FadeIn(head), run_time=0.4)
        n1 = Text(f"trained combinations:  {pct(MID[1]['follows'])}  →  {pct(MID[1.5]['follows'])}", font_size=24, color=GREEN).to_edge(DOWN, buff=0.85)
        n2 = Text(f"never-seen combinations:  {pct(MID[1]['held_out_follows'])}  →  {pct(MID[1.5]['held_out_follows'])}", font_size=24, color=GREEN).next_to(n1, DOWN, buff=0.15)
        self.at("lifts")
        self.play(FadeIn(n1), run_time=0.4)
        self.at("combinations")
        self.play(FadeIn(n2), run_time=0.4)
        self.at("wrong")
        self.play(FadeIn(l1), LaggedStart(*[FadeIn(t) for t in r1], lag_ratio=0.06), run_time=0.7)
        self.at("guidance")
        self.play(FadeIn(l2), LaggedStart(*[FadeIn(t) for t in r2], lag_ratio=0.06), run_time=0.7)
        self.end_section()

    # ------------------------------------------------------------------ 5. too far
    def too_far(self):
        self.section(5)
        self.clear_stage()
        names = json.load(open(f"{CODE_DIR}/results_mid6k.json"))["show"]
        g = Group()
        for k in range(3):
            rr = row(f"{OUT}/mid6k_w3_requests.png", h=1.0, r=k, buff=0.08)
            g.add(Group(small(names[k], COND_C, 30, 0.6).next_to(rr, LEFT, buff=0.3), rr))
        g.arrange(DOWN, buff=0.15, aligned_edge=RIGHT).move_to([0.3, 0.5, 0])
        head = Text("half-trained model, w = 3", font_size=26, color=RED_B).to_edge(UP, buff=0.4)
        self.at("turn")
        self.play(FadeIn(head), FadeIn(g), run_time=0.7)
        res = Text(f"{pct(MID[3]['follows'])} exactly as asked", font_size=28, color=RED).to_edge(DOWN, buff=0.8)
        self.at("10")
        self.play(FadeIn(res), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 6. finished model
    def finished(self):
        self.section(6)
        self.clear_stage()
        ws = [0, 1, 1.5, 2, 3, 5, 8]
        ax = Axes(x_range=[0, 8, 1], y_range=[0, 1, 0.25], x_length=9, y_length=4.2, tips=False,
                  axis_config={"color": GREY_C, "include_numbers": False}).move_to([0, -0.2, 0])
        xl = VGroup(*[Text(f"{w:g}", font=MONO, font_size=18).next_to(ax.c2p(w, 0), DOWN, buff=0.15) for w in (0, 1, 2, 3, 5, 8)])
        yl = VGroup(*[Text(f"{v}%", font=MONO, font_size=18).next_to(ax.c2p(0, v / 100), LEFT, buff=0.15) for v in (0, 50, 100)])
        xt = Text("guidance scale w", font_size=20, color=GREY_B).next_to(ax, DOWN, buff=0.55)
        yt = Text("exactly as asked", font_size=20, color=GREY_B).next_to(ax, UP, buff=0.2).align_to(ax, LEFT)

        def curve(R, col):
            pts = [ax.c2p(w, R[w]["follows"]) for w in ws]
            return VGroup(VMobject(color=col, stroke_width=5).set_points_as_corners(pts),
                          *[Dot(p, color=col, radius=0.07) for p in pts])
        cm, cf = curve(MID, MID_C), curve(FULL, FULL_C)
        lm = Text("— half-trained, 6,000 steps", font_size=22, color=MID_C).move_to(ax.c2p(5.2, 0.95), aligned_edge=LEFT)
        lf = Text("— finished, 20,000 steps", font_size=22, color=FULL_C).move_to(ax.c2p(5.2, 0.83), aligned_edge=LEFT)
        self.play(FadeIn(ax), FadeIn(xl), FadeIn(yl), FadeIn(xt), FadeIn(yt), Create(cm), FadeIn(lm), run_time=0.8)
        self.at("fully")
        self.play(Create(cf), FadeIn(lf), run_time=0.8)
        vals = VGroup(*[Text(f"w = {w:g}:  {pct(FULL[w]['follows'])}", font=MONO, font_size=20, color=FULL_C) for w in (1, 2, 3)])
        vals.arrange(DOWN, aligned_edge=LEFT, buff=0.12).move_to(ax.c2p(5.4, 0.55), aligned_edge=UL)
        for k, (cue, w) in enumerate((("97", 1), ("59", 2), ("four", 3))):
            self.at(cue)
            self.play(FadeIn(vals[k]), Flash(cf[1 + ws.index(w)], color=FULL_C), run_time=0.5)
        self.end_section()

    # ------------------------------------------------------------------ 7. why
    def why(self):
        self.section(7)
        self.clear_stage()
        P = f"{OUT}/predictions.png"
        names = ["nothing said", "request", "guided, w = 3"]
        cols = [UNC_C, COND_C, RED_B]
        g = Group()
        for r_ in range(3):
            rr = Group(*[tile(P, i, 1.05, row=r_) for i in range(5)]).arrange(RIGHT, buff=0.12)
            g.add(Group(Text(names[r_], font_size=20, color=cols[r_]).next_to(rr, LEFT, buff=0.3), rr))
        g.arrange(DOWN, buff=0.15, aligned_edge=RIGHT).move_to([0.2, 0.9, 0])
        ts = VGroup(*[Text(f"t = {t}", font=MONO, font_size=16, color=GREY_B).next_to(g[0][1][i], UP, buff=0.12)
                      for i, t in enumerate(FIG["predictions"]["steps_t"])])
        arr = Arrow(g[0][1][0].get_corner(UL) + 0.55 * UP, g[0][1][-1].get_corner(UR) + 0.55 * UP, buff=0, color=GREY_C, stroke_width=3)
        al = small("sampling →  less noise", GREY_B, 26).next_to(arr, UP, buff=0.05)
        self.at("first")
        self.play(FadeIn(g[:2]), FadeIn(ts), GrowArrow(arr), FadeIn(al), run_time=0.8)
        self.at("exaggerating")
        self.play(FadeIn(g[2]), run_time=0.6)
        wrong = Text("asked for a circle,\ndrew a triangle", font_size=20, color=RED_B, line_spacing=0.8).next_to(g[2][1][-1], RIGHT, buff=0.25)
        self.at("undo")
        self.play(FadeIn(wrong), run_time=0.4)
        A = FULL[3]["per_attribute"]
        pa = Text(f"finished model, w = 3:  shape {pct(A[0])}   color {pct(A[1])}   position {pct(A[2])}", font_size=22).to_edge(DOWN, buff=0.9)
        self.at("position")
        self.play(FadeIn(pa), run_time=0.4)
        self.play(Indicate(pa, color=RED_B), run_time=0.8)
        self.end_section()

    # ------------------------------------------------------------------ 8. late only
    def late(self):
        self.section(8)
        self.clear_stage()
        L = f"{OUT}/ladder.png"
        labels = [("w = 1", COND_C), ("w = 2", RED_B), ("w = 3", RED_B), ("w = 3, only t < 500", GREEN)]
        g = Group()
        for r_, (txt, col) in enumerate(labels):
            rr = row(L, h=0.85, r=r_, buff=0.07)
            g.add(Group(Text(txt, font=MONO, font_size=18, color=col).next_to(rr, LEFT, buff=0.3), rr))
        g.arrange(DOWN, buff=0.12, aligned_edge=RIGHT).move_to([0.6, 0.8, 0])
        head = small("finished model, same eight requests, same noise", WHITE, 32).to_edge(UP, buff=0.3)
        self.play(FadeIn(head), FadeIn(g[:3]), run_time=0.6)
        self.at("settled")
        self.play(FadeIn(g[3]), run_time=0.6)
        n3 = Text(f"w = 3:  {pct(FULL[3]['follows'])}  →  {pct(FULL[3]['late']['follows'])}", font=MONO, font_size=24, color=GREEN).move_to([-2.6, -2.7, 0])
        n2 = Text(f"w = 2:  {pct(FULL[2]['follows'])}  →  {pct(FULL[2]['late']['follows'])}", font=MONO, font_size=24, color=GREEN).move_to([2.6, -2.7, 0])
        self.at("78")
        self.play(FadeIn(n3), run_time=0.4)
        self.at("96")
        self.play(FadeIn(n2), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 9. real models
    def real(self):
        self.section(9)
        self.clear_stage()
        a = VGroup(Text("Stable Diffusion", font_size=28, color=YELLOW),
                   Text("default guidance scale: 7.5", font=MONO, font_size=24)).arrange(DOWN, buff=0.2).move_to([0, 1.3, 0])
        b = VGroup(Text("guidance in a limited interval", font_size=28, color=GREEN),
                   Text("skip the noisiest steps (Kynkäänniemi et al., 2024)", font_size=22)).arrange(DOWN, buff=0.2).move_to([0, -1.2, 0])
        self.at("real")
        self.play(FadeIn(a[0]), run_time=0.4)
        self.at("default")
        self.play(FadeIn(a[1]), run_time=0.4)
        self.at("recent")
        self.play(FadeIn(b[0]), run_time=0.4)
        self.at("noisiest")
        self.play(FadeIn(b[1]), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 10. code
    def code(self):
        self.section(10)
        self.clear_stage()
        code, hl = code_panel(CODE, font_size=18)
        code.move_to([0, 0.2, 0])
        self.play(FadeIn(code, shift=0.2 * UP), run_time=0.4)
        self.at("run")
        self.play(Create(hl), run_time=0.2)
        self.play(highlight(hl, code, 1), run_time=0.3)
        self.at("mix")
        self.play(highlight(hl, code, 3), run_time=0.3)
        self.at("difference")
        self.play(highlight(hl, code, 6), run_time=0.3)
        self.end_section()

    # ------------------------------------------------------------------ 11. outro
    def outro(self):
        self.section(11)
        self.clear_stage(run_time=0.4)
        card_ = next_up_card(NEXT)
        self.at("next")
        self.play(FadeIn(card_, shift=0.2 * UP), run_time=0.6)
        self.end_section()
        finish(self, hold=1.5)
