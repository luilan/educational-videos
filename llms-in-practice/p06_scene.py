"""LLMs in Practice, episode 6 — Chunking and Retrieval: Why RAG Fails, and Fixes.

Render from the repo root:  ./render.sh llms-in-practice p06
Every result, token count and chunk boundary on screen comes from code/p06_chunking/chunking.py
(three made-up handbooks, all-MiniLM-L6-v2, ten questions with known answers).
"""
from manim import *

from common import MODEL_COLOR, MONO, TOKEN_COLOR, code_panel, finish, highlight, next_up_card
from intro import play_token_intro
from p06_script import NEXT, TAGLINE, TITLE
from voiced_scene import VoicedScene

SERIES_LABEL = "LLMs IN PRACTICE  ·  EPISODE 6"
RESULTS = [("whole document", 3, 10, 255), ("fixed 30 words", 17, 8, 30), ("one per section", 17, 10, 38),
           ("sentence + neighbours", 63, 10, 28)]
CODE = """def by_section(text):
    return [s.strip() for s in text.split("\\n\\n")]     # blank lines

def sentence_window(text):
    s = sentences(text)
    return [" ".join(s[max(0, i - 1):i + 2])           # + neighbours
            for i in range(len(s))]"""


def doc_icon(w=0.5, h=0.65, color=GREY_B, n=4):
    page = Rectangle(width=w, height=h, stroke_color=color, stroke_width=1.5, fill_color=GREY_E, fill_opacity=1)
    lines = VGroup(*[Line(ORIGIN, (0.45 if k % 3 == 2 else 0.65) * w * RIGHT, stroke_width=1.2, color=color)
                     for k in range(n)]).arrange(DOWN, buff=h * 0.08, aligned_edge=LEFT)
    return VGroup(page, lines.move_to(page))


def chunk_box(text, color=GREY_D, font_size=18, width=None):
    label = Text(text, font=MONO, font_size=font_size)
    box = RoundedRectangle(corner_radius=0.1, width=width or label.width + 0.4, height=label.height + 0.35,
                           stroke_color=color, stroke_width=2, fill_color=color, fill_opacity=0.2)
    return VGroup(box, label.move_to(box))


def result_row(name, hits, words, color=WHITE):
    return VGroup(Text(name, font_size=26, color=color), Text(f"{hits} / 10", font=MONO, font_size=26, color=color),
                  Text(f"{words} words", font=MONO, font_size=26, color=color))


class ChunkingVideo(VoicedScene):
    VIDEO = "p06"

    def construct(self):
        play_token_intro(self, TITLE, 6, TAGLINE, label=SERIES_LABEL)
        self.hook()          # 1
        self.experiment()    # 2
        self.whole()         # 3
        self.fixed()         # 4
        self.natural()       # 5
        self.lessons()       # 6
        self.failures()      # 7
        self.code()          # 8
        self.measure()       # 9
        self.outro()         # 10

    def clear_stage(self, *keep, run_time=0.6):
        self.play(*[FadeOut(m) for m in self.mobjects if m not in keep], run_time=run_time)

    def row_at(self, row, y):
        row[0].move_to([-5.8, y, 0], aligned_edge=LEFT)
        row[1].move_to([1.3, y, 0])
        row[2].move_to([4.4, y, 0])
        return row

    # ------------------------------------------------------------------ 1. hook
    def hook(self):
        self.section(1)
        docs = VGroup(*[doc_icon(1.0, 1.35, n=8) for _ in range(3)]).arrange(RIGHT, buff=0.7).move_to([-3.6, 0.4, 0])
        names = VGroup(*[Text(t, font_size=22, color=GREY_B).next_to(d, DOWN, buff=0.15)
                         for t, d in zip(["handbook", "manual", "contract"], docs)])
        self.at("long")
        self.play(LaggedStart(*[FadeIn(d, shift=0.2 * UP) for d in docs], lag_ratio=0.2), FadeIn(names), run_time=0.8)
        window = RoundedRectangle(corner_radius=0.2, width=2.6, height=2.0, stroke_color=YELLOW, stroke_width=4)
        window.move_to([3.4, 0.4, 0])
        w_label = Text("prompt", font_size=24, color=YELLOW).next_to(window, UP, buff=0.15)
        self.at("paste")
        self.play(Create(window), FadeIn(w_label), run_time=0.5)
        self.play(Indicate(docs, color=RED, scale_factor=1.05), run_time=0.5)
        pieces = VGroup(*[Rectangle(width=1.6, height=0.32, stroke_width=0, fill_color=TOKEN_COLOR, fill_opacity=0.8)
                          for _ in range(3)]).arrange(DOWN, buff=0.12).move_to(window)
        self.at("pieces")
        self.play(*[TransformFromCopy(docs[i], pieces[i]) for i in range(3)], run_time=0.8)
        cut = Text("how you cut decides whether RAG works", font_size=30, color=YELLOW).move_to([0, -2.6, 0])
        self.at("cut")
        self.play(FadeIn(cut, shift=0.2 * UP), run_time=0.5)
        self.end_section()

    # ------------------------------------------------------------------ 2. the experiment
    def experiment(self):
        self.section(2)
        self.clear_stage()
        books = VGroup(*[VGroup(doc_icon(0.9, 1.2, c, 7), Text(t, font_size=22, color=c)).arrange(DOWN, buff=0.15)
                         for t, c in [("bakery", GOLD), ("bike shop", BLUE_C), ("gym", GREEN_C)]])
        books.arrange(RIGHT, buff=0.8).move_to([-3.4, 1.3, 0])
        made = Text("three made-up handbooks", font_size=22, color=GREY_B).next_to(books, UP, buff=0.25)
        self.at("library")
        self.play(FadeIn(made), LaggedStart(*[FadeIn(b, shift=0.2 * UP) for b in books], lag_ratio=0.2), run_time=0.9)
        qs = VGroup(*[Text(q, font_size=20) for q in ["When can I get gluten-free bread?",
                                                      "How much is a cake for 12 people?",
                                                      "What time does the morning baker start?", "…"]])
        qs.arrange(DOWN, aligned_edge=LEFT, buff=0.15)
        q_head = Text("10 questions, known answers", font_size=24, color=YELLOW)
        q_block = VGroup(q_head, qs).arrange(DOWN, aligned_edge=LEFT, buff=0.25).move_to([3.4, 1.3, 0])
        self.at("10")
        self.play(FadeIn(q_block, shift=0.2 * LEFT), run_time=0.6)
        checks = VGroup(Text("1. is the answer in the top chunk?", font_size=28),
                        Text("2. how many words go into the prompt?", font_size=28))
        checks.arrange(DOWN, aligned_edge=LEFT, buff=0.3).move_to([0, -2.0, 0])
        self.at("top")
        self.play(FadeIn(checks[0], shift=0.2 * UP), run_time=0.4)
        self.at("words")
        self.play(FadeIn(checks[1], shift=0.2 * UP), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 3. whole documents
    def whole(self):
        self.section(3)
        self.clear_stage()
        head = Text("strategy 1 · whole documents", font_size=30).to_edge(UP, buff=0.4)
        self.at("cutting")
        self.play(FadeIn(head), run_time=0.4)
        row = self.row_at(result_row("whole document", 10, 255), 2.3)
        self.at("every")
        self.play(FadeIn(row[0]), FadeIn(row[1]), run_time=0.5)
        self.at("255")
        self.play(FadeIn(row[2]), Indicate(row[2], color=RED), run_time=0.7)
        nine = Text("≈ 9× more than needed", font_size=24, color=RED_B).next_to(row[2], DOWN, buff=0.2)
        self.at("nine")
        self.play(FadeIn(nine), run_time=0.4)

        unit = 11.0 / 327
        bar = Rectangle(width=256 * unit, height=0.7, stroke_width=0, fill_color=GOLD, fill_opacity=0.8)
        lost = Rectangle(width=71 * unit, height=0.7, stroke_width=0, fill_color=RED, fill_opacity=0.8)
        VGroup(bar, lost).arrange(RIGHT, buff=0).move_to([0, -0.6, 0])
        limit = DashedLine(bar.get_corner(UR) + 0.5 * UP, bar.get_corner(DR) + 0.5 * DOWN, color=YELLOW,
                           stroke_width=4)
        lim_l = Text("embedding model reads 256 tokens", font_size=22, color=YELLOW).next_to(limit, UP, buff=0.1)
        bar_l = Text("bakery handbook: 327 tokens", font_size=22, color=GOLD).next_to(bar, UP, buff=0.15)
        bar_l.align_to(bar, LEFT)
        self.at("256")
        self.play(GrowFromEdge(bar, LEFT), Create(limit), FadeIn(lim_l), run_time=0.8)
        self.at("327")
        self.play(GrowFromEdge(lost, LEFT), FadeIn(bar_l), run_time=0.6)
        gone = Text("student discount · BELLA10 code · job ad\nnever reach the vector", font_size=24, color=RED_B,
                    line_spacing=0.8).next_to(lost, DOWN, buff=0.65).align_to(lost, RIGHT)
        self.at("student")
        self.play(FadeIn(gone, shift=0.2 * UP), run_time=0.6)
        self.end_section()

    # ------------------------------------------------------------------ 4. fixed-size chunks
    def fixed(self):
        self.section(4)
        self.clear_stage()
        head = Text("strategy 2 · every 30 words", font_size=30).to_edge(UP, buff=0.4)
        row = self.row_at(result_row("fixed 30 words", 8, 30), 2.3)
        self.at("30")
        self.play(FadeIn(head), run_time=0.4)
        self.at("eight")
        self.play(FadeIn(row), run_time=0.4)
        self.play(row[1].animate.set_color(RED), run_time=0.3)
        pairs = [("… We accept cash, cards and mobile payments. Students get a 10",
                  "percent discount with a valid student card. …"),
                 ("… Jobs We are hiring a morning baker. The", "shift starts at 4:00, and experience with …")]
        blocks = VGroup()
        for a, b in pairs:
            ca, cb = chunk_box(a, BLUE_C), chunk_box(b, TEAL_C)
            pair = VGroup(ca, cb).arrange(DOWN, buff=0.12, aligned_edge=LEFT)
            blocks.add(pair)
        blocks.arrange(DOWN, buff=0.55, aligned_edge=LEFT).move_to([0, -0.4, 0])
        scissors = VGroup(*[DashedLine(p[0].get_corner(DL) + 0.3 * LEFT, p[0].get_corner(DR) + 0.3 * RIGHT,
                                       color=RED, stroke_width=3).shift(0.06 * DOWN) for p in blocks])
        for i, (cue1, cue2) in enumerate([("students", "percent"), ("baker", "four")]):
            self.at(cue1)
            self.play(FadeIn(blocks[i][0], shift=0.2 * RIGHT), run_time=0.5)
            self.at(cue2)
            self.play(FadeIn(blocks[i][1], shift=0.2 * RIGHT), Create(scissors[i]), run_time=0.5)
        half = Text("facts cut in half", font_size=30, color=RED_B).to_edge(DOWN, buff=0.45)
        self.at("half")
        self.play(FadeIn(half, shift=0.2 * UP), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 5. natural boundaries
    def natural(self):
        self.section(5)
        self.clear_stage()
        head = Text("cut along the text's own structure", font_size=30).to_edge(UP, buff=0.4)
        self.at("structure")
        self.play(FadeIn(head), run_time=0.4)
        rows = VGroup(*[self.row_at(result_row(n, h, w, GREY_B if k < 2 else WHITE), 1.9 - k * 0.65)
                        for k, (n, c, h, w) in enumerate(RESULTS)])
        self.play(FadeIn(rows[0]), FadeIn(rows[1]), run_time=0.4)
        sec = chunk_box("Payment and discounts\nWe accept cash, cards and mobile payments. Students get a\n"
                        "10 percent discount with a valid student card. The code …", GREEN_C, 18)
        sec.move_to([0, -1.6, 0])
        sec_l = Text("one section, heading included", font_size=20, color=GREEN_B).next_to(sec, UP, buff=0.12)
        self.at("section")
        self.play(FadeIn(sec, shift=0.2 * UP), FadeIn(sec_l), run_time=0.5)
        self.at("ten")
        self.play(FadeIn(rows[2]), rows[2][1].animate.set_color(GREEN), run_time=0.5)
        self.at("thirty")
        self.play(Indicate(rows[2][2], color=GREEN), run_time=0.5)
        self.at("neighbors")
        self.play(FadeOut(sec), FadeOut(sec_l), run_time=0.3)
        sents = VGroup(*[Rectangle(width=1.4, height=0.35, stroke_width=1, stroke_color=BLACK, fill_color=GREY_C,
                                   fill_opacity=0.8) for _ in range(6)]).arrange(RIGHT, buff=0.1).move_to([0, -1.2, 0])
        windows = VGroup(*[SurroundingRectangle(VGroup(*sents[max(0, i - 1):i + 2]), buff=0.06 + 0.06 * (i % 2),
                                                color=[TEAL_C, GOLD][i % 2], stroke_width=3) for i in range(1, 5)])
        win_l = Text("each sentence + its neighbours: chunks overlap", font_size=20, color=GREY_A)
        win_l.next_to(sents, DOWN, buff=0.45)
        self.play(FadeIn(sents), run_time=0.3)
        self.at("overlap")
        self.play(LaggedStart(*[Create(w) for w in windows], lag_ratio=0.25), FadeIn(win_l), run_time=1.0)
        self.at("also")
        self.play(FadeIn(rows[3]), rows[3][1].animate.set_color(GREEN), run_time=0.5)
        self.at("twenty")
        self.play(Indicate(rows[3][2], color=GREEN), run_time=0.5)
        self.end_section()

    # ------------------------------------------------------------------ 6. lessons
    def lessons(self):
        self.section(6)
        self.clear_stage()
        tips = VGroup(*[Text(t, font_size=32) for t in ["cut at natural boundaries", "keep the headings: they carry the topic",
                                                         "overlap a little", "one answer per chunk, small and precise"]])
        tips.arrange(DOWN, aligned_edge=LEFT, buff=0.5).move_to([0, 0, 0])
        bullets = VGroup(*[Dot(color=YELLOW).next_to(t, LEFT, buff=0.3) for t in tips])
        for i, cue in enumerate(["natural", "headings", "overlap", "size"]):
            self.at(cue)
            self.play(FadeIn(tips[i], shift=0.2 * RIGHT), FadeIn(bullets[i]), run_time=0.45)
        self.end_section()

    # ------------------------------------------------------------------ 7. other failures
    def failures(self):
        self.section(7)
        self.clear_stage()
        head = Text("other ways retrieval fails", font_size=30).to_edge(UP, buff=0.4)
        self.at("retrieval")
        self.play(FadeIn(head), run_time=0.4)
        rows = [("different words than the document", "rewrite the question · add keyword search"),
                ("the answer needs two chunks", "retrieve more chunks"),
                ("too much retrieved", "the key fact gets lost in the middle (ep. 2)")]
        grid = VGroup(*[VGroup(Text(p, font_size=26, color=RED_B), Text("→ " + f, font_size=26, color=GREEN_B))
                        .arrange(DOWN, aligned_edge=LEFT, buff=0.15) for p, f in rows])
        grid.arrange(DOWN, aligned_edge=LEFT, buff=0.55).move_to([0, -0.2, 0])
        for i, (c1, c2) in enumerate([("different", "rewrite"), ("two", "retrieve"), ("much", "middle")]):
            self.at(c1)
            self.play(FadeIn(grid[i][0], shift=0.2 * RIGHT), run_time=0.4)
            self.at(c2)
            self.play(FadeIn(grid[i][1], shift=0.2 * RIGHT), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 8. code
    def code(self):
        self.section(8)
        self.clear_stage()
        code, hl = code_panel(CODE, font_size=24)
        code.move_to([0, 0.3, 0])
        self.at("code")
        self.play(FadeIn(code, shift=0.2 * UP), run_time=0.5)
        self.at("blank")
        hl.match_y(code.line_numbers[1])
        self.play(Create(hl), run_time=0.4)
        self.at("sentences")
        self.play(highlight(hl, code, 4), run_time=0.4)
        self.at("neighbors")
        self.play(highlight(hl, code, 5), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 9. measure
    def measure(self):
        self.section(9)
        self.clear_stage()
        big = Text("measure", font_size=64, color=YELLOW).move_to([0, 2.0, 0])
        self.at("measure")
        self.play(FadeIn(big, scale=1.2), run_time=0.5)
        loop = VGroup(Text("questions with known answers", font_size=28), Text("check what retrieval returns", font_size=28),
                      Text("change one thing at a time", font_size=28), Text("keep what works", font_size=28))
        loop.arrange(DOWN, buff=0.35).move_to([0, -0.8, 0])
        arrows = VGroup(*[Arrow(a.get_bottom(), b.get_top(), buff=0.06, color=GREY_B, stroke_width=3,
                                max_tip_length_to_length_ratio=0.35) for a, b in zip(loop[:-1], loop[1:])])
        for i, cue in enumerate(["questions", "check", "change", "keep"]):
            self.at(cue)
            self.play(FadeIn(loop[i], shift=0.2 * UP), *([GrowArrow(arrows[i - 1])] if i else []), run_time=0.4)
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
