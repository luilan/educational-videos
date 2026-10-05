"""LLMs in Practice, episode 5 — RAG: Giving the Model a Library.

Render from the repo root:  ./render.sh llms-in-practice p05
Every score, token count and answer on screen comes from code/p05_rag/rag.py (all-MiniLM-L6-v2 retrieval;
Qwen2.5-1.5B-Instruct and 0.5B-Instruct readers, greedy). Bella's Bakery is fictional.
"""
from manim import *

from common import MODEL_COLOR, MONO, TOKEN_COLOR, code_panel, finish, highlight, next_up_card
from intro import play_token_intro
from p05_script import NEXT, TAGLINE, TITLE
from voiced_scene import VoicedScene

SERIES_LABEL = "LLMs IN PRACTICE  ·  EPISODE 5"
QUESTION = "Can I buy gluten-free bread at Bella's Bakery on Wednesday?"
LIBRARY = ["Opens at 7:30 on weekdays, 9:00 on weekends.", "Closed on Mondays.",
           "Sourdough loaf: 6 euros, baked every morning.", "Gluten-free bread is only available on Fridays.",
           "Delivery within 5 km for orders over 30 euros.", "Croissants are made with French butter."]
SCORES = [0.62, 0.60, 0.41, 0.70, 0.17, 0.18]
NO_RAG = "I'm sorry, but as an AI language model, I don't have access to\nreal-time information about Bella's Bakery …"
ANSWER_15 = "No, you cannot buy gluten-free bread at Bella's Bakery on\nWednesday because it is only available on Fridays."
ANSWER_05 = ("Yes, you can buy gluten-free bread at Bella's Bakery on Wednesday.\n"
             "The information states that Bella's Bakery opens at 7:30 on weekdays …")
CODE = """library_vectors = embed(LIBRARY)                  # once

def rag(question, k=2):
    hits = top_k(library_vectors @ embed([question])[0], k)
    context = "\\n".join("- " + LIBRARY[i] for i in hits)
    prompt = (INSTRUCTIONS + "\\n\\nInformation:\\n" + context
              + "\\n\\nQuestion: " + question)
    return generate(prompt)"""


def bubble(text, color, font_size=22):
    label = Text(text, font_size=font_size, line_spacing=0.8)
    box = RoundedRectangle(corner_radius=0.2, width=label.width + 0.6, height=label.height + 0.45, stroke_width=0,
                           fill_color=color, fill_opacity=0.9)
    return VGroup(box, label.move_to(box))


def labeled_box(text, color=MODEL_COLOR, width=2.4, height=1.0, font_size=24):
    label = Text(text, font_size=font_size, line_spacing=0.8)
    box = RoundedRectangle(corner_radius=0.2, width=max(width, label.width + 0.5), height=height, stroke_color=color,
                           fill_color=color, fill_opacity=0.25)
    return VGroup(box, label.move_to(box))


def doc_icon(w=0.5, h=0.65, color=GREY_B):
    page = Rectangle(width=w, height=h, stroke_color=color, stroke_width=1.5, fill_color=GREY_E, fill_opacity=1)
    lines = VGroup(*[Line(ORIGIN, (0.45 if k % 3 == 2 else 0.65) * w * RIGHT, stroke_width=1.2, color=color)
                     for k in range(4)]).arrange(DOWN, buff=h * 0.1, aligned_edge=LEFT)
    return VGroup(page, lines.move_to(page))


class RagVideo(VoicedScene):
    VIDEO = "p05"

    def construct(self):
        play_token_intro(self, TITLE, 5, TAGLINE, label=SERIES_LABEL)
        self.hook()        # 1
        self.no_rag()      # 2
        self.idea()        # 3
        self.retrieval()   # 4
        self.prompt()      # 5
        self.answer()      # 6
        self.twist()       # 7
        self.code()        # 8
        self.why()         # 9
        self.outro()       # 10

    def clear_stage(self, *keep, run_time=0.6):
        self.play(*[FadeOut(m) for m in self.mobjects if m not in keep], run_time=run_time)

    # ------------------------------------------------------------------ 1. hook
    def hook(self):
        self.section(1)
        model = labeled_box("model", width=2.4, height=1.2, font_size=30).move_to([-3.6, 0.3, 0])
        train = VGroup(*[doc_icon() for _ in range(9)]).arrange_in_grid(3, 3, buff=0.12).next_to(model, DOWN, buff=0.4)
        cutoff = Text("training data, up to a cutoff date", font_size=22, color=GREY_B).next_to(train, DOWN, buff=0.2)
        self.at("training")
        self.play(FadeIn(model), LaggedStart(*[FadeIn(d) for d in train], lag_ratio=0.05), run_time=0.8)
        self.at("cut")
        self.play(FadeIn(cutoff), run_time=0.4)
        unknown = VGroup(*[VGroup(doc_icon(0.6, 0.78, GOLD), Text(t, font_size=22, color=GOLD)).arrange(RIGHT, buff=0.3)
                           for t in ("your company's documents", "your notes", "today's news")])
        unknown.arrange(DOWN, buff=0.45, aligned_edge=LEFT).move_to([3.3, 0.3, 0])
        qs = VGroup(*[Text("?", font_size=34, color=RED_B).next_to(u, LEFT, buff=0.25) for u in unknown])
        for i, cue in enumerate(["companys", "notes", "news"]):
            self.at(cue)
            self.play(FadeIn(unknown[i], shift=0.2 * LEFT), FadeIn(qs[i]), run_time=0.45)
        self.end_section()

    # ------------------------------------------------------------------ 2. without RAG
    def no_rag(self):
        self.section(2)
        self.clear_stage()
        shop = Text("Bella's Bakery  (made up)", font_size=26, color=GOLD).to_edge(UP, buff=0.5)
        self.at("bakery")
        self.play(FadeIn(shop), run_time=0.4)
        q = bubble(QUESTION, BLUE_E).move_to([1.0, 1.6, 0])
        self.at("buy")
        self.play(FadeIn(q, shift=0.2 * UP), run_time=0.5)
        a = bubble(NO_RAG, GREY_D, 20).move_to([-0.6, -0.2, 0])
        tag = Text("real answer, Qwen2.5-1.5B, no RAG", font_size=18, color=GREY_B).next_to(a, DOWN, buff=0.15)
        self.at("without")
        self.play(FadeIn(a, shift=0.2 * UP), FadeIn(tag), run_time=0.6)
        worse = Text("or worse: invent something plausible", font_size=26, color=RED_B).move_to([0, -2.4, 0])
        self.at("invent")
        self.play(FadeIn(worse, shift=0.2 * UP), run_time=0.5)
        self.end_section()

    # ------------------------------------------------------------------ 3. the idea
    def idea(self):
        self.section(3)
        self.clear_stage()
        title = Text("RAG · retrieval-augmented generation", font_size=34, t2c={"RAG": YELLOW}).to_edge(UP, buff=0.5)
        self.at("rag")
        self.play(FadeIn(title, shift=0.2 * DOWN), run_time=0.6)
        lib = VGroup(*[doc_icon() for _ in range(6)]).arrange_in_grid(2, 3, buff=0.12)
        lib_l = Text("library", font_size=22, color=GREY_B).next_to(lib, DOWN, buff=0.15)
        library = VGroup(lib, lib_l).move_to([-5.2, 0, 0])
        search = labeled_box("1 · search", TEAL_C, width=2.2, font_size=24).move_to([-2.5, 0, 0])
        prompt = labeled_box("2 · prompt\n+ documents", BLUE_C, width=2.4, height=1.3, font_size=24).move_to([0.5, 0, 0])
        model = labeled_box("model", MODEL_COLOR, width=1.8, font_size=26).move_to([3.2, 0, 0])
        ans = bubble("answer", GREEN_E, 24).move_to([5.5, 0, 0])
        arrows = VGroup(*[Arrow(a.get_right(), b.get_left(), buff=0.12, color=GREY_B)
                          for a, b in [(library, search), (search, prompt), (prompt, model), (model, ans)]])
        self.at("search")
        self.play(FadeIn(library), GrowArrow(arrows[0]), FadeIn(search), run_time=0.7)
        self.at("paste")
        self.play(GrowArrow(arrows[1]), FadeIn(prompt), run_time=0.6)
        self.at("answer")
        self.play(GrowArrow(arrows[2]), FadeIn(model), GrowArrow(arrows[3]), FadeIn(ans), run_time=0.7)
        self.end_section()

    # ------------------------------------------------------------------ 4. retrieval
    def retrieval(self):
        self.section(4)
        self.clear_stage()
        head = Text("retrieval = episode 4's embedding search", font_size=28).to_edge(UP, buff=0.45)
        self.at("embeddings")
        self.play(FadeIn(head), run_time=0.4)
        rows = VGroup()
        for text, s in zip(LIBRARY, SCORES):
            bar = Rectangle(width=s * 4.2, height=0.35, stroke_width=0, fill_color=TEAL_D, fill_opacity=0.8)
            rows.add(VGroup(Text(text, font_size=22), bar, Text(f"{s:.2f}", font=MONO, font_size=20)))
        for r in rows:
            r[0].move_to([-6.6, 0, 0], aligned_edge=LEFT)
            r[1].move_to([1.7, 0, 0], aligned_edge=LEFT)
            r[2].next_to(r[1], RIGHT, buff=0.15)
        for i, r in enumerate(rows):
            r.set_y(1.9 - i * 0.65)
        self.at("six")
        self.play(LaggedStart(*[FadeIn(r[0], shift=0.2 * RIGHT) for r in rows], lag_ratio=0.1), run_time=0.9)
        q = Text(QUESTION, font_size=22, color=YELLOW).move_to([0, -2.4, 0])
        self.at("question")
        self.play(FadeIn(q, shift=0.2 * UP), run_time=0.4)
        self.at("closest")
        self.play(*[GrowFromEdge(r[1], LEFT) for r in rows], *[FadeIn(r[2]) for r in rows], run_time=0.8)
        for idx, cue in [(3, "gluten"), (0, "opening")]:
            frame = SurroundingRectangle(rows[idx], buff=0.1, color=YELLOW)
            self.at(cue)
            self.play(Create(frame), rows[idx][1].animate.set_fill(YELLOW), run_time=0.5)
        self.end_section()

    # ------------------------------------------------------------------ 5. the prompt
    def prompt(self):
        self.section(5)
        self.clear_stage()
        def block(rows):
            return VGroup(*[Text(t, font=MONO, font_size=18, color=c) for t, c in rows]).arrange(
                DOWN, aligned_edge=LEFT, buff=0.14)
        blocks = VGroup(block([("Answer the question using only the information below.", GREY_A),
                               ("If the answer is not there, say you don't know.", GREY_A)]),
                        block([("Information:", WHITE),
                               ("- Gluten-free bread is only available on Fridays.", TEAL_B),
                               ("- Bella's Bakery opens at 7:30 on weekdays and at 9:00 on weekends.", TEAL_B)]),
                        block([("Question: " + QUESTION, BLUE_B)])).arrange(DOWN, aligned_edge=LEFT, buff=0.4)
        box = RoundedRectangle(corner_radius=0.2, width=blocks.width + 0.7, height=blocks.height + 0.7,
                               stroke_color=GREY_C, fill_color=GREY_E, fill_opacity=1)
        panel = VGroup(box, blocks.move_to(box)).move_to([0, 0.4, 0])
        texts = VGroup(*blocks[0], VGroup(), *blocks[1], VGroup(), *blocks[2])
        self.at("prompt")
        self.play(FadeIn(box), run_time=0.4)
        self.at("instruction")
        self.play(FadeIn(texts[0:2]), run_time=0.5)
        self.at("retrieved")
        self.play(FadeIn(texts[3:6], shift=0.2 * RIGHT), run_time=0.5)
        self.at("question")
        self.play(FadeIn(texts[7], shift=0.2 * RIGHT), run_time=0.4)
        count = Text("101 tokens, all context (episode 1)", font_size=28, color=YELLOW).next_to(panel, DOWN, buff=0.4)
        self.at("101")
        self.play(FadeIn(count, shift=0.2 * UP), run_time=0.5)
        self.end_section()

    # ------------------------------------------------------------------ 6. the answer
    def answer(self):
        self.section(6)
        self.clear_stage()
        model = labeled_box("Qwen2.5-1.5B-Instruct", MODEL_COLOR, height=0.9, font_size=24).move_to([0, 2.3, 0])
        self.at("small")
        self.play(FadeIn(model, shift=0.2 * DOWN), run_time=0.5)
        a = bubble(ANSWER_15, GREEN_E, 24).move_to([0, 0.4, 0])
        self.at("no")
        self.play(FadeIn(a[0]), AddTextLetterByLetter(a[1], run_time=2.0))
        ok = Text("✓ correct", font_size=36, color=GREEN).next_to(a, DOWN, buff=0.45)
        self.at("correct")
        self.play(FadeIn(ok, scale=1.3), run_time=0.4)
        never = Text("from a fact it never saw in training", font_size=26, color=GREY_A).next_to(ok, DOWN, buff=0.3)
        self.at("never")
        self.play(FadeIn(never), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 7. the twist
    def twist(self):
        self.section(7)
        self.clear_stage()
        warn = Text("a warning", font_size=32, color=RED_B).to_edge(UP, buff=0.5)
        self.at("warning")
        self.play(FadeIn(warn), run_time=0.4)
        model = labeled_box("Qwen2.5-0.5B-Instruct · same prompt", MODEL_COLOR, height=0.8, font_size=22)
        model.move_to([0, 2.0, 0])
        self.at("smaller")
        self.play(FadeIn(model, shift=0.2 * DOWN), run_time=0.5)
        a = bubble(ANSWER_05, RED_E, 21).move_to([0, 0.4, 0])
        self.at("yes")
        self.play(FadeIn(a[0]), AddTextLetterByLetter(a[1], run_time=2.0))
        wrong = Text("✗ wrong, with the right fact in its prompt", font_size=28, color=RED).next_to(a, DOWN, buff=0.4)
        self.at("wrong")
        self.play(FadeIn(wrong, scale=1.2), run_time=0.5)
        tips = VGroup(Text("check answers", font_size=28, color=GREEN_B),
                      Text("ask for sources", font_size=28, color=GREEN_B)).arrange(RIGHT, buff=1.2)
        tips.next_to(wrong, DOWN, buff=0.45)
        self.at("check")
        self.play(FadeIn(tips[0], shift=0.2 * UP), run_time=0.4)
        self.at("sources")
        self.play(FadeIn(tips[1], shift=0.2 * UP), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 8. code
    def code(self):
        self.section(8)
        self.clear_stage()
        code, hl = code_panel(CODE, font_size=20)
        code.move_to([0, 0.2, 0])
        self.at("embed")
        self.play(FadeIn(code, shift=0.2 * UP), run_time=0.5)
        hl.match_y(code.line_numbers[0])
        self.play(Create(hl), run_time=0.3)
        self.at("retrieve")
        self.play(highlight(hl, code, 3), run_time=0.4)
        self.at("join")
        self.play(highlight(hl, code, 4), run_time=0.4)
        self.at("instructions")
        self.play(highlight(hl, code, 5), run_time=0.4)
        self.at("generate")
        self.play(highlight(hl, code, 7), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 9. why it matters
    def why(self):
        self.section(9)
        self.clear_stage()
        lines = VGroup(Text("private or fresh information", font_size=30),
                       Text("no retraining", font_size=30, color=GREEN_B)).arrange(DOWN, buff=0.35).move_to([0, 2.3, 0])
        self.at("private")
        self.play(FadeIn(lines[0]), run_time=0.5)
        self.at("retraining")
        self.play(FadeIn(lines[1]), run_time=0.4)
        old = bubble("Gluten-free bread is only available on Fridays.", GREY_D, 22).move_to([0, 0.5, 0])
        new = bubble("Gluten-free bread: Wednesdays and Fridays.", TEAL_E, 22).move_to([0, 0.5, 0])
        upd = Text("update a document → the next answer uses it (illustrative)", font_size=22, color=GREY_B)
        upd.next_to(old, DOWN, buff=0.3)
        self.at("update")
        self.play(FadeIn(old), run_time=0.3)
        self.play(Transform(old, new), FadeIn(upd), run_time=0.7)
        dep = Text("everything depends on retrieving the right pieces", font_size=28, color=YELLOW).move_to([0, -2.2, 0])
        self.at("depends")
        self.play(FadeIn(dep, shift=0.2 * UP), run_time=0.5)
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
