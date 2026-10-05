"""LLMs in Practice, episode 1 — Prompts Are Just Context.

Render from the repo root:  ./render.sh llms-in-practice p01
Every number on screen comes from code/p01_prompt_is_context/what_the_model_sees.py
(Qwen2.5-0.5B-Instruct chat template and tokenizer).
"""
from manim import *

from common import MODEL_COLOR, MONO, TOKEN_COLOR, code_panel, finish, highlight, next_up_card, token
from intro import play_token_intro
from p01_script import NEXT, TAGLINE, TITLE
from voiced_scene import VoicedScene

SERIES_LABEL = "LLMs IN PRACTICE  ·  EPISODE 1"
SPECIAL = ORANGE
ROLE_COLORS = {"system": GREY_B, "user": BLUE_C, "assistant": GREEN_C}
MESSAGES = [("system", "You are a friendly cooking assistant."),
            ("user", "How long should I boil an egg?"),
            ("assistant", "About 7 minutes for a jammy yolk."),
            ("user", "And for hard-boiled?")]
# First 12 real tokens of the templated chat (token ID, text) and the real totals.
FIRST_TOKENS = [(151644, "<|im_start|>"), (8948, "system"), (198, "\\n"), (2610, "You"), (525, " are"),
                (264, " a"), (11657, " friendly"), (17233, " cooking"), (17847, " assistant"), (13, "."),
                (151645, "<|im_end|>"), (198, "\\n")]
TOTAL_TOKENS, TURN1_TOKENS = 55, 28
CODE = """tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
messages = [{"role": "system", "content": "You are a friendly ..."},
            {"role": "user", "content": "How long should I boil an egg?"}, ...]
text = tok.apply_chat_template(messages, tokenize=False,
                               add_generation_prompt=True)
ids = tok(text)["input_ids"]
print(len(ids))   # 55"""


# ---------------------------------------------------------------------------- helpers
def bubble(text, color, align, font_size=24):
    label = Text(text, font_size=font_size)
    box = RoundedRectangle(corner_radius=0.25, width=label.width + 0.6, height=label.height + 0.45,
                           stroke_width=0, fill_color=color, fill_opacity=0.9)
    return VGroup(box, label.move_to(box))


def role_card(role, text, width=5.6):
    color = ROLE_COLORS[role]
    tag = Text(role, font_size=20, color=color, weight=BOLD)
    body = Text(text, font_size=22)
    content = VGroup(tag, body).arrange(DOWN, aligned_edge=LEFT, buff=0.1)
    box = RoundedRectangle(corner_radius=0.15, width=max(width, content.width + 0.5), height=content.height + 0.35,
                           stroke_color=color,
                           stroke_width=2, fill_color=color, fill_opacity=0.12)
    content.move_to(box).align_to(box, LEFT).shift(0.25 * RIGHT)
    return VGroup(box, tag, body)


def template_lines():
    """The real Qwen chat template output, one Text per line, special tokens coloured."""
    rows = []
    for role, text in MESSAGES:
        rows.append(f"<|im_start|>{role}")
        rows.append(f"{text}<|im_end|>")
    rows.append("<|im_start|>assistant")
    lines = VGroup(*[Text(r, font=MONO, font_size=18, t2c={"<|im_start|>": SPECIAL, "<|im_end|>": SPECIAL})
                     for r in rows])
    return lines.arrange(DOWN, aligned_edge=LEFT, buff=0.14)


def panel(content, pad=0.35, color=GREY_C):
    box = RoundedRectangle(corner_radius=0.2, width=content.width + 2 * pad, height=content.height + 2 * pad,
                           stroke_color=color, stroke_width=2, fill_color=GREY_E, fill_opacity=1)
    content.move_to(box).set_z_index(2)
    return VGroup(box, content)


def bar(n_tokens, unit, colors, label):
    """A request drawn as a bar of coloured segments; colors = [(tokens, color), ...]."""
    segs = VGroup(*[Rectangle(width=n * unit, height=0.55, stroke_width=1, stroke_color=BLACK, fill_color=c,
                              fill_opacity=0.85) for n, c in colors]).arrange(RIGHT, buff=0)
    name = Text(label, font_size=22, color=GREY_A).next_to(segs, LEFT, buff=0.3)
    count = Text(f"{n_tokens} tokens", font_size=24).next_to(segs, RIGHT, buff=0.3)
    return VGroup(name, segs, count)


def context_block(text, color, width=5.2, height=0.7):
    box = RoundedRectangle(corner_radius=0.12, width=width, height=height, stroke_color=color, fill_color=color,
                           fill_opacity=0.3)
    return VGroup(box, Text(text, font_size=24).move_to(box))


class PromptIsContextVideo(VoicedScene):
    VIDEO = "p01"

    def construct(self):
        play_token_intro(self, TITLE, 1, TAGLINE, label=SERIES_LABEL)
        self.hook()              # 1
        self.flatten()           # 2
        self.open_turn()         # 3
        self.tokens()            # 4
        self.no_memory()         # 5
        self.everything()        # 6
        self.code()              # 7
        self.prompting()         # 8
        self.outro()             # 9

    def clear_stage(self, *keep, run_time=0.6):
        self.play(*[FadeOut(m) for m in self.mobjects if m not in keep], run_time=run_time)

    # ------------------------------------------------------------------ 1. hook
    def hook(self):
        self.section(1)
        phone = RoundedRectangle(corner_radius=0.4, width=8.6, height=6.2, stroke_color=GREY_B, stroke_width=3)
        bubbles = VGroup(bubble(MESSAGES[1][1], BLUE_E, RIGHT), bubble(MESSAGES[2][1], GREY_D, LEFT),
                         bubble(MESSAGES[3][1], BLUE_E, RIGHT))
        bubbles.arrange(DOWN, buff=0.45)
        for b, side in zip(bubbles, (RIGHT, LEFT, RIGHT)):
            b.align_to(phone, side).shift(-0.35 * side)
        VGroup(phone, bubbles).move_to(ORIGIN)
        bubbles.move_to(phone).align_to(phone, UP).shift(0.6 * DOWN)
        for b, side in zip(bubbles, (RIGHT, LEFT, RIGHT)):
            b.align_to(phone, side).shift(-0.35 * side)
        self.at("chat")
        self.play(Create(phone), run_time=0.6)
        self.at("conversation")
        self.play(FadeIn(bubbles[0], shift=0.3 * UP), run_time=0.5)
        self.at("bubbles")
        self.play(FadeIn(bubbles[1], shift=0.3 * UP), run_time=0.5)
        self.at("memory")
        self.play(FadeIn(bubbles[2], shift=0.3 * UP), run_time=0.5)

        self.at("never")
        self.play(VGroup(phone, bubbles).animate.set_opacity(0.15), run_time=0.7)
        strip = Text("<|im_start|>system You are a friendly cooking assistant.<|im_end|> <|im_start|>user How long ...",
                     font=MONO, font_size=20, t2c={"<|im_start|>": SPECIAL, "<|im_end|>": SPECIAL})
        strip.set(width=13.2)
        strip_box = SurroundingRectangle(strip, buff=0.2, corner_radius=0.1, color=GREY_B, fill_color=GREY_E,
                                         fill_opacity=1)
        self.strip = VGroup(strip_box, strip).set_z_index(3)
        self.at("one")
        self.play(FadeIn(self.strip, scale=0.9), run_time=0.7)
        self.end_section()

    # ------------------------------------------------------------------ 2. the chat template
    def flatten(self):
        self.section(2)
        self.clear_stage()
        cards = VGroup(*[role_card(r, t) for r, t in MESSAGES]).arrange(DOWN, buff=0.25).move_to([-3.6, 0, 0])
        cues = ["system", "ask", "answers", "follow"]
        for card, cue in zip(cards, cues):
            self.at(cue)
            self.play(FadeIn(card, shift=0.3 * RIGHT), run_time=0.5)

        self.doc = panel(template_lines()).move_to([2.55, 0, 0])
        title = Text("what the model reads", font_size=24, color=GREY_B).next_to(self.doc, UP, buff=0.2)
        self.at("flattens")
        self.play(cards.animate.scale(0.62).move_to([-4.95, 0, 0]).set_opacity(0.5),
                  FadeIn(self.doc[0]), FadeIn(title), run_time=0.6)
        self.play(LaggedStart(*[FadeIn(l, shift=0.2 * RIGHT) for l in self.doc[1]], lag_ratio=0.12), run_time=1.4)
        self.doc_title = title

        marks = VGroup(*[SurroundingRectangle(self.doc[1][i][:12], buff=0.04, color=SPECIAL, stroke_width=2)
                         for i in range(0, 9, 2)])
        label = Text("special tokens: who is speaking", font_size=22, color=SPECIAL)
        label.next_to(self.doc, DOWN, buff=0.25)
        self.at("special")
        self.play(Create(marks), FadeIn(label, shift=0.2 * UP), run_time=0.7)
        self.cards, self.marks, self.special_label = cards, marks, label
        self.end_section()

    # ------------------------------------------------------------------ 3. the open turn
    def open_turn(self):
        self.section(3)
        self.play(FadeOut(self.cards), FadeOut(self.marks), FadeOut(self.special_label),
                  VGroup(self.doc, self.doc_title).animate.move_to([0, 0.35, 0]), run_time=0.6)
        last = self.doc[1][-1]
        frame = SurroundingRectangle(last, buff=0.08, color=YELLOW, stroke_width=3)
        self.at("end")
        self.play(Create(frame), run_time=0.5)
        note = Text("an open\nassistant turn", font_size=24, color=YELLOW, line_spacing=0.8)
        note.next_to(self.doc, LEFT, buff=0.35).match_y(last)
        self.at("marker")
        self.play(FadeIn(note, shift=0.2 * LEFT), run_time=0.5)

        reply = Text("Hard-boiled: about 10 minutes.", font=MONO, font_size=20, color=YELLOW)
        reply.next_to(last, DOWN, buff=0.14, aligned_edge=LEFT)
        illus = Text("illustrative reply", font_size=20, color=GREY_B).to_corner(DR, buff=0.35)
        self.at("continue")
        self.play(self.doc[0].animate.stretch_to_fit_height(self.doc[0].height + 0.45).shift(0.225 * DOWN),
                  run_time=0.4)
        self.at("reply")
        self.play(AddTextLetterByLetter(reply, run_time=1.2), FadeIn(illus))
        big = Text("answering = continuing the text", font_size=32, t2c={"continuing": YELLOW})
        big.to_edge(DOWN, buff=0.45)
        self.at("continuing")
        self.play(FadeIn(big, shift=0.2 * UP), run_time=0.6)
        self.end_section()

    # ------------------------------------------------------------------ 4. tokens
    def tokens(self):
        self.section(4)
        self.clear_stage()
        toks = VGroup(*[token(t, color=SPECIAL if t.startswith("<|") else TOKEN_COLOR, font_size=20)
                        for _, t in FIRST_TOKENS]).arrange(RIGHT, buff=0.08)
        more = Text("…", font_size=32).next_to(toks, RIGHT, buff=0.15)
        row = VGroup(toks, more).move_to([0, 1.2, 0])
        if row.width > 13.4:
            row.scale_to_fit_width(13.4)
        ids = VGroup(*[Text(str(i), font=MONO, font_size=16, color=GREY_A).next_to(t, DOWN, buff=0.2)
                       for (i, _), t in zip(FIRST_TOKENS, toks)])
        self.at("tokens")
        self.play(LaggedStart(*[FadeIn(t, shift=0.2 * DOWN) for t in toks], lag_ratio=0.06), FadeIn(more),
                  run_time=1.1)
        count = Text(f"{TOTAL_TOKENS} tokens", font_size=56, color=YELLOW).move_to([0, -0.9, 0])
        self.at("55")
        self.play(FadeIn(count, scale=1.3), run_time=0.5)
        self.at("number")
        self.play(LaggedStart(*[FadeIn(i, shift=0.15 * UP) for i in ids], lag_ratio=0.05), run_time=0.8)
        world = Text("the model's entire world", font_size=30, color=GREY_A).next_to(count, DOWN, buff=0.4)
        self.at("world")
        self.play(FadeIn(world, shift=0.2 * UP), Circumscribe(VGroup(ids), color=YELLOW), run_time=0.8)
        self.end_section()

    # ------------------------------------------------------------------ 5. no memory
    def no_memory(self):
        self.section(5)
        self.clear_stage()
        title = Text("no memory between messages", font_size=36).to_edge(UP, buff=0.6)
        self.at("memory")
        self.play(FadeIn(title, shift=0.2 * DOWN), run_time=0.6)

        unit = 0.105
        req1 = bar(TURN1_TOKENS, unit, [(TURN1_TOKENS, BLUE_D)], "message 1").move_to([0, 1.0, 0])
        req2 = bar(TOTAL_TOKENS, unit, [(TURN1_TOKENS, BLUE_D), (TOTAL_TOKENS - TURN1_TOKENS, TEAL_D)], "message 2")
        req2.move_to([0, -0.6, 0])
        req2[1].align_to(req1[1], LEFT)
        req2[0].next_to(req2[1], LEFT, buff=0.3)
        req2[2].next_to(req2[1], RIGHT, buff=0.3)
        self.at("whole")
        self.play(FadeIn(req1[0]), GrowFromEdge(req1[1], LEFT), run_time=0.6)
        self.at("top")
        again = req1[1].copy()
        self.play(FadeIn(req2[0]), again.animate.move_to(req2[1][0]), run_time=0.7)
        resent = Text("sent again", font_size=20, color=WHITE).move_to(req2[1][0])
        self.play(GrowFromEdge(req2[1][1], LEFT), FadeIn(resent), run_time=0.5)
        self.remove(again)
        self.add(req2[1])
        self.at("28")
        self.play(FadeIn(req1[2], shift=0.2 * LEFT), run_time=0.4)
        self.at("55")
        self.play(FadeIn(req2[2], shift=0.2 * LEFT), run_time=0.4)
        grows = Text("every turn resends the whole chat", font_size=28, color=YELLOW).to_edge(DOWN, buff=0.8)
        self.at("longer")
        self.play(FadeIn(grows, shift=0.2 * UP), run_time=0.6)
        self.end_section()

    # ------------------------------------------------------------------ 6. everything is context
    def everything(self):
        self.section(6)
        self.clear_stage()
        window = RoundedRectangle(corner_radius=0.25, width=6.4, height=5.2, stroke_color=YELLOW, stroke_width=3)
        window.move_to([-2.2, -0.2, 0])
        head = Text("the context", font_size=30, color=YELLOW).next_to(window, UP, buff=0.15)
        blocks = VGroup(context_block("system prompt", GREY_B), context_block("earlier turns", BLUE_C),
                        context_block("pasted documents", TEAL_C), context_block("tool results", GOLD))
        blocks.arrange(DOWN, buff=0.3).move_to(window)
        self.at("text")
        self.play(Create(window), FadeIn(head), run_time=0.6)
        for cue, b in zip(["system", "earlier", "documents", "results"], blocks):
            self.at(cue)
            self.play(FadeIn(b, shift=0.2 * DOWN), run_time=0.4)

        outside = VGroup(*[Text(s, font_size=24, color=GREY_B) for s in
                           ("yesterday's chat", "your other files", "anything not sent")]).arrange(DOWN, buff=0.45)
        outside.move_to([4.6, -0.2, 0])
        cross = Cross(outside, stroke_color=RED, stroke_width=5)
        self.at("isnt")
        self.play(FadeIn(outside, shift=0.2 * LEFT), run_time=0.5)
        self.at("cant")
        self.play(Create(cross), run_time=0.5)
        self.end_section()

    # ------------------------------------------------------------------ 7. code
    def code(self):
        self.section(7)
        self.clear_stage()
        code, hl = code_panel(CODE, font_size=20)
        code.move_to([0, 0.3, 0])
        self.at("code")
        self.play(FadeIn(code, shift=0.2 * UP), run_time=0.6)
        self.at("messages")
        hl.match_y(code.line_numbers[1])
        self.play(Create(hl), run_time=0.4)
        self.at("template")
        self.play(highlight(hl, code, 3), run_time=0.4)
        self.at("tokenize")
        self.play(highlight(hl, code, 5), run_time=0.4)
        self.at("count")
        self.play(highlight(hl, code, 6), run_time=0.4)
        repo = Text("code/p01_prompt_is_context  ·  link below", font_size=24, color=GREY_A)
        repo.next_to(code, DOWN, buff=0.35)
        self.at("repository")
        self.play(FadeIn(repo, shift=0.2 * UP), run_time=0.5)
        self.end_section()

    # ------------------------------------------------------------------ 8. prompting
    def prompting(self):
        self.section(8)
        self.clear_stage()
        not_this = Text("giving orders to a mind", font_size=34, color=GREY_B).move_to([0, 2.4, 0])
        strike = Line(not_this.get_left(), not_this.get_right(), color=RED, stroke_width=5)
        self.at("orders")
        self.play(FadeIn(not_this), run_time=0.5)
        self.at("mind")
        self.play(Create(strike), run_time=0.4)

        doc_text = VGroup(*[Text(s, font=MONO, font_size=22) for s in
                            ("Q: How long to soft-boil an egg?", "A: 6 minutes.",
                             "Q: How long to hard-boil an egg?", "A:")]).arrange(DOWN, aligned_edge=LEFT, buff=0.18)
        doc_text[2:].shift(0.3 * DOWN)
        doc = panel(doc_text).move_to([-2.0, -0.3, 0])
        start = Text("the beginning of a document", font_size=28).next_to(doc, UP, buff=0.25)
        self.at("writing")
        self.play(FadeIn(doc, shift=0.2 * UP), FadeIn(start), run_time=0.7)
        nxt = Text(" 10 minutes.", font=MONO, font_size=22, color=YELLOW)
        nxt.next_to(doc_text[-1], RIGHT, buff=0.12)
        likely = Text("most likely\ncontinuation", font_size=24, color=YELLOW, line_spacing=0.8)
        likely.next_to(doc, RIGHT, buff=0.6)
        self.at("continuation")
        self.play(AddTextLetterByLetter(nxt, run_time=0.8), FadeIn(likely, shift=0.2 * LEFT))
        example = SurroundingRectangle(doc_text[:2], buff=0.08, color=GREEN, stroke_width=2)
        ex_label = Text("a good example", font_size=22, color=GREEN).next_to(doc, DOWN, buff=0.25)
        self.at("clear")
        self.play(Indicate(start, color=GREEN), run_time=0.6)
        self.at("examples")
        self.play(Create(example), FadeIn(ex_label), run_time=0.6)
        self.end_section()

    # ------------------------------------------------------------------ 9. outro
    def outro(self):
        self.section(9)
        self.at("limit")
        self.clear_stage(run_time=0.5)
        card = next_up_card(NEXT)
        self.at("next")
        self.play(FadeIn(card, shift=0.2 * UP), run_time=0.6)
        self.end_section()
        finish(self, hold=1.5)
