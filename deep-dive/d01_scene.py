"""How LLMs Work: Deep Dive, episode 1 — Byte-Pair Encoding, Step by Step.

Render from the repo root:  ./render.sh deep-dive d01
Every merge, count and token on screen comes from code/d01_bpe/bpe.py (trained on the first 200,000 characters of
Tiny Shakespeare; tested on the last 100,000) and GPT-2's real tokenizer.
"""
from manim import *

from common import MODEL_COLOR, MONO, TOKEN_COLOR, code_panel, finish, highlight, next_up_card, token
from intro import play_token_intro
from d01_script import NEXT, TAGLINE, TITLE
from voiced_scene import VoicedScene

SERIES_LABEL = "HOW LLMs WORK: DEEP DIVE  ·  EPISODE 1"
MERGES = [(1, "e", "␣", 5249, 194751), (2, "t", "h", 4065, 190686), (3, "t", "␣", 2936, 187750),
          (4, "s", "␣", 2797, 184953), (5, "o", "u", 2529, 182424)]
NAIVE = ["T", "o␣", "b", "e,␣", "or␣", "not␣", "to␣", "b", "e,␣", "that␣", "is␣", "the␣", "qu", "es", "tion", "."]
WORDS = ["To", "␣be", ",", "␣or", "␣not", "␣to", "␣be", ",", "␣that", "␣is", "␣the", "␣qu", "est", "ion", "."]
GPT2 = ["To", "␣be", ",", "␣or", "␣not", "␣to", "␣be", ",", "␣that", "␣is", "␣the", "␣question", "."]
CODE = """ids = list(text.encode("utf-8"))            # start: one token per byte
for step in range(N_MERGES):
    pair = Counter(zip(ids, ids[1:])).most_common(1)[0][0]
    ids = merge(ids, pair, 256 + step)          # replace it everywhere
    merges[pair] = 256 + step

def encode(text):   # replay the merges, earliest first
def decode(ids):    return b"".join(vocab[i] for i in ids).decode()"""


def token_row(pieces, color=TOKEN_COLOR, font_size=20, buff=0.06, max_width=13.4):
    row = VGroup(*[token(p, color=color, font_size=font_size) for p in pieces]).arrange(RIGHT, buff=buff)
    if row.width > max_width:
        row.scale_to_fit_width(max_width)
    return row


def count_bar(n, total=200_000, width=10.0, color=TOKEN_COLOR):
    return Rectangle(width=width * n / total, height=0.4, stroke_width=0, fill_color=color, fill_opacity=0.85)


class BPEVideo(VoicedScene):
    VIDEO = "d01"

    def construct(self):
        play_token_intro(self, TITLE, 1, TAGLINE, label=SERIES_LABEL)
        self.hook()          # 1
        self.bytes()         # 2
        self.algorithm()     # 3
        self.first()         # 4
        self.after()         # 5
        self.problem()       # 6
        self.words()         # 7
        self.gpt2()          # 8
        self.tradeoff()      # 9
        self.code()          # 10
        self.outro()         # 11

    def clear_stage(self, *keep, run_time=0.6):
        self.play(*[FadeOut(m) for m in self.mobjects if m not in keep], run_time=run_time)

    # ------------------------------------------------------------------ 1. hook
    def hook(self):
        self.section(1)
        row = token_row(["To", "␣be", ",", "␣or", "␣not", "␣to", "␣be"], font_size=26).move_to([0, 0.8, 0])
        self.at("tokens")
        self.play(LaggedStart(*[FadeIn(t, shift=0.2 * DOWN) for t in row], lag_ratio=0.1), run_time=0.8)
        q = Text("but where does the vocabulary come from?", font_size=30, color=YELLOW).move_to([0, -0.8, 0])
        self.at("scratch")
        self.play(FadeIn(q), run_time=0.5)
        m = Text("one merge at a time", font_size=26, color=GREY_B).next_to(q, DOWN, buff=0.4)
        self.at("merge")
        self.play(FadeIn(m), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 2. bytes
    def bytes(self):
        self.section(2)
        self.clear_stage()
        text = Text("To be", font_size=48).move_to([-4.0, 1.8, 0])
        raw = list("To be".encode("utf-8"))
        boxes = token_row([str(b) for b in raw], GREY_B, 26, 0.15).move_to([2.0, 1.8, 0])
        arrow = Arrow(text.get_right(), boxes.get_left(), buff=0.3, color=GREY_B)
        utf = Text("UTF-8", font_size=20, color=GREY_B).next_to(arrow, UP, buff=0.05)
        self.at("bytes")
        self.play(FadeIn(text), GrowArrow(arrow), FadeIn(utf), FadeIn(boxes, shift=0.2 * LEFT), run_time=0.8)
        vocab = Text("a byte: 256 possible values  →  start with 256 tokens", font_size=30).move_to([0, 0.3, 0])
        self.at("256")
        self.play(FadeIn(vocab), run_time=0.5)
        never = Text("nothing is ever unknown", font_size=28, color=GREEN_B).next_to(vocab, DOWN, buff=0.35)
        self.at("unknown")
        self.play(FadeIn(never), run_time=0.4)
        bar = count_bar(200_000).move_to([0, -2.0, 0])
        label = Text("200,000 characters of Shakespeare = 200,000 byte tokens", font_size=24).next_to(bar, UP, 0.15)
        self.at("200")
        self.play(GrowFromEdge(bar, LEFT), FadeIn(label), run_time=0.8)
        self.bar = bar
        self.end_section()

    # ------------------------------------------------------------------ 3. the algorithm
    def algorithm(self):
        self.section(3)
        self.clear_stage()
        steps = VGroup(*[Text(t, font_size=30) for t in ["1 · count every pair of neighbouring tokens",
                                                          "2 · take the most frequent pair",
                                                          "3 · give it a new token number",
                                                          "4 · replace it everywhere"]])
        steps.arrange(DOWN, aligned_edge=LEFT, buff=0.4).move_to([-0.5, 0.3, 0])
        for i, cue in enumerate(["count", "frequent", "new", "replace"]):
            self.at(cue)
            self.play(FadeIn(steps[i], shift=0.2 * RIGHT), run_time=0.4)
        loop = CurvedArrow(steps[3].get_right() + 0.3 * RIGHT, steps[0].get_right() + 0.3 * RIGHT, angle=PI / 2,
                           color=YELLOW)
        again = Text("repeat", font_size=26, color=YELLOW).next_to(loop, RIGHT, buff=0.15)
        self.at("again")
        self.play(Create(loop), FadeIn(again), run_time=0.6)
        self.end_section()

    # ------------------------------------------------------------------ 4. first merges
    def first(self):
        self.section(4)
        self.clear_stage()
        head = VGroup(*[Text(t, font_size=22, color=GREY_B) for t in ["merge", "pair", "new token", "seen",
                                                                     "text length"]])
        xs = [-5.6, -3.4, -0.9, 1.6, 4.4]
        for h, x in zip(head, xs):
            h.move_to([x, 2.6, 0])
        self.at("merges")
        self.play(FadeIn(head), run_time=0.4)
        rows = VGroup()
        for k, (n, a, b, seen, length) in enumerate(MERGES):
            y = 1.9 - k * 0.6
            cells = VGroup(Text(f"{n}", font=MONO, font_size=24),
                           VGroup(token(a, GREY_B, 22), Text("+", font_size=22), token(b, GREY_B, 22)).arrange(RIGHT, buff=0.1),
                           token(a + b, YELLOW, 22), Text(f"{seen:,}", font=MONO, font_size=24),
                           Text(f"{length:,}", font=MONO, font_size=24))
            for c, x in zip(cells, xs):
                c.move_to([x, y, 0])
            rows.add(cells)
        idnote = Text("“e␣” becomes token 256, the first new token", font_size=24, color=YELLOW).move_to([0, -1.5, 0])
        self.at("e")
        self.play(FadeIn(rows[0][:3]), run_time=0.5)
        self.at("times")
        self.play(FadeIn(rows[0][3:]), run_time=0.4)
        self.at("256")
        self.play(FadeIn(idnote), run_time=0.3)
        for k, cue in [(1, "h"), (2, "and"), (3, "s"), (4, "o")]:
            self.at(cue)
            self.play(FadeIn(rows[k], shift=0.2 * RIGHT), run_time=0.35)
        self.at("shorter")
        self.play(Indicate(VGroup(*[r[4] for r in rows]), color=YELLOW), run_time=0.6)
        self.end_section()

    # ------------------------------------------------------------------ 5. after 500 merges
    def after(self):
        self.section(5)
        self.clear_stage()
        b0 = count_bar(200_000, color=GREY_C).move_to([1.6, 2.2, 0])
        b1 = count_bar(81_132).align_to(b0, LEFT).set_y(1.5)
        l0 = Text("start: 200,000", font=MONO, font_size=20).next_to(b0, LEFT, buff=0.2)
        l1 = Text("500 merges: 81,132", font=MONO, font_size=20, color=YELLOW).next_to(b1, LEFT, buff=0.2)
        l1.align_to(l0, RIGHT)
        self.at("500")
        self.play(FadeIn(b0), FadeIn(l0), run_time=0.4)
        self.at("81")
        self.play(GrowFromEdge(b1, LEFT), FadeIn(l1), run_time=0.7)
        replay = Text("encoding new text: replay the merges, earliest first", font_size=26, color=GREY_A)
        replay.move_to([0, 0.5, 0])
        self.at("replay")
        self.play(FadeIn(replay), run_time=0.4)
        line = Text("“To be, or not to be, that is the question.”", font_size=26, color=BLUE_B).move_to([0, -0.4, 0])
        row = token_row(NAIVE, font_size=20).move_to([0, -1.4, 0])
        n = Text("16 tokens", font_size=28, color=YELLOW).next_to(row, DOWN, buff=0.35)
        self.at("16")
        self.play(FadeIn(line), LaggedStart(*[FadeIn(t) for t in row], lag_ratio=0.04), FadeIn(n), run_time=0.9)
        self.at("split")
        self.play(*[Indicate(row[i], color=RED) for i in (12, 13, 14)], run_time=0.8)
        self.end_section()

    # ------------------------------------------------------------------ 6. the problem
    def problem(self):
        self.section(6)
        self.clear_stage()
        head = Text("the longest tokens it learned", font_size=30).to_edge(UP, buff=0.6)
        self.at("longest")
        self.play(FadeIn(head), run_time=0.4)
        longest = VGroup(*[token(t, RED_C, 24) for t in [".↵↵MENENIUS:↵", ".↵↵SICINIUS:↵", ".↵↵CORIOLANUS:↵",
                                                          ".↵↵GLOUCESTER:↵"]]).arrange(DOWN, buff=0.3)
        longest.move_to([0, 0.3, 0])
        key = Text("↵ = new line", font_size=20, color=GREY_B).next_to(longest, RIGHT, buff=0.5)
        self.at("coriolanus")
        self.play(LaggedStart(*[FadeIn(t, shift=0.2 * UP) for t in longest], lag_ratio=0.2), FadeIn(key), run_time=0.8)
        glued = Text("punctuation + new lines + a name: one token", font_size=28, color=RED_B).move_to([0, -1.9, 0])
        self.at("glued")
        self.play(FadeIn(glued), run_time=0.4)
        why = Text("merges ignore word boundaries", font_size=28, color=YELLOW).next_to(glued, DOWN, buff=0.3)
        self.at("boundaries")
        self.play(FadeIn(why), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 7. pre-tokenization
    def words(self):
        self.section(7)
        self.clear_stage()
        head = Text("fix: split into words first, then merge only inside a word", font_size=28).to_edge(UP, buff=0.4)
        self.at("split")
        self.play(FadeIn(head), run_time=0.4)
        words = token_row(["To", "␣be", ",", "␣or", "␣not", "␣to", "␣be", ",", "␣that", "␣is", "␣the", "␣question", "."],
                          GREY_B, 20, 0.12).move_to([0, 2.2, 0])
        note = Text("the space stays at the start of the word", font_size=20, color=GREY_B).next_to(words, DOWN, 0.15)
        self.at("inside")
        self.play(FadeIn(words), FadeIn(note), run_time=0.6)
        first = VGroup(Text("first merges:", font_size=24, color=GREY_B),
                       *[token(t, YELLOW, 22) for t in ["␣t", "he", "ou", "␣a", "␣s"]]).arrange(RIGHT, buff=0.2)
        first.move_to([0, 0.8, 0])
        self.at("first")
        self.play(FadeIn(first), run_time=0.5)
        longest = VGroup(Text("longest tokens:", font_size=24, color=GREY_B),
                         *[token(t, GREEN_C, 22) for t in ["␣Senator", "␣Murderer", "CORIOLANUS", "␣Servingman"]])
        longest.arrange(RIGHT, buff=0.2).move_to([0, -0.2, 0])
        self.at("senator")
        self.play(FadeIn(longest), run_time=0.5)
        row = token_row(WORDS, font_size=20).move_to([0, -1.5, 0])
        n = Text("15 tokens", font_size=28, color=YELLOW).next_to(row, DOWN, buff=0.3)
        self.at("15")
        self.play(LaggedStart(*[FadeIn(t) for t in row], lag_ratio=0.04), FadeIn(n), run_time=0.8)
        self.at("real")
        self.play(*[Indicate(row[i], color=YELLOW) for i in (1, 4, 10)], run_time=0.7)
        self.end_section()

    # ------------------------------------------------------------------ 8. GPT-2
    def gpt2(self):
        self.section(8)
        self.clear_stage()
        head = Text("GPT-2: the same recipe, 50,000 merges, 40 GB of web text", font_size=28).to_edge(UP, buff=0.45)
        self.at("gpt")
        self.play(FadeIn(head), run_time=0.5)
        formula = Text("256 bytes + 50,000 merges + 1 end-of-text = 50,257 tokens", font=MONO, font_size=24,
                       color=YELLOW).move_to([0, 2.0, 0])
        self.at("vocabulary")
        self.play(FadeIn(formula), run_time=0.6)
        row = token_row(GPT2, font_size=22).move_to([0, 0.8, 0])
        n = Text("13 tokens", font_size=26, color=YELLOW).next_to(row, DOWN, buff=0.25)
        self.at("13")
        self.play(LaggedStart(*[FadeIn(t) for t in row], lag_ratio=0.05), FadeIn(n), run_time=0.8)
        self.at("single")
        self.play(Indicate(row[11], color=YELLOW, scale_factor=1.2), run_time=0.6)
        ours = count_bar(48_811, 48_811, 8.0, GREY_C).move_to([1.0, -1.4, 0])
        gpt = count_bar(32_324, 48_811, 8.0, TOKEN_COLOR).align_to(ours, LEFT).set_y(-2.1)
        lo = Text("ours, 500 merges: 48,811", font=MONO, font_size=20).next_to(ours, LEFT, buff=0.2)
        lg = Text("GPT-2: 32,324", font=MONO, font_size=20, color=BLUE_B).next_to(gpt, LEFT, buff=0.2).align_to(lo, RIGHT)
        cap = Text("tokens for 100,000 unseen characters of Shakespeare", font_size=20, color=GREY_B)
        cap.next_to(ours, UP, buff=0.15)
        self.at("third")
        self.play(FadeIn(cap), FadeIn(ours), FadeIn(lo), GrowFromEdge(gpt, LEFT), FadeIn(lg), run_time=0.8)
        self.end_section()

    # ------------------------------------------------------------------ 9. trade-off
    def tradeoff(self):
        self.section(9)
        self.clear_stage()
        q = Text("why not merge forever?", font_size=34, color=YELLOW).to_edge(UP, buff=0.6)
        self.at("forever")
        self.play(FadeIn(q), run_time=0.4)
        pro = VGroup(Text("bigger vocabulary", font_size=28), Text("✓ shorter sequences, less attention work",
                                                                    font_size=24, color=GREEN_B))
        con = VGroup(Text("but", font_size=28), Text("✗ one embedding row per token", font_size=24, color=RED_B),
                     Text("✗ rare tokens get little training", font_size=24, color=RED_B))
        pro.arrange(DOWN, aligned_edge=LEFT, buff=0.2).move_to([-3.4, 0.6, 0])
        con.arrange(DOWN, aligned_edge=LEFT, buff=0.2).move_to([3.0, 0.6, 0])
        self.at("shorter")
        self.play(FadeIn(pro), run_time=0.5)
        self.at("embedding")
        self.play(FadeIn(con[:2]), run_time=0.5)
        self.at("rare")
        self.play(FadeIn(con[2]), run_time=0.4)
        modern = Text("modern models: about 100,000 – 250,000 tokens (Qwen2.5: 151,936)", font_size=26,
                      color=YELLOW).move_to([0, -2.0, 0])
        self.at("modern")
        self.play(FadeIn(modern), run_time=0.5)
        self.end_section()

    # ------------------------------------------------------------------ 10. code
    def code(self):
        self.section(10)
        self.clear_stage()
        code, hl = code_panel(CODE, font_size=22)
        code.move_to([0, 0.3, 0])
        self.at("loop")
        self.play(FadeIn(code, shift=0.2 * UP), run_time=0.5)
        hl.match_y(code.line_numbers[1])
        self.play(Create(hl), run_time=0.3)
        self.at("count")
        self.play(highlight(hl, code, 2), run_time=0.4)
        self.at("merge")
        self.play(highlight(hl, code, 3), run_time=0.4)
        self.at("encoding")
        self.play(highlight(hl, code, 6), run_time=0.4)
        self.at("decoding")
        self.play(highlight(hl, code, 7), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 11. outro
    def outro(self):
        self.section(11)
        self.clear_stage(run_time=0.4)
        card = next_up_card(NEXT)
        self.at("next")
        self.play(FadeIn(card, shift=0.2 * UP), run_time=0.6)
        self.end_section()
        finish(self, hold=1.5)
