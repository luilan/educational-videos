"""Video 2 — Tokens: Chopping Text into Pieces.

Render from the repo root:  ./render.sh how-llms-work v02
"""

from manim import *

from intro import play_token_intro
from v02_script import TAGLINE, TITLE
from voiced_scene import VoicedScene

TOKEN_COLOR = BLUE_D
MONO = "DejaVu Sans Mono"
SP = "␣"  # visible leading space in GPT-2 tokens

_CODE_ROWS = [
    ("def train_bpe(words, num_merges):", ""),
    ("    merges = []", ""),
    ("    for _ in range(num_merges):", ""),
    ("        pairs = count_pairs(words)", "# neighbour pairs -> count"),
    ("        best = max(pairs, key=pairs.get)", "# most frequent pair"),
    ("        words = merge_pair(words, best)", "# glue it everywhere"),
    ("        merges.append(best)", ""),
    ("    return merges", ""),
]
BPE_CODE = "\n".join(f"{c}  {k}" if k else c for c, k in _CODE_ROWS)

SENTENCE = ["The", "cat", "sat", "on", "the"]
SENTENCE_IDS = [464, 3797, 3332, 319, 262]


# --- text / token helpers -------------------------------------------------------------------
def btext(s, font_size=28, color=WHITE, font=None):
    """Glyphs of `s` with a known baseline (.cap_off = glyph centre y minus cap-height middle)."""
    kw = {"font": font} if font else {}
    t = Text("H" + s, font_size=font_size, color=color, **kw)
    g = VGroup(*t[1:])
    g.cap_off = g.get_y() - t[0].get_y()
    return g


def seat(g, x, y, edge=None):
    """Put the cap-height middle of btext `g` at height y; centre it (or align its `edge`) at x."""
    ref_x = g.get_x() if edge is None else g.get_edge_center(edge)[0]
    g.shift([x - ref_x, y + g.cap_off - g.get_y(), 0])
    return g


def token(s, color=TOKEN_COLOR, font_size=28, height=0.65, min_width=0.6, pad=0.35, space_color=GREY_B):
    g = btext(s, font_size)
    if s.startswith(SP):
        g[0].set_color(space_color)
    box = RoundedRectangle(corner_radius=0.12, width=max(g.width + pad, min_width), height=height,
                           stroke_color=color, fill_color=color, fill_opacity=0.3)
    seat(g, 0, 0)
    return VGroup(box, g)


def char_unit(c, x, y, width=0.6, height=0.65, font_size=28, glyph_color=WHITE):
    box = RoundedRectangle(corner_radius=0.1, width=width, height=height, stroke_color=TOKEN_COLOR,
                           fill_color=TOKEN_COLOR, fill_opacity=0.3).move_to([x, y, 0])
    g = seat(btext(c, font_size, glyph_color), x, y)
    return box, g


def stat_line(mark, text, mark_color, font_size=28):
    m = Text(mark, font_size=font_size + 4, color=mark_color)
    t = Text(text, font_size=font_size, color=GREY_B)
    return VGroup(m, t).arrange(RIGHT, buff=0.25)


def header(text):
    return Text(text, font_size=36).to_edge(UP, buff=0.5)


def arrow(start, end, color=GREY_B):
    return Arrow(start, end, buff=0, color=color, stroke_width=4, max_tip_length_to_length_ratio=0.25)


def bracketed_column(entries, font_size=30):
    col = VGroup(*[Text(e, font=MONO, font_size=font_size) for e in entries]).arrange(DOWN, buff=0.22)
    h, w, lip = col.height + 0.3, col.width + 0.5, 0.15
    left = VMobject().set_points_as_corners([[-w / 2 + lip, h / 2, 0], [-w / 2, h / 2, 0],
                                             [-w / 2, -h / 2, 0], [-w / 2 + lip, -h / 2, 0]])
    right = left.copy().flip(UP).shift(w * RIGHT)
    brackets = VGroup(left, right).set_stroke(GREY_B, 3).move_to(col)
    return VGroup(brackets, col)


class TokensVideo(VoicedScene):
    VIDEO = "v02"

    # --- stage helpers ----------------------------------------------------------------------
    def wipe(self, *keep, anims=(), run_time=0.5):
        """Fade out everything on screen except `keep` (searched inside groups too)."""
        keep_set = set()
        for k in keep:
            keep_set.update(k.get_family())
        out = []

        def collect(mobs):
            for m in mobs:
                if m in keep_set:
                    continue
                if keep_set & set(m.get_family()):
                    collect(m.submobjects)
                else:
                    out.append(m)

        collect(self.mobjects)
        all_anims = [FadeOut(m) for m in out] + list(anims)
        if all_anims:
            self.play(*all_anims, run_time=run_time)

    def detach(self, whole, part):
        """Split a slice `part` off `whole` so both are separate top-level mobjects."""
        rest = VGroup(*[m for m in whole.submobjects if m not in part.submobjects])
        self.remove(whole)
        self.add(rest, part)
        return rest

    def construct(self):
        play_token_intro(self, TITLE, 2, TAGLINE)
        self.recap()
        self.text_to_numbers()
        self.characters()
        self.words()
        self.subwords()
        self.bpe_intro()
        self.bpe_example()
        self.bpe_code()
        self.vocabulary()
        self.strawberry()
        self.outro()

    # 1. Recap ---------------------------------------------------------------------------------
    def recap(self):
        self.section(1)
        claim = Text("An LLM predicts the next word", font_size=48).move_to(0.6 * UP)
        word = claim[-4:]
        self.at("llm")
        self.play(Write(claim), run_time=1.3)
        self.detach(claim, word)
        self.at("lie")
        strike = Line(word.get_left() + 0.1 * LEFT, word.get_right() + 0.1 * RIGHT, color=RED, stroke_width=6)
        self.play(Create(strike), word.animate.set_color(GREY_B), run_time=0.5)
        self.at("token")
        tok = Text("token", font_size=48, color=YELLOW).next_to(word, DOWN, buff=0.45)
        self.play(Write(tok), run_time=0.7)
        self.at("token")
        qmark = Text("?", font_size=48, color=YELLOW).next_to(tok, RIGHT, buff=0.12)
        self.play(FadeIn(qmark, scale=1.5), run_time=0.4)
        self.at("words")
        self.play(Indicate(word, color=GREY_A, scale_factor=1.15), run_time=0.6)
        self.end_section()

    # 2. Text must become numbers ----------------------------------------------------------------
    def text_to_numbers(self):
        self.section(2)
        self.wipe(run_time=0.6)
        title = Text("A neural network only understands numbers", font_size=32, color=GREY_B)
        title.to_edge(UP, buff=0.8)
        sent = Text("The cat sat on the", font_size=26)
        block = RoundedRectangle(corner_radius=0.15, width=sent.width + 0.5, height=0.9, stroke_color=GREY_B,
                                 fill_color=GREY_E, fill_opacity=0.5)
        sent.move_to(block)
        text_block = VGroup(block, sent)
        tk_box = RoundedRectangle(corner_radius=0.2, width=2.3, height=1.1, stroke_color=GREY_B,
                                  fill_color=GREY_E, fill_opacity=0.6)
        tk_q = Text("?", font_size=40, color=GREY_B).move_to(tk_box)
        ids = Text("[464, 3797, 3332, 319, 262]", font=MONO, font_size=22)
        a1 = Arrow(ORIGIN, 0.8 * RIGHT, buff=0, color=GREY_B, stroke_width=4)
        a2 = a1.copy()
        pipeline = VGroup(text_block, a1, tk_box, a2, ids).arrange(RIGHT, buff=0.12)
        pipeline.scale_to_fit_width(12.2).move_to(0.2 * DOWN)
        tk_q.move_to(tk_box)
        back = CurvedArrow(ids.get_bottom() + 0.2 * DOWN, block.get_bottom() + 0.15 * DOWN,
                           angle=-TAU / 5, color=GREY_B, stroke_width=4)

        self.at("numbers")
        self.play(Write(title), FadeIn(text_block, shift=0.2 * RIGHT), run_time=0.9)
        self.at("turn")
        self.play(GrowArrow(a1), FadeIn(tk_box), FadeIn(tk_q), run_time=0.7)
        self.at("list")
        self.play(GrowArrow(a2), FadeIn(ids, shift=0.2 * RIGHT), run_time=0.7)
        self.at("back")
        self.play(Create(back), run_time=0.8)
        self.at("tokenizers")
        tk_label = Text("tokenizer", font_size=30, color=YELLOW).move_to(tk_box)
        self.play(tk_box.animate.set_stroke(YELLOW, 5), ReplacementTransform(tk_q, tk_label), run_time=0.6)
        self.end_section()
        self.sent = sent

    # 3. One token per character ------------------------------------------------------------------
    def characters(self):
        self.section(3)
        sent = self.sent
        self.wipe(sent, anims=[sent.animate.scale(28 / 26).move_to(1.3 * UP)], run_time=0.7)
        title = header("Idea 1: one token per character")
        chars = list("The cat sat on the")
        step, width = 0.67, 0.6
        x0 = -(len(chars) * step - (step - width)) / 2 + width / 2
        boxes, glyphs, codes = VGroup(), [], VGroup()
        for k, c in enumerate(chars):
            x = x0 + k * step
            box, g = char_unit(SP if c == " " else c, x, 1.3, width=width,
                               glyph_color=GREY_B if c == " " else WHITE)
            boxes.add(box)
            glyphs.append(g)
            codes.add(Text(str(ord(c)), font=MONO, font_size=20, color=GREY_B).move_to([x, 0.62, 0]))
        visible = [g for c, g in zip(chars, glyphs) if c != " "]
        spaces = [g for c, g in zip(chars, glyphs) if c == " "]

        self.at("one")
        self.play(Write(title), run_time=0.7)
        self.at("character")
        self.play(LaggedStart(*[FadeIn(b) for b in boxes], lag_ratio=0.04),
                  *[ReplacementTransform(sent[j], g[0]) for j, g in enumerate(visible)],
                  *[FadeIn(g) for g in spaces], run_time=1.0)
        self.remove(*[g[0] for g in visible])
        self.add(*visible)
        self.play(LaggedStart(*[FadeIn(c, shift=0.1 * DOWN) for c in codes], lag_ratio=0.05), run_time=0.8)
        vocab = stat_line("✓", "vocabulary: a few hundred symbols", GREEN).move_to([-4.6, -0.4, 0], aligned_edge=LEFT)
        length = stat_line("✗", "18 tokens for 5 words", RED).move_to([-4.6, -1.15, 0], aligned_edge=LEFT)
        self.at("tiny")
        self.play(FadeIn(vocab, shift=0.2 * RIGHT), run_time=0.5)
        self.at("long")
        self.play(FadeIn(length, shift=0.2 * RIGHT), *[Indicate(b, color=RED, scale_factor=1.05) for b in boxes], run_time=0.7)
        self.at("meaning")
        t_unit = VGroup(boxes[6], glyphs[6]).copy()
        t_unit.generate_target()
        t_unit.target.scale(1.5).move_to([-1.3, -2.55, 0])
        meaning = Text("meaning?", font_size=36, color=YELLOW).next_to(t_unit.target, RIGHT, buff=0.45)
        dim = [b.animate.set_stroke(GREY_D).set_fill(GREY_E, 0.3) for b in boxes]
        dim += [g.animate.set_opacity(0.35) for g in glyphs] + [c.animate.set_opacity(0.35) for c in codes]
        self.play(MoveToTarget(t_unit), *dim, run_time=0.8)
        self.play(FadeIn(meaning, shift=0.2 * RIGHT), run_time=0.4)
        self.end_section()

    # 4. One token per word -----------------------------------------------------------------------
    def words(self):
        self.section(4)
        self.wipe()
        title = header("Idea 2: one token per word")
        row = VGroup(*[token(w, font_size=30) for w in SENTENCE]).arrange(RIGHT, buff=0.15).move_to(2.1 * UP)
        short = stat_line("✓", "short sequences: 5 tokens", GREEN).move_to([-4.6, 1.2, 0], aligned_edge=LEFT)
        vocab = stat_line("✗", "vocabulary: 500,000+ words?", RED).move_to([-4.6, 0.45, 0], aligned_edge=LEFT)
        batches = [["cat", "cats", "catnap", "kitty", "kitten", "dog", "dogs", "sat", "sitting", "mat"],
                   ["Catalina", "Kokoro", "Zelda", "Oaxaca", "Luigi"],
                   ["teh", "catz", "recieve", "definately", "thier"],
                   ["yeet", "rizz", "sus", "lol", "bussin"]]
        grid, k = [], 0
        for batch in batches:
            group = VGroup()
            for w in batch:
                x, y = -4.8 + 2.4 * (k % 5), -0.35 - 0.6 * (k // 5)
                group.add(seat(btext(w, 26, YELLOW), x, y))
                k += 1
            grid.append(group)

        self.at("one")
        self.play(Write(title), run_time=0.6)
        self.at("word")
        self.play(LaggedStart(*[FadeIn(t, shift=0.2 * DOWN) for t in row], lag_ratio=0.15), run_time=0.8)
        self.at("short")
        self.play(FadeIn(short, shift=0.2 * RIGHT), run_time=0.5)
        self.at("explodes")
        self.play(FadeIn(vocab, shift=0.2 * RIGHT),
                  LaggedStart(*[FadeIn(w, scale=0.7) for w in grid[0]], lag_ratio=0.12), run_time=1.0)
        self.play(*[w.animate.set_color(GREY_A) for w in grid[0]], run_time=0.3)
        for cue, i in [("name", 1), ("typo", 2), ("slang", 3)]:
            self.at(cue)
            self.play(LaggedStart(*[FadeIn(w, scale=0.7) for w in grid[i]], lag_ratio=0.15),
                      *[w.animate.set_color(GREY_A) for w in grid[i - 1]], run_time=0.6)
        self.at("training")
        new_word = token("catfluencer", font_size=30).move_to([-1.1, -1.5, 0], aligned_edge=RIGHT)
        link = arrow([-0.85, -1.5, 0], [0.45, -1.5, 0])
        unknown = token("[UNKNOWN]", color=RED, font_size=30).move_to([0.7, -1.5, 0], aligned_edge=LEFT)
        self.play(FadeOut(VGroup(*grid)), run_time=0.4)
        self.play(FadeIn(new_word, shift=0.2 * RIGHT), GrowArrow(link), FadeIn(unknown), run_time=0.7)
        self.at("read")
        cross = Text("✗", font_size=60, color=RED).next_to(unknown, RIGHT, buff=0.4)
        self.play(FadeIn(cross, scale=1.6), run_time=0.4)
        self.end_section()

    # 5. Subword tokens ---------------------------------------------------------------------------
    def subwords(self):
        self.section(5)
        self.wipe()
        title = header("Middle road: subword tokens")

        def row(word, pieces, y, count):
            w = seat(btext(word, 32), -2.35, y, edge=RIGHT)
            a = arrow([-2.05, y, 0], [-0.95, y, 0])
            toks = VGroup(*[token(p, font_size=30) for p in pieces]).arrange(RIGHT, buff=0.1)
            toks.move_to([-0.7, y, 0], aligned_edge=LEFT)
            c = Text(count, font_size=26, color=GREY_B).move_to([3.9, y, 0], aligned_edge=LEFT)
            return w, a, toks, c

        the = row("the", ["the"], 1.6, "1 token")
        tkz = row("tokenization", ["token", "ization"], 0.1, "2 tokens")
        cnp = row("catnap", ["cat", "n", "ap"], -1.4, "3 tokens")
        gpt = Text("splits shown: GPT-2's tokenizer", font_size=24, color=GREY_B).move_to(2.75 * DOWN)

        self.at("subword")
        self.play(Write(title), run_time=0.7)
        self.at("common")
        self.play(FadeIn(the[0]), GrowArrow(the[1]), FadeIn(the[2], shift=0.2 * RIGHT), run_time=0.8)
        self.at("token")
        self.play(FadeIn(the[3]), run_time=0.4)
        self.at("gpt")
        self.play(FadeIn(gpt), run_time=0.5)
        self.at("tokenization")
        self.play(FadeIn(tkz[0], shift=0.2 * RIGHT), run_time=0.5)
        self.at("two")
        self.play(GrowArrow(tkz[1]), LaggedStart(*[FadeIn(t, shift=0.2 * RIGHT) for t in tkz[2]], lag_ratio=0.4),
                  FadeIn(tkz[3]), run_time=0.8)
        self.at("catnap")
        self.play(FadeIn(cnp[0], shift=0.2 * RIGHT), GrowArrow(cnp[1]), run_time=0.5)
        for cue, i in [("cat", 0), ("n", 1), ("app", 2)]:
            self.at(cue)
            extra = [FadeIn(cnp[3])] if i == 2 else []
            self.play(FadeIn(cnp[2][i], shift=0.2 * RIGHT), *extra, run_time=0.35)
        self.end_section()

    # 6. Byte pair encoding -----------------------------------------------------------------------
    def bpe_intro(self):
        self.section(6)
        self.wipe()
        title = header("Byte Pair Encoding (BPE)")
        corpus_label = Text("training text", font_size=24, color=GREY_B).move_to([-5.8, 2.35, 0], aligned_edge=LEFT)
        self.rows = []
        units = VGroup()
        for r, w in enumerate(["low", "lowest", "newest", "widest"]):
            y = 1.5 - 0.95 * r
            row = []
            for k, c in enumerate(w):
                box, g = char_unit(c, -5.5 + 0.7 * k, y, width=0.62, font_size=30)
                row.append({"box": box, "glyphs": g, "s": c})
                units.add(VGroup(box, g))
            self.rows.append(row)

        table_label = Text("pair counts", font_size=24, color=GREY_B).move_to([1.3, 2.35, 0], aligned_edge=LEFT)
        table = VGroup()
        for i, (a, b, n) in enumerate([("e", "s", 3), ("s", "t", 3), ("l", "o", 2), ("o", "w", 2), ("w", "e", 2)]):
            y = 1.5 - 0.65 * i
            ta = token(a, font_size=24, height=0.5, min_width=0.48, pad=0.2).move_to([1.55, y, 0])
            tb = token(b, font_size=24, height=0.5, min_width=0.48, pad=0.2).move_to([2.08, y, 0])
            bar = Rectangle(width=0.8 * n, height=0.34, stroke_width=0, fill_color=TEAL, fill_opacity=0.85)
            bar.move_to([2.6, y, 0], aligned_edge=LEFT)
            num = seat(btext(str(n), 26, GREY_A), bar.get_right()[0] + 0.2, y, edge=LEFT)
            table.add(VGroup(ta, tb, bar, num))
        more = Text("… plus 5 pairs seen once", font_size=22, color=GREY_B)
        more.move_to([1.3, -1.8, 0], aligned_edge=LEFT)

        self.at("byte")
        self.play(Write(title), run_time=1.0)
        self.at("characters")
        self.play(FadeIn(corpus_label), LaggedStart(*[FadeIn(u, scale=0.8) for u in units], lag_ratio=0.03),
                  run_time=1.0)
        self.at("count")
        self.play(FadeIn(table_label),
                  LaggedStart(*[AnimationGroup(FadeIn(r[0]), FadeIn(r[1]), GrowFromEdge(r[2], LEFT), FadeIn(r[3]))
                                for r in table], lag_ratio=0.15), FadeIn(more), run_time=1.3)
        self.at("frequent")
        top = SurroundingRectangle(table[0], color=YELLOW, buff=0.1, corner_radius=0.1)
        self.play(Create(top), run_time=0.6)
        self.end_section()
        self.table = VGroup(table_label, table, more, top)

    # 7. Worked example ---------------------------------------------------------------------------
    def pair_marks(self, a, b):
        marks = VGroup()
        for row in self.rows:
            for u, v in zip(row, row[1:]):
                if u["s"] == a and v["s"] == b:
                    marks.add(SurroundingRectangle(VGroup(u["box"], v["box"]), color=YELLOW, buff=0.07,
                                                   corner_radius=0.14))
        return marks

    def fuse(self, a, b):
        """Animations that glue every neighbouring (a, b) pair in the corpus into one token."""
        anims, fused = [], []
        for row in self.rows:
            i = 0
            while i < len(row) - 1:
                u, v = row[i], row[i + 1]
                if u["s"] == a and v["s"] == b:
                    left, right = u["box"].get_left()[0], v["box"].get_right()[0]
                    y = u["box"].get_y()
                    nb = RoundedRectangle(corner_radius=0.1, width=right - left, height=u["box"].height,
                                          stroke_color=YELLOW, fill_color=TOKEN_COLOR, fill_opacity=0.3)
                    nb.move_to([(left + right) / 2, y, 0])
                    target = seat(btext(a + b, 30), nb.get_x(), y)
                    old = [*u["glyphs"], *v["glyphs"]]
                    anims += [Transform(u["box"], nb), FadeOut(v["box"])]
                    anims += [g.animate.move_to(t) for g, t in zip(old, target)]
                    merged = {"box": u["box"], "glyphs": VGroup(*old), "s": a + b}
                    row[i:i + 2] = [merged]
                    fused.append(merged)
                i += 1
        return anims, fused

    def bpe_example(self):
        self.section(7)
        merge_label = Text("merges", font_size=24, color=GREY_B).move_to([1.3, 2.35, 0], aligned_edge=LEFT)
        entries = [Text(t, font_size=30, t2c={r: YELLOW}).move_to([1.3, 1.5 - 0.65 * i, 0], aligned_edge=LEFT)
                   for i, (t, r) in enumerate([("1.  e + s → es", "es"), ("2.  es + t → est", "est"),
                                               ("3.  l + o → lo", "lo"), ("4.  lo + w → low", "low")])]

        self.at("e")
        marks = self.pair_marks("e", "s")
        self.play(Create(marks), run_time=0.6)
        self.at("three")
        self.play(Indicate(self.table[1][0][3], color=YELLOW, scale_factor=1.5), run_time=0.6)
        self.at("so")
        self.play(FadeOut(self.table), run_time=0.4)
        self.at("merge")
        anims, last = self.fuse("e", "s")
        self.play(*anims, FadeOut(marks), FadeIn(merge_label), FadeIn(entries[0], shift=0.2 * RIGHT),
                  run_time=0.9)
        self.at("t")
        marks = self.pair_marks("es", "t")
        self.play(Create(marks), run_time=0.5)
        self.at("merge")
        for (a, b), entry in [(("es", "t"), entries[1]), (("l", "o"), entries[2]), (("lo", "w"), entries[3])]:
            if a == "l":
                self.at("l")
            elif a == "lo":
                self.at("w")
            reset = [u["box"].animate.set_stroke(TOKEN_COLOR) for u in last]
            anims, last = self.fuse(a, b)
            self.play(*reset, *anims, FadeIn(entry, shift=0.2 * RIGHT), *([FadeOut(marks)] if marks is not None else []),
                      run_time=0.8)
            marks = None
        self.at("low")
        low = [u for row in self.rows for u in row if u["s"] == "low"]
        est = [u for row in self.rows for u in row if u["s"] == "est"]
        self.play(*[u["box"].animate.set_stroke(YELLOW, 5) for u in low],
                  Indicate(entries[3][-3:], color=YELLOW), run_time=0.6)
        self.at("est")
        self.play(*[u["box"].animate.set_stroke(TOKEN_COLOR, 4) for u in low],
                  *[u["box"].animate.set_stroke(YELLOW, 5) for u in est],
                  Indicate(entries[1][-3:], color=YELLOW), run_time=0.6)
        self.end_section()

    # 8. BPE training in code ---------------------------------------------------------------------
    def bpe_code(self):
        self.section(8)
        code = Code(code_string=BPE_CODE, language="python", formatter_style="monokai",
                    background="window", paragraph_config={"font_size": 21})
        if code.width > 12.8:  # font 21 is ~13.4 wide; this keeps it at an effective ~20 pt
            code.scale_to_fit_width(12.8)
        code.move_to(0.35 * UP)
        row_h = abs(code.line_numbers[0].get_y() - code.line_numbers[1].get_y())
        hl = Rectangle(width=code.code_lines.width + 0.25, height=row_h, color=YELLOW, stroke_width=2)
        hl.align_to(code.code_lines, LEFT).shift(0.12 * LEFT).match_y(code.line_numbers[3])
        code.code_lines.set_z_index(1)  # draw text over the highlight so its edge never hides "_"
        note = Text("num_merges: tens of thousands", font_size=26, color=GREY_B)
        note.next_to(code, DOWN, buff=0.4)

        self.at("code")
        self.wipe(anims=[FadeIn(code, shift=0.3 * UP)], run_time=0.8)
        self.at("count")
        self.play(Create(hl), run_time=0.5)
        for cue, line in [("pick", 4), ("merge", 5), ("repeat", 2)]:
            self.at(cue)
            self.play(hl.animate.match_y(code.line_numbers[line]), run_time=0.4)
        self.at("thousands")
        self.play(FadeIn(note, shift=0.2 * UP), run_time=0.5)
        self.end_section()

    # 9. Vocabulary and IDs -----------------------------------------------------------------------
    def vocabulary(self):
        self.section(9)
        self.wipe()
        title = Text("vocabulary", font_size=34).move_to([-5.9, 3.2, 0], aligned_edge=LEFT)
        head = VGroup(Text("ID", font_size=22, color=GREY_B).move_to([-3.4, 2.5, 0], aligned_edge=RIGHT),
                      Text("token", font_size=22, color=GREY_B).move_to([-2.9, 2.5, 0], aligned_edge=LEFT))
        id_col, tok_col = VGroup(), VGroup()
        for i, (n, s) in enumerate([(262, SP + "the"), (319, SP + "on"), (464, "The"),
                                    (3332, SP + "sat"), (3797, SP + "cat"), (9246, "cat")]):
            y = 1.9 - 0.6 * i
            id_col.add(seat(btext(str(n), 26, GREY_A, font=MONO), -3.4, y, edge=RIGHT))
            tok_col.add(token(s, font_size=24, height=0.5, pad=0.3).move_to([-2.9, y, 0], aligned_edge=LEFT))
        dots = Text("⋮", font_size=30, color=GREY_B).move_to([-3.1, -1.75, 0])
        table = VGroup(title, head, id_col, tok_col, dots)
        gpt2 = Text("GPT-2: 50,257 tokens", font_size=34, t2c={"50,257": YELLOW})
        gpt2.move_to([0.6, 1.2, 0], aligned_edge=LEFT)
        newer = Text("newer models: 100,000+", font_size=34, t2c={"100,000+": YELLOW})
        newer.move_to([0.6, 0.2, 0], aligned_edge=LEFT)

        self.at("vocabulary")
        self.play(FadeIn(title), FadeIn(head),
                  LaggedStart(*[FadeIn(VGroup(a, b), shift=0.15 * DOWN) for a, b in zip(id_col, tok_col)],
                              lag_ratio=0.12), FadeIn(dots), run_time=1.1)
        self.at("50")
        self.play(FadeIn(gpt2, shift=0.2 * RIGHT), run_time=0.6)
        self.at("100")
        self.play(FadeIn(newer, shift=0.2 * RIGHT), run_time=0.6)
        self.at("id")
        self.play(Indicate(id_col, color=YELLOW, scale_factor=1.1), run_time=0.7)

        plain = VGroup(*[token(w, font_size=30) for w in SENTENCE]).arrange(RIGHT, buff=0.35).move_to(1.3 * UP)
        spaced = VGroup(*[token(("" if i == 0 else SP) + w, font_size=30, space_color=YELLOW)
                          for i, w in enumerate(SENTENCE)]).arrange(RIGHT, buff=0.3).move_to(1.3 * UP)
        ids = VGroup(*[Text(str(n), font=MONO, font_size=28, color=YELLOW).next_to(t, DOWN, buff=0.35)
                       for n, t in zip(SENTENCE_IDS, plain)])
        self.at("sentence")
        self.wipe(run_time=0.4)
        self.play(LaggedStart(*[FadeIn(t, shift=0.2 * DOWN) for t in plain], lag_ratio=0.1), run_time=0.6)
        self.at("list")
        self.play(LaggedStart(*[FadeIn(n, shift=0.2 * DOWN) for n in ids], lag_ratio=0.1), run_time=0.7)
        self.at("spaces")
        anims = []
        for i, (p, s) in enumerate(zip(plain, spaced)):
            anims.append(Transform(p[0], s[0]))
            anims += [g.animate.move_to(t) for g, t in zip(p[1], s[1][-len(p[1]):])]
            if i > 0:
                anims.append(FadeIn(s[1][0], scale=1.5))
            anims.append(ids[i].animate.next_to(s[0], DOWN, buff=0.35))
        self.play(*anims, run_time=0.8)

        def compare(s, n, y):
            t = token(s, font_size=30, space_color=YELLOW).move_to([-0.7, y, 0], aligned_edge=RIGHT)
            a = arrow([-0.45, y, 0], [0.65, y, 0])
            num = Text(n, font=MONO, font_size=30, color=YELLOW).move_to([0.9, y, 0], aligned_edge=LEFT)
            return VGroup(t, a, num)

        self.at("cat")
        c1 = compare(SP + "cat", "3797", -1.3)
        self.play(FadeIn(c1, shift=0.2 * UP), run_time=0.6)
        self.at("cat")
        c2 = compare("cat", "9246", -2.3)
        self.play(FadeIn(c2, shift=0.2 * UP), run_time=0.6)
        self.end_section()

    # 10. Strawberry ------------------------------------------------------------------------------
    def strawberry(self):
        self.section(10)
        self.wipe()
        question = Text("How many r's are in strawberry?", font_size=34).move_to(3.0 * UP)
        pieces = ["How", SP + "many", SP + "r", "'s", SP + "are", SP + "in", SP + "strawberry", "?"]
        toks = VGroup(*[token(p, font_size=28) for p in pieces]).arrange(RIGHT, buff=0.1).move_to(1.9 * UP)
        straw = toks[6]
        letters = "strawberry"
        step, width = 0.64, 0.56
        x0 = -(len(letters) * step - (step - width)) / 2 + width / 2
        lboxes, lglyphs = VGroup(), VGroup()
        for k, c in enumerate(letters):
            box, g = char_unit(c, x0 + k * step, 0.2, width=width, height=0.6, font_size=30,
                               glyph_color=YELLOW if c == "r" else WHITE)
            box.set_stroke(GREY_C).set_fill(GREY_E, 0.4)
            lboxes.add(box)
            lglyphs.add(g)
        cover_label = btext(SP + "strawberry", 40)
        cover_label[0].set_color(GREY_B)
        cover_box = RoundedRectangle(corner_radius=0.15, width=lboxes.width + 0.5, height=1.05,
                                     stroke_color=TOKEN_COLOR, fill_color=interpolate_color(BLACK, TOKEN_COLOR, 0.3),
                                     fill_opacity=0.93).move_to(lboxes)
        seat(cover_label, cover_box.get_x(), cover_box.get_y())
        cover = VGroup(cover_box, cover_label)
        caption = Text("one token  ·  ID 41236", font_size=30, t2c={"41236": YELLOW}).move_to(1.2 * DOWN)

        self.at("ask")
        self.play(Write(question), run_time=0.9)
        self.at("strawberry")
        self.play(LaggedStart(*[FadeIn(t, shift=0.2 * DOWN) for t in toks], lag_ratio=0.08), run_time=0.8)
        self.at("letters")
        self.play(FadeIn(lboxes),
                  *[ReplacementTransform(src.copy(), dst[0]) for src, dst in zip(straw[1][1:], lglyphs)],
                  run_time=0.9)
        self.remove(*[g[0] for g in lglyphs])
        self.add(lglyphs)
        self.at("gpt")
        self.play(ReplacementTransform(straw.copy(), cover), lglyphs.animate.set_opacity(0.4), run_time=0.8)
        self.at("single")
        self.play(cover_box.animate.set_stroke(YELLOW, 5), straw[0].animate.set_stroke(YELLOW, 5),
                  FadeIn(caption, shift=0.2 * UP), run_time=0.6)
        self.end_section()

    # 11. Outro -----------------------------------------------------------------------------------
    def outro(self):
        self.section(11)
        self.wipe()
        lst = Text("[464, 3797, 3332, 319, 262]", font=MONO, font_size=36).move_to(0.3 * UP)
        num = lst[5:9]
        self.at("list")
        self.play(FadeIn(lst, shift=0.2 * UP), run_time=0.6)
        self.at("3797")
        rest = self.detach(lst, num)
        self.play(FadeOut(rest), num.animate.scale(2.2).move_to([-3.0, 0.3, 0]),
                  run_time=0.8)
        w, h, tip = num.width / 2 + 0.45, num.height / 2 + 0.45, 0.6
        c = num.get_center()
        tag = Polygon(*[c + np.array(p) for p in [(-w, h, 0), (w, h, 0), (w, -h, 0), (-w, -h, 0), (-w - tip, 0, 0)]],
                      color=YELLOW, stroke_width=4).round_corners(0.12)
        hole = Circle(radius=0.1, color=YELLOW, stroke_width=4).move_to(c + np.array([-w - tip + 0.35, 0, 0]))
        self.at("label")
        self.play(Create(tag), Create(hole), run_time=0.8)
        cat_q = Text("cat?", font_size=48, color=GREY_B).move_to([2.2, 0.3, 0])
        self.at("cat")
        self.play(FadeIn(cat_q, shift=0.2 * LEFT), run_time=0.5)
        vec = bracketed_column(["0.12", "−0.53", "0.88", "0.07", "⋮"]).move_to([2.8, 0.3, 0])
        link = arrow([tag.get_right()[0] + 0.25, 0.3, 0], [vec.get_left()[0] - 0.25, 0.3, 0])
        self.at("vector")
        self.play(FadeOut(cat_q), GrowArrow(link), FadeIn(vec, shift=0.3 * RIGHT), run_time=0.8)

        next_label = Text("Next up", font_size=30, color=GREY_B)
        next_title = Text("Embeddings: Words as Vectors", font_size=44)
        card = VGroup(next_label, next_title).arrange(DOWN, buff=0.35)
        self.at("embeddings")
        self.wipe(run_time=0.4)
        self.play(FadeIn(card, shift=0.2 * UP), run_time=0.5)
        self.end_section()
        self.wait(1.0)
        self.play(FadeOut(card))
        self.wait(0.5)
