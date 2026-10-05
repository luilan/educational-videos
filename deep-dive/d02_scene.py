"""How LLMs Work: Deep Dive, episode 2 — Bytes, Unicode, and Why Strawberry Is Hard.

Render from the repo root:  ./render.sh deep-dive d02
Every byte, token count, token split and model answer on screen comes from code/d02_bytes_unicode/bytes_unicode.py
(GPT-2 and Qwen2.5 tokenizers; Qwen2.5-1.5B-Instruct, greedy). Emoji are drawn from the Noto Color Emoji font.
"""
import os
import tempfile

from manim import *
from PIL import Image, ImageDraw, ImageFont

from common import MODEL_COLOR, MONO, TOKEN_COLOR, code_panel, finish, highlight, next_up_card, token
from intro import play_token_intro
from d02_script import NEXT, TAGLINE, TITLE
from voiced_scene import VoicedScene

SERIES_LABEL = "HOW LLMs WORK: DEEP DIVE  ·  EPISODE 2"
EMOJI_FONT = "/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf"
CHARS = [("a", "U+0061", ["01100001"]), ("é", "U+00E9", ["11000011", "10101001"]),
         ("ж", "U+0436", ["11010000", "10110110"]), ("中", "U+4E2D", ["11100100", "10111000", "10101101"]),
         ("🍓", "U+1F353", ["11110000", "10011111", "10001101", "10010011"])]
LANGS = [("English", "The cat is sleeping on the warm windowsill.", 43, 10, 10),
         ("Italian", "Il gatto dorme sul davanzale caldo.", 35, 13, 12),
         ("Russian", "Кошка спит на тёплом подоконнике.", 61, 38, 16),
         ("Chinese", "猫在温暖的窗台上睡觉。", 33, 25, 8),
         ("Emoji", "🐱💤🪟☀️", 18, 12, 7)]
SPELL = "Sure! Here is the spelling of \"strawberry\" letter by letter:\n\ns - t - r - o - w - a - b - e\n\n… There is 1 'r' at the beginning."
CODE = """for ch in ["a", "é", "中", "\\U0001F353"]:      # the last one is the strawberry emoji
    print(ch, len(ch.encode("utf-8")), "bytes")

tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
pieces = [tok.decode([i]) for i in tok("strawberry")["input_ids"]]   # str | aw | berry
n_tokens = len(tok("Кошка спит на тёплом подоконнике.")["input_ids"])   # 16"""


def emoji(ch, height=0.6):
    """An emoji as an image (Text cannot draw colour emoji)."""
    path = os.path.join(tempfile.gettempdir(), f"emoji_{ord(ch[0]):x}.png")
    if not os.path.exists(path):
        font = ImageFont.truetype(EMOJI_FONT, 109)
        im = Image.new("RGBA", (136, 128), (0, 0, 0, 0))
        ImageDraw.Draw(im).text((0, 0), ch, font=font, embedded_color=True)
        im.crop(im.getbbox()).save(path)
    return ImageMobject(path).scale_to_fit_height(height)


def glyph(ch, font_size=40):
    code = ord(ch[0])
    is_emoji = code >= 0x1F000 or 0x2600 <= code < 0x2800        # emoji blocks (not CJK, which Text can draw)
    return emoji(ch, font_size / 60) if is_emoji else Text(ch, font_size=font_size)


def labeled_box(text, color=MODEL_COLOR, width=2.4, height=1.0, font_size=26):
    label = Text(text, font_size=font_size, line_spacing=0.8)
    box = RoundedRectangle(corner_radius=0.2, width=max(width, label.width + 0.5), height=height, stroke_color=color,
                           fill_color=color, fill_opacity=0.25)
    return VGroup(box, label.move_to(box))


def bubble(text, color, font_size=20):
    label = Text(text, font_size=font_size, line_spacing=0.8)
    box = RoundedRectangle(corner_radius=0.18, width=label.width + 0.5, height=label.height + 0.4, stroke_width=0,
                           fill_color=color, fill_opacity=0.88)
    return VGroup(box, label.move_to(box))


class BytesUnicodeVideo(VoicedScene):
    VIDEO = "d02"

    def construct(self):
        play_token_intro(self, TITLE, 2, TAGLINE, label=SERIES_LABEL)
        self.hook()          # 1
        self.unicode()       # 2
        self.utf8()          # 3
        self.cost()          # 4
        self.strawberry()    # 5
        self.test()          # 6
        self.spelling()      # 7
        self.lessons()       # 8
        self.code()          # 9
        self.outro()         # 10

    def clear_stage(self, *keep, run_time=0.6):
        self.play(*[FadeOut(m) for m in self.mobjects if m not in keep], run_time=run_time)

    # ------------------------------------------------------------------ 1. hook
    def hook(self):
        self.section(1)
        q = bubble("How many times does the letter r appear in strawberry?", BLUE_E, 28).move_to([0, 1.4, 0])
        berry = emoji("🍓", 1.4).move_to([0, -0.6, 0])
        self.at("strawberry")
        self.play(FadeIn(q, shift=0.2 * DOWN), FadeIn(berry), run_time=0.6)
        fam = Text("a famous way to trip up language models", font_size=26, color=GREY_B).move_to([0, -2.0, 0])
        self.at("famous")
        self.play(FadeIn(fam), run_time=0.4)
        no = Text("it doesn't read letters", font_size=34, color=YELLOW).move_to([0, -2.8, 0])
        self.at("doesnt")
        self.play(FadeIn(no, shift=0.2 * UP), run_time=0.5)
        self.end_section()

    # ------------------------------------------------------------------ 2. Unicode
    def unicode(self):
        self.section(2)
        self.clear_stage()
        head = Text("Unicode: every character has a number (a code point)", font_size=30).to_edge(UP, buff=0.6)
        self.at("unicode")
        self.play(FadeIn(head), run_time=0.5)
        cols = Group()
        for ch, cp, _ in CHARS:
            cols.add(Group(glyph(ch, 60), Text(cp, font=MONO, font_size=24, color=YELLOW)).arrange(DOWN, buff=0.35))
        cols.arrange(RIGHT, buff=1.0).move_to([0, 0.3, 0])
        dec = Text("= 97", font=MONO, font_size=22, color=GREY_B).next_to(cols[0], DOWN, buff=0.2)
        self.at("97")
        self.play(FadeIn(cols[0]), FadeIn(dec), run_time=0.5)
        self.at("accent")
        self.play(LaggedStart(*[FadeIn(c, shift=0.2 * UP) for c in cols[1:]], lag_ratio=0.25), run_time=1.2)
        mil = Text("more than a million possible code points", font_size=26, color=GREY_A).move_to([0, -2.2, 0])
        self.at("million")
        self.play(FadeIn(mil), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 3. UTF-8
    def utf8(self):
        self.section(3)
        self.clear_stage()
        head = Text("UTF-8: one to four bytes per character", font_size=30).to_edge(UP, buff=0.5)
        self.at("utf")
        self.play(FadeIn(head), run_time=0.4)
        rows = Group()
        for ch, _, bits in CHARS:
            bytes_ = VGroup()
            for b in bits:
                lead = len(b) - len(b.lstrip("1")) + 1 if b.startswith("1") else 1
                t = Text(b, font=MONO, font_size=22)
                t[:lead].set_color(YELLOW)                     # the leading bits say how many bytes follow
                box = SurroundingRectangle(t, buff=0.08, color=GREY_B, stroke_width=1.5)
                bytes_.add(VGroup(box, t))
            bytes_.arrange(RIGHT, buff=0.15)
            rows.add(Group(glyph(ch, 36), bytes_, Text(f"{len(bits)} byte" + ("s" if len(bits) > 1 else ""),
                                                        font_size=22, color=GREY_B)))
        for k, r in enumerate(rows):
            y = 2.0 - k * 0.85
            r[0].move_to([-5.8, y, 0])
            r[1].move_to([-4.9, y, 0], aligned_edge=LEFT)
            r[2].move_to([5.6, y, 0])
        for k, cue in [(0, "plain"), (1, "accented"), (3, "chinese"), (4, "emoji")]:
            self.at(cue)
            anims = [FadeIn(rows[k], shift=0.2 * RIGHT)]
            if k == 1:
                anims.append(FadeIn(rows[2], shift=0.2 * RIGHT))
            self.play(*anims, run_time=0.45)
        lead = Text("yellow bits: how many bytes belong together", font_size=24, color=YELLOW).move_to([0, -2.4, 0])
        self.at("bits")
        self.play(FadeIn(lead), run_time=0.4)
        any_ = Text("so 256 byte tokens can spell any language", font_size=26, color=GREEN_B).next_to(lead, DOWN, 0.2)
        self.at("256")
        self.play(FadeIn(any_), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 4. cost per language
    def cost(self):
        self.section(4)
        self.clear_stage()
        head = Text("one sentence, five languages: how many tokens?", font_size=30).to_edge(UP, buff=0.4)
        self.at("languages")
        self.play(FadeIn(head), run_time=0.4)
        hdr = VGroup(*[Text(t, font_size=20, color=GREY_B) for t in ["language", "bytes", "GPT-2", "Qwen2.5"]])
        xs = [-5.9, 2.2, 3.8, 5.6]
        for h, x in zip(hdr, xs):
            h.move_to([x, 2.5, 0])
        self.play(FadeIn(hdr), run_time=0.3)
        rows = Group()
        for k, (lang, s, nb, g, q) in enumerate(LANGS):
            y = 1.8 - k * 0.75
            if lang == "Emoji":
                sent = Group(*[emoji(c, 0.42) for c in ["🐱", "💤", "🪟", "☀️"]]).arrange(RIGHT, buff=0.12)
            else:
                sent = Text(s, font_size=22)
                if sent.width > 5.9:
                    sent.scale_to_fit_width(5.9)
                elif sent.height < 0.3:
                    sent.scale(1.15)
            name = Text(lang, font_size=22, color=GREY_B)
            cells = Group(name, sent, Text(str(nb), font=MONO, font_size=24),
                          Text(str(g), font=MONO, font_size=24, color=RED_B if g > 20 else WHITE),
                          Text(str(q), font=MONO, font_size=24, color=GREEN_B))
            name.move_to([-6.4, y, 0], aligned_edge=LEFT)
            sent.move_to([-5.1, y, 0], aligned_edge=LEFT)
            for c, x in zip(cells[2:], xs[1:]):
                c.move_to([x, y, 0])
            rows.add(cells)
        self.play(*[FadeIn(r[:3]) for r in rows], run_time=0.6)
        for k, cue in [(0, "10"), (2, "38"), (3, "25")]:
            self.at(cue)
            self.play(FadeIn(rows[k][3], scale=1.3), run_time=0.35)
        self.play(FadeIn(rows[1][3]), FadeIn(rows[4][3]), run_time=0.3)
        note = Text("GPT-2 learned its merges mostly from English", font_size=22, color=GREY_A).move_to([0, -2.4, 0])
        self.at("merges")
        self.play(FadeIn(note), run_time=0.4)
        self.at("kwen")
        self.play(*[FadeIn(r[4]) for r in rows], run_time=0.6)
        self.at("16")
        self.play(Indicate(rows[2][4], color=GREEN), run_time=0.5)
        self.at("eight")
        self.play(Indicate(rows[3][4], color=GREEN), run_time=0.5)
        price = Text("same meaning, very different price and context use", font_size=24, color=YELLOW)
        price.next_to(note, DOWN, buff=0.2)
        self.at("price")
        self.play(FadeIn(price), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 5. strawberry tokens
    def strawberry(self):
        self.section(5)
        self.clear_stage()
        berry = emoji("🍓", 1.0).move_to([-5.4, 1.8, 0])
        self.at("strawberry")
        self.play(FadeIn(berry), run_time=0.3)
        one = VGroup(Text("“ strawberry” (with a space):", font_size=26), token("␣strawberry", TOKEN_COLOR, 28))
        one.arrange(RIGHT, buff=0.4).move_to([0.6, 1.8, 0])
        self.at("single")
        self.play(FadeIn(one), run_time=0.5)
        three = VGroup(Text("“strawberry” (no space):", font_size=26),
                       VGroup(*[token(t, TOKEN_COLOR, 28) for t in ["str", "aw", "berry"]]).arrange(RIGHT, buff=0.1))
        three.arrange(RIGHT, buff=0.4).move_to([0.6, 0.6, 0])
        self.at("three")
        self.play(FadeIn(three), run_time=0.5)
        letters = VGroup(*[Text(c, font=MONO, font_size=34) for c in "strawberry"]).arrange(RIGHT, buff=0.18)
        letters.move_to([0, -0.9, 0])
        hidden = Text("10 letters the model never sees", font_size=26, color=RED_B).next_to(letters, DOWN, buff=0.3)
        self.at("never")
        self.play(FadeIn(letters), FadeIn(hidden), run_time=0.5)
        self.play(letters.animate.set_opacity(0.2), run_time=0.4)
        ids = Text("it sees token numbers, and must have learned how each is spelled", font_size=24, color=YELLOW)
        ids.move_to([0, -2.6, 0])
        self.at("numbers")
        self.play(FadeIn(ids), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 6. the test
    def test(self):
        self.section(6)
        self.clear_stage()
        head = Text("Qwen2.5-1.5B-Instruct: count the letter", font_size=30).to_edge(UP, buff=0.5)
        self.at("asked")
        self.play(FadeIn(head), run_time=0.4)
        data = [("strawberry", "r", 3, 3, ["␣strawberry"]), ("bookkeeper", "e", 3, 3, ["␣book", "keeper"]),
                ("mississippi", "s", 4, 4, ["␣miss", "issippi"]), ("nevertheless", "e", 4, 3, ["␣nevertheless"])]
        rows = VGroup()
        for k, (w, l, true, said, toks) in enumerate(data):
            ok = true == said
            row = VGroup(Text(f"{w}, letter {l}", font_size=24),
                         Text(f"true {true}", font=MONO, font_size=24),
                         Text(f"model {said} {'✓' if ok else '✗'}", font=MONO, font_size=24, color=GREEN if ok else RED),
                         VGroup(*[token(t, GREY_B, 20) for t in toks]).arrange(RIGHT, buff=0.08))
            y = 1.5 - k * 0.8
            row[0].move_to([-6.2, y, 0], aligned_edge=LEFT)
            row[1].move_to([-1.0, y, 0])
            row[2].move_to([1.3, y, 0])
            row[3].move_to([3.6, y, 0], aligned_edge=LEFT)
            rows.add(row)
        self.at("three")
        self.play(FadeIn(rows[0][:3]), run_time=0.4)
        fam = Text("famous question: possibly memorised", font_size=22, color=GREY_B).move_to([0, -2.0, 0])
        self.at("famous")
        self.play(FadeIn(fam), run_time=0.4)
        self.at("bookkeeper")
        self.play(FadeIn(rows[1][:3]), FadeIn(rows[2][:3]), run_time=0.5)
        self.at("nevertheless")
        self.play(FadeIn(rows[3][:3]), run_time=0.4)
        self.at("never")
        self.play(*[FadeIn(r[3]) for r in rows], run_time=0.6)
        self.end_section()

    # ------------------------------------------------------------------ 7. spelling it out
    def spelling(self):
        self.section(7)
        self.clear_stage()
        q = Text("“Spell the word strawberry letter by letter, then count the r's.”", font_size=24, color=BLUE_B)
        q.to_edge(UP, buff=0.6)
        self.at("spell")
        self.play(FadeIn(q), run_time=0.4)
        a = bubble(SPELL, RED_E, 22).move_to([0, 1.4, 0])
        self.at("wrote")
        self.play(FadeIn(a, shift=0.2 * UP), run_time=0.6)
        bad = Text("misspelled: s-t-r-o-w-a-b-e, and counted one r", font_size=26, color=RED_B).next_to(a, DOWN, 0.3)
        self.at("one")
        self.play(FadeIn(bad), run_time=0.4)
        spaced = VGroup(*[token(c, TOKEN_COLOR, 24) for c in ["s", "␣t", "␣r", "␣a", "␣w", "␣b", "␣e", "␣r", "␣r", "␣y"]])
        spaced.arrange(RIGHT, buff=0.08).move_to([0, -1.85, 0])
        lab = Text("one token per letter", font_size=22, color=GREY_B).next_to(spaced, UP, buff=0.15)
        self.at("spaces")
        self.play(FadeIn(spaced), FadeIn(lab), run_time=0.5)
        four = Text("answer: 4 ✗", font=MONO, font_size=30, color=RED).next_to(spaced, DOWN, buff=0.35)
        self.at("four")
        self.play(FadeIn(four, scale=1.2), run_time=0.4)
        struggle = Text("seeing the letters is necessary, not sufficient", font_size=24, color=YELLOW).to_edge(DOWN, buff=0.2)
        self.at("struggles")
        self.play(FadeIn(struggle), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 8. lessons
    def lessons(self):
        self.section(8)
        self.clear_stage()
        head = Text("tokens hide letters, so these are hard:", font_size=30).to_edge(UP, buff=0.6)
        self.at("hide")
        self.play(FadeIn(head), run_time=0.4)
        hard = VGroup(*[token(t, RED_C, 26) for t in ["spelling", "counting letters", "rhyming", "reversing a word",
                                                       "odd number chunks"]]).arrange_in_grid(2, 3, buff=0.35)
        hard.move_to([0, 0.8, 0])
        self.at("spelling")
        self.play(LaggedStart(*[FadeIn(h) for h in hard[:4]], lag_ratio=0.15), run_time=0.8)
        self.at("numbers")
        self.play(FadeIn(hard[4]), run_time=0.3)
        fixes = VGroup(Text("fixes:", font_size=28, color=GREEN_B), token("call a tool (a short piece of code)", GREEN_C, 24),
                       token("a larger model", GREEN_C, 24), token("check", GREEN_C, 24)).arrange(RIGHT, buff=0.3)
        fixes.move_to([0, -1.5, 0])
        self.at("tool")
        self.play(FadeIn(fixes[:2]), run_time=0.4)
        self.at("larger")
        self.play(FadeIn(fixes[2:]), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 9. code
    def code(self):
        self.section(9)
        self.clear_stage()
        code, hl = code_panel(CODE, font_size=20)
        code.move_to([0, 0.3, 0])
        self.play(FadeIn(code, shift=0.2 * UP), run_time=0.5)
        hl.match_y(code.line_numbers[1])
        self.at("bytes")
        self.play(Create(hl), run_time=0.3)
        self.at("tokenizer")
        self.play(highlight(hl, code, 4), run_time=0.4)
        self.at("count")
        self.play(highlight(hl, code, 5), run_time=0.4)
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
