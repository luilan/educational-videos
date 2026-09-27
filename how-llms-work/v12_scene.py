"""Video 12 — Build a Tiny GPT.

Render from the repo root:  ./render.sh how-llms-work v12
"""
import json
import os
import re

import numpy as np
from manim import *

from common import MODEL_COLOR, MONO, TOKEN_COLOR, code_panel, finish, highlight, next_up_card
from intro import play_token_intro
from v12_script import NEXT, TAGLINE, TITLE
from v14_script import TITLE as BONUS_2
from voiced_scene import VoicedScene

# ---------------------------------------------------------------------------- the real run
TINY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tiny_gpt")
with open(os.path.join(TINY, "input.txt")) as f:
    TEXT = f.read()
CHARS = sorted(set(TEXT))                                  # the vocabulary (65 symbols)
STOI = {c: i for i, c in enumerate(CHARS)}
with open(os.path.join(TINY, "training_log.json")) as f:
    LOG = json.load(f)
PARAMS = LOG["params"]                                     # 818,241
VAL = {int(step): loss for step, loss in LOG["val"]}
SAMPLES = {int(k): v.lstrip("\n").split("\n") for k, v in LOG["samples"].items()}
with open(os.path.join(TINY, "tiny_gpt.py")) as f:
    _SRC = [line.rstrip() for line in f]
_SRC_OK = set(_SRC) | {re.sub(r"\s+#.*$", "", line) for line in _SRC}


def excerpt(src, indent=0):
    """Assert every shown line is verbatim from tiny_gpt.py (after re-indenting); '...' marks skipped lines."""
    for line in src.split("\n"):
        if line.strip() in ("", "..."):
            continue
        assert " " * indent + line in _SRC_OK, line
    return src


TOKENIZER = excerpt('''chars = sorted(set(text))
vocab_size = len(chars)
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for c, i in stoi.items()}
encode = lambda s: [stoi[c] for c in s]
decode = lambda ids: "".join(itos[i] for i in ids)''')

TINYGPT = excerpt('''class TinyGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block() for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx):
        x = self.tok_emb(idx) + self.pos_emb(torch.arange(idx.shape[1]))
        return self.head(self.ln_f(self.blocks(x)))''')

ATTN_FORWARD = excerpt('''def forward(self, x):
    B, T, C = x.shape
    q, k, v = self.qkv(x).split(n_embd, dim=2)
    q, k, v = (t.view(B, T, n_head, C // n_head).transpose(1, 2) for t in (q, k, v))
    att = q @ k.transpose(-2, -1) / (k.size(-1) ** 0.5)
    att = att.masked_fill(~self.mask[:T, :T], float("-inf"))
    att = F.softmax(att, dim=-1)
    y = (att @ v).transpose(1, 2).reshape(B, T, C)
    return self.proj(y)''', indent=4)

ATTN_INIT = excerpt('''self.qkv = nn.Linear(n_embd, 3 * n_embd)
self.proj = nn.Linear(n_embd, n_embd)''', indent=8)

MLP_CODE = excerpt('''class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.up = nn.Linear(n_embd, 4 * n_embd)
        self.down = nn.Linear(4 * n_embd, n_embd)

    def forward(self, x):
        return self.down(F.gelu(self.up(x)))''')

BLOCK_CODE = excerpt('''class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1, self.attn = nn.LayerNorm(n_embd), Attention()
        self.ln2, self.mlp = nn.LayerNorm(n_embd), MLP()

    def forward(self, x):
        x = x + self.attn(self.ln1(x))                      # tokens talk
        x = x + self.mlp(self.ln2(x))                       # each token thinks
        return x''')

HYPER = excerpt('''block_size, batch_size = 64, 32
n_embd, n_head, n_layer = 128, 4, 4''')

TRAIN = excerpt('''opt = torch.optim.AdamW(model.parameters(), lr=lr)
...
for step in range(max_steps + 1):
    ...
    x, y = get_batch(train_data)
    logits = model(x)                                       # forward pass
    loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
    opt.zero_grad()
    loss.backward()                                         # backpropagation
    opt.step()                                              # nudge every weight''')

EPISODES = ["next token", "tokens", "embeddings", "position", "attention", "attention math",
            "multi-head", "MLP", "block", "logits", "training"]
MONOKAI = ["#F92672", "#A6E22E", "#66D9EF", "#E6DB74", "#F8F8F2", "#AE81FF"]

# terminal (sections 10-12)
TERM_FS = 20
TERM_LINE_SPACING = 0.75
S0_LINES = SAMPLES[0][:6]
S250_LINES = SAMPLES[250][:7]
S5000_LINES = SAMPLES[5000][:15]                           # up to "CLARENCE:" and its first line
TERM_CHARS = max(len(line) for line in S5000_LINES)        # trim everything else to this width


# ---------------------------------------------------------------------------- helpers
def label(text, font_size=24, color=GREY_B, **kw):
    return Text(text, font_size=font_size, color=color, **kw)


def row_height(code):
    return abs(code.line_numbers[0].get_y() - code.line_numbers[1].get_y())


def set_hl(hl, code, i, j=None):
    """Put the highlight over lines i..j (0-based, inclusive) without animating."""
    j = i if j is None else j
    hl.stretch_to_fit_height(row_height(code) * (j - i + 1))
    hl.set_y((code.line_numbers[i].get_y() + code.line_numbers[j].get_y()) / 2)
    return hl


def span(hl, code, i, j=None):
    """Animate the highlight to cover lines i..j (also resets a multi-line highlight to one line)."""
    j = i if j is None else j
    return hl.animate.stretch_to_fit_height(row_height(code) * (j - i + 1)).set_y(
        (code.line_numbers[i].get_y() + code.line_numbers[j].get_y()) / 2)


def panel(src, font_size=20, pos=ORIGIN, title=None, max_width=12.8, line=0):
    """Monokai panel moved to `pos`, highlight parked on `line`, optional title in the window bar."""
    code, hl = code_panel(src, font_size, max_width)
    shift = np.array(pos, dtype=float) - code.get_center()
    code.shift(shift)
    hl.shift(shift)
    set_hl(hl, code, line)
    group = VGroup(code)
    if title:
        cap = Text(title, font=MONO, font_size=20, color=GREY_B)
        cap.move_to(code.get_top() + 0.22 * DOWN).align_to(code, RIGHT).shift(0.3 * LEFT)
        cap.set_z_index(3)
        group.add(cap)
    return code, hl, group


def char_box(c, w=0.6, h=0.65, font_size=28, color=TOKEN_COLOR):
    shown = {"\n": "⏎", " ": "␣"}.get(c, c)
    box = RoundedRectangle(corner_radius=0.1, width=w, height=h, stroke_color=color, stroke_width=2,
                           fill_color=color, fill_opacity=0.3)
    glyph = Text(shown, font=MONO, font_size=font_size, color=GREY_B if c.isspace() else WHITE)
    return VGroup(box, glyph.move_to(box))


def badge(text, color=TEAL_D, font_size=20, height=0.28):
    t = Text(text, font_size=font_size)
    box = RoundedRectangle(corner_radius=0.08, width=t.width + 0.3, height=height, stroke_color=color,
                           stroke_width=2, fill_color=color, fill_opacity=0.35)
    return VGroup(box, t.move_to(box)).set_z_index(3)


def episode_card(n, name):
    box = RoundedRectangle(corner_radius=0.12, width=2.9, height=0.95, stroke_color=TOKEN_COLOR,
                           fill_color=TOKEN_COLOR, fill_opacity=0.2)
    txt = VGroup(Text(f"video {n}", font_size=20, color=GREY_B), Text(name, font_size=26)).arrange(DOWN, buff=0.08)
    return VGroup(box, txt.move_to(box))


def file_icon(w=1.9, h=2.4, fold=0.45):
    body = Polygon([-w / 2, h / 2, 0], [w / 2 - fold, h / 2, 0], [w / 2, h / 2 - fold, 0], [w / 2, -h / 2, 0],
                   [-w / 2, -h / 2, 0], stroke_color=GREY_A, stroke_width=3, fill_color=GREY_E, fill_opacity=1)
    ear = Polygon([w / 2 - fold, h / 2, 0], [w / 2 - fold, h / 2 - fold, 0], [w / 2, h / 2 - fold, 0],
                  stroke_color=GREY_A, stroke_width=3, fill_color=GREY_D, fill_opacity=1)
    pattern = [(0, 0.9), (0.2, 0.7), (0.4, 0.8), (0.4, 0.55), (0, 0.0), (0.2, 0.75), (0.4, 0.95), (0.4, 0.5)]
    lines = VGroup()
    for k, (ind, length) in enumerate(pattern):
        if length == 0:
            continue
        x0, y = -w / 2 + 0.22 + ind, h / 2 - 0.62 - k * 0.22
        lines.add(Line([x0, y, 0], [x0 + length, y, 0], stroke_width=5, color=MONOKAI[k % len(MONOKAI)]))
    return VGroup(body, ear, lines)


def laptop():
    screen = RoundedRectangle(corner_radius=0.1, width=2.6, height=1.7, stroke_color=GREY_A, stroke_width=3,
                              fill_color=GREY_E, fill_opacity=1)
    base = Polygon([-1.45, 0, 0], [1.45, 0, 0], [1.75, -0.22, 0], [-1.75, -0.22, 0], stroke_color=GREY_A,
                   stroke_width=3, fill_color=GREY_D, fill_opacity=1)
    base.next_to(screen, DOWN, buff=0.04)
    chip = Square(0.75, stroke_color=TEAL_C, stroke_width=3, fill_color=TEAL_E, fill_opacity=0.6)
    pins = VGroup()
    for k in range(4):
        off = -0.24 + k * 0.16
        pins.add(Line([off, 0.375, 0], [off, 0.5, 0]), Line([off, -0.375, 0], [off, -0.5, 0]),
                 Line([0.375, off, 0], [0.5, off, 0]), Line([-0.375, off, 0], [-0.5, off, 0]))
    pins.set_stroke(TEAL_C, 3)
    cpu = Text("CPU", font_size=20, weight=BOLD).move_to(chip)
    return VGroup(screen, base, VGroup(chip, pins, cpu).move_to(screen))


def glyphs(text_mob, s, sub, occurrence=0):
    """The glyph slice of `text_mob` (built from string `s`) that spells `sub` (Text has no glyphs for whitespace)."""
    idx = -1
    for _ in range(occurrence + 1):
        idx = s.index(sub, idx + 1)
    start = len(re.sub(r"\s", "", s[:idx]))
    return text_mob[start:start + len(re.sub(r"\s", "", sub))]


def check_line(text, color=GREEN):
    return Text(text, font_size=26, color=GREY_A, t2c={"✓": color})


def model_box(w, h, n_layers, color, stripe_h, stripe_buff):
    box = RoundedRectangle(corner_radius=0.15, width=w, height=h, stroke_color=color, fill_color=color,
                           fill_opacity=0.12)
    stripes = VGroup(*[RoundedRectangle(corner_radius=0.03, width=w * 0.72, height=stripe_h, stroke_width=0,
                                        fill_color=color, fill_opacity=0.65) for _ in range(n_layers)])
    stripes.arrange(DOWN, buff=stripe_buff).move_to(box)
    return VGroup(box, stripes)


# ---------------------------------------------------------------------------- scene
class TinyGPTVideo(VoicedScene):
    VIDEO = "v12"

    def clear_anims(self, keep=()):
        anims = []
        for m in list(self.mobjects):
            if m in keep:
                continue
            m.clear_updaters()
            anims.append(FadeOut(m))
        return anims

    def construct(self):
        play_token_intro(self, TITLE, 12, TAGLINE)
        self.s1_pieces()
        self.s2_data()
        self.s3_tokenizer()
        self.s4_embeddings()
        self.s5_attention()
        self.s6_mlp_block()
        self.s7_output()
        self.s8_size()
        self.s9_training()
        self.s10_step0()
        self.s11_step250()
        self.s12_step5000()
        self.s13_outro()
        finish(self)

    # 1. Eleven videos -> one file ---------------------------------------------------------
    def s1_pieces(self):
        self.section(1)
        cards = VGroup(*[episode_card(i + 1, name) for i, name in enumerate(EPISODES)])
        rows = VGroup(cards[0:4], cards[4:8], cards[8:11])
        for r in rows:
            r.arrange(RIGHT, buff=0.3)
        rows.arrange(DOWN, buff=0.3).move_to([0, 0.2, 0])
        self.at("11")
        self.play(LaggedStart(*[FadeIn(c, shift=0.2 * UP) for c in cards], lag_ratio=0.08), run_time=1.4)

        icon = file_icon().move_to([-3.4, 0.5, 0])
        name = Text("tiny_gpt.py", font=MONO, font_size=26).next_to(icon, DOWN, buff=0.3)
        self.at("together")
        self.play(FadeIn(icon, scale=0.8), FadeIn(name),
                  LaggedStart(*[c.animate.scale(0.12).move_to(icon).set_opacity(0) for c in cards], lag_ratio=0.04),
                  run_time=1.0)
        self.remove(*cards)
        lines = Text("≈ 100 lines", font_size=30, color=YELLOW).next_to(name, DOWN, buff=0.25)
        self.at("100")
        self.play(FadeIn(lines, shift=0.2 * UP), run_time=0.5)

        comp = laptop().move_to([3.4, 0.6, 0])
        comp_lab = label("an ordinary computer", 26).next_to(comp, DOWN, buff=0.35)
        arrow = Arrow(icon.get_right() + 0.2 * RIGHT, [comp.get_left()[0] - 0.2, icon.get_y(), 0], buff=0,
                      color=GREY_B, stroke_width=5)
        train = label("train", 26, color=WHITE).next_to(arrow, UP, buff=0.15)
        self.at("train")
        self.play(GrowArrow(arrow), FadeIn(train), run_time=0.5)
        self.at("computer")
        self.play(FadeIn(comp, shift=0.3 * LEFT), FadeIn(comp_lab), run_time=0.6)
        self.end_section()

    # 2. The data: every character is a token ---------------------------------------------
    def s2_data(self):
        self.section(2)
        first, second = TEXT.split("\n")[:2]               # "First Citizen:" / "Before we proceed ..."
        box = RoundedRectangle(corner_radius=0.12, width=10.8, height=1.35, stroke_color=GREY_C, stroke_width=2,
                               fill_color=GREY_E, fill_opacity=1).move_to([0, 2.05, 0])
        fname = Text("input.txt", font=MONO, font_size=22, color=GREY_B).next_to(box, UP, buff=0.15)
        fname.align_to(box, LEFT)
        self.play(*self.clear_anims(), FadeIn(box), FadeIn(fname), run_time=0.5)

        count = Text(f"{len(TEXT):,} characters", font_size=28, color=YELLOW).next_to(box, UP, buff=0.12)
        count.align_to(box, RIGHT)
        self.at("million")
        self.play(FadeIn(count, shift=0.2 * DOWN), run_time=0.5)

        l1 = Text(first, font=MONO, font_size=24)
        l2 = Text(second, font=MONO, font_size=24)
        text = VGroup(l1, l2).arrange(DOWN, buff=0.18, aligned_edge=LEFT)
        text.move_to(box).align_to(box, LEFT).shift(0.4 * RIGHT)
        self.at("shakespeares")
        self.play(Write(l1), Write(l2), run_time=1.0)

        row = VGroup(*[char_box(c) for c in first]).arrange(RIGHT, buff=0.1).move_to([0, 0.55, 0])
        targets = [cb[1] for c, cb in zip(first, row) if not c.isspace()]
        spaces = [cb[1] for c, cb in zip(first, row) if c.isspace()]
        self.at("character")
        self.play(*[FadeIn(cb[0]) for cb in row], *[FadeIn(s) for s in spaces],
                  *[ReplacementTransform(g.copy(), t) for g, t in zip(l1, targets)],
                  l1.animate.set_color(YELLOW), run_time=0.8)
        note = label("1 character = 1 token", 26).move_to([0, -0.2, 0])
        self.at("token")
        self.play(FadeIn(note, shift=0.15 * UP), run_time=0.4)

        grid = VGroup(*[char_box(c, w=0.46, h=0.46, font_size=22) for c in CHARS])
        grid.arrange_in_grid(rows=5, cols=13, buff=0.08).move_to([0, -2.0, 0])
        vocab = label("the whole vocabulary: 65 symbols", 26, color=WHITE).move_to(note)
        self.at("65")
        self.play(LaggedStart(*[FadeIn(g, scale=0.7) for g in grid], lag_ratio=0.012),
                  FadeTransform(note, vocab), run_time=0.8)
        self.end_section()

    # 3. Tokenizer ---------------------------------------------------------------------------
    def s3_tokenizer(self):
        self.section(3)
        code, hl, win = panel(TOKENIZER, 24, [0, 1.3, 0], title="tiny_gpt.py")
        self.play(*self.clear_anims(), FadeIn(win, shift=0.2 * UP), run_time=0.5)
        set_hl(hl, code, 2, 3)
        self.at("dictionaries")
        self.play(Create(hl), run_time=0.4)

        ids = [STOI[c] for c in "hi"]
        src = Text('"hi"', font=MONO, font_size=32)
        nums = Text(f"[{', '.join(map(str, ids))}]", font=MONO, font_size=32, color=YELLOW)
        back = Text('"hi"', font=MONO, font_size=32)
        a1 = Arrow(ORIGIN, 2.2 * RIGHT, buff=0, color=GREY_B)
        a2 = Arrow(ORIGIN, 2.2 * RIGHT, buff=0, color=GREY_B)
        chain = VGroup(src, a1, nums, a2, back).arrange(RIGHT, buff=0.3).move_to([0, -2.3, 0])
        enc = Text("encode", font=MONO, font_size=22, color=GREY_B).next_to(a1, UP, buff=0.12)
        dec = Text("decode", font=MONO, font_size=22, color=GREY_B).next_to(a2, UP, buff=0.12)
        self.at("id")
        self.play(span(hl, code, 4), FadeIn(src), GrowArrow(a1), FadeIn(enc), FadeIn(nums, shift=0.2 * RIGHT),
                  run_time=0.5)
        self.at("back")
        self.play(span(hl, code, 5), GrowArrow(a2), FadeIn(dec), FadeIn(back, shift=0.2 * RIGHT), run_time=0.5)
        self.end_section()

    # 4. Embeddings --------------------------------------------------------------------------
    def tinygpt_panel(self):
        return panel(TINYGPT, 20, [0, 0.6, 0], title="class TinyGPT")

    def s4_embeddings(self):
        self.section(4)
        code, hl, win = self.tinygpt_panel()
        self.play(*self.clear_anims(), FadeIn(win, shift=0.2 * UP), run_time=0.5)
        set_hl(hl, code, 3, 4)
        self.at("two")
        self.play(Create(hl), run_time=0.4)
        self.at("tokens")
        self.play(span(hl, code, 3), run_time=0.3)
        self.at("positions")
        self.play(highlight(hl, code, 4), run_time=0.3)
        self.at("added")
        self.play(highlight(hl, code, 10), run_time=0.3)
        b3 = badge("video 3").next_to(code.code_lines[3], RIGHT, buff=0.5).set_y(code.line_numbers[3].get_y())
        b4 = badge("video 4").next_to(b3, DOWN, buff=0).set_y(code.line_numbers[4].get_y())
        self.at("three")
        self.play(FadeIn(b3, shift=0.2 * LEFT), run_time=0.3)
        self.at("four")
        self.play(FadeIn(b4, shift=0.2 * LEFT), run_time=0.3)
        self.end_section()

    # 5. Attention ---------------------------------------------------------------------------
    def s5_attention(self):
        self.section(5)
        code, hl, win = panel(ATTN_FORWARD, 20, [0, 1.25, 0], title="class Attention", max_width=13.6)
        self.play(*self.clear_anims(), FadeIn(win, shift=0.2 * UP), run_time=0.5)
        set_hl(hl, code, 2)
        self.at("queries")
        self.play(Create(hl), run_time=0.4)
        icode, ihl, iwin = panel(ATTN_INIT, 20, [1.4, -2.45, 0])
        where = label("in __init__:", 24).next_to(icode, LEFT, buff=0.35)
        self.at("linear")
        self.play(FadeIn(iwin, shift=0.2 * UP), FadeIn(where), FadeIn(ihl), run_time=0.5)
        self.at("heads")
        self.play(highlight(hl, code, 3), run_time=0.3)
        self.at("mask")
        self.play(highlight(hl, code, 5), run_time=0.3)
        self.at("softmax")
        self.play(highlight(hl, code, 6), run_time=0.3)
        self.end_section()

    # 6. MLP and block -----------------------------------------------------------------------
    def s6_mlp_block(self):
        self.section(6)
        code, hl, win = panel(MLP_CODE, 24, [0, 1.75, 0], title="class MLP")
        self.play(*self.clear_anims(), FadeIn(win, shift=0.2 * UP), run_time=0.5)

        def bar(n, color):
            return Rectangle(width=n / 128 * 1.1, height=0.42, stroke_width=0, fill_color=color, fill_opacity=0.85)

        y = -2.3
        b1 = bar(128, BLUE_C).move_to([-4.8, y, 0])
        b2 = bar(512, TEAL_C).move_to([0, y, 0])
        b3 = bar(128, BLUE_C).move_to([4.8, y, 0])
        a1 = Arrow(b1.get_right(), b2.get_left(), buff=0.15, color=GREY_B)
        a2 = Arrow(b2.get_right(), b3.get_left(), buff=0.15, color=GREY_B)
        n1 = label("128", 22).next_to(b1, DOWN, buff=0.15)
        n2 = label("4 × 128 = 512", 22).next_to(b2, DOWN, buff=0.15)
        n3 = label("128", 22).next_to(b3, DOWN, buff=0.15)
        up = Text("up", font=MONO, font_size=22).next_to(a1, UP, buff=0.1)
        down = Text("down", font=MONO, font_size=22).next_to(a2, UP, buff=0.1)
        gelu = Text("GELU", font_size=24, color=YELLOW).next_to(b2, UP, buff=0.12)

        set_hl(hl, code, 3)
        self.at("expands")
        self.play(Create(hl), FadeIn(b1), FadeIn(n1), GrowArrow(a1), FadeIn(up),
                  GrowFromEdge(b2, LEFT), FadeIn(n2), run_time=0.8)
        self.at("jell")
        self.play(highlight(hl, code, 7), b2.animate.set_fill(YELLOW_D, 0.85), FadeIn(gelu), run_time=0.4)
        self.at("projects")
        self.play(highlight(hl, code, 4), GrowArrow(a2), FadeIn(down), FadeIn(b3), FadeIn(n3), run_time=0.5)

        bcode, bhl, bwin = panel(BLOCK_CODE, 20, [0, 1.55, 0], title="class Block", max_width=13.1)
        self.at("block")
        self.play(*self.clear_anims(), FadeIn(bwin, shift=0.2 * UP), run_time=0.5)
        set_hl(bhl, bcode, 7, 8)
        self.at("residual")
        self.play(Create(bhl), run_time=0.4)
        self.at("layer")
        self.play(span(bhl, bcode, 3, 4), run_time=0.3)

        hcode, hhl, hwin = panel(HYPER, 20, [-2.9, -2.35, 0], line=1)
        stack = VGroup(*[RoundedRectangle(corner_radius=0.08, width=2.2, height=0.36, stroke_color=MODEL_COLOR,
                                          fill_color=MODEL_COLOR, fill_opacity=0.35) for _ in range(4)])
        stack.arrange(UP, buff=0.1).move_to([3.3, -2.35, 0])
        stack_lab = VGroup(*[Text("Block", font_size=20).move_to(b) for b in stack])
        times = Text("× 4", font_size=34, color=YELLOW).next_to(stack, RIGHT, buff=0.3)
        self.at("stack")
        self.play(FadeIn(hwin, shift=0.2 * UP), FadeIn(hhl),
                  LaggedStart(*[FadeIn(VGroup(b, t), shift=0.25 * UP) for b, t in zip(stack, stack_lab)],
                              lag_ratio=0.2), run_time=0.7)
        self.at("four")
        self.play(FadeIn(times, scale=1.3), run_time=0.3)
        self.end_section()

    # 7. Output head -------------------------------------------------------------------------
    def s7_output(self):
        self.section(7)
        code, hl, win = self.tinygpt_panel()
        self.play(*self.clear_anims(), FadeIn(win, shift=0.2 * UP), run_time=0.5)
        set_hl(hl, code, 6)
        self.at("norm")
        self.play(Create(hl), run_time=0.4)
        self.at("linear")
        self.play(highlight(hl, code, 7), run_time=0.3)
        out = Text("→ 65 logits, one per character", font_size=32, color=YELLOW).move_to([0, -2.6, 0])
        self.at("65")
        self.play(FadeIn(out, shift=0.2 * UP), run_time=0.5)
        self.end_section()

    # 8. How big is it? (log scale) ----------------------------------------------------------
    def s8_size(self):
        self.section(8)
        x0, unit = -6.0, 1.0                                # x of 10^0, scene units per power of ten

        def xl(n):
            return x0 + unit * np.log10(n)

        axis_y = -2.3
        axis = Line([x0, axis_y, 0], [xl(1e12) + 0.2, axis_y, 0], color=GREY_B, stroke_width=2)
        ticks, grid, tick_labs = VGroup(), VGroup(), VGroup()
        for p, txt in zip(range(0, 13, 3), ["1", "1K", "1M", "1B", "1T"]):
            x = xl(10 ** p)
            ticks.add(Line([x, axis_y - 0.08, 0], [x, axis_y + 0.08, 0], color=GREY_B, stroke_width=2))
            grid.add(DashedLine([x, axis_y, 0], [x, 2.7, 0], color=GREY_D, stroke_width=1.5, dash_length=0.08))
            tick_labs.add(label(txt, 22).next_to([x, axis_y, 0], DOWN, buff=0.18))
        head = Text("parameters", font_size=32).to_corner(UL, buff=0.5)
        scale_cap = label("log scale", 24).to_corner(UR, buff=0.5)
        self.play(*self.clear_anims(), FadeIn(VGroup(grid, axis, ticks, tick_labs, head, scale_cap)), run_time=0.5)

        def bar(n, y, color):
            return Rectangle(width=xl(n) - x0, height=0.45, stroke_width=0, fill_color=color,
                             fill_opacity=0.85).move_to([x0, y, 0], aligned_edge=LEFT)

        rows = [(1.6, PARAMS, BLUE_C), (0.15, 124e6, TEAL_C), (-1.3, 3e11, PURPLE_B)]
        bars = [bar(n, y, c) for y, n, c in rows]
        labs = [Text(f"tiny GPT: {PARAMS:,}", font_size=28), Text("GPT-2 small: 124 million", font_size=28)]
        for lab, (y, _, _) in zip(labs, rows):
            lab.move_to([x0, y + 0.55, 0], aligned_edge=LEFT)
        big_a = Text("largest today:", font_size=28).move_to([x0, rows[2][0] + 0.55, 0], aligned_edge=LEFT)
        big_b = Text("hundreds of billions", font_size=28, color=YELLOW).next_to(big_a, RIGHT, buff=0.2)

        self.at("818")
        self.play(GrowFromEdge(bars[0], LEFT), FadeIn(labs[0]), run_time=0.8)
        self.at("124")
        self.play(GrowFromEdge(bars[1], LEFT), FadeIn(labs[1]), run_time=0.8)
        self.at("largest")
        self.play(GrowFromEdge(bars[2], LEFT), FadeIn(big_a), run_time=1.0)
        self.at("billions")
        self.play(FadeIn(big_b, shift=0.2 * LEFT), run_time=0.5)
        self.end_section()

    # 9. Training loop -----------------------------------------------------------------------
    def s9_training(self):
        self.section(9)
        code, hl, win = panel(TRAIN, 20, [0, 0.9, 0], title="tiny_gpt.py", max_width=13.2)
        self.play(*self.clear_anims(), FadeIn(win, shift=0.2 * UP), run_time=0.5)
        set_hl(hl, code, 2)
        self.at("loop")
        self.play(Create(hl), run_time=0.4)
        self.at("chunks")
        self.play(highlight(hl, code, 4), run_time=0.3)
        self.at("predict")
        self.play(highlight(hl, code, 5), run_time=0.3)
        self.at("loss")
        self.play(highlight(hl, code, 6), run_time=0.3)
        self.at("backpropagate")
        self.play(span(hl, code, 7, 8), run_time=0.3)
        self.at("step")
        self.play(span(hl, code, 9), run_time=0.3)
        self.at("atom")
        self.play(highlight(hl, code, 0), run_time=0.3)
        steps = Text("5,000 steps", font_size=32, color=YELLOW)
        rest = Text("· 45 min on an 8-core CPU", font_size=32)
        cap = VGroup(steps, rest).arrange(RIGHT, buff=0.25).move_to([0, -2.75, 0])
        self.at("5")
        self.play(highlight(hl, code, 2), FadeIn(steps, shift=0.2 * UP), run_time=0.4)
        self.at("hour")
        self.play(FadeIn(rest, shift=0.2 * UP), run_time=0.4)
        self.end_section()

    # 10-12. Samples from the real run -------------------------------------------------------
    def terminal(self):
        """Dark terminal box sized for the step-5000 sample; returns (box, rule, x_left, y_top_of_body)."""
        body, _ = self.term_body(S5000_LINES)
        w = TERM_CHARS * Text("M" * 10, font=MONO, font_size=TERM_FS).width / 10 + 0.8
        h = body.height + 1.35
        box = RoundedRectangle(corner_radius=0.15, width=w, height=h, stroke_color=GREY_C, stroke_width=2,
                               fill_color=GREY_E, fill_opacity=1)
        box.move_to([-6.6 + w / 2, 3.6 - h / 2, 0])
        rule = Line(box.get_corner(UL) + [0.2, -0.75, 0], box.get_corner(UR) + [-0.2, -0.75, 0],
                    color=GREY_C, stroke_width=1.5)
        temp = label("sample · temperature 0.8", 20).move_to(box.get_top() + 0.4 * DOWN)
        temp.align_to(box, RIGHT).shift(0.3 * LEFT)
        return box, rule, temp

    def term_body(self, lines):
        shown = [line[:TERM_CHARS].rstrip() for line in lines]
        return Text("\n".join(shown), font=MONO, font_size=TERM_FS, line_spacing=TERM_LINE_SPACING), "\n".join(shown)

    def term_header(self, box, step):
        head = Text(f"step {step} · val loss {VAL[step]:.2f}", font=MONO, font_size=22, color=YELLOW)
        return head.move_to(box.get_top() + 0.4 * DOWN).align_to(box, LEFT).shift(0.3 * RIGHT)

    def place_body(self, body, box):
        return body.next_to(box.get_top(), DOWN, buff=0.95).align_to(box, LEFT).shift(0.4 * RIGHT)

    def captions_at(self, box, texts):
        col = VGroup(*texts).arrange(DOWN, buff=0.45, aligned_edge=LEFT)
        return col.next_to(box, RIGHT, buff=0.45).align_to(box, UP).shift(0.5 * DOWN)

    def s10_step0(self):
        self.section(10)
        self.term_box, rule, temp = self.terminal()
        self.head = self.term_header(self.term_box, 0)
        self.play(*self.clear_anims(), FadeIn(VGroup(self.term_box, rule, temp, self.head)), run_time=0.5)
        self.body, _ = self.term_body(S0_LINES)
        self.place_body(self.body, self.term_box)
        self.at("writes")
        self.play(FadeIn(self.body), run_time=0.4)
        cap = label("pure noise", 30, color=WHITE)
        self.captions_at(self.term_box, [cap])
        self.at("noise")
        self.play(FadeIn(cap, shift=0.2 * UP), run_time=0.4)
        self.caps = [cap]
        self.end_section()

    def swap_sample(self, step, lines):
        head = self.term_header(self.term_box, step)
        body, s = self.term_body(lines)
        self.place_body(body, self.term_box)
        self.play(FadeOut(self.head), FadeIn(head), FadeOut(self.body), FadeIn(body),
                  *[FadeOut(c) for c in self.caps], run_time=0.5)
        self.head, self.body = head, body
        return body, s

    def s11_step250(self):
        self.section(11)
        body, s = self.swap_sample(250, S250_LINES)
        caps = [check_line("common letters ✓"), check_line("spaces ✓"), check_line("short lines ✓"),
                Text("made-up words", font_size=26, color=YELLOW)]
        self.captions_at(self.term_box, caps)
        for cue, cap in zip(["common", "spaces", "short"], caps):
            self.at(cue)
            self.play(FadeIn(cap, shift=0.2 * UP), run_time=0.4)
        made_up = [glyphs(body, s, w) for w in ("thamfors", "hakeate", "asivrisked")]
        self.at("made")
        self.play(FadeIn(caps[3], shift=0.2 * UP), *[m.animate.set_color(YELLOW) for m in made_up], run_time=0.4)
        self.caps = caps
        self.end_section()

    def s12_step5000(self):
        self.section(12)
        body, s = self.swap_sample(5000, S5000_LINES)
        caps = [check_line("real words ✓"), check_line("character names ✓"),
                Text("the shape of a play", font_size=26, color=GREY_A),
                Text("learned only by\npredicting the\nnext character", font_size=26, color=YELLOW,
                     line_spacing=0.6)]
        self.captions_at(self.term_box, caps)
        self.at("words")
        self.play(FadeIn(caps[0], shift=0.2 * UP), run_time=0.4)
        names = [glyphs(body, s, n) for n in ("GLOUCESTER:", "Lord:", "CLARENCE:")]
        self.at("names")
        self.play(FadeIn(caps[1], shift=0.2 * UP), *[n.animate.set_color(YELLOW) for n in names], run_time=0.4)
        self.at("play")
        self.play(FadeIn(caps[2], shift=0.2 * UP), run_time=0.4)
        self.at("predicting")
        self.play(FadeIn(caps[3], shift=0.2 * UP), run_time=0.5)
        self.end_section()

    # 13. Outro ------------------------------------------------------------------------------
    def s13_outro(self):
        self.section(13)
        tiny = model_box(1.5, 1.5, 4, BLUE_C, 0.16, 0.12).move_to([0, 0.4, 0])
        tiny_lab = Text("tiny GPT", font_size=28).next_to(tiny, DOWN, buff=0.25)
        self.play(*self.clear_anims(), FadeIn(tiny, scale=0.8), FadeIn(tiny_lab), run_time=0.5)
        built = check_line("built from scratch ✓").next_to(tiny_lab, DOWN, buff=0.3)
        self.at("scratch")
        self.play(FadeIn(built, shift=0.2 * UP), run_time=0.4)

        huge = model_box(4.2, 5.0, 24, PURPLE_B, 0.12, 0.07).move_to([3.9, -0.2, 0])
        huge_lab = Text("real LLMs", font_size=28).next_to(huge, UP, buff=0.2)
        recipe = Text("same recipe", font_size=32, color=YELLOW).move_to([-1.35, 2.75, 0])
        self.at("same")
        shift = np.array([-5.3, -0.1, 0]) - VGroup(tiny, tiny_lab).get_center()
        self.play(VGroup(tiny, tiny_lab).animate.shift(shift), built.animate.shift(shift).set_opacity(0),
                  FadeIn(huge, shift=0.3 * LEFT), FadeIn(huge_lab), FadeIn(recipe), run_time=0.8)
        arrows, words = VGroup(), VGroup()
        for y, txt in zip([1.1, 0.0, -1.1], ["more data", "more layers", "more compute"]):
            a = Arrow([-4.2, y - 0.1, 0], [1.5, y - 0.1, 0], buff=0, color=GREY_B, stroke_width=4)
            arrows.add(a)
            words.add(Text(txt, font_size=26).next_to(a, UP, buff=0.08))
        self.at("data")
        self.play(GrowArrow(arrows[0]), FadeIn(words[0]), run_time=0.4)
        self.at("layers")
        self.play(GrowArrow(arrows[1]), FadeIn(words[1]),
                  LaggedStart(*[Indicate(s, color=YELLOW, scale_factor=1.05) for s in huge[1]], lag_ratio=0.03),
                  run_time=0.8)
        self.at("compute")
        self.play(GrowArrow(arrows[2]), FadeIn(words[2]), run_time=0.4)

        card = next_up_card(NEXT)
        self.at("bonus")
        self.play(*self.clear_anims(), FadeIn(card, shift=0.2 * UP), run_time=0.8)
        then = label(f"then: {BONUS_2}", 26).next_to(card, DOWN, buff=0.6)
        self.at("chatbot")
        self.play(FadeIn(then, shift=0.2 * UP), run_time=0.5)
        self.end_section()
