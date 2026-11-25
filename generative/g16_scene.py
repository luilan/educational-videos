"""How Generative Models Work: Images and Sound, episode 16 — Audio Tokens: Neural Codecs.

Render from the repo root:  ./render.sh generative g16
Numbers, pictures and the five audio clips come from code/g16_neural_codecs (results.json, tokens.json,
out/train_log.json, out/mel_*.png, out/clip_*.wav).
"""
import json

import numpy as np
import soundfile as sf
from manim import *

from common import MONO, code_panel, finish, highlight, next_up_card
from gen_common import pic
from intro import play_token_intro
from g16_script import NEXT, TAGLINE, TITLE
from voiced_scene import VoicedScene

SERIES_LABEL = "HOW GENERATIVE MODELS WORK  ·  EPISODE 16"
CODE_DIR = "code/g16_neural_codecs"
OUT = f"{CODE_DIR}/out"
R = json.load(open(f"{CODE_DIR}/results.json"))
TK = json.load(open(f"{CODE_DIR}/tokens.json"))
S = R["systems"]
WAVE_C, ENC_C, CB_C, DEC_C, TOK_C, REAL_C = TEAL_C, BLUE_C, YELLOW, ORANGE, PURPLE_B, GREEN_C
CODE = """z = encoder(x)                       # 24,000 samples/s → 75 vectors/s
residual, tokens = z, []
for book in codebooks[:n]:           # residual vector quantization
    i = book.nearest(residual)       # one token (10 bits) per frame
    tokens.append(i)
    residual = residual - book[i]
y = decoder(sum(book[i] for book, i in zip(codebooks, tokens)))"""


def small(text, color=GREY_B, size=30, scale=0.55):
    return Text(text, font_size=size, color=color).scale(scale)


def pct(v, d=0):
    return f"{v * 100:.{d}f}%"


def wave_bars(x, n=240, width=11.0, height=1.6, color=WAVE_C):
    x = np.asarray(x)
    env = np.abs(x[:len(x) // n * n]).reshape(n, -1).max(1)
    xs = np.linspace(-width / 2, width / 2, n)
    s = height / 2 / env.max()
    return VGroup(*[Line([a, -v * s, 0], [a, v * s, 0], stroke_width=1.6, color=color) for a, v in zip(xs, env)])


def token_grid(codes, cols, cell=0.3, books=8):
    g = VGroup()
    for r in range(books):
        for c in range(cols):
            v = codes[r][c]
            col = interpolate_color(BLUE_E, YELLOW, (v % 97) / 96)
            g.add(Square(cell, stroke_width=0.6, stroke_color=BLACK, fill_color=col, fill_opacity=0.95).move_to([c * cell, -r * cell, 0]))
    return g


class CodecVideo(VoicedScene):
    VIDEO = "g16"

    def construct(self):
        play_token_intro(self, TITLE, 16, TAGLINE, label=SERIES_LABEL)
        self.hook()          # 1
        self.encoder()       # 2
        self.vq()            # 3
        self.rvq()           # 4
        self.decoder()       # 5
        self.results()       # 6
        self.listen()        # 7
        self.bottleneck()    # 8
        self.real()          # 9
        self.language()      # 10
        self.code()          # 11
        self.outro()         # 12

    def section(self, i):
        print(f"SECTION {i} starts at {self.renderer.time:.2f}")
        super().section(i)

    def clear_stage(self, *keep, run_time=0.6):
        self.play(*[FadeOut(m) for m in self.mobjects if m not in keep], run_time=run_time)

    def play_clip(self, name, label, pos, color=WHITE):
        secs = sf.info(f"{OUT}/clip_{name}.wav").duration
        lab = Text(label, font_size=24, color=color).move_to(pos)
        track = Line(pos + 0.5 * DOWN + 2 * LEFT, pos + 0.5 * DOWN + 2 * RIGHT, color=GREY_D, stroke_width=6)
        bar = Line(track.get_start(), track.get_start() + 0.001 * RIGHT, color=color, stroke_width=6)
        icon = Text("♪", font_size=34, color=color).next_to(lab, LEFT, buff=0.25)
        self.play(FadeIn(lab), FadeIn(icon), FadeIn(track), run_time=0.3)
        print(f"CLIP {name} at {self.renderer.time:.2f}")
        self.add_sound(f"{OUT}/clip_{name}.wav")
        self.add(bar)
        self.play(bar.animate.put_start_and_end_on(track.get_start(), track.get_end()), run_time=secs, rate_func=linear)
        self.wait(0.3)
        g = VGroup(lab, icon, track, bar)
        self.play(FadeOut(g), run_time=0.25)

    # ------------------------------------------------------------------ 1. hook
    def hook(self):
        self.section(1)
        x, _ = sf.read(f"{OUT}/clip_original.wav")
        self.x = x
        llm = VGroup(*[VGroup(Square(0.6, color=TOK_C, fill_opacity=0.2), Text(t, font_size=18)) for t in ("The", " cat", " sat", " ?")]).arrange(RIGHT, buff=0.1)
        for b in llm:
            b[1].move_to(b[0])
        llm.move_to([0, 2.4, 0])
        ll = small("a language model writes tokens", TOK_C, 28).next_to(llm, DOWN, buff=0.15)
        self.at("tokens")
        self.play(FadeIn(llm), FadeIn(ll), run_time=0.5)
        w = wave_bars(x[:SR1], width=11).move_to([0, 0.1, 0])
        wl = small("one second of our narrator: 24,000 numbers", WAVE_C, 28).next_to(w, DOWN, buff=0.15)
        self.at("second")
        self.play(Create(w, lag_ratio=0.002), FadeIn(wl), run_time=0.9)
        grid = token_grid(TK["codes"], 25, cell=0.26).move_to([0, -2.4, 0])
        grid.move_to([1.6, -2.4, 0])
        gl = small("a neural codec:\na few hundred tokens", CB_C, 28).next_to(grid, LEFT, buff=0.4)
        self.at("codec")
        self.play(FadeIn(grid, lag_ratio=0.01), FadeIn(gl), run_time=0.8)
        self.end_section()

    # ------------------------------------------------------------------ 2. encoder
    def encoder(self):
        self.section(2)
        self.clear_stage()
        C = R["codec"]
        w = wave_bars(self.x[:SR1 // 4], n=120, width=3.2, height=1.4).move_to([-5.0, 1.2, 0])
        wl = small("24,000 samples/s", WAVE_C, 26).next_to(w, DOWN, buff=0.15)
        self.play(FadeIn(w), FadeIn(wl), run_time=0.4)
        blocks = VGroup()
        for k, s in enumerate((2, 4, 5, 8)):
            b = RoundedRectangle(width=1.15, height=0.9 + 0.25 * k, corner_radius=0.1, color=ENC_C, fill_color=ENC_C, fill_opacity=0.12)
            blocks.add(VGroup(b, Text(f"÷{s}", font=MONO, font_size=24, color=ENC_C).move_to(b)))
        blocks.arrange(RIGHT, buff=0.2).move_to([0.2, 1.2, 0])
        self.at("encoder")
        self.play(GrowArrow(Arrow(w.get_right(), blocks.get_left(), buff=0.1, color=GREY_B)), run_time=0.3)
        self.at("strides")
        self.play(LaggedStart(*[FadeIn(b) for b in blocks], lag_ratio=0.2), run_time=0.8)
        prod = small("2 · 4 · 5 · 8 = 320 samples → one vector", GREY_A, 28).next_to(blocks, DOWN, buff=0.3)
        self.at("320")
        self.play(FadeIn(prod), run_time=0.4)
        vec = VGroup(*[Rectangle(width=0.32, height=0.06, stroke_width=0, fill_color=interpolate_color(BLUE_E, WHITE, ((i * 37) % 64) / 64), fill_opacity=1) for i in range(64)]).arrange(DOWN, buff=0.005).move_to([5.0, 1.2, 0])
        vl = small(f"{C['latent_dim']} numbers", ENC_C, 26).next_to(vec, DOWN, buff=0.15)
        self.at("64")
        self.play(GrowArrow(Arrow(blocks.get_right(), vec.get_left(), buff=0.15, color=GREY_B)), FadeIn(vec), FadeIn(vl), run_time=0.5)
        row = VGroup(*[vec.copy().scale(0.6) for _ in range(20)]).arrange(RIGHT, buff=0.12).move_to([0, -1.6, 0])
        rl = small(f"{C['frames_per_second']:.0f} vectors a second · one every {1000 / C['frames_per_second']:.1f} ms", ENC_C, 28).next_to(row, DOWN, buff=0.2)
        self.at("75")
        self.play(LaggedStart(*[FadeIn(r) for r in row], lag_ratio=0.05), FadeIn(rl), run_time=0.8)
        self.end_section()

    # ------------------------------------------------------------------ 3. vector quantization
    def vq(self):
        self.section(3)
        self.clear_stage()
        rng = np.random.default_rng(3)
        pts = rng.normal(0, 1, (160, 2))
        pts = pts / np.abs(pts).max() * 2.6
        center = np.array([-2.6, -0.2, 0])
        dots = VGroup(*[Dot(center + np.array([a, b, 0]), radius=0.045, color=CB_C) for a, b in pts])
        box = Square(6.0, color=GREY_D).move_to(center)
        bl = small(f"codebook: {R['codec']['codebook_size']:,} learned vectors (shown in 2D)", CB_C, 26).next_to(box, UP, buff=0.15)
        self.at("nearest")
        self.play(FadeIn(box), LaggedStart(*[FadeIn(d) for d in dots], lag_ratio=0.004), FadeIn(bl), run_time=0.8)
        z = np.array([0.9, 0.7])
        k = int(np.argmin(((pts - z) ** 2).sum(1)))
        star = Star(5, outer_radius=0.16, color=ENC_C, fill_opacity=1).move_to(center + np.array([*z, 0]))
        sl = small("encoder vector", ENC_C, 24).next_to(star, UR, buff=0.05)
        self.play(FadeIn(star), FadeIn(sl), run_time=0.4)
        hit = dots[k]
        self.play(star.animate.move_to(hit.get_center()), Flash(hit, color=WHITE), run_time=0.6)
        tok = int(TK["codes"][0][0])
        t = VGroup(Text("keep only its number:", font_size=26),
                   Text(f"token {tok}", font=MONO, font_size=40, color=CB_C),
                   small(f"one of {R['codec']['codebook_size']:,} → {R['codec']['bits_per_token']} bits", GREY_B, 28)).arrange(DOWN, buff=0.25).move_to([3.6, 0, 0])
        self.at("number")
        self.play(FadeIn(t[0]), FadeIn(t[1]), run_time=0.5)
        self.at("10")
        self.play(FadeIn(t[2]), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 4. residual VQ
    def rvq(self):
        self.section(4)
        self.clear_stage()
        rem = np.array(R["rvq_remaining_energy"])
        ax = Axes(x_range=[0, 8.6, 1], y_range=[0, 6, 1], x_length=7.4, y_length=3.6, tips=False, axis_config={"color": GREY_D}).move_to([-2.2, -0.2, 0])
        bars = VGroup()
        labels = VGroup()
        for i, r in enumerate(rem):
            h = 3.6 * min(r * 100, 6) / 6
            b = Rectangle(width=0.62, height=max(h, 0.02), stroke_width=0, fill_color=CB_C, fill_opacity=0.9).move_to(ax.c2p(i + 1, 0), aligned_edge=DOWN)
            bars.add(b)
            labels.add(Text(f"{r * 100:.1f}%", font=MONO, font_size=16).next_to(b, UP, buff=0.08))
        xl = VGroup(*[Text(str(i + 1), font=MONO, font_size=16, color=GREY_B).next_to(ax.c2p(i + 1, 0), DOWN, buff=0.12) for i in range(8)])
        title = small("left unexplained after each codebook", GREY_A, 30).next_to(ax, UP, buff=0.3)
        xt = small("codebooks used", GREY_B, 26).next_to(xl, DOWN, buff=0.1)
        self.at("one")
        self.play(Create(ax), FadeIn(xl), FadeIn(xt), FadeIn(title), GrowFromEdge(bars[0], DOWN), FadeIn(labels[0]), run_time=0.7)
        cap = small(f"codebook 1 captures {pct(1 - rem[0])}", WHITE, 28).move_to([3.9, 2.0, 0])
        self.play(FadeIn(cap), run_time=0.3)
        self.at("residual", "missed")
        self.play(GrowFromEdge(bars[1], DOWN), FadeIn(labels[1]), run_time=0.5)
        self.at("third")
        self.play(GrowFromEdge(bars[2], DOWN), FadeIn(labels[2]), run_time=0.5)
        self.at("eight")
        self.play(LaggedStart(*[AnimationGroup(GrowFromEdge(b, DOWN), FadeIn(l)) for b, l in zip(bars[3:], labels[3:])], lag_ratio=0.15), run_time=0.9)
        C = R["codec"]
        rate = VGroup(Text("8 tokens per frame", font_size=24),
                      Text(f"× {C['frames_per_second']:.0f} frames = 600 tokens/s", font=MONO, font_size=20),
                      Text(f"× {C['bits_per_token']} bits = {C['kbps_per_codebook'] * 8:g} kbit/s", font=MONO, font_size=20, color=CB_C),
                      small("(raw 16-bit audio: 384 kbit/s)", GREY_B, 26)).arrange(DOWN, buff=0.18).move_to([4.0, -0.4, 0])
        self.at("tokens")
        self.play(FadeIn(rate[0]), FadeIn(rate[1]), run_time=0.4)
        self.at("kilobits")
        self.play(FadeIn(rate[2]), FadeIn(rate[3]), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 5. decoder and training
    def decoder(self):
        self.section(5)
        self.clear_stage()
        grid = token_grid(TK["codes"], 12, cell=0.22).move_to([-5.0, 1.8, 0])
        gl = small("tokens", CB_C, 26).next_to(grid, UP, buff=0.1)
        dec = RoundedRectangle(width=2.6, height=1.0, corner_radius=0.12, color=DEC_C).move_to([-0.6, 1.8, 0])
        dl = Text("decoder", font_size=24, color=DEC_C).move_to(dec)
        w = wave_bars(self.x[:SR1 // 4], n=120, width=3.2, height=1.2).move_to([4.4, 1.8, 0])
        self.at("decoder")
        self.play(FadeIn(grid), FadeIn(gl), GrowArrow(Arrow(grid.get_right(), dec.get_left(), buff=0.15, color=GREY_B)), FadeIn(dec), FadeIn(dl), run_time=0.6)
        self.play(GrowArrow(Arrow(dec.get_right(), w.get_left(), buff=0.15, color=GREY_B)), FadeIn(w), run_time=0.5)
        m1 = pic(f"{OUT}/mel_original.png", 1.3).stretch_to_fit_width(3.6).move_to([-3.6, -1.2, 0])
        m2 = pic(f"{OUT}/mel_ours_8.png", 1.3).stretch_to_fit_width(3.6).move_to([1.6, -1.2, 0])
        l1 = small("input", GREY_B, 26).next_to(m1, UP, buff=0.1)
        l2 = small("output", GREY_B, 26).next_to(m2, UP, buff=0.1)
        vs = Text("≈ ?", font_size=34).move_to([-1.0, -1.2, 0])
        self.at("spectrograms")
        self.play(FadeIn(m1), FadeIn(m2), FadeIn(l1), FadeIn(l2), FadeIn(vs), run_time=0.6)
        cb = small("codebook vectors drift toward the vectors assigned to them", CB_C, 26).move_to([0, -2.55, 0])
        self.at("drifts")
        self.play(FadeIn(cb), run_time=0.4)
        C = R["codec"]
        dr = small("each step: a random number of codebooks (1–8)", GREY_A, 26).next_to(cb, DOWN, buff=0.12)
        self.at("randomly")
        self.play(FadeIn(dr), run_time=0.4)
        tr = small(f"{C['encoder_parameters'] + C['decoder_parameters']:,} parameters · {C['train_steps']:,} steps · "
                   f"{C['train_minutes_of_audio']:.0f} min of speech, 8 voices", WHITE, 26).move_to([4.2, 0.4, 0]).scale_to_fit_width(5.6)
        self.at("trained")
        self.play(FadeIn(tr), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 6. results
    def results(self):
        self.section(6)
        self.clear_stage()
        keys = ["ours_1", "ours_2", "ours_4", "ours_8"]
        top = max(10, np.ceil(max(S[k]["words_changed"] for k in keys) * 100 / 5) * 5)
        ax = Axes(x_range=[0, 4.6, 1], y_range=[0, top, top / 2], x_length=7.6, y_length=3.6, tips=False, axis_config={"color": GREY_D}).move_to([-0.8, -0.3, 0])
        xl = VGroup(*[VGroup(Text(f"{n} book{'s' if n > 1 else ''}", font_size=16, color=GREY_B),
                             Text(f"{S[k]['kbps']:g} kbit/s", font=MONO, font_size=14, color=GREY_B)).arrange(DOWN, buff=0.05).next_to(ax.c2p(i + 1, 0), DOWN, buff=0.12)
                    for i, (n, k) in enumerate(zip((1, 2, 4, 8), keys))])
        yl = VGroup(*[Text(f"{v:g}%", font=MONO, font_size=16, color=GREY_B).next_to(ax.c2p(0, v), LEFT, buff=0.1) for v in (0, top / 2, top)])
        title = VGroup(Text("words Whisper hears differently than in the original", font_size=24),
                       small("40 sentences the codec never heard", GREY_B, 26)).arrange(DOWN, buff=0.12).to_edge(UP, buff=0.35)
        self.at("never")
        self.play(FadeIn(title), Create(ax), FadeIn(xl), FadeIn(yl), run_time=0.5)
        bars = VGroup()
        for i, k in enumerate(keys):
            v = S[k]["words_changed"] * 100
            b = Rectangle(width=0.9, height=max(0.02, 3.6 * min(v, top) / top), stroke_width=0, fill_color=CB_C, fill_opacity=0.9).move_to(ax.c2p(i + 1, 0), aligned_edge=DOWN)
            bars.add(VGroup(b, Text(f"{v:.1f}%", font=MONO, font_size=18).next_to(b, UP, buff=0.08)))
        self.at("one")
        self.play(FadeIn(bars[0]), run_time=0.4)
        self.at("two")
        self.play(FadeIn(bars[1]), run_time=0.4)
        self.play(FadeIn(bars[2]), FadeIn(bars[3]), run_time=0.6)
        self.end_section()

    # ------------------------------------------------------------------ 7. listen
    def listen(self):
        self.section(7)
        self.clear_stage()
        t = Text(f"“{R['example']['text']}”", font_size=26).to_edge(UP, buff=0.8)
        self.play(FadeIn(t), run_time=0.4)
        self.end_section()
        for name, label, col in (("original", "original (384 kbit/s)", WHITE), ("ours_1", "1 codebook (0.75 kbit/s)", RED_B),
                                 ("ours_8", "8 codebooks (6 kbit/s)", CB_C)):
            self.play_clip(name, label, np.array([0, -0.4, 0]), col)

    # ------------------------------------------------------------------ 8. bottleneck
    def bottleneck(self):
        self.section(8)
        self.clear_stage()
        keys = [("ours_1", "1 codebook"), ("ours_2", "2 codebooks"), ("ours_4", "4 codebooks"), ("ours_8", "8 codebooks"),
                ("ours_unquantized_0", "no rounding at all")]
        top = max(S[k]["mel_distance"] for k, _ in keys)
        rows = VGroup()
        for k, name in keys:
            v = S[k]["mel_distance"]
            col = DEC_C if k.endswith("_0") else CB_C
            rows.add(VGroup(Text(name, font_size=22, color=col),
                            Rectangle(width=7 * v / top, height=0.4, stroke_width=0, fill_color=col, fill_opacity=0.85),
                            Text(f"{v:.3f}", font=MONO, font_size=18)))
        for k, r in enumerate(rows):
            y = 1.3 - 0.65 * k
            r[0].move_to([-2.6, y, 0], aligned_edge=RIGHT)
            r[1].move_to([-2.3, y, 0], aligned_edge=LEFT)
            r[2].next_to(r[1], RIGHT, buff=0.15)
        title = Text("mel distance to the original (lower is better)", font_size=24).to_edge(UP, buff=0.5)
        self.at("beyond")
        self.play(FadeIn(title), LaggedStart(*[FadeIn(r) for r in rows[:4]], lag_ratio=0.15), run_time=0.8)
        self.at("rounding")
        self.play(FadeIn(rows[4]), run_time=0.4)
        con = small("the small decoder is the limit, not the bits", YELLOW, 32).to_edge(DOWN, buff=0.6)
        self.at("limit")
        self.play(FadeIn(con), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 9. real codecs
    def real(self):
        self.section(9)
        self.clear_stage()
        P = R["parameters"]
        C = R["codec"]
        hdr = ["", "parameters", "frames/s", "kbit/s", "words changed", "mel dist."]
        rows = [("ours (8 books)", f"{(C['encoder_parameters'] + C['decoder_parameters']) / 1e6:.1f}M", "75", "6", S["ours_8"]),
                ("EnCodec", f"{P['encodec'] / 1e6:.1f}M", "75", "6", S["encodec_6"]),
                ("EnCodec", f"{P['encodec'] / 1e6:.1f}M", "75", "1.5", S["encodec_1.5"]),
                ("Mimi (8 books)", f"{P['mimi'] / 1e6:.1f}M", "12.5", "1.1", S["mimi_8"])]
        tab = VGroup()
        tab.add(VGroup(*[Text(h, font_size=18, color=GREY_B) for h in hdr]))
        for name, p, f, kb, s in rows:
            tab.add(VGroup(Text(name, font_size=20, color=CB_C if name.startswith("ours") else REAL_C),
                           *[Text(v, font=MONO, font_size=18) for v in (p, f, kb, pct(s["words_changed"], 1), f"{s['mel_distance']:.3f}")]))
        xs = [-4.6, -1.9, -0.2, 1.3, 3.0, 4.9]
        for i, row in enumerate(tab):
            for x, cell in zip(xs, row):
                cell.move_to([x, 1.6 - 0.6 * i, 0])
        self.at("bigger")
        self.play(FadeIn(tab[0]), FadeIn(tab[1]), run_time=0.4)
        judges = small("trained with judges: the GAN game of episode 4", GREY_A, 28).move_to([0, -1.7, 0])
        self.at("judges")
        self.play(FadeIn(judges), run_time=0.4)
        self.at("metas")
        self.play(FadeIn(tab[2]), FadeIn(tab[3]), run_time=0.5)
        self.at("mimi")
        self.play(FadeIn(tab[4]), run_time=0.4)
        mel_col = VGroup(*[tab[i][5] for i in (1, 2, 4)])
        self.at("distance")
        self.play(Create(SurroundingRectangle(mel_col, buff=0.12, color=YELLOW)), tab[1][5].animate.set_color(YELLOW), run_time=0.5)
        self.end_section()
        self.play(FadeOut(judges), run_time=0.2)
        self.play_clip("encodec_6", "EnCodec, 6 kbit/s", np.array([-3.4, -2.2, 0]), REAL_C)
        self.play_clip("mimi_8", "Mimi, 1.1 kbit/s", np.array([3.4, -2.2, 0]), REAL_C)

    # ------------------------------------------------------------------ 10. speech as language
    def language(self):
        self.section(10)
        self.clear_stage()
        E = R["example"]
        t = Text(f"“{E['text']}”  ·  {E['seconds']:.1f} s", font_size=22).to_edge(UP, buff=0.5)
        self.play(FadeIn(t), run_time=0.3)
        counts = [("text (GPT-2 tokens)", E["text_tokens_gpt2"], TOK_C),
                  ("EnCodec, 6 kbit/s", E["encodec_frames"] * E["encodec_6k_codebooks"], REAL_C),
                  ("Mimi, 8 books", E["mimi_frames"] * E["mimi_codebooks"], REAL_C)]
        top = max(c for _, c, _ in counts)
        rows = VGroup()
        for name, c, col in counts:
            rows.add(VGroup(Text(name, font_size=20, color=col), Rectangle(width=max(0.05, 6.2 * c / top), height=0.42, stroke_width=0, fill_color=col, fill_opacity=0.85),
                            Text(f"{c:,} tokens", font=MONO, font_size=18)))
        for r in rows:
            r[0].move_to([-3.6, 0, 0], aligned_edge=RIGHT)
            r[1].next_to(r[0], RIGHT, buff=0.25)
            r[2].next_to(r[1], RIGHT, buff=0.15)
        rows.arrange(DOWN, aligned_edge=LEFT, buff=0.3).move_to([0.6, 1.0, 0])
        self.at("text")
        self.play(FadeIn(rows[0]), run_time=0.4)
        self.at("codec")
        self.play(FadeIn(rows[1]), run_time=0.4)
        self.at("mimi")
        self.play(FadeIn(rows[2]), run_time=0.4)
        lm = VGroup(RoundedRectangle(width=6.6, height=0.9, corner_radius=0.12, color=TOK_C),
                    Text("transformer: predict the next audio token", font_size=20, color=TOK_C))
        lm[1].move_to(lm[0])
        lm.move_to([0, -1.5, 0])
        ex = small("VALL-E (EnCodec tokens) · Moshi (Mimi tokens)", GREY_A, 28).next_to(lm, DOWN, buff=0.25)
        self.at("transformer")
        self.play(FadeIn(lm), run_time=0.4)
        self.at("volley")
        self.play(FadeIn(ex), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 11. code
    def code(self):
        self.section(11)
        self.clear_stage()
        code, hl = code_panel(CODE, font_size=17)
        code.move_to([0, 0.2, 0])
        self.play(FadeIn(code, shift=0.2 * UP), run_time=0.4)
        self.play(Create(hl), run_time=0.2)
        for i, w in ((0, "encode"), (3, "quantize"), (6, "decode")):  # Whisper hears "In code" as "encode"
            self.at(w)
            self.play(highlight(hl, code, i), run_time=0.3)
        self.end_section()

    # ------------------------------------------------------------------ 12. outro
    def outro(self):
        self.section(12)
        self.clear_stage(run_time=0.4)
        card_ = next_up_card(NEXT)
        self.at("next")
        self.play(FadeIn(card_, shift=0.2 * UP), run_time=0.6)
        self.end_section()
        finish(self, hold=1.5)


SR1 = 24000
