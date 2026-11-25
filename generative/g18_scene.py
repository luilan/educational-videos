"""How Generative Models Work: Images and Sound, episode 18 — Speech to Text: Inside Whisper (series finale).

Render from the repo root:  ./render.sh generative g18
Numbers and pictures come from code/g18_inside_whisper (results.json, attention.json, out/*.png, out/noise_0.wav).
"""
import json

import numpy as np
import soundfile as sf
from manim import *

from common import MONO, code_panel, finish, highlight
from gen_common import pic
from intro import play_token_intro
from g18_script import TAGLINE, TITLE
from voiced_scene import VoicedScene

SERIES_LABEL = "HOW GENERATIVE MODELS WORK  ·  EPISODE 18"
CODE_DIR = "code/g18_inside_whisper"
OUT = f"{CODE_DIR}/out"
R = json.load(open(f"{CODE_DIR}/results.json"))
A = json.load(open(f"{CODE_DIR}/attention.json"))
MEL_C, ENC_C, DEC_C, TOK_C, TIME_C = PURPLE_B, BLUE_C, ORANGE, TEAL_C, YELLOW
CODE = """mel = log_mel(resample(audio, 16000), pad_to=30)      # 80 × 3,000
h = encoder(mel)                                     # 1,500 × 768
ids = [SOT, NO_TIMESTAMPS]
while ids[-1] != EOT:
    ids.append(argmax(decoder(ids, h)))              # one token at a time
starts = dtw(-mean(cross_attention[align_heads]))    # word times"""


def small(text, color=GREY_B, size=30, scale=0.55):
    return Text(text, font_size=size, color=color).scale(scale)


def colormap(v):
    stops = np.array([[0.02, 0.02, 0.1], [0.35, 0.1, 0.5], [0.85, 0.35, 0.25], [1.0, 0.9, 0.35]])
    p = np.clip(v, 0, 1) * (len(stops) - 1)
    i = np.minimum(p.astype(int), len(stops) - 2)
    f = (p - i)[..., None]
    return ((stops[i] * (1 - f) + stops[i + 1] * f) * 255).astype(np.uint8)


def chip(text, color, size=20, mono=True):
    t = Text(text, font=MONO if mono else "DejaVu Serif", font_size=size, color=color)
    b = SurroundingRectangle(t, buff=0.1, corner_radius=0.08, color=color, stroke_width=1.5)
    return VGroup(b, t)


class WhisperVideo(VoicedScene):
    VIDEO = "g18"

    def construct(self):
        play_token_intro(self, TITLE, 18, TAGLINE, label=SERIES_LABEL)
        self.hook()          # 1
        self.input()         # 2
        self.encoder()       # 3
        self.decoder()       # 4
        self.greedy()        # 5
        self.training()      # 6
        self.attention()     # 7
        self.dtw()           # 8
        self.accuracy()      # 9
        self.errors()        # 10
        self.noise()         # 11
        self.silence()       # 12
        self.code()          # 13
        self.finale()        # 14

    def section(self, i):
        print(f"SECTION {i} starts at {self.renderer.time:.2f}")
        super().section(i)

    def clear_stage(self, *keep, run_time=0.6):
        self.play(*[FadeOut(m) for m in self.mobjects if m not in keep], run_time=run_time)

    # ------------------------------------------------------------------ 1. hook
    def hook(self):
        self.section(1)
        words = [w for w in self.sections[0]["words"] if w["t"] <= 3.0]
        head = Text("voice/g18/sections.json: this sentence, as Whisper timed it", font_size=22, color=GREY_B).to_edge(UP, buff=0.6)
        self.play(FadeIn(head), run_time=0.4)
        line = NumberLine(x_range=[0, 3.2, 0.5], length=10.5, include_numbers=False, color=GREY_D).move_to([-0.4, -1.6, 0])
        ticks = VGroup(*[Text(f"{t:g} s", font=MONO, font_size=16, color=GREY_B).next_to(line.n2p(t), DOWN, buff=0.15) for t in (0, 1, 2, 3)])
        self.play(Create(line), FadeIn(ticks), run_time=0.4)
        chips = VGroup()
        for k, w in enumerate(words):
            c = chip(w["w"], TOK_C, 18)
            c.move_to(line.n2p(w["t"]) + UP * (0.5 + 0.55 * (k % 3)), aligned_edge=LEFT + DOWN)
            chips.add(VGroup(c, Line(line.n2p(w["t"]), c.get_corner(DL), color=TOK_C, stroke_width=1.5)))
        for k, w in enumerate(words):
            self.wait_until(self.sec_start + w["t"])
            self.play(FadeIn(chips[k]), run_time=0.15)
        self.at("open")
        box = SurroundingRectangle(VGroup(line, chips), buff=0.25, color=YELLOW)
        self.play(Create(box), run_time=0.5)
        self.end_section()

    # ------------------------------------------------------------------ 2. input
    def input(self):
        self.section(2)
        self.clear_stage()
        I = R["input"]
        img = pic(f"{OUT}/input_30s.png", 1.7).stretch_to_fit_width(12.4).move_to([0, 0.4, 0])
        lab = Text("30 seconds · 16,000 samples a second → log-mel spectrogram", font_size=24).to_edge(UP, buff=0.5)
        self.at("30")
        self.play(FadeIn(lab), run_time=0.4)
        self.at("turned")
        self.play(FadeIn(img), run_time=0.6)
        dims = VGroup(Text("80 bands", font=MONO, font_size=20, color=MEL_C).rotate(PI / 2).next_to(img, LEFT, buff=0.1),
                      Text(f"{I['mel_shape'][1]:,} columns (100 a second)", font=MONO, font_size=20, color=MEL_C).next_to(img, DOWN, buff=0.15))
        self.at("80")
        self.play(FadeIn(dims[0]), run_time=0.3)
        self.at("3")
        self.play(FadeIn(dims[1]), run_time=0.3)
        frac = I["frames_with_sound"] / I["mel_shape"][1]
        x0 = img.get_left()[0]
        snd = Rectangle(width=12.4 * frac, height=1.85, color=YELLOW, stroke_width=3).move_to([x0 + 12.4 * frac / 2, 0.4, 0])
        sl = small(f"our sentence: {I['seconds']:.1f} s", YELLOW, 28).next_to(snd, UP, buff=0.12)
        self.at("five")
        self.play(Create(snd), FadeIn(sl), run_time=0.5)
        pad = small("padding: silence up to 30 s", GREY_B, 28).move_to([x0 + 12.4 * (frac + 1) / 2, -1.3, 0])
        self.at("padding")
        self.play(FadeIn(pad), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 3. encoder
    def encoder(self):
        self.section(3)
        self.clear_stage()
        M = R["model"]
        mel = pic(f"{OUT}/input_sound.png", 1.1).stretch_to_fit_width(5.0).move_to([-4.0, 2.2, 0])
        ml = small("log-mel: 3,000 columns", MEL_C, 26).next_to(mel, UP, buff=0.1)
        self.play(FadeIn(mel), FadeIn(ml), run_time=0.4)
        conv = VGroup(*[RoundedRectangle(width=2.6, height=0.55, corner_radius=0.1, color=ENC_C) for _ in range(2)]).arrange(DOWN, buff=0.12).move_to([0.6, 2.2, 0])
        cl = VGroup(Text("conv · stride 1", font=MONO, font_size=16, color=ENC_C).move_to(conv[0]),
                    Text("conv · stride 2", font=MONO, font_size=16, color=ENC_C).move_to(conv[1]))
        self.at("convolutions")
        self.play(GrowArrow(Arrow(mel.get_right(), conv.get_left(), buff=0.15, color=GREY_B)), FadeIn(conv), FadeIn(cl[0]), run_time=0.5)
        self.at("stride")
        self.play(FadeIn(cl[1]), Indicate(conv[1], color=YELLOW), run_time=0.5)
        vecs = VGroup(*[Rectangle(width=0.06, height=1.0, stroke_width=0, fill_color=ENC_C, fill_opacity=0.4 + 0.5 * ((i * 7) % 5) / 5) for i in range(100)]).arrange(RIGHT, buff=0.02).move_to([0, 0.3, 0])
        vl = small(f"{M['conv'][1][1]}-number vectors: {R['encoder']['shape'][1]:,}, one every 20 ms", ENC_C, 28).next_to(vecs, DOWN, buff=0.15)
        self.at("1")
        self.play(LaggedStart(*[FadeIn(v) for v in vecs], lag_ratio=0.01), FadeIn(vl), run_time=0.8)
        stack = VGroup(*[Rectangle(width=8.0, height=0.12, color=ENC_C, fill_color=ENC_C, fill_opacity=0.25, stroke_width=1) for _ in range(12)]).arrange(DOWN, buff=0.05).move_to([0, -2.0, 0])
        sl = Text(f"12 transformer layers · encoder {M['encoder_parameters'] / 1e6:.0f}M parameters", font_size=22, color=ENC_C).next_to(stack, DOWN, buff=0.15)
        self.at("transformer")
        self.play(LaggedStart(*[FadeIn(s) for s in stack], lag_ratio=0.05), FadeIn(sl), run_time=0.7)
        arcs = VGroup(*[ArcBetweenPoints(vecs[a].get_top(), vecs[b].get_top(), angle=-PI / 3, color=YELLOW, stroke_width=1.5) for a, b in ((10, 60), (30, 85), (50, 15), (75, 40))])
        self.at("attend")
        self.play(LaggedStart(*[Create(a) for a in arcs], lag_ratio=0.15), run_time=0.7)
        self.end_section()

    # ------------------------------------------------------------------ 4. decoder
    def decoder(self):
        self.section(4)
        self.clear_stage()
        M = R["model"]
        enc = RoundedRectangle(width=4.0, height=1.0, corner_radius=0.12, color=ENC_C).move_to([-4.2, 1.8, 0])
        el = Text("encoder: 1,500 vectors", font_size=20, color=ENC_C).move_to(enc)
        dec = RoundedRectangle(width=5.2, height=1.0, corner_radius=0.12, color=DEC_C).move_to([2.6, 1.8, 0])
        dl = Text(f"decoder: a language model · {M['decoder_parameters'] / 1e6:.0f}M", font_size=20, color=DEC_C).move_to(dec)
        self.at("decoder")
        self.play(FadeIn(enc), FadeIn(el), FadeIn(dec), FadeIn(dl), run_time=0.5)
        ca = Arrow(enc.get_right(), dec.get_left(), buff=0.15, color=YELLOW)
        cl = small("cross-attention: every layer looks at the audio", YELLOW, 26).next_to(ca, UP, buff=0.5)
        self.at("cross")
        self.play(GrowArrow(ca), FadeIn(cl), run_time=0.5)
        toks = R["decoding"]["tokens"]
        sot = VGroup(chip(toks[0], GREY_A, 18), chip(toks[1], GREY_A, 18)).arrange(RIGHT, buff=0.15).move_to([-2.5, -0.3, 0])
        self.at("start")
        self.play(FadeIn(sot[0]), run_time=0.3)
        self.at("no")
        self.play(FadeIn(sot[1]), run_time=0.3)
        nxt = VGroup(Text("→", font_size=30, color=GREY_B), chip(" ?", DEC_C, 20)).arrange(RIGHT, buff=0.2).next_to(sot, RIGHT, buff=0.2)
        self.at("predicts")
        self.play(FadeIn(nxt), run_time=0.4)
        self.sot = sot
        self.end_section()

    # ------------------------------------------------------------------ 5. greedy steps
    def greedy(self):
        self.section(5)
        self.clear_stage(self.sot)
        self.play(self.sot.animate.to_edge(UP, buff=0.5).to_edge(LEFT, buff=0.5), run_time=0.4)
        steps = R["decoding"]["steps"]
        row = VGroup()
        for s in steps:
            row.add(chip(s["token"].replace(" ", "·"), TOK_C if not s["token"].startswith("<|") else GREY_A, 18))
        row.arrange(RIGHT, buff=0.08).scale_to_fit_width(12.6).move_to([0, 1.8, 0])

        def bars(i):
            s = steps[i]
            opts = [(s["token"], s["p"])] + [(a, p) for a, p in s["alternatives"]]
            g = VGroup()
            for k, (t, p) in enumerate(opts):
                lab = Text(t.replace(" ", "·"), font=MONO, font_size=20, color=TOK_C if k == 0 else GREY_B)
                bar = Rectangle(width=max(0.03, 6 * p), height=0.32, stroke_width=0, fill_color=TOK_C if k == 0 else GREY_C, fill_opacity=0.9)
                pct = Text(f"{p * 100:.0f}%", font=MONO, font_size=18)
                g.add(VGroup(lab, bar, pct))
            for k, item in enumerate(g):
                item[0].move_to([-4.5, -0.3 - 0.5 * k, 0], aligned_edge=RIGHT)
                item[1].move_to([-4.2, -0.3 - 0.5 * k, 0], aligned_edge=LEFT)
                item[2].next_to(item[1], RIGHT, buff=0.15)
            return g

        cur = None
        for i, cue in ((0, "ys"), (1, "per"), (2, "listens")):
            self.at(cue)
            new = bars(i)
            self.play(FadeIn(row[i]), FadeOut(cur) if cur else Wait(0.01), FadeIn(new), run_time=0.3)
            cur = new
        self.at("likely")
        self.play(LaggedStart(*[FadeIn(row[i]) for i in range(3, 4)], lag_ratio=0.2), run_time=0.3)
        self.at("30")
        new = bars(4)
        self.play(FadeIn(row[4]), FadeOut(cur), FadeIn(new), run_time=0.3)
        self.play(Indicate(row[4], color=YELLOW), run_time=0.4)
        cur = new
        self.at("end")
        self.play(LaggedStart(*[FadeIn(r) for r in row[5:]], lag_ratio=0.08), run_time=1.0)
        self.end_section()

    # ------------------------------------------------------------------ 6. training and time tokens
    def training(self):
        self.section(6)
        self.clear_stage()
        h = Text("680,000 hours", font_size=48, color=YELLOW).move_to([0, 2.0, 0])
        hl = small("of transcribed audio from the web (Radford et al., 2022)", GREY_B, 28).next_to(h, DOWN, buff=0.2)
        self.at("680")
        self.play(FadeIn(h), FadeIn(hl), run_time=0.5)
        obj = small("objective: predict the next token", WHITE, 30).next_to(hl, DOWN, buff=0.4)
        self.at("predict")
        self.play(FadeIn(obj), run_time=0.4)
        ts = R["timestamp_tokens"]
        parts = ts.split("|>")
        first, rest = parts[0] + "|>", "|>".join(parts[1:])
        body, last = rest.rsplit("<|", 1)
        line = VGroup(chip(first, TIME_C, 20), Text(body.strip(), font_size=20), chip("<|" + last, TIME_C, 20)).arrange(RIGHT, buff=0.15)
        line.scale_to_fit_width(min(12.5, line.width)).move_to([0, -1.0, 0])
        self.at("time")
        self.play(FadeIn(line), run_time=0.6)
        st = small("time tokens: one for every 20 ms of the 30 s window", TIME_C, 28).next_to(line, DOWN, buff=0.3)
        self.at("steps")
        self.play(FadeIn(st), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 7. cross-attention map
    def heatmap(self):
        m = np.array(A["matrix"])
        img = colormap(m)
        img = np.repeat(np.repeat(img, 12, axis=0), 2, axis=1)
        im = ImageMobject(img)
        im.set_resampling_algorithm(RESAMPLING_ALGORITHMS["nearest"])
        im.stretch_to_fit_width(9.6).stretch_to_fit_height(4.4).move_to([0.9, -0.3, 0])
        return im, m.shape

    def attention(self):
        self.section(7)
        self.clear_stage()
        q = Text("where does each word's time come from?", font_size=26).to_edge(UP, buff=0.4)
        self.play(FadeIn(q), run_time=0.4)
        im, (n_tok, n_fr) = self.heatmap()
        labs = VGroup(*[Text(p.replace(" ", "·"), font=MONO, font_size=14, color=TOK_C).move_to(
            [im.get_left()[0] - 0.1, im.get_top()[1] - (i + 0.5) * im.height / n_tok, 0], aligned_edge=RIGHT) for i, p in enumerate(A["pieces"])])
        xl = VGroup(*[Text(f"{t} s", font=MONO, font_size=14, color=GREY_B).move_to([im.get_left()[0] + im.width * t * 50 / n_fr, im.get_bottom()[1] - 0.2, 0]) for t in range(0, 6)])
        hl = small(f"{R['model']['alignment_heads']} alignment heads, averaged · rows: tokens · columns: 20 ms of audio", GREY_B, 26).next_to(im, DOWN, buff=0.45)
        self.at("19")
        self.play(FadeIn(im), FadeIn(labs), FadeIn(xl), FadeIn(hl), run_time=0.8)
        self.at("diagonal")
        diag = DashedLine(im.get_corner(UL), im.get_corner(DR), color=WHITE, stroke_width=2)
        self.play(Create(diag), run_time=0.6)
        self.play(FadeOut(diag), run_time=0.3)
        self.im, self.n = im, (n_tok, n_fr)
        self.end_section()

    # ------------------------------------------------------------------ 8. DTW
    def dtw(self):
        self.section(8)
        im = self.im
        n_tok, n_fr = self.n
        pt = lambda i, j: [im.get_left()[0] + (j + 0.5) * im.width / n_fr, im.get_top()[1] - (i + 0.5) * im.height / n_tok, 0]
        path = A["path"]
        line = VMobject(stroke_color=WHITE, stroke_width=3).set_points_as_corners([pt(i, j) for i, j in path])
        self.at("dynamic")
        self.play(Create(line), run_time=2.0)
        T = R["timestamps_example"]
        first = {}
        for i, j in path:
            first.setdefault(i, j)
        word_rows = [i for i, p in enumerate(A["pieces"]) if p.startswith(" ") or i == 0]
        dots = VGroup(*[Dot(pt(i, first[i]), radius=0.07, color=TIME_C) for i in word_rows])
        self.at("reaches")
        self.play(LaggedStart(*[FadeIn(d, scale=1.5) for d in dots], lag_ratio=0.08), run_time=0.8)
        ex = small(f"“{T['words'][1]}”: starts at {T['whisper_dtw_start'][1]:.2f} s", TIME_C, 28).to_corner(UR, buff=0.5).shift(DOWN * 0.5)
        self.at("start")
        self.play(FadeIn(ex), Indicate(dots[1], color=WHITE), run_time=0.5)
        self.end_section()

    # ------------------------------------------------------------------ 9. accuracy
    def accuracy(self):
        self.section(9)
        self.clear_stage()
        S = R["timestamp_accuracy"]
        h = np.array(S["histogram_ms"])
        ax = Axes(x_range=[-400, 400, 100], y_range=[0, h.max() * 1.1, 50], x_length=10, y_length=3.6, tips=False,
                  axis_config={"color": GREY_D}).move_to([0, -0.6, 0])
        xl = VGroup(*[Text(f"{v:+d} ms" if v else "0", font=MONO, font_size=16, color=GREY_B).next_to(ax.c2p(v, 0), DOWN, buff=0.15) for v in (-400, -200, 0, 200, 400)])
        bars = VGroup(*[Rectangle(width=10 / 16 * 0.9, height=max(0.01, 3.6 * c / (h.max() * 1.1)), stroke_width=0, fill_color=TIME_C if k < 8 else RED_B,
                                  fill_opacity=0.9).move_to(ax.c2p(-400 + 50 * k + 25, 0), aligned_edge=DOWN) for k, c in enumerate(h)])
        title = Text("Whisper's word start − Kokoro's true start", font_size=24).to_edge(UP, buff=0.4)
        self.at("truth")
        self.play(FadeIn(title), Create(ax), FadeIn(xl), run_time=0.5)
        self.at("275")
        self.play(LaggedStart(*[GrowFromEdge(b, DOWN) for b in bars], lag_ratio=0.04), run_time=1.0)
        stats = VGroup(Text(f"median error {S['median_abs_ms']:.0f} ms", font=MONO, font_size=22),
                       Text(f"{sum(h[:8]) / sum(h) * 100:.0f}% early · mean {S['mean_signed_ms']:+.0f} ms", font=MONO, font_size=22, color=TIME_C)).arrange(DOWN, aligned_edge=LEFT, buff=0.12).to_corner(UR, buff=0.6).shift(DOWN * 0.6)
        self.at("115")
        self.play(FadeIn(stats[0]), run_time=0.3)
        self.at("early")
        self.play(FadeIn(stats[1]), run_time=0.3)
        cue = small("every cue fires about 0.1 s before its word", YELLOW, 30).to_edge(DOWN, buff=0.3)
        self.at("cue")
        self.play(FadeIn(cue), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 10. word errors
    def errors(self):
        self.section(10)
        self.clear_stage()
        N = R["narration"]
        head = Text(f"this series' narration: {N['episodes']} episodes, {N['words']:,} words", font_size=26).to_edge(UP, buff=0.5)
        self.at("across")
        self.play(FadeIn(head), run_time=0.4)
        g = VGroup()
        for name, color in (("small.en", TOK_C), ("tiny.en", GREY_B)):
            w = N[name]["word_error_rate"]
            g.add(VGroup(Text(name, font=MONO, font_size=22, color=color),
                         Rectangle(width=w * 100 * 1.2, height=0.45, stroke_width=0, fill_color=color, fill_opacity=0.9),
                         Text(f"{w * 100:.1f}% of words wrong", font=MONO, font_size=20)).arrange(RIGHT, buff=0.25))
        g.arrange(DOWN, aligned_edge=LEFT, buff=0.35).move_to([0, 1.0, 0])
        self.at("small")
        self.play(FadeIn(g[0]), run_time=0.4)
        self.at("tiny")
        self.play(FadeIn(g[1]), run_time=0.4)
        top = [e for e in N["small.en"]["top_errors"] if e[0].split(" → ")[0] and e[0].split(" → ")[1]][:5]
        rows = VGroup(*[Text(f"{e[0]}   ×{e[1]}", font=MONO, font_size=20, color=GREY_A) for e in top]).arrange(DOWN, aligned_edge=LEFT, buff=0.12)
        rl = small("most frequent (said → heard):", GREY_B, 26)
        box = VGroup(rl, rows).arrange(DOWN, aligned_edge=LEFT, buff=0.2).move_to([0, -1.6, 0])
        self.at("favorite")
        self.play(FadeIn(box), run_time=0.5)
        self.play(Indicate(rows[0], color=YELLOW), run_time=0.6)
        self.end_section()

    # ------------------------------------------------------------------ 11. noise
    def noise(self):
        self.section(11)
        self.clear_stage()
        Nz = R["noise"]
        keys = ["clean", "20", "10", "5", "0", "-5"]
        ax = Axes(x_range=[0, 5, 1], y_range=[0, 20, 5], x_length=8, y_length=3.2, tips=False, axis_config={"color": GREY_D}).move_to([0, 0.6, 0])
        xl = VGroup(*[Text(k if k == "clean" else f"{k} dB", font=MONO, font_size=16, color=GREY_B).next_to(ax.c2p(i, 0), DOWN, buff=0.15) for i, k in enumerate(keys)])
        yl = VGroup(*[Text(f"{v}%", font=MONO, font_size=16, color=GREY_B).next_to(ax.c2p(0, v), LEFT, buff=0.1) for v in (0, 10, 20)])
        pts = [ax.c2p(i, Nz[k] * 100) for i, k in enumerate(keys)]
        ln = VMobject(stroke_color=TOK_C, stroke_width=4).set_points_as_corners(pts)
        dots = VGroup(*[Dot(p, color=TOK_C) for p in pts])
        vals = VGroup(*[Text(f"{Nz[k] * 100:.0f}%", font=MONO, font_size=18).next_to(p, UP, buff=0.12) for k, p in zip(keys, pts)])
        title = Text("word errors as white noise gets louder (20 sentences)", font_size=24).to_edge(UP, buff=0.4)
        self.at("noise")
        self.play(FadeIn(title), Create(ax), FadeIn(xl), FadeIn(yl), run_time=0.5)
        self.at("near")
        self.play(Create(VMobject().set_points_as_corners(pts[:4]).set_stroke(TOK_C, 4)), FadeIn(dots[:4]), FadeIn(vals[:4]), run_time=0.8)
        self.at("6", "six")
        self.play(Create(VMobject().set_points_as_corners(pts[3:5]).set_stroke(TOK_C, 4)), FadeIn(dots[4]), FadeIn(vals[4]), run_time=0.5)
        eq = small("0 dB: noise as loud as the voice", YELLOW, 28).next_to(dots[4], RIGHT, buff=0.3).shift(DOWN * 0.6)
        self.at("loud")
        self.play(FadeIn(eq), Create(VMobject().set_points_as_corners(pts[4:]).set_stroke(TOK_C, 4)), FadeIn(dots[5]), FadeIn(vals[5]), run_time=0.6)
        self.end_section()
        pos = np.array([0, -2.4, 0])
        secs = sf.info(f"{OUT}/noise_0.wav").duration
        lab = Text("0 dB", font_size=24, color=YELLOW).move_to(pos)
        track = Line(pos + 0.5 * DOWN + 2 * LEFT, pos + 0.5 * DOWN + 2 * RIGHT, color=GREY_D, stroke_width=6)
        bar = Line(track.get_start(), track.get_start() + 0.001 * RIGHT, color=YELLOW, stroke_width=6)
        icon = Text("♪", font_size=34, color=YELLOW).next_to(lab, LEFT, buff=0.25)
        self.play(FadeIn(lab), FadeIn(icon), FadeIn(track), run_time=0.3)
        print(f"CLIP noise_0 at {self.renderer.time:.2f}")
        self.add_sound(f"{OUT}/noise_0.wav")
        self.add(bar)
        self.play(bar.animate.put_start_and_end_on(track.get_start(), track.get_end()), run_time=secs, rate_func=linear)
        self.wait(0.3)

    # ------------------------------------------------------------------ 12. silence
    def silence(self):
        self.section(12)
        self.clear_stage()
        ns = R["no_speech"]
        flat = Line([-5, 1.6, 0], [5, 1.6, 0], color=GREY_C, stroke_width=3)
        fl = small("10 seconds of silence", GREY_B, 28).next_to(flat, UP, buff=0.2)
        self.at("silence")
        self.play(Create(flat), FadeIn(fl), run_time=0.5)
        out = Text(f"Whisper: “{ns['silence']['text']}”", font_size=40, color=RED_B).move_to([0, 0.2, 0])
        self.at("you")
        self.play(FadeIn(out, scale=1.3), run_time=0.4)
        pr = small(f"(its own no-speech probability: {ns['silence']['no_speech_prob'][0]:.2f})", GREY_B, 26).next_to(out, DOWN, buff=0.2)
        self.at("hallucinates")
        self.play(FadeIn(pr), run_time=0.4)
        vad = Text("with a voice detector first: “”  (nothing)", font_size=26, color=GREEN).move_to([0, -1.9, 0])
        self.at("voice")
        self.play(FadeIn(vad), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 13. code
    def code(self):
        self.section(13)
        self.clear_stage()
        code, hl = code_panel(CODE, font_size=16)
        code.move_to([0, 0.2, 0])
        self.play(FadeIn(code, shift=0.2 * UP), run_time=0.4)
        self.play(Create(hl), run_time=0.2)
        for i, w in ((0, "log"), (1, "encoder"), (4, "decode"), (5, "align")):
            self.at(w)
            self.play(highlight(hl, code, i), run_time=0.3)
        self.end_section()

    # ------------------------------------------------------------------ 14. finale
    def finale(self):
        self.section(14)
        self.clear_stage(run_time=0.4)
        parts = [("Part 1", "learning to generate", "autoencoders · VAEs · GANs", BLUE_C),
                 ("Part 2", "diffusion", "noise · denoisers · sampling · flow matching", ORANGE),
                 ("Part 3", "text to image", "conditioning · guidance · CLIP · latents · DiT", PURPLE_B),
                 ("Part 4", "sound", "spectrograms · codecs · Kokoro · Whisper", TEAL_C)]
        cards = VGroup()
        for a, b, c, col in parts:
            box = RoundedRectangle(width=2.9, height=1.9, corner_radius=0.15, color=col, fill_color=col, fill_opacity=0.08)
            t = VGroup(Text(a, font_size=18, color=col), Text(b, font_size=22), Text(c, font_size=13, color=GREY_B)).arrange(DOWN, buff=0.15)
            for line in t[1:]:
                line.scale_to_fit_width(min(line.width, 2.6))
            cards.add(VGroup(box, t.move_to(box)))
        cards.arrange(RIGHT, buff=0.25).move_to([0, 0.9, 0])
        self.at("series")
        self.play(LaggedStart(*[FadeIn(c, shift=0.2 * UP) for c in cards], lag_ratio=0.2), run_time=1.4)
        t = Text("Thanks for watching", font_size=40).move_to([0, -1.6, 0])
        self.at("thanks")
        self.play(FadeIn(t), run_time=0.6)
        self.end_section()
        finish(self, hold=2.0)
