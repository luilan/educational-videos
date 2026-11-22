"""How Generative Models Work: Images and Sound, episode 15 — Sound as Data.

Render from the repo root:  ./render.sh generative g15
Numbers, pictures and the four audio clips come from code/g15_sound (results.json, waveform.json, mel_edges.json,
out/*.png, out/clip_*.wav).
"""
import json

import numpy as np
from manim import *

from common import MONO, code_panel, finish, highlight, next_up_card
from gen_common import pic
from intro import play_token_intro
from g15_script import NEXT, TAGLINE, TITLE
from voiced_scene import VoicedScene

SERIES_LABEL = "HOW GENERATIVE MODELS WORK  ·  EPISODE 15"
CODE_DIR = "code/g15_sound"
OUT = f"{CODE_DIR}/out"
R = json.load(open(f"{CODE_DIR}/results.json"))
W = json.load(open(f"{CODE_DIR}/waveform.json"))
EDGES = json.load(open(f"{CODE_DIR}/mel_edges.json"))["edges_hz"]
WAVE_C, FREQ_C, MEL_C, PHASE_C = TEAL_C, ORANGE, PURPLE_B, YELLOW
CLIP_S = R["clip"]["seconds"]
CODE = """frames = [x[i:i + 600] * hann for i in range(0, len(x) - 600, 240)]
spec = abs(rfft(frames, 1024))        # 25 ms windows every 10 ms
mel = mel_filters @ spec ** 2         # 80 triangles, equal steps in mel
log_mel = log10(mel)                  # what Whisper reads"""


def small(text, color=GREY_B, size=30, scale=0.55):
    return Text(text, font_size=size, color=color).scale(scale)


def num(v):
    return f"{v:,.0f}"


def wave_bars(env, width=12.0, height=2.4, color=WAVE_C):
    env = np.asarray(env)
    xs = np.linspace(-width / 2, width / 2, len(env))
    s = height / 2 / env.max()
    return VGroup(*[Line([x, -v * s, 0], [x, v * s, 0], stroke_width=1.6, color=color) for x, v in zip(xs, env)])


class SoundVideo(VoicedScene):
    VIDEO = "g15"

    def construct(self):
        play_token_intro(self, TITLE, 15, TAGLINE, label=SERIES_LABEL)
        self.hook()            # 1
        self.waveform()        # 2
        self.closeup()         # 3
        self.sampling()        # 4
        self.listen_rates()    # 5
        self.spectrogram()     # 6
        self.reading()         # 7
        self.mel()             # 8
        self.phase()           # 9
        self.griffin_lim()     # 10
        self.vocoder()         # 11
        self.code()            # 12
        self.outro()           # 13

    def section(self, i):
        print(f"SECTION {i} starts at {self.renderer.time:.2f}")
        super().section(i)

    def clear_stage(self, *keep, run_time=0.6):
        self.play(*[FadeOut(m) for m in self.mobjects if m not in keep], run_time=run_time)

    def play_clip(self, name, label, pos, color=WHITE):
        """Play out/clip_<name>.wav with a label and a progress bar, between narration sections."""
        lab = Text(label, font_size=24, color=color).move_to(pos)
        track = Line(pos + 0.5 * DOWN + 2 * LEFT, pos + 0.5 * DOWN + 2 * RIGHT, color=GREY_D, stroke_width=6)
        bar = Line(track.get_start(), track.get_start() + 0.001 * RIGHT, color=color, stroke_width=6)
        icon = Text("♪", font_size=34, color=color).next_to(lab, LEFT, buff=0.25)
        self.play(FadeIn(lab), FadeIn(icon), FadeIn(track), run_time=0.3)
        print(f"CLIP {name} at {self.renderer.time:.2f}")
        self.add_sound(f"{OUT}/clip_{name}.wav")
        self.add(bar)
        self.play(bar.animate.put_start_and_end_on(track.get_start(), track.get_end()), run_time=CLIP_S, rate_func=linear)
        self.wait(0.3)
        return VGroup(lab, icon, track, bar)

    # ------------------------------------------------------------------ 1. hook
    def hook(self):
        self.section(1)
        chip = Text("PART 4 · SOUND", font_size=26, color=WAVE_C).to_edge(UP, buff=0.6)
        self.at("part")
        self.play(FadeIn(chip), run_time=0.4)
        vals = W["closeup"][::6][:18]
        nums = VGroup(*[Text(f"{v:+.3f}", font=MONO, font_size=20, color=WAVE_C) for v in vals]).arrange(RIGHT, buff=0.25)
        nums.scale_to_fit_width(12.5).move_to([0, 1.0, 0])
        self.at("list")
        self.play(LaggedStart(*[FadeIn(n) for n in nums], lag_ratio=0.05), run_time=1.0)
        rate = Text("24,000 numbers every second", font_size=30).move_to([0, 0.1, 0])
        self.at("24")
        self.play(FadeIn(rate), run_time=0.4)
        ax = Line([-5, -2, 0], [5, -2, 0], color=GREY_C)
        xs = np.linspace(-5, 5, 18)
        k = 1.1 / max(abs(v) for v in vals)
        dots = VGroup(*[Dot([x, -2 + v * k, 0], radius=0.06, color=WAVE_C) for x, v in zip(xs, vals)])
        stems = VGroup(*[Line([x, -2, 0], [x, -2 + v * k, 0], color=WAVE_C, stroke_width=2) for x, v in zip(xs, vals)])
        pl = small("each number: how far the air is pushed, at one instant", GREY_B, 28).next_to(ax, DOWN, buff=1.3)
        self.at("pushed")
        self.play(Create(ax), LaggedStart(*[Create(s) for s in stems], lag_ratio=0.03), FadeIn(dots), FadeIn(pl), run_time=0.9)
        self.end_section()

    # ------------------------------------------------------------------ 2. waveform
    def waveform(self):
        self.section(2)
        self.clear_stage()
        bars = wave_bars(W["envelope"]).move_to([0, 0.4, 0])
        txt = Text(f"“{R['waveform']['text']}”", font_size=18, color=GREY_B).scale_to_fit_width(12).to_edge(UP, buff=0.5)
        self.at("sentence")
        self.play(FadeIn(txt), Create(bars, lag_ratio=0.002), run_time=1.2)
        info = Text(f"{R['waveform']['seconds']:.1f} seconds · {num(R['waveform']['samples'])} numbers",
                    font=MONO, font_size=24).move_to([0, -1.9, 0])
        self.at("seven")
        self.play(FadeIn(info), run_time=0.4)
        hop = W["envelope_hop_s"]
        env = np.asarray(W["envelope"])
        n = len(env)
        loud = int(np.argmax(env))
        quiet = [i for i in range(n // 4, n) if env[max(0, i - 6):i + 6].max() < 0.02][0]
        def mark(i, color, text):
            x = -6 + 12 * i / (n - 1)
            r = Rectangle(width=0.5, height=2.8, color=color, stroke_width=2).move_to([x, 0.4, 0])
            return VGroup(r, small(text, color, 26).next_to(r, DOWN, buff=0.15))
        self.at("loud")
        ml = mark(loud, YELLOW, "loud: swings wide")
        self.play(Create(ml), run_time=0.4)
        self.at("pauses")
        mq = mark(quiet, GREEN, "pause: near zero")
        self.play(Create(mq), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 3. close-up
    def closeup(self):
        self.section(3)
        self.clear_stage()
        y = np.asarray(W["closeup"])
        ax = Axes(x_range=[0, 30, 5], y_range=[-0.6, 0.6, 0.3], x_length=11, y_length=3.6, tips=False,
                  axis_config={"color": GREY_C}).move_to([0, 0.2, 0])
        xl = VGroup(*[Text(f"{v} ms", font=MONO, font_size=16).next_to(ax.c2p(v, -0.6), DOWN, buff=0.15) for v in (0, 10, 20, 30)])
        t = np.arange(len(y)) / 24
        curve = ax.plot_line_graph(t, y, add_vertex_dots=False, line_color=WAVE_C, stroke_width=2.5)
        self.at("zoom")
        self.play(Create(ax), FadeIn(xl), run_time=0.4)
        self.at("milliseconds")
        self.play(Create(curve), run_time=1.2)
        f0 = R["pitch"]["median_hz"]
        # mark repeats: successive peaks of the close-up
        ac = np.correlate(y - y.mean(), y - y.mean(), "full")[len(y) - 1:]
        per = int(np.argmax(ac[60:200]) + 60)
        p0 = int(np.argmax(y[:per]))
        ticks = VGroup(*[DashedLine(ax.c2p(t[i], -0.55), ax.c2p(t[i], 0.58), color=YELLOW, stroke_width=1.5)
                         for i in range(p0, len(y), per)])
        self.at("repeats")
        self.play(LaggedStart(*[Create(k) for k in ticks], lag_ratio=0.1), run_time=0.7)
        lab = Text(f"one repeat every {1000 * per / 24000:.1f} ms → {24000 / per:.0f} per second", font_size=24,
                   color=YELLOW).to_edge(DOWN, buff=1.0)
        self.at("200")
        self.play(FadeIn(lab), run_time=0.4)
        lab2 = small(f"pitch · whole sentence: median {f0:.0f} Hz, most between {R['pitch']['p10_hz']:.0f} and "
                     f"{R['pitch']['p90_hz']:.0f} Hz", WHITE, 30).to_edge(DOWN, buff=0.45)
        self.at("pitch")
        self.play(FadeIn(lab2), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 4. sampling
    def sampling(self):
        self.section(4)
        self.clear_stage()
        ax = Axes(x_range=[0, 1, 0.25], y_range=[-1.2, 1.2, 1], x_length=11, y_length=2.6, tips=False,
                  axis_config={"color": GREY_C}).move_to([0, -0.4, 0])
        xl = VGroup(*[Text(f"{v:g} ms", font=MONO, font_size=16).next_to(ax.c2p(v, -1.2), DOWN, buff=0.12) for v in (0, 0.5, 1)])
        rule = Text("more than 2 samples per cycle", font_size=26).to_edge(UP, buff=0.5)
        self.at("two")
        self.play(FadeIn(rule), run_time=0.4)
        lim = Text("24,000 samples/s → up to 12,000 Hz", font=MONO, font_size=22, color=WAVE_C).next_to(rule, DOWN, buff=0.3)
        self.at("12")
        self.play(FadeIn(lim), run_time=0.4)
        tone = ax.plot(lambda x: np.sin(2 * np.pi * 5 * x), x_range=[0, 1, 0.002], color=FREQ_C, stroke_width=2.5)
        tl = Text("5,000 Hz tone", font_size=20, color=FREQ_C).next_to(ax, UP, buff=0.15).align_to(ax, LEFT)
        self.at("tone")
        self.play(Create(ax), FadeIn(xl), Create(tone), FadeIn(tl), run_time=0.8)
        ts = np.arange(9) / 8
        dots = VGroup(*[Dot(ax.c2p(x, np.sin(2 * np.pi * 5 * x)), radius=0.08, color=WHITE) for x in ts])
        dl = Text("8,000 samples/s", font_size=20).next_to(tl, RIGHT, buff=0.6)
        self.at("8")
        self.play(FadeIn(dots), FadeIn(dl), run_time=0.5)
        alias = ax.plot(lambda x: -np.sin(2 * np.pi * 3 * x), x_range=[0, 1, 0.002], color=RED_B, stroke_width=3)
        al = Text("what the samples say: 3,000 Hz", font_size=20, color=RED_B).next_to(dl, RIGHT, buff=0.6)
        self.at("3")
        self.play(tone.animate.set_stroke(opacity=0.3), Create(alias), FadeIn(al), run_time=0.8)
        nm = Text("aliasing → filter out what is too high, then sample", font_size=24, color=YELLOW).to_edge(DOWN, buff=0.5)
        self.at("aliasing")
        self.play(FadeIn(nm), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 5. listen: sample rates
    def listen_rates(self):
        self.section(5)
        self.clear_stage()
        e4 = R["sample_rates"]["energy_above_4000_hz"]
        head = Text("8,000 samples/s → nothing above 4,000 Hz", font_size=28).to_edge(UP, buff=0.6)
        self.at("8")
        self.play(FadeIn(head), run_time=0.4)
        bar = Rectangle(width=8, height=0.5, stroke_width=0, fill_color=WAVE_C, fill_opacity=0.8).move_to([0, 1.4, 0])
        cut = Rectangle(width=8 * e4, height=0.5, stroke_width=0, fill_color=RED_B, fill_opacity=0.9).align_to(bar, RIGHT).move_to(bar, aligned_edge=RIGHT)
        bl = small("all of the sentence's energy", WAVE_C, 26).next_to(bar, DOWN, buff=0.15).align_to(bar, LEFT)
        cl = small(f"above 4,000 Hz: {e4 * 100:.1f}%", RED_B, 26).next_to(cut, DOWN, buff=0.15).align_to(cut, RIGHT)
        self.at("6")
        self.play(FadeIn(bar), FadeIn(cut), FadeIn(bl), FadeIn(cl), run_time=0.5)
        self.end_section()
        a = self.play_clip("original", "24,000 samples per second", np.array([0, -0.6, 0]), WAVE_C)
        b = self.play_clip("8k", "8,000 samples per second", np.array([0, -2.2, 0]), RED_B)

    # ------------------------------------------------------------------ 6. spectrogram
    def spectrogram(self):
        self.section(6)
        self.clear_stage()
        bars = wave_bars(W["envelope"], width=11, height=1.2).move_to([0.4, 2.6, 0])
        self.at("raw")
        self.play(FadeIn(bars), run_time=0.4)
        win = Rectangle(width=11 * 0.025 / R["waveform"]["seconds"] * 6, height=1.4, color=YELLOW, stroke_width=2)
        win.move_to(bars.get_left() + 0.3 * RIGHT)
        wl = small("25 ms window, moved 10 ms at a time", YELLOW, 26).next_to(bars, DOWN, buff=0.15)
        self.at("windows")
        self.play(Create(win), FadeIn(wl), run_time=0.4)
        self.play(win.animate.shift(1.6 * RIGHT), run_time=0.8)
        sp = pic(f"{OUT}/spectrogram.png", 4.0)
        sp.stretch_to_fit_width(11).move_to([0.4, -0.9, 0])
        fl = small("Fourier transform of each window", FREQ_C, 26).next_to(wl, RIGHT, buff=0.5)
        self.at("fourier")
        self.play(FadeIn(fl), run_time=0.3)
        self.at("spectrogram")
        self.play(FadeIn(sp), run_time=0.8)
        ta = Arrow(sp.get_corner(DL) + 0.3 * DOWN, sp.get_corner(DL) + 0.3 * DOWN + 2 * RIGHT, buff=0, color=WHITE, stroke_width=3)
        tla = small("time", WHITE, 26).next_to(ta, RIGHT, buff=0.1)
        fa = Arrow(sp.get_corner(DL) + 0.3 * LEFT, sp.get_corner(DL) + 0.3 * LEFT + 2 * UP, buff=0, color=WHITE, stroke_width=3)
        fla = small("frequency\n0 to 12,000 Hz", WHITE, 24).next_to(fa, LEFT, buff=0.1)
        self.at("time")
        self.play(GrowArrow(ta), FadeIn(tla), run_time=0.3)
        self.at("frequency")
        self.play(GrowArrow(fa), FadeIn(fla), run_time=0.3)
        self.at("color")
        self.play(FadeIn(small("brighter = louder", YELLOW, 26).next_to(ta, RIGHT, buff=1.4)), run_time=0.3)
        self.end_section()

    # ------------------------------------------------------------------ 7. reading it
    def reading(self):
        self.section(7)
        keep = [m for m in self.mobjects if isinstance(m, ImageMobject)]
        sp = keep[0]
        self.play(*[FadeOut(m) for m in self.mobjects if m is not sp],
                  sp.animate.stretch_to_fit_height(5.4).stretch_to_fit_width(8.6).move_to([2.0, 0, 0]), run_time=0.6)
        h, w = sp.height, sp.width
        def at_(fx, fy):                       # fraction of time, fraction of 0–12 kHz
            return sp.get_corner(DL) + RIGHT * w * fx + UP * h * fy
        l1 = Text("harmonics:\n200, 400, 600 Hz …", font_size=22, color=WHITE).move_to([-4.6, -2.0, 0])
        b1 = Arrow(l1.get_right(), at_(0.385, 0.06), buff=0.1, color=WHITE, stroke_width=4)
        self.at("stripes")
        self.play(GrowArrow(b1), FadeIn(l1), run_time=0.5)
        bend = Ellipse(width=1.0, height=1.0, color=YELLOW, stroke_width=3).move_to(at_(0.405, 0.1))
        bl = Text("they bend\nwith the pitch", font_size=22, color=YELLOW).move_to([-4.6, -0.2, 0])
        ba = Arrow(bl.get_right(), bend.get_left(), buff=0.1, color=YELLOW, stroke_width=4)
        self.at("bend")
        self.play(Create(bend), FadeIn(bl), GrowArrow(ba), run_time=0.4)
        s0 = 0.1 / R["waveform"]["seconds"]
        hs = Rectangle(width=0.35, height=h * 0.6, color=FREQ_C, stroke_width=3).move_to(at_(s0 + 0.002, 0.65))
        hl = Text("the hiss of “s”", font_size=22, color=FREQ_C).move_to([-4.6, 1.6, 0])
        ha = Arrow(hl.get_right(), hs.get_left(), buff=0.1, color=FREQ_C, stroke_width=4)
        self.at("hiss")
        self.play(Create(hs), FadeIn(hl), GrowArrow(ha), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 8. mel
    def mel(self):
        self.section(8)
        self.clear_stage()
        ax = NumberLine(x_range=[0, 12000, 2000], length=11, include_numbers=False, color=GREY_C).move_to([0, 1.2, 0])
        xl = VGroup(*[Text(f"{v // 1000}k" if v else "0", font=MONO, font_size=16).next_to(ax.n2p(v), DOWN, buff=0.12)
                      for v in range(0, 12001, 2000)])
        hz = small("Hz", GREY_B, 26).next_to(ax, RIGHT, buff=0.2)
        self.at("ears")
        self.play(Create(ax), FadeIn(xl), FadeIn(hz), run_time=0.5)
        tris = VGroup()
        for i in range(len(EDGES) - 2):
            lo, c, hi = EDGES[i:i + 3]
            tris.add(Polygon(ax.n2p(lo), ax.n2p(c) + 1.4 * UP, ax.n2p(hi), color=MEL_C, stroke_width=1.2,
                             fill_color=MEL_C, fill_opacity=0.06))
        self.at("mel")
        self.play(LaggedStart(*[Create(t) for t in tris], lag_ratio=0.01), run_time=1.2)
        n1 = small(f"{R['mel']['first_filter_width_hz']:.0f} Hz wide", MEL_C, 28).next_to(ax.n2p(0), UP, buff=1.6).shift(0.6 * RIGHT)
        n2 = small(f"{R['mel']['last_filter_width_hz']:.0f} Hz wide", MEL_C, 28).next_to(ax.n2p(11500), UP, buff=1.6)
        lab = Text("80 filters, equally spaced in mel", font_size=24, color=MEL_C).to_edge(UP, buff=0.4)
        self.at("80")
        self.play(FadeIn(lab), run_time=0.3)
        self.at("52")
        self.play(FadeIn(n1), run_time=0.3)
        self.at("877")
        self.play(FadeIn(n2), run_time=0.3)
        m = pic(f"{OUT}/mel.png", 2.2)
        m.stretch_to_fit_width(11).move_to([0, -1.5, 0])
        ml = small("log-mel spectrogram: 80 rows, 100 columns per second", WHITE, 26).next_to(m, DOWN, buff=0.15)
        self.at("log")
        self.play(FadeIn(m), FadeIn(ml), run_time=0.6)
        cnt = Text("24,000 → 8,000 numbers per second", font=MONO, font_size=22, color=YELLOW).next_to(ml, DOWN, buff=0.15)
        self.at("turns")
        self.play(FadeIn(cnt), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 9. phase
    def phase(self):
        self.section(9)
        self.clear_stage()
        ax1 = Axes(x_range=[0, 2, 1], y_range=[-1.2, 1.2, 1], x_length=4.8, y_length=1.6, tips=False,
                   axis_config={"color": GREY_C}).move_to([-3.3, 1.6, 0])
        ax2 = ax1.copy().move_to([3.3, 1.6, 0])
        w1 = ax1.plot(lambda x: np.sin(2 * np.pi * x), color=WAVE_C)
        w2 = ax2.plot(lambda x: np.sin(2 * np.pi * x + 1.9), color=PHASE_C)
        kept = small("kept: how strong (same for both)", WAVE_C, 28).move_to([0, 2.9, 0])
        self.at("strong")
        self.play(Create(ax1), Create(w1), Create(ax2), Create(w2), FadeIn(kept), run_time=0.7)
        lost = small("lost: the phase, where each wave is in its cycle", PHASE_C, 28).next_to(VGroup(ax1, ax2), DOWN, buff=0.25)
        self.at("phase")
        self.play(FadeIn(lost), run_time=0.4)
        rnd = Text("random phases", font_size=24, color=RED_B).move_to([-3.3, -0.6, 0])
        self.at("random")
        self.play(FadeIn(rnd), run_time=0.3)
        wh = Text(f"Whisper: “{R['phase']['random_phase']['whisper'][:33]}…”", font_size=18).move_to([2.0, -0.6, 0])
        ok = small(f"word errors: {R['phase']['random_phase']['word_error_rate'] * 100:.0f}%", GREEN, 28).next_to(wh, DOWN, buff=0.15)
        self.at("whisper")
        self.play(FadeIn(wh), FadeIn(ok), run_time=0.4)
        self.end_section()
        self.play_clip("random_phase", "random phases", np.array([0, -2.4, 0]), RED_B)

    # ------------------------------------------------------------------ 10. Griffin-Lim
    def griffin_lim(self):
        self.section(10)
        self.clear_stage()
        mag = Text("magnitudes", font_size=22, color=WAVE_C)
        ph = Text("phase guess", font_size=22, color=PHASE_C)
        snd = Text("sound", font_size=22)
        stft = Text("its spectrogram", font_size=22, color=FREQ_C)
        pts = [[-4.2, 1.8, 0], [0, 1.8, 0], [4.2, 1.8, 0], [0, -0.2, 0]]
        boxes = VGroup()
        for t, p in zip((mag, snd, stft, ph), pts):
            b = SurroundingRectangle(t, buff=0.2, corner_radius=0.12, color=t.color)
            boxes.add(VGroup(b, t).move_to(p))
        arr = VGroup(Arrow(boxes[0].get_right(), boxes[1].get_left(), buff=0.15, color=GREY_B),
                     Arrow(boxes[1].get_right(), boxes[2].get_left(), buff=0.15, color=GREY_B),
                     CurvedArrow(boxes[2].get_bottom(), boxes[3].get_right(), angle=-0.8, color=GREY_B),
                     CurvedArrow(boxes[3].get_left(), boxes[0].get_bottom(), angle=-0.8, color=GREY_B))
        keep = small("keep its phase, put the true magnitudes back", GREY_B, 26).next_to(boxes[3], DOWN, buff=0.3)
        self.at("griffin")
        self.play(FadeIn(boxes[0]), FadeIn(boxes[3]), run_time=0.5)
        self.at("guess")
        self.play(GrowArrow(arr[0]), FadeIn(boxes[1]), GrowArrow(arr[1]), FadeIn(boxes[2]), Create(arr[2]), Create(arr[3]),
                  FadeIn(keep), run_time=0.8)
        P = R["phase"]
        d = VGroup(Text("mismatch with the true magnitudes", font_size=20, color=GREY_B),
                   Text(f"random phases   {P['random_phase']['spectral_distance']:.2f}", font=MONO, font_size=22, color=RED_B),
                   Text(f"Griffin-Lim ×60 {P['griffin_lim_60']['spectral_distance']:.2f}", font=MONO, font_size=22, color=GREEN)
                   ).arrange(DOWN, aligned_edge=LEFT, buff=0.15).move_to([0, -2.0, 0])
        x60 = Text("× 60", font_size=30, color=YELLOW).move_to([0, 0.9, 0])
        self.at("60")
        self.play(FadeIn(x60), FadeIn(d), run_time=0.5)
        self.end_section()
        self.play(*[FadeOut(m) for m in (d, keep)], run_time=0.3)
        self.play_clip("griffin_lim_60", "Griffin-Lim, 60 rounds", np.array([0, -2.2, 0]), GREEN)

    # ------------------------------------------------------------------ 11. vocoder
    def vocoder(self):
        self.section(11)
        self.clear_stage()
        m = pic(f"{OUT}/mel.png", 1.4).stretch_to_fit_width(3.6).move_to([-4.4, 0.4, 0])
        ml = small("spectrogram", MEL_C, 28).next_to(m, DOWN, buff=0.2)
        v = RoundedRectangle(width=3.4, height=1.4, corner_radius=0.15, color=YELLOW, fill_color=YELLOW, fill_opacity=0.1).move_to([0, 0.4, 0])
        vl = Text("vocoder\n(a trained model)", font_size=20, color=YELLOW, line_spacing=0.8).move_to(v)
        wb = wave_bars(W["envelope"][:140], width=3.6, height=1.4).move_to([4.4, 0.4, 0])
        wl = small("natural sound", WAVE_C, 28).next_to(wb, DOWN, buff=0.2)
        self.at("spectrogram")
        self.play(FadeIn(m), FadeIn(ml), run_time=0.4)
        self.at("vocoder")
        self.play(GrowArrow(Arrow(m.get_right(), v.get_left(), buff=0.15, color=GREY_B)), FadeIn(v), FadeIn(vl), run_time=0.5)
        self.play(GrowArrow(Arrow(v.get_right(), wb.get_left(), buff=0.15, color=GREY_B)), FadeIn(wb), FadeIn(wl), run_time=0.5)
        ep = Text("our narrator's vocoder: episode 17", font_size=24, color=WHITE).to_edge(DOWN, buff=0.8)
        self.at("narrator")
        self.play(FadeIn(ep), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 12. code
    def code(self):
        self.section(12)
        self.clear_stage()
        code, hl = code_panel(CODE, font_size=18)
        code.move_to([0, 0.2, 0])
        self.play(FadeIn(code, shift=0.2 * UP), run_time=0.4)
        self.play(Create(hl), run_time=0.2)
        self.at("windows")
        self.play(highlight(hl, code, 0), run_time=0.3)
        self.at("fourier")
        self.play(highlight(hl, code, 1), run_time=0.3)
        self.at("mel")
        self.play(highlight(hl, code, 2), run_time=0.3)
        self.end_section()

    # ------------------------------------------------------------------ 13. outro
    def outro(self):
        self.section(13)
        self.clear_stage(run_time=0.4)
        card_ = next_up_card(NEXT)
        self.at("next")
        self.play(FadeIn(card_, shift=0.2 * UP), run_time=0.6)
        self.end_section()
        finish(self, hold=1.5)
