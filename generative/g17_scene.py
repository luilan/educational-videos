"""How Generative Models Work: Images and Sound, episode 17 — Text to Speech: Inside Our Narrator.

Render from the repo root:  ./render.sh generative g17
Numbers, curves and the three audio clips come from code/g17_inside_our_narrator (results.json, curves.json, out/*.wav).
"""
import json

import numpy as np
import soundfile as sf
from manim import *

from common import MONO, code_panel, finish, highlight, next_up_card
from intro import play_token_intro
from g17_script import NEXT, TAGLINE, TITLE
from voiced_scene import VoicedScene

SERIES_LABEL = "HOW GENERATIVE MODELS WORK  ·  EPISODE 17"
CODE_DIR = "code/g17_inside_our_narrator"
OUT = f"{CODE_DIR}/out"
R = json.load(open(f"{CODE_DIR}/results.json"))
CV = json.load(open(f"{CODE_DIR}/curves.json"))
PH_C, DUR_C, F0_C, EN_C, SND_C, STY_C = TEAL_C, ORANGE, YELLOW, GREEN_C, BLUE_C, PURPLE_B
CODE = """ids = vocab[phonemize(text)]                  # 58 symbols
dur = duration_predictor(bert(ids), style[128:])  # frames per phoneme
x = repeat(encoder(ids), dur)                  # one vector per 25 ms frame
f0, energy = f0n_predictor(x, style[128:])     # 80 values a second
audio = decoder(x, f0, energy, style[:128])    # 24,000 samples a second"""


def small(text, color=GREY_B, size=30, scale=0.55):
    return Text(text, font_size=size, color=color).scale(scale)


def mil(n):
    return f"{n / 1e6:.1f}M"


def curve(values, width, height, color, lo=None, hi=None, stroke=3):
    v = np.asarray(values, float)
    lo = v.min() if lo is None else lo
    hi = v.max() if hi is None else hi
    xs = np.linspace(-width / 2, width / 2, len(v))
    ys = (v - lo) / (hi - lo) * height - height / 2
    return VMobject(stroke_color=color, stroke_width=stroke).set_points_as_corners([[x, y, 0] for x, y in zip(xs, ys)])


def style_bar(vec, width=11.0, height=0.55):
    v = np.asarray(vec)
    s = np.abs(v).max()
    cells = VGroup()
    w = width / len(v)
    for i, x in enumerate(v):
        c = interpolate_color(BLUE_D, ORANGE, (x / s + 1) / 2)
        cells.add(Rectangle(width=w, height=height, stroke_width=0, fill_color=c, fill_opacity=1).move_to([-width / 2 + (i + 0.5) * w, 0, 0]))
    return cells


class NarratorVideo(VoicedScene):
    VIDEO = "g17"

    def construct(self):
        play_token_intro(self, TITLE, 17, TAGLINE, label=SERIES_LABEL)
        self.hook()          # 1
        self.phonemes()      # 2
        self.style()         # 3
        self.durations()     # 4
        self.alignment()     # 5
        self.curves()        # 6
        self.flat()          # 7
        self.higher()        # 8
        self.swap()          # 9
        self.decoder()       # 10
        self.training()      # 11
        self.code()          # 12
        self.outro()         # 13

    def section(self, i):
        print(f"SECTION {i} starts at {self.renderer.time:.2f}")
        super().section(i)

    def clear_stage(self, *keep, run_time=0.6):
        self.play(*[FadeOut(m) for m in self.mobjects if m not in keep], run_time=run_time)

    def play_clip(self, name, label, pos, color=WHITE):
        """Play out/<name>.wav with a label and a progress bar, between narration sections."""
        secs = sf.info(f"{OUT}/{name}.wav").duration
        lab = Text(label, font_size=24, color=color).move_to(pos)
        track = Line(pos + 0.5 * DOWN + 2 * LEFT, pos + 0.5 * DOWN + 2 * RIGHT, color=GREY_D, stroke_width=6)
        bar = Line(track.get_start(), track.get_start() + 0.001 * RIGHT, color=color, stroke_width=6)
        icon = Text("♪", font_size=34, color=color).next_to(lab, LEFT, buff=0.25)
        self.play(FadeIn(lab), FadeIn(icon), FadeIn(track), run_time=0.3)
        print(f"CLIP {name} at {self.renderer.time:.2f}")
        self.add_sound(f"{OUT}/{name}.wav")
        self.add(bar)
        self.play(bar.animate.put_start_and_end_on(track.get_start(), track.get_end()), run_time=secs, rate_func=linear)
        self.wait(0.3)
        return VGroup(lab, icon, track, bar)

    # ------------------------------------------------------------------ 1. hook
    def hook(self):
        self.section(1)
        P = R["parameters"]
        title = Text("Kokoro v1.0", font_size=40).to_edge(UP, buff=0.6)
        total = Text(f"{P['total']:,} parameters", font=MONO, font_size=26, color=SND_C).next_to(title, DOWN, buff=0.25)
        self.at("kokoro")
        self.play(FadeIn(title), run_time=0.4)
        self.at("82")
        self.play(FadeIn(total), run_time=0.4)
        parts = [("phoneme Bert", P["bert"] + P["bert_encoder"], PH_C), ("prosody predictor", P["predictor"], DUR_C),
                 ("text encoder", P["text_encoder"], STY_C), ("decoder", P["decoder"], SND_C)]
        W = 11.0
        bars, x = VGroup(), -W / 2
        for name, n, c in parts:
            w = W * n / P["total"]
            r = Rectangle(width=w, height=0.7, stroke_width=1.5, stroke_color=BLACK, fill_color=c, fill_opacity=0.85)
            r.move_to([x + w / 2, -0.3, 0])
            lab = VGroup(small(name, c, 26), Text(mil(n), font=MONO, font_size=16, color=c)).arrange(DOWN, buff=0.08)
            lab.next_to(r, DOWN if len(bars) % 2 == 0 else UP, buff=0.2)
            bars.add(VGroup(r, lab))
            x += w
        self.at("runs")
        self.play(LaggedStart(*[FadeIn(b) for b in bars], lag_ratio=0.15), run_time=1.0)
        flow = Text("text → phonemes → durations → pitch and energy → decoder → sound", font_size=22,
                    color=GREY_A).to_edge(DOWN, buff=0.6)
        self.at("open")
        self.play(FadeIn(flow), run_time=0.5)
        self.end_section()

    # ------------------------------------------------------------------ 2. phonemes
    def phonemes(self):
        self.section(2)
        self.clear_stage()
        head = Text("text → phonemes (symbols for sounds)", font_size=28).to_edge(UP, buff=0.5)
        self.play(FadeIn(head), run_time=0.4)
        sp = R["phonemes"]["spelling"]
        rows = VGroup()
        for w in ("though", "tough", "through", "thought"):
            rows.add(VGroup(Text(w, font_size=30), Text("→", font_size=26, color=GREY_B),
                            Text(sp[w], font=MONO, font_size=30, color=PH_C)).arrange(RIGHT, buff=0.35))
        rows.arrange(RIGHT, buff=0.7).scale_to_fit_width(12.4).move_to([0, 1.3, 0])
        for i, w in enumerate(("though", "tough", "through", "thought")):
            self.at(w)
            self.play(FadeIn(rows[i]), run_time=0.3)
        self.at("share")
        ough = VGroup(*[SurroundingRectangle(r[0][-4:] if w != "thought" else r[0][2:6], buff=0.05, color=YELLOW)
                        for r, w in zip(rows, ("though", "tough", "through", "thought"))])
        self.play(Create(ough), run_time=0.5)
        Pm = R["phonemes"]
        txt = Text(Pm["text"], font_size=30).move_to([0, -0.4, 0])
        n1 = small(f"{len(Pm['text'])} characters", GREY_B, 28).next_to(txt, DOWN, buff=0.15)
        ph = Text(Pm["phonemes"], font=MONO, font_size=30, color=PH_C)
        ph.scale_to_fit_width(min(ph.width, 12.0)).move_to([0, -1.8, 0])
        n2 = small(f"{Pm['symbols']} symbols  ·  ˈ ˌ = stress", PH_C, 28).next_to(ph, DOWN, buff=0.15)
        self.at("sentences")
        self.play(FadeIn(txt), run_time=0.4)
        self.at("51")
        self.play(FadeIn(n1), run_time=0.3)
        self.at("become")
        self.play(TransformFromCopy(txt, ph), run_time=0.8)
        self.at("58")
        self.play(FadeIn(n2), run_time=0.3)
        self.end_section()

    # ------------------------------------------------------------------ 3. style vector
    def style(self):
        self.section(3)
        self.clear_stage()
        bar = style_bar(CV["style"]).move_to([0, 0.6, 0])
        lab = Text("af_heart: one style vector, 256 numbers", font_size=26).next_to(bar, UP, buff=0.5)
        self.at("vector")
        self.play(FadeIn(lab), LaggedStart(*[FadeIn(c) for c in bar], lag_ratio=0.003), run_time=1.0)
        left, right = VGroup(*bar[:128]), VGroup(*bar[128:])
        bl = Brace(left, DOWN, color=SND_C)
        tl = VGroup(Text("how it sounds", font_size=24, color=SND_C), small("128 numbers → decoder", GREY_B, 26)).arrange(DOWN, buff=0.1).next_to(bl, DOWN, buff=0.15)
        br = Brace(right, DOWN, color=DUR_C)
        tr = VGroup(Text("how it speaks", font_size=24, color=DUR_C), small("128 numbers → rhythm, melody", GREY_B, 26)).arrange(DOWN, buff=0.1).next_to(br, DOWN, buff=0.15)
        self.at("first")
        self.play(GrowFromCenter(bl), FadeIn(tl), run_time=0.5)
        self.at("second")
        self.play(GrowFromCenter(br), FadeIn(tr), run_time=0.5)
        self.end_section()

    # ------------------------------------------------------------------ 4. durations
    def durations(self):
        self.section(4)
        self.clear_stage()
        D = R["durations"]
        d = np.array(D["frames"])
        labels = D["labels"]
        bert = VGroup(RoundedRectangle(width=4.6, height=0.8, corner_radius=0.12, color=PH_C),
                      Text("phoneme Bert: reads in context", font_size=20, color=PH_C))
        bert[1].move_to(bert[0])
        bert.to_edge(UP, buff=0.5)
        self.at("bert")
        self.play(FadeIn(bert), run_time=0.4)
        W, H = 12.4, 2.8
        w = W / len(d)
        base_y = -1.6
        bars = VGroup(*[Rectangle(width=w * 0.85, height=max(0.02, H * v / d.max()), stroke_width=0,
                                  fill_color=DUR_C, fill_opacity=0.9).move_to([-W / 2 + (i + 0.5) * w, base_y, 0], aligned_edge=DOWN)
                        for i, v in enumerate(d)])
        labs = VGroup(*[Text(l if len(l) == 1 and l != " " else "·", font=MONO, font_size=14, color=PH_C).move_to([-W / 2 + (i + 0.5) * w, base_y - 0.22, 0])
                        for i, l in enumerate(labels)])
        unit = small("frames per phoneme  ·  1 frame = 25 ms", DUR_C, 28).next_to(bert, DOWN, buff=0.3)
        self.at("prediction")
        self.play(FadeIn(labs), FadeIn(unit), run_time=0.4)
        self.at("lasts")
        self.play(LaggedStart(*[GrowFromEdge(b, DOWN) for b in bars], lag_ratio=0.02), run_time=1.2)
        twos = VGroup(*[bars[i] for i, v in enumerate(d) if v == 2])
        self.at("most")
        self.play(twos.animate.set_fill(YELLOW), run_time=0.4)
        t2 = small(f"{int((d[1:-1] == 2).sum())} of {len(d) - 2} phonemes: 2 frames = 50 ms", YELLOW, 28).move_to([0, 0.85, 0])
        self.play(FadeIn(t2), run_time=0.3)
        i_s = len(d) - 3
        self.at("final")
        self.play(twos.animate.set_fill(DUR_C), Indicate(bars[i_s], color=RED_B), run_time=0.6)
        t13 = small(f"final s: {d[i_s]} frames", RED_B, 28).next_to(bars[i_s], UP, buff=0.12).shift(LEFT * 0.9)
        self.play(FadeIn(t13), run_time=0.3)
        tot = Text(f"{int(d.sum())} frames = {d.sum() * 0.025:.2f} s", font=MONO, font_size=24).to_edge(DOWN, buff=0.35)
        self.at("whole")
        self.play(FadeIn(tot), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 5. alignment
    def alignment(self):
        self.section(5)
        self.clear_stage()
        D = R["durations"]
        # the word "narrator": its phonemes and durations
        w = R["words"][1]
        start = int(round(w["start"] / 0.025))
        n = len(w["phonemes"])
        k0 = 1 + R["phonemes"]["phonemes"].index(w["phonemes"])
        syms, durs = D["labels"][k0:k0 + n], D["frames"][k0:k0 + n]
        cols = [PH_C, DUR_C, STY_C, SND_C, YELLOW, GREEN_C, RED_B, PINK]
        top = VGroup(*[VGroup(Square(0.62, color=cols[i % len(cols)], fill_opacity=0.25),
                              Text(s, font=MONO, font_size=26)) for i, s in enumerate(syms)]).arrange(RIGHT, buff=0.12)
        for b in top:
            b[1].move_to(b[0])
        top.move_to([0, 1.6, 0])
        tl = small("“narrator”: one vector per phoneme", GREY_B, 28).next_to(top, UP, buff=0.3)
        self.at("each")
        self.play(FadeIn(tl), FadeIn(top), run_time=0.5)
        cells = VGroup()
        for i, (s, k) in enumerate(zip(syms, durs)):
            for _ in range(k):
                cells.add(VGroup(Rectangle(width=0.42, height=0.62, color=cols[i % len(cols)], fill_opacity=0.25, stroke_width=2),
                                 Text(s, font=MONO, font_size=18)))
        cells.arrange(RIGHT, buff=0.04).move_to([0, -0.4, 0])
        for c in cells:
            c[1].move_to(c[0])
        self.at("copied")
        srcs = []
        j = 0
        for i, k in enumerate(durs):
            srcs += [top[i]] * k
        self.play(*[TransformFromCopy(srcs[j], cells[j]) for j in range(len(cells))], run_time=1.2)
        bl = small(f"{sum(durs)} frames × 25 ms", DUR_C, 28).next_to(cells, DOWN, buff=0.3)
        self.play(FadeIn(bl), run_time=0.3)
        arrow = Arrow([-5.5, -2.4, 0], [5.5, -2.4, 0], color=GREY_B, buff=0)
        at = small("time →", GREY_B, 28).next_to(arrow, DOWN, buff=0.1)
        self.at("time")
        self.play(GrowArrow(arrow), FadeIn(at), run_time=0.5)
        self.end_section()

    # ------------------------------------------------------------------ 6. pitch and energy
    def curve_panel(self):
        f0 = np.array(CV["f0"])
        lo, hi = 0, 340
        ax = Axes(x_range=[0, len(f0) / 80, 0.5], y_range=[lo, hi, 100], x_length=11, y_length=2.6, tips=False,
                  axis_config={"color": GREY_D}).move_to([0, 0.9, 0])
        yl = VGroup(*[Text(f"{v} Hz", font=MONO, font_size=14, color=GREY_B).next_to(ax.c2p(0, v), LEFT, buff=0.1) for v in (100, 200, 300)])
        return ax, yl, lo, hi

    def f0_curve(self, ax, values, color, stroke=3):
        v = np.asarray(values, float)
        t = np.arange(len(v)) / 80
        pts = [ax.c2p(x, max(0, y)) for x, y in zip(t, v)]
        return VMobject(stroke_color=color, stroke_width=stroke).set_points_as_corners(pts)

    def curves(self):
        self.section(6)
        self.clear_stage()
        ax, yl, lo, hi = self.curve_panel()
        self.ax, self.yl = ax, yl
        f0 = self.f0_curve(ax, CV["f0"], F0_C)
        lab = Text("pitch", font_size=24, color=F0_C).next_to(ax, UP, buff=0.1).align_to(ax, LEFT)
        rate = small(f"{len(CV['f0'])} values in {R['audio']['seconds']:.2f} s = 80 a second", GREY_B, 26).next_to(lab, RIGHT, buff=0.5)
        self.at("80")
        self.play(Create(ax), FadeIn(yl), FadeIn(rate), run_time=0.5)
        self.at("pitch")
        self.play(FadeIn(lab), Create(f0), run_time=1.0)
        med = DashedLine(ax.c2p(0, R["pitch"]["median_hz"]), ax.c2p(len(CV["f0"]) / 80, R["pitch"]["median_hz"]), color=GREY_B)
        ml = small(f"median {R['pitch']['median_hz']:.0f} Hz", GREY_A, 26).next_to(med, RIGHT, buff=0.1).shift(LEFT * 1.6 + UP * 0.25)
        self.at("190")
        self.play(Create(med), FadeIn(ml), run_time=0.4)
        en = np.array(CV["energy"])
        e = curve(en, 11, 1.4, EN_C).move_to([0, -2.1, 0])
        el = Text("energy", font_size=24, color=EN_C).next_to(e, UP, buff=0.1).align_to(ax, LEFT)
        self.at("energy")
        self.play(FadeIn(el), Create(e), run_time=1.0)
        self.f0_line = f0
        self.end_section()

    # ------------------------------------------------------------------ 7. flat pitch
    def flat(self):
        self.section(7)
        keep = [self.ax, self.yl, self.f0_line]
        self.clear_stage(*keep)
        ed = R["edits"]
        lab = Text("edit before the decoder", font_size=26).to_edge(UP, buff=0.4)
        self.at("edit")
        self.play(FadeIn(lab), run_time=0.3)
        flat = self.f0_curve(self.ax, CV["flat_f0"], RED_B)
        self.at("flatten")
        self.play(self.f0_line.animate.set_stroke(opacity=0.3), Create(flat), run_time=0.8)
        wh = small(f"Whisper: “{ed['flat_pitch']['whisper']}”  ·  word errors {ed['flat_pitch']['word_error_rate'] * 100:.0f}%", GREEN, 28).move_to([0, -1.2, 0])
        self.at("whisper")
        self.play(FadeIn(wh), run_time=0.4)
        self.end_section()
        clip = self.play_clip("flat_pitch", "flat pitch", np.array([0, -2.4, 0]), RED_B)
        self.play(FadeOut(clip), FadeOut(flat), FadeOut(wh), FadeOut(lab), run_time=0.3)

    # ------------------------------------------------------------------ 8. higher pitch
    def higher(self):
        self.section(8)
        ed = R["edits"]
        hi = self.f0_curve(self.ax, CV["higher_f0"], PINK)
        lab = Text("pitch × 1.5", font_size=26, color=PINK).to_edge(UP, buff=0.4)
        self.at("multiply")
        self.play(FadeIn(lab), Create(hi), run_time=0.8)
        m = small(f"measured: {ed['original']['measured_median_pitch_hz']:.0f} Hz → {ed['higher_pitch']['measured_median_pitch_hz']:.0f} Hz",
                  WHITE, 30).move_to([0, -1.2, 0])
        self.at("204")
        self.play(FadeIn(m), run_time=0.4)
        same = small(f"same words, same {ed['higher_pitch']['seconds']:.2f} s", GREY_B, 28).next_to(m, DOWN, buff=0.15)
        self.at("same")
        self.play(FadeIn(same), run_time=0.3)
        self.end_section()
        self.play_clip("higher_pitch", "pitch × 1.5", np.array([0, -2.6, 0]), PINK)

    # ------------------------------------------------------------------ 9. another voice
    def swap(self):
        self.section(9)
        self.clear_stage()
        ours = style_bar(CV["style"]).move_to([0, 1.7, 0])
        mixed = VGroup(*style_bar(CV["other_style"])[:128], *style_bar(CV["style"])[128:]).move_to([0, 0.2, 0])
        l1 = small("af_heart", GREY_B, 28).next_to(ours, UP, buff=0.15)
        self.play(FadeIn(ours), FadeIn(l1), run_time=0.4)
        l2 = VGroup(Text("am_michael's first half", font_size=22, color=SND_C), Text("+ our second half", font_size=22, color=DUR_C)).arrange(RIGHT, buff=0.3).next_to(mixed, UP, buff=0.15)
        self.at("swap")
        self.play(TransformFromCopy(ours, mixed), FadeIn(l2), run_time=0.9)
        ed = R["edits"]
        st = VGroup(Text(f"pitch {ed['other_voice_sound']['measured_median_pitch_hz']:.0f} Hz  ·  {ed['other_voice_sound']['seconds']:.2f} s", font=MONO, font_size=22),
                    small(f"(swap both halves: {ed['other_voice_all']['measured_median_pitch_hz']:.0f} Hz, {ed['other_voice_all']['seconds']:.2f} s)", GREY_B, 26)
                    ).arrange(DOWN, buff=0.15).move_to([0, -1.0, 0])
        self.at("pitch")
        self.play(FadeIn(st[0]), run_time=0.4)
        self.at("only")
        self.play(FadeIn(st[1]), run_time=0.3)
        self.end_section()
        self.play_clip("other_voice_sound", "another voice's sound, our melody", np.array([0, -2.6, 0]), SND_C)

    # ------------------------------------------------------------------ 10. decoder
    def decoder(self):
        self.section(10)
        self.clear_stage()
        P = R["parameters"]
        box = RoundedRectangle(width=6.6, height=1.0, corner_radius=0.15, color=SND_C, fill_color=SND_C, fill_opacity=0.12).move_to([0, 2.6, 0])
        bl = Text(f"decoder · {mil(P['decoder'])} parameters ({P['decoder'] / P['total'] * 100:.0f}%)", font_size=22, color=SND_C).move_to(box)
        self.at("decoder")
        self.play(FadeIn(box), FadeIn(bl), run_time=0.4)
        har = np.array(CV["harmonic_source"])
        src = curve(har, 5.2, 1.2, F0_C, stroke=2).move_to([-3.4, 0.9, 0])
        sl = small("source: sine waves at the pitch", F0_C, 26).next_to(src, UP, buff=0.12)
        self.at("sine")
        self.play(Create(src), FadeIn(sl), run_time=0.8)
        up = small("× 2 · × 10 · × 6 → 4,800 steps a second", GREY_A, 26).move_to([3.2, 1.2, 0])
        self.at("upsamples")
        self.play(FadeIn(up), run_time=0.4)
        # 11 magnitudes + 11 phases per 5 samples
        mags = VGroup(*[Rectangle(width=0.28, height=0.2 + 0.9 * abs(np.sin(i * 1.3)) * np.exp(-i / 5), stroke_width=0, fill_color=ORANGE, fill_opacity=0.9) for i in range(11)]).arrange(RIGHT, buff=0.05, aligned_edge=DOWN)
        phs = VGroup(*[Circle(radius=0.12, color=PURPLE_B, stroke_width=2).add(Line(ORIGIN, 0.12 * np.array([np.cos(i * 2.1), np.sin(i * 2.1), 0]), color=PURPLE_B, stroke_width=2)) for i in range(11)]).arrange(RIGHT, buff=0.06)
        grp = VGroup(VGroup(small("strength", ORANGE, 24), mags).arrange(RIGHT, buff=0.2),
                     VGroup(small("phase", PURPLE_B, 24), phs).arrange(RIGHT, buff=0.2)).arrange(DOWN, aligned_edge=LEFT, buff=0.25)
        grp.move_to([3.2, 0.0, 0])
        gl = small("11 frequencies, every 5 samples", GREY_A, 26).next_to(grp, DOWN, buff=0.15)
        self.at("strength")
        self.play(FadeIn(grp[0]), run_time=0.3)
        self.at("phase")
        self.play(FadeIn(grp[1]), run_time=0.3)
        self.at("11")
        self.play(FadeIn(gl), run_time=0.3)
        audio = np.array(CV["audio"])
        wav = curve(audio, 11, 1.3, TEAL_C, stroke=2).move_to([0, -2.4, 0])
        il = small("inverse Fourier transform → 24,000 samples a second (30 ms shown)", TEAL_C, 26).next_to(wav, UP, buff=0.1)
        self.at("inverse")
        self.play(Create(wav), FadeIn(il), run_time=0.8)
        note = small("episode 15: Griffin-Lim guessed the phase · here it is predicted", YELLOW, 26).move_to([-3.2, -0.6, 0])
        self.at("griffin")
        self.play(FadeIn(note), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 11. training and speed
    def training(self):
        self.section(11)
        self.clear_stage()
        g = VGroup(RoundedRectangle(width=2.8, height=1.0, corner_radius=0.15, color=SND_C), Text("Kokoro", font_size=24, color=SND_C))
        j = VGroup(RoundedRectangle(width=3.4, height=1.0, corner_radius=0.15, color=RED_B), Text("judges", font_size=24, color=RED_B))
        real = small("real recordings", GREEN, 28)
        for m in (g, j):
            m[1].move_to(m[0])
        g.move_to([-4.2, 1.7, 0])
        j.move_to([2.6, 1.7, 0])
        real.move_to([2.6, 3.0, 0])
        q = small("real or generated?", RED_B, 28).next_to(j, DOWN, buff=0.2)
        self.at("judges")
        self.play(FadeIn(g), FadeIn(j), GrowArrow(Arrow(g.get_right(), j.get_left(), buff=0.15, color=GREY_B)), run_time=0.6)
        self.at("real")
        self.play(FadeIn(real), GrowArrow(Arrow(real.get_bottom(), j.get_top(), buff=0.08, color=GREEN)), FadeIn(q), run_time=0.5)
        ep = small("episode 4: the forger and the judge", GREY_B, 28).move_to([-1, 0.4, 0])
        self.at("forger")
        self.play(FadeIn(ep), run_time=0.4)
        T = R["timing_s"]
        tot = sum(T.values())
        W = 11.0
        bars, x = VGroup(), -W / 2
        for key, name, c in (("bert", "Bert", PH_C), ("predictor", "predictor", DUR_C), ("text_encoder", "text enc.", STY_C), ("decoder", "decoder", SND_C)):
            w = W * T[key] / tot
            r = Rectangle(width=max(w, 0.02), height=0.6, stroke_width=1, stroke_color=BLACK, fill_color=c, fill_opacity=0.85).move_to([x + w / 2, -1.6, 0])
            bars.add(r)
            if w > 1.2:
                bars.add(Text(f"{name} {T[key]:.2f} s", font_size=18, color=BLACK).move_to(r))
            x += w
        head = small(f"{R['audio']['seconds']:.2f} s of speech took {tot:.2f} s on this CPU (8 threads)", WHITE, 30).move_to([0, -0.7, 0])
        self.at("cpu")
        self.play(FadeIn(head), LaggedStart(*[FadeIn(b) for b in bars], lag_ratio=0.1), run_time=0.8)
        dl = small(f"decoder: {T['decoder'] / tot * 100:.0f}% of the time", SND_C, 28).move_to([0, -2.5, 0])
        self.at("decoder")
        self.play(FadeIn(dl), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 12. code
    def code(self):
        self.section(12)
        self.clear_stage()
        code, hl = code_panel(CODE, font_size=17)
        code.move_to([0, 0.2, 0])
        self.play(FadeIn(code, shift=0.2 * UP), run_time=0.4)
        self.play(Create(hl), run_time=0.2)
        for i, w in enumerate(("phonemes", "durations", "repeat", "predict", "decode")):
            self.at(w)
            self.play(highlight(hl, code, i), run_time=0.3)
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
