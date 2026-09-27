"""Video 9 — The Transformer Block.

Render from the repo root:  ./render.sh how-llms-work v09
"""
import numpy as np
from manim import *

from common import code_panel, finish, highlight, next_up_card, token
from intro import play_token_intro
from v09_script import NEXT, TAGLINE, TITLE
from voiced_scene import VoicedScene

MONO = "DejaVu Sans Mono"
ATTN_COLOR = YELLOW
MLP_COLOR = GREEN
LN_COLOR = PURPLE_B
EARLY_COLOR = TEAL_C
LATE_COLOR = PURPLE_B
BLOCK_TINTS = [BLUE_C, TEAL_C, GREEN_C, GOLD_C, RED_C, PURPLE_B,
               YELLOW_C, PINK, "#6F7BF7", ORANGE, MAROON_C, LIGHT_BROWN]

# Residual stream geometry (sections 2-8)
S_Y, S_H = -0.5, 0.7           # centre line and band height
S_X0, S_X1 = -6.5, 6.5         # band ends; the right end is an arrow tip
TIP = 0.4
BOX_Y = 1.35                   # height of the attention / MLP branches
VEC_X, PRED_X = -5.85, 5.75    # token vector at the start, prediction at the end
LABEL_Y = S_Y - 0.95           # labels under the stream ends
# Each half: branch-off x, box centre x, box width, add-node x
HALF_A = dict(xr=-4.3, bx=-2.0, bw=2.3, xa=-0.35)
HALF_B = dict(xr=0.6, bx=2.35, bw=1.7, xa=3.7)
BLOCK_XS = [-4.73 + 0.86 * k for k in range(12)]
VEC0 = [0.7, -0.4, 0.2, 0.9, -0.8]

CODE = """def layer_norm(x, gain, bias, eps=1e-5):
    mean = x.mean(-1, keepdims=True)
    var = x.var(-1, keepdims=True)
    return gain * (x - mean) / np.sqrt(var + eps) + bias

def block(x, p):
    x = x + attention(layer_norm(x, *p.ln1), *p.attn)   # tokens talk
    x = x + mlp(layer_norm(x, *p.ln2), *p.mlp)          # each token thinks
    return x

for p in blocks:  # 12 blocks in GPT-2 small
    x = block(x, p)"""


# ---------------------------------------------------------------------------- helpers
def dim(color, a=0.25):
    """An opaque, darkened version of a colour (box fills that hide the wires behind them)."""
    return interpolate_color(ManimColor(BLACK), ManimColor(color), a)


def part_box(text, color, width, height=0.95, font_size=30):
    label = Text(text, font_size=font_size)
    box = RoundedRectangle(corner_radius=0.15, width=width, height=height, stroke_color=color,
                           stroke_width=3, fill_color=dim(color), fill_opacity=1)
    return VGroup(box, label.move_to(box)).set_z_index(4)


def stream_band(x0=S_X0, x1=S_X1, y=S_Y, h=S_H):
    band = Polygon([x0, y - h / 2, 0], [x1 - TIP, y - h / 2, 0], [x1, y, 0],
                   [x1 - TIP, y + h / 2, 0], [x0, y + h / 2, 0],
                   stroke_color=BLUE_C, stroke_width=1.5, stroke_opacity=0.8)
    band.set_fill([BLUE_E, BLUE_D, BLUE_C], opacity=0.55)
    band.set_sheen_direction(RIGHT)
    return band


def val_color(v):
    return interpolate_color(ManimColor(GREY_D), ManimColor(BLUE_C if v >= 0 else RED_C), min(1.0, abs(v)))


def vec_column(vals, x, y=S_Y, cell=0.24, w=0.36):
    col = VGroup(*[Rectangle(width=w, height=cell, stroke_color=WHITE, stroke_width=1.5,
                             fill_color=val_color(v), fill_opacity=1) for v in vals])
    return col.arrange(DOWN, buff=0).move_to([x, y, 0]).set_z_index(6)


def plus_node(x, y=S_Y, r=0.22):
    circ = Circle(radius=r, stroke_color=WHITE, stroke_width=3, fill_color=BLACK, fill_opacity=1)
    bars = VGroup(Line(0.6 * r * LEFT, 0.6 * r * RIGHT, stroke_width=3),
                  Line(0.6 * r * DOWN, 0.6 * r * UP, stroke_width=3))
    return VGroup(circ, bars).move_to([x, y, 0]).set_z_index(5)


def wire_arrow(start, end):
    return Arrow(start, end, buff=0, color=GREY_B, stroke_width=4, tip_length=0.2,
                 max_tip_length_to_length_ratio=0.5, max_stroke_width_to_length_ratio=20)


def read_path(xr, box):
    """Branch off the stream: up from the band at xr, then right into the box."""
    up = Line([xr, S_Y + S_H / 2, 0], [xr, BOX_Y + 0.02, 0], color=GREY_B, stroke_width=4)
    return VGroup(up, wire_arrow([xr, BOX_Y, 0], box.get_left())).set_z_index(3)


def write_path(box, xa, r=0.22):
    """Out of the box to the right, then down into the add node on the stream."""
    out = Line(box.get_right(), [xa + 0.02, BOX_Y, 0], color=GREY_B, stroke_width=4)
    return VGroup(out, wire_arrow([xa, BOX_Y, 0], [xa, S_Y + r, 0])).set_z_index(3)


def draw_path(path, run_time=0.7):
    return AnimationGroup(Create(path[0]), GrowArrow(path[1]), lag_ratio=1.0, run_time=run_time)


def flash_box(part, run_time=0.3):
    return part[0].animate(rate_func=there_and_back, run_time=run_time).set_stroke(YELLOW, width=7).scale(1.12)


def flash_plus(node, run_time=0.3):
    return node.animate(rate_func=there_and_back, run_time=run_time).set_stroke(YELLOW).scale(1.35)


def mini_block(x, y=S_Y, tint=GREY_B):
    outer = RoundedRectangle(corner_radius=0.08, width=0.62, height=1.3, stroke_color=tint,
                             stroke_width=2.5, fill_color=GREY_E, fill_opacity=0.95)
    bars = VGroup(*[RoundedRectangle(corner_radius=0.04, width=0.17, height=0.85, stroke_width=0,
                                     fill_color=c, fill_opacity=0.75) for c in (ATTN_COLOR, MLP_COLOR)])
    bars.arrange(RIGHT, buff=0.1).move_to(outer)
    return VGroup(outer, bars).move_to([x, y, 0]).set_z_index(4)


def flow_line(y, x0=S_X0, x1=S_X1 - TIP, color=BLUE_A, width=3, reverse=False):
    a, b = ([x1, y, 0], [x0, y, 0]) if reverse else ([x0, y, 0], [x1, y, 0])
    return Line(a, b, stroke_color=color, stroke_width=width).set_z_index(2)


def gradient_arrow(x0, x1, y, c0, c1, width=6):
    line = Line([x0, y, 0], [x1 - 0.22, y, 0], stroke_width=width)
    line.set_stroke([c0, c1])
    line.set_sheen_direction(RIGHT)
    tip = Triangle(stroke_width=0, fill_color=c1, fill_opacity=1).rotate(-PI / 2)
    tip.stretch_to_fit_width(0.26).stretch_to_fit_height(0.3).move_to([x1 - 0.13, y, 0])
    return VGroup(line, tip)


class TransformerBlockVideo(VoicedScene):
    VIDEO = "v09"

    def clear_anims(self, keep=()):
        return [FadeOut(m) for m in list(self.mobjects) if m not in keep]

    def construct(self):
        play_token_intro(self, TITLE, 9, TAGLINE)

        # 1. Intro: two halves -------------------------------------------------------------------------
        self.section(1)
        att1 = part_box("Attention", ATTN_COLOR, 2.6, 1.1, 34).move_to([-3.4, 1.0, 0])
        mlp1 = part_box("MLP", MLP_COLOR, 2.6, 1.1, 34).move_to([3.4, 1.0, 0])
        att_sub = Text("tokens talk", font_size=28, color=GREY_B).move_to([-3.4, -0.25, 0])
        mlp_sub = Text("each token thinks", font_size=28, color=GREY_B).move_to([3.4, -0.25, 0])
        dots_a = VGroup(*[Dot(radius=0.11, color=BLUE_C) for _ in range(4)]).arrange(RIGHT, buff=0.45)
        dots_a.move_to([-3.4, -1.75, 0])
        arcs_a = VGroup(*[ArcBetweenPoints(dots_a[3].get_top() + 0.06 * UP, dots_a[j].get_top() + 0.06 * UP,
                                           angle=PI * 0.7, color=ATTN_COLOR, stroke_width=3)
                          for j in range(3)])
        dots_m = dots_a.copy().move_to([3.4, -1.75, 0])
        rings_m = VGroup(*[Circle(radius=0.24, color=MLP_COLOR, stroke_width=3).move_to(d) for d in dots_m])
        self.at("attention")
        self.play(FadeIn(att1, shift=0.2 * DOWN), FadeIn(att_sub), FadeIn(dots_a), run_time=0.6)
        self.play(LaggedStart(*[Create(a) for a in arcs_a], lag_ratio=0.25), run_time=0.7)
        self.at("mlp")
        self.play(FadeIn(mlp1, shift=0.2 * DOWN), FadeIn(mlp_sub), FadeIn(dots_m), run_time=0.6)
        self.play(LaggedStart(*[Create(r) for r in rings_m], lag_ratio=0.2), run_time=0.7)
        wire = wire_arrow([-0.5, 1.0, 0], [0.5, 1.0, 0])
        self.at("wire")
        self.play(att1.animate.move_to([-1.8, 1.0, 0]), mlp1.animate.move_to([1.8, 1.0, 0]),
                  FadeOut(VGroup(att_sub, mlp_sub, dots_a, arcs_a, dots_m, rings_m)), run_time=0.65)
        self.play(GrowArrow(wire), run_time=0.3)
        self.end_section()

        # 2. The residual stream -----------------------------------------------------------------------
        self.section(2)
        self.play(*self.clear_anims(), run_time=0.5)
        band = stream_band()
        heading = Text("the residual stream", font_size=36).move_to([0, 3.2, 0])
        self.at("stream")
        self.play(GrowFromEdge(band, LEFT), FadeIn(heading, shift=0.2 * DOWN), run_time=1.0)
        self.at("flowing")
        self.play(LaggedStart(*[ShowPassingFlash(flow_line(S_Y + dy), time_width=0.35)
                                for dy in (0.18, -0.12, 0.05)], lag_ratio=0.25), run_time=1.5)
        vec = vec_column(VEC0, VEC_X)
        emb_label = Text("embedding", font_size=22, color=GREY_B).move_to([VEC_X, LABEL_Y, 0])
        self.at("embedding")
        self.play(FadeIn(vec, shift=0.2 * RIGHT), FadeIn(emb_label), run_time=0.6)
        pred_label = Text("prediction", font_size=22, color=GREY_B).move_to([PRED_X, LABEL_Y, 0])
        self.at("end")
        self.play(FadeIn(pred_label, shift=0.15 * UP), run_time=0.45)
        self.end_section()

        # 3. Add, don't replace ------------------------------------------------------------------------
        self.section(3)
        att = part_box("Attention", ATTN_COLOR, HALF_A["bw"]).move_to([HALF_A["bx"], BOX_Y, 0])
        mlp = part_box("MLP", MLP_COLOR, HALF_B["bw"]).move_to([HALF_B["bx"], BOX_Y, 0])
        read_a, write_a = read_path(HALF_A["xr"], att), write_path(att, HALF_A["xa"])
        read_b, write_b = read_path(HALF_B["xr"], mlp), write_path(mlp, HALF_B["xa"])
        plus_a, plus_b = plus_node(HALF_A["xa"]), plus_node(HALF_B["xa"])
        self.at("attention")
        self.play(FadeIn(att, shift=0.2 * DOWN), run_time=0.5)
        self.at("mlp")
        self.play(FadeIn(mlp, shift=0.2 * DOWN), run_time=0.5)
        heading3 = Text("add, don't replace", font_size=36).move_to(heading)
        self.at("replace")
        self.play(FadeTransform(heading, heading3), run_time=0.6)
        self.at("read")
        self.play(draw_path(read_a), run_time=0.7)
        self.at("compute")
        self.play(flash_box(att, run_time=0.6))
        self.at("add")
        self.play(draw_path(write_a, run_time=0.5), FadeIn(plus_a, scale=0.5, run_time=0.5))
        self.play(flash_plus(plus_a, run_time=0.35))
        eq1 = Text("x = x + attention(x)", font=MONO, font_size=30, t2c={"attention": ATTN_COLOR})
        eq2 = Text("x = x + mlp(x)", font=MONO, font_size=30, t2c={"mlp": MLP_COLOR})
        eq1.move_to([0, -2.25, 0]).align_to([-2.85, 0, 0], LEFT)
        eq2.move_to([0, -2.95, 0]).align_to(eq1, LEFT)
        self.at("attention")
        self.play(Write(eq1), run_time=0.6)
        self.at("then")
        self.play(draw_path(read_b, run_time=0.45), run_time=0.45)
        self.play(draw_path(write_b, run_time=0.45), FadeIn(plus_b, scale=0.5, run_time=0.45))
        self.at("mlp")
        self.play(Write(eq2), run_time=0.6)
        self.end_section()

        # 4. Why adding matters --------------------------------------------------------------------------
        self.section(4)
        self.play(FadeOut(eq1), FadeOut(eq2), run_time=0.4)
        glow = VGroup(flow_line(S_Y, color=BLUE_A, width=22), flow_line(S_Y, color=BLUE_A, width=5))
        glow[0].set_stroke(opacity=0.25)
        glow.set_z_index(1)
        self.at("original")
        self.play(Create(glow), run_time=1.0)
        hw_label = Text("← gradient highway", font_size=26, color=YELLOW).move_to([-0.4, LABEL_Y, 0])
        self.at("path")
        self.play(ShowPassingFlash(flow_line(S_Y, color=YELLOW, width=14, reverse=True), time_width=0.3),
                  FadeIn(hw_label, shift=0.2 * LEFT), run_time=1.1)
        self.at("network")
        self.play(ShowPassingFlash(flow_line(S_Y, color=YELLOW, width=14, reverse=True), time_width=0.3),
                  run_time=1.1)
        self.at("deep")
        self.play(ShowPassingFlash(flow_line(S_Y, color=YELLOW, width=14, reverse=True), time_width=0.3),
                  run_time=1.1)
        trainable = Text("deep stacks stay trainable", font_size=28).move_to([-0.4, -2.4, 0])
        self.at("trainable")
        self.play(FadeIn(trainable, shift=0.15 * UP), run_time=0.6)
        self.end_section()

        # 5. Layer norm ----------------------------------------------------------------------------------
        self.section(5)
        self.play(FadeOut(glow), FadeOut(hw_label), FadeOut(trainable), run_time=0.4)
        ln_a = part_box("LN", LN_COLOR, 0.8, 0.6, 24).move_to([HALF_A["xr"], BOX_Y, 0])
        ln_b = part_box("LN", LN_COLOR, 0.8, 0.6, 24).move_to([HALF_B["xr"], BOX_Y, 0])
        heading5 = Text("layer norm", font_size=36).move_to(heading3)
        self.at("norm")
        self.play(FadeTransform(heading3, heading5), FadeIn(ln_a, scale=0.6), FadeIn(ln_b, scale=0.6),
                  run_time=0.6)
        x_in = Text("[2, 4, 6, 8]", font=MONO, font_size=30)
        x_out = Text("[−1.34, −0.45, 0.45, 1.34]", font=MONO, font_size=30, color=PURPLE_A)
        arrow5 = Arrow(ORIGIN, RIGHT, buff=0, color=GREY_B, stroke_width=4, max_tip_length_to_length_ratio=0.25)
        row = VGroup(x_in, arrow5, x_out).arrange(RIGHT, buff=0.35).move_to([-0.3, -2.0, 0])
        ln_tag = Text("LN", font_size=22, color=LN_COLOR).next_to(arrow5, UP, buff=0.1)
        in_stats = Text("mean 5, std ≈ 2.24", font_size=22, color=GREY_B).next_to(x_in, DOWN, buff=0.3)
        out_stats = Text("mean 0, std 1", font_size=26, color=YELLOW).next_to(x_out, DOWN, buff=0.3)
        gain = Text("× gain + bias (learned)", font_size=26).next_to(out_stats, DOWN, buff=0.3)
        self.at("rescales")
        self.play(FadeIn(x_in, shift=0.15 * UP), run_time=0.5)
        self.play(GrowArrow(arrow5), FadeIn(ln_tag), FadeIn(x_out, shift=0.3 * RIGHT), FadeIn(in_stats),
                  run_time=0.7)
        self.at("mean")
        self.play(FadeIn(out_stats[:6]), run_time=0.4)          # "mean 0," (Text drops spaces)
        self.at("deviation")
        self.play(FadeIn(out_stats[6:]), run_time=0.4)          # "std 1"
        self.at("learned")
        self.play(FadeIn(gain, shift=0.15 * UP), run_time=0.6)
        healthy = Text("keeps numbers in a healthy range", font_size=24, color=GREY_B).move_to([0, 2.55, 0])
        self.at("healthy")
        self.play(FadeIn(healthy, shift=0.15 * DOWN), run_time=0.6)
        self.at("flow")
        self.play(LaggedStart(*[ShowPassingFlash(flow_line(S_Y + dy), time_width=0.35)
                                for dy in (0.15, -0.1)], lag_ratio=0.3), run_time=1.2)
        self.end_section()

        # 6. The block -----------------------------------------------------------------------------------
        self.section(6)
        self.play(FadeOut(VGroup(row, ln_tag, in_stats, out_stats, gain, healthy, heading5)), run_time=0.4)
        outline = DashedVMobject(RoundedRectangle(corner_radius=0.25, width=9.2, height=3.3, stroke_color=GREY_B,
                                                  stroke_width=2.5), num_dashes=70)
        outline.move_to([-0.4, 0.5, 0])
        block_label = Text("transformer block", font_size=30).next_to(outline, UP, buff=0.2)
        self.at("block")
        self.play(Create(outline), FadeIn(block_label, shift=0.15 * DOWN), run_time=0.6)
        for cue, anim in [("norm", lambda: flash_box(ln_a)), ("attention", lambda: flash_box(att)),
                          ("add", lambda: flash_plus(plus_a)), ("norm", lambda: flash_box(ln_b)),
                          ("mlp", lambda: flash_box(mlp)), ("add", lambda: flash_plus(plus_b))]:
            self.at(cue)
            self.play(anim())
        self.end_section()

        # 7. Stacking ------------------------------------------------------------------------------------
        self.section(7)
        detail = VGroup(ln_a, ln_b, att, mlp, read_a, write_a, read_b, write_b, plus_a, plus_b, outline)
        blocks = VGroup(*[mini_block(x) for x in BLOCK_XS])
        first4 = [1, 4, 7, 10]            # the first copies already sit in their final slots
        self.at("stack")
        self.play(FadeTransform(detail, blocks[first4[0]]), FadeOut(block_label), run_time=0.7)
        self.play(LaggedStart(*[TransformFromCopy(blocks[first4[0]], blocks[k]) for k in first4[1:]],
                              lag_ratio=0.2), run_time=0.8)
        small = Text("GPT-2 small: 12 blocks", font_size=32).move_to([0, 2.6, 0])
        self.at("12")
        self.play(LaggedStart(*[FadeIn(blocks[k], scale=0.6) for k in range(12) if k not in first4],
                              lag_ratio=0.1), FadeIn(small, shift=0.15 * DOWN), run_time=0.9)
        largest = Text("largest GPT-2: 48", font_size=28, color=GREY_B).move_to([0, 1.9, 0])
        self.at("48")
        self.play(FadeIn(largest, shift=0.15 * DOWN), run_time=0.5)
        own = Text("each with its own weights", font_size=26).move_to([0, 0.85, 0])
        self.at("own")
        self.play(*[b[0].animate.set_stroke(c, width=3).set_fill(dim(c, 0.4), 1)
                    for b, c in zip(blocks, BLOCK_TINTS)], FadeIn(own), run_time=0.7)
        rng = np.random.default_rng(9)
        vals = np.array(VEC0, dtype=float)
        traveller = vec_column(vals, VEC_X)
        self.add(traveller)
        steps = []
        for b in blocks:
            vals = np.clip(vals + rng.normal(0, 0.2, len(vals)), -1, 1)
            steps.append(Transform(traveller, vec_column(vals, b.get_x()), rate_func=linear, run_time=0.14))
        steps.append(Transform(traveller, vec_column(vals, PRED_X), run_time=0.25))
        self.at("refines")
        self.play(Succession(*steps))
        self.end_section()

        # 8. What the layers do --------------------------------------------------------------------------
        self.section(8)
        self.play(FadeOut(VGroup(small, largest, own, emb_label, pred_label)), run_time=0.4)
        rough = Text("rough pattern", font_size=26, color=GREY_B).move_to([0, 1.2, 0])
        self.at("rough")
        self.play(FadeIn(rough), run_time=0.5)
        grad = gradient_arrow(BLOCK_XS[0] - 0.31, BLOCK_XS[-1] + 0.31, -1.6, EARLY_COLOR, LATE_COLOR)
        early_box = SurroundingRectangle(blocks[:4], color=EARLY_COLOR, buff=0.1, corner_radius=0.1)
        late_box = SurroundingRectangle(blocks[8:], color=LATE_COLOR, buff=0.1, corner_radius=0.1)
        left = VGroup(Text("earlier:", font_size=26, color=EARLY_COLOR),
                      Text("grammar, nearby words", font_size=26)).arrange(DOWN, aligned_edge=LEFT, buff=0.18)
        left.next_to(grad, DOWN, buff=0.35).align_to(grad, LEFT)
        right = VGroup(Text("later:", font_size=26, color=LATE_COLOR), Text("abstract meaning,", font_size=26),
                       Text("next-token prediction", font_size=26)).arrange(DOWN, aligned_edge=RIGHT, buff=0.18)
        right.next_to(grad, DOWN, buff=0.35).align_to(grad, RIGHT)
        self.at("earlier")
        self.play(Create(grad[0]), FadeIn(grad[1]), Create(early_box), FadeIn(left[0]), run_time=0.8)
        self.at("grammar")
        self.play(FadeIn(left[1], shift=0.1 * UP), run_time=0.5)
        self.at("later")
        self.play(Create(late_box), FadeIn(right[0]), run_time=0.6)
        self.at("abstract")
        self.play(FadeIn(right[1], shift=0.1 * UP), run_time=0.5)
        self.at("predicting")
        self.play(FadeIn(right[2], shift=0.1 * UP), run_time=0.5)
        self.end_section()

        # 9. Code ----------------------------------------------------------------------------------------
        self.section(9)
        code, hl = code_panel(CODE, font_size=20)
        code.move_to(ORIGIN)
        hl.align_to(code.code_lines, LEFT).shift(0.12 * LEFT).match_y(code.line_numbers[6])
        self.play(*self.clear_anims(), FadeIn(code), run_time=0.7)
        self.at("block")
        self.play(Create(hl), run_time=0.4)
        self.at("attention")
        self.play(hl.animate(rate_func=there_and_back).set_stroke(width=6), run_time=0.5)
        self.at("mlp")
        self.play(highlight(hl, code, 7), run_time=0.4)
        self.at("norm")
        self.play(highlight(hl, code, 0), run_time=0.4)
        row_h = abs(code.line_numbers[0].get_y() - code.line_numbers[1].get_y())
        loop_y = (code.line_numbers[10].get_y() + code.line_numbers[11].get_y()) / 2
        self.at("loop")
        self.play(hl.animate.stretch_to_fit_height(2 * row_h).set_y(loop_y), run_time=0.4)
        self.wait_until(self.sec_start + self.sec["dur"])      # clear the code in the pause after the line
        self.play(*self.clear_anims(), run_time=0.45)
        self.end_section()

        # 10. Outro -------------------------------------------------------------------------------------
        self.section(10)
        y10 = 0.3
        band10 = stream_band(-6.5, 2.3, y10)
        xs10 = [-5.1 + 0.58 * k for k in range(12)]
        blocks10 = VGroup(*[mini_block(x, y10, c) for x, c in zip(xs10, BLOCK_TINTS)])
        for b, c in zip(blocks10, BLOCK_TINTS):
            b[0].set_fill(dim(c, 0.4), 1)
        blocks10.scale(0.72)
        for b, x in zip(blocks10, xs10):
            b.move_to([x, y10, 0])
        vec10 = vec_column(VEC0, -6.05, y10).scale(0.8)
        self.play(FadeIn(band10), FadeIn(blocks10), FadeIn(vec10), run_time=0.4)
        rng = np.random.default_rng(12)
        out_vals = np.clip(np.array(VEC0) + rng.normal(0, 0.5, len(VEC0)), -1, 1)
        out_vec = vec_column(out_vals, xs10[-1], y10).scale(0.8)
        self.at("last")
        self.play(FadeIn(out_vec, scale=0.8), run_time=0.3)
        self.play(out_vec.animate.move_to([3.2, y10, 0]), run_time=0.9)
        self.at("vector")
        self.play(Circumscribe(out_vec, color=YELLOW, buff=0.1), run_time=0.8)
        view = Text("the model's view of what comes next", font_size=24, color=GREY_B).move_to([2.9, -1.35, 0])
        self.at("view")
        self.play(FadeIn(view, shift=0.15 * UP), run_time=0.5)
        q_arrow = wire_arrow(out_vec.get_right() + 0.15 * RIGHT, [4.7, y10, 0])
        qmark = Text("?", font_size=80, color=YELLOW).move_to([5.3, y10, 0])
        self.at("vector")
        self.play(GrowArrow(q_arrow), FadeIn(qmark, scale=0.6), run_time=0.5)
        words = VGroup(*[token(w) for w in ("mat", "floor", "sofa")]).arrange(DOWN, buff=0.15)
        words.move_to([5.5, y10, 0])
        self.at("turning")
        self.play(FadeTransform(qmark, words), run_time=0.6)
        card = next_up_card(NEXT)
        self.at("words")
        self.play(*self.clear_anims(), FadeIn(card), run_time=0.8)
        self.end_section()
        finish(self)
