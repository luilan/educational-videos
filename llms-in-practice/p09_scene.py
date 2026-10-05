"""LLMs in Practice, episode 9 — Fine-Tuning vs Prompting, and LoRA in One Picture.

Render from the repo root:  ./render.sh llms-in-practice p09
Every loss, parameter count, file size and sample on screen comes from code/p09_lora/lora.py
(the How LLMs Work tiny GPT trained on Shakespeare, then fine-tuned on made-up recipes for 600 steps).
"""
from manim import *

from common import MODEL_COLOR, MONO, TOKEN_COLOR, code_panel, finish, highlight, next_up_card
from intro import play_token_intro
from p09_script import NEXT, TAGLINE, TITLE
from voiced_scene import VoicedScene

SERIES_LABEL = "LLMs IN PRACTICE  ·  EPISODE 9"
BASE_SAMPLE = "RECIPE:\nGo'er her man lord, let seen is eath\nAnd him be made your stormer and what have shing of\n'He would gave you and frait in their that as"
LORA_SAMPLE = "RECIPE: PLUM AND BARLEY PASTA\nIngredients:\n- two cups of leek\n- two cups of plum\n- half a cup of cream\nSteps:\nFirst, fry the plum gently."
FULL_SAMPLE = "RECIPE: LEMON AND LEEK PASTA\nIngredients:\n- three handfuls of mushroom\n- one cup of apple\n- half a cup of milk\nSteps:\nFirst, roast the plum …"
CODE = """class LoRALinear(nn.Module):
    def __init__(self, base, r=4, alpha=8):
        super().__init__()
        self.base, self.scale = base, alpha / r          # base is frozen
        self.A = nn.Parameter(torch.randn(r, base.in_features) / base.in_features ** 0.5)
        self.B = nn.Parameter(torch.zeros(base.out_features, r))     # starts at zero

    def forward(self, x):
        return self.base(x) + (x @ self.A.T @ self.B.T) * self.scale"""


def labeled_box(text, color=MODEL_COLOR, width=2.4, height=1.0, font_size=26):
    label = Text(text, font_size=font_size, line_spacing=0.8)
    box = RoundedRectangle(corner_radius=0.2, width=max(width, label.width + 0.5), height=height, stroke_color=color,
                           fill_color=color, fill_opacity=0.25)
    return VGroup(box, label.move_to(box))


def sample_panel(text, color=GREY_C, font_size=18):
    label = Text(text, font=MONO, font_size=font_size, line_spacing=0.7)
    box = RoundedRectangle(corner_radius=0.15, width=label.width + 0.5, height=label.height + 0.4, stroke_color=color,
                           fill_color=GREY_E, fill_opacity=1)
    return VGroup(box, label.move_to(box))


def loss_bar(label, value, color, scale=1.0):
    bar = Rectangle(width=value * scale, height=0.42, stroke_width=0, fill_color=color, fill_opacity=0.85)
    name = Text(label, font_size=22).next_to(bar, LEFT, buff=0.25)
    num = Text(f"{value:.2f}", font=MONO, font_size=22).next_to(bar, RIGHT, buff=0.15)
    return VGroup(name, bar, num)


def grid(rows, cols, cell, color, opacity=0.6):
    return VGroup(*[Square(cell, stroke_width=1, stroke_color=BLACK, fill_color=color, fill_opacity=opacity)
                    for _ in range(rows * cols)]).arrange_in_grid(rows, cols, buff=0)


class LoraVideo(VoicedScene):
    VIDEO = "p09"

    def construct(self):
        play_token_intro(self, TITLE, 9, TAGLINE, label=SERIES_LABEL)
        self.hook()          # 1
        self.compare()       # 2
        self.experiment()    # 3
        self.full()          # 4
        self.picture()       # 5
        self.numbers()       # 6
        self.switch()        # 7
        self.code()          # 8
        self.when()          # 9
        self.outro()         # 10

    def clear_stage(self, *keep, run_time=0.6):
        self.play(*[FadeOut(m) for m in self.mobjects if m not in keep], run_time=run_time)

    def place_bars(self, bars, x=-1.0, y=0.0, buff=0.3):
        for i, b in enumerate(bars):
            b[1].move_to([x, y - i * (0.42 + buff), 0], aligned_edge=LEFT)
            b[0].next_to(b[1], LEFT, buff=0.25)
            b[2].next_to(b[1], RIGHT, buff=0.15)
        return bars

    # ------------------------------------------------------------------ 1. hook
    def hook(self):
        self.section(1)
        inputs = VGroup(*[labeled_box(t, c, width=2.6, height=0.7, font_size=22) for t, c in
                          [("prompts", BLUE_C), ("documents", TEAL_C), ("tool results", GOLD)]])
        inputs.arrange(DOWN, buff=0.25).move_to([-4.2, 0.6, 0])
        model = labeled_box("model\n(unchanged)", GREY_B, width=2.6, height=1.4).move_to([1.0, 0.6, 0])
        arrows = VGroup(*[Arrow(i.get_right(), model.get_left(), buff=0.15, color=GREY_B) for i in inputs])
        self.at("changed")
        self.play(FadeIn(model), run_time=0.4)
        for cue, k in [("prompts", 0), ("documents", 1), ("tool", 2)]:
            self.at(cue)
            self.play(FadeIn(inputs[k], shift=0.2 * RIGHT), GrowArrow(arrows[k]), run_time=0.35)
        self.at("never")
        self.play(Indicate(model, color=YELLOW), run_time=0.6)
        ft = Text("change the model itself: fine-tuning", font_size=32, color=YELLOW).move_to([0, -2.3, 0])
        self.at("fine")
        self.play(FadeIn(ft, shift=0.2 * UP), model[0].animate.set_stroke(YELLOW), run_time=0.6)
        self.end_section()

    # ------------------------------------------------------------------ 2. prompting vs fine-tuning
    def compare(self):
        self.section(2)
        self.clear_stage()
        cols = [("prompting", BLUE_C, ["cheap, instant", "try it first"]),
                ("RAG", TEAL_C, ["adds knowledge", "(episode 5)"]),
                ("fine-tuning", GOLD, ["changes the weights", "style · format · skill", "costs training time",
                                       "needs examples"])]
        cards = VGroup()
        for title, color, lines in cols:
            head = Text(title, font_size=30, color=color)
            body = VGroup(*[Text(l, font_size=22) for l in lines]).arrange(DOWN, buff=0.18)
            content = VGroup(head, body).arrange(DOWN, buff=0.35)
            box = RoundedRectangle(corner_radius=0.2, width=3.9, height=3.6, stroke_color=color, fill_color=color,
                                   fill_opacity=0.1)
            content.move_to(box).align_to(box, UP).shift(0.35 * DOWN)
            cards.add(VGroup(box, head, body))
        cards.arrange(RIGHT, buff=0.35).move_to([0, 0, 0])
        for cue, k in [("prompting", 0), ("rag", 1), ("weights", 2)]:
            self.at(cue)
            self.play(FadeIn(cards[k][0]), FadeIn(cards[k][1]), FadeIn(cards[k][2][0] if k == 2 else cards[k][2]),
                      run_time=0.45)
        for cue, k in [("style", 1), ("costs", 2), ("examples", 3)]:
            self.at(cue)
            self.play(FadeIn(cards[2][2][k], shift=0.1 * UP), run_time=0.35)
        self.end_section()

    # ------------------------------------------------------------------ 3. the experiment
    def experiment(self):
        self.section(3)
        self.clear_stage()
        gpt = labeled_box("tiny GPT\n818,241 parameters", MODEL_COLOR, width=3.4, height=1.3, font_size=24)
        gpt.move_to([-4.0, 1.6, 0])
        note = Text("How LLMs Work, ep. 12", font_size=20, color=GREY_B).next_to(gpt, DOWN, buff=0.15)
        self.at("tiny")
        self.play(FadeIn(gpt), FadeIn(note), run_time=0.5)
        trained = Text("trained on Shakespeare", font_size=24, color=GOLD).next_to(gpt, UP, buff=0.2)
        self.at("shakespeare")
        self.play(FadeIn(trained), run_time=0.4)
        sample = sample_panel(BASE_SAMPLE).move_to([2.2, 1.3, 0])
        self.at("recipe")
        self.play(FadeIn(sample[0]), FadeIn(sample[1][:7]), run_time=0.4)
        self.at("writes")
        self.play(AddTextLetterByLetter(sample[1][7:], run_time=1.5))
        loss = Text("loss on made-up recipes: 2.77", font_size=32, color=YELLOW).move_to([0, -1.6, 0])
        self.at("77")
        self.play(FadeIn(loss, shift=0.2 * UP), run_time=0.5)
        self.end_section()

    # ------------------------------------------------------------------ 4. full fine-tuning
    def full(self):
        self.section(4)
        self.clear_stage()
        head = Text("option 1 · full fine-tuning", font_size=30).to_edge(UP, buff=0.4)
        self.at("one")
        self.play(FadeIn(head), run_time=0.4)
        weights = grid(8, 12, 0.22, GREY_C, 0.5).move_to([-4.3, 1.3, 0])
        w_label = Text("all 818,241 weights change", font_size=20, color=GOLD).next_to(weights, DOWN, buff=0.15)
        self.at("600")
        self.play(FadeIn(weights), run_time=0.4)
        self.at("800")
        self.play(LaggedStart(*[c.animate.set_fill(GOLD, 0.8) for c in weights], lag_ratio=0.005), FadeIn(w_label),
                  run_time=1.0)
        sample = sample_panel(FULL_SAMPLE, GOLD, 16).move_to([3.4, 1.3, 0])
        bars = self.place_bars(VGroup(loss_bar("recipes, before", 2.77, GREY_B), loss_bar("recipes, after", 0.25, GREEN),
                                      loss_bar("Shakespeare, before", 1.60, GREY_B),
                                      loss_bar("Shakespeare, after", 4.40, RED)), x=-1.9, y=-0.95)
        self.at("drops")
        self.play(FadeIn(bars[0]), FadeIn(bars[1]), run_time=0.5)
        self.at("perfect")
        self.play(FadeIn(sample, shift=0.2 * LEFT), run_time=0.5)
        self.at("jumps")
        self.play(FadeIn(bars[2]), FadeIn(bars[3]), run_time=0.5)
        forgot = Text("it forgot", font_size=30, color=RED).next_to(bars[3], RIGHT, buff=0.5)
        self.at("forgot")
        self.play(FadeIn(forgot, scale=1.2), run_time=0.4)
        copy = Text("+ a whole new copy of the model", font_size=24, color=GREY_A).next_to(bars, DOWN, buff=0.3)
        self.at("copy")
        self.play(FadeIn(copy), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 5. LoRA in one picture
    def picture(self):
        self.section(5)
        self.clear_stage()
        head = Text("option 2 · LoRA: low-rank adaptation", font_size=30).to_edge(UP, buff=0.4)
        self.at("two")
        self.play(FadeIn(head), run_time=0.4)
        W = grid(8, 8, 0.38, GREY_C, 0.5).move_to([-4.6, 0.3, 0])
        W_l = Text("W  (frozen)", font_size=26, color=GREY_A).next_to(W, DOWN, buff=0.2)
        self.at("freeze")
        self.play(FadeIn(W), FadeIn(W_l), run_time=0.5)
        BA_pos = np.array([1.9, -0.4, 0])
        A = grid(1, 8, 0.38, TEAL_C, 0.8).move_to(BA_pos + [0, 1.52 + 0.45, 0])
        B = grid(8, 1, 0.38, GOLD, 0.8).move_to(BA_pos + [-1.52 - 0.45, 0, 0])
        A_l = Text("matrix A  (r × in)", font_size=22, color=TEAL_B).next_to(A, UP, buff=0.15)
        B_l = Text("matrix B\n(out × r)", font_size=22, color=GOLD, line_spacing=0.8).next_to(B, LEFT, buff=0.2)
        self.at("matrices")
        self.play(FadeIn(A), FadeIn(A_l), FadeIn(B), FadeIn(B_l), run_time=0.7)
        BA = grid(8, 8, 0.38, YELLOW, 0.25).move_to(BA_pos)
        BA_l = Text("B × A: a small correction", font_size=22, color=YELLOW).next_to(BA, DOWN, buff=0.2)
        eq = Text("output = W x  +  (B A x) · scale", font=MONO, font_size=26).move_to([1.9, -3.1, 0])
        self.at("product")
        self.play(FadeIn(BA, scale=0.8), FadeIn(BA_l), run_time=0.6)
        self.play(FadeIn(eq), run_time=0.5)
        zero = Text("B starts at 0:\nno change at first", font_size=24, color=GOLD, line_spacing=0.8).move_to([-4.0, -2.7, 0])
        self.at("zero")
        self.play(FadeIn(zero), B.animate.set_fill(GOLD, 0.15), run_time=0.5)
        self.at("trained")
        self.play(Indicate(A, color=YELLOW), Indicate(B, color=YELLOW), B.animate.set_fill(GOLD, 0.8), run_time=0.8)
        self.end_section()

    # ------------------------------------------------------------------ 6. the numbers
    def numbers(self):
        self.section(6)
        self.clear_stage()
        trainable = VGroup(Text("32,768 trainable numbers", font_size=40, color=YELLOW),
                           Text("4.0% of the model (rank 4, every attention and MLP layer)", font_size=22, color=GREY_A))
        trainable.arrange(DOWN, buff=0.2).move_to([0, 2.4, 0])
        self.at("32")
        self.play(FadeIn(trainable[0], scale=1.1), FadeIn(trainable[1]), run_time=0.6)
        files = VGroup(Text("adapter file: 128 KB", font_size=28, color=TEAL_B),
                       Text("whole model: 3,196 KB", font_size=28, color=GREY_B)).arrange(RIGHT, buff=1.0)
        files.move_to([0, 1.0, 0])
        self.at("128")
        self.play(FadeIn(files), run_time=0.5)
        bars = self.place_bars(VGroup(loss_bar("base model", 2.77, GREY_B, 2.0), loss_bar("full fine-tuning", 0.25, GOLD, 2.0),
                                      loss_bar("LoRA", 0.27, TEAL_C, 2.0)), x=-0.6, y=-0.5)
        title = Text("loss on recipes", font_size=24, color=GREY_B).next_to(bars, UP, buff=0.25)
        self.play(FadeIn(title), FadeIn(bars[0]), FadeIn(bars[1]), run_time=0.4)
        self.at("27")
        self.play(FadeIn(bars[2], shift=0.2 * RIGHT), run_time=0.5)
        self.at("almost")
        self.play(Indicate(bars[2][2], color=YELLOW), Indicate(bars[1][2], color=YELLOW), run_time=0.7)
        self.end_section()

    # ------------------------------------------------------------------ 7. switch it off
    def switch(self):
        self.section(7)
        self.clear_stage()
        base = labeled_box("base model\n(frozen)", MODEL_COLOR, width=3.0, height=1.4).move_to([-3.8, 1.2, 0])
        adapter = labeled_box("recipe adapter", TEAL_C, width=2.6, height=0.8, font_size=22).next_to(base, RIGHT, buff=0.1)
        self.play(FadeIn(base), FadeIn(adapter), run_time=0.4)
        self.at("off")
        self.play(adapter.animate.shift(1.0 * RIGHT).set_opacity(0.25), run_time=0.6)
        back = Text("Shakespeare loss: 1.60, exactly as before", font_size=30, color=GREEN).move_to([0, -0.2, 0])
        self.at("exactly")
        self.play(FadeIn(back, shift=0.2 * UP), run_time=0.5)
        self.at("never")
        self.play(Indicate(base, color=GREEN), run_time=0.6)
        plugs = VGroup(*[labeled_box(t, c, width=2.6, height=0.7, font_size=20) for t, c in
                         [("recipes", TEAL_C), ("legal letters", BLUE_C), ("customer 1, 2, 3 …", GOLD)]])
        plugs.arrange(RIGHT, buff=0.3).move_to([0, -2.0, 0])
        swap = Text("one base model, many small adapters", font_size=26, color=YELLOW).next_to(plugs, UP, buff=0.3)
        self.at("swap")
        self.play(FadeIn(swap), run_time=0.4)
        for cue, k in [("recipes", 0), ("legal", 1), ("customer", 2)]:
            self.at(cue)
            self.play(FadeIn(plugs[k], shift=0.2 * UP), run_time=0.35)
        self.end_section()

    # ------------------------------------------------------------------ 8. code
    def code(self):
        self.section(8)
        self.clear_stage()
        code, hl = code_panel(CODE, font_size=20)
        code.move_to([0, 0.2, 0])
        self.at("wrap")
        self.play(FadeIn(code, shift=0.2 * UP), run_time=0.5)
        hl.match_y(code.line_numbers[0])
        self.play(Create(hl), run_time=0.3)
        self.at("freeze")
        self.play(highlight(hl, code, 3), run_time=0.4)
        self.at("random")
        self.play(highlight(hl, code, 4), run_time=0.4)
        self.at("zero")
        self.play(highlight(hl, code, 5), run_time=0.4)
        self.at("output")
        self.play(highlight(hl, code, 8), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 9. when to use what
    def when(self):
        self.section(9)
        self.clear_stage()
        steps = VGroup(*[labeled_box(t, c, width=8.4, height=h, font_size=26) for t, c, h in
                         [("1 · prompting: always start here", BLUE_C, 0.95),
                          ("2 · RAG: knowledge, especially facts that change", TEAL_C, 0.95),
                          ("3 · LoRA fine-tuning: style, format, narrow skills\n(with hundreds of good examples)", GOLD,
                           1.3)]])
        steps.arrange(DOWN, buff=0.35).move_to([0, 0, 0])
        for cue, k in [("prompting", 0), ("rag", 1), ("lora", 2)]:
            self.at(cue)
            self.play(FadeIn(steps[k], shift=0.2 * UP), run_time=0.45)
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
