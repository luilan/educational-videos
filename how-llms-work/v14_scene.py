"""Video 14 — From GPT to Chatbot (series finale).

Render from the repo root:  ./render.sh how-llms-work v14
"""
import random

from manim import *

from common import MODEL_COLOR, MONO, TOKEN_COLOR, code_panel, finish, token
from intro import play_token_intro
from v14_script import TAGLINE, TITLE
from voiced_scene import VoicedScene

SPECIAL_COLOR = ORANGE
CHAT_Q = "Why do cats love boxes?"
CHAT_A = "Boxes feel safe and warm, so a cat can relax."
TEMPLATE = ["<|user|>", CHAT_Q, "<|assistant|>", CHAT_A]
STRIP_ROWS = [["<|user|>", "Why", "do", "cats", "love", "boxes", "?", "<|assistant|>"],
              ["Boxes", "feel", "safe", "and", "warm", ",", "so", "a", "cat", "can", "relax", "."]]
TEST_CODE = """def add(a, b):
    return a + b

assert add(2, 3) == 5"""


# ---------------------------------------------------------------------------- helpers
def chat_bubble(text, color, align, font_size=26):
    """Video 1 chat bubble: user = BLUE_E aligned RIGHT, assistant = GREY_D aligned LEFT."""
    label = Text(text, font_size=font_size, line_spacing=0.8)
    box = RoundedRectangle(corner_radius=0.25, width=label.width + 0.6, height=label.height + 0.5,
                           stroke_width=0, fill_color=color, fill_opacity=0.9)
    return VGroup(box, label.move_to(box).align_to(box, align).shift(-0.3 * align))


def labeled_box(text, color=MODEL_COLOR, width=2.6, height=1.1, font_size=30):
    label = Text(text, font_size=font_size)
    box = RoundedRectangle(corner_radius=0.2, width=max(width, label.width + 0.5), height=height,
                           stroke_color=color, fill_color=color, fill_opacity=0.25)
    return VGroup(box, label.move_to(box))


def stage_label(text, font_size=34):
    return Text(text, font_size=font_size).to_edge(UP, buff=0.45)


def doc_icon(w=0.42, h=0.55):
    page = Rectangle(width=w, height=h, stroke_color=GREY_B, stroke_width=1.5, fill_color=GREY_E, fill_opacity=1)
    lines = VGroup(*[Line(ORIGIN, (0.45 if k % 3 == 2 else 0.65) * w * RIGHT, stroke_width=1.2, color=GREY_B)
                     for k in range(4)]).arrange(DOWN, buff=h * 0.1, aligned_edge=LEFT)
    return VGroup(page, lines.move_to(page))


def mini_pair(pick_right):
    """A tiny A/B comparison: two answer slips, the preferred one outlined green."""
    a = Rectangle(width=0.3, height=0.22, stroke_color=GREY_B, stroke_width=1.5, fill_color=GREY_D, fill_opacity=0.9)
    b = a.copy()
    (b if pick_right else a).set_stroke(GREEN, 3)
    return VGroup(a, b).arrange(RIGHT, buff=0.06)


def pair_grid(rows, cols, buff, seed):
    rng = random.Random(seed)
    return VGroup(*[mini_pair(k == 0 or rng.random() < 0.6) for k in range(rows * cols)]).arrange_in_grid(
        rows, cols, buff=buff)


def answer_card(letter, text, width=5.0, height=1.8):
    tag = Text(letter, font_size=30, weight=BOLD, color=GREY_B)
    body = Text(text, font_size=24, line_spacing=0.8)
    content = VGroup(tag, body).arrange(DOWN, aligned_edge=LEFT, buff=0.2)
    box = RoundedRectangle(corner_radius=0.2, width=width, height=height, stroke_color=GREY_B, stroke_width=2,
                           fill_color=GREY_D, fill_opacity=0.9)
    content.move_to(box).align_to(box, LEFT).shift(0.35 * RIGHT)
    return VGroup(box, tag, body)


def small_answer(text, width=1.9, height=0.7):
    box = RoundedRectangle(corner_radius=0.15, width=width, height=height, stroke_color=GREY_B, stroke_width=2,
                           fill_color=GREY_D, fill_opacity=0.9)
    label = Text(text, font_size=22).move_to(box).align_to(box, LEFT).shift(0.25 * RIGHT)
    return VGroup(box, label)


def person_icon(color=GREY_A):
    head = Circle(radius=0.2, stroke_width=0, fill_color=color, fill_opacity=1)
    body = AnnularSector(inner_radius=0, outer_radius=0.38, angle=PI, color=color)
    body.next_to(head, DOWN, buff=0.06)
    return VGroup(head, body)


def robot_icon(color=TEAL_C):
    head = RoundedRectangle(corner_radius=0.15, width=1.0, height=0.8, stroke_color=color,
                            fill_color=color, fill_opacity=0.25)
    eyes = VGroup(*[Dot(radius=0.08, color=color) for _ in range(2)]).arrange(RIGHT, buff=0.3)
    eyes.move_to(head).shift(0.1 * UP)
    mouth = Line(0.2 * LEFT, 0.2 * RIGHT, color=color, stroke_width=3).move_to(head).shift(0.18 * DOWN)
    antenna = Line(ORIGIN, 0.25 * UP, color=color, stroke_width=3).next_to(head, UP, buff=0)
    tip = Dot(radius=0.07, color=color).move_to(antenna.get_end())
    return VGroup(head, eyes, mouth, antenna, tip)


def principles_doc():
    lines = VGroup(Text("principles", font_size=26, weight=BOLD),
                   *[Text(s, font_size=22, color=GREY_A) for s in ("1. be helpful", "2. be honest", "3. avoid harm")])
    lines.arrange(DOWN, aligned_edge=LEFT, buff=0.2)
    page = Rectangle(width=lines.width + 0.6, height=lines.height + 0.6, stroke_color=GREY_B, stroke_width=2,
                     fill_color=GREY_E, fill_opacity=1)
    return VGroup(page, lines.move_to(page))


def check(font_size=36):
    return Text("✓", font_size=font_size, color=GREEN)


def stack_box(text, color, width=3.9, height=0.75, font_size=24):
    box = RoundedRectangle(corner_radius=0.15, width=width, height=height, stroke_color=color,
                           fill_color=color, fill_opacity=0.25)
    return VGroup(box, Text(text, font_size=font_size).move_to(box))


def map_box(num, name, width=None):
    """Series-map box in the style of video 1's "The road ahead"."""
    label = Text(name, font_size=20)
    box = RoundedRectangle(corner_radius=0.15, width=width or max(1.5, label.width + 0.45), height=0.9,
                           stroke_color=GREY_B, fill_color=GREY_E, fill_opacity=0.6)
    ep = Text(num, font_size=20, color=GREY_B).next_to(box, UP, buff=0.1)
    return VGroup(box, label.move_to(box), ep)


def light(group, color=TOKEN_COLOR):
    return group[0].animate.set_stroke(color).set_fill(color, 0.35)


def unlight(group):
    return group[0].animate.set_stroke(GREY_B).set_fill(GREY_E, 0.6)


class ChatbotVideo(VoicedScene):
    VIDEO = "v14"

    def construct(self):
        play_token_intro(self, TITLE, 14, TAGLINE)
        self.base_model()        # 1
        self.pretraining()       # 2
        self.fine_tuning()       # 3
        self.chat_template()     # 4
        self.preferences()       # 5
        self.rlhf_and_dpo()      # 6
        self.other_signals()     # 7
        self.same_machine()      # 8
        self.recap()             # 9
        self.farewell()          # 10

    # ------------------------------------------------------------------ utilities
    def clear_stage(self, *keep, extra=(), run_time=0.6):
        self.play(*[FadeOut(m) for m in self.mobjects if m not in keep], *extra, run_time=run_time)

    def regroup(self, group):
        """Replace separately-added children with their parent group so it can be transformed as one."""
        self.remove(*group.submobjects)
        self.add(group)
        return group

    # ------------------------------------------------------------------ 1. the base model
    def base_model(self):
        self.section(1)
        line = Text("The cat sat on the mat and fell asleep.", font_size=30)
        line.move_to([-3.3, 2.5, 0], aligned_edge=LEFT)
        prompt, nxt, rest = line[:14], line[14:17], line[17:]
        nxt.set_color(YELLOW)
        rest.set_color(YELLOW)
        self.base = labeled_box("base model", width=2.6, height=1.0).move_to([-5.0, 2.5, 0])
        base_note = Text("next-token predictor", font_size=20, color=GREY_B).next_to(self.base, DOWN, buff=0.15)

        self.at("predict")
        self.play(FadeIn(prompt, shift=0.2 * RIGHT), run_time=0.4)
        self.at("next")
        self.play(FadeIn(nxt, scale=1.3), run_time=0.4)
        self.at("base")
        self.play(FadeIn(self.base, scale=0.8), FadeIn(base_note), run_time=0.7)
        self.at("autocomplete")
        self.play(AddTextLetterByLetter(rest, run_time=1.0))

        question = chat_bubble("What is the capital of France?", BLUE_E, RIGHT).to_edge(RIGHT, buff=0.6).set_y(1.0)
        paris = chat_bubble("Paris.", GREY_D, LEFT).to_edge(LEFT, buff=0.6).set_y(-0.15)
        quiz = chat_bubble("What is the capital of Germany?\nWhat is the capital of Spain?\n"
                           "What is the capital of Italy?", GREY_D, LEFT)
        quiz.next_to(paris, DOWN, buff=0.35, aligned_edge=LEFT)
        note = Text("illustrative", font_size=22, color=GREY_B).to_corner(DR, buff=0.35)
        self.at("question")
        self.play(FadeIn(question, shift=0.3 * UP), run_time=0.6)
        self.at("answer")
        self.play(FadeIn(paris.set_opacity(0.4), shift=0.2 * UP), run_time=0.5)
        self.at("continue")
        self.play(paris.animate.set_opacity(0.15), run_time=0.5)
        self.at("questions")
        self.play(FadeIn(quiz[0], run_time=0.3), AddTextLetterByLetter(quiz[1], run_time=1.6))
        self.at("quiz")
        self.play(FadeIn(note), run_time=0.5)
        self.end_section()

    # ------------------------------------------------------------------ 2. pretraining
    def pretraining(self):
        self.section(2)
        self.clear_stage(self.base, extra=[self.base.animate.scale(1.15).move_to([2.3, 0.9, 0])], run_time=0.8)
        self.stage = stage_label("stage 1 · pretraining")
        docs = VGroup(*[doc_icon() for _ in range(24)]).arrange_in_grid(4, 6, buff=0.12).move_to([-4.2, 0.9, 0])
        feed = Arrow(docs.get_right(), self.base.get_left(), buff=0.2, color=GREY_B)
        self.at("pre")
        self.play(FadeIn(self.stage, shift=0.2 * DOWN),
                  LaggedStart(*[FadeIn(d, scale=0.6) for d in docs], lag_ratio=0.03), GrowArrow(feed), run_time=0.8)
        rng = random.Random(14)
        flying = VGroup(*[docs[i].copy() for i in rng.sample(range(24), 12)])
        self.play(LaggedStart(*[f.animate.move_to(self.base).scale(0.2).set_opacity(0) for f in flying],
                              lag_ratio=0.1, run_time=1.3))
        self.remove(flying)

        frame = Rectangle(width=11, height=0.5, stroke_color=GREY_B, stroke_width=2).move_to([0, -2.55, 0])
        fill = Rectangle(width=11 * 0.97, height=0.5, stroke_width=0, fill_color=MODEL_COLOR, fill_opacity=0.7)
        fill.align_to(frame, LEFT).match_y(frame)
        rest = Rectangle(width=11 * 0.03, height=0.5, stroke_width=0, fill_color=GREY_C, fill_opacity=0.8)
        rest.next_to(fill, RIGHT, buff=0)
        fill_label = Text("pretraining", font_size=22).move_to(fill)
        bar_title = Text("training compute", font_size=22, color=GREY_B).next_to(frame, UP, buff=0.15)
        bar_title.align_to(frame, LEFT)
        rest_label = Text("later stages", font_size=20, color=GREY_B).next_to(frame, UP, buff=0.15)
        rest_label.align_to(frame, RIGHT)
        most = Text("≈ almost all the compute", font_size=26, color=YELLOW).next_to(frame, DOWN, buff=0.2)
        self.at("computing")
        self.play(Create(frame), GrowFromEdge(fill, LEFT), FadeIn(rest), FadeIn(fill_label), FadeIn(bar_title),
                  FadeIn(rest_label), FadeIn(most, shift=0.2 * UP), run_time=1.0)

        know = Text("knowledge", font_size=26, color=GREY_A).next_to(self.base, DOWN, buff=0.25)
        self.at("knowledge")
        self.play(Indicate(self.base, color=YELLOW, scale_factor=1.06), FadeIn(know, shift=0.2 * UP), run_time=0.8)

        tri = Text("trillions of tokens", font_size=28).next_to(docs, DOWN, buff=0.25)
        eg = Text("e.g. Llama 3: ~15T tokens", font_size=20, color=GREY_B).next_to(tri, DOWN, buff=0.12)
        self.at("trillions")
        self.play(FadeIn(tri, shift=0.2 * UP), FadeIn(eg), run_time=0.7)
        self.end_section()

    # ------------------------------------------------------------------ 3. supervised fine-tuning
    def fine_tuning(self):
        self.section(3)
        self.clear_stage(self.base, self.stage, extra=[self.base.animate.move_to([4.5, 0.9, 0])], run_time=0.8)
        new_stage = stage_label("stage 2 · supervised fine-tuning")
        self.at("supervised")
        self.play(FadeTransform(self.stage, new_stage), run_time=0.6)
        self.stage = new_stage

        same = Text("same model", font_size=30).scale(1.15).move_to(self.base[0])
        self.at("same")
        self.play(Transform(self.base[1], same), Circumscribe(self.base, color=YELLOW), run_time=0.8)
        loss = Text("same next-token loss", font_size=26, color=YELLOW).next_to(self.base, DOWN, buff=0.3)
        self.at("loss")
        self.play(FadeIn(loss, shift=0.2 * UP), run_time=0.5)

        user = chat_bubble("How long should I boil an egg?", BLUE_E, RIGHT).move_to([1.9, 1.7, 0], aligned_edge=RIGHT)
        bot = chat_bubble("About 7 minutes for a jammy yolk,\nor 10 for hard-boiled.", GREY_D, LEFT)
        bot.move_to([-6.3, 0.05, 0], aligned_edge=LEFT)
        frame = DashedVMobject(SurroundingRectangle(VGroup(user, bot), buff=0.25, corner_radius=0.2,
                                                    color=GREY_B, stroke_width=2), num_dashes=70)
        into = Arrow(frame.get_right(), self.base.get_left(), buff=0.12, color=GREY_B)
        self.at("conversations")
        self.play(Create(frame), GrowArrow(into), run_time=0.6)
        self.at("user")
        self.play(FadeIn(user, shift=0.3 * UP), run_time=0.5)
        self.at("assistant")
        self.play(FadeIn(bot[0], run_time=0.3), AddTextLetterByLetter(bot[1], run_time=1.2))

        fmt = Text("format: who speaks, and when", font_size=28, t2c={"format:": YELLOW})
        sty = Text("style: helpful, clear, friendly", font_size=28, t2c={"style:": YELLOW})
        fmt.move_to([-6.1, -2.0, 0], aligned_edge=LEFT)
        sty.next_to(fmt, DOWN, buff=0.3, aligned_edge=LEFT)
        self.at("format")
        self.play(FadeIn(fmt, shift=0.2 * UP), run_time=0.5)
        self.at("style")
        self.play(FadeIn(sty, shift=0.2 * UP), run_time=0.5)
        self.end_section()

    # ------------------------------------------------------------------ 4. chat templates
    def chat_template(self):
        self.section(4)
        self.clear_stage(self.stage)
        lines = VGroup(*[Text(s, font=MONO, font_size=26) for s in TEMPLATE]).arrange(DOWN, buff=0.3, aligned_edge=LEFT)
        panel_box = RoundedRectangle(corner_radius=0.2, width=lines.width + 0.8, height=lines.height + 0.7,
                                     stroke_color=GREY_C, stroke_width=2, fill_color=GREY_E, fill_opacity=1)
        panel = VGroup(panel_box, lines.move_to(panel_box)).move_to([0.5, 0.6, 0])
        lines.set_z_index(2)
        roles = VGroup(*[SurroundingRectangle(lines[i], buff=0.08, corner_radius=0.1, stroke_color=SPECIAL_COLOR,
                                              fill_color=SPECIAL_COLOR, fill_opacity=0.3) for i in (0, 2)])
        roles.set_z_index(1)
        spec = Text("special\ntokens", font_size=22, color=SPECIAL_COLOR, line_spacing=0.8)
        spec.move_to([-5.6, lines[1].get_y(), 0])
        spec_arrows = VGroup(*[Arrow(spec.get_right(), r.get_left(), buff=0.1, color=SPECIAL_COLOR, stroke_width=3,
                                     max_tip_length_to_length_ratio=0.3) for r in roles]).set_z_index(1)
        varies = Text("format varies by model", font_size=22, color=GREY_B)
        varies.next_to(panel, DOWN, buff=0.3).align_to(panel, RIGHT)

        self.at("text")
        self.play(FadeIn(panel, shift=0.3 * UP), run_time=0.7)
        self.at("special")
        self.play(FadeIn(roles), FadeIn(spec), *[GrowArrow(a) for a in spec_arrows], run_time=0.6)
        self.at("speaking")
        self.play(FadeIn(varies), run_time=0.5)

        target = SurroundingRectangle(lines[3], buff=0.1, color=YELLOW, fill_color=YELLOW, fill_opacity=0.12)
        target.set_z_index(1)
        target_label = Text("trained to predict this part", font_size=24, color=YELLOW)
        target_label.next_to(panel, DOWN, buff=0.3).align_to(lines[3], LEFT)
        self.at("predict")
        self.play(Create(target), lines[1].animate.set_opacity(0.45), run_time=0.5)
        self.at("assistants")
        self.play(FadeIn(target_label, shift=0.2 * UP), run_time=0.5)

        colors = [[SPECIAL_COLOR] + [TOKEN_COLOR] * 6 + [SPECIAL_COLOR], [YELLOW] * 12]
        rows = VGroup(*[VGroup(*[token(w, color=c, font_size=22) for w, c in zip(ws, cs)]).arrange(RIGHT, buff=0.08)
                        for ws, cs in zip(STRIP_ROWS, colors)])
        rows.arrange(DOWN, buff=0.3, aligned_edge=LEFT)
        slot = VGroup(DashedVMobject(RoundedRectangle(corner_radius=0.12, width=0.7, height=0.65, color=YELLOW)),
                      Text("?", font_size=26, color=YELLOW))
        slot.next_to(rows[1], RIGHT, buff=0.08)
        slot[1].move_to(slot[0])
        VGroup(rows, slot).move_to([0, 0.8, 0])
        doc_label = Text("just one long document", font_size=32).next_to(rows, DOWN, buff=0.7)
        doc_label.set_x(0)
        src = VGroup(panel, roles, target)
        self.remove(panel, roles, target)
        self.add(src)
        self.at("chat")
        self.play(FadeOut(VGroup(spec, spec_arrows, varies, target_label)), run_time=0.5)
        self.at("document")
        self.play(FadeTransform(src, rows), FadeIn(doc_label, shift=0.2 * UP), run_time=0.7)
        self.at("continue")
        self.play(Create(slot[0]), FadeIn(slot[1]), run_time=0.5)
        self.end_section()

    # ------------------------------------------------------------------ 5. preferences
    def preferences(self):
        self.section(5)
        self.clear_stage(self.stage, run_time=0.3)
        new_stage = stage_label("stage 3 · preferences")
        self.at("3")
        self.play(FadeTransform(self.stage, new_stage), run_time=0.6)
        self.stage = new_stage

        prompt = chat_bubble(CHAT_Q, BLUE_E, RIGHT).move_to([0, 2.2, 0])
        self.at("judgment")
        self.play(FadeIn(prompt, shift=0.2 * UP), run_time=0.5)
        card_a = answer_card("A", "Because they are cats.").move_to([-3.1, 0.0, 0])
        card_b = answer_card("B", "Boxes feel safe and warm,\nso a cat can relax.").move_to([3.1, 0.0, 0])
        self.at("compare")
        self.play(LaggedStart(FadeIn(card_a, shift=0.2 * UP), FadeIn(card_b, shift=0.2 * UP), lag_ratio=0.3),
                  run_time=0.8)

        person = person_icon().move_to([0, 0.0, 0])
        tick = check(44).move_to(card_b[0].get_corner(UR) + [-0.4, -0.4, 0]).set_z_index(3)
        self.at("pick")
        self.play(FadeIn(person, shift=0.2 * UP), FadeIn(tick, scale=1.5), card_b[0].animate.set_stroke(GREEN, 4),
                  card_a.animate.fade(0.5), run_time=0.7)

        self.grid = pair_grid(5, 10, (0.3, 0.3), seed=5).move_to([0, 0.35, 0])
        many = Text("many thousands of comparisons", font_size=26, color=GREY_A).next_to(self.grid, DOWN, buff=0.4)
        cards = VGroup(card_a, card_b)
        self.remove(card_a, card_b)
        self.add(cards)
        self.at("thousands")
        self.play(FadeOut(VGroup(prompt, person, tick)), FadeTransform(cards, self.grid[0]), run_time=0.6)
        self.play(LaggedStart(*[FadeIn(p, scale=0.5) for p in self.grid[1:]], lag_ratio=0.04, run_time=1.2),
                  FadeIn(many))
        self.regroup(self.grid)

        learns = Text("the model learns what “better” means", font_size=30, t2c={"“better”": YELLOW})
        learns.next_to(many, DOWN, buff=0.45)
        self.at("learns")
        self.play(FadeIn(learns, shift=0.2 * UP), run_time=0.6)
        self.end_section()

    # ------------------------------------------------------------------ 6. RLHF and DPO
    def rlhf_and_dpo(self):
        self.section(6)
        pile = pair_grid(4, 3, (0.15, 0.2), seed=6).move_to([-5.45, 1.3, 0])
        pile_label = Text("preference pairs", font_size=22, color=GREY_B).next_to(pile, DOWN, buff=0.2)
        self.clear_stage(self.stage, self.grid, extra=[ReplacementTransform(self.grid, pile), FadeIn(pile_label)],
                         run_time=0.9)

        title = Text("RLHF · reinforcement learning from human feedback", font_size=30,
                     t2c={"RLHF": YELLOW}).to_edge(UP, buff=0.45)
        short = title[:4].copy().move_to(self.stage)
        self.at("rlhf")
        self.play(FadeOut(self.stage), FadeIn(short, scale=1.2), run_time=0.5)
        self.at("reinforcement")
        self.play(ReplacementTransform(short, title[:4]), FadeIn(title[4:], shift=0.2 * LEFT), run_time=0.8)
        self.at("human")
        self.play(Indicate(pile, color=GREEN, scale_factor=1.08), run_time=0.7)

        reward = labeled_box("reward model", color=GOLD, width=2.7, height=1.1, font_size=24).move_to([-2.25, 1.3, 0])
        to_reward = Arrow(pile.get_right(), reward.get_left(), buff=0.15, color=GREY_B)
        self.at("reward")
        self.play(GrowArrow(to_reward), FadeIn(reward, scale=0.8), run_time=0.7)

        chatbot = labeled_box("chatbot", color=MODEL_COLOR, width=2.2, height=1.1, font_size=28).move_to([5.1, 1.3, 0])
        answers = VGroup(small_answer("answer 1"), small_answer("answer 2")).arrange(DOWN, buff=0.5).move_to([0.85, 1.3, 0])
        self.at("answers")
        self.play(FadeIn(chatbot, scale=0.8),
                  LaggedStart(*[FadeIn(a, shift=0.4 * LEFT) for a in answers], lag_ratio=0.3), run_time=0.7)

        score_lines = VGroup(*[DashedLine(reward.get_right(), a.get_left(), buff=0.1, color=GREY_B, stroke_width=2)
                               for a in answers])
        scores = VGroup(Text("0.2", font_size=26, color=RED), Text("0.9", font_size=26, color=GREEN))
        for s, a in zip(scores, answers):
            s.next_to(a[0], RIGHT, buff=0.15)
        self.at("prefer")
        self.play(Create(score_lines), LaggedStart(*[FadeIn(s, scale=1.4) for s in scores], lag_ratio=0.4), run_time=0.7)

        shifted = chatbot.copy().shift(0.3 * LEFT + 0.25 * DOWN)
        nudge = Arrow(shifted.get_left(), scores[1].get_right(), buff=0.12, color=GREEN, stroke_width=5)
        nudge_label = Text("nudge", font_size=24, color=GREEN).move_to([nudge.get_center()[0] + 0.1, 0.25, 0])
        self.at("nudge")
        self.play(chatbot.animate.move_to(shifted), GrowArrow(nudge), FadeIn(nudge_label),
                  answers[1][0].animate.set_stroke(GREEN), run_time=0.8)
        self.at("higher")
        self.play(Indicate(scores[1], color=GREEN, scale_factor=1.5), run_time=0.7)

        rlhf_parts = VGroup(to_reward, reward, score_lines, answers, scores, nudge, nudge_label)
        self.at("simpler")
        self.play(rlhf_parts.animate.fade(0.6), run_time=0.6)

        y = -1.9
        p0 = pile_label.get_bottom() + 0.12 * DOWN
        p3 = chatbot.get_bottom() + 0.12 * DOWN
        corners = [p0, [p0[0], y, 0], [p3[0], y, 0]]
        path = VGroup(Line(corners[0], corners[1], color=TEAL, stroke_width=5),
                      Line(corners[1], corners[2], color=TEAL, stroke_width=5),
                      Arrow(corners[2], p3, buff=0, color=TEAL, stroke_width=5, max_tip_length_to_length_ratio=0.15))
        dpo = Text("DPO: learn from the pairs directly", font_size=28, t2c={"DPO:": TEAL})
        dpo.next_to(path[1], DOWN, buff=0.25)
        self.at("dpo")
        self.play(Succession(Create(path[0]), Create(path[1]), GrowArrow(path[2])), FadeIn(dpo, shift=0.2 * UP),
                  run_time=1.2)
        self.at("directly")
        self.play(*[ShowPassingFlash(p.copy().set_color(YELLOW), time_width=0.5) for p in path[:2]],
                  Indicate(chatbot, color=YELLOW, scale_factor=1.06), run_time=0.8)
        self.end_section()

    # ------------------------------------------------------------------ 7. other training signals
    def other_signals(self):
        self.section(7)
        self.clear_stage()
        ai_head = Text("AI feedback", font_size=30).move_to([-3.4, 2.6, 0])
        robot = robot_icon().scale(1.2).move_to([-5.0, 0.9, 0])
        self.at("ai")
        self.play(FadeIn(ai_head, shift=0.2 * DOWN), FadeIn(robot, scale=0.8), run_time=0.6)

        doc = principles_doc().move_to([-2.0, 0.9, 0])
        guide = Arrow(doc.get_left(), robot.get_right(), buff=0.2, color=GREY_B)
        eg = Text("e.g. Constitutional AI", font_size=22, color=GREY_B).next_to(doc, DOWN, buff=0.3)
        self.at("principles")
        self.play(FadeIn(doc, shift=0.2 * LEFT), GrowArrow(guide), FadeIn(eg), run_time=0.7)

        divider = Line([0.35, 2.9, 0], [0.35, -3.0, 0], color=GREY_D, stroke_width=2)
        rl_head = Text("RL on checkable answers", font_size=30).move_to([3.6, 2.6, 0])
        self.at("checkable")
        self.play(Create(divider), FadeIn(rl_head, shift=0.2 * DOWN), run_time=0.6)

        math = Text("17 × 24 = 408", font_size=36)
        math_ok = VGroup(math, check(40).next_to(math, RIGHT, buff=0.3)).move_to([3.6, 1.45, 0])
        self.at("math")
        self.play(FadeIn(math, shift=0.2 * UP), FadeIn(math_ok[1], scale=1.5), run_time=0.5)

        code, _ = code_panel(TEST_CODE, font_size=20)
        code.move_to([3.6, -0.45, 0])
        passed = Text("tests pass ✓", font_size=28, color=GREEN).next_to(code, DOWN, buff=0.3)
        self.at("code")
        self.play(FadeIn(code, shift=0.2 * UP), FadeIn(passed, shift=0.2 * UP), run_time=0.7)
        self.end_section()

    # ------------------------------------------------------------------ 8. same machine, different data
    def same_machine(self):
        self.section(8)
        self.clear_stage()
        x = -0.4
        tokens = stack_box("tokens", TOKEN_COLOR).move_to([x, -2.3, 0])
        emb = stack_box("embeddings", TEAL_D).move_to([x, -1.1, 0])
        block_box = RoundedRectangle(corner_radius=0.2, width=3.9, height=1.5, stroke_color=MODEL_COLOR,
                                     fill_color=MODEL_COLOR, fill_opacity=0.2).move_to([x, 0.45, 0])
        block_title = Text("transformer block × N", font_size=22).move_to(block_box).align_to(block_box, UP).shift(0.15 * DOWN)
        attn = stack_box("attention", BLUE_C, width=1.65, height=0.55, font_size=20)
        mlp = stack_box("MLP", GREEN_C, width=1.65, height=0.55, font_size=20)
        subs = VGroup(attn, mlp).arrange(RIGHT, buff=0.2).next_to(block_title, DOWN, buff=0.18)
        block = VGroup(block_box, block_title, subs)
        logits = stack_box("logits → next token", GOLD).move_to([x, 2.1, 0])
        parts = [tokens, emb, block, logits]
        arrows = VGroup(*[Arrow(a.get_top(), b.get_bottom(), buff=0.06, color=GREY_B, stroke_width=3,
                                max_tip_length_to_length_ratio=0.35) for a, b in zip(parts[:-1], parts[1:])])
        self.at("machinery")
        self.play(LaggedStart(*[FadeIn(p, shift=0.2 * UP) for p in parts], lag_ratio=0.25),
                  LaggedStart(*[GrowArrow(a) for a in arrows], lag_ratio=0.25), run_time=1.2)

        same = VGroup(Text("same transformer", font_size=28), Text("at every stage", font_size=22, color=GREY_B))
        same.arrange(DOWN, buff=0.15).move_to([-4.65, 0.45, 0])
        self.at("same")
        self.play(FadeIn(same, shift=0.2 * RIGHT), run_time=0.6)
        for cue, part in [("embeddings", emb), ("attention", attn), ("prediction", logits)]:
            self.at(cue)
            self.play(Indicate(part, color=YELLOW, scale_factor=1.08), run_time=0.7)

        head = Text("data & training signal", font_size=22, color=GREY_B)
        labels = VGroup(*[Text(s, font_size=24) for s in
                          ("pretraining: internet text", "SFT: conversations", "preferences / RL: feedback")])
        labels.arrange(DOWN, buff=0.45, aligned_edge=LEFT).set_color(YELLOW)
        VGroup(head, labels).arrange(DOWN, buff=0.4, aligned_edge=LEFT).move_to([1.95, 0.45, 0], aligned_edge=LEFT)
        for i, cue in enumerate(["data", "training", "change"]):
            self.at(cue)
            anims = [FadeIn(labels[i], shift=0.3 * LEFT)]
            if i == 0:
                anims.append(FadeIn(head))
            else:
                anims.append(labels[i - 1].animate.set_color(GREY_B))
            self.play(*anims, run_time=0.45)
        self.end_section()

    # ------------------------------------------------------------------ 9. series recap (video 1 map)
    def recap(self):
        self.section(9)
        self.clear_stage()
        heading = Text("The whole journey", font_size=40).to_edge(UP, buff=0.5)
        row1 = VGroup(*[map_box(n, name) for n, name in
                        [("2", "Tokens"), ("3", "Embeddings"), ("4", "Position"), ("5–7", "Attention"),
                         ("8", "MLP"), ("10", "Probabilities")]]).arrange(RIGHT, buff=0.4).move_to([0, 1.3, 0])
        links1 = VGroup(*[Arrow(a[0].get_right(), b[0].get_left(), buff=0.03, color=GREY_C, stroke_width=3,
                                max_tip_length_to_length_ratio=0.45) for a, b in zip(row1[:-1], row1[1:])])
        block = DashedVMobject(SurroundingRectangle(VGroup(row1[3], row1[4]), buff=0.15, color=GREY_B,
                                                    corner_radius=0.15), num_dashes=60)
        block_label = Text("9 · Transformer block × N", font_size=22, color=GREY_B).next_to(block, DOWN, buff=0.12)
        row2 = VGroup(*[map_box(n, name, width=2.3) for n, name in
                        [("11", "Training"), ("12", "Tiny GPT"), ("13", "KV cache"), ("14", "Chatbot")]])
        row2.arrange(RIGHT, buff=0.9).move_to([0, -1.9, 0])
        links2 = VGroup(*[Arrow(a[0].get_right(), b[0].get_left(), buff=0.08, color=GREY_C, stroke_width=3,
                                max_tip_length_to_length_ratio=0.3) for a, b in zip(row2[:-1], row2[1:])])
        self.at("journey")
        parts = [m for pair in zip(row1, [*links1, VGroup(block, block_label)]) for m in pair]
        parts += [m for pair in zip(row2, [*links2, VGroup()]) for m in pair][:-1]
        self.play(Write(heading), LaggedStart(*[FadeIn(m, shift=0.2 * UP) for m in parts], lag_ratio=0.12),
                  run_time=1.2)

        steps = [("tokens", row1[0]), ("vectors", row1[1]), ("positions", row1[2]), ("attention", row1[3]),
                 ("mlps", row1[4])]
        prev = None
        for cue, box in steps:
            self.at(cue)
            self.play(light(box), *([unlight(prev)] if prev else []), run_time=0.5)
            prev = box
        self.at("blocks")
        self.play(unlight(prev), block.animate.set_stroke(ORANGE, 4), block_label.animate.set_color(ORANGE),
                  run_time=0.6)
        self.at("probabilities")
        self.play(light(row1[5]), run_time=0.5)

        y = -0.15
        a0, a1 = row1[5][0].get_bottom() + 0.05 * DOWN, row1[0][0].get_bottom() + 0.05 * DOWN
        loop = VGroup(Line(a0, [a0[0], y, 0], color=YELLOW, stroke_width=4),
                      Line([a0[0], y, 0], [a1[0], y, 0], color=YELLOW, stroke_width=4),
                      Arrow([a1[0], y, 0], a1, buff=0, color=YELLOW, stroke_width=4, max_tip_length_to_length_ratio=0.3))
        loop_label = Text("pick a token, repeat", font_size=22, color=YELLOW).next_to(loop[1], DOWN, buff=0.12)
        self.at("pick")
        self.play(Succession(Create(loop[0]), Create(loop[1]), GrowArrow(loop[2])), FadeIn(loop_label), run_time=0.8)
        self.at("repeat")
        self.play(unlight(row1[5]), light(row2[2], TEAL), run_time=0.5)
        self.at("training")
        self.play(unlight(row2[2]), light(row2[0], GREEN), loop.animate.set_color(GREY_B),
                  loop_label.animate.set_color(GREY_B), run_time=0.5)
        self.at("number")
        self.play(unlight(row2[0]), light(row2[1], GOLD), run_time=0.5)
        self.at("fine")
        self.play(unlight(row2[1]), light(row2[3], YELLOW), run_time=0.6)
        self.at("assistant")
        self.play(Indicate(row2[3], color=YELLOW, scale_factor=1.12), run_time=0.9)
        self.end_section()

    # ------------------------------------------------------------------ 10. farewell
    def farewell(self):
        self.section(10)
        card = VGroup(Text("How LLMs Work", font_size=48),
                      Text("Thanks for watching", font_size=30, color=GREY_B)).arrange(DOWN, buff=0.35)
        self.at("llm")
        self.clear_stage(run_time=0.5)
        self.play(FadeIn(card[0], scale=0.9), run_time=0.6)
        self.at("thanks")
        self.play(FadeIn(card[1], shift=0.2 * UP), run_time=0.6)
        self.end_section()
        finish(self, hold=2.0)
