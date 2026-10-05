"""LLMs in Practice, episode 8 — Agents: Think, Act, Observe.

Render from the repo root:  ./render.sh llms-in-practice p08
Every trace line, token count and answer on screen comes from code/p08_agent/agent.py
(Qwen2.5-3B-Instruct, greedy; VALIDATE False for the first try, True for the second) and, for the invented shops,
the same agent run with Qwen2.5-1.5B-Instruct. The shops, calendar and map are made up.
"""
from manim import *

from common import MODEL_COLOR, MONO, code_panel, finish, highlight, next_up_card
from intro import play_token_intro
from p08_script import NEXT, TAGLINE, TITLE
from voiced_scene import VoicedScene

SERIES_LABEL = "LLMs IN PRACTICE  ·  EPISODE 8"
GOAL = "I want to buy gluten-free bread today.\nWhere can I get it, and how long is the walk?"
FIRST = [("find_shops(day='today')", "shops: []", RED_B),
         ("get_walking_time('shop for gluten-free bread')", "30 minutes", RED_B)]
FIRST_ANSWER = ("Today there are no shops in Milan that sell gluten-free bread. The walk to the\n"
                "nearest shop selling gluten-free bread would take approximately 30 minutes.")
INVENTED = ("1. Bread & Butter - Located at Via XX Settembre 20, … This shop offers various\n"
            "   types of bread including gluten-free options.\n"
            "2. La Piazza - Situated at Via della Spiga 35, 20124 Milano (Milan), …")
SECOND = [(1, "find_shops(day='today')", "error: day must be a weekday name … call get_today first", RED_B),
          (1, "get_walking_time('shop for gluten-free bread')", "error: unknown place … use a shop name", RED_B),
          (2, "get_today()", "Wednesday", GREEN_B),
          (2, "find_shops(day='today')", "error: day must be a weekday name …", RED_B),
          (3, "get_today()", "Wednesday", GREEN_B),
          (4, "find_shops(day='Wednesday')", "shops: ['Pane Vivo']", GREEN_B),
          (5, "get_walking_time('Pane Vivo')", "18 minutes", GREEN_B)]
SECOND_ANSWER = "Today is Wednesday, and you can buy gluten-free bread at Pane Vivo.\nThe walk will take approximately 18 minutes."
TOKENS = [422, 551, 652, 691, 760, 814]
CODE = """for step in range(MAX_STEPS):
    reply = generate(messages)                     # think
    calls = find_tool_calls(reply)
    if not calls:
        return reply                               # final answer
    for call in calls:
        result = TOOLS[call["name"]](**call["arguments"])    # act
        messages += [call_message(call), tool_response(result)]  # observe"""
VALIDATION = """def find_shops(product, day):
    if day not in WEEKDAYS:
        return {"error": f"day must be a weekday name such as Monday, "
                         f"not {day!r}. Call get_today first."}
    ..."""


def bubble(text, color, font_size=22, mono=False):
    label = Text(text, font_size=font_size, line_spacing=0.8, **({"font": MONO} if mono else {}))
    box = RoundedRectangle(corner_radius=0.2, width=label.width + 0.6, height=label.height + 0.45, stroke_width=0,
                           fill_color=color, fill_opacity=0.9)
    return VGroup(box, label.move_to(box))


def labeled_box(text, color=MODEL_COLOR, width=2.4, height=1.0, font_size=26):
    label = Text(text, font_size=font_size, line_spacing=0.8)
    box = RoundedRectangle(corner_radius=0.2, width=max(width, label.width + 0.5), height=height, stroke_color=color,
                           fill_color=color, fill_opacity=0.25)
    return VGroup(box, label.move_to(box))


def trace_line(call, result, color, step=None, font_size=18):
    parts = [Text(call, font=MONO, font_size=font_size), Text("→ " + result, font=MONO, font_size=font_size,
                                                              color=color)]
    if step is not None:
        parts.insert(0, Text(f"step {step}", font=MONO, font_size=font_size, color=GREY_B))
    return VGroup(*parts).arrange(RIGHT, buff=0.3)


class AgentVideo(VoicedScene):
    VIDEO = "p08"

    def construct(self):
        play_token_intro(self, TITLE, 8, TAGLINE, label=SERIES_LABEL)
        self.hook()          # 1
        self.loop()          # 2
        self.task()          # 3
        self.first()         # 4
        self.smaller()       # 5
        self.fix()           # 6
        self.second()        # 7
        self.cost()          # 8
        self.code()          # 9
        self.lessons()       # 10
        self.outro()         # 11

    def clear_stage(self, *keep, run_time=0.6):
        self.play(*[FadeOut(m) for m in self.mobjects if m not in keep], run_time=run_time)

    # ------------------------------------------------------------------ 1. hook
    def hook(self):
        self.section(1)
        one = VGroup(bubble("question", BLUE_E), Arrow(LEFT, RIGHT, color=GREY_B), bubble("1 tool call", GREY_D),
                     Arrow(LEFT, RIGHT, color=GREY_B), bubble("answer", GREEN_E)).arrange(RIGHT, buff=0.2)
        one.move_to([0, 2.0, 0])
        self.at("last")
        self.play(FadeIn(one), run_time=0.6)
        steps = VGroup(*[labeled_box(f"step {i}", GREY_B, width=1.6, height=0.8, font_size=22) for i in range(1, 5)])
        steps.arrange(RIGHT, buff=0.6).move_to([0, 0, 0])
        links = VGroup(*[Arrow(a.get_right(), b.get_left(), buff=0.08, color=YELLOW) for a, b in zip(steps, steps[1:])])
        self.at("several")
        self.play(LaggedStart(*[FadeIn(s, shift=0.2 * RIGHT) for s in steps], lag_ratio=0.2), run_time=0.8)
        self.at("depends")
        self.play(LaggedStart(*[GrowArrow(l) for l in links], lag_ratio=0.2), run_time=0.6)
        name = Text("an agent: a model choosing tools, in a loop, until a goal is reached", font_size=26,
                    color=YELLOW).move_to([0, -2.0, 0])
        self.at("loop")
        self.play(FadeIn(name, shift=0.2 * UP), run_time=0.6)
        self.end_section()

    # ------------------------------------------------------------------ 2. the loop
    def loop(self):
        self.section(2)
        self.clear_stage()
        c, r, gap = np.array([-1.6, -0.3, 0]), 2.3, 0.75
        angles = [PI / 2 - k * 2 * PI / 3 for k in range(3)]
        nodes = VGroup(labeled_box("think\nmodel decides", MODEL_COLOR, width=2.6, height=1.2, font_size=24),
                       labeled_box("act\nprogram runs a tool", GREEN_C, width=2.6, height=1.2, font_size=24),
                       labeled_box("observe\nresult → context", BLUE_C, width=2.6, height=1.2, font_size=24))
        for n, a in zip(nodes, angles):
            n.move_to(c + r * np.array([np.cos(a), np.sin(a), 0]))
        arcs = VGroup()
        for k in range(3):
            a0, a1 = angles[k] - gap, angles[k] - 2 * PI / 3 + gap
            arcs.add(Arc(radius=r, start_angle=a0, angle=a1 - a0, arc_center=c, color=GREY_B, stroke_width=4)
                     .add_tip(tip_length=0.25))
        for cue, k in [("think", 0), ("act", 1), ("observe", 2)]:
            self.at(cue)
            self.play(FadeIn(nodes[k], scale=0.9), run_time=0.4)
        self.at("repeat")
        self.play(LaggedStart(*[Create(a) for a in arcs], lag_ratio=0.3), run_time=0.9)
        exit_box = VGroup(Text("stop when:", font_size=24, color=GREY_B), Text("final answer", font_size=26, color=GREEN_B),
                          Text("or a step limit", font_size=26, color=RED_B)).arrange(DOWN, aligned_edge=LEFT, buff=0.2)
        exit_box.move_to([4.6, -0.2, 0])
        self.at("answer")
        self.play(FadeIn(exit_box[:2]), run_time=0.4)
        self.at("limit")
        self.play(FadeIn(exit_box[2]), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 3. the task
    def task(self):
        self.section(3)
        self.clear_stage()
        goal = bubble(GOAL, BLUE_E, 24).move_to([0, 2.3, 0])
        self.at("buy")
        self.play(FadeIn(goal, shift=0.2 * DOWN), run_time=0.6)
        tools = VGroup(labeled_box("get_today()", GREY_B, width=3.0, height=0.8, font_size=22),
                       labeled_box("find_shops(product, day)", GREY_B, width=3.0, height=0.8, font_size=22),
                       labeled_box("get_walking_time(destination)", GREY_B, width=3.0, height=0.8, font_size=22))
        tools.arrange(RIGHT, buff=0.7).move_to([0, 0.2, 0])
        made = Text("three made-up tools", font_size=22, color=GREY_B).next_to(tools, UP, buff=0.2)
        self.at("tools")
        self.play(FadeIn(made), run_time=0.3)
        for cue, k in [("date", 0), ("shops", 1), ("walking", 2)]:
            self.at(cue)
            self.play(FadeIn(tools[k], shift=0.2 * UP), run_time=0.35)
        arrows = VGroup(*[Arrow(a.get_right(), b.get_left(), buff=0.08, color=YELLOW) for a, b in zip(tools, tools[1:])])
        order = VGroup(*[Text(t, font_size=26, color=YELLOW) for t in ["1 · the day", "2 · the shops", "3 · the walk"]])
        for t, box in zip(order, tools):
            t.next_to(box, DOWN, buff=0.35)
        self.at("order")
        self.play(*[GrowArrow(a) for a in arrows], run_time=0.5)
        for cue, k in [("day", 0), ("shops", 1), ("walk", 2)]:
            self.at(cue)
            self.play(FadeIn(order[k]), Indicate(tools[k], color=YELLOW), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 4. first try
    def first(self):
        self.section(4)
        self.clear_stage()
        head = Text("first try · Qwen2.5-3B · tools without checks", font_size=28).to_edge(UP, buff=0.4)
        self.at("3")
        self.play(FadeIn(head), run_time=0.4)
        lines = VGroup(*[trace_line(c, r, col) for c, r, col in FIRST]).arrange(DOWN, aligned_edge=LEFT, buff=0.3)
        lines.move_to([0, 1.5, 0])
        self.at("findshops")
        self.play(FadeIn(lines[0][0]), run_time=0.4)
        self.at("nothing")
        self.play(FadeIn(lines[0][1]), run_time=0.4)
        self.at("walk")
        self.play(FadeIn(lines[1]), run_time=0.5)
        ans = bubble(FIRST_ANSWER, RED_E, 20).move_to([0, -0.4, 0])
        self.at("answer")
        self.play(FadeIn(ans, shift=0.2 * UP), run_time=0.5)
        wrong = Text("✗ wrong: Pane Vivo sells it on Wednesdays", font_size=28, color=RED).next_to(ans, DOWN, buff=0.4)
        self.at("wrong")
        self.play(FadeIn(wrong, scale=1.2), run_time=0.5)
        conf = Text("confident, and wrong", font_size=32, color=YELLOW).next_to(wrong, DOWN, buff=0.35)
        self.at("confident")
        self.play(FadeIn(conf), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 5. smaller model
    def smaller(self):
        self.section(5)
        self.clear_stage()
        head = Text("Qwen2.5-1.5B, after getting the day:", font_size=28).to_edge(UP, buff=0.8)
        self.at("smaller")
        self.play(FadeIn(head), run_time=0.4)
        inv = bubble(INVENTED, RED_E, 20, mono=True).move_to([0, 0.3, 0])
        tag = Text("invented shops and addresses", font_size=30, color=RED).next_to(inv, DOWN, buff=0.4)
        self.at("invented")
        self.play(FadeIn(inv, shift=0.2 * UP), FadeIn(tag), run_time=0.7)
        self.end_section()

    # ------------------------------------------------------------------ 6. the fix
    def fix(self):
        self.section(6)
        self.clear_stage()
        head = Text("the fix is in the tools", font_size=34, color=YELLOW).to_edge(UP, buff=0.5)
        self.at("tools")
        self.play(FadeIn(head), run_time=0.4)
        code, hl = code_panel(VALIDATION, font_size=22)
        code.move_to([0, 0.6, 0])
        self.at("check")
        self.play(FadeIn(code), run_time=0.5)
        hl.match_y(code.line_numbers[1])
        self.play(Create(hl), run_time=0.3)
        self.at("why")
        self.play(highlight(hl, code, 2), run_time=0.4)
        errs = VGroup(Text("“day must be a weekday name … Call get_today first.”", font_size=24, color=GREEN_B),
                      Text("“unknown place … Use a shop name returned by find_shops.”", font_size=24, color=GREEN_B))
        errs.arrange(DOWN, buff=0.25).next_to(code, DOWN, buff=0.45)
        self.at("day")
        self.play(FadeIn(errs[0], shift=0.2 * UP), run_time=0.4)
        self.at("unknown")
        self.play(FadeIn(errs[1], shift=0.2 * UP), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 7. second try
    def second(self):
        self.section(7)
        self.clear_stage()
        head = Text("second try · same model · tools with helpful errors", font_size=28).to_edge(UP, buff=0.35)
        self.at("second")
        self.play(FadeIn(head), run_time=0.4)
        lines = VGroup(*[trace_line(c, r, col, s, 16) for s, c, r, col in SECOND]).arrange(DOWN, aligned_edge=LEFT,
                                                                                          buff=0.18)
        lines.move_to([0, 0.9, 0])
        if lines.width > 13.4:
            lines.scale_to_fit_width(13.4)
        for idx, cue in [((0, 1), "errors"), ((2,), "wednesday"), ((3,), "mistake"), ((4,), "three"),
                         ((5,), "four"), ((6,), "five")]:
            self.at(cue)
            self.play(*[FadeIn(lines[i], shift=0.2 * RIGHT) for i in idx], run_time=0.45)
        ans = bubble(SECOND_ANSWER, GREEN_E, 20).next_to(lines, DOWN, buff=0.35)
        step6 = Text("step 6", font=MONO, font_size=16, color=GREY_B).next_to(ans, LEFT, buff=0.25)
        self.at("six")
        self.play(FadeIn(ans, shift=0.2 * UP), FadeIn(step6), run_time=0.5)
        tally = Text("6 steps · 3 errors · each fixed by reading the error", font_size=26, color=YELLOW)
        tally.to_edge(DOWN, buff=0.35)
        self.at("steps")
        self.play(FadeIn(tally), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 8. cost
    def cost(self):
        self.section(8)
        self.clear_stage()
        head = Text("every step resends everything so far", font_size=30).to_edge(UP, buff=0.5)
        self.at("context")
        self.play(FadeIn(head), run_time=0.4)
        bars = VGroup(*[Rectangle(width=0.9, height=t / 200, stroke_width=0, fill_color=BLUE_C, fill_opacity=0.85)
                        for t in TOKENS]).arrange(RIGHT, buff=0.5, aligned_edge=DOWN).move_to([0, -2.5, 0], aligned_edge=DOWN)
        labels = VGroup(*[Text(str(t), font=MONO, font_size=20).next_to(b, UP, buff=0.1) for t, b in zip(TOKENS, bars)])
        steps = VGroup(*[Text(f"step {i}", font_size=20, color=GREY_B).next_to(b, DOWN, buff=0.12)
                         for i, b in enumerate(bars, 1)])
        self.at("422")
        self.play(GrowFromEdge(bars[0], DOWN), FadeIn(labels[0]), FadeIn(steps[0]), run_time=0.4)
        self.play(LaggedStart(*[AnimationGroup(GrowFromEdge(b, DOWN), FadeIn(l), FadeIn(s))
                                for b, l, s in zip(bars[1:], labels[1:], steps[1:])], lag_ratio=0.3), run_time=1.4)
        self.at("814")
        self.play(Indicate(labels[-1], color=YELLOW, scale_factor=1.4), run_time=0.5)
        note = Text("tokens per step: slow, costly, and long runs can overflow the window (ep. 2)", font_size=22,
                    color=GREY_A).next_to(head, DOWN, buff=0.3)
        self.at("slow")
        self.play(FadeIn(note), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 9. code
    def code(self):
        self.section(9)
        self.clear_stage()
        code, hl = code_panel(CODE, font_size=20)
        code.move_to([0, 0.3, 0])
        self.at("limit")
        self.play(FadeIn(code, shift=0.2 * UP), run_time=0.5)
        hl.match_y(code.line_numbers[0])
        self.play(Create(hl), run_time=0.3)
        self.at("generate")
        self.play(highlight(hl, code, 1), run_time=0.4)
        self.at("final")
        self.play(highlight(hl, code, 4), run_time=0.4)
        self.at("run")
        self.play(highlight(hl, code, 6), run_time=0.4)
        self.at("append")
        self.play(highlight(hl, code, 7), run_time=0.4)
        self.at("around")
        self.play(highlight(hl, code, 0), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 10. lessons
    def lessons(self):
        self.section(10)
        self.clear_stage()
        tips = VGroup(*[Text(t, font_size=28) for t in ["a step limit", "tools that check inputs, with helpful errors",
                                                         "a small, clear toolset", "a log of every step",
                                                         "a human approves actions with consequences",
                                                         "a capable enough model"]])
        tips.arrange(DOWN, aligned_edge=LEFT, buff=0.32).move_to([0.3, 0, 0])
        dots = VGroup(*[Text("✓", font_size=28, color=GREEN).next_to(t, LEFT, buff=0.3) for t in tips])
        for i, cue in enumerate(["limit", "check", "small", "log", "human", "capable"]):
            self.at(cue)
            self.play(FadeIn(tips[i], shift=0.2 * RIGHT), FadeIn(dots[i]), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 11. outro
    def outro(self):
        self.section(11)
        self.clear_stage(run_time=0.4)
        card = next_up_card(NEXT)
        self.at("next")
        self.play(FadeIn(card, shift=0.2 * UP), run_time=0.6)
        self.end_section()
        finish(self, hold=1.5)
