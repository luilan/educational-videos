"""LLMs in Practice, episode 7 — Tool Use: How a Model Calls a Function.

Render from the repo root:  ./render.sh llms-in-practice p07
Every prompt, tool call, token count and reply on screen comes from code/p07_tool_use/tool_use.py
(Qwen2.5-1.5B-Instruct, its real chat template, greedy). The weather service is made up.
"""
from manim import *

from common import MODEL_COLOR, MONO, code_panel, finish, highlight, next_up_card
from intro import play_token_intro
from p07_script import NEXT, TAGLINE, TITLE
from voiced_scene import VoicedScene

SERIES_LABEL = "LLMs IN PRACTICE  ·  EPISODE 7"
TAG = ORANGE
Q1 = "What is the weather like in Milan right now? Do I need an umbrella?"
Q2 = "Do I need an umbrella in Milan right now?"
ANSWER = ("The current temperature in Milan is 18°C and it's currently experiencing\n"
          "light rain. It might be raining lightly so you might want to bring an\numbrella if you plan on going out.")
NO_CALL = ("To determine if you need an umbrella in Milan right now, I would need to\n"
           "check the current weather conditions for that specific location. Could you\n"
           "please provide me with your location so I can fetch the latest weather …")
SYSTEM = ["# Tools", "", "You may call one or more functions to assist with the user query.", "",
          "You are provided with function signatures within <tools></tools> XML tags:", "<tools>",
          '{"type": "function", "function": {"name": "get_weather",',
          ' "description": "Get the current weather in a city.",',
          ' "parameters": {"type": "object", "properties": {"city": {"type": "string",',
          ' "description": "The name of the city, e.g. Milan"}}, "required": ["city"]}}}', "</tools>", "",
          "For each function call, return a json object with function name and",
          "arguments within <tool_call></tool_call> XML tags: …"]
CODE = """while True:
    reply = generate(messages)
    if "<tool_call>" not in reply:
        return reply                              # a normal answer
    call = json.loads(between(reply, "<tool_call>", "</tool_call>"))
    result = TOOLS[call["name"]](**call["arguments"])     # run it
    messages += [tool_call_message(call), tool_response(result)]"""


def bubble(text, color, font_size=22):
    label = Text(text, font_size=font_size, line_spacing=0.8)
    box = RoundedRectangle(corner_radius=0.2, width=label.width + 0.6, height=label.height + 0.45, stroke_width=0,
                           fill_color=color, fill_opacity=0.9)
    return VGroup(box, label.move_to(box))


def labeled_box(text, color=MODEL_COLOR, width=2.4, height=1.0, font_size=26):
    label = Text(text, font_size=font_size, line_spacing=0.8)
    box = RoundedRectangle(corner_radius=0.2, width=max(width, label.width + 0.5), height=height, stroke_color=color,
                           fill_color=color, fill_opacity=0.25)
    return VGroup(box, label.move_to(box))


def mono_panel(lines, font_size=18, color=GREY_C, t2c=None):
    texts = VGroup(*[Text(l, font=MONO, font_size=font_size, t2c=t2c or {}) if l else
                     Rectangle(width=0.1, height=font_size / 110, stroke_width=0, fill_opacity=0) for l in lines])
    texts.arrange(DOWN, aligned_edge=LEFT, buff=0.1)
    box = RoundedRectangle(corner_radius=0.2, width=texts.width + 0.6, height=texts.height + 0.5, stroke_color=color,
                           fill_color=GREY_E, fill_opacity=1)
    return VGroup(box, texts.move_to(box))


TAGS = {"<tool_call>": TAG, "</tool_call>": TAG, "<tools>": TAG, "</tools>": TAG, "<tool_response>": TAG,
        "</tool_response>": TAG}


class ToolUseVideo(VoicedScene):
    VIDEO = "p07"

    def construct(self):
        play_token_intro(self, TITLE, 7, TAGLINE, label=SERIES_LABEL)
        self.hook()         # 1
        self.trick()        # 2
        self.describe()     # 3
        self.call()         # 4
        self.run()          # 5
        self.back()         # 6
        self.failure()      # 7
        self.code()         # 8
        self.safety()       # 9
        self.outro()        # 10

    def clear_stage(self, *keep, run_time=0.6):
        self.play(*[FadeOut(m) for m in self.mobjects if m not in keep], run_time=run_time)

    # ------------------------------------------------------------------ 1. hook
    def hook(self):
        self.section(1)
        model = labeled_box("model", width=2.6, height=1.2, font_size=32).move_to([-2.9, 0.4, 0])
        tin = Text("text in", font_size=24, color=GREY_B).next_to(model, LEFT, buff=0.9)
        tout = Text("text out", font_size=24, color=GREY_B).next_to(model, RIGHT, buff=0.9)
        a1 = Arrow(tin.get_right(), model.get_left(), buff=0.1, color=GREY_B)
        a2 = Arrow(model.get_right(), tout.get_left(), buff=0.1, color=GREY_B)
        q = Text("today's weather?", font_size=28, color=YELLOW).next_to(model, UP, buff=0.5)
        self.at("weather")
        self.play(FadeIn(model), FadeIn(q, shift=0.2 * DOWN), run_time=0.6)
        nots = VGroup(*[Text(t, font_size=24, color=RED_B) for t in ["✗ no window", "✗ no internet", "✗ no clock"]])
        nots.arrange(RIGHT, buff=0.5).next_to(model, DOWN, buff=0.5)
        for i, cue in enumerate(["window", "internet", "clock"]):
            self.at(cue)
            self.play(FadeIn(nots[i]), run_time=0.3)
        self.at("text")
        self.play(FadeIn(tin), GrowArrow(a1), GrowArrow(a2), FadeIn(tout), run_time=0.6)
        can = VGroup(*[Text(t, font_size=26, color=GREEN_B) for t in ["✓ check the weather", "✓ search the web",
                                                                        "✓ run code"]])
        can.arrange(DOWN, aligned_edge=LEFT, buff=0.3).move_to([4.5, 0.4, 0])
        self.at("assistants")
        self.play(LaggedStart(*[FadeIn(c, shift=0.2 * LEFT) for c in can], lag_ratio=0.4), run_time=1.2)
        how = Text("how?", font_size=40, color=YELLOW).move_to([4.5, -2.2, 0])
        self.at("how")
        self.play(FadeIn(how, scale=1.3), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 2. the trick
    def trick(self):
        self.section(2)
        self.clear_stage()
        model = labeled_box("model", width=2.4, height=1.1).move_to([-4.0, 0.3, 0])
        req = bubble("a request,\nin a trained format", GREY_D, 22).move_to([0, 0.3, 0])
        app = labeled_box("your program\ndoes the work", GREEN_C, width=3.0, height=1.3, font_size=24).move_to([4.2, 0.3, 0])
        a1 = Arrow(model.get_right(), req.get_left(), buff=0.15, color=GREY_B)
        a2 = Arrow(req.get_right(), app.get_left(), buff=0.15, color=GREY_B)
        never = Text("the model never runs anything", font_size=30, color=YELLOW).to_edge(UP, buff=0.8)
        self.at("never")
        self.play(FadeIn(model), FadeIn(never), run_time=0.5)
        self.at("writes")
        self.play(GrowArrow(a1), FadeIn(req, shift=0.2 * RIGHT), run_time=0.5)
        self.at("program")
        self.play(GrowArrow(a2), FadeIn(app, shift=0.2 * RIGHT), run_time=0.5)
        self.end_section()

    # ------------------------------------------------------------------ 3. describe the tools
    def describe(self):
        self.section(3)
        self.clear_stage()
        step = Text("1 · describe the tools", font_size=30, color=YELLOW).to_edge(UP, buff=0.35)
        self.at("one")
        self.play(FadeIn(step), run_time=0.4)
        sig = mono_panel(['def get_weather(city: str):', '    """Get the current weather in a city."""'], 20, TEAL_C)
        sig.move_to([0, 2.0, 0])
        self.at("name")
        self.play(FadeIn(sig), run_time=0.5)
        panel = mono_panel(SYSTEM, 15, GREY_C, TAGS).move_to([0, -0.9, 0])
        label = Text("in the system prompt (real Qwen2.5 template)", font_size=20, color=GREY_B).next_to(panel, UP, 0.1)
        label.align_to(panel, RIGHT)
        self.at("pastes")
        self.play(sig.animate.scale(0.8).to_corner(UL, buff=0.35).shift(0.75 * DOWN), FadeIn(panel[0]), FadeIn(label),
                  run_time=0.6)
        groups = [(range(0, 4), "call"), (range(4, 11), "signatures"), (range(11, 14), "write")]
        for rows, cue in groups:
            self.at(cue)
            self.play(LaggedStart(*[FadeIn(panel[1][i]) for i in rows], lag_ratio=0.1), run_time=0.6)
        more = Text("just more text in the context", font_size=26, color=YELLOW).next_to(panel, DOWN, buff=0.2)
        self.at("context")
        self.play(FadeIn(more), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 4. the model writes a call
    def call(self):
        self.section(4)
        self.clear_stage()
        step = Text("2 · the model writes a call", font_size=30, color=YELLOW).to_edge(UP, buff=0.35)
        q = bubble(Q1, BLUE_E, 22).move_to([0.8, 1.9, 0])
        self.at("two")
        self.play(FadeIn(step), run_time=0.4)
        self.at("weather")
        self.play(FadeIn(q, shift=0.2 * UP), run_time=0.5)
        reply = mono_panel(["<tool_call>", '{"name": "get_weather", "arguments": {"city": "Milan"}}', "</tool_call>"],
                           22, TAG, TAGS).move_to([-0.6, -0.3, 0])
        tag = Text("the model's real reply", font_size=20, color=GREY_B).next_to(reply, DOWN, buff=0.15)
        self.at("reply")
        self.play(FadeIn(reply[0]), run_time=0.3)
        self.at("request")
        self.play(FadeIn(reply[1], shift=0.2 * RIGHT), FadeIn(tag), run_time=0.6)
        self.at("milan")
        self.play(Indicate(reply[1][1], color=YELLOW, scale_factor=1.05), run_time=0.6)
        self.end_section()

    # ------------------------------------------------------------------ 5. the app runs it
    def run(self):
        self.section(5)
        self.clear_stage()
        step = Text("3 · your program runs it", font_size=30, color=YELLOW).to_edge(UP, buff=0.35)
        self.at("three")
        self.play(FadeIn(step), run_time=0.4)
        parse = labeled_box("parse JSON", GREY_B, width=2.4, font_size=24).move_to([-4.5, 0.6, 0])
        func = labeled_box('get_weather("Milan")', GREEN_C, width=3.6, font_size=24).move_to([0, 0.6, 0])
        result = mono_panel(['{"city": "Milan",', ' "temperature_c": 18,', ' "sky": "light rain"}'], 22, GREEN_C)
        result.move_to([4.5, 0.6, 0])
        arrows = VGroup(Arrow(parse.get_right(), func.get_left(), buff=0.15, color=GREY_B),
                        Arrow(func.get_right(), result.get_left(), buff=0.15, color=GREY_B))
        self.at("json")
        self.play(FadeIn(parse), run_time=0.4)
        self.at("runs")
        self.play(GrowArrow(arrows[0]), FadeIn(func), run_time=0.5)
        made = Text("a made-up weather service", font_size=20, color=GREY_B).next_to(func, DOWN, buff=0.2)
        self.at("made")
        self.play(FadeIn(made), run_time=0.3)
        self.at("18")
        self.play(GrowArrow(arrows[1]), FadeIn(result, shift=0.2 * LEFT), run_time=0.6)
        only = Text("the only moment anything happens in the world", font_size=28, color=YELLOW).move_to([0, -2.2, 0])
        self.at("only")
        self.play(FadeIn(only, shift=0.2 * UP), Indicate(func, color=YELLOW), run_time=0.7)
        self.end_section()

    # ------------------------------------------------------------------ 6. back into the context
    def back(self):
        self.section(6)
        self.clear_stage()
        step = Text("4 · the result goes back into the context", font_size=30, color=YELLOW).to_edge(UP, buff=0.35)
        self.at("four")
        self.play(FadeIn(step), run_time=0.4)
        rows = [("system: tools …", GREY_B), ("user: " + Q1[:40] + " …", BLUE_C),
                ("assistant: <tool_call> get_weather(Milan) </tool_call>", TAG),
                ("tool: <tool_response> 18 °C, light rain </tool_response>", GREEN_C), ("assistant: …", WHITE)]
        stack = VGroup()
        for text, color in rows:
            label = Text(text, font=MONO, font_size=18, color=color)
            box = RoundedRectangle(corner_radius=0.1, width=9.6, height=0.5, stroke_color=color, stroke_width=1.5,
                                   fill_color=color, fill_opacity=0.1)
            stack.add(VGroup(box, label.move_to(box).align_to(box, LEFT).shift(0.2 * RIGHT)))
        stack.arrange(DOWN, buff=0.1).move_to([0, 1.0, 0])
        self.play(FadeIn(stack[:3]), run_time=0.4)
        self.at("response")
        self.play(FadeIn(stack[3], shift=0.2 * UP), run_time=0.5)
        self.at("continues")
        self.play(FadeIn(stack[4]), run_time=0.3)
        count = Text("264 tokens", font_size=30, color=YELLOW).next_to(stack, RIGHT, buff=0.2).shift(0.3 * LEFT)
        count.next_to(stack, DOWN, buff=0.2).align_to(stack, RIGHT)
        self.at("264")
        self.play(FadeIn(count), run_time=0.4)
        ans = bubble(ANSWER, GREEN_E, 20).move_to([0, -1.9, 0])
        self.at("answers")
        self.play(FadeIn(ans[0]), AddTextLetterByLetter(ans[1], run_time=2.0))
        self.end_section()

    # ------------------------------------------------------------------ 7. failure
    def failure(self):
        self.section(7)
        self.clear_stage()
        q = bubble(Q2, BLUE_E, 24).move_to([1.5, 2.3, 0])
        self.at("vaguely")
        self.play(FadeIn(q, shift=0.2 * UP), run_time=0.5)
        reply = bubble(NO_CALL, RED_E, 20).move_to([-0.6, 0.6, 0])
        self.at("didnt")
        self.play(FadeIn(reply, shift=0.2 * UP), run_time=0.6)
        no = Text("no tool call: it asked for the location it was given", font_size=24, color=RED_B)
        no.next_to(reply, DOWN, buff=0.25)
        self.at("location")
        self.play(FadeIn(no), run_time=0.4)
        tips = VGroup(Text("larger models call tools more reliably", font_size=24),
                      Text("clear tool descriptions help", font_size=24),
                      Text("handle: no call · broken JSON", font_size=24)).arrange(DOWN, aligned_edge=LEFT, buff=0.2)
        tips.set_color(GREEN_B).move_to([0, -2.4, 0])
        for i, cue in enumerate(["larger", "clear", "handle"]):
            self.at(cue)
            self.play(FadeIn(tips[i], shift=0.2 * RIGHT), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 8. code
    def code(self):
        self.section(8)
        self.clear_stage()
        code, hl = code_panel(CODE, font_size=20)
        code.move_to([0, 0.3, 0])
        self.at("loop")
        self.play(FadeIn(code, shift=0.2 * UP), run_time=0.5)
        hl.match_y(code.line_numbers[1])
        self.at("generate")
        self.play(Create(hl), run_time=0.3)
        self.at("done")
        self.play(highlight(hl, code, 3), run_time=0.4)
        self.at("parse")
        self.play(highlight(hl, code, 4), run_time=0.4)
        self.at("run")
        self.play(highlight(hl, code, 5), run_time=0.4)
        self.at("append")
        self.play(highlight(hl, code, 6), run_time=0.4)
        self.at("again")
        self.play(highlight(hl, code, 1), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 9. safety
    def safety(self):
        self.section(9)
        self.clear_stage()
        head = Text("the model chooses what to call · you choose what it can do", font_size=28).to_edge(UP, buff=0.5)
        self.at("chooses")
        self.play(FadeIn(head), run_time=0.5)
        safe = labeled_box("get_weather\nread-only: harmless", GREEN_C, width=4.0, height=1.4, font_size=24)
        risky = labeled_box("send_email · make_payment\nactions: real consequences", RED_C, width=5.0, height=1.4,
                            font_size=24)
        VGroup(safe, risky).arrange(RIGHT, buff=0.8).move_to([0, 1.0, 0])
        self.at("harmless")
        self.play(FadeIn(safe, shift=0.2 * UP), run_time=0.5)
        self.at("sending")
        self.play(FadeIn(risky, shift=0.2 * UP), run_time=0.5)
        checks = VGroup(*[Text(t, font_size=26) for t in ["✓ ask the user to confirm", "✓ check the arguments",
                                                           "✓ never run model code without limits"]])
        checks.arrange(DOWN, aligned_edge=LEFT, buff=0.3).move_to([0, -1.6, 0]).set_color(YELLOW)
        for i, cue in enumerate(["confirm", "arguments", "never"]):
            self.at(cue)
            self.play(FadeIn(checks[i], shift=0.2 * RIGHT), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 10. outro
    def outro(self):
        self.section(10)
        self.clear_stage(run_time=0.4)
        line = Text("several tools + a goal + this loop = an agent", font_size=32, color=YELLOW).move_to([0, 1.2, 0])
        self.at("several")
        self.play(FadeIn(line), run_time=0.6)
        card = next_up_card(NEXT).move_to([0, -0.8, 0])
        self.at("next")
        self.play(FadeIn(card, shift=0.2 * UP), run_time=0.5)
        self.end_section()
        finish(self, hold=1.5)
