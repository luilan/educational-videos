"""How AI Agents Work: Inside the Harness, episode 3 — Tool Calling Under the Hood.

Render from the repo root:  ./render.sh agents a03
Quoted text is real: Qwen3's chat template output (show_prompt.py), a raw Qwen3 4B sample (raw_samples.py, seed 0),
and a Codex CLI 0.163.0 request captured through the logging proxy (codex_*.json). Token counts: Qwen3 tokenizer.
Codex code: openai/codex @ 89461ea, codex-rs/core/src/tools/handlers/mod.rs.
"""
from manim import *

from common import MONO, code_panel, finish, highlight, next_up_card
from intro import play_token_intro
from a03_script import NEXT, TAGLINE, TITLE
from voiced_scene import VoicedScene

SERIES_LABEL = "HOW AI AGENTS WORK  ·  EPISODE 3"
MODEL_C, HARNESS_C, TOOL_C = PURPLE_B, TEAL_C, GOLD
SAMPLES = [388, 459, 494, 251, 406, 300, 256, 324, 268, 288]   # tokens written per raw sample (reasoning + call)
CALL_TOKENS = 25                                                # the <tool_call> block itself
PROMPT = """<|im_start|>system
You are a coding agent working in a project folder. ...
# Tools
You may call one or more functions to assist with the user query.
You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"type": "function", "function": {"name": "shell", "description": "Run a bash command ...",
 "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, ...}}}
</tools>
For each function call, return a json object with function name and arguments
within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call><|im_end|>
<|im_start|>user
The test in test_stats.py fails. Run it with: python3 test_stats.py.<|im_end|>"""
SAMPLE = """Okay, let me try to figure out how to help the user. They mentioned that the
test in test_stats.py fails and they want me to run it ... (1,570 characters)
... I think that's it. Let me output the tool call.
</think>"""
CALL = """<tool_call>
{"name": "shell", "arguments": {"command": "python3 test_stats.py"}}
</tool_call>"""
ROUNDTRIP = """<|im_start|>assistant
<tool_call>
{"name": "shell", "arguments": {"command": "python3 test_stats.py"}}
</tool_call><|im_end|>
<|im_start|>user
<tool_response>
exit code 1
AssertionError
</tool_response><|im_end|>"""
PARAMS = ["cmd (required)", "workdir", "yield_time_ms", "max_output_tokens", "sandbox_permissions", "justification",
          "prefix_rule", "login", "shell", "tty"]
CODEX_CALL = """{"type": "function_call", "name": "exec_command",
 "arguments": "{\\"cmd\\":\\"ls\\",\\"workdir\\":\\"/work\\"}", "call_id": "call_2tfwp3m3"}"""
CODEX_RESULT = """{"type": "function_call_output", "call_id": "call_2tfwp3m3",
 "output": "Chunk ID: 876fef
Wall time: 0.0000 seconds
Process exited with code 0
Original token count: 3
Output:
hello.py"}"""
CODE = """def parse_tool_call(text):
    m = re.search(r"<tool_call>(.*?)</tool_call>", text, re.S)
    if not m:
        return None, "no tool call"
    try:
        return json.loads(m.group(1)), None
    except json.JSONDecodeError as e:
        return None, f"failed to parse function arguments: {e}\""""


def card(title, body, color=GREY_B, fs=14, title_fs=16):
    t = Text(body, font=MONO, font_size=fs, line_spacing=0.85)
    hd = Text(title, font_size=title_fs, color=color)
    g = VGroup(hd, t).arrange(DOWN, aligned_edge=LEFT, buff=0.15)
    r = RoundedRectangle(width=g.width + 0.45, height=g.height + 0.35, corner_radius=0.12, color=color, fill_opacity=0.07)
    return VGroup(r, g.move_to(r))


def hbar(label, value, scale, color, y, x0=-3.0):
    b = Rectangle(width=value * scale, height=0.45, stroke_width=0, fill_color=color, fill_opacity=0.9).move_to([x0, y, 0], aligned_edge=LEFT)
    return VGroup(Text(label, font_size=18).next_to(b, LEFT, buff=0.2).align_to([x0 - 0.2, 0, 0], RIGHT), b,
                  Text(f"{value}", font=MONO, font_size=18).next_to(b, RIGHT, buff=0.15))


class ToolCallingVideo(VoicedScene):
    VIDEO = "a03"

    def construct(self):
        play_token_intro(self, TITLE, 3, TAGLINE, label=SERIES_LABEL)
        self.hook()        # 1
        self.reads()       # 2
        self.writes()      # 3
        self.parsing()     # 4
        self.back()        # 5
        self.codex_tool()  # 6
        self.codex_result()  # 7
        self.wrong()       # 8
        self.code()        # 9
        self.outro()       # 10

    def clear_stage(self, *keep, run_time=0.6):
        self.play(*[FadeOut(m) for m in self.mobjects if m not in keep], run_time=run_time)

    # ------------------------------------------------------------------ 1
    def hook(self):
        self.section(1)
        m = RoundedRectangle(width=2.6, height=1.2, corner_radius=0.18, color=MODEL_C, fill_opacity=0.15)
        mt = Text("model", font_size=26, color=MODEL_C).move_to(m)
        t = Text("text in → text out", font_size=24, color=GREY_A).next_to(m, DOWN, buff=0.4)
        self.at("episode")
        self.play(FadeIn(m), FadeIn(mt), run_time=0.5)
        self.at("text")
        self.play(FadeIn(t), run_time=0.4)
        q = Text("so what is a “tool call”, seen from the model?", font_size=28, color=YELLOW).to_edge(DOWN, buff=0.9)
        self.at("side")
        self.play(FadeIn(q), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 2
    def reads(self):
        self.section(2)
        self.clear_stage()
        h = Text("what the model reads (Qwen3 chat template, our one tool)", font_size=22).to_edge(UP, buff=0.35)
        self.at("reads")
        self.play(FadeIn(h), run_time=0.3)
        p = card("rendered prompt", PROMPT, color=GREY_B, fs=12).move_to([0, 0.15, 0])
        self.at("template")
        self.play(FadeIn(p), run_time=0.8)
        self.at("signatures")
        self.play(Indicate(p[1][1], color=TOOL_C, scale_factor=1.0), run_time=0.6)
        tok = VGroup(Text("without the tool: 58 tokens", font=MONO, font_size=18, color=GREY_B),
                     Text("with the tool: 197 tokens (+139)", font=MONO, font_size=18, color=TOOL_C)).arrange(RIGHT, buff=0.8)
        tok.to_edge(DOWN, buff=0.3)
        self.at("139")
        self.play(FadeIn(tok), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 3
    def writes(self):
        self.section(3)
        self.clear_stage()
        h = Text("what the model writes: 10 raw samples, no server parsing", font_size=24).to_edge(UP, buff=0.45)
        self.at("writes")
        self.play(FadeIn(h), run_time=0.3)
        s = card("sample 0 (raw output)", SAMPLE, color=GREY_B, fs=13)
        c = Text(CALL, font=MONO, font_size=14, color=TOOL_C, line_spacing=0.85)
        VGroup(s, c).arrange(DOWN, aligned_edge=LEFT, buff=0.15).move_to([-2.6, 1.0, 0])
        self.at("sampled")
        self.play(FadeIn(s), run_time=0.5)
        self.at("same")
        self.play(FadeIn(c), run_time=0.4)
        ok = Text("10 / 10 valid calls, all identical", font_size=20, color=GREEN_B).next_to(c, DOWN, buff=0.25).align_to(c, LEFT)
        self.play(FadeIn(ok), run_time=0.3)
        # stacked bars: reasoning (grey) + the call (gold), tokens per sample
        x0, y0, H, W = 2.2, -3.0, 3.9, 4.6
        top = max(SAMPLES)
        bw = W / len(SAMPLES)
        g = VGroup(Line([x0, y0, 0], [x0 + W, y0, 0], color=GREY_D, stroke_width=1))
        for i, v in enumerate(SAMPLES):
            hr = H * (v - CALL_TOKENS) / top
            hc = H * CALL_TOKENS / top
            x = x0 + bw * (i + 0.5)
            g.add(Rectangle(width=bw * 0.6, height=hr, stroke_width=0, fill_color=GREY_C, fill_opacity=0.8).move_to([x, y0 + hr / 2, 0]),
                  Rectangle(width=bw * 0.6, height=hc, stroke_width=0, fill_color=TOOL_C, fill_opacity=1).move_to([x, y0 + hr + hc / 2 + 0.02, 0]))
        lab = VGroup(Text("tokens written per sample", font_size=16, color=GREY_A),
                     VGroup(Square(0.18, fill_color=GREY_C, fill_opacity=0.8, stroke_width=0), Text("reasoning", font_size=15),
                            Square(0.18, fill_color=TOOL_C, fill_opacity=1, stroke_width=0), Text("the call (~25)", font_size=15)).arrange(RIGHT, buff=0.12)
                     ).arrange(DOWN, buff=0.1).next_to(g, UP, buff=0.15)
        rng = Text("251–494", font=MONO, font_size=16).next_to(g, DOWN, buff=0.12)
        self.at("25")
        self.play(FadeIn(g, lag_ratio=0.05), FadeIn(lab), run_time=0.9)
        self.at("250")
        self.play(FadeIn(rng), Indicate(VGroup(*[g[1 + 2 * i] for i in range(10)]), color=WHITE, scale_factor=1.0), run_time=0.8)
        note = Text("even with thinking switched off", font_size=20, color=YELLOW).to_edge(DOWN, buff=0.15)
        self.at("switched")
        self.play(FadeIn(note), run_time=0.3)
        self.end_section()

    # ------------------------------------------------------------------ 4
    def parsing(self):
        self.section(4)
        self.clear_stage()
        raw = card("raw text from the model", "...Let me output the tool call.\n</think>\n<tool_call>\n{\"name\": \"shell\", \"arguments\":\n {\"command\": \"python3 test_stats.py\"}}\n</tool_call>", fs=14).move_to([-3.2, 1.6, 0])
        self.at("find")
        self.play(FadeIn(raw), run_time=0.5)
        step = VGroup(Text("1. find the <tool_call> tags", font_size=20), Text("2. parse the JSON inside", font_size=20)).arrange(DOWN, aligned_edge=LEFT, buff=0.2)
        step.move_to([3.4, 1.6, 0])
        self.at("tags")
        self.play(FadeIn(step[0]), run_time=0.3)
        self.at("parses")
        self.play(FadeIn(step[1]), run_time=0.3)
        st = card("structured call for the harness", 'name:      "shell"\narguments: "{\\"command\\": \\"python3 test_stats.py\\"}"\ncall_id:   "call_..."', color=TOOL_C, fs=16).move_to([0, -1.6, 0])
        a1 = Arrow(raw.get_right(), step.get_left(), buff=0.15, color=GREY_B)
        a2 = Arrow(step.get_bottom() + 0.2 * DOWN, st.get_top() + 0.3 * RIGHT, buff=0.15, color=GREY_B)
        self.at("structured")
        self.play(GrowArrow(a1), GrowArrow(a2), FadeIn(st), run_time=0.6)
        self.end_section()

    # ------------------------------------------------------------------ 5
    def back(self):
        self.section(5)
        self.clear_stage()
        h = Text("the result goes back as text too", font_size=24).to_edge(UP, buff=0.5)
        self.at("result")
        self.play(FadeIn(h), run_time=0.3)
        rt = card("what the model reads on its next call (new part)", ROUNDTRIP, fs=14).move_to([-2.3, 0.2, 0])
        self.at("wrapped")
        self.play(FadeIn(rt), run_time=0.6)
        n1 = Text("+43 tokens", font=MONO, font_size=26, color=TOOL_C).move_to([5.0, 0.9, 0])
        n2 = Text("the reasoning is not kept,\nonly the call", font_size=20, color=GREY_A, line_spacing=0.9).move_to([5.0, -0.4, 0])
        self.at("43")
        self.play(FadeIn(n1), run_time=0.3)
        self.at("reasoning")
        self.play(FadeIn(n2), run_time=0.3)
        self.end_section()

    # ------------------------------------------------------------------ 6
    def codex_tool(self):
        self.section(6)
        self.clear_stage()
        h = Text("Codex's main tool: exec_command", font_size=26).to_edge(UP, buff=0.5)
        self.at("exec")
        self.play(FadeIn(h), run_time=0.3)
        ps = VGroup(*[Text(p, font=MONO, font_size=18, color=TOOL_C if i == 0 else WHITE) for i, p in enumerate(PARAMS)])
        ps.arrange_in_grid(5, 2, buff=(0.6, 0.25), col_alignments="ll").move_to([-2.2, 0.5, 0])
        self.at("10")
        self.play(FadeIn(ps, lag_ratio=0.1), run_time=1.6)
        b1 = hbar("exec_command", 421, 0.009, TOOL_C, -2.0, x0=-1.0)
        b2 = hbar("our shell tool", 58, 0.009, TEAL_C, -2.7, x0=-1.0)
        cap = Text("tokens in the tool definition", font_size=16, color=GREY_A).next_to(b1, UP, buff=0.15).align_to(b1[1], LEFT)
        self.at("421")
        self.play(FadeIn(cap), FadeIn(b1), run_time=0.4)
        self.at("hours")
        self.play(FadeIn(b2), run_time=0.3)
        self.end_section()

    # ------------------------------------------------------------------ 7
    def codex_result(self):
        self.section(7)
        self.clear_stage()
        c1 = card("the model's call, as Codex received it", CODEX_CALL, color=TOOL_C, fs=15).move_to([0, 1.9, 0])
        c2 = card("the result Codex sends back", CODEX_RESULT, color=HARNESS_C, fs=15).move_to([0, -0.9, 0])
        self.at("result")
        self.play(FadeIn(c1), run_time=0.4)
        self.at("id")
        self.play(FadeIn(c2), run_time=0.5)
        self.at("header")
        self.play(Indicate(c2[1][1], color=HARNESS_C, scale_factor=1.0), run_time=0.8)
        self.end_section()

    # ------------------------------------------------------------------ 8
    def wrong(self):
        self.section(8)
        self.clear_stage()
        bad = card("a call with broken JSON", '{"name": "exec_command", "arguments": "{\\"cmd\\": \\"ls}"}', color=RED_B, fs=16).move_to([0, 2.0, 0])
        self.at("valid")
        self.play(FadeIn(bad), run_time=0.4)
        src = card("Codex · tools/handlers/mod.rs", 'serde_json::from_str(arguments).map_err(|err| {\n    FunctionCallError::RespondToModel(format!(\n        "failed to parse function arguments: {err}"))', color=GREY_B, fs=15).move_to([0, 0.1, 0])
        self.at("crash")
        self.play(FadeIn(src), run_time=0.5)
        res = Text("tool result: “failed to parse function arguments: …” (the parser's error)", font=MONO, font_size=16, color=YELLOW).move_to([0, -1.7, 0])
        self.at("failed")
        self.play(FadeIn(res), run_time=0.4)
        again = Text("the model reads the error and can try again", font_size=22, color=GREEN_B).move_to([0, -2.6, 0])
        self.at("again")
        self.play(FadeIn(again), run_time=0.3)
        self.end_section()

    # ------------------------------------------------------------------ 9
    def code(self):
        self.section(9)
        self.clear_stage()
        code, hl = code_panel(CODE, font_size=22)
        code.move_to([0, 0.2, 0])
        self.play(FadeIn(code, shift=0.2 * UP), run_time=0.5)
        hl.match_y(code.line_numbers[1])
        self.at("search")
        self.play(Create(hl), run_time=0.3)
        self.at("json")
        self.play(highlight(hl, code, 5), run_time=0.3)
        self.at("error")
        self.play(highlight(hl, code, 7), run_time=0.3)
        self.end_section()

    # ------------------------------------------------------------------ 10
    def outro(self):
        self.section(10)
        self.clear_stage(run_time=0.4)
        card_ = next_up_card(NEXT)
        self.at("next")
        self.play(FadeIn(card_, shift=0.2 * UP), run_time=0.6)
        self.end_section()
        finish(self, hold=1.5)
