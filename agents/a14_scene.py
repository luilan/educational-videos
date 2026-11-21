"""How AI Agents Work: Inside the Harness, episode 14 — Putting It Together (series finale).

Render from the repo root:  ./render.sh agents a14
Runs: code/a14_putting_it_together/tiny_agent.py (2026-11-21, Qwen3 4B on Ollama; logs in run_2026-11-21.txt).
Codex first-request sizes: 7,852 tokens (episode 4) and 7,928 (episode 9), Codex CLI 0.163.0 through the logging proxy.
"""
from manim import *

from common import MONO, code_panel, finish, highlight
from intro import play_token_intro
from a14_script import TAGLINE, TITLE
from voiced_scene import VoicedScene

SERIES_LABEL = "HOW AI AGENTS WORK  ·  EPISODE 14"
MODEL_C, HARNESS_C, TOOL_C = PURPLE_B, TEAL_C, GOLD
PARTS = [("the loop + stop conditions", 2), ("tool calls, errors as results", 3), ("search-first + output cap", 4),
         ("apply_patch", 5), ("sessions", 6), ("sandbox (bubblewrap)", 7), ("rules + approvals", 8), ("AGENTS.md", 9),
         ("compaction", 10), ("MCP servers", 11), ("parallel calls", 12)]
RUN1 = [("mcp__tickets__get_ticket {\"id\": \"T-42\"}", "the ticket", GREEN_B),
        ("exec_command: grep -rn 'invoice' .", "exit 0", GREEN_B),
        ("exec_command: cat invoice.py", "exit 0", GREEN_B),
        ("exec_command: patch -p1 < <(echo \"*** Begin Patch …\")", "exit 2", RED_B),
        ("(last reply: reasoning about line numbers)", "", GREY_A)]
CODE = """def run(self, task):
    history = self.history_start(task)          # AGENTS.md + the task
    for step in range(1, MAX_STEPS + 1):
        r = respond(history, self.tools, INSTRUCTIONS)
        calls = [o for o in r["output"] if o["type"] == "function_call"]
        history += r["output"]
        if not calls:                           # no tool call: done
            return final_text(r)
        with ThreadPoolExecutor() as pool:      # parallel calls
            outs = list(pool.map(self.call_tool, calls))
        for c, out in zip(calls, outs):
            history.append({"type": "function_call_output",
                            "call_id": c["call_id"], "output": out})
        if r["usage"]["input_tokens"] > WINDOW * 9 // 10:
            history = self.compact(history, task)"""


class PuttingItTogetherVideo(VoicedScene):
    VIDEO = "a14"

    def construct(self):
        play_token_intro(self, TITLE, 14, TAGLINE, label=SERIES_LABEL)
        self.hook()        # 1
        self.parts()       # 2
        self.task()        # 3
        self.first()       # 4
        self.run1()        # 5
        self.run2()        # 6
        self.did()         # 7
        self.pattern()     # 8
        self.code()        # 9
        self.end()         # 10

    def clear_stage(self, *keep, run_time=0.6):
        self.play(*[FadeOut(m) for m in self.mobjects if m not in keep], run_time=run_time)

    # ------------------------------------------------------------------ 1
    def hook(self):
        self.section(1)
        f = Text("tiny_agent.py", font=MONO, font_size=44, color=HARNESS_C).move_to([0, 0.6, 0])
        s = Text("293 lines · Python standard library only", font_size=26, color=GREY_A).next_to(f, DOWN, buff=0.5)
        self.at("one", "file")
        self.play(FadeIn(f, shift=0.2 * UP), run_time=0.5)
        self.at("three", "300")
        self.play(FadeIn(s), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 2
    def parts(self):
        self.section(2)
        self.clear_stage()
        rows = VGroup(*[VGroup(Text(f"ep {n}", font_size=18, color=GREY_B), Text(t, font_size=22)).arrange(RIGHT, buff=0.3)
                        for t, n in PARTS])
        rows.arrange_in_grid(6, 2, buff=(1.0, 0.3), col_alignments="ll", flow_order="dr").move_to([0, 0, 0])
        cues = ["loop", "tool", "search", "patches", "sessions", "sandbox", "rules", "agents", "compaction", "mcp", "parallel"]
        for r, c in zip(rows, cues):
            self.at(c)
            self.play(FadeIn(r), run_time=0.25)
        self.end_section()

    # ------------------------------------------------------------------ 3
    def task(self):
        self.section(3)
        self.clear_stage()
        t = Text("“Ticket T-42 describes a bug in this project. Fix it.”", font_size=28, color=YELLOW).move_to([0, 1.8, 0])
        self.at("task")
        self.play(FadeIn(t), run_time=0.4)
        items = VGroup(Text("the ticket: in an MCP server (tickets_mcp.py)", font_size=23),
                       Text("AGENTS.md: run_tests.py, a CHANGELOG line, a small change", font_size=23),
                       Text("every command: inside the bubblewrap sandbox", font_size=23)).arrange(DOWN, aligned_edge=LEFT, buff=0.35)
        items.move_to([0, -0.4, 0])
        for it, c in zip(items, ["mcp", "changelog", "sandbox"]):
            self.at(c)
            self.play(FadeIn(it), run_time=0.3)
        self.end_section()

    # ------------------------------------------------------------------ 4
    def first(self):
        self.section(4)
        self.clear_stage()
        h = Text("the first request, before the model has done anything", font_size=26).to_edge(UP, buff=0.6)
        self.play(FadeIn(h), run_time=0.3)
        def bar(y, v, c, lab):
            b = Rectangle(width=v * 0.0011, height=0.6, stroke_width=0, fill_color=c, fill_opacity=0.9).move_to([-5.0, y, 0], aligned_edge=LEFT)
            return VGroup(b, Text(lab, font_size=22).next_to(b, UP, buff=0.12, aligned_edge=LEFT))
        a = bar(0.6, 606, HARNESS_C, "tiny_agent: 606 tokens")
        b = bar(-1.0, 7852, TOOL_C, "Codex CLI: 7,852 tokens (episode 4), 7,928 (episode 9)")
        self.at("six", "600")
        self.play(FadeIn(a), run_time=0.4)
        self.at("eight", "8")
        self.play(FadeIn(b), run_time=0.5)
        self.end_section()

    # ------------------------------------------------------------------ 5
    def run1(self):
        self.section(5)
        self.clear_stage()
        h = Text("run 1", font_size=28).to_edge(UP, buff=0.5)
        self.play(FadeIn(h), run_time=0.3)
        rows = VGroup(*[VGroup(Text(c, font=MONO, font_size=21), Text(o, font=MONO, font_size=19, color=col)).arrange(RIGHT, buff=0.4)
                        for c, o, col in RUN1]).arrange(DOWN, aligned_edge=LEFT, buff=0.28).move_to([0, 0.3, 0])
        if rows.width > 13:
            rows.scale_to_fit_width(13)
        self.at("ticket")
        self.play(FadeIn(rows[0]), run_time=0.3)
        self.at("searched")
        self.play(FadeIn(rows[1]), run_time=0.3)
        self.at("read")
        self.play(FadeIn(rows[2]), run_time=0.3)
        self.at("wrapped")
        self.play(FadeIn(rows[3]), run_time=0.4)
        n = Text("not “apply_patch”, so the harness never intercepted it", font_size=21, color=TOOL_C).to_edge(DOWN, buff=1.1)
        self.at("recognize")
        self.play(FadeIn(n), run_time=0.3)
        self.at("spent")
        self.play(FadeIn(rows[4]), run_time=0.3)
        self.end_section()

    # ------------------------------------------------------------------ 6
    def run2(self):
        self.section(6)
        self.clear_stage()
        h = Text("runs 2 and 3", font_size=28).to_edge(UP, buff=0.5)
        self.play(FadeIn(h), run_time=0.3)
        rows = VGroup(Text("mcp__tickets__get_ticket {\"id\": \"T-42\"}", font=MONO, font_size=19),
                      Text("exec_command: grep -rn 'invoice' .", font=MONO, font_size=19),
                      Text("(an empty reply)", font=MONO, font_size=19, color=GREY_A)).arrange(DOWN, aligned_edge=LEFT, buff=0.3).move_to([0, 0.8, 0])
        self.at("ticket")
        self.play(FadeIn(rows[:2]), run_time=0.4)
        self.at("nothing")
        self.play(FadeIn(rows[2]), run_time=0.3)
        res = VGroup(Text("all three runs: tests still failing · no CHANGELOG line", font_size=24, color=RED_B),
                     Text("nothing outside the project touched", font_size=24, color=GREEN_B)).arrange(DOWN, buff=0.3).move_to([0, -1.5, 0])
        self.at("all")
        self.play(FadeIn(res[0]), run_time=0.3)
        self.at("outside")
        self.play(FadeIn(res[1]), run_time=0.3)
        self.end_section()

    # ------------------------------------------------------------------ 7
    def did(self):
        self.section(7)
        self.clear_stage()
        rows = VGroup(*[Text(t, font_size=26, color=GREEN_B) for t in ("✓ the ticket arrived through MCP", "✓ the search found the file", "✓ the sandbox held")])
        rows.arrange(DOWN, aligned_edge=LEFT, buff=0.35).move_to([0, 1.2, 0])
        for r, c in zip(rows, ["mcp", "search", "sandbox"]):
            self.at(c)
            self.play(FadeIn(r), run_time=0.3)
        a = Text("a harness can make an agent safe, cheap and checkable", font_size=26).move_to([0, -0.9, 0])
        b = Text("it cannot make a 4-billion-parameter model careful", font_size=26, color=YELLOW).next_to(a, DOWN, buff=0.35)
        self.at("safe")
        self.play(FadeIn(a), run_time=0.4)
        self.at("cannot")
        self.play(FadeIn(b), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 8
    def pattern(self):
        self.section(8)
        self.clear_stage()
        rows = VGroup(*[VGroup(Text(a, font_size=22), Text(b, font_size=20, color=HARNESS_C)).arrange(RIGHT, buff=0.5) for a, b in
                        [("guessed line numbers", "ep 4: grep -n gives real ones"), ("rewrote files it had never read", "ep 5: patches refuse unseen context"),
                         ("wrote tool calls as plain text", "ep 5: nothing ran"), ("knew rules it did not follow", "ep 9: checks and hooks")]])
        rows.arrange(DOWN, aligned_edge=LEFT, buff=0.4).move_to([0, 0.2, 0])
        for r in rows:
            r[1].align_to(rows.get_left() + RIGHT * 5.6, LEFT)
        for r, c in zip(rows, ["guest", "rewrote", "wrote", "new"]):
            self.at(c)
            self.play(FadeIn(r), run_time=0.3)
        self.end_section()

    # ------------------------------------------------------------------ 9
    def code(self):
        self.section(9)
        self.clear_stage()
        code, hl = code_panel(CODE, font_size=18)
        code.move_to([0, 0.1, 0])
        self.play(FadeIn(code, shift=0.2 * UP), run_time=0.5)
        hl.match_y(code.line_numbers[3])
        self.at("send")
        self.play(Create(hl), run_time=0.3)
        self.at("run")
        self.play(highlight(hl, code, 9), run_time=0.3)
        self.at("append")
        self.play(highlight(hl, code, 11), run_time=0.3)
        self.at("stop")
        self.play(highlight(hl, code, 6), run_time=0.3)
        self.end_section()

    # ------------------------------------------------------------------ 10
    def end(self):
        self.section(10)
        self.clear_stage(run_time=0.4)
        e = Text("How AI Agents Work: Inside the Harness · 14 episodes", font_size=30).move_to([0, 1.6, 0])
        self.at("end")
        self.play(FadeIn(e), run_time=0.4)
        parts = VGroup(*[Text(t, font_size=21, color=c) for t, c in (("loop", MODEL_C), ("tools", TOOL_C), ("search & edit", BLUE_B),
                                                                       ("commands", GREEN_B), ("sandbox & approvals", RED_B),
                                                                       ("context", HARNESS_C), ("evals", YELLOW))])
        parts.arrange(RIGHT, buff=0.4).move_to([0, 0.5, 0])
        if parts.width > 13:
            parts.scale_to_fit_width(13)
        self.play(LaggedStart(*[FadeIn(p) for p in parts], lag_ratio=0.2), run_time=1.4)
        c = Text("every number from code you can run, with a model small enough for a laptop", font_size=22, color=GREY_A).move_to([0, -0.6, 0])
        self.at("number")
        self.play(FadeIn(c), run_time=0.4)
        t = Text("Thanks for watching", font_size=40).move_to([0, -1.9, 0])
        self.at("thanks")
        self.play(FadeIn(t), run_time=0.5)
        self.end_section()
        finish(self, hold=2.0)
