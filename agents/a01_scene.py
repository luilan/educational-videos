"""How AI Agents Work: Inside the Harness, episode 1 — What a Harness Is.

Render from the repo root:  ./render.sh agents a01
Numbers and quoted outputs come from runs on 2026-10-19: code/a01_what_is_a_harness (no_harness.py, tiny_harness.py)
and Codex CLI 0.163.0 traced through a logging proxy, all with Qwen3 4B on Ollama 0.40.6 (CPU, in containers without
internet). Token counts of Codex's first request: Qwen tokenizer on the captured request; "seen" = Ollama's count.
"""
from manim import *

from common import MONO, code_panel, finish, highlight, next_up_card
from intro import play_token_intro
from a01_script import NEXT, TAGLINE, TITLE
from voiced_scene import VoicedScene

SERIES_LABEL = "HOW AI AGENTS WORK  ·  EPISODE 1"
MODEL_C, HARNESS_C, TOOL_C = PURPLE_B, TEAL_C, GOLD
STATS_PY = '''def mean(xs):
    """Average of a non-empty list of numbers."""
    return sum(xs) / (len(xs) - 1)'''
CODE = """while True:
    reply = model(history, tools)
    calls = tool_calls(reply)
    if not calls:
        break              # plain words: done
    for call in calls:
        history.append(run(call))"""
# first Codex request, tokens (Qwen tokenizer): instructions, skills, environment, task, nine tool definitions
PARTS = [("instructions", 3553, BLUE_C), ("skills", 522, PURPLE_B), ("tool definitions", 4079, GOLD), ("env + task", 128, GREEN_C)]


def box(label, color, w=2.6, h=1.2, fs=26):
    r = RoundedRectangle(width=w, height=h, corner_radius=0.18, color=color, fill_opacity=0.15)
    return VGroup(r, Text(label, font_size=fs, color=color).move_to(r))


def card(title, body, color=GREY_B, fs=18, w=None):
    t = Text(body, font=MONO, font_size=fs, line_spacing=0.9)
    hd = Text(title, font_size=16, color=color)
    g = VGroup(hd, t).arrange(DOWN, aligned_edge=LEFT, buff=0.15)
    r = RoundedRectangle(width=w or g.width + 0.5, height=g.height + 0.4, corner_radius=0.12, color=color, fill_opacity=0.08)
    return VGroup(r, g.move_to(r))


class HarnessVideo(VoicedScene):
    VIDEO = "a01"

    def construct(self):
        play_token_intro(self, TITLE, 1, TAGLINE, label=SERIES_LABEL)
        self.hook()        # 1
        self.alone()       # 2
        self.trick()       # 3
        self.tiny()        # 4
        self.real()        # 5
        self.roles()       # 6
        self.warning()     # 7
        self.code()        # 8
        self.outro()       # 9

    def clear_stage(self, *keep, run_time=0.6):
        self.play(*[FadeOut(m) for m in self.mobjects if m not in keep], run_time=run_time)

    # ------------------------------------------------------------------ 1. hook
    def hook(self):
        self.section(1)
        m = box("model", MODEL_C).move_to([0, 0.6, 0])
        tin = Text("text in", font_size=22, color=GREY_A).next_to(m, LEFT, buff=1.2)
        tout = Text("text out", font_size=22, color=GREY_A).next_to(m, RIGHT, buff=1.2)
        a1 = Arrow(tin.get_right(), m.get_left(), buff=0.15, color=GREY_B)
        a2 = Arrow(m.get_right(), tout.get_left(), buff=0.15, color=GREY_B)
        self.at("language")
        self.play(FadeIn(m), run_time=0.5)
        cants = VGroup(*[Text(f"✗ {t}", font_size=22, color=RED_B) for t in ("open a file", "run a command", "check its work")]).arrange(RIGHT, buff=0.7)
        cants.move_to([0, -1.0, 0])
        self.at("open")
        self.play(FadeIn(cants, lag_ratio=0.3), run_time=1.2)
        self.at("reads")
        self.play(FadeIn(tin), GrowArrow(a1), GrowArrow(a2), FadeIn(tout), run_time=0.6)
        h = RoundedRectangle(width=10.5, height=4.6, corner_radius=0.3, color=HARNESS_C, stroke_width=3).move_to([0, 0.1, 0])
        hl = Text("the harness", font_size=26, color=HARNESS_C).next_to(h, UP, buff=0.12).align_to(h, LEFT)
        self.at("everything")
        self.play(FadeOut(cants), Create(h), FadeIn(hl), run_time=0.8)
        self.end_section()

    # ------------------------------------------------------------------ 2. the model alone
    def alone(self):
        self.section(2)
        self.clear_stage()
        left = VGroup(card("stats.py", STATS_PY, fs=12), card("python3 test_stats.py", "AssertionError", color=RED_B, fs=14)).arrange(DOWN, buff=0.25, aligned_edge=LEFT)
        left.move_to([-3.75, 0.6, 0])
        self.at("failing")
        self.play(FadeIn(left, lag_ratio=0.2), run_time=0.7)
        m = Text("Qwen3 4B · no tools", font_size=22, color=MODEL_C).move_to([3.5, 2.5, 0])
        self.at("small")
        self.play(FadeIn(m), run_time=0.4)
        reply = card("the model's answer (excerpt)", "Key fix: the bug was in the\n`median` function's even-length\nlist handling ...\nAfter applying this fix, the\ntest passes successfully.", color=MODEL_C, fs=13)
        reply.move_to([3.5, 0.6, 0])
        self.at("invents")
        self.play(FadeIn(reply, shift=0.2 * LEFT), run_time=0.6)
        notes = VGroup(Text("✗ there is no median()", font_size=20, color=RED_B),
                       Text("✗ the test still fails", font_size=20, color=RED_B)).arrange(DOWN, aligned_edge=LEFT, buff=0.12).next_to(reply, DOWN, buff=0.3)
        self.at("isnt")
        self.play(FadeIn(notes[0]), run_time=0.3)
        self.at("passes")
        self.play(FadeIn(notes[1]), run_time=0.3)
        n = Text("it never saw a single file", font_size=26, color=YELLOW).move_to([0, -3.0, 0])
        self.at("never")
        self.play(FadeIn(n), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 3. the trick
    def trick(self):
        self.section(3)
        self.clear_stage()
        m = box("model", MODEL_C).move_to([-3.6, 0.4, 0])
        hb = box("harness", HARNESS_C).move_to([3.6, 0.4, 0])
        tool = card("tool description", 'shell(command)\n"Run a bash command"', color=TOOL_C, fs=15).move_to([-4.9, 2.95, 0])
        self.at("describe")
        self.play(FadeIn(m), FadeIn(hb), FadeIn(tool), run_time=0.6)
        call = Text('{"command": "python3 test_stats.py"}', font=MONO, font_size=17, color=TOOL_C)
        a1 = CurvedArrow(m.get_top() + 0.1 * UP, hb.get_top() + 0.1 * UP, angle=-PI / 4, color=TOOL_C)
        call.next_to(a1, UP, buff=0.1)
        self.at("call")
        self.play(Create(a1), FadeIn(call), run_time=0.7)
        run = Text("runs it", font_size=20, color=HARNESS_C).next_to(hb, DOWN, buff=0.2)
        self.at("runs")
        self.play(FadeIn(run), run_time=0.3)
        res = Text("AssertionError ...", font=MONO, font_size=17, color=GREY_A)
        a2 = CurvedArrow(hb.get_bottom() + 0.5 * DOWN, m.get_bottom() + 0.1 * DOWN, angle=-PI / 4, color=GREY_B)
        res.next_to(a2, DOWN, buff=0.1)
        self.at("pastes")
        self.play(Create(a2), FadeIn(res), run_time=0.7)
        done = Text("…until the model answers in plain words", font_size=24, color=YELLOW).move_to([0, -3.1, 0])
        self.at("until")
        self.play(FadeIn(done), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 4. a tiny harness
    def tiny(self):
        self.section(4)
        self.clear_stage()
        h = Text("tiny_harness.py · 40 lines · one tool: shell", font_size=24).to_edge(UP, buff=0.6)
        self.at("whole")
        self.play(FadeIn(h), run_time=0.4)
        steps = [("1", "$ python3 test_stats.py", "exit 1 · AssertionError", RED_B),
                 ("2", "$ echo 'def mean(nums): return sum(nums) / len(nums)' > stats.py", "exit 0", GREEN_B),
                 ("3", "model answers in words", "done", GREY_A)]
        rows = VGroup()
        for n, cmd, res, col in steps:
            rows.add(VGroup(Text(f"call {n}", font_size=18, color=MODEL_C), Text(cmd, font=MONO, font_size=15),
                            Text(res, font=MONO, font_size=15, color=col)).arrange(RIGHT, buff=0.35))
        rows.arrange(DOWN, aligned_edge=LEFT, buff=0.35).move_to([0, 0.7, 0])
        self.at("three")
        self.play(FadeIn(rows[0]), run_time=0.4)
        self.at("rewrites")
        self.play(FadeIn(rows[1]), run_time=0.4)
        self.play(FadeIn(rows[2]), run_time=0.3)
        chk = Text("harness check: python3 test_stats.py → test passes ✓", font_size=22, color=GREEN_B).move_to([0, -1.2, 0])
        self.at("checks")
        self.play(FadeIn(chk), run_time=0.4)
        note = Text("but it never read stats.py first: the docstring is gone", font_size=22, color=GOLD).move_to([0, -2.2, 0])
        self.at("notice")
        self.play(FadeIn(note), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 5. a real harness
    def real(self):
        self.section(5)
        self.clear_stage()
        h = Text("Codex CLI 0.163 (open source) · the same task", font_size=24).to_edge(UP, buff=0.6)
        self.at("codex")
        self.play(FadeIn(h), run_time=0.4)
        task = card("your task", "The test in test_stats.py fails. Run it ...\n147 characters", color=GREEN_C, fs=16).move_to([0, 2.2, 0])
        self.at("147")
        self.play(FadeIn(task), run_time=0.4)
        total = sum(p[1] for p in PARTS)
        W, x0, y = 11.0, -5.5, -0.2
        segs, labels = VGroup(), VGroup()
        x = x0
        for name, n, col in PARTS:
            w = W * n / total
            r = Rectangle(width=max(w - 0.03, 0.03), height=0.7, stroke_width=0, fill_color=col, fill_opacity=0.85).move_to([x + w / 2, y, 0])
            segs.add(r)
            lab = VGroup(Text(name, font_size=16), Text(f"{n:,}", font=MONO, font_size=16, color=GREY_A)).arrange(DOWN, buff=0.06)
            labels.add(lab.next_to(r, DOWN, buff=0.18 if w > 1.0 else 0.9))   # narrow segments: label lower, no collisions
            x += w
        cap = Text(f"what the model receives: {total:,} tokens", font_size=22, color=YELLOW).next_to(segs, UP, buff=0.35)
        self.at("wrapped")
        self.play(FadeIn(cap), FadeIn(segs, lag_ratio=0.2), run_time=1.0)
        self.play(FadeIn(labels), run_time=0.5)
        tools = Text("exec_command · write_stdin · request_user_input · view_image · multi_agent_v1 ·\nget_goal · create_goal · update_goal · web_search",
                     font=MONO, font_size=15, color=GOLD, line_spacing=0.9).move_to([0, -2.8, 0])
        self.at("nine")
        self.play(FadeIn(tools), run_time=0.5)
        self.end_section()

    # ------------------------------------------------------------------ 6. who does what
    def roles(self):
        self.section(6)
        self.clear_stage()
        items = ["the loop", "the tools", "sandbox and approvals", "instructions", "the context window", "the record of what happened"]
        hcol = VGroup(Text("the harness", font_size=28, color=HARNESS_C),
                      *[Text("· " + t, font_size=22) for t in items]).arrange(DOWN, aligned_edge=LEFT, buff=0.22).move_to([-2.6, 0, 0])
        mcol = VGroup(Text("the model", font_size=28, color=MODEL_C), Text("· choosing the next words", font_size=22)).arrange(DOWN, aligned_edge=LEFT, buff=0.22)
        mcol.move_to([3.4, 1.2, 0])
        self.at("harness")
        self.play(FadeIn(hcol[0]), run_time=0.3)
        self.play(FadeIn(hcol[1:], lag_ratio=0.25), run_time=2.2)
        self.at("model")
        self.play(FadeIn(mcol), run_time=0.5)
        self.end_section()

    # ------------------------------------------------------------------ 7. a warning from the trace
    def warning(self):
        self.section(7)
        self.clear_stage()
        h = Text("what the local model server did with Codex's request", font_size=24).to_edge(UP, buff=0.6)
        self.at("details")
        self.play(FadeIn(h), run_time=0.4)
        W, x0 = 8.4, -5.2                                   # bars share a left edge; length = tokens
        sent = Rectangle(width=W, height=0.5, stroke_width=0, fill_color=GREY_B, fill_opacity=0.7).move_to([x0 + W / 2, 1.5, 0])
        w2 = W * 3779 / 7854
        seen = Rectangle(width=w2, height=0.5, stroke_width=0, fill_color=MODEL_C, fill_opacity=0.9).move_to([x0 + w2 / 2, 0.8, 0])
        ls = Text("sent 7,854 tokens", font=MONO, font_size=18).next_to(sent, RIGHT, buff=0.15)
        lv = Text("seen 3,779 (48%)", font=MONO, font_size=18, color=MODEL_C).next_to(seen, RIGHT, buff=0.15)
        sent_g = VGroup(sent, ls)
        self.at("dropped")
        self.play(FadeIn(sent_g), run_time=0.4)
        self.at("half")
        self.play(GrowFromEdge(seen, LEFT), FadeIn(lv), run_time=0.7)
        why = Text("“developer” messages silently dropped: all of Codex's instructions", font_size=18, color=GREY_A).move_to([0, 0.05, 0])
        self.play(FadeIn(why), run_time=0.3)
        patch = card("with the instructions restored: the reply (plain text)", "*** Begin Patch\n*** Update File: /work/stats.py\n...\n*** End Patch\nI'll now verify the fix by rerunning the test.", color=MODEL_C, fs=14)
        patch.move_to([-1.6, -1.9, 0])
        self.at("restored")
        self.play(FadeIn(patch), run_time=0.5)
        x = Text("no tool call\n→ nothing ran", font_size=24, color=RED_B, line_spacing=0.9).next_to(patch, RIGHT, buff=0.6)
        self.at("nothing")
        self.play(FadeIn(x), run_time=0.3)
        self.end_section()

    # ------------------------------------------------------------------ 8. code
    def code(self):
        self.section(8)
        self.clear_stage()
        code, hl = code_panel(CODE, font_size=24)
        code.move_to([0, 0.3, 0])
        self.play(FadeIn(code, shift=0.2 * UP), run_time=0.5)
        hl.match_y(code.line_numbers[0])
        self.at("loop")
        self.play(Create(hl), run_time=0.3)
        self.at("tool")
        self.play(highlight(hl, code, 5), run_time=0.3)
        self.at("result")
        self.play(highlight(hl, code, 6), run_time=0.3)
        self.end_section()

    # ------------------------------------------------------------------ 9. outro
    def outro(self):
        self.section(9)
        self.clear_stage(run_time=0.4)
        card_ = next_up_card(NEXT)
        self.at("next")
        self.play(FadeIn(card_, shift=0.2 * UP), run_time=0.6)
        self.end_section()
        finish(self, hold=1.5)
