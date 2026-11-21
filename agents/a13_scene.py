"""How AI Agents Work: Inside the Harness, episode 13 — Evaluating Agents.

Render from the repo root:  ./render.sh agents a13
Results: code/a13_evaluating_agents/eval.py (2026-11-21, Qwen3 4B on Ollama, 4 tasks × 2 attempts; results in
run_2026-11-21.txt and results_2026-11-21.json). Episode 4's numbers from code/a04_reading_and_searching.
"""
from manim import *

from common import MONO, code_panel, finish, highlight, next_up_card
from intro import play_token_intro
from a13_script import NEXT, TAGLINE, TITLE
from voiced_scene import VoicedScene

SERIES_LABEL = "HOW AI AGENTS WORK  ·  EPISODE 13"
MODEL_C, HARNESS_C, TOOL_C = PURPLE_B, TEAL_C, GOLD
# task, attempt 1, attempt 2, mean tokens, mean seconds
RESULTS = [("stats", True, False, 1064, 781), ("fix", False, True, 9404, 2318), ("search", False, False, 3831, 1312),
           ("server", True, True, 633, 592)]
TASKS = [("stats", "ep 2", "fix mean() in stats.py", "tests pass, test file unchanged"),
         ("fix", "ep 5", "fix total() in invoice.py", "tests pass, test file unchanged"),
         ("search", "ep 4", "which line names the figure images?", "answer has study_guide.py and 42"),
         ("server", "ep 6", "start server.py, report the version", "answer has 1.4.2")]
CODE = """def tests_pass_unchanged(test_file):
    def grade(work, answer):
        same = (work / test_file).read_text() == original(test_file)
        ok = subprocess.run(["python3", test_file], cwd=work).returncode == 0
        return ok and same, ("" if same else "test file was modified")
    return grade

def answer_contains(*facts):
    def grade(work, answer):
        missing = [f for f in facts if f not in answer]
        return not missing, ("missing " + ", ".join(missing) if missing else "")
    return grade"""


class EvaluatingAgentsVideo(VoicedScene):
    VIDEO = "a13"

    def construct(self):
        play_token_intro(self, TITLE, 13, TAGLINE, label=SERIES_LABEL)
        self.hook()        # 1
        self.what()        # 2
        self.suite()       # 3
        self.graders()     # 4
        self.results()     # 5
        self.passk()       # 6
        self.variance()    # 7
        self.cost()        # 8
        self.using()       # 9
        self.code()        # 10
        self.outro()       # 11

    def clear_stage(self, *keep, run_time=0.6):
        self.play(*[FadeOut(m) for m in self.mobjects if m not in keep], run_time=run_time)

    # ------------------------------------------------------------------ 1
    def hook(self):
        self.section(1)
        rows = VGroup(Text("ep 4, search mode: line 224, line 224, line 224", font_size=22),
                      Text("ep 5, rewrite: 12 lines left · a file that no longer runs", font_size=22),
                      Text("ep 6, blocking: right by accident · told the user to do it", font_size=22)).arrange(DOWN, aligned_edge=LEFT, buff=0.35)
        rows.move_to([0, 0.8, 0])
        self.at("same")
        self.play(FadeIn(rows, lag_ratio=0.3), run_time=1.0)
        a = Text("one run is an anecdote", font_size=30, color=YELLOW).move_to([0, -1.4, 0])
        self.at("anecdote")
        self.play(FadeIn(a), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 2
    def what(self):
        self.section(2)
        self.clear_stage()
        items = VGroup(*[Text(t, font_size=28, color=c) for t, c in [("a fixed set of tasks", WHITE), ("a grader for each", HARNESS_C),
                                                                      ("several attempts per task", TOOL_C)]]).arrange(DOWN, buff=0.5).move_to([0, 0.5, 0])
        for it, cue in zip(items, ["tasks", "grader", "attempts"]):
            self.at(cue)
            self.play(FadeIn(it), run_time=0.3)
        n = Text("graders check the outcome, not the transcript", font_size=24, color=GREY_A).to_edge(DOWN, buff=0.8)
        self.at("outcome")
        self.play(FadeIn(n), run_time=0.3)
        self.end_section()

    # ------------------------------------------------------------------ 3
    def suite(self):
        self.section(3)
        self.clear_stage()
        h = Text("eval.py: four tasks from earlier episodes", font_size=26).to_edge(UP, buff=0.5)
        self.play(FadeIn(h), run_time=0.3)
        rows = VGroup()
        for name, ep, desc, _ in TASKS:
            rows.add(VGroup(Text(name, font=MONO, font_size=22, color=TOOL_C), Text(ep, font_size=19, color=GREY_B), Text(desc, font_size=21)))
        for r in rows:
            r.arrange(RIGHT, buff=0.4)
        rows.arrange(DOWN, aligned_edge=LEFT, buff=0.35).move_to([0, 0.2, 0])
        for r in rows:
            r[1].align_to(rows.get_left() + RIGHT * 1.6, LEFT)
            r[2].align_to(rows.get_left() + RIGHT * 2.7, LEFT)
        cues = ["average", "invoice", "line", "server"]
        for r, c in zip(rows, cues):
            self.at(c)
            self.play(FadeIn(r), run_time=0.3)
        ag = Text("the agent: episode 2's harness (shell tool, 30 s timeout, 10 model calls)", font_size=20, color=GREY_A).to_edge(DOWN, buff=0.6)
        self.at("harness")
        self.play(FadeIn(ag), run_time=0.3)
        self.end_section()

    # ------------------------------------------------------------------ 4
    def graders(self):
        self.section(4)
        self.clear_stage()
        h = Text("graders", font_size=28).to_edge(UP, buff=0.5)
        self.play(FadeIn(h), run_time=0.3)
        rows = VGroup(*[VGroup(Text(name, font=MONO, font_size=22, color=TOOL_C), Text(g, font_size=22)).arrange(RIGHT, buff=0.5)
                        for name, _, _, g in TASKS]).arrange(DOWN, aligned_edge=LEFT, buff=0.35).move_to([0, 0.4, 0])
        for r in rows:
            r[1].align_to(rows.get_left() + RIGHT * 1.6, LEFT)
        self.at("tests")
        self.play(FadeIn(rows[:2]), run_time=0.4)
        cheat = Text("otherwise “fixing” the test would pass", font_size=22, color=RED_B).move_to([0, -1.6, 0])
        self.at("otherwise")
        self.play(FadeIn(cheat), run_time=0.3)
        self.at("questions")
        self.play(FadeIn(rows[2:]), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 5
    def results(self):
        self.section(5)
        self.clear_stage()
        heads = ["task", "attempt 1", "attempt 2"]
        xs = [-3.6, -0.8, 1.6]
        hdr = VGroup(*[Text(t, font_size=21, color=GREY_A).move_to([x, 2.2, 0]) for t, x in zip(heads, xs)])
        self.play(FadeIn(hdr), run_time=0.3)
        rows = VGroup()
        for i, (name, a1, a2, _, _) in enumerate(RESULTS):
            y = 1.4 - 0.75 * i
            rows.add(VGroup(Text(name, font=MONO, font_size=24).move_to([xs[0], y, 0]),
                            Text("PASS" if a1 else "FAIL", font=MONO, font_size=24, color=GREEN_B if a1 else RED_B).move_to([xs[1], y, 0]),
                            Text("PASS" if a2 else "FAIL", font=MONO, font_size=24, color=GREEN_B if a2 else RED_B).move_to([xs[2], y, 0])))
        order = {"server": 3, "fixes": (0, 1), "search": 2}
        self.at("server")
        self.play(FadeIn(rows[3]), run_time=0.3)
        self.at("fixes")
        self.play(FadeIn(rows[0]), FadeIn(rows[1]), run_time=0.4)
        self.at("search")
        self.play(FadeIn(rows[2]), run_time=0.3)
        tot = Text("overall: 4 of 8 (pass@1 = 0.50)", font_size=26, color=YELLOW).move_to([0, -2.4, 0])
        self.at("overall")
        self.play(FadeIn(tot), run_time=0.3)
        self.end_section()

    # ------------------------------------------------------------------ 6
    def passk(self):
        self.section(6)
        self.clear_stage()
        a = VGroup(Text("pass@1", font=MONO, font_size=30, color=TOOL_C), Text("share of attempts that succeed", font_size=24),
                   Text("0.50", font=MONO, font_size=30)).arrange(RIGHT, buff=0.6).move_to([0, 1.3, 0])
        b = VGroup(Text("pass^k", font=MONO, font_size=30, color=HARNESS_C), Text("did every attempt succeed?", font_size=24),
                   Text("server only", font=MONO, font_size=26)).arrange(RIGHT, buff=0.6).move_to([0, -0.1, 0])
        self.at("one")
        self.play(FadeIn(a), run_time=0.4)
        self.at("power", "every")
        self.play(FadeIn(b), run_time=0.4)
        n = Text("if it has to work every time, pass^k is the number", font_size=24, color=YELLOW).to_edge(DOWN, buff=0.9)
        self.at("matters")
        self.play(FadeIn(n), run_time=0.3)
        self.end_section()

    # ------------------------------------------------------------------ 7
    def variance(self):
        self.section(7)
        self.clear_stage()
        h = Text("the invoice fix, twice", font_size=28).to_edge(UP, buff=0.5)
        self.play(FadeIn(h), run_time=0.3)
        def bar(y, v, c, lab):
            b = Rectangle(width=v * 0.0008, height=0.5, stroke_width=0, fill_color=c, fill_opacity=0.9).move_to([-4.6, y, 0], aligned_edge=LEFT)
            return VGroup(b, Text(lab, font_size=22).next_to(b, UP, buff=0.12, aligned_edge=LEFT))
        b1 = bar(1.0, 7754, RED_C, "attempt 1: FAIL · 4 calls · 7,754 tokens")
        b2 = bar(-0.6, 11053, GREEN_C, "attempt 2: PASS · 6 calls · 11,053 tokens")
        self.at("failed")
        self.play(FadeIn(b1), run_time=0.4)
        self.at("passed")
        self.play(FadeIn(b2), run_time=0.4)
        s = Text("same task, same model, same harness", font_size=24, color=GREY_A).to_edge(DOWN, buff=1.0)
        self.at("same")
        self.play(FadeIn(s), run_time=0.3)
        self.end_section()

    # ------------------------------------------------------------------ 8
    def cost(self):
        self.section(8)
        self.clear_stage()
        c = VGroup(Text("8 attempts", font_size=30), Text("2 h 47 min on our CPU", font_size=30, color=TOOL_C),
                   Text("29,864 tokens sent", font_size=30, color=TOOL_C)).arrange(DOWN, buff=0.45).move_to([0, 0.6, 0])
        self.at("hours")
        self.play(FadeIn(c), run_time=0.5)
        n = Text("2 attempts per task can't tell an improvement from luck", font_size=24, color=YELLOW).to_edge(DOWN, buff=0.9)
        self.at("luck")
        self.play(FadeIn(n), run_time=0.3)
        self.end_section()

    # ------------------------------------------------------------------ 9
    def using(self):
        self.section(9)
        self.clear_stage()
        loop = VGroup(*[Text(t, font_size=26) for t in ("change one thing", "rerun the same suite", "compare")]).arrange(RIGHT, buff=0.9)
        loop.move_to([0, 1.2, 0])
        arrows = VGroup(*[Arrow(loop[i].get_right(), loop[i + 1].get_left(), buff=0.1, color=GREY_B) for i in range(2)])
        self.at("change")
        self.play(FadeIn(loop[0]), run_time=0.3)
        self.at("rerun")
        self.play(GrowArrow(arrows[0]), FadeIn(loop[1]), GrowArrow(arrows[1]), FadeIn(loop[2]), run_time=0.5)
        ex = VGroup(Text("episode 4, search instructions:  0 of 3 right", font_size=24, color=RED_B),
                     Text("+ “read around matches, follow values back”:  2 of 2 right", font_size=24, color=GREEN_B)).arrange(DOWN, aligned_edge=LEFT, buff=0.3)
        ex.move_to([0, -1.0, 0])
        self.at("fours", "four")
        self.play(FadeIn(ex), run_time=0.5)
        self.end_section()

    # ------------------------------------------------------------------ 10
    def code(self):
        self.section(10)
        self.clear_stage()
        code, hl = code_panel(CODE, font_size=19)
        code.move_to([0, 0.1, 0])
        self.play(FadeIn(code, shift=0.2 * UP), run_time=0.5)
        hl.match_y(code.line_numbers[3])
        self.at("tests")
        self.play(Create(hl), run_time=0.3)
        self.at("compare")
        self.play(highlight(hl, code, 2), run_time=0.3)
        self.at("fail")
        self.play(highlight(hl, code, 4), run_time=0.3)
        self.end_section()

    # ------------------------------------------------------------------ 11
    def outro(self):
        self.section(11)
        self.clear_stage(run_time=0.4)
        card_ = next_up_card(NEXT)
        self.at("next")
        self.play(FadeIn(card_, shift=0.2 * UP), run_time=0.6)
        self.end_section()
        finish(self, hold=1.5)
