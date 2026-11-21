"""How AI Agents Work: Inside the Harness, episode 9 — Instructions and Memory.

Render from the repo root:  ./render.sh agents a09
Runs: code/a09_instructions_and_memory/instructions_harness.py (2026-11-20/21, Qwen3 4B on Ollama; logs in
run_2026-11-21.txt). Codex request: codex_request_2026-11-21.json (Codex CLI 0.163.0 through the logging proxy).
Codex: openai/codex @ 89461ea (core/src/agents_md.rs, config/src/config_toml.rs, hooks/src/lib.rs, core/src/hook_runtime.rs).
"""
from manim import *

from common import MONO, code_panel, finish, highlight, next_up_card
from intro import play_token_intro
from a09_script import NEXT, TAGLINE, TITLE
from voiced_scene import VoicedScene

SERIES_LABEL = "HOW AI AGENTS WORK  ·  EPISODE 9"
MODEL_C, HARNESS_C, TOOL_C = PURPLE_B, TEAL_C, GOLD
AGENTS_MD = """# Notes for coding agents

- Run the tests with `python3 run_tests.py`, never by calling a test
  file directly: it also checks formatting.
- Every fix gets one line in CHANGELOG.md, under "Unreleased",
  in the form `- Fixed: <what was wrong>`.
- Keep the change as small as possible."""
REQUEST = [("developer", "Codex's base instructions", "16,979 chars", GREY_B),
           ("developer", "<skills_instructions> …", "2,316 chars", GREY_B),
           ("user", "# AGENTS.md instructions for /work\n<INSTRUCTIONS> …the 3 rules… </INSTRUCTIONS>\n<environment_context> …", "742 chars", HARNESS_C),
           ("user", "What command runs the tests in this project? Answer in one line.", "64 chars", YELLOW)]
# (setup, runs, fixed, used run_tests.py, CHANGELOG line)
TABLE = [("no AGENTS.md", "1", "1", "–", "–"),
         ("in the folder only", "1", "0", "0", "0"),
         ("sent the Codex way", "5", "3*", "0", "3 (at the end of the file)"),
         ("system instructions", "2", "1", "0", "0")]
CODE = """def agents_md_message(folder, text):
    # Codex: codex-rs/core/src/agents_md.rs
    return {"role": "user",
            "content": f"# AGENTS.md instructions for {folder}\\n\\n"
                       f"<INSTRUCTIONS>\\n{text}\\n</INSTRUCTIONS>"}

rules = (work / "AGENTS.md").read_text()
history = [agents_md_message(work, rules),
           {"role": "user", "content": TASK}]"""


def card(title, body, color=GREY_B, fs=17, title_fs=17):
    t = Text(body, font=MONO, font_size=fs, line_spacing=0.85)
    hd = Text(title, font_size=title_fs, color=color)
    g = VGroup(hd, t).arrange(DOWN, aligned_edge=LEFT, buff=0.15)
    r = RoundedRectangle(width=g.width + 0.45, height=g.height + 0.35, corner_radius=0.12, color=color, fill_opacity=0.07)
    return VGroup(r, g.move_to(r))


class InstructionsMemoryVideo(VoicedScene):
    VIDEO = "a09"

    def construct(self):
        play_token_intro(self, TITLE, 9, TAGLINE, label=SERIES_LABEL)
        self.hook()        # 1
        self.loading()     # 2
        self.request()     # 3
        self.rules()       # 4
        self.setups()      # 5
        self.results()     # 6
        self.knowing()     # 7
        self.hooks()       # 8
        self.code()        # 9
        self.outro()       # 10

    def clear_stage(self, *keep, run_time=0.6):
        self.play(*[FadeOut(m) for m in self.mobjects if m not in keep], run_time=run_time)

    # ------------------------------------------------------------------ 1
    def hook(self):
        self.section(1)
        items = VGroup(*[Text(t, font_size=26) for t in ("how to run the tests", "what goes in the changelog", "what not to touch")])
        items.arrange(DOWN, aligned_edge=LEFT, buff=0.35).move_to([-2.5, 1.0, 0])
        self.at("tests")
        self.play(FadeIn(items[0]), run_time=0.3)
        self.at("change", "changelog")
        self.play(FadeIn(items[1]), run_time=0.3)
        self.at("touch")
        self.play(FadeIn(items[2]), run_time=0.3)
        f = Text("AGENTS.md", font=MONO, font_size=40, color=HARNESS_C).move_to([0, -1.6, 0])
        self.at("file")
        self.play(FadeIn(f, shift=0.2 * UP), run_time=0.5)
        self.end_section()

    # ------------------------------------------------------------------ 2
    def loading(self):
        self.section(2)
        self.clear_stage()
        h = Text("how Codex finds it (core/src/agents_md.rs)", font_size=28).to_edge(UP, buff=0.5)
        self.play(FadeIn(h), run_time=0.3)
        lines = [("repo/", 0, "← project root (.git)", WHITE), ("AGENTS.md", 2, "", HARNESS_C), ("service/", 2, "", WHITE),
                 ("AGENTS.md", 4, "", HARNESS_C), ("api/", 4, "← working folder", WHITE)]
        cw = Text("x" * 20, font=MONO, font_size=22).width / 20
        tree = VGroup()
        for name, ind, note, col in lines:
            t = Text(name, font=MONO, font_size=22, color=col)
            row = VGroup(t)
            if note:
                row.add(Text(note, font_size=20, color=GREY_B).next_to(t, RIGHT, buff=0.4))
            tree.add(row)
        tree.arrange(DOWN, aligned_edge=LEFT, buff=0.16)
        for row, (_, ind, _, _) in zip(tree, lines):
            row.shift(RIGHT * cw * ind)
        tree.move_to([-2.2, 1.0, 0])
        self.at("root")
        self.play(FadeIn(tree), run_time=0.5)
        notes = VGroup(Text("root → working folder, joined in order", font_size=22),
                       Text("at most 32 KiB (project_doc_max_bytes)", font_size=22),
                       Text("skipped for untrusted projects", font_size=22, color=RED_B)).arrange(DOWN, aligned_edge=LEFT, buff=0.35)
        notes.move_to([0, -1.9, 0])
        self.at("joins")
        self.play(FadeIn(notes[0]), run_time=0.3)
        self.at("thirty", "32")
        self.play(FadeIn(notes[1]), run_time=0.3)
        self.at("skips")
        self.play(FadeIn(notes[2]), run_time=0.3)
        self.end_section()

    # ------------------------------------------------------------------ 3
    def request(self):
        self.section(3)
        self.clear_stage()
        h = Text("a Codex request, captured (input items, in order)", font_size=26).to_edge(UP, buff=0.45)
        self.play(FadeIn(h), run_time=0.3)
        rows = VGroup()
        for role, text, size, c in REQUEST:
            t = Text(text, font=MONO, font_size=16, color=c, line_spacing=0.85)
            r = VGroup(Text(role, font=MONO, font_size=17, color=GREY_A), t, Text(size, font_size=16, color=GREY_B))
            rows.add(r)
        for r in rows:
            r[1].next_to(r[0], RIGHT, buff=0.4)
            r[2].next_to(r[1], RIGHT, buff=0.4)
        rows.arrange(DOWN, aligned_edge=LEFT, buff=0.4).move_to([0, -0.1, 0])
        for r in rows:
            r[1].align_to(rows.get_left() + RIGHT * 1.7, LEFT)
            r[2].next_to(r[1], RIGHT, buff=0.4)
        if rows.width > 13:
            rows.scale_to_fit_width(13)
        self.at("user")
        self.play(FadeIn(rows[2]), run_time=0.4)
        self.at("instructions")
        self.play(FadeIn(rows[0]), run_time=0.3)
        self.at("skills")
        self.play(FadeIn(rows[1]), run_time=0.3)
        self.at("task")
        self.play(FadeIn(rows[3]), run_time=0.3)
        self.end_section()

    # ------------------------------------------------------------------ 4
    def rules(self):
        self.section(4)
        self.clear_stage()
        c = card("demo/AGENTS.md", AGENTS_MD, color=HARNESS_C, fs=20).move_to([0, 0.1, 0])
        self.at("three")
        self.play(FadeIn(c), run_time=0.6)
        self.end_section()

    # ------------------------------------------------------------------ 5
    def setups(self):
        self.section(5)
        self.clear_stage()
        h = Text("the same bug fix, four setups", font_size=28).to_edge(UP, buff=0.5)
        self.play(FadeIn(h), run_time=0.3)
        s = VGroup(*[Text(t, font_size=24) for t in ("1. no AGENTS.md", "2. AGENTS.md in the folder, not sent",
                                                    "3. sent as a user message (the Codex way)", "4. put in the system instructions")])
        s.arrange(DOWN, aligned_edge=LEFT, buff=0.35).move_to([0, -0.1, 0])
        for t, cue in zip(s, ["no", "folder", "codecs", "system"]):
            self.at(cue)
            self.play(FadeIn(t), run_time=0.3)
        self.end_section()

    # ------------------------------------------------------------------ 6
    def results(self):
        self.section(6)
        self.clear_stage()
        heads = ["setup", "runs", "fixed", "used run_tests.py", "CHANGELOG line"]
        xs = [-6.2, -1.9, -0.8, 0.4, 3.1]
        y0 = 2.3
        hdr = VGroup(*[Text(t, font_size=19, color=GREY_A).move_to([x, y0, 0], aligned_edge=LEFT) for t, x in zip(heads, xs)])
        rows = VGroup()
        for i, r in enumerate(TABLE):
            rows.add(VGroup(*[Text(v, font_size=21, color=(RED_B if (j == 3 and v == "0") else WHITE)).move_to([x, y0 - 0.65 * (i + 1), 0], aligned_edge=LEFT)
                              for j, (v, x) in enumerate(zip(r, xs))]))
        foot = Text("* one run crashed in our harness before its final check", font_size=15, color=GREY_B).next_to(rows, DOWN, buff=0.15).align_to(rows, LEFT)
        self.play(FadeIn(hdr), FadeIn(rows), FadeIn(foot), run_time=0.6)
        cost = Text("sending AGENTS.md: +107 tokens on every call (288 vs 181 for the first)", font_size=20, color=GREY_A).move_to([0, -1.35, 0])
        self.at("hundred")
        self.play(FadeIn(cost), run_time=0.3)
        self.at("never")
        self.play(*[Indicate(r[3], color=RED_B, scale_factor=1.0) for r in rows[1:]], run_time=0.8)
        self.at("changelog")
        self.play(Indicate(rows[2][4], color=TOOL_C, scale_factor=1.0), run_time=0.8)
        note = Text("echo '- Fixed: …' >> CHANGELOG.md   → after “## 1.3.0”, not under “## Unreleased”", font=MONO,
                    font_size=17, color=TOOL_C).move_to([0, -2.1, 0])
        self.at("appended")
        self.play(FadeIn(note), run_time=0.3)
        opened = Text("folder only: AGENTS.md was never opened", font_size=20, color=GREY_A).move_to([0, -2.7, 0])
        self.at("folder")
        self.play(FadeIn(opened), run_time=0.3)
        self.end_section()

    # ------------------------------------------------------------------ 7
    def knowing(self):
        self.section(7)
        self.clear_stage()
        q = card("Codex CLI + Qwen3 4B", "What command runs the tests in this project? Answer in one line.", color=YELLOW, fs=20).move_to([0, 1.4, 0])
        self.at("asked")
        self.play(FadeIn(q), run_time=0.4)
        a = card("answer", "Yes, the AGENTS.md says: \"Run the tests with `python3 run_tests.py`, ...\"\n→ `python3 run_tests.py`",
                 color=GREEN_B, fs=19).move_to([0, -0.4, 0])
        self.at("answered")
        self.play(FadeIn(a), run_time=0.4)
        k = Text("knowing the rule ≠ following it", font_size=30, color=YELLOW).to_edge(DOWN, buff=0.8)
        self.at("following")
        self.play(FadeIn(k), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 8
    def hooks(self):
        self.section(8)
        self.clear_stage()
        h = Text("Codex hooks: 14 events (codex-rs/hooks)", font_size=26).to_edge(UP, buff=0.4)
        self.play(FadeIn(h), run_time=0.3)
        ev = VGroup(Text("Stop: can refuse to end the turn, with a reason", font_size=21),
                    Text("PreToolUse: “Command blocked by PreToolUse hook: {reason}”", font_size=21)).arrange(DOWN, aligned_edge=LEFT, buff=0.25)
        ev.move_to([0, 1.7, 0])
        self.at("stop")
        self.play(FadeIn(ev[0]), run_time=0.3)
        self.at("pre")
        self.play(FadeIn(ev[1]), run_time=0.3)
        run = VGroup(Text("our stop check, run 3:", font_size=20, color=GREY_A),
                     Text("Stop blocked: no line was added to CHANGELOG.md under \"Unreleased\"", font=MONO, font_size=17, color=RED_B),
                     Text("$ echo '- Fixed: Incorrect total calculation …' >> CHANGELOG.md", font=MONO, font_size=17)).arrange(DOWN, aligned_edge=LEFT, buff=0.18)
        run.move_to([0, -0.4, 0])
        self.at("fired")
        self.play(FadeIn(run[0]), FadeIn(run[1]), run_time=0.4)
        self.at("added")
        self.play(FadeIn(run[2]), run_time=0.3)
        bug = Text("our check only tested that CHANGELOG.md changed", font_size=22, color=TOOL_C).move_to([0, -1.9, 0])
        self.at("only")
        self.play(FadeIn(bug), run_time=0.3)
        lesson = Text("a check is only as good as what it tests", font_size=26, color=YELLOW).to_edge(DOWN, buff=0.45)
        self.at("good")
        self.play(FadeIn(lesson), run_time=0.3)
        self.end_section()

    # ------------------------------------------------------------------ 9
    def code(self):
        self.section(9)
        self.clear_stage()
        code, hl = code_panel(CODE, font_size=21)
        code.move_to([0, 0.1, 0])
        self.play(FadeIn(code, shift=0.2 * UP), run_time=0.5)
        hl.match_y(code.line_numbers[2])
        self.at("message")
        self.play(Create(hl), run_time=0.3)
        self.at("file")
        self.play(highlight(hl, code, 6), run_time=0.3)
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
