"""How AI Agents Work: Inside the Harness, episode 10 — The Context Budget.

Render from the repo root:  ./render.sh agents a10
Runs: code/a10_context_budget/budget_harness.py (2026-11-21, Qwen3 4B on Ollama; logs in run_2026-11-21.txt; a simulated
4,096-token window). Codex: openai/codex @ 89461ea (models-manager/src/model_info.rs fallback metadata,
protocol/src/openai_models.rs auto_compact_token_limit, prompts/templates/compact/prompt.md).
"""
from manim import *

from common import MONO, code_panel, finish, highlight, next_up_card
from intro import play_token_intro
from a10_script import NEXT, TAGLINE, TITLE
from voiced_scene import VoicedScene

SERIES_LABEL = "HOW AI AGENTS WORK  ·  EPISODE 10"
MODEL_C, HARNESS_C, TOOL_C = PURPLE_B, TEAL_C, GOLD
WINDOW = 4096
PROMPT = """You are performing a CONTEXT CHECKPOINT COMPACTION. Create a handoff
summary for another LLM that will resume the task.
Include:
- Current progress and key decisions made
- Important context, constraints, or user preferences
- What remains to be done (clear next steps)
- Any critical data, examples, or references needed to continue"""
SUMMARY = """# CONTEXT CHECKPOINT COMPACTION HANDOFF SUMMARY
## Current Progress & Key Decisions
- Confirmed answers from the code files:
  - Agents series cover image: `youtube/covers/AgentsCover.jpg`
  - MIN_CHAPTER value: `10`
  - MAX_TITLE_WIDTH value: `12.8`
..."""
CODE = """def compact(history):
    r = respond(history + [{"role": "user", "content": COMPACT_PROMPT}])
    summary = text_of(r).split("</think>")[-1].strip()
    return [{"role": "user", "content": TASK},
            {"role": "user", "content": SUMMARY_PREFIX + summary}]

...
if used > WINDOW * 9 // 10:          # Codex: 90% of the window
    history = compact(history)"""


def card(title, body, color=GREY_B, fs=17, title_fs=17):
    t = Text(body, font=MONO, font_size=fs, line_spacing=0.85)
    hd = Text(title, font_size=title_fs, color=color)
    g = VGroup(hd, t).arrange(DOWN, aligned_edge=LEFT, buff=0.15)
    r = RoundedRectangle(width=g.width + 0.45, height=g.height + 0.35, corner_radius=0.12, color=color, fill_opacity=0.07)
    return VGroup(r, g.move_to(r))


def budget_bars(values, labels, colors, scale=0.00042, x0=-4.6, y0=-2.6, bw=1.1, gap=0.9):
    g = VGroup()
    line_y = y0 + WINDOW * scale
    for i, (v, lab, c) in enumerate(zip(values, labels, colors)):
        b = Rectangle(width=bw, height=v * scale, stroke_width=0, fill_color=c, fill_opacity=0.9)
        b.move_to([x0 + i * (bw + gap), y0 + v * scale / 2, 0])
        g.add(VGroup(b, Text(f"{v:,}", font=MONO, font_size=17).next_to(b, UP, buff=0.08),
                     Text(lab, font_size=16, color=GREY_A).next_to(b, DOWN, buff=0.12)))
    w = DashedLine([x0 - 0.9, line_y, 0], [x0 + len(values) * (bw + gap) - 0.2, line_y, 0], color=RED_B)
    wl = Text("window: 4,096", font_size=17, color=RED_B).next_to(w.get_start(), UP, buff=0.08, aligned_edge=LEFT)
    return g, VGroup(w, wl)


class ContextBudgetVideo(VoicedScene):
    VIDEO = "a10"

    def construct(self):
        play_token_intro(self, TITLE, 10, TAGLINE, label=SERIES_LABEL)
        self.hook()        # 1
        self.codex()       # 2
        self.setup_()      # 3
        self.unlimited()   # 4
        self.compaction()  # 5
        self.compact_runs()  # 6
        self.catch()       # 7
        self.capped()      # 8
        self.lesson()      # 9
        self.code()        # 10
        self.outro()       # 11

    def clear_stage(self, *keep, run_time=0.6):
        self.play(*[FadeOut(m) for m in self.mobjects if m not in keep], run_time=run_time)

    # ------------------------------------------------------------------ 1
    def hook(self):
        self.section(1)
        win = Rectangle(width=8, height=4.2, color=GREY_B).move_to([0, 0.2, 0])
        wl = Text("context window", font_size=20, color=GREY_B).next_to(win, UP, buff=0.1)
        self.at("everything")
        self.play(Create(win), FadeIn(wl), run_time=0.5)
        blocks = VGroup(*[Rectangle(width=7.6, height=h, stroke_width=0, fill_color=c, fill_opacity=0.8)
                          for h, c in [(0.5, HARNESS_C), (0.3, YELLOW), (0.9, TOOL_C), (1.3, TOOL_C), (1.6, TOOL_C)]])
        blocks.arrange(DOWN, buff=0.06).align_to(win.get_top() + DOWN * 0.2, UP)
        labs = ["instructions", "task", "tool result", "tool result", "tool result"]
        tl = VGroup(*[Text(t, font_size=16).move_to(b) for t, b in zip(labs, blocks)])
        self.at("instructions")
        self.play(FadeIn(blocks[0]), FadeIn(tl[0]), run_time=0.3)
        self.at("task")
        self.play(FadeIn(blocks[1]), FadeIn(tl[1]), run_time=0.3)
        self.at("tool")
        self.play(FadeIn(blocks[2:]), FadeIn(tl[2:]), run_time=0.6)
        q = Text("something has to go", font_size=28, color=YELLOW).to_edge(DOWN, buff=0.4)
        self.at("something")
        self.play(FadeIn(q), run_time=0.3)
        self.end_section()

    # ------------------------------------------------------------------ 2
    def codex(self):
        self.section(2)
        self.clear_stage()
        h = Text("Codex, for a model it has no metadata for (fallback)", font_size=26).to_edge(UP, buff=0.5)
        self.play(FadeIn(h), run_time=0.3)
        rows = VGroup(Text("context_window: 272,000 tokens", font=MONO, font_size=24),
                      Text("effective_context_window_percent: 95", font=MONO, font_size=24),
                      Text("auto-compact at 90% of the window", font=MONO, font_size=24, color=HARNESS_C)).arrange(DOWN, aligned_edge=LEFT, buff=0.45)
        rows.move_to([0, -0.2, 0])
        for r, cue in zip(rows, ["272", "95", "compacts"]):
            self.at(cue)
            self.play(FadeIn(r), run_time=0.3)
        self.end_section()

    # ------------------------------------------------------------------ 3
    def setup_(self):
        self.section(3)
        self.clear_stage()
        w = Text("our window: 4,096 tokens (simulated; the server allows 32,768)", font_size=24).to_edge(UP, buff=0.6)
        self.at("smaller")
        self.play(FadeIn(w), run_time=0.4)
        t = card("task", "Answer three questions about this project. Read the files to find out.\n"
                 "1. Which cover image file does the agents series use? (stepstone/build_packs.py)\n"
                 "2. What is the value of MIN_CHAPTER? (youtube/build_manifests.py)\n"
                 "3. What is the value of MAX_TITLE_WIDTH? (framework/intro.py)", color=YELLOW, fs=17).move_to([0, -0.5, 0])
        self.at("three")
        self.play(FadeIn(t), run_time=0.5)
        self.end_section()

    # ------------------------------------------------------------------ 4
    def unlimited(self):
        self.section(4)
        self.clear_stage()
        h = Text("no budget (unlimited mode)", font_size=28).to_edge(UP, buff=0.4)
        self.play(FadeIn(h), run_time=0.3)
        cmds = VGroup(*[Text(f"$ cat {f}   → {n} chars", font=MONO, font_size=18) for f, n in
                        [("stepstone/build_packs.py", "14,859"), ("youtube/build_manifests.py", "13,092"), ("framework/intro.py", "1,800")]])
        cmds.arrange(DOWN, aligned_edge=LEFT, buff=0.15).move_to([0.6, 2.2, 0])
        self.at("printed")
        self.play(FadeIn(cmds), run_time=0.5)
        bars, win = budget_bars([224, 7887], ["call 1", "call 2"], [BLUE_D, RED_C], x0=-1.0)
        self.at("next")
        self.play(FadeIn(bars[0]), Create(win), run_time=0.5)
        self.play(FadeIn(bars[1]), run_time=0.4)
        ok = Text("3 of 3 right: only because the real server had room", font_size=22, color=GREY_A).to_edge(DOWN, buff=0.12)
        self.at("answered")
        self.play(FadeIn(ok), run_time=0.3)
        self.end_section()

    # ------------------------------------------------------------------ 5
    def compaction(self):
        self.section(5)
        self.clear_stage()
        p = card("Codex · prompts/templates/compact/prompt.md", PROMPT, color=HARNESS_C, fs=17).move_to([0, 1.2, 0])
        self.at("summary", "handoff")
        self.play(FadeIn(p), run_time=0.6)
        after = VGroup(Text("then: [task] + [“Another language model started to solve this problem …” + summary]",
                            font_size=20, color=GREY_A))
        after.move_to([0, -1.6, 0])
        self.at("starts")
        self.play(FadeIn(after), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 6
    def compact_runs(self):
        self.section(6)
        self.clear_stage()
        s = card("summary from compact run 2 (Qwen3 4B, excerpt)", SUMMARY, color=GREEN_B, fs=16)
        s.scale_to_fit_width(min(s.width, 8.2)).move_to([-2.2, 0.9, 0])
        self.at("summary")
        self.play(FadeIn(s), run_time=0.5)
        bars, win = budget_bars([224, 545], ["call 1", "call 2"], [BLUE_D, GREEN_C], x0=3.0, bw=0.8, gap=0.6)
        self.at("next")
        self.play(FadeIn(bars), Create(win), run_time=0.5)
        r = Text("3 runs: summaries of ~1,100 characters; next call 545–595 tokens; 3 of 3 right every time",
                 font_size=20, color=GREY_A).to_edge(DOWN, buff=0.35)
        self.at("right")
        self.play(FadeIn(r), run_time=0.3)
        self.end_section()

    # ------------------------------------------------------------------ 7
    def catch(self):
        self.section(7)
        self.clear_stage()
        h = Text("the compaction request itself", font_size=28).to_edge(UP, buff=0.4)
        self.at("look")
        self.play(FadeIn(h), run_time=0.3)
        bars, win = budget_bars([224, 7835, 545], ["call 1", "compaction", "call 2"], [BLUE_D, RED_C, GREEN_C], x0=-2.6)
        self.at("7", "seven")
        self.play(FadeIn(bars), Create(win), run_time=0.6)
        r = Text("to summarize, the model must read it all first", font_size=22, color=YELLOW).move_to([0, 2.5, 0])
        self.at("read")
        self.play(FadeIn(r), run_time=0.3)
        c = Text("Codex: compact at 90% + cap every command's output", font_size=22, color=HARNESS_C).next_to(r, DOWN, buff=0.25)
        self.at("early")
        self.play(FadeIn(c), run_time=0.3)
        self.end_section()

    # ------------------------------------------------------------------ 8
    def capped(self):
        self.section(8)
        self.clear_stage()
        h = Text("capped mode: every tool result cut to 2,000 characters", font_size=26).to_edge(UP, buff=0.4)
        self.at("capped")
        self.play(FadeIn(h), run_time=0.3)
        bars, win = budget_bars([224, 1850], ["call 1", "call 2"], [BLUE_D, TEAL_C], x0=3.9, bw=0.8, gap=0.6)
        self.at("largest")
        self.play(FadeIn(bars), Create(win), run_time=0.5)
        full = Rectangle(width=6.6, height=0.45, stroke_width=0, fill_color=GREY_D, fill_opacity=1).move_to([-2.6, 1.2, 0])
        head = Rectangle(width=0.45, height=0.45, stroke_width=0, fill_color=TEAL_C, fill_opacity=1).align_to(full, LEFT).match_y(full)
        tail = head.copy().align_to(full, RIGHT)
        hit = Rectangle(width=0.08, height=0.45, stroke_width=0, fill_color=RED_C, fill_opacity=1).move_to(full.get_left() + RIGHT * 6.6 * 59 / 301)
        fl = Text("stepstone/build_packs.py: 301 lines", font=MONO, font_size=16).next_to(full, UP, buff=0.1).align_to(full, LEFT)
        hl = Text("line 59: \"cover\": \"youtube/covers/AgentsCover.jpg\"  (cut)", font=MONO, font_size=15, color=RED_B).next_to(full, DOWN, buff=0.15).align_to(full, LEFT)
        self.at("cover")
        self.play(FadeIn(full), FadeIn(head), FadeIn(tail), FadeIn(fl), run_time=0.4)
        self.at("middle")
        self.play(FadeIn(hit), FadeIn(hl), run_time=0.4)
        ans = Text("answer (both runs): cover.jpg  ✗   ·   2 of 3 right", font_size=22, color=RED_B).move_to([-2.2, -1.0, 0])
        self.at("answered")
        self.play(FadeIn(ans), run_time=0.3)
        src = Text("from the docstring at the top: “cover.jpg   series artwork (when available)”", font_size=17, color=GREY_A).next_to(ans, DOWN, buff=0.25)
        if src.get_left()[0] < -6.9:
            src.shift(RIGHT * (-6.9 - src.get_left()[0]))
        self.at("top")
        self.play(FadeIn(src), run_time=0.3)
        self.end_section()

    # ------------------------------------------------------------------ 9
    def lesson(self):
        self.section(9)
        self.clear_stage()
        rows = VGroup(Text("a summary can lose details", font_size=28, color=GREEN_B),
                      Text("a cap cuts the middle", font_size=28, color=TEAL_C),
                      Text("search reads only the lines that matter (episode 4)", font_size=28, color=YELLOW)).arrange(DOWN, buff=0.5)
        for r, cue in zip(rows, ["summary", "cap", "searching"]):
            self.at(cue)
            self.play(FadeIn(r), run_time=0.3)
        self.end_section()

    # ------------------------------------------------------------------ 10
    def code(self):
        self.section(10)
        self.clear_stage()
        code, hl = code_panel(CODE, font_size=20)
        code.move_to([0, 0.1, 0])
        self.play(FadeIn(code, shift=0.2 * UP), run_time=0.5)
        hl.match_y(code.line_numbers[1])
        self.at("one")
        self.play(Create(hl), run_time=0.3)
        self.at("shorter")
        self.play(highlight(hl, code, 3), run_time=0.3)
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
