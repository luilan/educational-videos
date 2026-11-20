"""How AI Agents Work: Inside the Harness, episode 4 — Reading and Searching Code.

Render from the repo root:  ./render.sh agents a04
Every command, token count and answer is from code/a04_reading_and_searching/search_harness.py (2026-11-20, Qwen3 4B on
Ollama, 3 runs per mode; logs in run_2026-11-20.txt). Project size: count_tokens.py (Qwen3 tokenizer).
Codex: openai/codex @ 89461ea (protocol/src/prompts/base_instructions/default.md, utils/string/src/truncate.rs,
core/src/unified_exec/mod.rs).
"""
from manim import *

from common import MONO, SERIF, code_panel, finish, highlight, next_up_card
from intro import play_token_intro
from a04_script import NEXT, TAGLINE, TITLE
from voiced_scene import VoicedScene

SERIES_LABEL = "HOW AI AGENTS WORK  ·  EPISODE 4"
MODEL_C, HARNESS_C, TOOL_C = PURPLE_B, TEAL_C, GOLD
FILES = [("framework/answer_keys.py", 2430), ("framework/common.py", 815), ("framework/intro.py", 540),
         ("framework/intro_styles.py", 1783), ("framework/review_mid.py", 334), ("framework/review_sheet.py", 348),
         ("framework/study_guide.py", 3944), ("framework/voice.py", 454), ("framework/voiced_scene.py", 407),
         ("stepstone/build_packs.py", 3664), ("stepstone/publish.py", 992), ("youtube/build_manifests.py", 3361)]
NAIVE_CMDS = [("ls", "28 chars"), ("ls stepstone", "26 chars"), ("cat stepstone/publish.py", "4,046 chars"),
              ("cat stepstone/build_packs.py", "14,859 chars")]
NAIVE_SENT = [191, 228, 268, 1295, 4996]          # tokens sent per model call, naive run 1 (run 3 identical)
SEARCH_SENT = [239, 741, 819]                     # search run 1
GREP_OUT = """$ grep -rn "figure" .
stepstone/build_packs.py:88:def figures_in(text, figs):
stepstone/build_packs.py:208:  for f in c["figures"]:
framework/study_guide.py:41:  for fig in c.get("figures", []):
framework/study_guide.py:106:figure { margin: 6pt auto; ...
framework/study_guide.py:222:def figure_html(fig):
framework/study_guide.py:224:  return (f'<figure{cls}><img ...
... 15 matching lines, 1,504 chars"""
CTX_CMDS = [("grep -rn 'study-guide'", "223 chars"), ("grep -rn -C3 'figure'", "5,343 chars, 2,024 shown"),
            ("grep -rn -C3 'fig\\[\"file\"\\]'", "1,056 chars")]
CODEX_CMDS = [("ls /work", 7852), ("ls /work/stepstone", 9999), ("cat /work/stepstone/build_packs.py", 10572)]
CODEX_RULE = ("When searching for text or files, prefer using `rg` or `rg --files`\n"
              "respectively because `rg` is much faster than alternatives like `grep`.\n"
              "(If the `rg` command is not found, then use alternatives.)")
CODE = """LIMIT = 2000   # characters kept per tool result

def truncate_middle(text, limit=LIMIT):
    if len(text) <= limit:
        return text
    head = text[:limit // 2]
    tail = text[-(limit - limit // 2):]
    cut = len(text) - limit
    return f"{head}\\n…{cut} chars truncated…\\n{tail}\""""


def card(title, body, color=GREY_B, fs=14, title_fs=16, mono=True):
    t = Text(body, font=MONO if mono else SERIF, font_size=fs, line_spacing=0.85)
    hd = Text(title, font_size=title_fs, color=color)
    g = VGroup(hd, t).arrange(DOWN, aligned_edge=LEFT, buff=0.15)
    r = RoundedRectangle(width=g.width + 0.45, height=g.height + 0.35, corner_radius=0.12, color=color, fill_opacity=0.07)
    return VGroup(r, g.move_to(r))


def bars(values, color, scale, x0, y0, bw=0.42, gap=0.14, label=True):
    g = VGroup()
    for i, v in enumerate(values):
        h = max(v * scale, 0.04)
        b = Rectangle(width=bw, height=h, stroke_width=0, fill_color=color, fill_opacity=0.9)
        b.move_to([x0 + i * (bw + gap), y0 + h / 2, 0])
        g.add(b)
        if label:
            g.add(Text(f"{v:,}", font=MONO, font_size=13).next_to(b, UP, buff=0.06))
    return g


class ReadingSearchingVideo(VoicedScene):
    VIDEO = "a04"

    def construct(self):
        play_token_intro(self, TITLE, 4, TAGLINE, label=SERIES_LABEL)
        self.hook()        # 1
        self.naive()       # 2
        self.invented()    # 3
        self.search()      # 4
        self.results()     # 5
        self.near_miss()   # 6
        self.context()     # 7
        self.budget()      # 8
        self.codex()       # 9
        self.code()        # 10
        self.outro()       # 11

    def clear_stage(self, *keep, run_time=0.6):
        self.play(*[FadeOut(m) for m in self.mobjects if m not in keep], run_time=run_time)

    # ------------------------------------------------------------------ 1
    def hook(self):
        self.section(1)
        h = Text("demo_project/: 12 files, 1,455 lines, 19,072 tokens", font_size=30).to_edge(UP, buff=0.4)
        rows = VGroup()
        top = max(t for _, t in FILES)
        for name, tok in FILES:
            n = Text(name, font=MONO, font_size=18)
            b = Rectangle(width=2.6 * tok / top, height=0.2, stroke_width=0, fill_color=BLUE_D, fill_opacity=0.9)
            v = Text(f"{tok:,}", font=MONO, font_size=16, color=GREY_B)
            rows.add(VGroup(n, b, v))
        for r in rows:
            r[1].next_to(r[0], RIGHT, buff=0.2)
            r[2].next_to(r[1], RIGHT, buff=0.12)
        rows.arrange(DOWN, aligned_edge=LEFT, buff=0.1).move_to([-2.4, -0.45, 0])
        for r in rows:  # align bars in one column
            r[1].align_to([rows.get_left()[0] + 4.4, 0, 0], LEFT)
            r[2].next_to(r[1], RIGHT, buff=0.12)
        self.at("demo")
        self.play(FadeIn(h), run_time=0.4)
        self.at("files")
        self.play(FadeIn(rows, lag_ratio=0.06), run_time=1.4)
        q = card("the question", "Which line decides the file name\nof a figure image extracted\nfor the study-guide PDF?",
                 color=YELLOW, fs=24, mono=False).move_to([4.2, 1.2, 0])
        self.at("question")
        self.play(FadeIn(q), run_time=0.5)
        note = Text("every file it prints stays\nin the conversation and is\nsent again on every call", font_size=24,
                    color=GREY_A, line_spacing=0.9).move_to([4.2, -1.5, 0])
        self.at("every")
        self.play(FadeIn(note), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 2
    def naive(self):
        self.section(2)
        self.clear_stage()
        h = Text("no guidance (naive mode)", font_size=26).to_edge(UP, buff=0.4)
        self.at("first")
        self.play(FadeIn(h), run_time=0.3)
        cmds = VGroup(*[VGroup(Text("$ " + c, font=MONO, font_size=22), Text("→ " + o, font=MONO, font_size=19, color=GREY_B))
                        .arrange(RIGHT, buff=0.3) for c, o in NAIVE_CMDS]).arrange(DOWN, aligned_edge=LEFT, buff=0.22)
        cmds.move_to([-2.6, 1.3, 0])
        self.at("listed")
        self.play(FadeIn(cmds[0]), FadeIn(cmds[1]), run_time=0.4)
        self.at("printed")
        self.play(FadeIn(cmds[2]), FadeIn(cmds[3]), run_time=0.4)
        self.play(Indicate(cmds[3][1], color=RED_B, scale_factor=1.0), run_time=0.6)
        b = bars(NAIVE_SENT, RED_C, 0.00048, 2.2, -2.9)
        cap = Text("tokens sent per model call", font_size=18, color=GREY_A).next_to(b, DOWN, buff=0.15).align_to(b, LEFT)
        tot = Text("total 6,978", font=MONO, font_size=22, color=RED_B).next_to(b, RIGHT, buff=0.4).align_to(b, DOWN)
        self.at("five")
        self.play(FadeIn(b, lag_ratio=0.15), FadeIn(cap), run_time=0.8)
        self.at("almost")
        self.play(FadeIn(tot), run_time=0.3)
        ans = Text("answer: f\"{vid}_{f['t']:07.2f}.jpg\"\n        in stepstone/build_packs.py", font=MONO, font_size=20)
        ans.move_to([-3.3, -1.2, 0])
        self.at("answer")
        self.play(FadeIn(ans), run_time=0.4)
        self.at("wrong")
        wrong = Text("✗ wrong file: that line names StepStone's pack images", font_size=20, color=RED_B).next_to(ans, DOWN, buff=0.25).align_to(ans, LEFT)
        self.play(FadeIn(wrong), run_time=0.3)
        self.end_section()

    # ------------------------------------------------------------------ 3
    def invented(self):
        self.section(3)
        self.clear_stage()
        h = Text("the line numbers in the naive answers", font_size=26).to_edge(UP, buff=0.4)
        self.play(FadeIn(h), run_time=0.3)
        rows = VGroup(
            VGroup(Text("run 1: build_packs.py, line 108", font=MONO, font_size=24), Text("real line: 209", font_size=22, color=GREY_B)),
            VGroup(Text("run 3: build_packs.py, line 177", font=MONO, font_size=24), Text("real line: 209", font_size=22, color=GREY_B)),
            VGroup(Text("run 2: stepstone/study_guide.py, line 25", font=MONO, font_size=24), Text("no such file", font_size=22, color=RED_B)),
        )
        for r in rows:
            r.arrange(RIGHT, buff=0.6)
        rows.arrange(DOWN, aligned_edge=LEFT, buff=0.45).move_to([0, 0.9, 0])
        self.at("108")
        self.play(FadeIn(rows[0]), run_time=0.4)
        self.at("177")
        self.play(FadeIn(rows[1]), run_time=0.4)
        cat = card("cat stepstone/build_packs.py (what the model saw)", "\"\"\"Build StepStone packs ...\n...\n            name = f\"{vid}_{f['t']:07.2f}.jpg\"\n...",
                   fs=17).move_to([-1.0, -1.9, 0])
        nolines = Text("no line numbers", font_size=24, color=YELLOW).next_to(cat, RIGHT, buff=0.3)
        self.at("cat")
        self.play(FadeIn(cat), FadeIn(nolines), run_time=0.5)
        self.at("third")
        self.play(FadeIn(rows[2]), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 4
    def search(self):
        self.section(4)
        self.clear_stage()
        ins = card("added to the instructions (search mode)",
                   "The project is too big to read whole. Search first: use `grep -rn`\nto find candidate lines, then read "
                   "only the lines you need with\n`sed -n 'START,ENDp' FILE`. Never print whole files.", color=HARNESS_C, fs=19)
        ins.to_edge(UP, buff=0.5)
        self.at("same")
        self.play(FadeIn(ins), run_time=0.5)
        g = card("grep -rn PATTERN .", "framework/study_guide.py:41:        for fig in c.get(\"figures\", []):\n"
                                       "^ file                   ^ line", color=TOOL_C, fs=17).move_to([0, -0.5, 0])
        s = card("sed -n '40,43p' FILE", "        for c in concepts:\n            for fig in c.get(\"figures\", []):\n"
                                         "                fig[\"file\"] = ...\n                if not os.path.exists(...", color=TOOL_C, fs=17).move_to([0, -2.55, 0])
        self.at("shows")
        self.play(FadeIn(g), run_time=0.5)
        self.at("range")
        self.play(FadeIn(s), run_time=0.5)
        self.end_section()

    # ------------------------------------------------------------------ 5
    def results(self):
        self.section(5)
        self.clear_stage()
        out = card("what the model got back (run 1)", GREP_OUT, fs=17)
        out.scale_to_fit_width(min(out.width, 8.6)).to_corner(UL, buff=0.45)
        self.at("searched")
        self.play(FadeIn(out), run_time=0.6)
        nb = bars([6978], RED_C, 0.00045, 3.6, -3.1, bw=0.7, label=False)
        sb = bars([1799, 2314, 1910], TEAL_C, 0.00045, 4.7, -3.1, bw=0.45, gap=0.1, label=False)
        labs = VGroup(Text("naive (runs 1, 3)", font_size=15).next_to(nb, LEFT, buff=0.15).align_to(nb, DOWN),
                      Text("search (runs 1–3)", font_size=15).next_to(sb, UP, buff=0.4))
        vals = VGroup(Text("6,978", font=MONO, font_size=14).next_to(nb, UP, buff=0.06),
                      Text("1,799 · 2,314 · 1,910", font=MONO, font_size=13).next_to(sb, UP, buff=0.06))
        cap = Text("tokens sent in total", font_size=18, color=GREY_A).move_to([4.6, -3.75, 0])
        self.at("2", "around")
        self.play(FadeIn(nb), FadeIn(sb), FadeIn(labs), FadeIn(vals), FadeIn(cap), run_time=0.8)
        real = Text("every line number quoted: real", font_size=26, color=GREEN_B).move_to([-2.6, -2.4, 0])
        self.at("real")
        self.play(FadeIn(real), run_time=0.3)
        self.end_section()

    # ------------------------------------------------------------------ 6
    def near_miss(self):
        self.section(6)
        self.clear_stage()
        h = Text("search mode: 3 runs, the same answer every time", font_size=28).to_edge(UP, buff=0.4)
        self.play(FadeIn(h), run_time=0.3)
        src = [("40", "    for c in concepts:"), ("41", "        for fig in c.get(\"figures\", []):"),
               ("42", "            fig[\"file\"] = f\"img/{vid}_{fig['t']:07.2f}.jpg\""), ("", "  ⋮"),
               ("224", "    return (f'<figure{cls}><img src=\"{fig[\"file\"]}\"> ...")]
        lines = VGroup(*[VGroup(Text(n, font=MONO, font_size=18, color=GREY_B), Text(t.lstrip(), font=MONO, font_size=18))
                         for n, t in src])
        char_w = Text("x" * 20, font=MONO, font_size=18).width / 20       # Pango drops leading spaces: indent by hand
        for ln in lines:
            ln[1].next_to(ln[0], RIGHT, buff=0.35)
        lines.arrange(DOWN, aligned_edge=LEFT, buff=0.22).to_edge(LEFT, buff=0.7).shift(0.6 * UP)
        for ln, (_, t) in zip(lines, src):
            ln[1].align_to(lines[0][0].get_left() + RIGHT * (0.75 + char_w * (len(t) - len(t.lstrip()) - 4)), LEFT)
        title = Text("framework/study_guide.py", font_size=20, color=GREY_B).next_to(lines, UP, buff=0.3).align_to(lines, LEFT)
        box = SurroundingRectangle(VGroup(lines, title), buff=0.25, corner_radius=0.12, color=GREY_B, fill_opacity=0.07)
        self.play(FadeIn(box), FadeIn(title), FadeIn(lines), run_time=0.5)
        their = Text("↑ their answer (line 224): the name is used here", font_size=21, color=RED_B).next_to(box, DOWN, buff=0.2).align_to(box, LEFT)
        self.at("224")
        self.play(FadeIn(their), Indicate(lines[4], color=RED_B, scale_factor=1.0), run_time=0.6)
        right = Text("← made here", font_size=21, color=GREEN_B).next_to(box, RIGHT, buff=0.15).match_y(lines[2])
        m41 = Text("← grep matched", font_size=21, color=GREY_B).next_to(box, RIGHT, buff=0.15).match_y(lines[1])
        self.at("42")
        self.play(FadeIn(right), lines[2].animate.set_color(GREEN_B), run_time=0.5)
        self.at("below")
        self.play(FadeIn(m41), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 7
    def context(self):
        self.section(7)
        self.clear_stage()
        ins = card("added to the instructions (context mode)",
                   "Use `grep -rn -C 3` so you see a few lines around each match.\nWhen you find where a value is used, "
                   "search again for where it is set.", color=HARNESS_C, fs=19).to_edge(UP, buff=0.5)
        self.at("two")
        self.play(FadeIn(ins), run_time=0.5)
        cmds = VGroup(*[VGroup(Text("$ " + c, font=MONO, font_size=21), Text("→ " + o, font=MONO, font_size=18, color=GREY_B))
                        .arrange(RIGHT, buff=0.3) for c, o in CTX_CMDS]).arrange(DOWN, aligned_edge=LEFT, buff=0.25)
        cmds.move_to([-0.6, -0.2, 0])
        lab = Text("run 1:", font_size=20, color=GREY_A).next_to(cmds, UP, buff=0.25).align_to(cmds, LEFT)
        self.at("one")
        self.play(FadeIn(lab), FadeIn(cmds[0]), FadeIn(cmds[1]), run_time=0.5)
        self.at("fig")
        self.play(FadeIn(cmds[2]), run_time=0.3)
        self.play(Indicate(cmds[2][0], color=GREEN_B, scale_factor=1.0), run_time=0.6)
        r2 = Text("run 2: grep -rn -C 3 'figure' *  showed lines 38–44 → enough", font_size=20, color=GREY_A)
        r2.next_to(cmds, DOWN, buff=0.35).align_to(cmds, LEFT)
        self.at("other")
        self.play(FadeIn(r2), run_time=0.4)
        ans = Text("✓ both runs: framework/study_guide.py, line 42  (3,046 and 1,932 tokens)", font_size=22,
                   color=GREEN_B).move_to([0, -2.6, 0])
        self.at("both")
        self.play(FadeIn(ans), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 8
    def budget(self):
        self.section(8)
        self.clear_stage()
        h = Text("an output budget for every tool result", font_size=28).to_edge(UP, buff=0.45)
        self.play(FadeIn(h), run_time=0.3)
        full = Rectangle(width=10, height=0.6, stroke_width=0, fill_color=GREY_D, fill_opacity=1).move_to([0, 1.4, 0])
        fl = Text("grep -rn -C3 'figure': 5,343 chars", font=MONO, font_size=20).next_to(full, UP, buff=0.15)
        self.at("over")
        self.play(FadeIn(full), FadeIn(fl), run_time=0.5)
        head = Rectangle(width=1.9, height=0.6, stroke_width=0, fill_color=TEAL_C, fill_opacity=1)
        mid = Text("…3,343 chars truncated…", font=MONO, font_size=20, color=YELLOW)
        tail = Rectangle(width=1.9, height=0.6, stroke_width=0, fill_color=TEAL_C, fill_opacity=1)
        kept = VGroup(head, mid, tail).arrange(RIGHT, buff=0.3).move_to([0, -0.3, 0])
        kl = VGroup(Text("first 1,000", font_size=19).next_to(head, DOWN, buff=0.1),
                    Text("last 1,000", font_size=19).next_to(tail, DOWN, buff=0.1))
        self.at("beginning")
        self.play(FadeIn(head), FadeIn(tail), FadeIn(kl), run_time=0.5)
        self.at("cut")
        self.play(FadeIn(mid), run_time=0.4)
        ours = Text("ours: 2,000 characters per result", font_size=22).move_to([0, -1.9, 0])
        cx = Text("Codex: max_output_tokens, default 10,000 per command; “…N tokens truncated…”", font_size=22,
                  color=GREY_A).move_to([0, -2.6, 0])
        self.at("ours")
        self.play(FadeIn(ours), run_time=0.3)
        self.at("codex")
        self.play(FadeIn(cx), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 9
    def codex(self):
        self.section(9)
        self.clear_stage()
        q = card("Codex · base_instructions/default.md", CODEX_RULE, color=HARNESS_C, fs=18, title_fs=18).to_edge(UP, buff=0.45)
        self.at("instructions")
        self.play(FadeIn(q), run_time=0.6)
        self.at("prefer")
        self.play(Indicate(q[1][1], color=HARNESS_C, scale_factor=1.0), run_time=0.8)
        hd = Text("Codex CLI + Qwen3 4B, same task:", font_size=22, color=GREY_A).move_to([-2.6, -0.2, 0])
        rows = VGroup(*[VGroup(Text("$ " + c, font=MONO, font_size=19),
                               Text(f"(request: {t:,} tokens)", font_size=17, color=GREY_B)).arrange(RIGHT, buff=0.3)
                        for c, t in CODEX_CMDS]).arrange(DOWN, aligned_edge=LEFT, buff=0.2)
        rows.next_to(hd, DOWN, buff=0.25).align_to(hd, LEFT)
        self.at("small")
        self.play(FadeIn(hd), FadeIn(rows[0]), FadeIn(rows[1]), run_time=0.5)
        self.at("printed")
        self.play(FadeIn(rows[2]), run_time=0.3)
        self.play(Indicate(rows[2][0], color=RED_B, scale_factor=1.0), run_time=0.6)
        note = Text("instructions only help as much as the model follows them", font_size=22, color=YELLOW).to_edge(DOWN, buff=0.4)
        self.at("only")
        self.play(FadeIn(note), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 10
    def code(self):
        self.section(10)
        self.clear_stage()
        code, hl = code_panel(CODE, font_size=22)
        code.move_to([0, 0.2, 0])
        self.play(FadeIn(code, shift=0.2 * UP), run_time=0.5)
        hl.match_y(code.line_numbers[3])
        self.at("long")
        self.play(Create(hl), run_time=0.3)
        self.at("head")
        self.play(highlight(hl, code, 5), run_time=0.3)
        self.at("note")
        self.play(highlight(hl, code, 8), run_time=0.3)
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
