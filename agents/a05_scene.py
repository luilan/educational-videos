"""How AI Agents Work: Inside the Harness, episode 5 — Editing Files.

Render from the repo root:  ./render.sh agents a05
Every command, size and outcome is from code/a05_editing_files/edit_harness.py (2026-11-20, Qwen3 4B on Ollama; logs in
run_2026-11-20.txt). The 277-character patch is the fix the model worked out in patch run 2, applied by our apply_patch.
Codex: openai/codex @ 89461ea (codex-rs/apply-patch, core/src/tools/handlers/apply_patch.rs, apply_patch.lark).
"""
from manim import *

from common import MONO, code_panel, finish, highlight, next_up_card
from intro import play_token_intro
from a05_script import NEXT, TAGLINE, TITLE
from voiced_scene import VoicedScene

SERIES_LABEL = "HOW AI AGENTS WORK  ·  EPISODE 5"
MODEL_C, HARNESS_C, TOOL_C = PURPLE_B, TEAL_C, GOLD
BUG = "        return round(self.subtotal() - self.discount() + self.vat_total() + self.discount(), 2)"
FIX = "        return round(self.subtotal() - self.discount() + self.vat_total(), 2)"
PATCH = """apply_patch <<'EOF'
*** Begin Patch
*** Update File: invoice.py
@@ def total(self):
-        return round(self.subtotal() - self.discount() + self.vat_total() + self.discount(), 2)
+        return round(self.subtotal() - self.discount() + self.vat_total(), 2)
*** End Patch
EOF"""
DIFF3 = """-\"\"\"Invoices for a small print shop: line items, ...\"\"\"
+"Invoices for a small print shop: line items, ..."
-@dataclass
-class Item:
+dataclass Item:
-@dataclass
-class Customer:
+dataclass Customer:
... 14 lines removed, 7 added; the file no longer runs"""
CODE = """def seek(lines, pattern, start):
    for norm in (lambda s: s, str.rstrip, str.strip):
        for i in range(start, len(lines) - len(pattern) + 1):
            if all(norm(lines[i + j]) == norm(p)
                   for j, p in enumerate(pattern)):
                return i
    return None

# None: the patch is refused"""


def card(title, body, color=GREY_B, fs=16, title_fs=17, mono=True):
    t = Text(body, font=MONO if mono else None, font_size=fs, line_spacing=0.85) if mono else Text(body, font_size=fs, line_spacing=0.85)
    hd = Text(title, font_size=title_fs, color=color)
    g = VGroup(hd, t).arrange(DOWN, aligned_edge=LEFT, buff=0.15)
    r = RoundedRectangle(width=g.width + 0.45, height=g.height + 0.35, corner_radius=0.12, color=color, fill_opacity=0.07)
    return VGroup(r, g.move_to(r))


def fit(m, w=12.6):
    if m.width > w:
        m.scale_to_fit_width(w)
    return m


def cmd_rows(rows, fs=20):
    g = VGroup(*[VGroup(Text("$ " + c, font=MONO, font_size=fs), Text(o, font_size=fs - 2, color=GREY_B)).arrange(RIGHT, buff=0.3)
                 for c, o in rows]).arrange(DOWN, aligned_edge=LEFT, buff=0.2)
    return g


class EditingFilesVideo(VoicedScene):
    VIDEO = "a05"

    def construct(self):
        play_token_intro(self, TITLE, 5, TAGLINE, label=SERIES_LABEL)
        self.hook()        # 1
        self.rewrite()     # 2
        self.rewrite3()    # 3
        self.cost()        # 4
        self.patch()       # 5
        self.intercept()   # 6
        self.matching()    # 7
        self.patch_runs()  # 8
        self.right()       # 9
        self.code()        # 10
        self.outro()       # 11

    def clear_stage(self, *keep, run_time=0.6):
        self.play(*[FadeOut(m) for m in self.mobjects if m not in keep], run_time=run_time)

    # ------------------------------------------------------------------ 1
    def hook(self):
        self.section(1)
        h = Text("demo/invoice.py: 108 lines, 3,755 characters", font_size=28).to_edge(UP, buff=0.5)
        self.at("bug")
        self.play(FadeIn(h), run_time=0.4)
        l1 = Text("    def total(self):", font=MONO, font_size=20)
        cw = Text("x" * 20, font=MONO, font_size=20).width / 20
        parts = [Text("return round(self.subtotal() - self.discount() + self.vat_total()", font=MONO, font_size=20),
                 Text("+ self.discount()", font=MONO, font_size=20, color=RED_B), Text(", 2)", font=MONO, font_size=20)]
        l2 = VGroup(parts[0], parts[1].next_to(parts[0], RIGHT, buff=cw), parts[2].next_to(parts[1], RIGHT, buff=0.02))
        parts[1].align_to(parts[0], DOWN); parts[2].align_to(parts[0], DOWN)
        g = VGroup(l1, l2).arrange(DOWN, aligned_edge=LEFT, buff=0.2)
        l2.shift(RIGHT * 0.45)
        fit(g).move_to([0, 0.4, 0])
        nums = VGroup(Text("65", font=MONO, font_size=18, color=GREY_B).next_to(l1, LEFT, buff=0.3),
                      Text("66", font=MONO, font_size=18, color=GREY_B).next_to(l1, LEFT, buff=0.3).match_y(l2))
        self.play(FadeIn(g), FadeIn(nums), run_time=0.5)
        bad = SurroundingRectangle(parts[1], color=RED_B, buff=0.06)
        note = Text("the total adds the discount back", font_size=24, color=RED_B).next_to(g, DOWN, buff=0.6)
        self.at("adds")
        self.play(Create(bad), FadeIn(note), run_time=0.5)
        q = Text("how does an agent change one line?", font_size=28, color=YELLOW).to_edge(DOWN, buff=0.8)
        self.at("how")
        self.play(FadeIn(q), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 2
    def rewrite(self):
        self.section(2)
        self.clear_stage()
        ins = card("rewrite mode: the instructions say", "To edit a file, write the complete new file with a heredoc:\n"
                   "cat > FILE <<'EOF'\n...whole file...\nEOF", color=HARNESS_C, fs=19).to_edge(UP, buff=0.5)
        self.at("simplest")
        self.play(FadeIn(ins), run_time=0.5)
        rows = cmd_rows([("python3 test_invoice.py", "exit 1"),
                         ("cat > invoice.py <<'EOF' …", "277 chars, never read the file"),
                         ("cat > invoice.py <<'EOF' …", "the same, 10 more times")], fs=20).move_to([-2.2, -0.6, 0])
        lab = Text("run 1", font_size=20, color=GREY_A).next_to(rows, UP, buff=0.25).align_to(rows, LEFT)
        self.at("first")
        self.play(FadeIn(lab), FadeIn(rows), run_time=0.6)
        before = Rectangle(width=0.7, height=108 * 0.03, stroke_width=0, fill_color=BLUE_D, fill_opacity=0.9)
        after = Rectangle(width=0.7, height=12 * 0.03, stroke_width=0, fill_color=RED_C, fill_opacity=0.9)
        bars = VGroup(before, after).arrange(RIGHT, buff=0.6, aligned_edge=DOWN).move_to([4.6, -1.1, 0])
        bl = VGroup(Text("108 lines", font_size=17).next_to(before, DOWN, buff=0.1),
                    Text("12 lines", font_size=17).next_to(after, DOWN, buff=0.1))
        self.at("replaced")
        self.play(FadeIn(before), FadeIn(bl[0]), run_time=0.3)
        self.play(TransformFromCopy(before, after), FadeIn(bl[1]), run_time=0.7)
        self.end_section()

    # ------------------------------------------------------------------ 3
    def rewrite3(self):
        self.section(3)
        self.clear_stage()
        rows = cmd_rows([("cat invoice.py", "read it"), ("cat > invoice.py <<'EOF' …", "4,334 chars"),
                         ("cat > invoice.py <<'EOF' …", "4,052 chars  × 3, then stopped as stuck")], fs=20)
        lab = Text("run 3", font_size=20, color=GREY_A)
        g = VGroup(lab, rows).arrange(DOWN, aligned_edge=LEFT, buff=0.2).to_corner(UL, buff=0.6)
        self.at("did")
        self.play(FadeIn(lab), FadeIn(rows[0]), run_time=0.4)
        self.at("typed")
        self.play(FadeIn(rows[1]), FadeIn(rows[2]), run_time=0.5)
        d = card("diff against the original file", DIFF3, color=RED_B, fs=18).move_to([0.6, -1.3, 0])
        self.at("dropped")
        self.play(FadeIn(d), run_time=0.6)
        self.end_section()

    # ------------------------------------------------------------------ 4
    def cost(self):
        self.section(4)
        self.clear_stage()
        h = Text("characters the model must write for one fix", font_size=28).to_edge(UP, buff=0.6)
        self.play(FadeIn(h), run_time=0.3)
        vals = [("whole file (rewrite)", 3755, RED_C), ("the patch", 277, TEAL_C), ("the changed line", len(FIX), GREEN_C)]
        rows = VGroup()
        for name, v, c in vals:
            b = Rectangle(width=max(v * 0.0019, 0.05), height=0.55, stroke_width=0, fill_color=c, fill_opacity=0.9)
            rows.add(VGroup(Text(name, font_size=22), b, Text(f"{v:,}", font=MONO, font_size=20)))
        for r in rows:
            r[1].next_to([-2.9, 0, 0], RIGHT, buff=0)
            r[0].next_to(r[1], LEFT, buff=0.3)
        rows.arrange(DOWN, buff=0.45, aligned_edge=LEFT)
        for r in rows:
            r[1].align_to([-2.9, 0, 0], LEFT)
            r[0].next_to(r[1], LEFT, buff=0.3)
            r[2].next_to(r[1], RIGHT, buff=0.2)
        rows.move_to([0.3, -0.1, 0])
        self.at("costs")
        self.play(FadeIn(rows[0]), run_time=0.5)
        note = Text("every line copied is a chance to copy it wrong", font_size=24, color=YELLOW).to_edge(DOWN, buff=0.8)
        self.at("chance")
        self.play(FadeIn(note), run_time=0.3)
        self.at("fix")
        self.play(FadeIn(rows[2]), run_time=0.4)
        self.end_section()
        self.cost_rows = rows

    # ------------------------------------------------------------------ 5
    def patch(self):
        self.section(5)
        self.at("patches")
        self.play(FadeIn(self.cost_rows[1]), run_time=0.4)
        self.wait(1.0)
        self.clear_stage(run_time=0.4)
        p = fit(card("a patch (Codex's apply_patch format)", PATCH, color=TOOL_C, fs=17)).move_to([0, 0.6, 0])
        self.play(FadeIn(p), run_time=0.5)
        body = p[1][1]
        notes = VGroup(Text("which file", font_size=20, color=GREY_A), Text("− removed   + added", font_size=20, color=GREY_A),
                       Text("@@ and unchanged lines: where", font_size=20, color=GREY_A)).arrange(RIGHT, buff=0.9).next_to(p, DOWN, buff=0.45)
        self.at("names")
        self.play(FadeIn(notes[0]), run_time=0.3)
        self.at("minus")
        self.play(FadeIn(notes[1]), run_time=0.3)
        self.at("unchanged")
        self.play(FadeIn(notes[2]), run_time=0.3)
        self.end_section()

    # ------------------------------------------------------------------ 6
    def intercept(self):
        self.section(6)
        self.clear_stage()
        def box(label, color, w=2.8):
            r = RoundedRectangle(width=w, height=1.0, corner_radius=0.15, color=color, fill_opacity=0.12)
            return VGroup(r, Text(label, font_size=22, color=color).move_to(r))
        model = box("model", MODEL_C).move_to([-4.8, 1.2, 0])
        harness = box("harness", HARNESS_C).move_to([0, 1.2, 0])
        shell = box("shell", GREY_B).move_to([4.8, 1.2, 0])
        self.at("codex")
        self.play(FadeIn(model), FadeIn(harness), FadeIn(shell), run_time=0.5)
        a1 = Arrow(model.get_right(), harness.get_left(), buff=0.1, color=GREY_B)
        call = Text("exec_command: apply_patch <<'EOF' …", font=MONO, font_size=18).move_to([-2.4, 2.25, 0])
        self.at("runs")
        self.play(GrowArrow(a1), FadeIn(call), run_time=0.5)
        x = Cross(Line(harness.get_right(), shell.get_left()), stroke_color=RED_B, scale_factor=0.3)
        a2 = DashedLine(harness.get_right(), shell.get_left(), color=GREY_D)
        caught = Text("caught here: parsed, checked, applied", font_size=22, color=HARNESS_C).next_to(harness, DOWN, buff=0.5)
        never = Text("never reaches the shell", font_size=18, color=RED_B).next_to(a2, UP, buff=0.6)
        self.at("catches")
        self.play(Create(a2), FadeIn(x), FadeIn(never), FadeIn(caught), run_time=0.6)
        res = Text("result: “Success. Updated the following files: M invoice.py”", font=MONO, font_size=17).move_to([0, -1.5, 0])
        src = Text("Codex: core/src/tools/handlers/apply_patch.rs · intercept_apply_patch", font_size=18, color=GREY_A).move_to([0, -2.4, 0])
        self.at("itself")
        self.play(FadeIn(res), run_time=0.4)
        self.at("same")
        self.play(FadeIn(src), run_time=0.3)
        self.end_section()

    # ------------------------------------------------------------------ 7
    def matching(self):
        self.section(7)
        self.clear_stage()
        h = Text("finding the unchanged lines in the file", font_size=28).to_edge(UP, buff=0.5)
        self.play(FadeIn(h), run_time=0.3)
        steps = VGroup(*[Text(t, font_size=24) for t in ("1. exact match", "2. ignore spaces at line ends",
                                                         "3. ignore spaces around the line",
                                                         "4. normalise punctuation (“ ” – → \" \" -)")])
        steps.arrange(DOWN, aligned_edge=LEFT, buff=0.35).move_to([-1.5, 0.3, 0])
        self.at("exact")
        self.play(FadeIn(steps[0]), run_time=0.3)
        self.at("ends")
        self.play(FadeIn(steps[1]), run_time=0.3)
        self.at("around")
        self.play(FadeIn(steps[2]), run_time=0.3)
        self.at("fancy")
        self.play(FadeIn(steps[3]), run_time=0.3)
        ref = Text("apply_patch failed: could not find these lines in the file: …", font=MONO, font_size=18,
                   color=RED_B).to_edge(DOWN, buff=1.1)
        sub = Text("the file is untouched; the error is the tool's result", font_size=20, color=GREY_A).next_to(ref, DOWN, buff=0.2)
        self.at("refused")
        self.play(FadeIn(ref), FadeIn(sub), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 8
    def patch_runs(self):
        self.section(8)
        self.clear_stage()
        h = Text(PATCH_RUNS_TITLE, font_size=28).to_edge(UP, buff=0.5)
        self.at("never")
        self.play(FadeIn(h), run_time=0.4)
        rows = VGroup(*[VGroup(Text(a, font_size=20, color=GREY_A), Text(b, font_size=20)).arrange(RIGHT, buff=0.4)
                        for a, b in PATCH_RUNS]).arrange(DOWN, aligned_edge=LEFT, buff=0.25)
        for r in rows:
            r[1].align_to(rows[0][0].get_left() + RIGHT * 2.3, LEFT)
        rows.move_to([0, 1.0, 0])
        self.play(FadeIn(rows, lag_ratio=0.2), run_time=0.8)
        draft = card("a patch the model drafted (read-first run 2)", "@@ -4,5 +4,4 @@\n-    return sum(self.items) + 10.0\n+    return sum(self.items)",
                     color=GREY_B, fs=18).move_to([-2.4, -1.8, 0])
        self.at("drafted")
        self.play(FadeIn(draft), run_time=0.5)
        ref = Text("our apply_patch:\n“could not find these lines”", font_size=21, color=RED_B, line_spacing=0.9).next_to(draft, RIGHT, buff=0.6)
        self.at("refused")
        self.play(FadeIn(ref), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 9
    def right(self):
        self.section(9)
        self.clear_stage()
        p = fit(card("the fix from patch run 2, sent as a patch", PATCH, color=GREEN_B, fs=17)).move_to([0, 0.9, 0])
        self.at("worked")
        self.play(FadeIn(p), run_time=0.6)
        cut = Text("(run 2's reply hit the token cap before the tool call)", font_size=20, color=GREY_A).next_to(p, DOWN, buff=0.3)
        self.at("cap")
        self.play(FadeIn(cut), run_time=0.3)
        res = VGroup(Text("277 characters", font=MONO, font_size=24, color=TOOL_C),
                     Text("1 line changed", font_size=24), Text("test: ok", font=MONO, font_size=24, color=GREEN_B)).arrange(RIGHT, buff=0.9)
        res.to_edge(DOWN, buff=0.8)
        self.at("277", "two")
        self.play(FadeIn(res[0]), FadeIn(res[1]), run_time=0.4)
        self.at("passes")
        self.play(FadeIn(res[2]), run_time=0.3)
        self.end_section()

    # ------------------------------------------------------------------ 10
    def code(self):
        self.section(10)
        self.clear_stage()
        code, hl = code_panel(CODE, font_size=22)
        code.move_to([0, 0.2, 0])
        self.play(FadeIn(code, shift=0.2 * UP), run_time=0.5)
        hl.match_y(code.line_numbers[1])
        self.at("search")
        self.play(Create(hl), run_time=0.3)
        self.at("strict")
        self.play(highlight(hl, code, 1), run_time=0.3)
        self.at("replace")
        self.play(highlight(hl, code, 5), run_time=0.3)
        self.at("arent", "say")
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


PATCH_RUNS_TITLE = "patch runs: the file was never damaged"
PATCH_RUNS = [("patch run 1", "no edit: drafted a patch for code it had not read"),
              ("patch run 2", "read the file, found the bug; reply cut at the token cap"),
              ("patch run 3", "no edit: wrote apply_patch as text, for code it had not read"),
              ("read-first 1", "no edit"),
              ("read-first 2", "no edit: wrote apply_patch as text, for code it had not read")]
