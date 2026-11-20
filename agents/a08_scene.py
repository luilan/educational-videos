"""How AI Agents Work: Inside the Harness, episode 8 — Approvals and Policies.

Render from the repo root:  ./render.sh agents a08
Rule checks are code/a08_approvals/execpolicy_2026-11-20.txt (`codex execpolicy check --rules project.rules ...`, Codex
CLI 0.163.0). Runs: approval_harness.py (2026-11-20, Qwen3 4B on Ollama; logs in run_2026-11-20.txt).
Codex: openai/codex @ 89461ea (execpolicy/README.md, core/src/exec_policy.rs, protocol/src/protocol.rs AskForApproval).
"""
from manim import *

from common import MONO, code_panel, finish, highlight, next_up_card
from intro import play_token_intro
from a08_script import NEXT, TAGLINE, TITLE
from voiced_scene import VoicedScene

SERIES_LABEL = "HOW AI AGENTS WORK  ·  EPISODE 8"
MODEL_C, HARNESS_C, TOOL_C = PURPLE_B, TEAL_C, GOLD
RULE = """prefix_rule(pattern = ["pip", "install"], decision = "prompt",
            justification = "installs code from the internet")"""
CHECKS = [("python3 test_invoice.py", "allow", GREEN_B), ("pip install requests", "prompt", TOOL_C),
          ("rm -rf build", "forbidden", RED_B), ("ls -la", "no rule matched", GREY_B)]
UNTRUSTED = [("npm install", "asked → approved"), ("npm test", "asked → approved"), ("npm init --yes", "asked → approved"),
             ("python3 test_invoice.py", "allowed by a rule")]
CODE = """def decide(cmd, policy, escalated):
    rule = rule_decision(cmd)               # 1. prefix rules
    if rule:
        return rule                         # allow / prompt / forbidden
    if is_dangerous(cmd):                   # 2. rm -rf, sudo, git reset --hard
        return "prompt"
    if policy == "untrusted":               # 3. the approval policy
        return "prompt"
    return "prompt" if escalated else "allow"   # on-request"""


def card(title, body, color=GREY_B, fs=17, title_fs=17):
    t = Text(body, font=MONO, font_size=fs, line_spacing=0.85)
    hd = Text(title, font_size=title_fs, color=color)
    g = VGroup(hd, t).arrange(DOWN, aligned_edge=LEFT, buff=0.15)
    r = RoundedRectangle(width=g.width + 0.45, height=g.height + 0.35, corner_radius=0.12, color=color, fill_opacity=0.07)
    return VGroup(r, g.move_to(r))


def two_col(pairs, fs=21, gap=5.6, colors=None):
    rows = VGroup(*[VGroup(Text(a, font=MONO, font_size=fs), Text(b, font_size=fs, color=(colors[i] if colors else GREY_A)))
                    for i, (a, b) in enumerate(pairs)])
    for r in rows:
        r[1].next_to(r[0], RIGHT, buff=0.4)
    rows.arrange(DOWN, aligned_edge=LEFT, buff=0.3)
    x = rows.get_left()[0] + gap
    for r in rows:
        r[1].align_to([x, 0, 0], LEFT).match_y(r[0])
    return rows


class ApprovalsVideo(VoicedScene):
    VIDEO = "a08"

    def construct(self):
        play_token_intro(self, TITLE, 8, TAGLINE, label=SERIES_LABEL)
        self.hook()        # 1
        self.rules()       # 2
        self.checks()      # 3
        self.policy()      # 4
        self.order()       # 5
        self.ours()        # 6
        self.untrusted()   # 7
        self.onrequest()   # 8
        self.onrequest2()  # 9
        self.gap()         # 10
        self.tradeoff()    # 11
        self.code()        # 12
        self.outro()       # 13

    def clear_stage(self, *keep, run_time=0.6):
        self.play(*[FadeOut(m) for m in self.mobjects if m not in keep], run_time=run_time)

    # ------------------------------------------------------------------ 1
    def hook(self):
        self.section(1)
        cmds = VGroup(*[Text(t, font=MONO, font_size=26) for t in ("pip install requests", "git push origin main", "rm -rf build")])
        cmds.arrange(DOWN, aligned_edge=LEFT, buff=0.4).move_to([-1.5, 0.8, 0])
        self.at("installing")
        self.play(FadeIn(cmds[0]), run_time=0.3)
        self.at("pushing")
        self.play(FadeIn(cmds[1]), run_time=0.3)
        self.at("deleting")
        self.play(FadeIn(cmds[2]), run_time=0.3)
        q = Text("run it? ask? refuse?", font_size=30, color=YELLOW).move_to([0, -1.5, 0])
        self.at("decide")
        self.play(FadeIn(q), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 2
    def rules(self):
        self.section(2)
        self.clear_stage()
        h = Text("Codex execpolicy: prefix rules (Starlark)", font_size=28).to_edge(UP, buff=0.5)
        self.at("rules")
        self.play(FadeIn(h), run_time=0.3)
        c = card("project.rules", RULE, color=HARNESS_C, fs=20).move_to([0, 0.6, 0])
        self.at("matches")
        self.play(FadeIn(c), run_time=0.5)
        ds = VGroup(Text("allow", font_size=26, color=GREEN_B), Text("prompt = ask the user", font_size=26, color=TOOL_C),
                    Text("forbidden", font_size=26, color=RED_B)).arrange(RIGHT, buff=1.0).move_to([0, -1.5, 0])
        self.at("allow")
        self.play(FadeIn(ds[0]), run_time=0.3)
        self.at("prompt")
        self.play(FadeIn(ds[1]), run_time=0.3)
        self.at("forbidden")
        self.play(FadeIn(ds[2]), run_time=0.3)
        self.end_section()

    # ------------------------------------------------------------------ 3
    def checks(self):
        self.section(3)
        self.clear_stage()
        h = Text("$ codex execpolicy check --rules project.rules <command>", font=MONO, font_size=22).to_edge(UP, buff=0.5)
        self.at("checked")
        self.play(FadeIn(h), run_time=0.3)
        rows = two_col([(c, d) for c, d, _ in CHECKS], fs=23, gap=6.2, colors=[c for _, _, c in CHECKS]).move_to([0, 0.4, 0])
        cues = ["tests", "pip", "forbidden", "ls"]
        for r, cue in zip(rows, cues):
            self.at(cue)
            self.play(FadeIn(r), run_time=0.3)
            if cue == "forbidden":
                j = Text("“never delete recursively; move files to ./trash instead”", font_size=19, color=GREY_A)
                j.next_to(rows, DOWN, buff=0.5)
                self.play(FadeIn(j), run_time=0.3)
        self.end_section()

    # ------------------------------------------------------------------ 4
    def policy(self):
        self.section(4)
        self.clear_stage()
        h = Text("no rule matched → the approval policy", font_size=28).to_edge(UP, buff=0.5)
        self.at("match")
        self.play(FadeIn(h), run_time=0.3)
        rows = two_col([("untrusted", "ask the user about every one"), ("on-request", "run in the sandbox; ask only if the model requests more"),
                        ("never", "never ask; the sandbox is the only limit")], fs=22, gap=3.0).move_to([0, -0.1, 0])
        for r, cue in zip(rows, ["untrusted", "request", "never"]):
            self.at(cue)
            self.play(FadeIn(r), run_time=0.3)
        self.end_section()

    # ------------------------------------------------------------------ 5
    def order(self):
        self.section(5)
        self.clear_stage()
        boxes = VGroup()
        for t, c in [("1. rules", HARNESS_C), ("2. dangerous?", RED_B), ("3. policy", TOOL_C)]:
            r = RoundedRectangle(width=3.2, height=1.1, corner_radius=0.15, color=c, fill_opacity=0.1)
            boxes.add(VGroup(r, Text(t, font_size=26, color=c).move_to(r)))
        boxes.arrange(RIGHT, buff=0.9).move_to([0, 0.8, 0])
        arrows = VGroup(*[Arrow(boxes[i].get_right(), boxes[i + 1].get_left(), buff=0.1, color=GREY_B) for i in range(2)])
        self.at("rules")
        self.play(FadeIn(boxes[0]), run_time=0.3)
        self.at("dangerous")
        self.play(GrowArrow(arrows[0]), FadeIn(boxes[1]), run_time=0.4)
        d = Text("ask (under never: refuse)", font_size=20, color=RED_B).next_to(boxes[1], DOWN, buff=0.3)
        self.at("asks")
        self.play(FadeIn(d), run_time=0.3)
        self.at("policy")
        self.play(GrowArrow(arrows[1]), FadeIn(boxes[2]), run_time=0.4)
        src = Text("codex-rs/core/src/exec_policy.rs", font=MONO, font_size=18, color=GREY_B).to_edge(DOWN, buff=0.8)
        self.add(src)
        self.end_section()

    # ------------------------------------------------------------------ 6
    def ours(self):
        self.section(6)
        self.clear_stage()
        h = Text("our harness: same order, episode 7's sandbox, a scripted user", font_size=26).to_edge(UP, buff=0.5)
        self.play(FadeIn(h), run_time=0.3)
        user = VGroup(Text("approves: anything that stays in the project", font_size=24, color=GREEN_B),
                      Text("denies: pip, curl, wget, sudo, ~/.bashrc, profile", font_size=24, color=RED_B)).arrange(DOWN, aligned_edge=LEFT, buff=0.4)
        user.move_to([0, 0, 0])
        self.at("approves")
        self.play(FadeIn(user[0]), run_time=0.3)
        self.at("denies")
        self.play(FadeIn(user[1]), run_time=0.3)
        self.end_section()

    # ------------------------------------------------------------------ 7
    def untrusted(self):
        self.section(7)
        self.clear_stage()
        h = Text("untrusted, run 1 (run 2 the same)", font_size=28).to_edge(UP, buff=0.5)
        self.at("untrusted")
        self.play(FadeIn(h), run_time=0.3)
        rows = two_col(UNTRUSTED, fs=22, gap=5.4, colors=[TOOL_C, TOOL_C, TOOL_C, GREEN_B]).move_to([0, 0.4, 0])
        self.at("four")
        self.play(FadeIn(rows[:3], lag_ratio=0.2), run_time=0.6)
        self.at("rule")
        self.play(FadeIn(rows[3]), run_time=0.3)
        res = Text("4 questions to the user · tests pass", font_size=26, color=GREY_A).to_edge(DOWN, buff=1.0)
        self.at("passed")
        self.play(FadeIn(res), run_time=0.3)
        self.end_section()

    # ------------------------------------------------------------------ 8
    def onrequest(self):
        self.section(8)
        self.clear_stage()
        h = Text("on-request, run 1", font_size=28).to_edge(UP, buff=0.5)
        self.play(FadeIn(h), run_time=0.3)
        rows = two_col([("npm install", "allowed (sandboxed)"), ("npm test", "allowed (sandboxed)"),
                        ("npm init -y", "allowed (sandboxed)"), ("python3 test_invoice.py", "allowed by a rule")],
                       fs=22, gap=5.4, colors=[GREY_A, GREY_A, GREY_A, GREEN_B]).move_to([0, 0.4, 0])
        self.at("nothing", "asked")
        self.play(FadeIn(rows, lag_ratio=0.2), run_time=0.8)
        res = Text("0 questions to the user · tests pass", font_size=26, color=GREY_A).to_edge(DOWN, buff=1.0)
        self.at("passed")
        self.play(FadeIn(res), run_time=0.3)
        self.end_section()

    # ------------------------------------------------------------------ 9
    def onrequest2(self):
        self.section(9)
        self.clear_stage()
        h = Text("on-request, run 2", font_size=28).to_edge(UP, buff=0.4)
        self.play(FadeIn(h), run_time=0.3)
        R = [("cat README.md", "allowed", GREY_A),
             ("pip install -r requirements.txt", "rule: prompt → user: denied", RED_B),
             ("pip install … [escalation requested] ×3", "user: denied", RED_B),
             ("python3 -m venv venv", "allowed", GREY_A),
             ("./venv/bin/pip install -r requirements.txt", "no rule matched → allowed", TOOL_C),
             ("./venv/bin/python3 test_invoice.py", "allowed · ok", GREEN_B)]
        rows = two_col([(a, b) for a, b, _ in R], fs=19, gap=6.6, colors=[c for _, _, c in R]).move_to([0, 0.5, 0])
        if rows.width > 12.8:
            rows.scale_to_fit_width(12.8)
        self.at("read", "readme")
        self.play(FadeIn(rows[0]), run_time=0.3)
        self.at("pip")
        self.play(FadeIn(rows[1]), run_time=0.3)
        self.at("escalate")
        self.play(FadeIn(rows[2]), run_time=0.3)
        just = Text("justification: “The pip install command requires elevated privileges to install dependencies …”",
                    font_size=17, color=GREY_B).next_to(rows, DOWN, buff=0.35)
        if just.width > 12.8:
            just.scale_to_fit_width(12.8)
        self.play(FadeIn(just), run_time=0.3)
        self.at("virtual")
        self.play(FadeIn(rows[3]), run_time=0.3)
        self.at("path")
        self.play(FadeIn(rows[4]), run_time=0.3)
        blk = Text("…inside the sandbox: no network, so the download failed", font_size=21, color=YELLOW).next_to(just, DOWN, buff=0.3)
        self.at("blocked")
        self.play(FadeIn(blk), run_time=0.3)
        self.at("anyway")
        self.play(FadeIn(rows[5]), run_time=0.3)
        self.end_section()

    # ------------------------------------------------------------------ 10
    def gap(self):
        self.section(10)
        self.clear_stage()
        h = Text("rules match a command's first words", font_size=28).to_edge(UP, buff=0.5)
        self.at("rules")
        self.play(FadeIn(h), run_time=0.3)
        a = card("$ codex execpolicy check --rules project.rules ./venv/bin/pip install …", '{"matchedRules": []}', color=GREY_B, fs=19).move_to([0, 1.0, 0])
        b = card("… --resolve-host-executables ./venv/bin/pip install …",
                 '"matchedPrefix": ["pip", "install"], "decision": "prompt",\n"resolvedProgram": ".../venv/bin/pip"', color=HARNESS_C, fs=19).move_to([0, -1.0, 0])
        self.at("slipped")
        self.play(FadeIn(a), run_time=0.4)
        self.at("resolve")
        self.play(FadeIn(b), run_time=0.4)
        sb = Text("and behind every rule, the sandbox", font_size=24, color=YELLOW).to_edge(DOWN, buff=0.5)
        self.at("behind")
        self.play(FadeIn(sb), run_time=0.3)
        self.end_section()

    # ------------------------------------------------------------------ 11
    def tradeoff(self):
        self.section(11)
        self.clear_stage()
        a = Text("ask about everything → people learn to click “yes”", font_size=28, color=TOOL_C)
        b = Text("ask only for escalations → needs a sandbox that holds", font_size=28, color=HARNESS_C)
        g = VGroup(a, b).arrange(DOWN, buff=0.8).move_to([0, 0.2, 0])
        self.at("everything")
        self.play(FadeIn(a), run_time=0.4)
        self.at("escalations")
        self.play(FadeIn(b), run_time=0.4)
        self.end_section()

    # ------------------------------------------------------------------ 12
    def code(self):
        self.section(12)
        self.clear_stage()
        code, hl = code_panel(CODE, font_size=20)
        code.move_to([0, 0.1, 0])
        self.play(FadeIn(code, shift=0.2 * UP), run_time=0.5)
        hl.match_y(code.line_numbers[1])
        self.at("rules")
        self.play(Create(hl), run_time=0.3)
        self.at("dangerous")
        self.play(highlight(hl, code, 4), run_time=0.3)
        self.at("policy")
        self.play(highlight(hl, code, 6), run_time=0.3)
        self.end_section()

    # ------------------------------------------------------------------ 13
    def outro(self):
        self.section(13)
        self.clear_stage(run_time=0.4)
        card_ = next_up_card(NEXT)
        self.at("next")
        self.play(FadeIn(card_, shift=0.2 * UP), run_time=0.6)
        self.end_section()
        finish(self, hold=1.5)
