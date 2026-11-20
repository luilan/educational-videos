"""Study guide content for How AI Agents Work: Inside the Harness, episode 8: Approvals and Policies.

Build:  python framework/study_guide.py agents a08 --video agents/media/videos/a08_scene/1080p60/ApprovalsVideo.mp4
Every number comes from code/a08_approvals (2026-11-20): `codex execpolicy check` (Codex CLI 0.163.0) on project.rules,
and approval_harness.py with Qwen3 4B on Ollama (2 untrusted runs, 2 on-request runs, episode 7's sandbox).
Codex code: openai/codex @ 89461ea.
"""

LESSON = {
    "series": "How AI Agents Work: Inside the Harness",
    "label": "Episode 8",
    "title": "Approvals and Policies",
    "tagline": "Who decides whether a command runs",
    "duration": "2:41",
    "intro": """<p>A sandbox refuses; but some commands really need more, and someone has to decide which. Codex CLI
decides in a fixed order: <b>prefix rules</b> (allow, prompt, forbidden), then a <b>dangerous-command</b> check, then
the <b>approval policy</b> (<code>untrusted</code>, <code>on-request</code>, <code>never</code>). This lesson checks
rules with Codex's own checker, runs a harness under two policies with a scripted user, and shows the gap a model
found between a rule and the sandbox.</p>""",
    "prereq": """<div class="box note"><b>Before you start:</b> episodes 1–7, especially 7 (Sandboxing). Code:
<code>code/a08_approvals</code> (Linux with bubblewrap; <code>codex execpolicy check</code> needs the Codex CLI
binary; the harness needs a local Ollama with <code>qwen3:4b</code>).</div>""",
}

CONCEPTS = [
    {
        "title": "Rules",
        "segment": (19.8, 47.2),
        "figures": [{"t": 28.4, "caption": "An execpolicy prefix rule: a pattern, a decision, a justification."},
                    {"t": 46.9, "caption": "`codex execpolicy check` on four commands: allow, prompt, forbidden, no "
                                           "rule matched."}],
        "body": [
            """<p>Codex's execpolicy rules are Starlark <code>prefix_rule(pattern=[…], decision=…,
justification=…)</code> calls. A rule matches the first words of a command; its decision is <code>allow</code>,
<code>prompt</code> (ask the user) or <code>forbidden</code>. For a forbidden rule the justification should suggest an
alternative, and it is shown to the model.</p>""",
            "{fig0}",
            """<p>With our five rules, <code>codex execpolicy check</code> gave: <code>python3 test_invoice.py</code> →
allow; <code>pip install requests</code> → prompt (<i>“installs code from the internet”</i>); <code>rm -rf
build</code> → forbidden (<i>“never delete recursively; move files to ./trash instead”</i>); <code>ls -la</code> → no
rule matched, so the decision is left to the policy.</p>""",
            "{fig1}",
            """<div class="box key"><b class="t">Key idea</b>Rules are the project's explicit list: what always runs,
what must be asked about, and what never runs.</div>""",
        ],
        "exercises": [
            {"kind": "mc", "q": "A rule's decision is <code>prompt</code>. What happens?",
             "options": ["The command runs", "The user is asked", "The command is refused", "The model is asked"],
             "answer": "B.", "why": "prompt means ask the user before running.", "key": {"choice": 1}},
            {"kind": "tf", "q": "“Every command matches some rule.”",
             "answer": "False.", "why": "ls -la matched none; such commands go to the approval policy.",
             "key": {"value": False}},
        ],
    },
    {
        "title": "Policies, and the order of decisions",
        "segment": (47.7, 84.4),
        "figures": [{"t": 63.5, "caption": "The approval policy decides commands no rule matched: untrusted, on-request, "
                                           "never."},
                    {"t": 72.2, "caption": "The order: rules, then the dangerous-command check, then the policy."}],
        "body": [
            """<p>For a command no rule matched, the <b>policy</b> decides: <code>untrusted</code> asks the user about
every one; <code>on-request</code> (Codex's default) runs it in the sandbox without asking, and asks only when the
model requests more (<code>sandbox_permissions: "require_escalated"</code> plus a justification); <code>never</code>
never asks and relies on the sandbox alone.</p>""",
            '<div class="figrow">{fig0}{fig1}</div>',
            """<p>The order (<code>core/src/exec_policy.rs</code>): rules first; then a dangerous-command check
(<code>rm -rf</code>, <code>sudo</code>…), which asks, or under <code>never</code> refuses; then the policy. Our harness
uses the same order with episode 7's sandbox and a scripted user that approves anything staying in the project and
denies downloads, <code>sudo</code> and shell-profile changes.</p>""",
            """<div class="box key"><b class="t">Key idea</b>Rules, then danger, then policy: explicit knowledge first,
general caution second, a default last.</div>""",
        ],
        "exercises": [
            {"kind": "order", "q": "Put Codex's decision steps in order.",
             "answer": "prefix rules → dangerous-command check → approval policy.",
             "why": "An explicit rule always wins; the policy is the fallback.",
             "key": {"items": ["prefix rules", "dangerous-command check", "approval policy"]}},
            {"kind": "mc", "q": "Under <code>on-request</code>, when is the user asked about an unmatched, harmless "
                                "command?",
             "options": ["Always", "Never", "When the model requests an escalation", "When it fails"],
             "answer": "C.", "why": "Otherwise it runs in the sandbox.", "key": {"choice": 2}},
        ],
    },
    {
        "title": "The policies in practice",
        "segment": (84.9, 124.8),
        "figures": [{"t": 96.9, "caption": "Untrusted: four questions, all about harmless npm commands; the test run "
                                           "allowed by a rule."},
                    {"t": 103.3, "caption": "On-request run 1: no questions; everything ran in the sandbox."},
                    {"t": 124.5, "caption": "On-request run 2: pip denied by rule and escalation; a venv's pip slipped "
                                            "past the rule and the sandbox stopped the download."}],
        "body": [
            """<p>Same task as episode 7. Under <code>untrusted</code>, both runs asked the user four times, all about
harmless <code>npm</code> commands; only <code>python3 test_invoice.py</code> was allowed by a rule. Under
<code>on-request</code>, run 1 asked nothing. Every run's tests passed.</p>""",
            '<div class="figrow">{fig0}{fig1}</div>',
            """<p>On-request run 2 read the README. <code>pip install -r requirements.txt</code> hit the prompt rule
and was denied; the model requested escalation three times with a justification (<i>“The pip install command requires
elevated privileges …”</i>), denied each time. Then it created a virtual environment and ran
<code>./venv/bin/pip install -r requirements.txt</code>: no rule matched, so it ran in the sandbox, where the download
failed for lack of network. The tests passed anyway.</p>""",
            "{fig2}",
            """<div class="box key"><b class="t">Key idea</b>Denials steer a model to alternatives, and some
alternatives get around the rules; the sandbox is what makes that safe.</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "On-request run 2 asked the user four times. How many were escalation requests?",
             "answer": "3.", "why": "One ask came from the pip rule, three from require_escalated.",
             "key": {"parts": [{"label": None, "value": 3, "tol": 0.5, "unit": None}]}},
            {"kind": "short", "q": "Why did <code>./venv/bin/pip install</code> not match the <code>pip install</code> "
                                   "rule?",
             "answer": "Prefix rules compare the command's first words; the first word was ./venv/bin/pip, not pip.",
             "why": "Rules match tokens literally unless paths are resolved."},
        ],
    },
    {
        "title": "Closing the gap, and the trade-off",
        "segment": (125.3, 155.5),
        "figures": [{"t": 140.0, "caption": "With --resolve-host-executables, Codex's checker maps ./venv/bin/pip to pip, "
                                            "and the rule matches: prompt."},
                    {"t": 155.2, "caption": "The decision in code: rules, the dangerous check, the policy."}],
        "body": [
            """<p><code>codex execpolicy check --rules project.rules ./venv/bin/pip install …</code> also finds no rule.
With <code>--resolve-host-executables</code> it resolves the program path to its name and the <code>pip
install</code> rule matches (<code>"resolvedProgram": ".../venv/bin/pip"</code>, decision prompt). Whatever the rules
miss, the sandbox still holds.</p>""",
            "{fig0}",
            """<p>The trade-off: asking about everything (untrusted) produces many questions about harmless commands,
and people learn to click “yes”; asking only for escalations (on-request) is quieter, and works because a sandbox does
the rest.</p>""",
            "{fig1}",
            """<div class="box key"><b class="t">Key idea</b>Approvals are for the rare commands that need more than the
sandbox allows; everything else should be safe to run without asking.</div>""",
        ],
        "exercises": [
            {"kind": "tf", "q": "“Without --resolve-host-executables, Codex's checker matched ./venv/bin/pip to the pip "
                                "rule.”",
             "answer": "False.", "why": "Without the flag: {\"matchedRules\": []}.", "key": {"value": False}},
            {"kind": "code", "q": "<b>Try it yourself.</b> Decide a command: rules, then danger, then policy.",
             "code": """def decide(cmd, policy, escalated):
    rule = rule_decision(cmd)
    if rule:
        return rule
    if is_dangerous(cmd):
        return "prompt"
    if policy == "untrusted":
        return "prompt"
    return "prompt" if escalated else "allow\"""",
             "answer": "approval_harness.py's decide() does exactly this.",
             "why": "A real harness would also resolve program paths before matching rules."},
        ],
    },
]
