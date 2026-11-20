# How AI Agents Work, episode 8 · Approvals and Policies

Who decides whether a command runs: rules first, then a dangerous-command check, then the approval policy, the order
Codex CLI uses (`codex-rs/core/src/exec_policy.rs`).

- `project.rules`: five Codex execpolicy prefix rules (Starlark): allow running the project's code and `git
  status/diff/log`; prompt for `pip install` and `git push`; forbid `rm -rf` with a hint. `execpolicy_2026-11-20.txt`
  is the output of `codex execpolicy check --rules project.rules <command>` (Codex CLI 0.163.0) for six commands.
- `approval_harness.py`: the tiny harness with a Codex-style decision for every command:
  1. prefix rules (the same rules as `project.rules`, in Python);
  2. a dangerous-command check (`rm -rf`, `sudo`, `git reset --hard`): ask;
  3. the policy: `untrusted` asks about every command no rule allows; `on-request` runs everything in the sandbox and
     asks only when the model sets `sandbox_permissions: "require_escalated"` (with a justification).
  The "user" is scripted so runs are repeatable: it approves anything that stays in the project and denies downloads
  (`pip`, `curl`, `wget`), `sudo` and changes to the shell profile. Every command runs inside episode 7's bubblewrap
  sandbox (`sandbox.py`); an approved escalation gets network access and a writable (fake) home.
- `llm.py`: one call to Ollama's Responses API, with a per-reply cap (`MAX_OUTPUT_TOKENS`, default 8,000).

```bash
codex execpolicy check --rules project.rules pip install requests
ollama pull qwen3:4b
python3 approval_harness.py untrusted
python3 approval_harness.py on-request
```

Needs Linux with bubblewrap. Python standard library only.

Results (2026-11-20):

| command | `codex execpolicy check` decision |
|---|---|
| `python3 test_invoice.py` | allow |
| `git status` | allow |
| `pip install requests` | prompt (*"installs code from the internet"*) |
| `git push origin main` | prompt |
| `rm -rf build` | forbidden (*"never delete recursively; move files to ./trash instead"*) |
| `ls -la` | no rule matched (left to the policy) |

Qwen3 4B on Ollama, CPU, the task of episode 7 (full output in `run_2026-11-20.txt`; runs vary):

| policy | run | commands | asked the user | outcome |
|---|---|---|---|---|
| untrusted | 1 | `npm install`, `npm test`, `npm init`, `npm init -y`, `python3 test_invoice.py` | 4 (all approved) | tests pass |
| untrusted | 2 | `npm install`, `npm test`, `npm test --loglevel=verbose`, `npm init --yes`, `python3 test_invoice.py` | 4 (all approved) | tests pass |
| on-request | 1 | `npm install`, `npm test`, `npm init -y`, `python3 test_invoice.py` | 0 | tests pass |
| on-request | 2 | `cat README.md`, `pip install -r requirements.txt` (rule: prompt → denied), the same with `sandbox_permissions: require_escalated` ×3 (denied), `python3 -m venv venv`, `./venv/bin/pip install -r requirements.txt` (no rule matched → ran in the sandbox, download failed), `./venv/bin/python3 test_invoice.py` | 4 (all denied) | tests pass, 10 model calls |

- On-request run 2 is the interesting one: after the rule and three escalation requests were denied (justification:
  *"The pip install command requires elevated privileges to install dependencies …"*), the model called pip by its
  path inside a venv. Prefix rules match a command's first words, so `./venv/bin/pip install` matched none of ours and
  ran; the sandbox's missing network stopped the download. Codex's checker resolves such paths when asked:
  `codex execpolicy check --resolve-host-executables … ./venv/bin/pip install …` matches the `pip install` rule
  (`"resolvedProgram": ".../venv/bin/pip"`, decision `prompt`); without the flag, no rule matched (both in
  `execpolicy_2026-11-20.txt`).
- Lesson from the lab: in untrusted run 1 the model ran `npm init` without `-y`, which waits for answers on standard
  input. Our job runner's standard input was the job list itself, and `npm init` read it, ending the queue. Commands
  now get `stdin=DEVNULL`.

How Codex CLI does it (openai/codex @ 89461ea):

- Policies (`AskForApproval` in `codex-rs/protocol/src/protocol.rs`): `untrusted` (ask unless a rule allows),
  `on-request` (the default: the model decides when to ask), `granular`, `never` (never ask; failures go back to the
  model).
- Decision for a command no rule matched (`render_decision_for_unmatched_command` in `core/src/exec_policy.rs`):
  dangerous → prompt (forbidden under `never`); `never` → allow (the sandbox protects); `untrusted` → prompt;
  `on-request` in a restricted sandbox → prompt only if the model requested a sandbox override.
- Rule language and CLI: `codex-rs/execpolicy/README.md`.
