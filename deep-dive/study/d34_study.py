"""Study guide content for How LLMs Work: Deep Dive, episode 34: Speculative Decoding.

Build:  python framework/study_guide.py deep-dive d34 --video deep-dive/media/videos/d34_scene/1080p60/SpeculativeVideo.mp4
Figures: {"t": seconds into the rendered video, "caption": ...}; placed in body blocks with {fig0}, {fig1}, ...
Exercise kinds: mc (options), tf, short, number, order, code. Every exercise has an answer and a why.
Every number comes from code/d34_speculative/speculative.py (draft Qwen2.5-0.5B, target Qwen2.5-1.5B, greedy, 128 new
tokens, CPU float32, 8 threads; times vary by machine) or from the arithmetic shown.
"""

LESSON = {
    "series": "How LLMs Work: Deep Dive",
    "label": "Episode 34",
    "title": "Speculative Decoding",
    "tagline": "A small model guesses, a big model checks",
    "duration": "2:27",
    "intro": """<p>This lesson answers one question: how can a big model write faster without changing its output? A
small <b>draft</b> model guesses a few tokens ahead; the big <b>target</b> model checks them all in one pass and keeps the
ones it agrees with. On Python code, Qwen2.5-1.5B wrote the same 128 tokens 2.3 times faster; on prose, guessing is
harder and too many guesses made it slower. With sampling, an accept/reject rule keeps the target's distribution
exactly.</p>""",
    "prereq": """<div class="box note"><b>Before you start:</b> Deep Dive episodes 31 (the KV cache, prefill vs decode)
and 32 (batching). Code: <code>code/d34_speculative</code> (Qwen2.5-0.5B and 1.5B, about 8 GB in RAM).</div>""",
}

CONCEPTS = [
    {
        "title": "Why checking is cheap",
        "segment": (8, 41),
        "figures": [{"t": 40.0, "caption": "One Qwen2.5-1.5B pass after 100 tokens: 332 ms for 1 new token, 199 ms for 5, 242 ms "
                                           "for 9 (this CPU)."}],
        "body": [
            """<p>A big model writes one token per pass, and each decode pass is slow because it reads all the weights for
little math (episode 31). But a pass over several new tokens reads the same weights once. On this CPU, one Qwen2.5-1.5B
pass for 1 new token took <b>332 ms</b>; for 5 tokens, <b>199 ms</b>; for 9, <b>242 ms</b>. (One token is even slower
here, the same matrix-vector oddity as episode 32; on a GPU it is about equal.)</p>""",
            "{fig0}",
            """<div class="box key"><b class="t">Key idea</b>Checking a handful of tokens costs about as much as writing one.</div>""",
        ],
        "exercises": [
            {"kind": "tf", "q": "“Speculative decoding (greedy) can change the big model's output.”",
             "answer": "False.", "why": "Only tokens the target itself would have picked are kept; in all 8 runs the output "
                                       "was identical token for token.", "key": {'value': False}},
        ],
    },
    {
        "title": "The algorithm and its speed",
        "segment": (41, 108),
        "figures": [{"t": 59.0, "caption": "A real round: the draft guesses “ates · from · the · land”; the target agrees on 3, "
                                           "then adds its own “Earth”."},
                    {"t": 77.8, "caption": "Python code: 44.0 s plain, 19.0 s with 4 guesses (96% accepted, 4.7 tokens per "
                                           "pass)."},
                    {"t": 91.0, "caption": "Prose: 1.46x with 2 guesses (63% accepted), slower than plain with 6 or 8."},
                    {"t": 107.0, "caption": "Each draft token costs 131 ms, 40% of a 332 ms target step."}],
        "body": [
            """<p>Each round, the draft (Qwen2.5-0.5B) guesses k tokens one by one. The target reads all of them in a
single pass and, at every position, gives the token it would have picked. Keep the guesses up to the first disagreement,
then add the target's own token: every pass yields at least 1 token and at most k + 1.</p>""",
            '<div class="figrow">{fig0}{fig1}</div>',
            """<table><tr><th>128 tokens</th><th>accepted</th><th>tokens per pass</th><th>time</th><th>speed-up</th></tr>
<tr><td>code, plain</td><td>–</td><td>1</td><td>44.0 s</td><td>1x</td></tr>
<tr><td>code, k = 4</td><td>96%</td><td>4.7</td><td>19.0 s</td><td>2.31x</td></tr>
<tr><td>prose, plain</td><td>–</td><td>1</td><td>42.8 s</td><td>1x</td></tr>
<tr><td>prose, k = 2</td><td>63%</td><td>2.2</td><td>29.3 s</td><td>1.46x</td></tr>
<tr><td>prose, k = 4</td><td>54%</td><td>3.1</td><td>31.3 s</td><td>1.37x</td></tr>
<tr><td>prose, k = 6</td><td>37%</td><td>3.2</td><td>45.9 s</td><td>0.93x</td></tr>
<tr><td>prose, k = 8</td><td>32%</td><td>3.6</td><td>47.3 s</td><td>0.91x</td></tr></table>""",
            '<div class="figrow">{fig2}{fig3}</div>',
            """<p>The guesses aren't free: the draft takes 131 ms per token here, 40% of a target step, and every rejected
guess is wasted time. The best k depends on how predictable the text is and how cheap the draft is.</p>""",
            """<div class="box key"><b class="t">Key idea</b>Speed-up ≈ tokens per pass ÷ (cost of k draft steps + one check,
in target steps).</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "k = 4, and the first 2 guesses match the target. How many tokens does this pass add?",
             "answer": "3.", "why": "The 2 accepted guesses plus the target's own token at the first disagreement.", "key": {'parts': [{'label': None, 'value': 3, 'tol': 0.5, 'unit': None}]}},
            {"kind": "number", "q": "Estimate the code speed-up with k = 4: a round costs 4 draft steps (131 ms) and one "
                                    "5-token check (199 ms) and yields 4.7 tokens; plain decoding takes 332 ms per token.",
             "answer": "About 2.2x.", "why": "4 × 131 + 199 = 723 ms for 4.7 tokens = 154 ms per token; 332 / 154 ≈ 2.2 "
                                            "(measured 2.31x).", "key": {'parts': [{'label': None, 'value': 2, 'tol': 0.5, 'unit': None}]}},
            {"kind": "short", "q": "Why did 6 guesses make prose slower than plain decoding?",
             "answer": "Only 37% were accepted, so most of the 6 draft steps (131 ms each) were wasted, while each pass still "
                       "gained only about 3 tokens.", "why": "Draft cost grows with k; accepted tokens stop growing."},
        ],
    },
    {
        "title": "Sampling: keeping the distribution exact",
        "segment": (108, 138),
        "figures": [{"t": 128.0, "caption": "Accept with probability min(1, p/q), else sample from max(0, p − q): 1M draws match "
                                            "the target; 75% accepted."}],
        "body": [
            """<p>With sampling, the draft samples x from its distribution q; accept it with probability min(1, p(x) /
q(x)), where p is the target's distribution. If it is rejected, sample from what's left over, max(0, p − q) renormalised.
The result follows p exactly. A million simulated draws with p = (0.50, 0.30, 0.15, 0.05) and q = (0.25, 0.50, 0.20, 0.05)
gave (0.501, 0.299, 0.150, 0.050), and 75.0% of guesses were accepted: the sum of min(p, q).</p>""",
            "{fig0}",
            """<p>In code, the greedy check is a short loop: run the target once over the guesses, count how many match
its own choices, and keep those plus its next token.</p>""",
            """<div class="box key"><b class="t">Key idea</b>The acceptance rate is the overlap between the draft's and the
target's distributions.</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "With p = (0.50, 0.30, 0.15, 0.05) and q = (0.25, 0.50, 0.20, 0.05): what fraction of "
                                    "guesses is accepted, and which token is drawn after a rejection?",
             "answer": "75%; always token 1.", "why": "Σ min(p, q) = 0.25 + 0.30 + 0.15 + 0.05 = 0.75; max(0, p − q) = "
                                                     "(0.25, 0, 0, 0).", "key": {'parts': [{'label': 'accepted', 'value': 75, 'tol': 0.5, 'unit': '%'}, {'label': 'token after rejection (numbered from 1)', 'value': 1, 'tol': 0.5, 'unit': None}]}},
            {"kind": "code", "q": "<b>Try it yourself.</b> Check that the rule reproduces p.",
             "code": """import torch
p = torch.tensor([0.50, 0.30, 0.15, 0.05]); q = torch.tensor([0.25, 0.50, 0.20, 0.05])
x = torch.multinomial(q, 100_000, replacement=True)
keep = torch.rand(100_000) < (p[x] / q[x]).clamp(max=1)
r = (p - q).clamp(min=0)
y = torch.where(keep, x, torch.multinomial(r / r.sum(), 100_000, replacement=True))
print(torch.bincount(y) / 100_000)""",
             "answer": "About [0.50, 0.30, 0.15, 0.05].", "why": "Accepted mass min(p, q) plus the leftover p − q on rejection "
                                                                "adds up to p."},
        ],
    },
]
