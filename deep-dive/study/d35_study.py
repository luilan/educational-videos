"""Study guide content for How LLMs Work: Deep Dive, episode 35: Decoding Strategies Compared.

Build:  python framework/study_guide.py deep-dive d35 --video deep-dive/media/videos/d35_scene/1080p60/DecodingVideo.mp4
Figures: {"t": seconds into the rendered video, "caption": ...}; placed in body blocks with {fig0}, {fig1}, ...
Exercise kinds: mc (options), tf, short, number, order, code. Every exercise has an answer and a why.
Every number comes from code/d35_decoding/decoding.py (GPT-2 small, 4 prompts × 120 new tokens, 3 seeded samples per prompt
for random methods; judge Qwen2.5-1.5B) or from the arithmetic shown.
"""

LESSON = {
    "series": "How LLMs Work: Deep Dive",
    "label": "Episode 35",
    "title": "Decoding Strategies Compared",
    "tagline": "Greedy, beams, temperature, top-k, top-p, min-p",
    "duration": "2:17",
    "intro": """<p>This lesson answers one question: given the model's probabilities, how should you pick the next token?
The same GPT-2 becomes a broken record with <b>greedy</b> decoding or <b>beam search</b> (69% and 76% repeated phrases), a
word-salad generator at <b>temperature</b> 1.5, and a reasonable storyteller with <b>truncated sampling</b>: top-k, top-p
or min-p. A bigger judge model finds the looping text the least surprising of all: the most likely text is not the best
text.</p>""",
    "prereq": """<div class="box note"><b>Before you start:</b> How LLMs Work episode 10 (from vectors back to words) and
LLMs in Practice episode 3 (sampling). Code: <code>code/d35_decoding</code> (GPT-2 small and Qwen2.5-1.5B).</div>""",
}

CONCEPTS = [
    {
        "title": "Greedy and beam search",
        "segment": (8, 67),
        "figures": [{"t": 36.5, "caption": "Two measurements: repetition (share of 4-token phrases already used) and judge "
                                           "surprise (Qwen2.5-1.5B's loss on the text)."},
                    {"t": 50.0, "caption": "Greedy: “He was wearing a black hat with a black belt.” again and again; 69% "
                                           "repeats."},
                    {"t": 65.5, "caption": "Beam search (4 beams): 76% repeats, yet the lowest judge surprise, 0.57."}],
        "body": [
            """<p>GPT-2 continues four story openings for 120 tokens. Two measurements: <b>repetition</b>, the share of
4-token phrases it has already used in the same text, and <b>judge surprise</b>, the mean loss of a bigger model
(Qwen2.5-1.5B) on the text: lower means more predictable.</p>""",
            "{fig0}",
            """<p><b>Greedy</b> decoding always takes the most likely token. It starts fine, then gets stuck: “He was
wearing a black hat with a black belt.” over and over; 69% of its phrases are repeats. <b>Beam search</b> keeps the 4
most likely partial texts and returns the most likely overall: 76% repeats. Yet the judge finds these texts the
<i>least</i> surprising (0.67 and 0.57): a loop is very predictable.</p>""",
            '<div class="figrow">{fig1}{fig2}</div>',
            """<div class="box key"><b class="t">Key idea</b>Maximising likelihood finds bland, repetitive text: the most
likely text is not the best text.</div>""",
        ],
        "exercises": [
            {"kind": "mc", "q": "Why does the judge give the looping greedy text such a low loss?",
             "options": ["It is well written", "Once a phrase repeats, every next token is easy to predict",
                         "The judge was trained on GPT-2's outputs", "Greedy text is shorter"],
             "answer": "B.", "why": "Repetition makes text predictable, so likelihood alone rewards it."},
        ],
    },
    {
        "title": "Temperature",
        "segment": (67, 85),
        "figures": [{"t": 83.5, "caption": "GPT-2's next-token probabilities at T = 0.7, 1.0, 1.5; judge surprise 2.50, 4.91, "
                                           "9.10."}],
        "body": [
            """<p>Sampling draws the next token at random in proportion to its probability; <b>temperature</b> T first
divides the logits: below 1 it sharpens the distribution, above 1 it flattens it. At 0.7 the text is varied and mostly
sensible (4% repeats, judge 2.50); at 1.0 it wanders into nonsense (4.91); at 1.5 it is word salad (9.10): “…Skooter
bipartisan Zionist Liberty Innovators…”.</p>""",
            "{fig0}",
            """<div class="box key"><b class="t">Key idea</b>The tail of unlikely tokens is huge: after “…opened the door
and”, the 12 most likely tokens hold only 32% of the probability.</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "Logits (2, 1, 0). Give the probabilities at T = 1, 0.5 and 2.",
             "answer": "(0.665, 0.245, 0.090); (0.867, 0.117, 0.016); (0.506, 0.307, 0.186).",
             "why": "softmax(logits / T): dividing by 0.5 doubles the gaps; dividing by 2 halves them."},
        ],
    },
    {
        "title": "Cutting the tail: top-k, top-p, min-p",
        "segment": (85, 127),
        "figures": [{"t": 102.0, "caption": "Tokens kept after “…door and” (top token 0.066) and after “…a black” (0.156): top-k "
                                            "40 / 40, top-p 608 / 275, min-p 28 / 8."},
                    {"t": 115.0, "caption": "All 8 strategies: repetition vs judge surprise."}],
        "body": [
            """<p>Truncation methods drop the unlikely tail before sampling. <b>Top-k</b> keeps the k most likely tokens;
<b>top-p</b> (nucleus) keeps the smallest set whose probabilities add up to p; <b>min-p</b> keeps tokens at least a
fraction (here 0.1) as likely as the top one. Only min-p's cut follows the model's confidence closely: after “…door
and” it keeps 28 tokens, after “…a black” (where “hat” has 0.156) only 8. Top-k always keeps 40; top-p keeps 608, then
275.</p>""",
            '<div class="figrow">{fig0}{fig1}</div>',
            """<table><tr><th>strategy</th><th>repetition</th><th>judge surprise</th></tr>
<tr><td>greedy</td><td>69%</td><td>0.67</td></tr><tr><td>beam search, 4 beams</td><td>76%</td><td>0.57</td></tr>
<tr><td>temperature 0.7</td><td>4%</td><td>2.50</td></tr><tr><td>temperature 1.0</td><td>0%</td><td>4.91</td></tr>
<tr><td>temperature 1.5</td><td>0%</td><td>9.10</td></tr><tr><td>top-k 40</td><td>2%</td><td>3.10</td></tr>
<tr><td>top-p 0.9</td><td>0%</td><td>3.87</td></tr><tr><td>min-p 0.1</td><td>10%</td><td>2.28</td></tr></table>""",
            """<p>All three stay varied, and min-p has the lowest judge surprise of the sampling methods. Judge surprise
is only a proxy: it can't tell a loop from good prose, which is why the two measurements are read together.</p>""",
            """<div class="box key"><b class="t">Key idea</b>Sample, but from a truncated distribution; a cut relative to
the top token adapts to the model's confidence.</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "Probabilities (0.5, 0.2, 0.15, 0.1, 0.05). How many tokens does top-p 0.9 keep? min-p "
                                    "0.1? min-p 0.3?",
             "answer": "4; 5; 3.", "why": "Top-p: cumulative 0.5, 0.7, 0.85, 0.95 reaches 0.9 at the 4th token. Min-p 0.1 "
                                         "keeps p ≥ 0.05; min-p 0.3 keeps p ≥ 0.15."},
            {"kind": "code", "q": "<b>Try it yourself.</b> Write min-p sampling and count the kept tokens.",
             "code": """import torch
logits = torch.tensor([2.0, 1.5, 1.0, 0.0, -1.0, -3.0])
p = logits.softmax(-1)
logits[p < 0.1 * p.max()] = float("-inf")
print((logits > float("-inf")).sum(), torch.multinomial(logits.softmax(-1), 1))""",
             "answer": "4 tokens kept (probabilities ≥ 0.1 × 0.46).", "why": "p ≈ (0.46, 0.28, 0.17, 0.06, 0.02, 0.003); "
                                                                            "the cut is 0.046."},
        ],
    },
]
