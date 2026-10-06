"""Study guide content for How LLMs Work: Deep Dive, episode 12: Attention Sinks.

Build:  python framework/study_guide.py deep-dive d12 --video deep-dive/media/videos/d12_scene/1080p60/AttentionSinksVideo.mp4
Figures: {"t": seconds into the rendered video, "caption": ...}; placed in body blocks with {fig0}, {fig1}, ...
Exercise kinds: mc (options), tf, short, number, order, code. Every exercise has an answer and a why.
Every number comes from code/d12_attention_sinks/attention_sinks.py (GPT-2 small, real weights, forward pass written by
hand and checked against transformers 4.57.1; torch 2.14.0, CPU) or the same functions with the exercise masks.
"""

LESSON = {
    "series": "How LLMs Work: Deep Dive",
    "label": "Episode 12",
    "title": "Attention Sinks",
    "tagline": "Why so many heads stare at the first token",
    "duration": "2:17",
    "intro": """<p>This lesson answers one question: why do so many attention heads stare at the first token, and why
does it matter? In GPT-2, 39% of all attention goes to the first token, whatever that token is. Softmax must add up to
one, so a head with nothing useful to look at parks its attention on the one token every query can see: an
<b>attention sink</b>, whose values are tiny. Drop it from a sliding-window cache and the model collapses (perplexity 48
→ 2,133); keep the first four tokens and it recovers (43).</p>""",
    "prereq": """<div class="box note"><b>Before you start:</b> Deep Dive episodes 8 (one head, entry by entry) and 11
(sliding windows). Code: <code>code/d12_attention_sinks</code> (GPT-2 small, about 500 MB).</div>""",
}

CONCEPTS = [
    {
        "title": "How much attention goes to the first token",
        "segment": (8, 53),
        "figures": [{"t": 38.0, "caption": "Attention on the first token in all 144 heads: 39% on average, 59 heads above "
                                           "half, little in layers 0–2."},
                    {"t": 52.0, "caption": "Change the first token (line break, zebra, The, comma) or cut mid-sentence: "
                                           "always about 0.39."}],
        "body": [
            """<p>GPT-2 reads 256 tokens of Shakespeare. Averaged over all 144 heads (queries 16–255), <b>39%</b> of the
attention goes to the very first token, and <b>59 heads</b> give it more than half. The first layers barely do it
(0.01–0.16 per layer); layers 5 to 10 give it more than half (0.51 to 0.61).</p>""",
            '<div class="figrow">{fig0}{fig1}</div>',
            """<p>Is the first token special? Replace it with a line break, <i>zebra</i>, <i>The</i> or a comma: 0.394,
0.392, 0.394, 0.393. Start the text mid-sentence: 0.395. It is not the word; it is the <b>position</b>.</p>""",
            """<div class="box key"><b class="t">Key idea</b>Many heads send a large share of their attention to the
first position, regardless of its content.</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "If 39% of attention goes to the first token on average, how much is left for the "
                                    "other 255 tokens together?",
             "answer": "61%.", "why": "Each row of weights sums to 1 (100%).", "key": {'parts': [{'label': None, 'value': 61, 'tol': 0.5, 'unit': '%'}]}},
            {"kind": "tf", "q": "“GPT-2 attends to the first token because it is usually an important word like "
                                "“The”.”",
             "answer": "False.", "why": "With a line break, “zebra” or a comma as the first token, the share stays at "
                                       "about 0.39.", "key": {'value': False}},
        ],
    },
    {
        "title": "Why a sink, and why it is harmless",
        "segment": (53, 80),
        "figures": [{"t": 66.5, "caption": "Softmax must add up to 1: a head with nothing to look at parks its attention on "
                                           "the token every query can see."},
                    {"t": 79.3, "caption": "The first token's value vectors are small (1.42 vs 6.42), so attending to it "
                                           "adds almost nothing."}],
        "body": [
            """<p><b>Softmax must add up to one.</b> A head that has nothing useful to look at for a given token still
has to put its attention somewhere. The first token is the one every query can see (because of the causal mask), so
the model learns to park attention there: an <b>attention sink</b>.</p>""",
            '<div class="figrow">{fig0}{fig1}</div>',
            """<p>Parking there is almost free: the first token's value vectors are small, an average size of
<b>1.42</b> against <b>6.42</b> for the other tokens (layers 2–11). Attending to the sink adds almost nothing to the
output: it works as a “do nothing” option.</p>""",
            """<div class="box key"><b class="t">Key idea</b>An attention sink is a learned way to say “nothing here”
under the constraint that the weights must sum to 1.</div>""",
        ],
        "exercises": [
            {"kind": "short", "q": "Why is the first token, rather than, say, the tenth, the one that becomes the sink?",
             "answer": "Because of the causal mask, the first token is the only one visible to every query, including the "
                       "earliest ones; a sink must be reachable from everywhere.",
             "why": "A token at position 10 is invisible to queries 0–9."},
            {"kind": "mc", "q": "What makes attending to the sink nearly a no-op?",
             "options": ["Its scores are negative", "Its value vectors are small", "It is masked", "It has no position "
                         "embedding"],
             "answer": "B.", "why": "The output is a weighted sum of values; a small value contributes little, whatever "
                                   "its weight.", "key": {'choice': 1}},
        ],
    },
    {
        "title": "Streaming, and four tokens that fix it",
        "segment": (80, 111),
        "figures": [{"t": 98.0, "caption": "A sliding window of 256 that drops the first tokens: perplexity 48 → 2,133. "
                                           "The model collapses."},
                    {"t": 109.5, "caption": "Keep the first four tokens plus the window: perplexity 43."}],
        "body": [
            """<p>To read an endless stream, a model can keep a sliding window: the last 256 tokens, dropping the oldest,
including the first. On 1,024 tokens of Shakespeare (loss measured on positions 512–1,022), GPT-2 with full attention
has a perplexity of <b>48</b>; with the window it jumps to <b>2,133</b>. The heads lose their sink and dump their
attention on whatever tokens remain.</p>""",
            '<div class="figrow">{fig0}{fig1}</div>',
            """<p>The fix: always keep the <b>first four tokens</b>, plus the window. Perplexity: <b>43</b>, even a little
better than full attention on this text. The trick is known as streaming with attention sinks (StreamingLLM). Some newer
designs give heads an explicit way to attend to nothing, such as a learned sink score for each head.</p>""",
            """<div class="box key"><b class="t">Key idea</b>Keep the sink tokens in any cache that evicts old tokens;
four tokens are enough to bring the model back.</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "With a window of 256 plus 4 sink tokens, at most how many keys does a query see?",
             "answer": "260.", "why": "The last 256 tokens plus tokens 0–3 (when they are not already in the window).", "key": {'parts': [{'label': None, 'value': 260, 'tol': 0.5, 'unit': None}]}},
            {"kind": "code", "q": "<b>Try it yourself.</b> In <code>attention_sinks.py</code>, keep only the first token "
                                  "(or the first two) instead of four. What perplexity do you get?",
             "code": """for n in (1, 2):
    mask = window | ((idx[None] < n) & (idx[None] <= idx[:, None]))
    lg, _, _ = forward(ids, mask)
    loss = torch.nn.functional.cross_entropy(lg[0, 512:-1], targets[512:])
    print(n, loss.exp())""",
             "answer": "1 sink: perplexity 47; 2 sinks: 45 (4 sinks: 43).",
             "why": "Even a single sink token prevents the collapse; GPT-2's sink is mostly the very first position."},
        ],
    },
    {
        "title": "The code",
        "segment": (111, 128),
        "figures": [{"t": 126.5, "caption": "One more term in the mask: allowed if in the window, or one of the first four "
                                            "tokens."}],
        "body": [
            """<p>In code, sinks are one more term in the attention mask: a key is allowed if it is in the window, or if
it is one of the first four tokens (and not in the future). The episode's code writes GPT-2's forward pass by hand so
it can take any mask; it matches <code>transformers</code> to within 2 × 10<sup>−4</sup> in the logits.</p>""",
            "{fig0}",
            """<div class="box key"><b class="t">Key idea</b>Masks are the lever for all of part 3: causal, windowed,
sparse, or windowed plus sinks.</div>""",
        ],
        "exercises": [
            {"kind": "order", "q": "Order these masks from fewest to most allowed pairs on 1,024 tokens: <i>full causal · "
                                   "window 256 + 4 sinks · window 256</i>.",
             "answer": "window 256 → window 256 + 4 sinks → full causal.",
             "why": "The sinks add a few pairs to the window; full causal allows every earlier key.", "key": {'items': ['window 256', 'window 256 + 4 sinks', 'full causal']}},
        ],
    },
]
