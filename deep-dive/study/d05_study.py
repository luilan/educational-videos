"""Study guide content for How LLMs Work: Deep Dive, episode 5: RoPE.

Build:  python framework/study_guide.py deep-dive d05 --video deep-dive/media/videos/d05_scene/1080p60/RopeVideo.mp4
Figures: {"t": seconds into the rendered video, "caption": ...}; placed in body blocks with {fig0}, {fig1}, ...
Exercise kinds: mc (options), tf, short, number, order, code. Every exercise has an answer and a why.
Every number comes from code/d05_rope/rope.py (Qwen2.5-0.5B config: head size 64, base 1,000,000; float64; checked
against transformers 4.57.1's Qwen2 rotary embedding) or the same functions run on other positions.
"""

LESSON = {
    "series": "How LLMs Work: Deep Dive",
    "label": "Episode 5",
    "title": "RoPE",
    "tagline": "Rotating vectors to encode order",
    "duration": "2:35",
    "intro": """<p>This lesson answers one question: how do modern models encode position without a learned table?
<b>RoPE</b> (rotary position embedding) <b>rotates</b> each query and key by an angle proportional to its position.
Because a dot product depends only on the angle between two vectors, the attention score then depends only on the
<b>relative distance</b> between tokens. A head's vector is split into pairs that rotate at many different speeds,
from a turn every 6 tokens to one every 4 million.</p>""",
    "prereq": """<div class="box note"><b>Before you start:</b> Deep Dive episode 4 (why attention needs position) and
Foundations F02 and F09 (dot product, sine, cosine and rotations). Code: <code>code/d05_rope</code> (only a config
download).</div>""",
}

CONCEPTS = [
    {
        "title": "Position as a rotation",
        "segment": (8, 42),
        "figures": [{"t": 40.7, "caption": "Rotate the vector by θ per position: the length stays the same; only the "
                                           "direction carries the position."}],
        "body": [
            """<p>GPT-2's learned table stops at 1,024 positions. Most modern models (Llama, Mistral, Qwen) use
<b>rotary position embedding</b>, RoPE, instead. Picture a query as an arrow with two numbers. At position 0, leave it
alone; at position 1, rotate it by a small angle θ; at position 2, by 2θ; and so on. The arrow keeps its length
(|q| = 8.960271 before and after); only its direction carries the position.</p>""",
            "{fig0}",
            """<div class="box key"><b class="t">Key idea</b>RoPE does not add a position vector: it turns the query and
key by position-dependent angles.</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "θ = 30° per position. By how many degrees is a query at position 5 rotated?",
             "answer": "150°.", "why": "5 × 30° = 150°.", "key": {'parts': [{'label': None, 'value': 150, 'tol': 0.5, 'unit': None}]}},
            {"kind": "tf", "q": "“Rotating a vector by RoPE changes its length.”",
             "answer": "False.", "why": "A rotation preserves length: |rope(q, 7)| = |q| = 8.960271 in the code.", "key": {'value': False}},
        ],
    },
    {
        "title": "Why the score depends only on distance",
        "segment": (42, 87),
        "figures": [{"t": 62.1, "caption": "Rotate the query by its position m and the key by n: the angle between them "
                                           "changes by (m − n)θ."},
                    {"t": 86.5, "caption": "Same distance, same score, wherever the pair is: +5.924725 for m − n = 3."}],
        "body": [
            """<p>Attention scores are <b>dot products</b>, and a dot product depends on the <b>angle between</b> two
vectors: q · k = |q| |k| cos(angle). Rotate the query by its position m and the key by its position n; the angle between
them changes by <b>(m − n) θ</b>. So the score depends only on how far apart the two tokens are, not on where they
are.</p>""",
            '<div class="figrow">{fig0}{fig1}</div>',
            """<p>Checked with a random query and key of head size 64: positions (5, 2), (105, 102) and (10,005, 10,002)
all give <b>+5.924725</b>; a distance of 1 gives a different score, +8.349663, again the same everywhere.</p>""",
            """<div class="box key"><b class="t">Key idea</b>rope(q, m) · rope(k, n) is a function of q, k and m − n
only: RoPE turns absolute rotations into relative attention.</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "In the code's example, what is the score for m = 1,000,003 and n = 1,000,000?",
             "answer": "+5.924725.", "why": "m − n = 3, the same distance as (5, 2) (checked by running it).", "key": {'parts': [{'label': None, 'value': 5.924725, 'tol': 5e-07, 'unit': None}]}},
            {"kind": "mc", "q": "Which pair has the same score as (53, 3)?",
             "options": ["(53, 4)", "(1053, 1003)", "(103, 3)", "(3, 53)"],
             "answer": "B.", "why": "Both are 50 apart; the code gives +11.24301 for each.", "key": {'choice': 1}},
            {"kind": "short", "q": "Why does the value vector not need to be rotated?",
             "answer": "Position only needs to affect who attends to whom, which is decided by the query-key scores; the "
                       "values are what is mixed once the weights are set.",
             "why": "RoPE is applied to queries and keys only, inside every attention layer."},
        ],
    },
    {
        "title": "Many speeds, like clock hands",
        "segment": (87, 124),
        "figures": [{"t": 109.6, "caption": "Four of Qwen2.5's 32 pairs after 60 positions: the fast pair has turned many "
                                            "times; the slowest has barely moved."},
                    {"t": 122.7, "caption": "RoPE rotates queries and keys only; nothing is added to the token "
                                            "embeddings."}],
        "body": [
            """<p>A head of 64 numbers makes <b>32 pairs</b>, each rotating at its own speed: frequency =
base<sup>−2i/64</sup> for pair i. In Qwen2.5 (base 1,000,000) the fastest pair makes a full turn every <b>6.3</b> tokens,
pair 8 every 199, pair 16 every 6,283, and the slowest every <b>4,080,185</b> tokens. Fast pairs track nearby order;
slow pairs measure long distances.</p>""",
            '<div class="figrow">{fig0}{fig1}</div>',
            """<p>RoPE is applied inside every attention layer, to the queries and keys only. Nothing is added to the token
embeddings, and the values are not rotated: position only shapes who attends to whom.</p>""",
            """<div class="box key"><b class="t">Key idea</b>Many rotation speeds give the model both fine and coarse
position information, like the second, minute and hour hands of a clock.</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "Pair 16 rotates at base<sup>−32/64</sup> radians per token with base 1,000,000. What "
                                    "is that, and how many tokens per full turn (2π radians)?",
             "answer": "0.001 rad/token; about 6,283 tokens.", "why": "1,000,000<sup>−0.5</sup> = 0.001; 2π / 0.001 ≈ "
                                                                    "6,283.2.", "key": {'parts': [{'label': 'rad/token', 'value': 0.001, 'tol': 0.0005, 'unit': None}, {'label': 'tokens per turn', 'value': 6283, 'tol': 125.66, 'unit': None}]}},
            {"kind": "short", "q": "Why are the fastest pairs alone not enough to encode position?",
             "answer": "They wrap around every few tokens, so positions 0 and 6 look almost the same to them; slow pairs "
                       "are needed to tell far-apart positions apart.",
             "why": "Like a second hand: it repeats every minute, so you also need the minute and hour hands."},
        ],
    },
    {
        "title": "Stretchable, and four lines of code",
        "segment": (124, 155),
        "figures": [{"t": 134.2, "caption": "Slow the rotations down, and the same angles cover a longer text."},
                    {"t": 148.2, "caption": "RoPE in four lines; it matches transformers' Qwen2 implementation to within "
                                            "2 × 10⁻⁷."}],
        "body": [
            """<p>Because position is just an angle, it can be <b>rescaled</b>: slow the rotations down, and the same
angles cover a longer text. That is how models stretch their context window (next episode).</p>""",
            '<div class="figrow">{fig0}{fig1}</div>',
            """<p>In code, RoPE is four lines: compute the angles, split the vector into two halves, and rotate each pair
with a cosine and a sine. The episode's version matches the one in Hugging Face's Qwen2 code to within two
ten-millionths (float rounding).</p>""",
            """<div class="box key"><b class="t">Key idea</b>RoPE has no learned parameters: it is pure geometry, which is
why it can be adjusted after training.</div>""",
        ],
        "exercises": [
            {"kind": "tf", "q": "“RoPE adds a learned table of parameters to the model, like GPT-2's position "
                                "embeddings.”",
             "answer": "False.", "why": "The rotation angles are computed from a formula (positions × fixed frequencies); "
                                       "nothing is learned.", "key": {'value': False}},
            {"kind": "code", "q": "<b>Try it yourself.</b> Using <code>rope</code>, <code>q</code> and <code>k</code> "
                                  "from <code>rope.py</code>, print the score for each pair below. Which scores are equal, "
                                  "and why?",
             "code": """for m, n in [(7, 4), (53, 3), (1053, 1003), (1_000_003, 1_000_000)]:
    print(m, n, torch.dot(rope(q, m), rope(k, n)).item())""",
             "answer": "(7, 4) and (1,000,003, 1,000,000) both give +5.924725 (distance 3); (53, 3) and (1053, 1003) both "
                       "give +11.24301 (distance 50).",
             "why": "Only m − n matters, even a million positions away."},
        ],
    },
]
