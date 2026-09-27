"""Study guide content for How LLMs Work, episode 4: Where Am I? Position.

Build:  python framework/study_guide.py how-llms-work v04
Figures: {"t": seconds into the rendered video, "caption": ...}; placed in body blocks with {fig0}, {fig1}, ...
Exercise kinds: mc (options), tf, short, number, order, code. Every exercise has an answer and a why.
"""

LESSON = {
    "series": "How LLMs Work",
    "label": "Episode 4",
    "title": "Where Am I? Position",
    "tagline": "How a model knows the order of words",
    "duration": "2:26",
    "intro": """<p>Embeddings tell the model <b>what</b> each token is, but not <b>where</b> it is: <i>“The cat sat on
the mat”</i> and <i>“The mat sat on the cat”</i> get exactly the same vectors. This lesson is about putting the
<b>order into the vectors themselves</b>: first two simple ideas, then the sine waves of the original transformer, and
finally the rotations (RoPE) used by most modern models.</p>""",
    "prereq": """<div class="box note"><b>Before you start:</b> this lesson builds on episode 3 (embeddings: one vector
per token, <code>x = E[ids]</code>). The maths is adding vectors, sine and cosine waves, and rotation: Foundations F01
(Vectors) and F09 (Waves and Rotations) cover it. Concept 6 reads a short NumPy function; F11 (NumPy in Three Minutes)
helps.</div>""",
}

CONCEPTS = [
    {
        "title": "Same tokens, different meaning",
        "segment": (8, 31),
        "figures": [{"t": 16.8, "caption": "Same tokens, different meaning, but identical vectors."},
                    {"t": 30.6, "caption": "Attention compares every token with every other; nothing says which came "
                                           "first."}],
        "body": [
            """<p><i>“The cat sat on the mat”</i> and <i>“The mat sat on the cat”</i> contain the same tokens but mean
completely different things. So far, our token vectors can't tell them apart: a token gets the same embedding wherever
it appears.</p>""",
            """<p>And it matters. <b>Attention</b>, the heart of the transformer (next episode), compares every token
with every other token, and nothing in that comparison says which one came first. So we have to <b>put the order into
the vectors themselves</b>.</p>""",
            '<div class="figrow">{fig0}{fig1}</div>',
            """<div class="box key"><b class="t">Key idea</b>Without position information, a sentence is just a
<b>bag of tokens</b>: word order is invisible. Each token vector must also carry <b>where</b> the token is.</div>""",
        ],
        "exercises": [
            {"kind": "tf", "q": "“Embeddings alone already tell <i>“The cat sat on the mat”</i> apart from <i>“The mat "
                                "sat on the cat”</i>.”",
             "answer": "False.", "why": "Both sentences contain the same tokens, so they get exactly the same "
                                        "vectors."},
            {"kind": "mc", "q": "Which pair of sentences would look identical to a model with no position "
                                "information?",
             "options": ["<i>The dog bit the man</i> / <i>The man bit the dog</i>",
                         "<i>The dog bit the man</i> / <i>A dog bit a man</i>",
                         "<i>The dog bit</i> / <i>The dog bites</i>", "<i>I like tea</i> / <i>I like coffee</i>"],
             "answer": "A.", "why": "Same tokens, different order. Every other pair differs in at least one token."},
            {"kind": "short", "q": "Why can't attention work out the word order by itself?",
             "answer": "Nothing in its comparisons says which token came first.",
             "why": "It compares every token with every other token; unless the vectors carry position, order is "
                    "lost."},
        ],
    },
    {
        "title": "Idea 1: add the position number",
        "segment": (31, 43),
        "figures": [{"t": 42.9, "caption": "At token 5000, the position (5000) drowns out the meaning (0.83) "
                                           "(illustrative values)."}],
        "body": [
            """<p>The most obvious idea: just add the position number to the token's vector: 1 for the first token,
2 for the second, and so on. But these numbers <b>grow without limit</b>. By token 5000, a meaning of (0.21, −0.47,
0.83) becomes (5000.21, 4999.53, 5000.83): the position <b>drowns out the meaning</b>.</p>""",
            "{fig0}",
            """<div class="box key"><b class="t">Key idea</b>A position signal must <b>stay small</b>, on the same scale
as the meaning. Raw position numbers grow without limit and swamp it.</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "The meaning vector is (0.21, −0.47, 0.83). With idea 1, what is the input vector "
                                    "at position 3?",
             "answer": "(3.21, 2.53, 3.83).", "why": "Add 3 to every number: −0.47 + 3 = 2.53."},
            {"kind": "mc", "q": "What goes wrong with idea 1 in a long text?",
             "options": ["Two positions get the same number", "The position numbers become huge and drown out the "
                         "meaning", "It needs a table with one row per position", "It only works for even positions"],
             "answer": "B.", "why": "By token 5000 the numbers are around 5000, while the meaning is below 1."},
            {"kind": "tf", "q": "“At position 5000, <i>cat</i> (0.21, −0.47, 0.83) and a word with meaning (0.25, "
                                "−0.40, 0.80) give input vectors that are almost identical.”",
             "answer": "True.", "why": "(5000.21, 4999.53, 5000.83) vs (5000.25, 4999.60, 5000.80): the small "
                                       "differences in meaning are lost next to 5000."},
        ],
    },
    {
        "title": "Idea 2: learn a vector per position",
        "segment": (43, 58),
        "figures": [{"t": 57.5, "caption": "Row 2 of the position table is added to cat's embedding."}],
        "body": [
            """<p>GPT-2 <b>learns</b> positions. It keeps a second table, the <b>position embeddings</b>, with one
vector for each position, and adds that vector to the token's embedding. For <i>cat</i> at position 2: (0.35, −0.12,
0.61, …) + (0.54, −0.59, 0.67, …) = (0.89, −0.71, 1.28, …).</p>""",
            "{fig0}",
            """<div class="box key"><b class="t">Key idea</b>Learned positions: <b>input = token embedding + position
embedding</b>. Simple, but the table has a fixed size (GPT-2: 1,024 positions, 0–1023), and the model can't handle
positions it never saw in training.</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "Using the table in the figure, compute the first three numbers of <i>cat</i> at "
                                    "position <b>0</b>. (cat's embedding starts 0.35, −0.12, 0.61.)",
             "answer": "(1.15, −0.10, 1.47).", "why": "Row 0 is (0.80, 0.02, 0.86): 0.35 + 0.80, −0.12 + 0.02, "
                                                      "0.61 + 0.86."},
            {"kind": "mc", "q": "GPT-2 is given a text of 1,500 tokens. What is the problem?",
             "options": ["Tokens after position 1023 have no position vector", "The position numbers get too big",
                         "Its vocabulary is too small", "There is no problem"],
             "answer": "A.", "why": "The table has 1,024 rows (positions 0–1023), all learned in training."},
            {"kind": "number", "q": "GPT-2's position table has 1,024 rows of 768 numbers. How many numbers must be "
                                    "learned?",
             "answer": "786,432.", "why": "1,024 × 768. Like the embedding matrix, every one of them is learned."},
            {"kind": "tf", "q": "“In GPT-2, the position vectors are computed from a fixed formula.”",
             "answer": "False.", "why": "They are learned in training, just like the token embeddings. (The formula "
                                        "is the next idea.)"},
        ],
    },
    {
        "title": "Waves at many speeds, like a clock",
        "segment": (58, 89),
        "figures": [{"t": 72.5, "caption": "Each pair: a sine and a cosine at its own frequency. Read off the waves at "
                                           "a position (here 6)."},
                    {"t": 88.5, "caption": "Like a clock: fast hands tell neighbours apart, slow hands track the long "
                                           "run."}],
        "body": [
            """<p>The original transformer paper used <b>waves</b>. Dimensions come in pairs: each pair follows a
<b>sine</b> and a <b>cosine</b> wave, and every pair has its <b>own frequency</b>. To encode a position, read off where
each wave is at that point. Sines and cosines always stay between −1 and 1, so unlike idea 1 the values never grow.</p>""",
            '<div class="figrow">{fig0}{fig1}</div>',
            """<p>It works like a clock. The second hand moves fast, the minute hand slower, the hour hand slowest.
Each hand alone is ambiguous, but read together they give the exact time. Likewise, <b>fast waves tell neighbours
apart</b>, and <b>slow waves track where you are in the long run</b>.</p>""",
            """<div class="box key"><b class="t">Key idea</b><b>Sinusoidal encoding</b>: position → sines and cosines at
many frequencies. Every value is between −1 and 1, and all the waves together pin down the exact position.</div>""",
        ],
        "exercises": [
            {"kind": "mc", "q": "Why is one fast wave not enough to encode position?",
             "options": ["Its values are too large", "It repeats quickly, so far-apart positions can look the same",
                         "It can't tell neighbours apart", "It needs a learned table"],
             "answer": "B.", "why": "Like the second hand, which is back in the same place every minute."},
            {"kind": "short", "q": "A clock's second hand points at 12 and its minute hand at 3. Give two different "
                                   "times this could be. Which hand settles it?",
             "answer": "E.g. 1:15:00 or 4:15:00; the hour hand.",
             "why": "The fast hands only fix the time within the hour. The slowest hand tells which hour."},
            {"kind": "tf", "q": "“In sinusoidal encoding, the values for position 5000 are much larger than for "
                                "position 5.”",
             "answer": "False.", "why": "Every value is a sine or cosine, always between −1 and 1."},
        ],
    },
    {
        "title": "Position is added to meaning",
        "segment": (89, 101),
        "figures": [{"t": 100.8, "caption": "cat at 2 and cat at 6: the same meaning, in a different place. "
                                            "PE values are real (d = 4)."}],
        "body": [
            """<p>The position vector is simply <b>added</b> to the token's embedding. With d = 4 numbers per token, the
real sinusoidal values are PE(2) = (0.909, −0.416, 0.020, 1.000) and PE(6) = (−0.279, 0.960, 0.060, 0.998). So
<i>cat</i> at position 2 and <i>cat</i> at position 6 start out as different vectors: the same meaning, in a
different place.</p>""",
            "{fig0}",
            """<div class="box key"><b class="t">Key idea</b><b>input vector = embedding(token) + PE(position)</b>: one
vector that carries both what the token is and where it is.</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "For d = 4, PE(0) = (sin 0, cos 0, sin 0, cos 0). Write PE(0), then compute "
                                    "<i>cat</i> at position 0, using cat's embedding (0.350, −0.120, 0.610, 0.080).",
             "lines": 2,
             "answer": "PE(0) = (0, 1, 0, 1); cat at 0 = (0.350, 0.880, 0.610, 1.080).",
             "why": "sin 0 = 0 and cos 0 = 1, so only the cosine dimensions change: −0.120 + 1 = 0.880, "
                    "0.080 + 1 = 1.080."},
            {"kind": "tf", "q": "“Adding position vectors changes the rows of the embedding matrix <code>E</code>.”",
             "answer": "False.", "why": "<code>E</code> is unchanged. The sum is computed for each input: "
                                        "<code>x = E[ids] + PE</code>."},
            {"kind": "mc", "q": "<i>cat</i> appears at positions 2 and 6 of a sentence. Which statement is true?",
             "options": ["Both get exactly the same input vector", "Their input vectors differ, because different "
                         "position vectors were added", "Only the first <i>cat</i> gets a position vector",
                         "The second <i>cat</i> is removed"],
             "answer": "B.", "why": "Same embedding, different PE: (1.259, −0.536, 0.630, 1.080) vs (0.071, 0.840, "
                                   "0.670, 1.078)."},
        ],
    },
    {
        "title": "Sinusoidal encoding in code",
        "segment": (101, 114),
        "figures": [{"t": 113.4, "size": "small", "caption": "The whole encoding in a few lines, added to the "
                                                             "embeddings on the last line."}],
        "body": [
            """<p>In code: for each position and each pair of dimensions, compute an angle. Put the <b>sine in the even
dimensions</b> and the <b>cosine in the odd ones</b>, then add the result to the embeddings.</p>""",
            "{fig0}",
            """<pre class="code">def positional_encoding(n_pos, d_model):
    pos = np.arange(n_pos)[:, None]           # 0, 1, 2, ...
    i = np.arange(0, d_model, 2)[None, :]     # even dimensions
    angle = pos / 10000 ** (i / d_model)
    pe = np.zeros((n_pos, d_model))
    pe[:, 0::2] = np.sin(angle)               # even dims: sine
    pe[:, 1::2] = np.cos(angle)               # odd dims: cosine
    return pe

x = E[ids] + positional_encoding(len(ids), d_model)</pre>""",
            """<div class="box key"><b class="t">Key idea</b><code>angle = pos / 10000 ** (i / d_model)</code>: the
first pair (i = 0) uses the angle pos itself, the fastest wave; each later pair divides by a bigger number, so its
wave is slower.</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "With <code>d_model = 4</code>, the pairs use i = 0 and i = 2. What are the two "
                                    "angles at position 2?",
             "answer": "2 and 0.02.", "why": "10000<sup>0/4</sup> = 1 and 10000<sup>2/4</sup> = 100, so PE(2) = "
                                             "(sin 2, cos 2, sin 0.02, cos 0.02) = (0.909, −0.416, 0.020, 1.000)."},
            {"kind": "number", "q": "A sine wave repeats each time its angle grows by 2π ≈ 6.28. With "
                                    "<code>d_model = 4</code>, about how many positions does it take for pair 1 to "
                                    "repeat? And pair 2?",
             "answer": "About 6.3 and about 628.", "why": "Pair 1's angle is pos; pair 2's is pos / 100, so it needs "
                                                          "100 times as many positions: the fast and the slow hand."},
            {"kind": "code", "q": "<b>Try it yourself.</b> Copy <code>positional_encoding</code> from above (with "
                                  "<code>import numpy as np</code>) and run the lines below. (a) What are rows 0 and 2 "
                                  "of <code>pe</code>? (b) What is the largest absolute value anywhere in "
                                  "<code>big</code>, which covers 10,000 positions? (c) Compute the three dot products. "
                                  "What do you notice?",
             "code": """pe = positional_encoding(8, 4)
print(np.round(pe[0], 3), np.round(pe[2], 3))        # (a)

big = positional_encoding(10000, 16)
print(np.abs(big).max())                             # (b)

p = positional_encoding(50, 16)
print(p[10] @ p[11], p[30] @ p[31], p[10] @ p[40])   # (c)""",
             "answer": "(a) [0, 1, 0, 1] and [0.909, −0.416, 0.02, 1.0] · (b) 1.0 · (c) ≈ 7.49, 7.49, 2.70",
             "why": """(a) Row 0 is sin 0 = 0 and cos 0 = 1 in every pair; row 2 is the PE(2) from the video.
(b) Even 10,000 positions in, nothing exceeds 1: no drowning, unlike idea 1. (c) Neighbouring positions score high,
far-apart ones lower, and the score depends only on the <b>distance</b> (10→11 and 30→31 are equal). Each pair
contributes sin·sin + cos·cos = cos(angle difference). RoPE, next, builds this “only the gap matters” idea right into
attention."""},
        ],
    },
    {
        "title": "RoPE: rotate instead of add",
        "segment": (114, 134),
        "figures": [{"t": 133.7, "size": "small", "caption": "Shift both by 3: the gap stays 3θ."}],
        "body": [
            """<p>Most modern models, like Llama, use <b>rotary position embedding (RoPE)</b>. Instead of adding a
vector, it <b>rotates</b> pairs of numbers by an angle that grows with position: angle = position · θ. A token at
position 2 is turned by 2θ, one at position 5 by 5θ.</p>""",
            "{fig0}",
            """<p>RoPE is applied to the vectors attention compares (queries and keys, episode 6). There only the
<b>difference</b> of the angles matters: 5θ − 2θ = 3θ, and at positions 5 and 8 it is still 3θ.</p>""",
            """<div class="box key"><b class="t">Key idea</b>RoPE <b>rotates</b> pairs of numbers by position · θ instead
of adding a vector. A comparison of two tokens then depends only on their <b>distance</b>.</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "With RoPE, token A is at position 4 and token B at position 10. By what angles "
                                    "are they rotated, and what angle gap does the comparison see?",
             "answer": "4θ and 10θ; a gap of 6θ.", "why": "Angle = position · θ, and the gap is (10 − 4)θ."},
            {"kind": "number", "q": "Take θ = 10°. Two tokens sit at positions 3 and 7. Later in the text, the same "
                                    "two tokens sit at positions 103 and 107. What gap does the comparison see each "
                                    "time?",
             "answer": "40° both times.", "why": "30° vs 70°, and 1030° vs 1070°: the distance is 4 positions either "
                                                 "way."},
            {"kind": "mc", "q": "How is RoPE different from sinusoidal encoding?",
             "options": ["It adds a larger position vector", "It rotates pairs of numbers by a position-dependent angle "
                         "instead of adding a vector", "It uses a learned table, like GPT-2",
                         "It ignores position"],
             "answer": "B.", "why": "Sinusoidal encoding adds PE(pos) to the embedding; RoPE turns pairs of numbers "
                                   "in the vectors attention compares."},
            {"kind": "tf", "q": "“With RoPE, what a comparison sees depends on the tokens' exact positions, not just "
                                "on how far apart they are.”",
             "answer": "False.", "why": "Only the difference of the angles matters, and that depends only on the "
                                        "distance."},
        ],
    },
]
