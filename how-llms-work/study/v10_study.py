"""Study guide content for How LLMs Work, episode 10: From Vectors Back to Words.

Build:  python framework/study_guide.py how-llms-work v10
Figures: {"t": seconds into the rendered video, "caption": ...}; placed in body blocks with {fig0}, {fig1}, ...
Exercise kinds: mc (options), tf, short, number, order, code. Every exercise has an answer and a why.
"""

LESSON = {
    "series": "How LLMs Work",
    "label": "Episode 10",
    "title": "From Vectors Back to Words",
    "tagline": "Logits, softmax, and choosing the next token",
    "duration": "2:14",
    "intro": """<p>After the last block, every token is still a vector. This lesson turns the last one back into a word:
the <b>unembedding matrix</b> gives <b>one score (logit) per token</b>, <b>softmax</b> turns the scores into
probabilities, and a <b>sampling rule</b> (greedy, temperature, top-k, top-p) picks the next token. That closes the
loop from episode 1.</p>""",
    "prereq": """<div class="box note"><b>Before you start:</b> this lesson builds on episode 1 (predict, pick, append,
repeat), episode 3 (the embedding matrix) and episode 9 (the stack of blocks). For the maths, Foundations F02 (The Dot
Product), F03 (Matrices), F05 (Exponentials and Logarithms), F06 (Probability and Sampling) and F07 (Softmax,
Properly) cover everything used here.</div>""",
}

CODE = """def next_token(h, W_unembed, temperature=0.8, top_k=50):
    logits = h @ W_unembed / temperature    # a score per token
    top = np.argsort(logits)[-top_k:]       # keep the k best
    p = np.exp(logits[top] - logits[top].max())
    p /= p.sum()                            # softmax
    return np.random.choice(top, p=p)       # sample one"""

CONCEPTS = [
    {
        "title": "From the last vector to logits",
        "segment": (8, 39),
        "figures": [{"t": 18.3, "caption": "Only the last vector, on “the”, is needed."},
                    {"t": 34.2, "caption": "One column, and one logit, per vocabulary token."}],
        "body": [
            """<p>After the last transformer block there is one vector per token. To predict what comes after
<i>“The cat sat on the”</i>, we only need <b>the last one</b>: the vector sitting on the word <i>the</i>.</p>""",
            '<div class="figrow">{fig0}{fig1}</div>',
            """<p>It goes through <b>one last layer norm</b>, then is multiplied by the <b>unembedding matrix</b>, which has
<b>one column for every token</b> in the vocabulary. The result is one score per token, <b>50,257 numbers</b> in GPT-2,
called <b>logits</b>. In GPT-2 this matrix is the <b>embedding matrix again</b>, reused (“tied weights”).</p>""",
            """<div class="box key"><b class="t">Key idea</b>Last vector → final layer norm → × unembedding matrix →
<b>one logit per vocabulary token</b>.</div>""",
        ],
        "exercises": [
            {"kind": "mc", "q": "Which vector is used to predict the word after <i>“The cat sat on the”</i>?",
             "options": ["The vector on the first word, <i>The</i>", "The average of all five vectors",
                         "The vector on the last token, <i>the</i>", "The vector on <i>cat</i>, the most important word"],
             "answer": "C.", "why": "The prediction for the next token is read from the last position's vector."},
            {"kind": "number", "q": "GPT-2 small has vectors of 768 numbers and a vocabulary of 50,257 tokens. "
                                    "(a) How many logits does one prediction produce? (b) How many numbers are in the "
                                    "unembedding matrix?",
             "answer": "(a) 50,257. (b) 768 × 50,257 = 38,597,376.",
             "why": "One column of 768 numbers per vocabulary token. With tied weights, these are the same numbers as "
                    "the embedding matrix."},
            {"kind": "tf", "q": "“In GPT-2, the unembedding matrix is learned separately from the embedding matrix.”",
             "answer": "False.", "why": "It is the embedding matrix again (transposed), reused: tied weights."},
            {"kind": "order", "q": "Put in order: <i>unembedding matrix · last transformer block · logits · final "
                                   "layer norm</i>.",
             "answer": "last transformer block → final layer norm → unembedding matrix → logits.",
             "why": "The last vector is normalized once more, then multiplied by the matrix to give the scores."},
        ],
    },
    {
        "title": "Each logit is a dot product",
        "segment": (40, 48),
        "figures": [{"t": 48.0, "caption": "In a 2-D picture: mat lines up best with h and gets the highest score; "
                                           "banana points away and gets a negative one."}],
        "body": [
            """<p>Multiplying by the matrix means taking a <b>dot product</b> with each token's column:
<b>logit = h · w<sub>token</sub></b>. It measures how well the final vector h <b>lines up</b> with that token's
direction. The better the match, the higher the score.</p>""",
            "{fig0}",
            """<p>In the video's 2-D picture, h = (2, 1). <i>mat</i> = (1.1, 0.8) scores 2·1.1 + 1·0.8 = <b>3.0</b>,
<i>floor</i> = (0.6, 1.2) scores <b>2.4</b>, and <i>banana</i> = (−0.8, −0.4), pointing the other way, scores
<b>−2.0</b>.</p>""",
            """<div class="box key"><b class="t">Key idea</b>A logit is a dot product: <b>lined up → high</b>, at right
angles → 0, <b>opposite → negative</b>.</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "With h = (2, 1), compute the logits for <i>sofa</i>, w = (0.5, 0.9), and for "
                                    "<i>bed</i>, w = (0.4, 0.8).",
             "answer": "1.9 and 1.6.",
             "why": "2·0.5 + 1·0.9 = 1.9 and 2·0.4 + 1·0.8 = 1.6: the values of the toy vocabulary in concept 3."},
            {"kind": "short", "q": "A new token <i>rug</i> has w = (1.2, 0.7). Would it beat <i>mat</i> (3.0)?",
             "answer": "Yes: its logit is 3.1.", "why": "2·1.2 + 1·0.7 = 3.1 > 3.0: it lines up with h even better."},
            {"kind": "mc", "q": "With h = (2, 1), which token direction gives the most negative logit?",
             "options": ["(2, 1)", "(1, −2)", "(−2, −1)", "(0, 0)"],
             "answer": "C.", "why": "(−2, −1) points exactly opposite to h: −5. (1, −2) is at right angles (0), (0, 0) "
                                   "gives 0, and (2, 1) gives +5."},
            {"kind": "tf", "q": "“A logit is always a number between 0 and 1.”",
             "answer": "False.", "why": "Logits can be any number, positive or negative (<i>banana</i>: −2.0). Softmax "
                                        "turns them into probabilities."},
        ],
    },
    {
        "title": "Softmax: scores to probabilities",
        "segment": (49, 62),
        "figures": [{"t": 62.3, "caption": "The toy vocabulary: logits on the left, probabilities after softmax on the "
                                           "right. They add up to 100 %."}],
        "body": [
            """<p>Logits can be any number. <b>Softmax</b> turns them into probabilities: <b>exponentiate</b> each one
(e<sup>logit</sup> is always positive), then <b>divide by the total</b>. Now every token has a probability, and they
add up to 1.</p>""",
            "{fig0}",
            """<p>In the toy vocabulary the eight values e<sup>logit</sup> add up to about 48.72, and e<sup>3.0</sup> ≈ 20.09,
so <i>mat</i> gets 20.09 / 48.72 ≈ <b>41.2 %</b>. Even <i>banana</i> (logit −2.0) gets a small positive share:
0.3 %.</p>""",
            """<div class="box key"><b class="t">Key idea</b>softmax: p = e<sup>logit</sup> / (sum of all
e<sup>logit</sup>). A bigger logit gives a bigger probability; every probability is positive; they add up to 1.</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "Compute the softmax of the logits [2, 1, 0], as percentages (e² ≈ 7.39, "
                                    "e¹ ≈ 2.72, e⁰ = 1).",
             "answer": "≈ 66.5 %, 24.5 %, 9.0 %.", "why": "The total is 11.11: 7.39 / 11.11, 2.72 / 11.11 and "
                                                          "1 / 11.11."},
            {"kind": "number", "q": "In the toy vocabulary, e<sup>2.4</sup> ≈ 11.02. Using the total 48.72, what is "
                                    "<i>floor</i>'s probability?",
             "answer": "≈ 22.6 %.", "why": "11.02 / 48.72 ≈ 0.226, as in the chart."},
            {"kind": "short", "q": "You add 10 to every logit. What happens to the probabilities?",
             "answer": "Nothing: they stay exactly the same.",
             "why": "e<sup>logit + 10</sup> = e<sup>10</sup> · e<sup>logit</sup>. Every term and the total get the same "
                    "factor, which cancels in the division: only the differences between logits matter."},
            {"kind": "tf", "q": "“A token with a negative logit gets a negative probability.”",
             "answer": "False.", "why": "e<sup>x</sup> is positive for every x: <i>banana</i>'s −2.0 becomes 0.3 %."},
        ],
    },
    {
        "title": "Greedy picking and temperature",
        "segment": (62, 86),
        "figures": [{"t": 81.0, "caption": "Low temperature (T = 0.5): sharper, mat takes 66 %."},
                    {"t": 85.7, "caption": "High temperature (T = 2): flatter, rare words get a chance."}],
        "body": [
            """<p>The simplest choice is <b>greedy</b>: always take the most likely token. But that tends to be
<b>repetitive and dull</b>. Instead we <b>sample</b>, and control how adventurous that is with the <b>temperature</b>
T: divide the logits by T before the softmax, <code>softmax(logits / T)</code>.</p>""",
            '<div class="figrow">{fig0}{fig1}</div>',
            """<p>A <b>low</b> T <b>sharpens</b> the distribution toward the top choice (<i>mat</i>: 41.2 % → 66.3 % at
T = 0.5). A <b>high</b> T <b>flattens</b> it, so rarer words get a chance (<i>banana</i>: 0.3 % → 2.2 % at T = 2).</p>""",
            """<div class="box key"><b class="t">Key idea</b><b>T &lt; 1</b>: sharper, closer to greedy. <b>T &gt; 1</b>:
flatter, more variety. T = 1: the model's own probabilities.</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "At T = 0.5, what do the logits of <i>mat</i> (3.0) and <i>floor</i> (2.4) become? "
                                    "How does the gap between them change?",
             "answer": "6.0 and 4.8: the gap doubles from 0.6 to 1.2.",
             "why": "Dividing by 0.5 doubles every logit, so the differences grow and the top choice pulls ahead."},
            {"kind": "number", "q": "At T = 1, <i>mat</i> (41.2 %) is about 1.8 times as likely as <i>floor</i> "
                                    "(22.6 %). Using the two charts above, how many times as likely is it at T = 0.5, "
                                    "and at T = 2?",
             "answer": "About 3.3 and about 1.35.",
             "why": "66.3 / 20.0 ≈ 3.3 and 26.4 / 19.5 ≈ 1.35. (Exactly: e<sup>1.2</sup> and e<sup>0.3</sup>, the gap "
                    "divided by T.)"},
            {"kind": "tf", "q": "“A very low temperature, such as T = 0.1, makes sampling behave almost like greedy "
                                "picking.”",
             "answer": "True.", "why": "At T = 0.1, <i>mat</i> gets 99.75 %, so it is picked almost every time."},
            {"kind": "short", "q": "You want a story generator to be more surprising. Should you raise or lower the "
                                   "temperature?",
             "answer": "Raise it (T &gt; 1).", "why": "A higher T flattens the distribution, so less likely words are "
                                                  "picked more often. A lower T (or greedy) makes the output more "
                                                  "predictable."},
        ],
    },
    {
        "title": "Top-k and top-p",
        "segment": (86, 99),
        "figures": [{"t": 91.5, "caption": "Top-k (k = 3): keep the three most likely tokens and rescale."},
                    {"t": 97.6, "caption": "Top-p (p = 0.9): keep tokens until the running total reaches 90 %."}],
        "body": [
            """<p>Two more tricks keep sampling sensible by cutting unlikely tokens before we sample. <b>Top-k</b> keeps
only the <b>k most likely</b> tokens. <b>Top-p</b> keeps the <b>smallest set</b> of top tokens whose probabilities
<b>add up to p</b> (at least p), say 90 %. The kept probabilities are rescaled to add up to 1, then we sample.</p>""",
            '<div class="figrow">{fig0}{fig1}</div>',
            """<p>With k = 3, <i>mat</i>, <i>floor</i> and <i>sofa</i> become 53.1 %, 29.2 % and 17.7 %. With p = 0.9, the
running total first reaches 90 % at <i>roof</i> (93.9 %), so five tokens stay.</p>""",
            """<div class="box key"><b class="t">Key idea</b>Top-k keeps a <b>fixed number</b> of tokens; top-p keeps
<b>as many as needed</b> to cover p. Cut, rescale, sample.</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "Use top-k with k = 2 on the T = 1 chart (<i>mat</i> 41.2 %, <i>floor</i> 22.6 %). "
                                    "What are the new probabilities?",
             "answer": "≈ 64.6 % and 35.4 %.",
             "why": "41.2 / 63.8 and 22.6 / 63.8: the kept tokens are rescaled to add up to 100 %."},
            {"kind": "number", "q": "The running totals at T = 1 are 41.2, 63.9, 77.6, 87.7, 93.9 and 98.5 %. How many "
                                    "tokens does top-p keep for p = 0.5? And for p = 0.8?",
             "answer": "2 and 4.", "why": "The smallest set that reaches p: 63.9 % ≥ 50 % after two tokens, "
                                          "87.7 % ≥ 80 % after four."},
            {"kind": "short", "q": "At T = 2 the chart is flatter: the running totals are 26.4, 45.9, 61.2, 74.3, 84.5, "
                                   "93.2 and 97.8 %. How many tokens does top-p (p = 0.9) keep now? And top-k (k = 3)?",
             "answer": "Six; still three.",
             "why": "Top-p adapts to the shape of the distribution (flatter means more tokens); top-k always keeps "
                    "exactly k."},
            {"kind": "tf", "q": "“After the cut, the kept probabilities are rescaled so they add up to 1 again.”",
             "answer": "True.", "why": "53.1 + 29.2 + 17.7 = 100 %: each kept token grows in proportion."},
        ],
    },
    {
        "title": "One sentence, many lessons",
        "segment": (100, 112),
        "figures": [{"t": 111.3, "caption": "During training, every position predicts its own next token, and each "
                                            "prediction is checked against the real one."}],
        "body": [
            """<p>During <b>training</b> the model does not predict just one next token. <b>Every position</b> predicts its
own next token <b>at the same time</b>, and each prediction is checked against the <b>real</b> next token in the
text.</p>""",
            "{fig0}",
            """<p>So one sentence gives the model <b>many lessons at once</b>: <i>“The cat sat on the mat”</i> (6 tokens)
gives 5 predictions to check. Episode 11 shows how the checking turns into learning.</p>""",
            """<div class="box key"><b class="t">Key idea</b>In training, a text of n tokens gives <b>n − 1</b>
next-token predictions at the same time, each checked against the real next token.</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "A training text has 10 tokens. How many next-token predictions can be checked?",
             "answer": "9.", "why": "Every position except the last has a real next token to be checked against."},
            {"kind": "short", "q": "In <i>“The cat sat on the mat”</i>, what should the position on <i>sat</i> "
                                   "predict? And the position on the second <i>the</i>?",
             "answer": "<i>on</i>; <i>mat</i>.", "why": "Each position's target is simply the next token in the text."},
            {"kind": "tf", "q": "“During training, the model generates the sentence one word at a time, and only the "
                                "final word is checked.”",
             "answer": "False.", "why": "All positions predict at the same time, and every prediction is checked."},
        ],
    },
    {
        "title": "Choosing the next token, in code",
        "segment": (112, 132),
        "figures": [{"t": 125.8, "caption": "The whole forward pass, from tokens to a sampled next token."}],
        "body": [
            """<p>In code: compute the logits and divide by the temperature, keep the top k, softmax, and sample one
token.</p>""",
            f'<pre class="code">{CODE}</pre>',
            """<p>That closes the loop from episode 1: <b>predict, pick, append, repeat</b>. We have now seen the
<b>whole forward pass</b>. But all those matrices started out as <b>random noise</b>. How do they learn? Episode 11:
training.</p>""",
            "{fig0}",
            """<div class="box key"><b class="t">Key idea</b><code>next_token</code>: logits / T → keep the top k →
softmax → sample. Put it inside the episode 1 loop and the model writes text.</div>""",
        ],
        "exercises": [
            {"kind": "order", "q": "Put the whole forward pass in order: <i>sample · blocks × N · tokens · unembed · "
                                   "+ position · embed</i>.",
             "answer": "tokens → embed → + position → blocks × N → unembed → sample.",
             "why": "Episodes 2 → 3 → 4 → 5–9 → 10."},
            {"kind": "mc", "q": "The line <code>p = np.exp(logits[top] - logits[top].max())</code> subtracts the "
                                "largest logit before exponentiating. Why is that safe?",
             "options": ["It changes the probabilities in favour of rarer tokens",
                         "Subtracting the same number from every logit does not change the softmax, and it keeps "
                         "e<sup>x</sup> from getting huge",
                         "It is a bug, fixed by the next line", "It is how the temperature is applied"],
             "answer": "B.", "why": "Concept 3, exercise 3.3: only differences between logits matter. The largest "
                                   "value becomes e<sup>0</sup> = 1, so nothing overflows."},
            {"kind": "code", "q": "<b>Try it yourself.</b> Here is a version of <code>next_token</code> that takes the "
                                  "logits directly (and also returns <code>top</code> and <code>p</code>), with the "
                                  "toy vocabulary. (a) What does the <code>print</code> show? (b) Call "
                                  "<code>next_token(logits)</code> 10,000 times and count the words. About what share "
                                  "is <i>mat</i>? Does <i>banana</i> ever come out? (c) Try "
                                  "<code>temperature=0.1</code>: what is <code>p</code> now, and which strategy from "
                                  "concept 4 does this approach?",
             "code": """import numpy as np

WORDS  = ["mat", "floor", "sofa", "bed", "roof", "table", "car", "banana"]
logits = np.array([3.0, 2.4, 1.9, 1.6, 1.1, 0.8, -0.5, -2.0])

def next_token(logits, temperature=1.0, top_k=3):
    logits = logits / temperature
    top = np.argsort(logits)[-top_k:]           # keep the k best
    p = np.exp(logits[top] - logits[top].max())
    p /= p.sum()                                # softmax
    return np.random.choice(top, p=p), top, p

tok, top, p = next_token(logits)
print(top, np.round(p, 3))""",
             "answer": "(a) [2 1 0] [0.177 0.292 0.531] · (b) about 53 % mat; banana never · (c) p ≈ [0. 0.002 0.998]: "
                       "almost greedy",
             "why": """(a) <code>argsort</code> sorts from smallest to largest, so the last three indices are sofa (2),
floor (1) and mat (0); p is the top-3 softmax, the 17.7 / 29.2 / 53.1 % of concept 5. (b) Sampling follows p, so mat
comes out about 5,300 times in 10,000; banana is cut by top-k, so its chance is exactly 0. For example:
<code>words = [WORDS[next_token(logits)[0]] for _ in range(10000)]</code>, then <code>words.count("mat")</code>.
(c) Dividing by 0.1 turns the gaps 0.6 and 1.1 into 6 and 11, so mat takes 99.8 %: nearly greedy."""},
        ],
    },
]
