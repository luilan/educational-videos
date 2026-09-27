"""Study guide content for How LLMs Work, episode 6: Attention II: The Math.

Build:  python framework/study_guide.py how-llms-work v06
Figures: {"t": seconds into the rendered video, "caption": ...}; placed in body blocks with {fig0}, {fig1}, ...
Exercise kinds: mc (options), tf, short, number, order, code. Every exercise has an answer and a why.
All numbers come from the video's worked example (v06_scene.py, SEED = 182528) and were checked with NumPy.
"""

CODE = """def attention(X, Wq, Wk, Wv):
    Q, K, V = X @ Wq, X @ Wk, X @ Wv            # project
    scores = Q @ K.T / np.sqrt(K.shape[-1])     # score and scale
    n = len(X)
    mask = np.triu(np.ones((n, n)), k=1).astype(bool)
    scores[mask] = -np.inf                      # no peeking ahead
    weights = np.exp(scores - scores.max(-1, keepdims=True))
    weights /= weights.sum(-1, keepdims=True)   # softmax, row by row
    return weights @ V                          # mix the values"""

LESSON = {
    "series": "How LLMs Work",
    "label": "Episode 6",
    "title": "Attention II: The Math",
    "tagline": "Queries, keys and values, one matrix at a time",
    "duration": "2:18",
    "intro": """<p>This lesson has one big idea: all of attention, for every token at once, is <b>a handful of matrix
multiplications</b>. Project the tokens to queries, keys and values, score every query against every key, scale, mask
the future, softmax each row, and mix the values: <b>softmax(Q Kᵀ / √d) · V</b>.</p>""",
    "prereq": """<div class="box note"><b>Before you start:</b> this lesson puts episode 5 (queries, keys, values,
weights that add up to 1, only looking backwards) into numbers. It uses the dot product (Foundations F02), matrix
multiplication (F03), e<sup>x</sup> (F05) and softmax (F07); F08 explains why scores grow with d. The code in
concept 7 uses NumPy (F11).</div>""",
}

CONCEPTS = [
    {
        "title": "From X to Q, K and V",
        "segment": (8, 42),
        "figures": [{"t": 29.4, "caption": "X: one row per token, d = 4 columns."},
                    {"t": 42.2, "caption": "Q, K and V: a query, key and value for every token."}],
        "body": [
            """<p>Now with actual numbers. The five tokens <i>The cat sat on the</i> are stacked as the rows of one
matrix, <b>X</b>: 5 rows (one per token) and <b>d</b> columns. The video uses d = 4, so every number fits on
screen.</p>""",
            '<div class="figrow">{fig0}{fig1}</div>',
            """<p>Multiply X by three learned 4 × 4 matrices: <b>Q = X · W<sub>Q</sub></b>, <b>K = X · W<sub>K</sub></b>,
<b>V = X · W<sub>V</sub></b>. Row i of Q is token i's query, row i of K its key, row i of V its value: every token,
all at once, in three matrix multiplications (Foundations F03).</p>""",
            """<div class="box key"><b class="t">Key idea</b>X (n × d) times learned d × d matrices gives <b>Q, K and V</b>
(each n × d). <b>One row per token</b>: its query, its key, its value.</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "X is 5 × 4 and W<sub>Q</sub> is 4 × 4. What is the shape of Q? What would it be "
                                    "for a text of 12 tokens?",
             "answer": "5 × 4; 12 × 4.", "why": "(n × d) · (d × d) = n × d: one 4-number query per token."},
            {"kind": "number", "q": "The first row of X (<i>The</i>) is (−1.5, 0.4, 0.1, 1.3) and the first column of "
                                    "W<sub>Q</sub> is (0.4, 0.6, 0.7, 0.2). Compute the first number of <i>The</i>'s "
                                    "query.",
             "answer": "−0.03.", "why": "−0.60 + 0.24 + 0.07 + 0.26 = −0.03, the top-left entry of Q in the video."},
            {"kind": "tf", "q": "“A longer text needs a bigger W<sub>Q</sub>.”",
             "answer": "False.", "why": "W<sub>Q</sub> is d × d whatever the text length. More tokens only means more "
                                        "rows in X, and so in Q, K and V."},
        ],
    },
    {
        "title": "Scores: Q times K transposed",
        "segment": (42, 59),
        "figures": [{"t": 58.6, "caption": "Q · Kᵀ: the 5 × 5 grid of scores. Row sat, column cat is "
                                           "q<sub>sat</sub> · k<sub>cat</sub> = 1.10."}],
        "body": [
            """<p>Next, compare every query with every key. That is a <b>single matrix multiplication</b>:
<b>scores = Q · Kᵀ</b>. Transposing K turns its rows (the keys) into columns, so entry (i, j) is row i of Q times
column j of Kᵀ, the dot product <b>q<sub>i</sub> · k<sub>j</sub></b>. The result is a 5 × 5 grid: row i, column j
says how well token i's question matches token j's key.</p>""",
            "{fig0}",
            """<div class="box key"><b class="t">Key idea</b><b>scores[i, j] = q<sub>i</sub> · k<sub>j</sub></b>.
Q · Kᵀ computes all n × n query–key dot products in one go.</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "Check the video's example: q<sub>sat</sub> = (0.50, −2.00, −1.09, −1.40) and "
                                    "k<sub>cat</sub> = (1.43, 1.00, −1.19, −0.78). Compute q<sub>sat</sub> · "
                                    "k<sub>cat</sub> to two decimals.",
             "answer": "1.10.", "why": "0.715 − 2.000 + 1.297 + 1.092 = 1.104 ≈ 1.10."},
            {"kind": "number", "q": "q<sub>The</sub> = (−0.03, 0.40, −0.90, 0.34) and k<sub>The</sub> = (−1.45, 1.26, "
                                    "1.47, 0.84). Compute the top-left score of the grid.",
             "answer": "−0.49.", "why": "0.044 + 0.504 − 1.323 + 0.286 = −0.490, as in the video."},
            {"kind": "mc", "q": "A text has 100 tokens and d = 4. What is the shape of Q · Kᵀ?",
             "options": ["100 × 4", "4 × 4", "100 × 100", "4 × 100"],
             "answer": "C.", "why": "(100 × 4) · (4 × 100) = 100 × 100: one score for every (query token, key token) "
                                   "pair."},
            {"kind": "tf", "q": "“The score of <i>sat</i> looking at <i>cat</i> equals the score of <i>cat</i> "
                                "looking at <i>sat</i>.”",
             "answer": "False.", "why": "sat → cat is q<sub>sat</sub> · k<sub>cat</sub> = 1.10, but cat → sat is "
                                        "q<sub>cat</sub> · k<sub>sat</sub> = −2.51. Queries and keys are different "
                                        "vectors, so the grid is not symmetric."},
        ],
    },
    {
        "title": "Scaling by √d",
        "segment": (59, 69),
        "figures": [{"t": 69.0, "caption": "Every score ÷ √4 = ÷ 2. Inset: row “the” turned into "
                                           "weights, without and with the scaling."}],
        "body": [
            """<p>Then divide every score by <b>√d</b>, the square root of the key size. Here d = 4, so we divide by 2.
Without this, scores grow with the vector length (F08), the softmax gets <b>too extreme</b>, and training struggles.
In the inset, row <i>the</i> turned straight into weights gives the top token 0.94, almost everything; after ÷ 2 it
gets 0.73 and the others keep a share.</p>""",
            "{fig0}",
            """<div class="box key"><b class="t">Key idea</b><b>scaled scores = Q · Kᵀ / √d</b>. This keeps the scores
moderate, so the softmax is not too sharp.</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "<i>sat</i>'s score for <i>cat</i> is 1.104. What is it after scaling (d = 4)?",
             "answer": "0.55.", "why": "1.104 ÷ √4 = 1.104 ÷ 2 = 0.552 ≈ 0.55, the value in the scaled grid."},
            {"kind": "number", "q": "A model uses keys of size 64. What number are its scores divided by?",
             "answer": "8.", "why": "√64 = 8."},
            {"kind": "mc", "q": "Why divide the scores by √d?",
             "options": ["So that every row of scores adds up to 1",
                         "Because scores grow with the vector length, which makes the softmax too extreme",
                         "To turn negative scores into positive ones", "To stop tokens from seeing the future"],
             "answer": "B.", "why": "A is the softmax's job, D is the mask's job, and dividing by a positive number "
                                   "never changes a sign."},
            {"kind": "tf", "q": "“Dividing a row of scores by 2 can change which token gets the largest weight.”",
             "answer": "False.", "why": "Dividing by a positive number keeps the order, so the top score stays on top "
                                        "(in row <i>the</i>, sat wins both times). Only the sharpness changes."},
        ],
    },
    {
        "title": "The causal mask",
        "segment": (69, 80),
        "figures": [{"t": 80.7, "caption": "Every score above the diagonal becomes −∞; after the "
                                           "softmax it will be exactly 0."}],
        "body": [
            """<p>Now the <b>causal mask</b>: episode 5's “only look backwards” rule, in numbers. A token must not see
the future, so every score <b>above the diagonal</b> (column j later than row i) is set to <b>−∞</b>. After the next
step, the softmax, those become <b>exactly zero</b>, because e<sup>−∞</sup> = 0.</p>""",
            "{fig0}",
            """<div class="box key"><b class="t">Key idea</b>Mask: <b>scores[i, j] = −∞ whenever j &gt; i</b>. After the
softmax, later tokens get weight exactly 0.</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "How many entries of the 5 × 5 grid are masked? How many would be masked for a "
                                    "100-token text?",
             "answer": "10; 4,950.", "why": "Everything above the diagonal: (25 − 5) ÷ 2 = 10, and "
                                           "(100 × 100 − 100) ÷ 2 = 4,950."},
            {"kind": "mc", "q": "Which of these scores gets masked?",
             "options": ["row sat, column cat", "row cat, column sat", "row the, column The", "row on, column on"],
             "answer": "B.", "why": "sat comes after cat, so cat may not look ahead at it. A and C look backwards; D is "
                                   "a token looking at itself, which is allowed."},
            {"kind": "short", "q": "Which row of the grid has no masked entries at all? Why?",
             "answer": "The last row (the final <i>the</i>).", "why": "No token comes after it, so it may look at "
                                                                      "every token, itself included."},
            {"kind": "tf", "q": "“Setting the masked scores to 0 instead of −∞ would work just as well.”",
             "answer": "False.", "why": "e<sup>0</sup> = 1, so a score of 0 would still get a real share of the weight. "
                                        "Only −∞ gives e<sup>−∞</sup> = 0."},
        ],
    },
    {
        "title": "Softmax, row by row",
        "segment": (80, 91),
        "figures": [{"t": 90.4, "caption": "The attention weights: every row is ≥ 0 and adds up to "
                                           "1.00."}],
        "body": [
            """<p>The <b>softmax</b> (F07) turns each row into weights: <b>exponentiate every score, then divide by the
row's total</b>. Every row is now positive (masked entries exactly 0) and <b>adds up to one</b>: episode 5's budget,
now computed. Row <i>sat</i>, for example, becomes 0.02, 0.70, 0.28: sat listens mostly to cat.</p>""",
            "{fig0}",
            """<div class="box key"><b class="t">Key idea</b><b>weight = exp(score) ÷ row total</b>, one row at a time.
Each row is all ≥ 0 and adds up to 1.</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "Row <i>cat</i> has scaled scores −1.53 and 0.25 (the rest is masked). Compute its "
                                    "two weights. Use e<sup>−1.53</sup> ≈ 0.217 and e<sup>0.25</sup> ≈ 1.284.",
             "answer": "0.14 and 0.86.", "why": "Total 0.217 + 1.284 = 1.501; 0.217 ÷ 1.501 = 0.14 and 1.284 ÷ 1.501 "
                                               "= 0.86, as in the video."},
            {"kind": "short", "q": "Why is row <i>The</i> exactly 1.00, 0, 0, 0, 0, whatever its score is?",
             "answer": "It has only one unmasked entry.", "why": "e<sup>s</sup> ÷ e<sup>s</sup> = 1 for its own score, "
                                                                 "and every masked entry gives e<sup>−∞</sup> = 0."},
            {"kind": "mc", "q": "You add 1 to every unmasked score in a row. What happens to that row's weights?",
             "options": ["They all grow", "They stay exactly the same", "The largest grows and the others shrink",
                         "They no longer add up to 1"],
             "answer": "B.", "why": "e<sup>s+1</sup> = e × e<sup>s</sup>, and the common factor e cancels in the "
                                   "division: only differences between scores matter (F07)."},
            {"kind": "tf", "q": "“A negative score gives a negative weight.”",
             "answer": "False.", "why": "e<sup>x</sup> &gt; 0 for every x: in row <i>sat</i>, the score −0.38 "
                                        "becomes the weight 0.28."},
        ],
    },
    {
        "title": "Mixing the values",
        "segment": (91, 103),
        "figures": [{"t": 97.5, "caption": "output = weights · V. Row sat is 0.02 · v<sub>The</sub> "
                                           "+ 0.70 · v<sub>cat</sub> + 0.28 · v<sub>sat</sub>."}],
        "body": [
            """<p>Finally, <b>multiply the weights by V</b>. Each token's output row is a <b>weighted blend of the
values it attends to</b>: out<sub>sat</sub> = 0.02 · v<sub>The</sub> + 0.70 · v<sub>cat</sub> + 0.28 ·
v<sub>sat</sub>. The first token can only see itself (weight 1.00), so it simply gets its own value back.</p>""",
            "{fig0}",
            """<div class="box key"><b class="t">Key idea</b><b>output = weights · V</b>. Row i is the weighted sum of
the values of token i and the tokens before it.</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "The first numbers of v<sub>The</sub>, v<sub>cat</sub> and v<sub>sat</sub> are "
                                    "−0.14, −0.07 and 1.15. With the weights 0.02, 0.70 and 0.28, compute the first "
                                    "number of out<sub>sat</sub>.",
             "answer": "0.27.", "why": "−0.003 − 0.049 + 0.322 = 0.270 ≈ 0.27, the video's value."},
            {"kind": "mc", "q": "The weights are 5 × 5 and V is 5 × 4. What is the shape of the output?",
             "options": ["5 × 5", "5 × 4", "4 × 4", "4 × 5"],
             "answer": "B.", "why": "(5 × 5) · (5 × 4) = 5 × 4: one new 4-number vector per token, the same shape as "
                                   "X."},
            {"kind": "short", "q": "Row <i>on</i> has weights 0.21, 0.04, 0.74, 0.01, 0. Whose value dominates "
                                   "out<sub>on</sub>, and whose value is not in it at all?",
             "answer": "sat's (0.74); the final <i>the</i>'s (weight 0).",
             "why": "The weights set each value's share. The final <i>the</i> comes after <i>on</i>, so it is masked."},
            {"kind": "tf", "q": "“out<sub>The</sub> equals v<sub>The</sub>.”",
             "answer": "True.", "why": "The first token's only weight is 1.00, on itself: 1.00 · v<sub>The</sub> = "
                                       "v<sub>The</sub>."},
        ],
    },
    {
        "title": "The whole formula, in NumPy",
        "segment": (103, 136),
        "figures": [{"t": 119.4, "caption": "One line, a handful of matrix multiplications, every token in "
                                            "parallel."},
                    {"t": 126.4, "caption": "The same thing in NumPy: project, score and scale, mask, softmax, "
                                            "mix."}],
        "body": [
            """<p>Put together, it is one line: <b>Attention(Q, K, V) = softmax(Q Kᵀ / √d) · V</b>, with the causal
mask applied before the softmax. Every token is handled <b>in parallel</b>, in a handful of matrix multiplications.
That is a big reason transformers run so well on GPUs.</p>""",
            '<div class="figrow">{fig0}{fig1}</div>',
            f"""<pre class="code">{CODE}</pre>""",
            """<p><code>np.triu(…, k=1)</code> marks the entries above the diagonal. The softmax subtracts each row's
max first, the overflow trick from F07; it does not change the weights. One attention layer asks one kind of question;
episode 7 asks several at once.</p>""",
            """<div class="box key"><b class="t">Key idea</b><b>softmax(Q Kᵀ / √d) · V</b>, plus the causal mask: all
tokens at once, in a few matrix multiplications.</div>""",
        ],
        "exercises": [
            {"kind": "order", "q": "Put the steps in the order the code runs them: <i>softmax · mask · multiply by V · "
                                   "Q · Kᵀ · divide by √d · project X to Q, K, V</i>.",
             "answer": "project → Q · Kᵀ → ÷ √d → mask → softmax → multiply by V.",
             "why": "Scores before weights, the mask before the softmax (so masked entries become 0), values last."},
            {"kind": "mc", "q": "Why does attention run so well on GPUs?",
             "options": ["It handles every token at once, as a few large matrix multiplications",
                         "It skips most tokens, so there is little to compute",
                         "It processes one token after another, in a loop",
                         "It uses no multiplications, only additions"],
             "answer": "A.", "why": "GPUs are built to do huge matrix multiplications in parallel, and that is "
                                   "exactly what attention is."},
            {"kind": "code", "q": "<b>Try it yourself.</b> The code runs the video's <code>attention</code> function on "
                                  "3 tokens with d = 2. All three weight matrices are the identity, so Q = K = V = X. (a) Predict row 1 of the output "
                                  "without computing anything. (b) By hand, compute token 2's two weights and its "
                                  "output row (remember to divide by √2), then run the code to check. (c) Comment out "
                                  "the line <code>scores[mask] = -np.inf</code> and run again. Which rows change, and "
                                  "why?",
             "code": "import numpy as np\n\n" + CODE + """

X = np.array([[1.0, 0.0],     # token 1
              [0.0, 1.0],     # token 2
              [1.0, 1.0]])    # token 3
I = np.eye(2)                 # Wq = Wk = Wv = identity: Q = K = V = X
print(attention(X, I, I, I).round(2))""",
             "answer": "(a) (1, 0) · (b) weights 0.33 and 0.67, output (0.33, 0.67) · (c) rows 1 and 2 change, to "
                       "(0.8, 0.6) and (0.6, 0.8); row 3 stays (0.75, 0.75)",
             "why": """The printed rows are <code>[1. 0.]</code>, <code>[0.33 0.67]</code> and <code>[0.75 0.75]</code>. (a) Token 1 sees only itself,
so it gets its own value, (1, 0). (b) Scores q<sub>2</sub> · k<sub>1</sub> = 0 and q<sub>2</sub> · k<sub>2</sub> = 1;
divided by √2 ≈ 1.41 they are 0 and 0.71. e<sup>0</sup> = 1 and e<sup>0.71</sup> ≈ 2.03, so the weights are
1 ÷ 3.03 = 0.33 and 2.03 ÷ 3.03 = 0.67, and the output is 0.33 · (1, 0) + 0.67 · (0, 1) = (0.33, 0.67). (c) Without the
mask, tokens 1 and 2 also mix in the values of later tokens. Token 3 is last: it already saw everything, so its row
does not change."""},
        ],
    },
]
