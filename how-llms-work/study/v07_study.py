"""Study guide content for How LLMs Work, episode 7: Multi-Head Attention.

Build:  python framework/study_guide.py how-llms-work v07
Figures: {"t": seconds into the rendered video, "caption": ...}; placed in body blocks with {fig0}, {fig1}, ...
Exercise kinds: mc (options), tf, short, number, order, code. Every exercise has an answer and a why.
All numbers and code outputs were checked with NumPy.
"""

CODE = """def multi_head_attention(X, Wq, Wk, Wv, Wo, n_heads):
    n, d = X.shape
    hd = d // n_heads  # 768 // 12 = 64
    def split(M):      # (n, d) -> (heads, n, hd)
        return M.reshape(n, n_heads, hd).transpose(1, 0, 2)
    Q, K, V = split(X @ Wq), split(X @ Wk), split(X @ Wv)
    scores = Q @ K.transpose(0, 2, 1) / np.sqrt(hd)    # every head at once
    scores += np.triu(np.full((n, n), -np.inf), k=1)   # causal mask
    w = np.exp(scores - scores.max(-1, keepdims=True))
    w /= w.sum(-1, keepdims=True)                      # softmax
    out = (w @ V).transpose(1, 0, 2).reshape(n, d)     # merge the heads
    return out @ Wo                                    # output projection"""

ONE_HEAD = """def attention(X, Wq, Wk, Wv):                  # episode 6: one head
    Q, K, V = X @ Wq, X @ Wk, X @ Wv
    scores = Q @ K.T / np.sqrt(K.shape[-1])
    n = len(X)
    scores[np.triu(np.ones((n, n)), k=1).astype(bool)] = -np.inf
    w = np.exp(scores - scores.max(-1, keepdims=True))
    w /= w.sum(-1, keepdims=True)
    return w @ V"""

LESSON = {
    "series": "How LLMs Work",
    "label": "Episode 7",
    "title": "Multi-Head Attention",
    "tagline": "Many questions, asked at the same time",
    "duration": "2:00",
    "intro": """<p>This lesson has one big idea: instead of one big attention, run <b>several smaller ones side by
side</b>, called <b>heads</b>. Each head has its own query, key and value matrices, so each can learn to look for
something different. Their results are glued back together, mixed by one more learned matrix, and added to the
token's vector.</p>""",
    "prereq": """<div class="box note"><b>Before you start:</b> this lesson builds directly on episodes 5 and 6: one
head is exactly the attention of episode 6. It uses matrix shapes and multiplication (Foundations F03), and the code
works with NumPy arrays that have an extra “heads” axis (F11).</div>""",
}

CONCEPTS = [
    {
        "title": "Several heads, side by side",
        "segment": (8, 29),
        "figures": [{"t": 18.8, "caption": "One word, several questions."},
                    {"t": 29.2, "caption": "Several heads, each with its own W<sub>Q</sub>, W<sub>K</sub>, "
                                           "W<sub>V</sub>."}],
        "body": [
            """<p>One attention <b>head</b> (episode 6) asks one kind of question: each token gets one row of weights,
so one blend of the others. But when the model reads <i>sat</i>, it might want to know several things at once. Who is
sitting? Where? What came just before?</p>""",
            '<div class="figrow">{fig0}{fig1}</div>',
            """<p>So instead of one big attention, we run several smaller ones <b>side by side</b>. Each one is called
a <b>head</b>, and each has its <b>own query, key and value matrices</b>.</p>""",
            """<div class="box key"><b class="t">Key idea</b><b>Multi-head attention</b> runs several attention heads
in parallel, each with its own learned W<sub>Q</sub>, W<sub>K</sub> and W<sub>V</sub>, so a token can ask several
questions at once.</div>""",
        ],
        "exercises": [
            {"kind": "mc", "q": "What does each head have its own copy of?",
             "options": ["The token vectors X", "The matrices W<sub>Q</sub>, W<sub>K</sub> and W<sub>V</sub>",
                         "The vocabulary of tokens", "The text, in a different order"],
             "answer": "B.", "why": "All heads read the same token vectors. Their own matrices let them ask "
                                   "different questions."},
            {"kind": "tf", "q": "“With a single head, <i>sat</i> gets one row of attention weights, so it can blend in "
                                "only one mix of the other tokens.”",
             "answer": "True.", "why": "One head = one set of weights per token = one kind of question. Several heads "
                                       "give several rows, so several different mixes."},
            {"kind": "short", "q": "In <i>“The dog quickly ran home”</i>, the token <i>ran</i> asks “who is "
                                   "running?” and “what came just before?”. Which token should each question find?",
             "answer": "<i>dog</i>; <i>quickly</i>.", "why": "Like <i>cat</i> and <i>slowly</i> for <i>sat</i> in "
                                                           "the video: two questions, two different answers, so "
                                                           "two heads."},
        ],
    },
    {
        "title": "Sharing out the vector",
        "segment": (29, 40),
        "figures": [{"t": 39.6, "caption": "GPT-2 small: 12 heads share 768 numbers, 64 each."}],
        "body": [
            """<p>Here's the trick: the heads <b>share out the vector</b>. GPT-2 small uses <b>12</b> heads on its
<b>768</b> numbers, so each head works with <b>768 ÷ 12 = 64</b> of them.</p>""",
            """<p>Precisely: the whole token vector is projected to a query, a key and a value of 768 numbers each
(as in episode 6), and each of those is cut into 12 pieces of 64, one piece per head. Every head then runs ordinary
attention on its own 64-number pieces, scaling by √64 = 8.</p>""",
            "{fig0}",
            """<div class="box key"><b class="t">Key idea</b><b>head size = d ÷ number of heads</b> (GPT-2 small:
768 ÷ 12 = 64). Each head gets its own 64-number query, key and value for every token.</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "A model has vectors of 1,024 numbers and 16 heads. What is the head size?",
             "answer": "64.", "why": "1,024 ÷ 16 = 64."},
            {"kind": "number", "q": "A model has d = 512 and a head size of 64. How many heads does it have?",
             "answer": "8.", "why": "512 ÷ 64 = 8."},
            {"kind": "tf", "q": "“Each GPT-2 small head only ever reads 64 of the token's original 768 numbers.”",
             "answer": "False.", "why": "X · W<sub>Q</sub> uses all 768 numbers; it is the resulting query (and key "
                                        "and value) that is cut into 64-number pieces. Every head's 64 numbers are "
                                        "computed from the whole vector."},
            {"kind": "number", "q": "For a 10-token text, how many attention scores does one head compute? And all "
                                    "12 heads together?",
             "answer": "100; 1,200.", "why": "Each head makes its own 10 × 10 grid (episode 6), and 12 × 100 = 1,200."},
        ],
    },
    {
        "title": "Different heads, different patterns",
        "segment": (40, 70),
        "figures": [{"t": 58.8, "caption": "Four heads, four patterns (row = token asking, column = token looked "
                                           "at)."},
                    {"t": 70.6, "caption": "On “sat”: three heads look in three different ways."}],
        "body": [
            """<p>Because each head learns its <b>own matrices</b>, each can learn a <b>different pattern</b>. In
trained models, researchers have found heads that look at the <b>previous token</b>, heads that connect a <b>verb to
its object</b>, and heads that spot a <b>repeated phrase</b> and predict how it continues (<b>induction</b> heads).
Many others are much harder to interpret. Nobody assigns these jobs: they come out of training.</p>""",
            '<div class="figrow">{fig0}{fig1}</div>',
            """<p>In the illustration, on <i>sat</i>, head 1 looks at <i>cat</i> (the one doing the sitting), head 2
looks one step back (<i>slowly</i>), and head 3 spreads its attention widely over the whole sentence.</p>""",
            """<div class="box key"><b class="t">Key idea</b>Each head <b>learns its own attention pattern</b> from data.
Some are easy to interpret (previous token, verb → object, induction); many are not.</div>""",
        ],
        "exercises": [
            {"kind": "mc", "q": "In a “previous token” head's grid, where are the bright cells?",
             "options": ["On the diagonal: each token looks at itself",
                         "Just left of the diagonal: row i, column i − 1",
                         "All in the first column", "Above the diagonal"],
             "answer": "B.", "why": "Token i looks at token i − 1. D is impossible: the causal mask keeps everything "
                                   "above the diagonal empty."},
            {"kind": "tf", "q": "“Engineers decide which head does what, for example ‘head 2 links verbs to their "
                                "objects’.”",
             "answer": "False.", "why": "The patterns are learned in training. Researchers find them afterwards, and "
                                        "many heads have no clear job at all."},
            {"kind": "short", "q": "A text contains <i>“… the Golden Gate Bridge … the Golden Gate”</i>. What would a "
                                   "repeated-phrase (induction) head help predict next, and why?",
             "lines": 2,
             "answer": "<i>Bridge</i>.", "why": "“the Golden Gate” appeared before, followed by <i>Bridge</i>. An "
                                              "induction head finds what followed the earlier copy and predicts that "
                                              "the phrase continues the same way."},
        ],
    },
    {
        "title": "Glue, mix and add",
        "segment": (70, 87),
        "figures": [{"t": 78.2, "caption": "Concatenate: 12 × 64 = 768 numbers."},
                    {"t": 86.4, "caption": "W<sub>O</sub> mixes the heads; the result is added to the token."}],
        "body": [
            """<p>Each head produces its own <b>64 numbers</b> for every token. We <b>glue them back together</b>
(<b>concatenate</b>) into one vector of 12 × 64 = <b>768</b>. Then one more learned matrix, the <b>output projection
W<sub>O</sub></b>, mixes what all the heads found, and the result is <b>added back to the token's vector</b>: added,
not replaced, just like the weighted mix in episode 5.</p>""",
            '<div class="figrow">{fig0}{fig1}</div>',
            """<div class="box key"><b class="t">Key idea</b><b>Concatenate</b> the heads (12 × 64 = 768) → multiply by
<b>W<sub>O</sub></b> → <b>add</b> the result to the token's vector.</div>""",
        ],
        "exercises": [
            {"kind": "order", "q": "Put in order: <i>concatenate the heads · every head runs attention · add to the "
                                   "token's vector · multiply by W<sub>O</sub> · project and split into heads</i>.",
             "lines": 2,
             "answer": "project and split → every head runs attention → concatenate → multiply by W<sub>O</sub> → add.",
             "why": "Split before the heads work, glue after; W<sub>O</sub> mixes the glued result, which is then "
                    "added."},
            {"kind": "number", "q": "In GPT-2 small, W<sub>O</sub> takes 768 numbers to 768 numbers. How many learned "
                                    "numbers does it hold?",
             "answer": "589,824.", "why": "It is a 768 × 768 matrix: 768 × 768 = 589,824."},
            {"kind": "mc", "q": "What is W<sub>O</sub> for?",
             "options": ["It mixes what the different heads found into one result", "It applies the causal mask",
                         "It turns the scores into weights that add up to 1", "It splits the vector into heads"],
             "answer": "A.", "why": "After gluing, each block of 64 numbers holds one head's findings; W<sub>O</sub> "
                                   "lets them combine. B and C happen inside every head, D before."},
            {"kind": "tf", "q": "“After multi-head attention, the token's vector is replaced by the result.”",
             "answer": "False.", "why": "The result is <b>added</b> to the token's vector (“residual: add, don't "
                                        "replace”), so the token keeps what it had and gains context."},
        ],
    },
    {
        "title": "Many views for the price of one",
        "segment": (87, 97),
        "figures": [{"t": 96.5, "caption": "12 heads of 64 cost about the same as 1 head of 768: many views, "
                                           "one price."}],
        "body": [
            """<p>The nice part: <b>twelve heads of size 64 cost about the same as one head of size 768</b>. Either
way, the query, key and value matrices hold <b>3 × 768 × 768</b> numbers: a full 768 × 768 W<sub>Q</sub> is simply the
twelve heads' 768 × 64 slices side by side. We get many points of view, for the price of one.</p>""",
            "{fig0}",
            """<div class="box key"><b class="t">Key idea</b>Splitting into heads adds <b>no extra weights</b>:
12 × (768 × 64) = 768 × 768. Twelve attention patterns for the price of one.</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "How many numbers are in one GPT-2 small layer's W<sub>Q</sub>, W<sub>K</sub> and "
                                    "W<sub>V</sub> together?",
             "answer": "1,769,472.", "why": "3 × 768 × 768 = 3 × 589,824 = 1,769,472."},
            {"kind": "number", "q": "One head's slice of W<sub>Q</sub> is 768 × 64. How many numbers is that? And for "
                                    "all 12 heads?",
             "answer": "49,152; 589,824.", "why": "768 × 64 = 49,152, and 12 × 49,152 = 589,824: exactly one "
                                                 "768 × 768 matrix."},
            {"kind": "tf", "q": "“Using 24 heads instead of 12 (still with d = 768) would double the query, key and "
                                "value weights.”",
             "answer": "False.", "why": "The head size halves to 768 ÷ 24 = 32, and 24 × 768 × 32 = 768 × 768 again."},
            {"kind": "mc", "q": "What do 12 heads of size 64 give you that one head of size 768 does not?",
             "options": ["Twelve separate attention patterns for every token", "Twelve times as many weights",
                         "Longer token vectors", "The ability to look at future tokens"],
             "answer": "A.", "why": "Same cost, but each token can now look in twelve different ways at once. B and C "
                                   "are false, and D is still forbidden by the mask."},
        ],
    },
    {
        "title": "Multi-head attention in code",
        "segment": (97, 117),
        "figures": [{"t": 115.3, "size": "small", "caption": "Attention lets tokens talk; the MLP then works on "
                                                             "each token on its own."}],
        "body": [
            """<p>In code, we <b>split</b> the vectors into heads, run attention on <b>every head at once</b>, then
<b>merge</b> the heads and apply the <b>output projection</b>. “Every head at once” works because the arrays get an
extra first axis for the heads: <code>split</code> turns an (n, d) matrix into shape (heads, n, hd), and the rest is
episode 6's attention, done for all heads in one go.</p>""",
            f"""<pre class="code">{CODE}</pre>""",
            """<p>Attention lets tokens share information. Once a token has gathered its context, it needs to work
with it on its own: that is the job of the other half of every layer, the <b>MLP</b> (episode 8).</p>""",
            "{fig0}",
            """<div class="box key"><b class="t">Key idea</b><b>split</b> (n, d) → (heads, n, hd), <b>attention</b> in
every head at once, <b>merge</b> back to (n, d), then <b>@ W<sub>O</sub></b>.</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "GPT-2 small reads a 7-token text (n = 7, d = 768, 12 heads). What is the shape of "
                                    "(a) <code>Q</code> after <code>split</code>, (b) <code>scores</code>, (c) "
                                    "<code>out</code> before <code>@ Wo</code>?",
             "lines": 2,
             "answer": "(a) (12, 7, 64). (b) (12, 7, 7). (c) (7, 768).",
             "why": "(heads, n, hd); one n × n grid per head; merging glues the heads back into one 768-number vector "
                    "per token."},
            {"kind": "tf", "q": "“In every layer, attention lets tokens share information, and the MLP then works on "
                                "each token on its own.”",
             "answer": "True.", "why": "That is the outro's picture: every layer is attention, then MLP."},
            {"kind": "code", "q": "<b>Try it yourself.</b> The code runs the video's <code>multi_head_attention</code> "
                                  "and episode 6's one-head <code>attention</code> on 3 tokens with d = 4, using a "
                                  "“do nothing” output projection (W<sub>O</sub> = identity). (a) With "
                                  "<code>n_heads=2</code>, what are <code>hd</code> and the shapes of <code>Q</code> "
                                  "(after <code>split</code>) and <code>scores</code>? Add prints to check. (b) What "
                                  "do the three <code>print</code> lines show? (c) Explain the two True/False results: "
                                  "why is one head exactly episode 6's attention, and why do two heads give something "
                                  "different from the same weights?",
             "code": "import numpy as np\n\n" + ONE_HEAD + "\n\n" + CODE + """

rng = np.random.default_rng(0)
n, d = 3, 4
X = rng.normal(size=(n, d))
Wq, Wk, Wv = rng.normal(size=(3, d, d))
Wo = np.eye(d)                                  # "do nothing" projection

one = multi_head_attention(X, Wq, Wk, Wv, Wo, n_heads=1)
two = multi_head_attention(X, Wq, Wk, Wv, Wo, n_heads=2)
print(one.shape, two.shape)
print(np.allclose(one, attention(X, Wq, Wk, Wv)))
print(np.allclose(one, two))""",
             "answer": "(a) hd = 2, Q (2, 3, 2), scores (2, 3, 3) · (b) (3, 4) (3, 4), then True, then False",
             "why": """(a) hd = 4 // 2 = 2; <code>split</code> gives (heads, n, hd) = (2, 3, 2), and each head has its own
3 × 3 grid. (b) Both outputs keep X's shape, (3, 4), so they can be added back to X. (c) With one head, hd = d,
<code>split</code> changes nothing, the scaling is √4 as in episode 6, and W<sub>O</sub> = identity changes nothing: it
<i>is</i> single-head attention (True). With two heads, each head scores with only its 2-number half of the query and
key (and divides by √2), so each gets its own weights and mixes its half of the values in its own way (False)."""},
        ],
    },
]
