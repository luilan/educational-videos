"""Study guide content for How LLMs Work, episode 13 (bonus): Making It Fast: the KV Cache.

Build:  python framework/study_guide.py how-llms-work v13
Figures: {"t": seconds into the rendered video, "caption": ...}; placed in body blocks with {fig0}, {fig1}, ...
Exercise kinds: mc (options), tf, short, number, order, code. Every exercise has an answer and a why.
"""

LESSON = {
    "series": "How LLMs Work",
    "label": "Episode 13",
    "title": "Making It Fast: the KV Cache",
    "tagline": "Why generation doesn't start over for every token",
    "duration": "2:07",
    "intro": """<p>This bonus lesson is about speed. Generating text means one model run per new token, and done naively
every run <b>redoes the work for the whole text</b>. The <b>KV cache</b> saves each token's keys and values, so every
step only has to process <b>one new token</b>. The price is memory.</p>""",
    "prereq": """<div class="box note"><b>Before you start:</b> this lesson builds on episode 1 (the generation loop),
episode 6 (queries, keys, values and the causal mask), episode 7 (attention heads) and episode 9 (layers stacked in
blocks). Concept 7 reads NumPy code; Foundations F11 (NumPy in Three Minutes) helps.</div>""",
}

CONCEPTS = [
    {
        "title": "The naive loop repeats work",
        "segment": (8, 21),
        "figures": [{"t": 20.2, "caption": "Naive generation: each step processes the whole text again. The red "
                                           "squares were already computed on an earlier step."}],
        "body": [
            """<p>Generating text means running the model <b>once for every new token</b> (the loop of episode 1).
Done naively, each run processes the <b>whole text again</b>, from the very first token. With the 5-token prompt
<i>“The cat sat on the”</i>, five steps process 5 + 6 + 7 + 8 + 9 = 35 tokens, and 26 of those computations had
already been done before: <b>repeated work</b>.</p>""",
            "{fig0}",
            """<div class="box key"><b class="t">Key idea</b>Without a cache, every new token costs a run over the
<b>entire text so far</b>. The work per step keeps growing, and most of it repeats earlier steps.</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "26 of the video's 35 naive computations are repeats. How many are new, and where "
                                    "do they come from?",
             "answer": "9 new: 5 on step 1, then 1 on each of the 4 later steps.",
             "why": "35 − 26 = 9. Step 1 processes the prompt for the first time; after that, only the one newly added "
                    "token is new on each step."},
            {"kind": "number", "q": "A new case: a 10-token prompt and 4 naive steps. How many token computations in "
                                    "total?",
             "answer": "46.", "why": "The steps process 10, 11, 12 and 13 tokens: 10 + 11 + 12 + 13 = 46."},
            {"kind": "tf", "q": "“In the naive loop, every step costs the same, however long the text is.”",
             "answer": "False.", "why": "Each step processes the whole text, which is one token longer than last time, "
                                        "so every step costs a little more than the one before."},
            {"kind": "mc", "q": "What exactly makes the naive loop wasteful?",
             "options": ["It predicts several tokens at once", "It redoes the old tokens' work, already done on "
                         "earlier steps", "It forgets the prompt after the first step", "It uses the causal mask"],
             "answer": "B.", "why": "The old tokens' work is recomputed on every step. The rest of this lesson is about "
                                   "not doing that."},
        ],
    },
    {
        "title": "Old tokens never change",
        "segment": (21, 34),
        "figures": [{"t": 33.9, "caption": "A new token (mat) adds one row and one masked column. The old rows are "
                                           "unchanged, and so are the old keys and values."}],
        "body": [
            """<p>With the <b>causal mask</b> (episode 6), every token looks only at itself and the tokens
<b>before</b> it, never at later ones. So when a new token arrives, nothing about the old tokens changes: the grid gains
<b>one new row</b> and a masked column, and the old tokens' <b>keys and values are exactly the same as last time</b>,
in every layer.</p>""",
            "{fig0}",
            """<div class="box key"><b class="t">Key idea</b>A token's keys and values depend only on that token and the
ones before it. Adding a token <b>never changes</b> anything about the earlier ones.</div>""",
        ],
        "exercises": [
            {"kind": "mc", "q": "A sixth token arrives in a 5 × 5 causal attention grid. What happens to the grid?",
             "options": ["Every row is recomputed with new weights", "A new row and a masked column are added; the "
                         "old rows stay the same", "Only the first row changes", "The oldest token is dropped"],
             "answer": "B.", "why": "The new token looks back at everything (a new row); the old tokens never look "
                                   "ahead, so their rows are unchanged."},
            {"kind": "number", "q": "In that same step (5 × 5 grid → 6 × 6), how many cells are new, and how many of the "
                                    "new cells hold real (non-zero) attention weights?",
             "answer": "11 new cells, 6 with real weights.",
             "why": "6 × 6 − 5 × 5 = 11. The new row (6 cells) is all real: the new token sees itself and the 5 before "
                    "it. The 5 new cells above it are masked to zero."},
            {"kind": "tf", "q": "“When <i>mat</i> is added, the key of <i>cat</i> must be recomputed, because "
                                "<i>cat</i> can now attend to <i>mat</i>.”",
             "answer": "False.", "why": "<i>cat</i> never looks ahead, so nothing about it changes: its key and value "
                                        "are exactly the same as before."},
            {"kind": "short", "q": "Imagine a model <b>without</b> the causal mask. When a new token arrives, would the "
                                   "old tokens' keys and values in the later layers stay the same? Explain.",
             "answer": "No.", "why": "The old tokens would now attend to the new one, so their outputs change, and later "
                                     "layers compute keys and values from those outputs. The cache only works because "
                                     "the past never sees the future."},
        ],
    },
    {
        "title": "The KV cache: one new row per step",
        "segment": (34, 64),
        "figures": [{"t": 52.2, "caption": "The new token's query is compared with all cached keys; the weights mix "
                                           "the cached values into its output."},
                    {"t": 63.7, "caption": "Six steps: one token per step with the cache (6 in total), the whole text "
                                           "every step without it (51)."}],
        "body": [
            """<p>So we save them: the <b>KV cache</b> stores every token's <b>key</b> and <b>value</b>, in every
layer. When the next token arrives, we compute only <b>its own</b> query, key and value. Its query is compared with
<b>all the cached keys</b>, and the weights mix <b>the cached values</b>: one new row of the grid, not the whole grid.
Each step now processes <b>one token</b> instead of the entire text, a big part of why chatbots can stream their
answers so quickly.</p>""",
            '<div class="figrow">{fig0}{fig1}</div>',
            """<div class="box key"><b class="t">Key idea</b>Each new token computes only its own query, key and value,
then attends over the cached keys and values: <b>one new row per step</b>, with exactly the same result as recomputing
everything.</div>""",
        ],
        "exercises": [
            {"kind": "mc", "q": "With a KV cache, which vectors are computed when the next token arrives?",
             "options": ["A query, key and value for every token", "Only the new token's query, key and value",
                         "Only the new token's key and value; its query is cached", "All keys and values, no queries"],
             "answer": "B.", "why": "Everything about the old tokens is already in the cache; only the new token is "
                                   "computed."},
            {"kind": "short", "q": "The cache stores keys and values, but no queries. Why is an old token's query never "
                                   "needed again?",
             "answer": "Only the new token asks a question.",
             "why": "A query is used to build that token's own row of the grid. Old rows are finished and never change, "
                    "so their queries are not needed again."},
            {"kind": "number", "q": "A new case: 100 steps from a 1-token prompt, so step <i>k</i> sees <i>k</i> tokens. "
                                    "How many token computations (a) without a cache, (b) with one?",
             "answer": "(a) 5,050. (b) 100.",
             "why": "Without a cache: 1 + 2 + … + 100 = 5,050. With it, one token per step: 100. The longer the text, "
                    "the bigger the saving."},
            {"kind": "tf", "q": "“The KV cache gives a slightly different, approximate result.”",
             "answer": "False.", "why": "Old keys and values never change, so the cached ones are exactly what a "
                                        "recomputation would give. Concept 7 checks this in code."},
        ],
    },
    {
        "title": "The price: memory",
        "segment": (64, 80),
        "figures": [{"t": 79.35, "caption": "Every token, in every layer: 18,432 numbers per token for GPT-2 small, and "
                                            "gigabytes for big models and long chats."}],
        "body": [
            """<p>Speed is paid for with <b>memory</b>. The cache holds a key and a value <b>for every token, in every
layer</b>. GPT-2 small has 12 layers and vectors of 768 numbers, so each token adds <b>2 (K and V) × 12 × 768 = 18,432
numbers</b>. For big models and long conversations, the cache can take up <b>gigabytes</b>.</p>""",
            "{fig0}",
            """<div class="box key"><b class="t">Key idea</b>Cache size = <b>2 × layers × vector size</b> numbers per
token, times the number of tokens. It grows with every token of the conversation.</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "How many numbers does GPT-2 small's cache hold for a conversation of 1,000 tokens?",
             "answer": "18,432,000 (about 18.4 million).", "why": "18,432 numbers per token × 1,000 tokens."},
            {"kind": "number", "q": "A new case: 32 layers and vectors of 4,096 numbers. (a) How many numbers per token? "
                                    "(b) At 2 bytes per number, how much memory for 8,000 tokens?",
             "answer": "(a) 262,144. (b) About 4.2 GB.",
             "why": "2 × 32 × 4,096 = 262,144. Then 262,144 × 8,000 × 2 bytes = 4,194,304,000 bytes ≈ 4.2 GB: "
                    "gigabytes, as the video says."},
            {"kind": "mc", "q": "The conversation becomes twice as long. What happens to the size of the cache?",
             "options": ["It stays the same", "It doubles", "It quadruples", "It halves"],
             "answer": "B.", "why": "Each token adds the same fixed amount (a key and a value per layer), so twice the "
                                   "tokens means twice the cache."},
            {"kind": "tf", "q": "“The cache stores one key and one value per token, shared by all the layers.”",
             "answer": "False.", "why": "Every layer computes its own keys and values, so every layer has its own cache. "
                                        "That is the × 12 in the formula."},
        ],
    },
    {
        "title": "Prefill and decode",
        "segment": (80, 90),
        "figures": [{"t": 89.4, "caption": "Prefill reads the whole prompt at once and fills the cache; decode adds one "
                                           "token at a time, reusing what is cached."}],
        "body": [
            """<p>So generation has two phases. <b>Prefill</b>: read the <b>whole prompt at once</b> and fill the cache
with its keys and values. The prompt is known in advance, so all its tokens can be processed together, in parallel
(as in episode 6). Then <b>decode</b>: <b>one token at a time</b>, reusing the cache.</p>""",
            "{fig0}",
            """<div class="box key"><b class="t">Key idea</b><b>Prefill</b>: the whole prompt in one go, filling the
cache. <b>Decode</b>: one token per step, reusing the cache.</div>""",
        ],
        "exercises": [
            {"kind": "order", "q": "Put in order: <i>decode, one token per step · the user sends a prompt · the cache now "
                                   "holds the prompt's keys and values · prefill</i>.",
             "answer": "user sends a prompt → prefill → the cache holds the prompt's keys and values → decode.",
             "why": "Prefill reads the prompt and fills the cache; only then can decode reuse it."},
            {"kind": "tf", "q": "“During prefill, the prompt's tokens are fed through the model one at a time.”",
             "answer": "False.", "why": "Prefill reads the whole prompt at once. One token at a time is decode."},
            {"kind": "short", "q": "Why can prefill handle all the prompt's tokens together, while decode must go one "
                                   "token at a time?",
             "answer": "The prompt is known in advance; generated tokens are not.",
             "why": "Each new token must first be picked from the previous step's prediction before it can be processed."},
            {"kind": "number", "q": "A new case: GPT-2 small receives a 200-token prompt. (a) How many tokens does each "
                                    "decode step process? (b) How many numbers are in the cache right after prefill?",
             "answer": "(a) 1. (b) 3,686,400.", "why": "Decode processes only the newest token. After prefill the cache "
                                                     "holds 200 tokens × 18,432 numbers = 3,686,400."},
        ],
    },
    {
        "title": "Shrinking the cache: grouped-query attention",
        "segment": (90, 101),
        "figures": [{"t": 100.4, "caption": "Standard: every query head has its own keys and values. Grouped-query: "
                                            "groups of query heads share them (illustrative)."}],
        "body": [
            """<p>Because the cache gets so large, modern models <b>shrink</b> it. Attention has several <b>heads</b>
(episode 7), each with its own queries, keys and values. <b>Grouped-query attention</b> (GQA) lets <b>several query
heads share the same keys and values</b>. In the video's illustration, 8 query heads share 2 key/value heads, so the
cache is <b>4× smaller</b>. Every head still asks its own question.</p>""",
            "{fig0}",
            """<div class="box key"><b class="t">Key idea</b>GQA: groups of query heads share one key/value head. The
cache shrinks by the factor <b>query heads ÷ key/value heads</b>.</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "A model has 32 query heads and 8 key/value heads. How many query heads share each "
                                    "key/value head, and how much smaller is the cache than with standard attention?",
             "answer": "4 per group; 4× smaller.", "why": "32 ÷ 8 = 4 query heads per group, and 8 instead of 32 "
                                                          "key/value heads are cached."},
            {"kind": "number", "q": "GPT-2 small has 12 layers, each with 12 heads of 64 numbers (12 × 64 = 768). If it "
                                    "used GQA with 4 key/value heads, how many numbers would its cache hold per token?",
             "answer": "6,144.", "why": "2 × 12 layers × (4 heads × 64) = 6,144: three times smaller than 18,432, "
                                       "because 12 ÷ 4 = 3."},
            {"kind": "tf", "q": "“Grouped-query attention works by using fewer query heads.”",
             "answer": "False.", "why": "The query heads stay the same; the keys and values are shared, and they are "
                                        "what the cache stores."},
            {"kind": "mc", "q": "Why do modern models use grouped-query attention?",
             "options": ["To make each head more accurate", "To make the KV cache smaller",
                         "So that no causal mask is needed", "To make the prompt shorter"],
             "answer": "B.", "why": "Sharing keys and values means fewer of them to store for every token."},
        ],
    },
    {
        "title": "The KV cache in code",
        "segment": (101, 114),
        "figures": [{"t": 113.2, "size": "small", "caption": "attend_new_token from the video: one new row of "
                                                             "attention, using the cache."}],
        "body": [
            """<p>The whole idea fits in one small function. <code>x_new</code> is the new token's vector and
<code>cache</code> holds two lists, <code>keys</code> and <code>values</code>.</p>""",
            "{fig0}",
            """<pre class="code">def attend_new_token(x_new, cache, Wq, Wk, Wv):
    q, k, v = x_new @ Wq, x_new @ Wk, x_new @ Wv
    cache.keys.append(k)                    # keep for later steps
    cache.values.append(v)
    K, V = np.stack(cache.keys), np.stack(cache.values)
    w = softmax(q @ K.T / np.sqrt(len(k)))  # vs all past tokens
    return w @ V</pre>""",
            """<p>Compute the new token's query, key and value; append the key and value to the cache; then attend over
everything cached so far. <b>No mask is needed</b>, because the cache only holds the past (and the new token
itself).</p>""",
            """<div class="box key"><b class="t">Key idea</b>One call = one new row of the attention grid: the scaled
score of <code>q</code> against every cached key, a softmax, and a weighted mix of the cached values.</div>""",
        ],
        "exercises": [
            {"kind": "short", "q": "Keys have 64 numbers each. After <code>attend_new_token</code> has run for the 7 tokens "
                                   "<i>The cat sat on the mat and</i>, what are the shapes of <code>K</code> and of "
                                   "<code>w</code>?",
             "answer": "<code>K</code>: (7, 64). <code>w</code>: (7,).",
             "why": "One cached key per token, stacked into rows; one weight for each cached token."},
            {"kind": "mc", "q": "Why does <code>attend_new_token</code> need no causal mask?",
             "options": ["Masks are only needed during training", "The cache only holds the new token and the ones "
                         "before it, so there is no future to hide", "The softmax ignores future tokens by itself",
                         "Dividing by <code>np.sqrt(len(k))</code> removes them"],
             "answer": "B.", "why": "Future tokens have not been generated yet, so they are not in the cache."},
            {"kind": "code", "q": "<b>Try it yourself.</b> The code below puts episode 6's full causal "
                                  "<code>attention</code> next to this episode's <code>attend_new_token</code>, and feeds "
                                  "5 random tokens through the cache one at a time. (a) What does the first "
                                  "<code>print</code> show, and what does it tell you? (b) What does the second "
                                  "<code>print</code> show? (c) Delete the line in <code>attention</code> that applies the "
                                  "mask and run it again. What changes, and why?",
             "code": """import numpy as np

def softmax(s):
    e = np.exp(s - s.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)

def attention(X, Wq, Wk, Wv):          # episode 6: the whole grid, masked
    Q, K, V = X @ Wq, X @ Wk, X @ Wv
    scores = Q @ K.T / np.sqrt(K.shape[-1])
    n = len(X)
    scores[np.triu(np.ones((n, n)), k=1).astype(bool)] = -np.inf
    return softmax(scores) @ V

class Cache:
    def __init__(self):
        self.keys, self.values = [], []

def attend_new_token(x_new, cache, Wq, Wk, Wv):   # this episode
    q, k, v = x_new @ Wq, x_new @ Wk, x_new @ Wv
    cache.keys.append(k)
    cache.values.append(v)
    K, V = np.stack(cache.keys), np.stack(cache.values)
    w = softmax(q @ K.T / np.sqrt(len(k)))
    return w @ V

rng = np.random.default_rng(0)
X = rng.normal(size=(5, 4))            # 5 tokens, 4 numbers each
Wq, Wk, Wv = (rng.normal(size=(4, 4)) for _ in range(3))

cache = Cache()
rows = [attend_new_token(x, cache, Wq, Wk, Wv) for x in X]
print(np.allclose(np.stack(rows), attention(X, Wq, Wk, Wv)))
print(len(cache.keys), np.stack(cache.keys).shape)""",
             "answer": "(a) True · (b) 5 (5, 4) · (c) the first print becomes False",
             "why": """(a) Feeding the tokens one at a time through the cache gives exactly the same five output rows
as full causal attention over the whole grid. (b) One key per token: 5 keys of 4 numbers, stacked into a 5 × 4 matrix.
(c) Without the mask, the full version lets each token see the tokens after it, but when the cache handles that token
the later ones do not exist yet. The two agree only because the causal mask keeps the past from seeing the future
(concept 2)."""},
        ],
    },
]
