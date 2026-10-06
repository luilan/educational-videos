"""Study guide content for How LLMs Work, episode 12: Build a Tiny GPT.

Build:  python framework/study_guide.py how-llms-work v12
Figures: {"t": seconds into the rendered video, "caption": ...}; placed in body blocks with {fig0}, {fig1}, ...
Exercise kinds: mc (options), tf, short, number, order, code. Every exercise has an answer and a why.
"""

LESSON = {
    "series": "How LLMs Work",
    "label": "Episode 12",
    "title": "Build a Tiny GPT",
    "tagline": "Every piece, in about a hundred lines of code",
    "duration": "2:19",
    "intro": """<p>This lesson puts the whole series into <b>one real program</b>: <code>tiny_gpt.py</code>, about a
hundred lines of PyTorch. It reads a million characters of Shakespeare, builds a GPT with <b>818,241 parameters</b> from
the pieces of episodes 2–10, trains it with the loop of episode 11 on an ordinary CPU, and watches it go from noise to
something that looks like a play.</p>""",
    "prereq": """<div class="box note"><b>Before you start:</b> this lesson assembles everything from episodes 1–11,
so watch those first; episode 11 (training) matters most. The maths is in Foundations F05 (Exponentials and
Logarithms), F10 (Slopes and Gradients), F12 (Neural Networks in Three Minutes), F13 (PyTorch and Autograd) and F14
(Train vs Validation Data). The hands-on exercises need Python with PyTorch.</div>""",
}

CONCEPTS = [
    {
        "title": "Every character is a token",
        "segment": (8, 35),
        "figures": [{"t": 28.7, "caption": "1,115,394 characters of Shakespeare. The whole vocabulary: 65 symbols."},
                    {"t": 34.3, "caption": "The tokenizer: characters to IDs (encode) and back (decode)."}],
        "body": [
            """<p>Over eleven episodes we built every piece of a GPT. <code>tiny_gpt.py</code> puts them together and
trains the result on an ordinary computer. The training text is <b>1,115,394 characters</b> of Shakespeare's plays. To
keep things tiny, <b>every character is a token</b>, so the vocabulary is just the <b>65 different characters</b> in the
file: newline, space, 10 punctuation marks, the digit 3, and the 52 capital and small letters.</p>""",
            '<div class="figrow">{fig0}{fig1}</div>',
            """<p>The tokenizer is <b>two small dictionaries</b>: <code>stoi</code> gives each character its position in
the sorted list, and <code>itos</code> maps back. <code>encode("hi")</code> is <code>[46, 47]</code>, and
<code>decode</code> turns that back into <i>hi</i>.</p>""",
            """<div class="box key"><b class="t">Key idea</b>Character-level tokens: the vocabulary is the <b>65
characters</b> of the text, sorted, and a token's ID is its place in that list. Encoding and decoding are dictionary
lookups.</div>""",
        ],
        "exercises": [
            {"kind": "mc", "q": "Why is the tiny GPT's vocabulary only 65 tokens?",
             "options": ["The model cannot handle more than 65 numbers", "Each token is one character, and the text "
                         "uses only 65 different characters", "Rare words were removed from the text",
                         "65 is the length of the context"],
             "answer": "B.", "why": "<code>sorted(set(text))</code> collects the distinct characters of the file.", "key": {'choice': 1}},
            {"kind": "tf", "q": "“With this tokenizer, the word <i>the</i> is a single token.”",
             "answer": "False.", "why": "Every character is a token: <i>t</i>, <i>h</i>, <i>e</i> are three. (The "
                                        "GPT-2 tokenizer of episode 2 makes <i>the</i> one token.)", "key": {'value': False}},
            {"kind": "number", "q": "How many tokens is the text <i>“First Citizen:”</i>?",
             "answer": "14.", "why": "One per character, and the space counts: 5 + 1 + 8.", "key": {'parts': [{'label': None, 'value': 14, 'tol': 0.5, 'unit': None}]}},
            {"kind": "code", "q": "<b>Try it yourself.</b> This code rebuilds the real 65-character vocabulary of "
                                  "<code>tiny_gpt.py</code> (the same list <code>sorted(set(text))</code> finds in the "
                                  "Shakespeare file) and its tokenizer. (a) Which IDs do the newline and the space get, "
                                  "and where do the capital and the small letters start? (b) Work out "
                                  "<code>encode(\"Hi!\")</code> by hand, then check. (c) What is "
                                  "<code>decode([39, 1, 41, 39, 58])</code>? (d) What happens with "
                                  "<code>encode(\"4 cats\")</code>, and why?",
             "code": r"""import string

chars = sorted("\n !$&',-.3:;?" + string.ascii_uppercase + string.ascii_lowercase)
vocab_size = len(chars)
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for c, i in stoi.items()}
encode = lambda s: [stoi[c] for c in s]
decode = lambda ids: "".join(itos[i] for i in ids)

print(vocab_size, encode("hi"), decode(encode("hi")))     # 65 [46, 47] hi""",
             "answer": "(a) 0 and 1; A is 13, a is 39 · (b) [20, 47, 2] · (c) “a cat” · (d) KeyError: '4'",
             "why": """(a) Sorting by character code puts newline and space first, then the punctuation and the 3
(IDs 2–12), then A–Z (13–38) and a–z (39–64). (b) H = 13 + 7 = 20, i = 39 + 8 = 47, ! = 2. (c) 39 = a, 1 = space,
41 = c, 39 = a, 58 = t. (d) The only digit in the Shakespeare text is 3, so 4 has no ID: the model can only read and
write the 65 characters it was trained on."""},
        ],
    },
    {
        "title": "Embeddings: token plus position",
        "segment": (35, 44),
        "figures": [{"t": 43.8, "size": "small",
                     "caption": "class TinyGPT: a token table (video 3) and a position table (video 4)."}],
        "body": [
            """<p>The model is the class <code>TinyGPT</code>. It starts with two <b>embedding tables</b>, as in
episodes 3 and 4: one row of learned numbers per <b>token</b> and one per <b>position</b>, <b>added</b> together in
<code>forward</code>.</p>""",
            "{fig0}",
            """<pre class="code">self.tok_emb = nn.Embedding(vocab_size, n_embd)     # video 3
self.pos_emb = nn.Embedding(block_size, n_embd)     # video 4 (learned, like GPT-2)
x = self.tok_emb(idx) + self.pos_emb(torch.arange(idx.shape[1]))</pre>""",
            """<p>So every character becomes <b>128 numbers</b> (<code>n_embd</code>) saying <i>which</i> character and
<i>where</i>. The position table has only <code>block_size = 64</code> rows: the model reads at most 64 characters at a
time.</p>""",
            """<div class="box key"><b class="t">Key idea</b><b>x = token embedding + position embedding</b>: one
128-number vector per character. The position table's 64 rows set the context length.</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "How many learned numbers are in the two tables together?",
             "answer": "16,512.", "why": "Token table 65 × 128 = 8,320; position table 64 × 128 = 8,192.", "key": {'parts': [{'label': None, 'value': 16512, 'tol': 0.5, 'unit': None}]}},
            {"kind": "short", "q": "Input <i>hi</i> = [46, 47]. Which two rows are added to make the vector for <i>i</i>?",
             "answer": "Row 47 of the token table + row 1 of the position table.",
             "why": "<i>i</i> is token 47 and sits at position 1 (positions count from 0)."},
            {"kind": "number", "q": "A batch is 32 sequences of 64 characters. What is the shape of <code>x</code> "
                                    "after the embeddings?",
             "answer": "32 × 64 × 128.", "why": "Each of the 32 × 64 characters becomes a vector of <code>n_embd</code> = 128 "
                                   "numbers.", "key": {'parts': [{'label': 'dim 1', 'value': 32, 'tol': 0.5, 'unit': None}, {'label': 'dim 2', 'value': 64, 'tol': 0.5, 'unit': None}, {'label': 'dim 3', 'value': 128, 'tol': 0.5, 'unit': None}]}},
            {"kind": "tf", "q": "“The tiny GPT can read a context of any length, since positions are added, not stored.”",
             "answer": "False.", "why": "<code>pos_emb</code> has only 64 rows, so position 64 has no row (PyTorch "
                                        "raises an error). That is why <code>generate</code> feeds only the last 64 "
                                        "characters: <code>idx[:, -block_size:]</code>.", "key": {'value': False}},
        ],
    },
    {
        "title": "Inside a block: attention and the MLP",
        "segment": (44, 63),
        "figures": [{"t": 56.5, "caption": "The MLP: 128 → 4 × 128 = 512 → 128."},
                    {"t": 62.8, "caption": "class Block, and 4 blocks stacked."}],
        "body": [
            """<p><b>Attention</b> (episodes 5–7) makes queries, keys and values with <b>one linear layer</b>
(128 → 3 × 128 = 384 numbers, split three ways), shared out over <b>4 heads</b> of 32 numbers. The scores q · k are
divided by √32; the <b>causal mask</b> sets scores for later positions to −∞, so after the <b>softmax</b> they get
weight 0: no peeking ahead.</p>""",
            """<p>The <b>MLP</b> (episode 8) expands each vector to 512 numbers (4×), applies <b>GELU</b>, and projects
back to 128. The <b>block</b> (episode 9) adds both onto the residual stream, each after a <b>layer norm</b>:
<code>x = x + self.attn(self.ln1(x))</code>, then <code>x = x + self.mlp(self.ln2(x))</code>. Four blocks are
stacked.</p>""",
            '<div class="figrow">{fig0}{fig1}</div>',
            """<div class="box key"><b class="t">Key idea</b>A block = <b>attention</b> (tokens talk: 4 heads, causal
mask) + <b>MLP</b> (each token thinks: 128 → 512 → 128), each added to the residual stream. The tiny GPT stacks 4.</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "<code>n_embd</code> = 128, <code>n_head</code> = 4. (a) How many numbers per head? "
                                    "(b) What are the scores divided by?",
             "answer": "(a) 32. (b) √32 ≈ 5.66.",
             "why": "<code>C // n_head</code> = 128 // 4 = 32, and the code divides by "
                    "<code>k.size(-1) ** 0.5</code> = √32.", "key": {'parts': [{'label': '(a)', 'value': 32, 'tol': 0.5, 'unit': None}, {'label': '(b)', 'value': 5.66, 'tol': 0.113, 'unit': None}]}},
            {"kind": "mc", "q": "What does <code>att.masked_fill(~self.mask[:T, :T], float(\"-inf\"))</code> achieve?",
             "options": ["It removes padding characters", "No position can attend to later ones, so it cannot see "
                         "what it must predict", "It keeps only the highest score in each row",
                         "It switches off some heads at random"],
             "answer": "B.", "why": "−∞ becomes 0 after the softmax. Without the mask, training would let the model "
                                   "read the answer.", "key": {'choice': 1}},
            {"kind": "number", "q": "With <code>n_embd</code> = 256 and 4 heads, how wide is the MLP's hidden layer, "
                                    "and how big is each head?",
             "answer": "1,024 and 64.", "why": "4 × 256 = 1,024 and 256 / 4 = 64.", "key": {'parts': [{'label': 'MLP hidden', 'value': 1024, 'tol': 0.5, 'unit': None}, {'label': 'head size', 'value': 64, 'tol': 0.5, 'unit': None}]}},
            {"kind": "tf", "q": "“In each block, the attention output replaces <code>x</code>.”",
             "answer": "False.", "why": "It is <i>added</i>: <code>x = x + self.attn(self.ln1(x))</code>. That running "
                                        "sum is the residual stream.", "key": {'value': False}},
        ],
    },
    {
        "title": "Logits out, and 818,241 parameters",
        "segment": (63, 81),
        "figures": [{"t": 70.4, "caption": "One last layer norm, then a linear layer: 65 logits, one per character."},
                    {"t": 81.0, "caption": "Parameters, on a log scale: from 818,241 to hundreds of billions."}],
        "body": [
            """<p>After the blocks come one last <b>layer norm</b> (<code>ln_f</code>) and a <b>linear layer</b>
(<code>head</code>) that turns each 128-number vector into <b>65 logits</b>, one per character. Softmax turns them into
next-character probabilities (episode 10).</p>""",
            '<div class="figrow">{fig0}{fig1}</div>',
            """<p>That is the whole model: <b>818,241 parameters</b>. GPT-2 small has 124 million, and the largest
models today have hundreds of billions. You can count them yourself: <code>nn.Linear(a, b)</code> has a × b weights
plus b biases, <code>nn.LayerNorm(128)</code> has 128 + 128, and <code>nn.Embedding(n, 128)</code> has n × 128.</p>""",
            """<div class="box key"><b class="t">Key idea</b>Output: <code>ln_f</code> + <code>head</code>, 128 → <b>65
logits</b> per position. In total <b>818,241 parameters</b>, almost all of them inside the 4 blocks.</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "How many parameters does <code>head = nn.Linear(128, 65)</code> have?",
             "answer": "8,385.", "why": "128 × 65 = 8,320 weights plus 65 biases.", "key": {'parts': [{'label': None, 'value': 8385, 'tol': 0.5, 'unit': None}]}},
            {"kind": "mc", "q": "For each position, what comes out of the model?",
             "options": ["The ID of the next character", "65 logits, one per character; softmax turns them into "
                         "probabilities", "A vector of 128 numbers", "64 probabilities, one per position"],
             "answer": "B.", "why": "<code>head</code> maps 128 numbers to <code>vocab_size</code> = 65 scores. "
                                   "Picking a character happens later, in <code>generate</code>.", "key": {'choice': 1}},
            {"kind": "number", "q": "About how many times more parameters does GPT-2 small (124 million) have?",
             "answer": "About 150 times.", "why": "124,000,000 / 818,241 ≈ 152: a bit more than two steps of ×10 on "
                                                  "the log scale.", "key": {'parts': [{'label': None, 'value': 150, 'tol': 3, 'unit': None}]}},
            {"kind": "code", "q": "<b>Try it yourself.</b> Count the parameters by hand, then let PyTorch check. The "
                                  "code builds the layers of one block with the real sizes. (a) Before running, work out "
                                  "<code>count(qkv)</code>. (b) Predict the second number printed: one whole block. "
                                  "(c) Add up the model: 4 blocks, the two embedding tables (concept 2), the final "
                                  "layer norm and the head (4.1). Do you get 818,241? What share is in the blocks?",
             "code": """import torch.nn as nn

n_embd, n_head, n_layer = 128, 4, 4
count = lambda m: sum(p.numel() for p in m.parameters())

qkv  = nn.Linear(n_embd, 3 * n_embd)      # Attention.qkv
proj = nn.Linear(n_embd, n_embd)          # Attention.proj
up   = nn.Linear(n_embd, 4 * n_embd)      # MLP.up
down = nn.Linear(4 * n_embd, n_embd)      # MLP.down
ln   = nn.LayerNorm(n_embd)               # a block has two: ln1, ln2

block = count(qkv) + count(proj) + count(up) + count(down) + 2 * count(ln)
print(count(qkv), block)""",
             "answer": "(a) 49,536 · (b) 198,272 · (c) 4 × 198,272 + 8,320 + 8,192 + 256 + 8,385 = 818,241; "
                       "about 97 % in the blocks",
             "why": """(a) 128 × 384 = 49,152 weights + 384 biases. (b) qkv 49,536 + proj 16,512 + up 66,048 +
down 65,664 + 2 × 256. (c) The blocks hold 793,088 of the 818,241. The causal mask is not counted: it is a fixed buffer
(<code>register_buffer</code>), not a learned parameter."""},
        ],
    },
    {
        "title": "Training on an ordinary CPU",
        "segment": (81, 95),
        "figures": [{"t": 95.0, "size": "small",
                     "caption": "The loop of episode 11, in tiny_gpt.py."}],
        "body": [
            """<p>Training is the loop of episode 11, line for line: each step grabs <b>32 random chunks of 64
characters</b>, predicts <b>every next character</b>, computes the cross-entropy loss, backpropagates and steps with
<b>AdamW</b>, a variant of Adam. The targets <code>y</code> are the same chunks one character further along:</p>""",
            "{fig0}",
            """<pre class="code">def get_batch(d):
    ix = torch.randint(len(d) - block_size - 1, (batch_size,))
    x = torch.stack([d[i:i + block_size] for i in ix])
    y = torch.stack([d[i + 1:i + block_size + 1] for i in ix])   # the next character
    return x, y</pre>""",
            """<div class="box key"><b class="t">Key idea</b>Each step: 32 chunks × 64 characters, <b>every position
predicts its next character</b>. The first 90 % of the text is for training; <b>5,000 steps took about 45 minutes</b>
on an ordinary 8-core CPU.</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "How many next-character predictions does one step score? About how many in "
                                    "5,000 steps?",
             "answer": "2,048 per step; about 10 million in all.", "why": "32 chunks × 64 positions = 2,048, and "
                                                                          "2,048 × 5,000 = 10,240,000.", "key": {'parts': [{'label': 'per step', 'value': 2048, 'tol': 0.5, 'unit': None}, {'label': '5,000 steps (millions)', 'value': 10, 'tol': 0.5, 'unit': None}]}},
            {"kind": "short", "q": "With <code>block_size</code> = 4, a chunk starts at the <i>F</i> of <i>“First "
                                   "Citizen:”</i>. What are <code>x</code> and <code>y</code>?",
             "answer": "x = “Firs”, y = “irst”.",
             "why": "y starts one character later: each character of y is the one that follows the matching "
                    "character of x."},
            {"kind": "number", "q": "<code>logits</code> is 32 × 64 × 65. What shape is "
                                    "<code>logits.view(-1, vocab_size)</code>?",
             "answer": "2,048 × 65.", "why": "All positions of all chunks in one list: one row of 65 logits per "
                                            "prediction, and the loss is the average over the 2,048 rows.", "key": {'parts': [{'label': 'rows', 'value': 2048, 'tol': 0.5, 'unit': None}, {'label': 'cols', 'value': 65, 'tol': 0.5, 'unit': None}]}},
            {"kind": "number", "q": "Of the 1,115,394 characters, the first 90 % are for training. How many does the "
                                    "model never train on?",
             "answer": "111,540.", "why": "<code>int(0.9 × 1,115,394)</code> = 1,003,854 for training; the other "
                                         "111,540 are the validation text.", "key": {'parts': [{'label': None, 'value': 111540, 'tol': 0.5, 'unit': None}]}},
        ],
    },
    {
        "title": "Watching it learn",
        "segment": (95, 137),
        "figures": [{"t": 109.0, "caption": "Step 250 (validation loss 2.20): made-up words."},
                    {"t": 119.8, "caption": "Step 5,000 (1.59): real words and names."}],
        "body": [
            """<p>The program prints samples as it trains (temperature 0.8, episode 10). <b>Step 0</b> (validation loss
4.41): pure noise, like <code>RYcb!FkBYGF</code>. <b>Step 250</b> (2.20): common letters, spaces in the right places,
short lines, but made-up words such as <i>thamfors</i>. <b>Step 5,000</b> (1.59): real words, character names such as
GLOUCESTER, and the shape of a play.</p>""",
            '<div class="figrow">{fig0}{fig1}</div>',
            """<p>It isn't Shakespeare, but it learned all this from nothing but <b>predicting the next
character</b>. Real LLMs use the <b>same recipe</b> with far more data, layers and compute; the two bonus episodes show
how they generate text quickly and how a raw GPT becomes a chatbot.</p>""",
            """<div class="box key"><b class="t">Key idea</b>Noise → common letters and spaces → words and structure,
learned only from next-character prediction. Real LLMs: <b>the same recipe, much bigger</b>.</div>""",
        ],
        "exercises": [
            {"kind": "order", "q": "Put these real samples in training order: "
                                   "<i>(A)</i> <code>Reabe my thamfors be make hakeate</code> · <i>(B)</i> "
                                   "<code>GLOUCESTER: Then shall we are 'twere?</code> · <i>(C)</i> "
                                   "<code>RYcb!FkBYGF</code>.",
             "answer": "C → A → B.", "why": "Noise at step 0, made-up words at step 250, real words and a character "
                                           "name at step 5,000.", "key": {'items': ['(C) RYcb!FkBYGF', '(A) Reabe my thamfors be make hakeate', "(B) GLOUCESTER: Then shall we are 'twere?"]}},
            {"kind": "number", "q": "Which p, given to the correct character every time, matches the loss at step 0 "
                                    "(4.41) and at step 5,000 (1.59)?",
             "answer": "About 0.012 (1 in 82) at the start, 0.20 (1 in 5) at the end.",
             "why": "loss = −ln p, so p = e<sup>−loss</sup>: e<sup>−4.41</sup> ≈ 0.012 and e<sup>−1.59</sup> ≈ 0.20.", "key": {'parts': [{'label': 'step 0', 'value': 0.012, 'tol': 0.0005, 'unit': None}, {'label': 'step 5,000', 'value': 0.2, 'tol': 0.005, 'unit': None}]}},
            {"kind": "short", "q": "At step 250 the model writes <i>thamfors</i> and <i>hakeate</i>. What has it "
                                   "learned, and what not yet?",
             "answer": "Common letters, spaces, short lines; not yet real words.",
             "why": "The easy statistics of the text come first; words and names need more training."},
            {"kind": "tf", "q": "“Real LLMs are trained with a completely different recipe from the tiny GPT.”",
             "answer": "False.", "why": "Same recipe: next-token prediction, the same kind of blocks, the same training "
                                        "loop. Just far more data, layers and compute.", "key": {'value': False}},
        ],
    },
]
