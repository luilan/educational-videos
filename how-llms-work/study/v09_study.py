"""Study guide content for How LLMs Work, episode 9: The Transformer Block.

Build:  python framework/study_guide.py how-llms-work v09
Figures: {"t": seconds into the rendered video, "caption": ...}; placed in body blocks with {fig0}, {fig1}, ...
Exercise kinds: mc (options), tf, short, number, order, code. Every exercise has an answer and a why.
"""

LESSON = {
    "series": "How LLMs Work",
    "label": "Episode 9",
    "title": "The Transformer Block",
    "tagline": "Attention and MLP, wired together and stacked",
    "duration": "1:58",
    "intro": """<p>We have both halves of a layer: <b>attention</b>, where tokens share information, and the <b>MLP</b>,
where each token processes it. This lesson wires them into a <b>transformer block</b>: each half <b>adds</b> its result
to a stream of vectors, with a <b>layer norm</b> in front of it. Then the block is <b>stacked</b>, 12 times in GPT-2
small.</p>""",
    "prereq": """<div class="box note"><b>Before you start:</b> this lesson builds on episodes 3–4 (each token becomes a
vector, plus its position), 5–7 (attention) and 8 (the MLP). For the maths, Foundations F01 (Vectors), F04 (Straight
Lines and Bends), F08 (Averages and Spread: the mean and standard deviation used by layer norm) and F11 (NumPy in Three
Minutes, for the code in concept 6) help.</div>""",
}

CODE = """def layer_norm(x, gain, bias, eps=1e-5):
    mean = x.mean(-1, keepdims=True)
    var = x.var(-1, keepdims=True)
    return gain * (x - mean) / np.sqrt(var + eps) + bias

def block(x, p):
    x = x + attention(layer_norm(x, *p.ln1), *p.attn)   # tokens talk
    x = x + mlp(layer_norm(x, *p.ln2), *p.mlp)          # each token thinks
    return x

for p in blocks:  # 12 blocks in GPT-2 small
    x = block(x, p)"""

CONCEPTS = [
    {
        "title": "The residual stream",
        "segment": (8, 26),
        "figures": [{"t": 14.8, "caption": "The two halves: attention (tokens talk) and the MLP (each token thinks)."},
                    {"t": 26.4, "caption": "The residual stream: each token's vector flows from its embedding to the "
                                           "prediction."}],
        "body": [
            """<p>A layer has two halves: <b>attention</b>, where tokens share information, and the <b>MLP</b>, where each
token processes it. The central idea for wiring them together is the <b>residual stream</b>.</p>""",
            '<div class="figrow">{fig0}{fig1}</div>',
            """<p>Picture each token's vector flowing along a stream, from its <b>embedding</b> at the start to the
<b>prediction</b> at the end. Every half along the way works on this vector (how, is concept 2).</p>""",
            """<div class="box key"><b class="t">Key idea</b>Each token has one vector that <b>flows through the whole
model</b>, from embedding to prediction: the <b>residual stream</b>. Attention and the MLP work on it along the
way.</div>""",
        ],
        "exercises": [
            {"kind": "mc", "q": "What flows along the residual stream?",
             "options": ["Each token's vector", "The probabilities for the next word", "The text, as a list of words",
                         "The attention weights of the first layer"],
             "answer": "A.", "why": "The stream carries each token's vector, from its embedding to the prediction."},
            {"kind": "order", "q": "Put these in the order a token's vector meets them: <i>prediction · embedding · "
                                   "attention and MLP halves</i>.",
             "answer": "embedding → attention and MLP halves → prediction.",
             "why": "The stream starts at the embedding and ends at the prediction; the halves sit along it."},
            {"kind": "short", "q": "Which half lets tokens share information, and which processes each token on its "
                                   "own?",
             "answer": "Attention shares; the MLP processes each token alone.",
             "why": "Attention is where tokens talk; the MLP is where each token thinks (episode 8)."},
            {"kind": "tf", "q": "“A text of 6 tokens has 6 vectors flowing along the residual stream, one per token.”",
             "answer": "True.", "why": "Each token's vector flows along the stream, from its own embedding to the "
                                       "prediction."},
        ],
    },
    {
        "title": "Add, don't replace",
        "segment": (27, 49),
        "figures": [{"t": 38.3, "caption": "Each half reads from the stream, computes, and adds its result back at ⊕."},
                    {"t": 49.3, "caption": "The unbroken stream is a direct path back through the network: a "
                                           "“gradient highway”."}],
        "body": [
            """<p>Attention and the MLP <b>don't replace</b> the vector. Each one <b>reads</b> from the stream, computes
something, and <b>adds</b> its result back: <code>x = x + attention(x)</code>, then <code>x = x + mlp(x)</code>.</p>""",
            '<div class="figrow">{fig0}{fig1}</div>',
            """<p>Adding matters. It <b>keeps the original information</b> around, so a half only has to add what is new,
and it gives training a <b>direct path back</b> through the network. That is a big part of what makes <b>deep
stacks</b> of layers trainable (training is episode 11).</p>""",
            """<div class="box key"><b class="t">Key idea</b>Each half <b>adds</b> to the stream:
<code>x = x + f(x)</code>. The original vector is never thrown away.</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "x = [1, 2]. Attention returns [0.5, −1]; then the MLP returns [0, 3]. What is x "
                                    "after each step?",
             "answer": "[1.5, 1], then [1.5, 4].", "why": "x + attention(x) = [1 + 0.5, 2 − 1]; then add [0, 3]."},
            {"kind": "tf", "q": "“After the attention half, the stream holds only attention's output; the original "
                                "vector is gone.”",
             "answer": "False.", "why": "The output is added to x, so the original is still part of the sum."},
            {"kind": "short", "q": "A half has learned nothing useful yet and always outputs zeros. What happens to x "
                                   "with <code>x&nbsp;=&nbsp;x&nbsp;+&nbsp;f(x)</code>? And with <code>x&nbsp;=&nbsp;f(x)</code>?",
             "lines": 2,
             "answer": "Adding: x passes through unchanged. Replacing: x becomes all zeros.",
             "why": "Adding lets information flow past a half that has nothing to say; replacing wipes it out."},
            {"kind": "mc", "q": "Which two reasons does the video give for adding?",
             "options": ["It makes the model faster and smaller",
                         "It keeps the original information, and gives training a direct path back through the network",
                         "It turns vectors into probabilities and removes the need for training",
                         "It lets the MLP see the other tokens"],
             "answer": "B.", "why": "Keeping the information and the direct path for training are why deep stacks can "
                                   "be trained."},
        ],
    },
    {
        "title": "Layer norm",
        "segment": (50, 65),
        "figures": [{"t": 64.3, "caption": "A layer norm (LN) sits before each half. [2, 4, 6, 8] becomes mean 0, "
                                           "standard deviation 1, then gets a learned gain and bias."}],
        "body": [
            """<p>Before each half there is a <b>layer norm</b> (LN). It rescales a vector so its numbers have a
<b>mean of 0</b> and a <b>standard deviation of 1</b> (subtract the mean, divide by the standard deviation), then
applies a learned scale and shift: the <b>gain</b> and the <b>bias</b>.</p>""",
            "{fig0}",
            """<p>Example: [2, 4, 6, 8] has mean 5. The squared distances from the mean are 9, 1, 1, 9, which average 5, so
the standard deviation is √5 ≈ 2.24. Then [−3, −1, 1, 3] / 2.24 = <b>[−1.34, −0.45, 0.45, 1.34]</b>. This keeps the
numbers in a healthy range as they flow through many layers.</p>""",
            """<div class="box key"><b class="t">Key idea</b>LN(x) = gain × (x − mean) / std + bias. Before the gain
and bias, the output always has <b>mean 0 and standard deviation 1</b>.</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "Apply layer norm (gain 1, bias 0) to [1, 3].",
             "answer": "[−1, 1].", "why": "The mean is 2; both numbers are 1 away from it, so the standard deviation is "
                                         "1; (x − 2) / 1."},
            {"kind": "short", "q": "Without calculating: what does layer norm (gain 1, bias 0) give for "
                                   "[12, 14, 16, 18]? And for [20, 40, 60, 80]?",
             "answer": "Both give [−1.34, −0.45, 0.45, 1.34].",
             "why": "They are [2, 4, 6, 8] shifted by 10 and scaled by 10. Subtracting the mean removes any shift; "
                    "dividing by the standard deviation removes any scale."},
            {"kind": "number", "q": "Apply a gain of 2 and a bias of 1 to [−1.34, −0.45, 0.45, 1.34]. Give each number "
                                    "to one decimal.",
             "answer": "[−1.7, 0.1, 1.9, 3.7].",
             "why": "2 × value + 1. The learned gain and bias let the model pick its own scale and shift."},
            {"kind": "tf", "q": "“Layer norm is applied just once, at the start of the model.”",
             "answer": "False.", "why": "There is one before each half: two in every block."},
        ],
    },
    {
        "title": "Wiring the block",
        "segment": (65, 71),
        "figures": [{"t": 70.5, "caption": "One transformer block: layer norm, attention, add; layer norm, MLP, add."}],
        "body": [
            """<p>Put it together and you have a <b>transformer block</b>: <b>layer norm, attention, add</b>; then
<b>layer norm, MLP, add</b>. With the layer norms, the two lines are:</p>""",
            """<pre class="code">x = x + attention(layer_norm(x))   # tokens talk
x = x + mlp(layer_norm(x))         # each token thinks</pre>""",
            "{fig0}",
            """<p>Notice where the layer norm sits: it normalizes the <b>copy that goes into each half</b>, not the stream
itself. The stream only ever has things <b>added</b> to it. (This arrangement is called <b>pre-norm</b>.)</p>""",
            """<div class="box key"><b class="t">Key idea</b>Block = <b>LN → attention → add, LN → MLP → add</b>. Two
halves, two layer norms, two additions.</div>""",
        ],
        "exercises": [
            {"kind": "order", "q": "Put the six steps of one block in order: <i>add · MLP · layer norm · attention · "
                                   "add · layer norm</i>.",
             "answer": "layer norm → attention → add → layer norm → MLP → add.",
             "why": "Each half: normalize a copy, compute, add the result to the stream."},
            {"kind": "mc", "q": "Which line is the first half of the block?",
             "options": ["<code>x = attention(layer_norm(x))</code>", "<code>x = x + attention(layer_norm(x))</code>",
                         "<code>x = layer_norm(x) + attention(x)</code>", "<code>x = x + layer_norm(x)</code>"],
             "answer": "B.", "why": "A replaces x instead of adding; C puts a normalized copy into the stream and "
                                   "feeds attention an un-normalized x; D has no attention at all."},
            {"kind": "tf", "q": "“The layer norm changes the vector stored in the residual stream.”",
             "answer": "False.", "why": "It normalizes the copy fed into the half. Only the half's output is added to "
                                        "the stream."},
        ],
    },
    {
        "title": "Stack it",
        "segment": (71, 95),
        "figures": [{"t": 82.7, "caption": "GPT-2 small stacks 12 blocks, each with its own weights."},
                    {"t": 94.3, "caption": "A rough pattern: earlier layers handle local things, later layers "
                                           "abstract meaning."}],
        "body": [
            """<p>Now <b>stack</b> the block. GPT-2 small has <b>12 blocks</b>; the largest GPT-2 has <b>48</b>. Every
block has <b>its own weights</b>, and each one refines the vectors a little more as they flow along the stream.</p>""",
            '<div class="figrow">{fig0}{fig1}</div>',
            """<p>Researchers see a <b>rough pattern</b>. Earlier layers tend to handle <b>local</b> things, like grammar
and nearby words. Later layers deal with more <b>abstract meaning</b>, and with <b>predicting the next token</b>.</p>""",
            """<div class="box key"><b class="t">Key idea</b>The same design, repeated N times, with <b>different weights
in every block</b>. Early blocks: local patterns. Late blocks: meaning and the next token.</div>""",
        ],
        "exercises": [
            {"kind": "tf", "q": "“The 12 blocks of GPT-2 small are copies of one block that share the same weights.”",
             "answer": "False.", "why": "Same design, but every block has its own weights."},
            {"kind": "number", "q": "From episode 8, one GPT-2 small layer has 4,718,592 MLP weights and 2,359,296 "
                                    "attention weights. About how many weights do the 12 blocks hold together "
                                    "(ignoring biases and layer norms)?",
             "answer": "84,934,656, about 85 million.", "why": "12 × (4,718,592 + 2,359,296) = 12 × 7,077,888."},
            {"kind": "number", "q": "How many layer norms are inside GPT-2 small's 12 blocks? And inside the largest "
                                    "GPT-2's 48?",
             "answer": "24 and 96.", "why": "Two per block, one before each half."},
            {"kind": "mc", "q": "According to the rough pattern, which job is more typical of the later layers?",
             "options": ["Grammar and nearby words", "Abstract meaning and predicting the next token",
                         "Splitting the text into tokens", "Looking up each token's embedding"],
             "answer": "B.", "why": "A is typical of the earlier layers; C and D happen before the first block."},
        ],
    },
    {
        "title": "The block in code",
        "segment": (95, 116),
        "figures": [{"t": 106.2, "size": "small", "caption": "The code from the video: layer norm, a block, and the "
                                                             "loop over all blocks."}],
        "body": [
            """<p>In code, a block is two lines: add attention of the normalized input, then add the MLP. The layer norm
is a few more lines, and the <b>whole model is just a loop</b> over the blocks.</p>""",
            "{fig0}",
            f'<pre class="code">{CODE}</pre>',
            """<p>After the last block, each token's vector holds the model's view of <b>what comes next</b>. But it is
still a vector: episode 10 turns it back into words.</p>""",
            """<div class="box key"><b class="t">Key idea</b><code>for p in blocks: x = block(x, p)</code>. The body of
the model is one loop, and each block is two “normalize, compute, add” steps.</div>""",
        ],
        "exercises": [
            {"kind": "short", "q": "What would go wrong if you deleted <code>x +</code> from both lines of "
                                   "<code>block</code>?",
             "lines": 2,
             "answer": "Each half would replace the vector instead of adding to it.",
             "why": "The original information would be lost at every half, and training would lose its direct path "
                    "back through the network (concept 2)."},
            {"kind": "tf", "q": "“After the last block, each token's vector is already a list of probabilities for "
                                "the next word.”",
             "answer": "False.", "why": "It is still a vector (768 numbers in GPT-2 small). Episode 10 turns it into "
                                        "probabilities."},
            {"kind": "code", "q": "<b>Try it yourself.</b> Below is the video's <code>layer_norm</code> and a toy block "
                                  "whose attention has learned nothing yet (it returns zeros) and whose “MLP” just "
                                  "returns 0.1 times its input. (a) Predict the first two printed lines. (b) Predict "
                                  "the third. Why is it not all zeros, although attention returns zeros? (c) Delete "
                                  "<code>x +</code> from both lines of <code>block</code> and run again. What comes "
                                  "out, and why?",
             "code": """import numpy as np

def layer_norm(x, gain, bias, eps=1e-5):
    mean = x.mean(-1, keepdims=True)
    var = x.var(-1, keepdims=True)
    return gain * (x - mean) / np.sqrt(var + eps) + bias

def attention(z): return np.zeros_like(z)   # has learned nothing yet
def mlp(z):       return 0.1 * z            # a tiny toy "MLP"

def block(x):
    x = x + attention(layer_norm(x, 1.0, 0.0))
    x = x + mlp(layer_norm(x, 1.0, 0.0))
    return x

x = np.array([2.0, 4.0, 6.0, 8.0])
y = layer_norm(x, 1.0, 0.0)
print(np.round(y, 2), round(y.mean(), 3), round(y.std(), 3))
print(np.round(layer_norm(x, 2.0, 1.0), 2))
print(np.round(block(x), 3))""",
             "answer": "(a) [-1.34 -0.45  0.45  1.34] 0.0 1.0 and [-1.68  0.11  1.89  3.68] · "
                       "(b) [1.866 3.955 6.045 8.134] · (c) [0. 0. 0. 0.]",
             "why": """(a) The example from concept 3: mean 0, standard deviation 1; then 2 × value + 1. (b) Attention
adds zeros, so x passes through unchanged; the MLP then adds 0.1 × [−1.34, −0.45, 0.45, 1.34]: a small refinement of
the original vector. (c) Without <code>x +</code>, attention replaces x with zeros, and nothing can bring the
information back: layer norm of all zeros is zeros (the tiny <code>eps</code> avoids dividing by zero), and 0.1 × 0
is 0."""},
        ],
    },
]
