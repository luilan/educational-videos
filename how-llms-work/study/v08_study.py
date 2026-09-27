"""Study guide content for How LLMs Work, episode 8: The MLP: Where Facts Live.

Build:  python framework/study_guide.py how-llms-work v08
Figures: {"t": seconds into the rendered video, "caption": ...}; placed in body blocks with {fig0}, {fig1}, ...
Exercise kinds: mc (options), tf, short, number, order, code. Every exercise has an answer and a why.
"""

LESSON = {
    "series": "How LLMs Work",
    "label": "Episode 8",
    "title": "The MLP: Where Facts Live",
    "tagline": "The part of the model that thinks on its own",
    "duration": "2:16",
    "intro": """<p>Attention moves information between tokens. This lesson is about the other half of every layer: the
<b>MLP</b>, which works on <b>each token's vector by itself</b>. It expands the vector, bends it with a simple nonlinear
function, and projects it back down. It holds most of each layer's weights, and much of the model's factual knowledge
seems to live there.</p>""",
    "prereq": """<div class="box note"><b>Before you start:</b> this lesson builds on episode 3 (every token is a vector of
768 numbers in GPT-2 small) and episodes 5–7 (attention). The maths is dot products and matrix multiplication:
Foundations F02 (The Dot Product), F03 (Matrices), F04 (Straight Lines and Bends) and F12 (Neural Networks in Three
Minutes) cover everything used here.</div>""",
}

CODE = """def mlp(x, W1, b1, W2, b2):
    h = gelu(x @ W1 + b1)      # expand 768 -> 3072, then bend
    return h @ W2 + b2         # project back 3072 -> 768"""

CONCEPTS = [
    {
        "title": "One token at a time",
        "segment": (8, 21),
        "figures": [{"t": 20.8, "caption": "Every token's vector goes through its own copy of the MLP, and every copy "
                                           "has the same weights."}],
        "body": [
            """<p>Attention moves information <b>between</b> tokens. The <b>MLP</b> does a different kind of work: it takes
<b>each token's vector on its own</b> and transforms it, without looking at any other token. Every position goes
through <b>the same MLP, with the same weights</b>.</p>""",
            "{fig0}",
            """<div class="box key"><b class="t">Key idea</b>Attention: tokens talk to each other. MLP: each token is
processed <b>by itself</b>, with the <b>same weights at every position</b>. It is also where much of the model's
<b>factual knowledge</b> seems to live (concept 6).</div>""",
        ],
        "exercises": [
            {"kind": "mc", "q": "Which sentence describes the MLP?",
             "options": ["It moves information between tokens",
                         "It processes each token's vector on its own, with the same weights at every position",
                         "It uses a different set of weights for each position in the text",
                         "It only processes the last token of the text"],
             "answer": "B.", "why": "A describes attention. C and D are wrong: it is one MLP, applied to every "
                                   "position."},
            {"kind": "tf", "q": "“Inside the MLP, the vector for <i>sat</i> can read information from the vector for "
                                "<i>cat</i>.”",
             "answer": "False.", "why": "The MLP sees only one token's vector. Moving information between tokens is "
                                        "attention's job."},
            {"kind": "number", "q": "A text has 5 tokens. In one layer, (a) how many times is the MLP applied, and "
                                    "(b) how many different sets of MLP weights are used?",
             "answer": "(a) 5 times. (b) 1 set.", "why": "One application per token, but it is the same MLP (the same "
                                                         "weights) every time."},
            {"kind": "short", "q": "Two tokens reach a layer's MLP with exactly the same vector. What can you say "
                                   "about the two outputs?",
             "answer": "They are identical.", "why": "The MLP sees only the vector and uses the same weights everywhere, "
                                                     "so the same input always gives the same output."},
        ],
    },
    {
        "title": "Expand, bend, project back",
        "segment": (21, 43),
        "figures": [{"t": 43.3, "caption": "The three steps, with the sizes in GPT-2 small: 768 numbers in, 3,072 in "
                                           "the middle, 768 out."}],
        "body": [
            """<p><b>MLP</b> stands for <b>multi-layer perceptron</b>. Here it is just three steps: <b>expand</b> the vector
to <b>four times</b> its size (a matrix multiplication), apply a simple <b>nonlinear function</b> to every number (the
“bend”), then <b>project</b> it back down to its original size (a second matrix multiplication).</p>""",
            "{fig0}",
            """<p>In GPT-2 small that is <b>768 → 3,072 → 768</b>.</p>""",
            """<div class="box key"><b class="t">Key idea</b>MLP = <b>expand (4×) → bend → project back</b>. The vector
leaves the MLP with the same size it came in with.</div>""",
        ],
        "exercises": [
            {"kind": "order", "q": "Put the three MLP steps in order: <i>project back down · expand to 4× · apply the "
                                   "nonlinear function</i>.",
             "answer": "expand to 4× → apply the nonlinear function → project back down.",
             "why": "Expand, bend, project."},
            {"kind": "number", "q": "GPT-2 small has vectors of 768 numbers. (a) How many numbers does the expanded "
                                    "(hidden) vector have? (b) How many does the MLP's output have?",
             "answer": "(a) 3,072. (b) 768.", "why": "4 × 768 = 3,072, and the projection brings it back to 768."},
            {"kind": "number", "q": "Another model has vectors of 1,024 numbers and uses the same 4× rule. How many "
                                    "numbers are in one token's hidden vector? And in the hidden vectors of a "
                                    "10-token text, all together?",
             "answer": "4,096 per token; 40,960 for 10 tokens.",
             "why": "4 × 1,024 = 4,096, and every token gets its own hidden vector: 10 × 4,096."},
            {"kind": "tf", "q": "“The MLP makes the vector permanently bigger: a token enters with 768 numbers and "
                                "leaves with 3,072.”",
             "answer": "False.", "why": "The 3,072 numbers exist only inside the MLP. The projection brings the vector "
                                        "back to 768."},
        ],
    },
    {
        "title": "Neurons as detectors",
        "segment": (44, 60),
        "figures": [{"t": 60.2, "caption": "Each hidden number is a neuron: a dot product w · x that measures how far the "
                                           "token's vector x points along the neuron's direction w."}],
        "body": [
            """<p>Each of the 3,072 hidden numbers is a <b>neuron</b>. A neuron has its own weight vector <b>w</b> (768
numbers, one column of W1) and takes a <b>dot product</b> with the token's vector: <b>neuron = w · x</b> (plus a bias).
It measures <b>how far x points along w</b>: large when they point the same way, zero at right angles, negative when
they point in opposite directions.</p>""",
            "{fig0}",
            """<p>So each neuron is like a <b>question</b> about the token: <i>is this about animals?</i> <i>Is this the
end of a sentence?</i> (illustrative labels).</p>""",
            """<div class="box key"><b class="t">Key idea</b>A neuron is <b>a dot product</b> with its own direction w. It
gives a big number when the token's vector <b>points along w</b>: a detector for one pattern.</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "A tiny neuron has w = [1, 0, 2]. Compute w · x for x = [0.5, −1, 1] and for "
                                    "x′ = [−1, 2, −1]. Which of the two tokens matches the neuron's pattern?",
             "answer": "2.5 and −3. The first (x).",
             "why": "1·0.5 + 0·(−1) + 2·1 = 2.5 and 1·(−1) + 0·2 + 2·(−1) = −3. A large positive value means the "
                    "vector points along w."},
            {"kind": "mc", "q": "Four token vectors all have the same length. Which one gives the neuron the largest "
                                "value?",
             "options": ["The one pointing the same way as w", "The one pointing the opposite way to w",
                         "The one at right angles to w", "They all give the same value"],
             "answer": "A.", "why": "Aligned gives a large positive dot product; right angles gives 0; opposite gives a "
                                   "negative one."},
            {"kind": "tf", "q": "“In GPT-2 small, each neuron's weight vector w has 768 numbers, and each MLP has "
                                "3,072 neurons.”",
             "answer": "True.", "why": "w must be as long as the token vector (768). The 3,072 such vectors together "
                                       "are the expand matrix: 3,072 × 768 = 2,359,296 weights."},
        ],
    },
    {
        "title": "GELU: quiet unless the pattern shows up",
        "segment": (61, 74),
        "figures": [{"t": 73.8, "caption": "GELU: positive inputs pass through (≈ x), negative inputs are squashed "
                                           "to ≈ 0. A neuron at −2 stays quiet; one at 2.5 fires."}],
        "body": [
            """<p>The nonlinear function is usually <b>GELU</b>, applied to every neuron separately. <b>Large positive</b>
inputs pass through almost unchanged (GELU(2.5) ≈ 2.5), and <b>negative</b> inputs are squashed close to zero
(GELU(−2) ≈ −0.05). Around zero it bends smoothly from one behaviour to the other.</p>""",
            "{fig0}",
            """<p>So a neuron <b>stays quiet</b> unless its pattern really shows up: only a clearly positive match is passed
on. The dashed line in the video is <b>ReLU</b>, max(0, x), a simpler cousin with a sharp corner at 0.</p>""",
            """<div class="box key"><b class="t">Key idea</b>GELU(x) ≈ x for large positive x and ≈ 0 for negative x.
A neuron <b>fires</b> only when its dot product is clearly positive.</div>""",
        ],
        "exercises": [
            {"kind": "mc", "q": "Which value is closest to GELU(3)?",
             "options": ["0", "1", "3", "9"],
             "answer": "C.", "why": "Large positive inputs pass through almost unchanged: GELU(3) ≈ 2.996."},
            {"kind": "number", "q": "Four neurons have dot products 2.5, −2, 4 and −3. Roughly what are the four values "
                                    "after GELU? Which neurons fire?",
             "lines": 2,
             "answer": "≈ 2.5, ≈ −0.05, ≈ 4, ≈ 0. The first and the third fire.",
             "why": "Exact values: 2.485, −0.045, 4.000, −0.004. Positive inputs pass, negative ones are squashed."},
            {"kind": "tf", "q": "“GELU turns every negative input into exactly 0.”",
             "answer": "False.", "why": "It squashes them <i>close to</i> zero (−0.05 at −2), not exactly to zero. "
                                        "Exactly 0 for every negative input is what ReLU does."},
        ],
    },
    {
        "title": "Why the bend matters",
        "segment": (74, 86),
        "figures": [{"t": 81.6, "caption": "Without the bend, two matrices collapse into one, and so does any stack "
                                           "of them."},
                    {"t": 85.4, "caption": "No bend: only straight lines. With the bend: curves."}],
        "body": [
            """<p>Without the bend, the MLP would be two matrix multiplications in a row, and those <b>collapse into
one</b>: W₂(W₁x) = (W₂W₁)x = Wx. Stacking more layers would add nothing: a chain of matrices is still one matrix,
which can only describe <b>straight lines</b>.</p>""",
            '<div class="figrow">{fig0}{fig1}</div>',
            """<div class="box key"><b class="t">Key idea</b>A matrix times a matrix is just another matrix. The
<b>nonlinear bend</b> between them breaks the collapse and lets the network learn <b>curves</b>: more than straight
lines.</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "Let W₁ = [[1, 2], [0, 1]] and W₂ = [[2, 0], [1, 1]] (written row by row). "
                                    "(a) Compute W = W₂W₁. (b) For x = [1, 1], check that W₂(W₁x) and Wx give the "
                                    "same result.",
             "lines": 2,
             "answer": "(a) W = [[2, 4], [1, 3]]. (b) Both give [6, 4].",
             "why": "W₁x = [3, 1], then W₂[3, 1] = [6, 4]; and Wx = [2 + 4, 1 + 3] = [6, 4]. Two matrices in a row act "
                    "like one."},
            {"kind": "short", "q": "In GPT-2 small, W1 is 768 × 3,072 and W2 is 3,072 × 768. If you removed GELU, "
                                   "the MLP would collapse into a single matrix. What shape, and how many numbers?",
             "answer": "768 × 768: 589,824 numbers.",
             "why": "Without the bend, the 4.7 million weights could only do what 589,824 can: the expansion would be "
                    "wasted."},
            {"kind": "tf", "q": "“Ten matrix multiplications in a row, with no nonlinearity between them, can learn "
                                "curves that a single matrix cannot.”",
             "answer": "False.", "why": "The ten matrices collapse into one, which can still only fit straight lines."},
            {"kind": "mc", "q": "What is the job of the nonlinear function in the MLP?",
             "options": ["To make the vector four times bigger",
                         "To stop the two matrix multiplications collapsing into one, so the network can learn curves",
                         "To turn the vector into probabilities", "To move information between tokens"],
             "answer": "B.", "why": "A is the expand matrix, C comes at the very end of the model (episode 10), "
                                   "D is attention."},
        ],
    },
    {
        "title": "Recalling facts",
        "segment": (86, 104),
        "figures": [{"t": 103.7, "caption": "Simplified picture: the “Eiffel Tower” neuron fires, and the down "
                                            "projection adds a push that means Paris."}],
        "body": [
            """<p>The <b>down projection</b> (the second matrix, W2) turns each <b>active</b> neuron into a <b>push</b> in
some direction. Every neuron has its own output direction (one row of W2), which is added in proportion to how
strongly the neuron fired. Quiet neurons add almost nothing.</p>""",
            "{fig0}",
            """<p>A simplified picture: if a neuron that detects <i>Eiffel Tower</i> fires, it might add a direction that
means <i>Paris</i>, so the vector now carries “Tower” + Paris. Researchers have found that facts like this are
<b>largely recalled in the MLP layers</b>.</p>""",
            """<div class="box key"><b class="t">Key idea</b>Detect, then push: a neuron fires on a pattern, and the down
projection <b>adds its direction</b> (for example, a fact) to the token's vector.</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "After GELU, three neurons have values h = [2, 0, 0.5]. Their output directions "
                                    "(rows of W2) are [1, 0], [0, 1] and [−2, 2]. What does the MLP add to the vector?",
             "answer": "[1, 1].",
             "why": "2·[1, 0] + 0·[0, 1] + 0.5·[−2, 2] = [2 − 1, 0 + 1] = [1, 1]. The quiet neuron adds nothing."},
            {"kind": "mc", "q": "In the simplified picture, what happens when the <i>Eiffel Tower</i> neuron fires?",
             "options": ["The token is replaced by the word Paris",
                         "A direction meaning Paris is added to the token's vector",
                         "Attention searches the other tokens for Paris",
                         "The model immediately outputs the word Paris"],
             "answer": "B.", "why": "The down projection adds a push: the vector keeps “Tower” and gains “Paris”."},
            {"kind": "tf", "q": "“Researchers have found that facts like <i>the Eiffel Tower is in Paris</i> are "
                                "largely recalled in the attention layers.”",
             "answer": "False.", "why": "In the MLP layers. Attention's job is moving information between tokens."},
        ],
    },
    {
        "title": "Where the weights are, in code",
        "segment": (104, 134),
        "figures": [{"t": 115.6, "caption": "Weights per layer in GPT-2 small: the MLP has twice as many as "
                                            "attention."}],
        "body": [
            """<p>The MLP also holds <b>most of each layer's weights</b>. In GPT-2 small its two matrices have
2 × 768 × 3,072 = <b>4,718,592</b> weights (≈ 4.7 M), while attention's four 768 × 768 matrices (query, key, value
and output) have 4 × 768 × 768 = <b>2,359,296</b> (≈ 2.4 M). That is <b>twice</b> as many: the MLP is about
<b>two-thirds</b> of each layer.</p>""",
            "{fig0}",
            """<p>In code, it is two matrix multiplications with GELU in between: expand and bend, then project back
(<code>gelu</code> is written out in exercise 7.4).</p>""",
            f'<pre class="code">{CODE}</pre>',
            """<p>So each layer has two halves: <b>attention, where tokens talk</b>, and <b>the MLP, where each token
thinks</b>. Episode 9 wires them together into a transformer block.</p>""",
            """<div class="box key"><b class="t">Key idea</b>MLP ≈ 4.7 M weights per layer, attention ≈ 2.4 M:
<b>2×</b>. In code: <code>gelu(x @ W1 + b1) @ W2 + b2</code>.</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "What fraction of a GPT-2 small layer's weights (MLP + attention) is in the MLP?",
             "answer": "2/3 (about 67 %).", "why": "4,718,592 / (4,718,592 + 2,359,296) = 2/3."},
            {"kind": "number", "q": "A model has vectors of 1,024 numbers, a 4× hidden layer, and attention with "
                                    "four 1,024 × 1,024 matrices. How many weights do the MLP and attention each "
                                    "have per layer? Is the MLP still twice as big?",
             "lines": 2,
             "answer": "MLP 8,388,608; attention 4,194,304. Yes, 2×.",
             "why": "2 × 1,024 × 4,096 and 4 × 1,024 × 1,024. With the 4× rule the MLP always has 8 × d × d weights "
                    "and attention 4 × d × d."},
            {"kind": "mc", "q": "In <code>h = gelu(x @ W1 + b1)</code>, x is one token's vector (768 numbers). What "
                                "does h hold?",
             "options": ["768 numbers: the output of the MLP", "3,072 numbers: one value per neuron, after the bend",
                         "50,257 numbers: one per word in the vocabulary", "12 numbers: one per layer"],
             "answer": "B.", "why": "x @ W1 expands to 3,072 neuron values, and gelu bends each one. The output is "
                                   "only 768 after <code>@ W2</code>."},
            {"kind": "code", "q": "<b>Try it yourself.</b> The first two functions are the code from the video; the "
                                  "rest builds a random MLP of GPT-2-small size and runs it on 5 tokens at once. "
                                  "(a) What does the first <code>print</code> show? (b) What does the second show, and "
                                  "what is the shape of <code>h</code> inside <code>mlp</code>? (c) The third prints "
                                  "<code>True</code>. What does that tell you about how the MLP treats tokens?",
             "code": """import numpy as np

def gelu(x):
    c = np.sqrt(2 / np.pi)
    return 0.5 * x * (1 + np.tanh(c * (x + 0.044715 * x**3)))

def mlp(x, W1, b1, W2, b2):
    h = gelu(x @ W1 + b1)      # expand 768 -> 3072, then bend
    return h @ W2 + b2         # project back 3072 -> 768

rng = np.random.default_rng(0)
W1, b1 = rng.normal(0, 0.02, (768, 3072)), np.zeros(3072)
W2, b2 = rng.normal(0, 0.02, (3072, 768)), np.zeros(768)
x = rng.normal(0, 1, (5, 768))            # 5 tokens, one row each

out = mlp(x, W1, b1, W2, b2)
print(np.round(gelu(np.array([-2.0, 0.0, 2.5])), 3))
print(out.shape, W1.size + b1.size + W2.size + b2.size)
print(np.allclose(out[2], mlp(x[2], W1, b1, W2, b2)))""",
             "answer": "(a) [-0.045  0.     2.485] · (b) (5, 768) 4722432; h is (5, 3072) · (c) each token is "
                       "processed on its own",
             "why": """(a) −2 is squashed to nearly 0, 0 stays 0, and 2.5 passes almost unchanged. (b) Each of the 5
tokens gets 3,072 neuron values in h and is projected back to 768. The count is the 4,718,592 weights plus
3,072 + 768 = 3,840 biases. (c) Running the MLP on token 2 alone gives exactly its row of the batch result: no token
affects another, and all use the same weights (concept 1)."""},
        ],
    },
]
