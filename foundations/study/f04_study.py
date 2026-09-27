"""Study guide content for How LLMs Work · Foundations, F04: Straight Lines and Bends.

Build:  python framework/study_guide.py foundations f04
Figures: {"t": seconds into the rendered video, "caption": ...}; placed in body blocks with {fig0}, {fig1}, ...
Exercise kinds: mc (options), tf, short, number, order, code. Every exercise has an answer and a why.
All numbers verified with NumPy (the scene's own asserts plus the hand calculations below).
"""

LESSON = {
    "series": "How LLMs Work · Foundations",
    "label": "F04",
    "title": "Straight Lines and Bends",
    "tagline": "Linear functions, and why networks need a bend",
    "duration": "1:49",
    "intro": """<p>This lesson has one big idea: <b>linear layers alone are not enough</b>. A matrix can stretch and
rotate, but a stack of matrices is always just one more matrix. Putting a simple <b>bend</b> between the layers, a
nonlinear function like ReLU or GELU, is what lets a neural network build curved, complicated functions.</p>""",
    "prereq": """<div class="box note"><b>Before you start:</b> this lesson uses vectors (F01) and multiplying a matrix
by a vector or by another matrix, including NumPy's <code>@</code> (F03). You will meet the ideas again in episode 8
(The MLP: Where Facts Live), whose layer is exactly “expand, bend with GELU, project back”, and in episode 9 (The
Transformer Block), where those layers are stacked.</div>""",
}

CONCEPTS = [
    {
        "title": "Linear: double in, double out",
        "segment": (8, 27),
        "figures": [{"t": 17.8, "caption": "f(x) = 2x: a straight line through the origin."},
                    {"t": 27.2, "caption": "A matrix is linear too: W(a + b) = Wa + Wb."}],
        "body": [
            """<p>A function takes an input and gives an output. The simplest kind is <b>linear</b>, like
<i>f</i>(x) = 2x: f(1) = 2 and f(2) = 4, so doubling the input doubles the output. Its graph is a <b>straight line
through the origin</b>. Multiplying a vector by a <b>matrix</b> W is linear too. <b>Scale</b> the input and the output
scales the same way, W(2a) = 2·Wa; <b>add</b> two inputs and their outputs add, W(a + b) = Wa + Wb.</p>""",
            '<div class="figrow">{fig0}{fig1}</div>',
            """<div class="box key"><b class="t">Key idea</b><b>Linear</b> means scaling and adding pass straight
through: f(2x) = 2·f(x) and f(a + b) = f(a) + f(b). Every matrix is linear.</div>""",
        ],
        "exercises": [
            {"kind": "mc", "q": "Which of these functions is linear?",
             "options": ["f(x) = 2x + 1", "f(x) = −3x", "f(x) = x²", "f(x) = 10"],
             "answer": "B.", "why": "−3x triples and flips every input, so doubling in doubles out. For 2x + 1, f(1) = 3 "
                                   "but f(2) = 5 (its line misses the origin); x² quadruples when x doubles; a constant "
                                   "never changes."},
            {"kind": "number", "q": "The video's matrix W has rows (1, −0.5) and (0.5, 1), and the vectors are "
                                    "a = (1, 0.5) and b = (0.5, 1.5). Compute Wa, Wb and W(a + b). Does W(a + b) = "
                                    "Wa + Wb?",
             "lines": 2,
             "answer": "Wa = (0.75, 1), Wb = (−0.25, 1.75), W(a + b) = W(1.5, 2) = (0.5, 2.75). Yes.",
             "why": "Each output number is a row of W dotted with the vector, e.g. 1·1 + (−0.5)·0.5 = 0.75. And "
                    "(0.75, 1) + (−0.25, 1.75) = (0.5, 2.75), exactly W(a + b)."},
            {"kind": "number", "q": "You know Wa = (0.75, 1). Without multiplying by W again, write down W(2a) and "
                                    "W(−a).",
             "answer": "W(2a) = (1.5, 2) and W(−a) = (−0.75, −1).",
             "why": "Scaling passes straight through a linear map: W(2a) = 2·Wa, and scaling by −1 flips the output too."},
            {"kind": "tf", "q": "“The graph of a linear function such as f(x) = 5x always passes through (0, 0).”",
             "answer": "True.", "why": "f(0) = 5·0 = 0. The same holds for any matrix: W times the zero vector is zero."},
        ],
    },
    {
        "title": "Linear layers collapse into one",
        "segment": (28, 38),
        "figures": [{"t": 34.5, "caption": "W₁ then W₂ is one matrix: their product."},
                    {"t": 36.5, "caption": "100 linear layers: still one matrix."}],
        "body": [
            """<p>Here's the catch. Apply W₁ to x, then W₂, and you get exactly what one <b>single</b> matrix gives,
their product: <b>W₂(W₁x) = (W₂W₁)x</b>. Stack a hundred linear layers and they still <b>collapse</b> into one
matrix, W₁₀₀⋯W₂W₁.</p>""",
            '<div class="figrow">{fig0}{fig1}</div>',
            """<div class="box key"><b class="t">Key idea</b>Linear after linear is still linear. With nothing in
between, a stack of matrices is <b>no more powerful than one</b>.</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "W₁ has rows (1, 2) and (0, 1); W₂ has rows (2, 0) and (1, 1); x = (1, 1). "
                                    "(a) Compute W₁x, then W₂(W₁x). (b) Compute the product W₂W₁, then (W₂W₁)x.",
             "lines": 2,
             "answer": "(a) W₁x = (3, 1), W₂(W₁x) = (6, 4). (b) W₂W₁ has rows (2, 4) and (1, 3); (W₂W₁)x = (6, 4).",
             "why": "Same answer both ways: the two layers act exactly like the single matrix W₂W₁."},
            {"kind": "number", "q": "Three layers act on single numbers: multiply by 3, then by −2, then by 0.5. "
                                    "What single layer does the same?",
             "answer": "Multiply by −3.",
             "why": "3 × (−2) × 0.5 = −3. However many multiplications you chain, they make one multiplication."},
            {"kind": "mc", "q": "A network has 100 linear layers and no bends. How many matrices do you need to "
                                "compute exactly the same outputs?",
             "options": ["100, one per layer", "50, since layers merge in pairs", "1", "It depends on the input"],
             "answer": "C.", "why": "The whole stack multiplies out to one matrix, W₁₀₀⋯W₂W₁, for every input."},
            {"kind": "tf", "q": "“Adding more linear layers, with nothing between them, lets a network represent more "
                                "complicated functions.”",
             "answer": "False.", "why": "They collapse into one matrix, so the network can do exactly what one layer "
                                       "can do, no more."},
        ],
    },
    {
        "title": "What a straight map can't do",
        "segment": (38, 50),
        "figures": [{"t": 45.0, "caption": "One linear map can stretch and rotate the grid, but straight lines stay "
                                           "straight: it can't bend."},
                    {"t": 50.3, "caption": "Points inside the circle vs outside it: every straight line leaves orange "
                                           "points on both sides."}],
        "body": [
            """<p>A single linear map can only do so much. It can <b>stretch</b> and <b>rotate</b> space, but it
<b>can't bend</b> it: straight grid lines stay straight.</p>""",
            """<p>That limits what it can decide. The blue points lie inside a circle and the orange points around it.
Try any straight line: orange points end up on <b>both</b> sides, so no straight line separates inside from outside.
The boundary we need is curved, and a stack of linear layers can't make one either (concept 2).</p>""",
            '<div class="figrow">{fig0}{fig1}</div>',
            """<div class="box key"><b class="t">Key idea</b>Linear maps only make <b>straight</b> things: straight
lines, flat boundaries. Problems like “inside vs outside a circle” need a <b>curve</b>.</div>""",
        ],
        "exercises": [
            {"kind": "mc", "q": "Which of these can a single matrix <b>not</b> do to the grid?",
             "options": ["Stretch it", "Rotate it", "Shear it (F03)", "Bend straight lines into curves"],
             "answer": "D.", "why": "Stretching, rotating and shearing all keep straight lines straight. Bending is "
                                   "exactly what a linear map can't do."},
            {"kind": "short", "q": "A one-dimensional version of the circle: the “inside” points are −0.5 and 0.5, "
                                   "the “outside” points are −2 and 2. Can one cut “x > c” put both outside points on "
                                   "one side and both inside points on the other? Explain.",
             "lines": 2,
             "answer": "No.", "why": "The outside points sit at both ends. Any cut that puts 2 on one side leaves −2 on "
                                     "the other side, together with the inside points. (Concept 6 solves this with bends.)"},
            {"kind": "tf", "q": "“One linear layer can't separate the circle, but 50 linear layers stacked "
                                "together can.”",
             "answer": "False.", "why": "The 50 layers collapse into one matrix (concept 2), with exactly the same limits."},
        ],
    },
    {
        "title": "The fix: a bend called ReLU",
        "segment": (51, 68),
        "figures": [{"t": 57.0, "caption": "The bend sits between the layers and acts on every number separately."},
                    {"t": 67.6, "caption": "ReLU(x) = max(0, x). With ReLU between them, stacked layers can't collapse."}],
        "body": [
            """<p>The fix is to add a <b>bend</b> between the layers: a <b>nonlinear</b> function, applied to
<b>every number separately</b>. The layer becomes x → W₁ → bend → W₂ → y.</p>""",
            """<p>The simplest bend is <b>ReLU</b>, ReLU(x) = max(0, x). It keeps positive numbers and replaces negative
ones with zero, so the vector [−2, −0.5, 0, 1, 3] becomes <b>[0, 0, 0, 1, 3]</b>. A tiny change, but ReLU is not
linear, so W₂·ReLU(W₁x) is no longer one matrix, and stacked layers can't collapse.</p>""",
            '<div class="figrow">{fig0}{fig1}</div>',
            """<div class="box key"><b class="t">Key idea</b>A <b>nonlinearity</b> (bend) between linear layers stops
them collapsing. ReLU = max(0, x), applied to each number on its own, is the simplest one.</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "Apply ReLU to the vector [4, −1, 0.5, −3].",
             "answer": "[4, 0, 0.5, 0].", "why": "Positive numbers pass unchanged; negative ones become 0."},
            {"kind": "number", "q": "Check that ReLU breaks the “adding” rule of concept 1: compute ReLU(3 + (−2)) and "
                                    "ReLU(3) + ReLU(−2).",
             "answer": "ReLU(1) = 1, but ReLU(3) + ReLU(−2) = 3 + 0 = 3.",
             "why": "They differ, so ReLU is not linear. That is precisely what stops the layers collapsing into one "
                    "matrix."},
            {"kind": "order", "q": "Put one bent layer in the order the data flows: <i>y · W₂ · x · ReLU · W₁</i>.",
             "answer": "x → W₁ → ReLU → W₂ → y.", "why": "Linear, then the bend, then linear again."},
            {"kind": "tf", "q": "“ReLU mixes the numbers of a vector: each output number depends on all the input "
                                "numbers.”",
             "answer": "False.", "why": "ReLU treats every number separately. The mixing is done by the matrices."},
        ],
    },
    {
        "title": "GELU: a smooth ReLU",
        "segment": (68, 79),
        "figures": [{"t": 79.3, "caption": "GELU (teal) against ReLU (dashed): the same far from 0, but smooth near 0, "
                                           "with a small dip below zero."}],
        "body": [
            """<p>Transformers usually use <b>GELU</b>, a smooth version of ReLU. Far from zero the two agree:
large positive inputs <b>pass through</b> (gelu(3) ≈ 2.996) and large negative ones <b>fade to zero</b>
(gelu(−3) ≈ −0.004). Near zero, instead of ReLU's sharp corner, there is a <b>gentle curve</b>, with a small dip below
zero down to about −0.17.</p>""",
            "{fig0}",
            """<div class="box key"><b class="t">Key idea</b>GELU ≈ ReLU for large inputs, but it is <b>smooth</b> (no
corner) around 0 and dips slightly below zero (minimum ≈ −0.17). It is still a bend, applied to every number
separately.</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "How far is GELU from ReLU at the video's two points? Compute relu(3) − gelu(3) "
                                    "and relu(−3) − gelu(−3).",
             "answer": "Both ≈ 0.004.", "why": "3 − 2.996 = 0.004 and 0 − (−0.004) = 0.004. Far from zero, GELU is "
                                               "within a few thousandths of ReLU."},
            {"kind": "mc", "q": "Given gelu(1) ≈ 0.841 and gelu(−1) ≈ −0.159, where do ReLU and GELU differ most?",
             "options": ["For large positive inputs, like 3", "For large negative inputs, like −3",
                         "Near zero, around −1 to 1", "Nowhere: they are the same function"],
             "answer": "C.", "why": "At ±1 they differ by about 0.16 (relu(1) = 1, relu(−1) = 0); at ±3 by only 0.004. "
                                   "The smooth curve replaces the corner at 0."},
            {"kind": "tf", "q": "“GELU never outputs a negative number.”",
             "answer": "False.", "why": "It has a small dip: its minimum is about −0.17 (near x ≈ −0.75), and "
                                       "gelu(−3) ≈ −0.004 is negative too, just tiny."},
            {"kind": "short", "q": "A GELU receives the vector [−3, 0, 3]. Roughly what comes out?",
             "answer": "About [−0.004, 0, 2.996], i.e. almost [0, 0, 3].",
             "why": "Each number is bent on its own, and this far from zero GELU acts almost exactly like ReLU."},
        ],
    },
    {
        "title": "Curves from many straight pieces",
        "segment": (80, 87),
        "figures": [{"t": 87.1, "caption": "A curve (blue) approximated by adding up ReLU pieces: the yellow bent line "
                                           "follows it closely."}],
        "body": [
            """<p>Each ReLU adds one <b>hinge</b>: flat on one side, sloped on the other. Add several hinges at different
places, with different slopes, and you get a line with <b>corners</b>. With enough corners, the straight pieces can
follow a curve very closely.</p>""",
            """<p>That is what bends between the layers buy. A network can build <b>curved shapes out of many simple
pieces</b>, and approximate almost any function. Without the bends, all it could ever make is one straight line.</p>""",
            "{fig0}",
            """<div class="box key"><b class="t">Key idea</b>ReLU hinges are simple pieces. Adding many of them gives bent
lines that can follow <b>almost any curve</b>.</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "Let g(x) = relu(x − 1) + relu(−x − 1). Compute g at x = −2, −0.5, 0.5 and 2. "
                                    "Does it separate the inside and outside points of exercise 3.2?",
             "lines": 2,
             "answer": "g = 1, 0, 0, 1. Yes.",
             "why": "Two hinges build a valley: 0 between −1 and 1, rising outside. The outside points get 1 and the "
                    "inside points 0, so “outside if g(x) > 0.5” works, which no single cut could do."},
            {"kind": "number", "q": "Let h(x) = relu(x) − 2·relu(x − 1). Compute h(0), h(1), h(2) and h(3). "
                                    "Describe the shape.",
             "answer": "0, 1, 0, −1: up to a peak at x = 1, then down.",
             "why": "The first hinge starts a rising line at 0; the second, from x = 1, subtracts twice the slope, so "
                    "the line turns downward. Two pieces, one corner."},
            {"kind": "mc", "q": "You add up several linear functions with no bends, such as 2x and −3x. What shape "
                                "is the result?",
             "options": ["A curve", "A straight line", "A hinge with one corner", "A circle"],
             "answer": "B.", "why": "2x − 3x = −x. Sums of linear functions stay linear; corners and curves need bends."},
        ],
    },
    {
        "title": "A layer in code",
        "segment": (88, 107),
        "figures": [{"t": 94.6, "size": "small", "caption": "The code from the video: ReLU in one line, then a layer: "
                                                            "matrix, bend, matrix."}],
        "body": [
            """<p>In NumPy, ReLU is one line: <code>np.maximum(0, x)</code> compares every number with 0 and keeps the
larger. A layer is a <b>matrix multiplication, then the bend, then the next matrix</b>. (In code the vector is
written first, <code>x @ W1</code>; it is still one matrix multiplication.)</p>""",
            "{fig0}",
            """<pre class="code">def relu(x):
    return np.maximum(0, x)

relu(np.array([-2, -0.5, 0, 1, 3]))   # [0, 0, 0, 1, 3]

h = relu(x @ W1)                      # linear, then bend
y = h @ W2                            # linear again</pre>""",
            """<p>This is the MLP of episode 8: <b>expand</b> (W1 makes the vector longer), <b>bend with GELU</b>, and
<b>project back</b> (W2 returns it to its original size). Episode 9 stacks these blocks.</p>""",
            """<div class="box key"><b class="t">Key idea</b>One layer = <code>relu(x @ W1) @ W2</code>: linear, bend,
linear. Remove the bend and the two matrices collapse into one, <code>W1 @ W2</code>.</div>""",
        ],
        "exercises": [
            {"kind": "short", "q": "Remove the bend: <code>y = (x @ W1) @ W2</code>. Which single matrix W gives the "
                                   "same y for every x?",
             "answer": "W = W1 @ W2.", "why": "Without the bend the layer is linear after linear, so it collapses into "
                                              "their product (concept 2)."},
            {"kind": "order", "q": "Put the steps of episode 8's MLP in order: <i>project back · bend with GELU · "
                                   "expand</i>.",
             "answer": "expand → bend with GELU → project back.",
             "why": "Matrix W1 (expand), the bend, then matrix W2 (back to the original size)."},
            {"kind": "code", "q": "<b>Try it yourself.</b> Here is a tiny layer: W1 expands 2 numbers to 3, W2 projects "
                                  "back to 2. (a) <b>By hand first:</b> what are <code>x @ W1</code>, <code>h</code> and "
                                  "<code>y</code>? Then run the code to check. (b) What does the last line print, and why "
                                  "are its two arrays equal? (c) Why is <code>y</code> different from "
                                  "<code>y_lin</code>? Which number in <code>x @ W1</code> made the difference?",
             "code": """import numpy as np

def relu(x):
    return np.maximum(0, x)

x  = np.array([1., 2.])
W1 = np.array([[1., -2.,  0.],
               [1.,  1., -1.]])    # 2 numbers -> 3: expand
W2 = np.array([[1., 0.],
               [2., 1.],
               [0., 3.]])          # 3 numbers -> 2: project back

h = relu(x @ W1)                   # linear, then bend
y = h @ W2                         # linear again
print(x @ W1, h, y)

y_lin = (x @ W1) @ W2              # the same layer without the bend
print(y_lin, x @ (W1 @ W2))""",
             "answer": "(a) x @ W1 = [3, 0, −2], h = [3, 0, 0], y = [3, 0] · (b) [3, −6] twice · (c) ReLU zeroed "
                       "the −2.",
             "why": """(a) Each entry of <code>x @ W1</code> is x dotted with a column of W1: 1·1 + 2·1 = 3,
1·(−2) + 2·1 = 0, 0 + 2·(−1) = −2. ReLU turns −2 into 0, and h @ W2 = 3·(1, 0) = (3, 0).
(b) Without the bend the two matrices collapse into one: W1 @ W2 has rows (−3, −2) and (3, −2), and x times it
gives (3, −6), the same as doing the two steps.
(c) In <code>y_lin</code>, the −2 in the third position adds −2 × (0, 3) = (0, −6). The bend removed it, so the layer
is no longer a single matrix."""},
        ],
    },
]
