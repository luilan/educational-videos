"""Study guide content for How LLMs Work · Foundations, F12: Neural Networks in Three Minutes.

Build:  python framework/study_guide.py foundations f12 --video <NeuralNetVideo.mp4>
Figures: {"t": seconds into the rendered video, "caption": ...}; placed in body blocks with {fig0}, {fig1}, ...
Exercise kinds: mc (options), tf, short, number, order, code. Every exercise has an answer and a why.
All numbers and code outputs were checked by running Python (NumPy, and torch for the parameter counts).
"""

LESSON = {
    "series": "How LLMs Work · Foundations",
    "label": "F12",
    "title": "Neural Networks in Three Minutes",
    "tagline": "Neurons, layers, weights, and learning",
    "duration": "2:00",
    "intro": """<p>This lesson has one big idea: a neural network is <b>just a function</b>, built from one very simple part
repeated many times. A <b>neuron</b> weighs its inputs, adds a bias and bends the result; neurons side by side make a
<b>layer</b>, and stacked layers make a <b>network</b>. Its numbers, the <b>parameters</b>, are not written by anyone:
they are learned from data.</p>""",
    "prereq": """<div class="box note"><b>Before you start:</b> this lesson builds on F02 (The Dot Product), F03 (Matrices)
and F04 (Straight Lines and Bends, where ReLU was introduced); the code uses NumPy's <code>@</code> from F11. Neural
networks return in episode 8 (The MLP: Where Facts Live), episode 11 (Training: Learning from Mistakes) and episode 12
(Build a Tiny GPT).</div>""",
}

CONCEPTS = [
    {
        "title": "A neuron: weigh, add, bend",
        "segment": (8, 42),
        "figures": [{"t": 27.6, "caption": "Weigh each input, add them up with the bias (Σ), bend with ReLU."},
                    {"t": 41.0, "caption": "The video's example: 1 + 3 − 1 = 3, and ReLU(3) = 3."}],
        "body": [
            """<p>Strip away the hype and a neural network is a <b>function</b> built from simple parts. The basic part is a
<b>neuron</b>: it multiplies each input by a <b>weight</b>, adds the results up with a <b>bias</b>, and passes the sum
through a <b>bend</b>, here <b>ReLU</b> (F04): positive numbers pass, negative numbers become 0.</p>""",
            '<div class="figrow">{fig0}{fig1}</div>',
            """<p>Inputs 2 and 3, weights 0.5 and 1, bias −1: 2 × 0.5 + 3 × 1 − 1 = 3. ReLU(3) = 3, so the neuron
<b>fires</b> with value 3. Had the sum been negative, it would output 0 and stay silent.</p>""",
            """<div class="box key"><b class="t">Key idea</b>A neuron computes <b>ReLU(x · w + b)</b>: the dot product of
the inputs with the weights, plus a bias, then the bend.</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "Same neuron (weights 0.5 and 1, bias −1), inputs 4 and 1. What does it output?",
             "answer": "2.", "why": "4 × 0.5 + 1 × 1 = 3, then 3 − 1 = 2. ReLU keeps positive numbers, so 2."},
            {"kind": "number", "q": "Same neuron, inputs 2 and −3. Compute the sum before the bend, and the output. "
                                    "Does the neuron fire?",
             "answer": "Sum −3, output 0: it does not fire.",
             "why": "2 × 0.5 + (−3) × 1 − 1 = 1 − 3 − 1 = −3, and ReLU turns every negative number into 0."},
            {"kind": "order", "q": "Put a neuron's steps in order: <i>apply ReLU · add the bias · multiply each input "
                                   "by its weight · add up the products</i>.",
             "answer": "multiply by weights → add up → add the bias → apply ReLU.",
             "why": "Weigh, add, bend: the bias shifts the sum before the bend decides what comes out."},
            {"kind": "mc", "q": "What does the bias do?",
             "options": ["It multiplies every input by the same number",
                         "It adds a fixed number to the weighted sum, shifting it up or down before the bend",
                         "It chooses which inputs the neuron ignores",
                         "It is the bend that turns negative numbers into 0"],
             "answer": "B.", "why": "In the example the bias −1 turned 4 into 3. Weights scale the inputs (A); ReLU "
                                   "is the bend (D)."},
        ],
    },
    {
        "title": "A layer is one matrix multiplication",
        "segment": (42, 54),
        "figures": [{"t": 53.2, "caption": "Three neurons read the same inputs 2 and 3. Their weights form the matrix "
                                           "W (row = input, column = neuron): out comes [3, 0, 0]."}],
        "body": [
            """<p>Put several neurons side by side, all reading the <b>same inputs</b>, and you have a <b>layer</b>. Each
neuron keeps its own weights. Write them as the <b>columns</b> of a matrix <b>W</b> (one row per input, one column per
neuron) and the biases as a vector <b>b</b>, and the whole layer is a single line: <b>ReLU(x · W + b)</b>.</p>""",
            "{fig0}",
            """<p>In the video, column 1 of W is the neuron from concept 1, giving 3. Column 2 gives
2 × (−1) + 3 × 0.3 + 0.5 = −0.6, column 3 gives 2 × 0.2 + 3 × (−0.4) + 0 = −0.8. After ReLU: <b>[3, 0, 0]</b>.</p>""",
            """<div class="box key"><b class="t">Key idea</b>A layer of neurons is <b>one matrix multiplication plus the
bend</b>: ReLU(x · W + b). It turns the input vector into one number per neuron, all at once.</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "Same layer (W = [[0.5, −1, 0.2], [1, 0.3, −0.4]], b = [−1, 0.5, 0]), new input "
                                    "x = [0, 2]. Compute x · W + b, then the layer's output.",
             "lines": 2,
             "answer": "x · W + b = [1, 1.1, −0.8]; output [1, 1.1, 0].",
             "why": "Column by column: 0 + 2 − 1 = 1; 0 + 0.6 + 0.5 = 1.1; 0 − 0.8 + 0 = −0.8. ReLU zeroes only the "
                    "negative one, so two neurons fire this time."},
            {"kind": "mc", "q": "A layer reads 4 inputs and has 10 neurons. What shape is W, and how many biases "
                                "are there?",
             "options": ["4 × 10, and 10 biases", "10 × 4, and 4 biases", "4 × 10, and 4 biases",
                         "10 × 10, and 10 biases"],
             "answer": "A.", "why": "One row per input (4), one column per neuron (10), and one bias per neuron (10)."},
            {"kind": "tf", "q": "“In x · W, each <i>row</i> of W holds the weights of one neuron.”",
             "answer": "False.", "why": "Each <i>column</i> is one neuron; each row belongs to one input. Column 1, "
                                        "[0.5, 1], is the neuron from concept 1."},
        ],
    },
    {
        "title": "Stacking layers",
        "segment": (54, 63),
        "figures": [{"t": 62.9, "caption": "Each layer's outputs feed the next layer."}],
        "body": [
            """<p>Stack layers, and the <b>outputs of one layer become the inputs of the next</b>. Why? <b>Early layers
find simple patterns; later layers combine them into more complex ones</b>, like strokes into shapes. The bend between
layers makes this work: without it, two matrix multiplications in a row collapse into one (F04).</p>""",
            "{fig0}",
            """<div class="box key"><b class="t">Key idea</b>A network is a chain of layers: <b>the output of one layer is
the input of the next</b>. Depth lets it build complex patterns out of simple ones.</div>""",
        ],
        "exercises": [
            {"kind": "short", "q": "The layer of concept 2 outputs [3, 0, 0]. How many inputs does the next layer "
                                   "read, and how many rows must its weight matrix have?",
             "answer": "3 inputs, so 3 rows.",
             "why": "One input per neuron of the previous layer, and W always has one row per input."},
            {"kind": "number", "q": "The next layer has a single neuron with weights [0.5, 4, −2] and bias 1. It reads "
                                    "[3, 0, 0]. What does it output?",
             "answer": "2.5.", "why": "3 × 0.5 + 0 × 4 + 0 × (−2) + 1 = 2.5, and ReLU keeps it. The silent neurons send "
                                     "0, so their weights (4 and −2) make no difference for this input."},
            {"kind": "mc", "q": "According to the video, how do the layers of a deep network typically divide the "
                                "work?",
             "options": ["Early layers find complex patterns, later layers simplify them",
                         "Early layers find simple patterns, later layers combine them into complex ones",
                         "Every layer finds the same patterns, for safety",
                         "Only the last layer finds patterns; the others pass the inputs through"],
             "answer": "B.", "why": "Simple pieces first (like strokes), then combinations of pieces (like shapes)."},
            {"kind": "tf", "q": "“Two layers with <i>no bend</i> between them can do more than one layer.”",
             "answer": "False.", "why": "Without the bend, the two matrix multiplications collapse into one matrix "
                                        "(F04). The bend is what makes depth worthwhile."},
        ],
    },
    {
        "title": "Parameters: the numbers that learn",
        "segment": (63, 75),
        "figures": [{"t": 74.9, "caption": "A weight on every edge, a bias (b) in every neuron."}],
        "body": [
            """<p>The weights and biases together are the network's <b>parameters</b>. They are the <b>only thing that
changes</b> when a network learns. The tiny GPT of episode 12 has <b>818,241</b>; big models have billions.</p>""",
            "{fig0}",
            """<p>A layer with <i>n</i> inputs and <i>m</i> neurons has <b>n × m weights</b> (one per connection) and
<b>m biases</b> (one per neuron).</p>""",
            """<div class="box key"><b class="t">Key idea</b><b>Parameters = weights + biases.</b> One layer has
inputs × neurons + neurons of them. Learning means changing these numbers, and nothing else.</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "How many parameters does the layer of concept 2 (2 inputs, 3 neurons) have?",
             "answer": "9.", "why": "2 × 3 = 6 weights (the entries of W) plus 3 biases."},
            {"kind": "number", "q": "The network in the figure has 3 inputs and three layers of 4 neurons each. "
                                    "How many parameters does it have in total?",
             "answer": "56.", "why": "Layer 1: 3 × 4 + 4 = 16. Layers 2 and 3: 4 × 4 + 4 = 20 each. "
                                    "16 + 20 + 20 = 56."},
            {"kind": "mc", "q": "Which of these is <b>not</b> a parameter of the network?",
             "options": ["The weight on the edge from input 1 to neuron 3", "The bias of neuron 2",
                         "The input numbers, such as 2 and 3", "An entry of the matrix W"],
             "answer": "C.", "why": "Inputs are the data the network reads; they change with every example. Parameters "
                                   "belong to the network itself."},
            {"kind": "tf", "q": "“When a network learns, it grows new neurons and connections.”",
             "answer": "False.", "why": "The shape stays fixed; only the values of the weights and biases change."},
        ],
    },
    {
        "title": "Inference and training",
        "segment": (75, 90),
        "figures": [{"t": 83.0, "caption": "Inference: forward to a prediction."},
                    {"t": 89.6, "caption": "Training: compare, then nudge."}],
        "body": [
            """<p>A network is used in two ways. <b>Inference</b> means running it: inputs flow forward and out comes a
<b>prediction</b>; nothing inside changes. <b>Training</b> means adjusting it: <b>compare</b> the prediction with the
right answer and <b>nudge every parameter</b> a little so the prediction gets better.</p>""",
            '<div class="figrow">{fig0}{fig1}</div>',
            """<p>In the video's illustration, the prediction 0.2 misses the right answer 1.0 by 0.8; one nudge later it
is 0.3, an error of 0.7. Training repeats this many times (F10, F13 and episode 11 show how).</p>""",
            """<div class="box key"><b class="t">Key idea</b><b>Inference</b>: forward only, parameters fixed.
<b>Training</b>: forward, compare with the right answer, nudge every parameter, and repeat.</div>""",
        ],
        "exercises": [
            {"kind": "mc", "q": "You ask a finished chatbot a question and it answers. What is the network doing?",
             "options": ["Training", "Inference", "Both at once", "Neither"],
             "answer": "B.", "why": "It runs forward to make predictions; its parameters stay as they are."},
            {"kind": "order", "q": "Put one training step in order: <i>nudge every parameter · compare the prediction "
                                   "with the right answer · run the inputs forward</i>.",
             "answer": "run forward → compare → nudge.",
             "why": "You need a prediction before you can compare it, and the comparison tells you how to nudge."},
            {"kind": "number", "q": "Continue the video's example: the prediction is 0.3 and the right answer 1.0. "
                                    "The next nudge moves the prediction to 0.45. What is the new error, and by how "
                                    "much did it shrink?",
             "answer": "Error 0.55, which is 0.15 smaller.", "why": "1.0 − 0.45 = 0.55, down from 0.7."},
            {"kind": "tf", "q": "“During inference, the parameters are nudged after every prediction.”",
             "answer": "False.", "why": "Nudging parameters is training. Inference only runs the network forward."},
        ],
    },
    {
        "title": "We choose the shape, data does the rest",
        "segment": (90, 98),
        "figures": [{"t": 97.6, "caption": "No hand-written rules. We set the shape (3 layers, 4 neurons each); "
                                           "training data turns the untrained network into a trained one."}],
        "body": [
            """<p>Nobody writes the rules inside a neural network: there is no <i>if word == “the”</i> anywhere. People
choose only the <b>shape</b>: how many layers, and how many neurons in each. The shape fixes how many parameters there
are. Their <b>values</b> are set by training on data, which turns an untrained network into a trained one.</p>""",
            "{fig0}",
            """<div class="box key"><b class="t">Key idea</b>People choose the <b>shape</b> (layers, neurons per layer).
The <b>training data</b> chooses the values of the parameters.</div>""",
        ],
        "exercises": [
            {"kind": "mc", "q": "Which of these is chosen by a person, not learned from data?",
             "options": ["The weight between input 1 and neuron 3", "The number of layers",
                         "The bias of neuron 2", "The output for a particular input"],
             "answer": "B.", "why": "The number of layers is part of the shape. Weights and biases are learned, and "
                                   "the outputs follow from them."},
            {"kind": "number", "q": "You choose the shape: 2 inputs → a layer of 3 neurons → a layer of 1 neuron. "
                                    "How many parameters will training have to set?",
             "answer": "13.", "why": "First layer 2 × 3 + 3 = 9, second layer 3 × 1 + 1 = 4, and 9 + 4 = 13."},
            {"kind": "tf", "q": "“To teach a network something new, an engineer writes a new rule into it.”",
             "answer": "False.", "why": "Nobody writes rules inside. You train it on new data and the parameters "
                                        "change."},
        ],
    },
    {
        "title": "Neurons and layers in code",
        "segment": (98, 105),
        "figures": [{"t": 104.6, "size": "small", "caption": "The code from the video: a neuron in one line, a whole "
                                                            "layer in one line."}],
        "body": [
            """<p>In code, a neuron is a dot product, a bias and a ReLU. A layer does many neurons at once: the same
line with a matrix <code>W</code> instead of a vector <code>w</code>, and <code>np.maximum</code>, which applies ReLU to
every number of the result.</p>""",
            "{fig0}",
            """<pre class="code">def neuron(x, w, b):
    return max(0.0, x @ w + b)             # weighted sum, then ReLU

x = np.array([2.0, 3.0])
neuron(x, np.array([0.5, 1.0]), -1.0)     # 3.0

def layer(x, W, b):
    return np.maximum(0, x @ W + b)        # many neurons at once</pre>""",
            """<div class="box key"><b class="t">Key idea</b>Neuron: <code>max(0.0, x @ w + b)</code>. Layer:
<code>np.maximum(0, x @ W + b)</code>. A network is layers called one after another.</div>""",
        ],
        "exercises": [
            {"kind": "mc", "q": "Why does <code>layer</code> use <code>np.maximum(0, …)</code> rather than Python's "
                                "<code>max(0.0, …)</code>?",
             "options": ["np.maximum is faster, but the result is the same",
                         "x @ W + b is an array: np.maximum applies ReLU to each element, but max cannot compare a "
                         "whole array with 0",
                         "np.maximum picks the neuron with the largest output",
                         "np.maximum also adds the bias"],
             "answer": "B.", "why": "<code>max(0.0, np.array([3, -0.6]))</code> raises a ValueError (“truth value of an "
                                   "array … is ambiguous”). <code>np.maximum</code> works element by element."},
            {"kind": "code", "q": "<b>Try it yourself.</b> The code below has the two functions from the video, the "
                                  "video's first layer (<code>W1</code>, <code>b1</code>) and a second layer with two "
                                  "neurons (<code>W2</code>, <code>b2</code>). (a) What does <code>layer(x, W1, b1)</code> "
                                  "return? Check that <code>neuron(x, W1[:, 0], b1[0])</code> matches its first number. "
                                  "(b) Write <code>network(x)</code>, which feeds the output of layer 1 into layer 2. "
                                  "What does it return for <code>x</code>? (c) What does it return for "
                                  "<code>np.array([0.0, 2.0])</code>? How many parameters does this network have?",
             "code": """import numpy as np

def neuron(x, w, b):
    return max(0.0, x @ w + b)             # weighted sum, then ReLU

def layer(x, W, b):
    return np.maximum(0, x @ W + b)        # many neurons at once

x  = np.array([2.0, 3.0])
W1 = np.array([[0.5, -1.0,  0.2],
               [1.0,  0.3, -0.4]])
b1 = np.array([-1.0, 0.5, 0.0])
W2 = np.array([[ 1.0, -2.0],
               [ 0.5,  1.0],
               [-1.0,  3.0]])
b2 = np.array([0.5, 1.0])""",
             "answer": "(a) [3. 0. 0.], and neuron(...) = 3.0 · (b) [3.5 0. ] · (c) [2.05 0.1 ], 17 parameters",
             "why": """(a) The numbers of concept 2; column 0 of W1 with bias b1[0] is the neuron of concept 1.
(b) <code>def network(x): return layer(layer(x, W1, b1), W2, b2)</code>. The hidden output [3, 0, 0] gives
[3 × 1 + 0.5, 3 × (−2) + 1] = [3.5, −5], and ReLU makes it [3.5, 0].
(c) The hidden output is now [1, 1.1, 0] (question 2.1), so layer 2 computes [1 + 0.55 + 0.5, −2 + 1.1 + 1] =
[2.05, 0.1]: both neurons fire. Parameters: 6 + 3 + 6 + 2 = 17 (<code>W1.size + b1.size + W2.size + b2.size</code>)."""},
        ],
    },
]
