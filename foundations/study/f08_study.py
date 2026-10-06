"""Study guide content for How LLMs Work · Foundations, F08: Averages and Spread.

Build:  python framework/study_guide.py foundations f08 --video <rendered mp4>
Figures: {"t": seconds into the rendered video, "caption": ...}; placed in body blocks with {fig0}, {fig1}, ...
Exercise kinds: mc (options), tf, short, number, order, code. Every exercise has an answer and a why.
All numbers were checked with NumPy (np.std divides by n, like the video's recipe).
"""

LESSON = {
    "series": "How LLMs Work · Foundations",
    "label": "F08",
    "title": "Averages and Spread",
    "tagline": "Mean, standard deviation, and keeping numbers tame",
    "duration": "1:56",
    "intro": """<p>This lesson has one big idea: two simple statistics describe a list of numbers, the <b>mean</b> (its
center) and the <b>standard deviation</b> (its spread), and together they let us keep numbers tame. Normalizing with them
is what layer norm does, and the same idea of spread explains why attention divides its scores by √d.</p>""",
    "prereq": """<div class="box note"><b>Before you start:</b> concepts 1–3 need only arithmetic and square roots.
Concepts 4 and 5 use the dot product from F02 (multiply matching numbers, then add) and softmax from F07 (bigger gaps
between scores make it sharper). These ideas are used in episode 6 (the √d in the attention formula) and episode 9
(layer norm in the transformer block).</div>""",
}

CONCEPTS = [
    {
        "title": "The mean: the center",
        "segment": (8, 27),
        "figures": [{"t": 15.0, "caption": "Through many layers, numbers can drift too big or too small."},
                    {"t": 27.7, "caption": "The mean of 2, 4, 6, 8 is 5, the center of the dots."}],
        "body": [
            """<p>Deep networks pass numbers through dozens of layers. If they drift too big (4812) or too small
(0.00001), training breaks. Two simple statistics help keep them in check: the <b>mean</b>, which says where the numbers
are centered, and the <b>standard deviation</b>, which says how spread out they are.</p>""",
            """<p>The <b>mean</b> is the average: add the numbers up and divide by how many there are. For 2, 4, 6, 8:
(2 + 4 + 6 + 8) ÷ 4 = 20 ÷ 4 = <b>5</b>.</p>""",
            '<div class="figrow">{fig0}{fig1}</div>',
            """<div class="box key"><b class="t">Key idea</b><b>mean = sum ÷ count</b>. It tells you where the numbers
are centered, but nothing about how spread out they are.</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "Find the mean of <b>1, 2, 3, 10</b>.",
             "answer": "4.", "why": "16 ÷ 4 = 4. One big number pulls the mean up: three of the four numbers are below "
                                   "it.", "key": {'parts': [{'label': None, 'value': 4, 'tol': 0.5, 'unit': None}]}},
            {"kind": "number", "q": "Add 10 to every number of the video's example: 12, 14, 16, 18. What is the new "
                                    "mean?",
             "answer": "15.", "why": "Every number moves up by 10, so the center moves up by 10 too: 60 ÷ 4 = 15.", "key": {'parts': [{'label': None, 'value': 15, 'tol': 0.5, 'unit': None}]}},
            {"kind": "tf", "q": "“Two lists with the same mean must be spread out in the same way.”",
             "answer": "False.", "why": "4, 6 and 0, 10 both have mean 5, but the second is far more spread out. That is "
                                        "why we need a second number, the standard deviation.", "key": {'value': False}},
            {"kind": "mc", "q": "Why do deep networks need to keep their numbers in check?",
             "options": ["Numbers that drift too big or too small through many layers make training break",
                         "Every layer's mean must be exactly 5",
                         "Negative numbers are not allowed inside a network",
                         "Computers can only store numbers between 0 and 1"],
             "answer": "A.", "why": "That is the problem the video opens with. Mean and standard deviation measure the "
                                   "drift, so we can undo it (concept 3).", "key": {'choice': 0}},
        ],
    },
    {
        "title": "Standard deviation: the typical distance",
        "segment": (28, 51),
        "figures": [{"t": 50.9, "size": "small", "caption": "Squares 9, 1, 1, 9; average 5; std = √5 ≈ 2.24."}],
        "body": [
            """<p>The <b>standard deviation</b> (std) measures spread: how far the numbers typically are from the mean.
The recipe: take each number's <b>distance from the mean</b>, <b>square</b> it, <b>average</b> the squares, and take
the <b>square root</b>.</p>""",
            """<p>For 2, 4, 6, 8 (mean 5): the distances are −3, −1, 1, 3; the squares are 9, 1, 1, 9; their average is
20 ÷ 4 = 5; so the std is √5 ≈ <b>2.24</b>. Squaring makes every distance count as positive (−3 and 3 both give 9);
the square root undoes the squaring.</p>""",
            "{fig0}",
            """<div class="box key"><b class="t">Key idea</b><b>std = √(average of the squared distances from the
mean)</b>. A small std means the numbers huddle close to the mean; a large std means they are spread out.</div>""",
        ],
        "exercises": [
            {"kind": "order", "q": "Put the steps of the std recipe in order: <i>square · take the square root · "
                                   "distance from the mean · average</i>.",
             "answer": "distance from the mean → square → average → take the square root.",
             "why": "Distances first, then square them, average the squares, and finish with the square root.", "key": {'items': ['distance from the mean', 'square', 'average', 'take the square root']}},
            {"kind": "number", "q": "Find the std of <b>1, 2, 3, 4, 5</b>.",
             "answer": "√2 ≈ 1.41.", "why": "Mean 3; distances −2, −1, 0, 1, 2; squares 4, 1, 0, 1, 4; average "
                                          "10 ÷ 5 = 2; √2 ≈ 1.41.", "key": {'parts': [{'label': None, 'value': 1.41, 'tol': 0.0282, 'unit': None}]}},
            {"kind": "mc", "q": "What is the std of <b>5, 5, 5, 5</b>?",
             "options": ["5", "1", "0", "2.24"],
             "answer": "C.", "why": "Every distance from the mean (5) is 0, so there is no spread at all.", "key": {'choice': 2}},
            {"kind": "tf", "q": "“Doubling every number (4, 8, 12, 16) doubles the std.”",
             "answer": "True.", "why": "The distances double (−6, −2, 2, 6), the squares average 20, and "
                                       "√20 ≈ 4.47 = 2 × 2.24.", "key": {'value': True}},
        ],
    },
    {
        "title": "Normalizing: mean 0, spread 1",
        "segment": (52, 74),
        "figures": [{"t": 68.2, "caption": "(x − 5) ÷ 2.24: the numbers now have mean 0 and std 1."},
                    {"t": 73.6, "caption": "Layer norm does this to every token's vector, then × gain + bias."}],
        "body": [
            """<p>To <b>normalize</b>, subtract the mean, then divide by the std. For 2, 4, 6, 8: (x − 5) ÷ 2.24 gives
about <b>−1.34, −0.45, 0.45, 1.34</b>. The mean is now 0 and the spread is 1, whatever the numbers were before.</p>""",
            """<p>That is exactly what <b>layer norm</b> does to every token's vector, followed by a learned
<b>scale and shift</b> (× gain + bias): the network itself learns which center and spread work best.</p>""",
            '<div class="figrow">{fig0}{fig1}</div>',
            """<div class="box key"><b class="t">Key idea</b><b>Normalize: (x − mean) ÷ std → mean 0, std 1.</b>
Layer norm = normalize each token's vector, then × gain + bias (both learned).</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "Normalize the two numbers <b>1, 3</b>.",
             "answer": "−1, 1.", "why": "Mean 2; distances −1 and 1; std = √1 = 1; so (1 − 2) ÷ 1 = −1 and "
                                       "(3 − 2) ÷ 1 = 1.", "key": {'parts': [{'label': '1', 'value': -1, 'tol': 0.5, 'unit': None}, {'label': '3', 'value': 1, 'tol': 0.5, 'unit': None}]}},
            {"kind": "mc", "q": "Which list normalizes to exactly the same numbers as 2, 4, 6, 8 "
                                "(−1.34, −0.45, 0.45, 1.34)?",
             "options": ["10, 20, 30, 40", "2, 4, 6, 9", "1, 2, 3", "2, 4, 6, 8, 10"],
             "answer": "A.", "why": "10, 20, 30, 40 is 5 × (2, 4, 6, 8). Subtracting the mean removes any shift and "
                                   "dividing by the std removes any scale, so only the pattern is left.", "key": {'choice': 0}},
            {"kind": "number", "q": "Layer norm has turned a vector into −1, 1. The learned gain is 2 and the bias is "
                                    "1. What comes out? What are its mean and std?",
             "lines": 2,
             "answer": "−1, 3 · mean 1, std 2.",
             "why": "−1 × 2 + 1 = −1 and 1 × 2 + 1 = 3. The bias sets the new center, the gain sets the new spread.", "key": {'parts': [{'label': '1st output', 'value': -1, 'tol': 0.5, 'unit': None}, {'label': '2nd output', 'value': 3, 'tol': 0.5, 'unit': None}, {'label': 'mean', 'value': 1, 'tol': 0.5, 'unit': None}, {'label': 'std', 'value': 2, 'tol': 0.5, 'unit': None}]}},
            {"kind": "tf", "q": "“After normalizing, every number lies between −1 and 1.”",
             "answer": "False.", "why": "Only the <i>typical</i> distance is 1. The video's own example gives −1.34 and "
                                        "1.34.", "key": {'value': False}},
        ],
    },
    {
        "title": "The spread of a sum grows like √n",
        "segment": (74, 85),
        "figures": [{"t": 84.9, "size": "small", "caption": "Simulated dot products: 4 times the terms, twice the "
                                                            "spread."}],
        "body": [
            """<p>Spread also explains a detail in attention. A dot product q · k = q₁k₁ + q₂k₂ + … + qₙkₙ adds up
<b>n products</b>. When the numbers are random, some products are positive and some negative, so they partly cancel.
The spread of the total still grows, but only with the <b>square root</b> of how many you add. The video simulated
20,000 random pairs (numbers with mean 0 and spread 1): n = 1 gives a spread of about 1, n = 4 about 2, n = 16
about 4.</p>""",
            "{fig0}",
            """<div class="box key"><b class="t">Key idea</b>Add up n random products and the spread of the total is
about <b>√n</b>. Four times as many terms only doubles the spread.</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "Following the pattern, what spread do you expect for n = 64? For n = 256?",
             "answer": "About 8 and about 16.", "why": "√64 = 8 and √256 = 16.", "key": {'parts': [{'label': 'n = 64', 'value': 8, 'tol': 0.5, 'unit': None}, {'label': 'n = 256', 'value': 16, 'tol': 0.5, 'unit': None}]}},
            {"kind": "mc", "q": "Going from n = 16 to n = 64 terms multiplies the spread by about…",
             "options": ["4", "2", "8", "16"],
             "answer": "B.", "why": "Four times as many terms: √64 ÷ √16 = 8 ÷ 4 = 2.", "key": {'choice': 1}},
            {"kind": "tf", "q": "“Add up 100 random products, each with a spread of about 1, and the total has a "
                                "spread of about 100.”",
             "answer": "False.", "why": "About √100 = 10: positive and negative products partly cancel.", "key": {'value': False}},
            {"kind": "number", "q": "GPT-2's vectors have 768 numbers. If you took the dot product of two random "
                                    "vectors that size, about how wide would the spread be?",
             "answer": "About 27.7.", "why": "√768 ≈ 27.7 (a simulation gives about 27.8).", "key": {'parts': [{'label': None, 'value': 27.7, 'tol': 0.554, 'unit': None}]}},
        ],
    },
    {
        "title": "Dividing by √d keeps softmax tame",
        "segment": (86, 103),
        "figures": [{"t": 90.5, "caption": "With 64 numbers, dot products spread about 8 times wider."},
                    {"t": 97.8, "caption": "÷ √64 = ÷ 8: spread about 1, and a well-behaved softmax."}],
        "body": [
            """<p>With vectors of 64 numbers, dot products spread about √64 = <b>8</b> times wider. Fed straight into
softmax, such big gaps make it far too sharp: in the video, one key gets 0.99 and the others almost nothing. Dividing
by the square root of the key size, <b>√d</b>, brings the spread back to about 1, which keeps the softmax well
behaved.</p>""",
            '<div class="figrow">{fig0}{fig1}</div>',
            """<p>In NumPy, mean and std are built in, and normalizing is one line:</p>""",
            """<pre class="code">x = np.array([2, 4, 6, 8])
x.mean(), x.std()                  # (5.0, 2.236)
(x - x.mean()) / x.std()           # [-1.34, -0.45, 0.45, 1.34]

q, k = np.random.randn(2, 64)
q @ k / np.sqrt(64)                # typically around ±1</pre>""",
            """<div class="box key"><b class="t">Key idea</b>Attention divides its scores by √d. That cancels the √d
growth of the spread, so softmax sees scores with a spread of about 1 and stays well behaved, not too sharp.</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "What do we divide the dot products by when the key size is d = 16? And when "
                                    "d = 128?",
             "answer": "4, and about 11.3.", "why": "√16 = 4 and √128 ≈ 11.3.", "key": {'parts': [{'label': 'd = 16', 'value': 4, 'tol': 0.5, 'unit': None}, {'label': 'd = 128', 'value': 11.3, 'tol': 0.226, 'unit': None}]}},
            {"kind": "mc", "q": "With d = 64 and no ÷ √d, what goes wrong?",
             "options": ["The scores spread so widely that softmax puts almost everything on one key",
                         "Softmax outputs negative numbers",
                         "The probabilities no longer add up to 1",
                         "All the dot products become 0"],
             "answer": "A.", "why": "Wide gaps make softmax sharp (F07). Dividing by √d = 8 is exactly softmax with "
                                   "temperature T = 8.", "key": {'choice': 0}},
            {"kind": "code", "q": "<b>Try it yourself.</b> Run the code below. (a) What do the first two "
                                  "<code>print</code>s show? (b) Predict the two numbers on the last line before running "
                                  "it. (c) Normalize <code>y = np.array([10, 20, 30, 40])</code> with the same one-liner. "
                                  "What do you get, and why?",
             "code": """import numpy as np
np.set_printoptions(precision=2)

x = np.array([2, 4, 6, 8])
print(x.mean(), x.std())
print((x - x.mean()) / x.std())

rng = np.random.default_rng(0)
d = 256
q = rng.standard_normal((10000, d))      # 10,000 random queries
k = rng.standard_normal((10000, d))      # 10,000 random keys
dots = (q * k).sum(axis=1)               # 10,000 dot products
print(dots.std(), (dots / np.sqrt(d)).std())""",
             "answer": "(a) 5.0 2.23606797749979 and [-1.34 -0.45  0.45  1.34] · (b) about 16 and about 1 "
                       "(15.94 and 1.00 with this seed) · (c) [-1.34 -0.45  0.45  1.34]",
             "why": """(a) The video's numbers. <code>x.std()</code> divides by the count n, exactly like the recipe.
(b) Each row of <code>q * k</code>, summed, is one dot product of 256 random numbers, so the spread is about √256 = 16;
dividing by √d = 16 brings it back to about 1. (c) 10, 20, 30, 40 is 5 × (2, 4, 6, 8), and normalizing removes both the
shift and the scale, so you get the same numbers."""},
        ],
    },
]
