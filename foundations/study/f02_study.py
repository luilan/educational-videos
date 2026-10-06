"""Study guide content for How LLMs Work · Foundations F02: The Dot Product.

Build:  python framework/study_guide.py foundations f02
Figures: {"t": seconds into the rendered video, "caption": ...}; placed in body blocks with {fig0}, {fig1}, ...
Exercise kinds: mc (options), tf, short, number, order, code. Every exercise has an answer and a why.
All numeric answers and code outputs were checked with NumPy.
"""

LESSON = {
    "series": "How LLMs Work · Foundations",
    "label": "F02",
    "title": "The Dot Product",
    "tagline": "One number that says how much two vectors agree",
    "duration": "1:52",
    "intro": """<p>This lesson has one big idea: the <b>dot product</b> turns two vectors into <b>one number</b> that
says how much they point the same way. The recipe is multiply matching numbers, then add. Divide by both lengths and you
get <b>cosine similarity</b>, a pure measure of direction. LLMs use dot products almost everywhere.</p>""",
    "prereq": """<div class="box note"><b>Before you start:</b> this builds on F01 (Vectors): lists of numbers,
arrows, and length by Pythagoras. The dot product is used in main episodes 3 (Embeddings), 5 and 6 (Attention), 8 (The
MLP) and 10 (From Vectors Back to Words). Concept 6 reads a few lines of NumPy.</div>""",
}

CONCEPTS = [
    {
        "title": "Multiply matching numbers, then add",
        "segment": (8, 37),
        "figures": [{"t": 27.2, "caption": "The recipe for a = (3, 1) and b = (2, 2): 3·2 = 6, 1·2 = 2, "
                                           "and 6 + 2 = 8."},
                    {"t": 36.3, "caption": "Same recipe for vectors of 768 numbers: 768 multiplications, then one "
                                           "sum."}],
        "body": [
            """<p>Given two vectors, a useful question is: <b>how much do they point the same way?</b> The
<b>dot product</b>, written <b>a · b</b>, answers it with a single number. The recipe: <b>multiply matching numbers,
then add everything up</b>. For a = (3, 1) and b = (2, 2): 3 × 2 + 1 × 2 = 6 + 2 = <b>8</b>.</p>""",
            """<p>It works in any number of dimensions, as long as both vectors have the <b>same number of
entries</b>, so every number has a partner. Two vectors of 768 numbers take 768 multiplications, then one big
sum.</p>""",
            '<div class="figrow">{fig0}{fig1}</div>',
            """<div class="box key"><b class="t">Key idea</b>a · b = <b>the sum of the products of matching
numbers</b>. Two vectors in, one number out.</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "Compute (1, 4) · (3, 2).",
             "answer": "11.", "why": "1 × 3 + 4 × 2 = 3 + 8 = 11.", "key": {'parts': [{'label': None, 'value': 11, 'tol': 0.5, 'unit': None}]}},
            {"kind": "number", "q": "(a) Compute (2, −1, 3) · (4, 5, 1). (b) Two vectors each have 768 numbers. How "
                                    "many multiplications and how many additions does their dot product take?",
             "lines": 2,
             "answer": "(a) 6 (b) 768 multiplications, 767 additions.",
             "why": "(a) 8 − 5 + 3 = 6. (b) One product per position; adding up 768 numbers takes 767 “+” signs.", "key": {'parts': [{'label': '(a)', 'value': 6, 'tol': 0.5, 'unit': None}, {'label': '(b) multiplications', 'value': 768, 'tol': 0.5, 'unit': None}, {'label': '(b) additions', 'value': 767, 'tol': 0.5, 'unit': None}]}},
            {"kind": "mc", "q": "Which of these dot products can you <b>not</b> compute?",
             "options": ["(1, 2) · (3, 4)", "(1, 2, 3) · (4, 5)", "(0, 0) · (5, 7)", "(−1, 2, 0) · (3, 3, 3)"],
             "answer": "B.", "why": "3 numbers against 2: the third has no partner. C is fine (it is 0), and so is D "
                                   "(it is 3).", "key": {'choice': 1}},
            {"kind": "tf", "q": "“a · b and b · a always give the same number.”",
             "answer": "True.", "why": "Each product is the same either way round (3 × 2 = 2 × 3), so the sum is "
                                       "the same.", "key": {'value': True}},
        ],
    },
    {
        "title": "Positive, zero, negative: agreement",
        "segment": (37, 50),
        "figures": [{"t": 46.5, "caption": "b turned to (−1, 3): at right angles to a, and a · b = 0."},
                    {"t": 49.8, "caption": "b = (−2, −1) points roughly opposite to a: a · b = −7."}],
        "body": [
            """<p>Geometrically, the dot product measures <b>agreement</b>. Keep a = (3, 1) and turn b:</p>""",
            """<table><tr><th>b</th><th>a · b</th><th>the arrows…</th></tr>
<tr><td>(3, 2)</td><td>3·3 + 1·2 = <b>11</b></td><td>point the same way: large positive</td></tr>
<tr><td>(−1, 3)</td><td>3·(−1) + 1·3 = <b>0</b></td><td>are at right angles: exactly zero</td></tr>
<tr><td>(−2, −1)</td><td>3·(−2) + 1·(−1) = <b>−7</b></td><td>point in opposite directions: negative</td></tr></table>""",
            '<div class="figrow">{fig0}{fig1}</div>',
            """<div class="box key"><b class="t">Key idea</b>a · b <b>&gt; 0</b>: they agree. a · b <b>= 0</b>: at
right angles. a · b <b>&lt; 0</b>: they point in opposite directions.</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "Compute (2, 1) · (−1, 2). What does the result tell you about the two arrows?",
             "answer": "0: they are at right angles.", "why": "−2 + 2 = 0, and a dot product of exactly zero means "
                                                              "the arrows are at right angles.", "key": {'parts': [{'label': None, 'value': 0, 'tol': 0.5, 'unit': None}]}},
            {"kind": "mc", "q": "a = (1, 1). Which b gives the most negative a · b?",
             "options": ["(2, 2)", "(1, −1)", "(−2, −1)", "(−1, 0)"],
             "answer": "C.", "why": "The dot products are 4, 0, −3 and −1. (−2, −1) points most nearly opposite to "
                                   "a, and it is long.", "key": {'choice': 2}},
            {"kind": "tf", "q": "“If a · b is negative, the angle between a and b is more than 90°.”",
             "answer": "True.", "why": "Zero is exactly 90°; below zero, the arrows lean away from each other.", "key": {'value': True}},
            {"kind": "number", "q": "Find the number k that makes (4, 2) · (1, k) = 0.",
             "answer": "k = −2.", "why": "4 × 1 + 2 × k = 0, so k = −2: the arrow (1, −2) is at right angles to "
                                         "(4, 2).", "key": {'parts': [{'label': None, 'value': -2, 'tol': 0.5, 'unit': None}]}},
        ],
    },
    {
        "title": "The shadow and the angle",
        "segment": (50, 67),
        "figures": [{"t": 59.2, "caption": "Project b onto a like a shadow: a · b = shadow length × length of a, "
                                           "8 ≈ 2.53 × 3.16."},
                    {"t": 66.3, "caption": "The second formula: ‖a‖ = √10 ≈ 3.16, ‖b‖ = √8 ≈ 2.83, "
                                           "θ ≈ 26.6°."}],
        "body": [
            """<p>Here's the picture: <b>project</b> b onto a, like a shadow. The dot product is the <b>length of that
shadow times the length of a</b>. For our example the shadow is about 2.53 long, a has length √10 ≈ 3.16, and
2.53 × 3.16 ≈ 8.</p>""",
            """<p>That gives a second formula: <b>a · b = ‖a‖ ‖b‖ cos θ</b>, where ‖a‖ is the length of a (F01) and θ
is the angle between the arrows. The cosine is 1 at 0°, 0 at 90° and −1 at 180°, which is why right angles give
exactly zero. Check: 3.16 × 2.83 × cos 26.6° ≈ 3.16 × 2.83 × 0.894 ≈ 8.</p>""",
            '<div class="figrow">{fig0}{fig1}</div>',
            """<div class="box key"><b class="t">Key idea</b>a · b = <b>‖a‖ ‖b‖ cos θ</b>: length × length ×
how well the directions line up. Same number as multiply-and-add, seen as geometry.</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "‖a‖ = 2, ‖b‖ = 5 and the angle between them is 60° (cos 60° = 0.5). What is "
                                    "a · b?",
             "answer": "5.", "why": "2 × 5 × 0.5 = 5.", "key": {'parts': [{'label': None, 'value': 5, 'tol': 0.5, 'unit': None}]}},
            {"kind": "number", "q": "a = (4, 0) and b = (3, 5). How long is b's shadow on a? Check that shadow × ‖a‖ "
                                    "equals a · b.",
             "lines": 2,
             "answer": "3; 3 × 4 = 12 = a · b.",
             "why": "a lies along the x-axis, so b's shadow is just b's x-part, 3. And a · b = 4 × 3 + 0 × 5 = 12.", "key": {'parts': [{'label': 'shadow', 'value': 3, 'tol': 0.5, 'unit': None}, {'label': 'a · b', 'value': 12, 'tol': 0.5, 'unit': None}]}},
            {"kind": "mc", "q": "Two vectors both have length 3. When is their dot product largest?",
             "options": ["When they are at right angles", "When they point the same way",
                         "When they point in opposite directions", "Never: it is always 9"],
             "answer": "B.", "why": "cos θ is largest (1) at θ = 0°, giving 3 × 3 × 1 = 9. At right angles it is 0; "
                                   "opposite, −9.", "key": {'choice': 1}},
            {"kind": "tf", "q": "“Making a twice as long, without changing its direction, doubles a · b.”",
             "answer": "True.", "why": "‖a‖ doubles and nothing else changes. Check: (6, 2) · (2, 2) = 16, twice 8.", "key": {'value': True}},
        ],
    },
    {
        "title": "Cosine similarity",
        "segment": (67, 81),
        "figures": [{"t": 80.9, "caption": "For our a and b, cosine similarity is 0.894: close to “same”."}],
        "body": [
            """<p>If we only care about <b>direction</b>, we divide the dot product by both lengths. What's left is the
cosine of the angle, called <b>cosine similarity</b>: cos θ = a · b / (‖a‖ ‖b‖). For a = (3, 1) and b = (2, 2):
8 / (√10 × √8) ≈ <b>0.894</b>. It always lies between −1 and 1: <b>1</b> means the same direction, <b>0</b>
unrelated (right angles), and <b>−1</b> opposite.</p>""",
            "{fig0}",
            """<div class="box key"><b class="t">Key idea</b>Cosine similarity = a · b / (‖a‖ ‖b‖): the dot product
with the lengths divided out. A score from <b>−1 to 1</b> for direction only.</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "What is the cosine similarity of (1, 0) and (1, 1)? Round to 3 decimal places.",
             "answer": "0.707.", "why": "a · b = 1, ‖a‖ = 1, ‖b‖ = √2, so 1 / √2 ≈ 0.707 (the angle is 45°).", "key": {'parts': [{'label': None, 'value': 0.707, 'tol': 0.0005, 'unit': None}]}},
            {"kind": "mc", "q": "Which pair has cosine similarity exactly −1?",
             "options": ["(1, 2) and (2, 4)", "(1, 2) and (−2, 1)", "(1, 2) and (−3, −6)", "(1, 2) and (−1, 2)"],
             "answer": "C.", "why": "(−3, −6) = −3 × (1, 2): exactly opposite. A is +1 (same direction), B is 0 "
                                   "(right angles), D is 0.6.", "key": {'choice': 2}},
            {"kind": "tf", "q": "“(3, 1) and (30, 10) have a larger cosine similarity than (3, 1) and (3, 1), "
                                "because the numbers are bigger.”",
             "answer": "False.", "why": "Both are exactly 1: same direction. Only the raw dot product grows "
                                        "(100 versus 10).", "key": {'value': False}},
            {"kind": "number", "q": "What is the cosine similarity of (3, 4) and (4, 3)?",
             "answer": "0.96.", "why": "a · b = 12 + 12 = 24, and both lengths are 5, so 24 / 25 = 0.96: very "
                                       "similar directions.", "key": {'parts': [{'label': None, 'value': 0.96, 'tol': 0.005, 'unit': None}]}},
        ],
    },
    {
        "title": "Dot products everywhere",
        "segment": (81, 93),
        "figures": [{"t": 92.9, "caption": "Three places LLMs use dot products."}],
        "body": [
            """<p>Language models use dot products everywhere. <b>Attention</b> compares a <i>query</i> vector with a
<i>key</i> vector (episodes 5 and 6). The <b>output layer</b> compares the final vector with a vector for <b>every
word</b> (episode 10). And each <b>MLP neuron</b> is a dot product of its weights with the input (episode 8).
Each time the question is: <b>how much do these two vectors agree?</b></p>""",
            "{fig0}",
            """<div class="box key"><b class="t">Key idea</b>Whenever an LLM compares two vectors (query and key,
vector and word, weights and input), it computes a <b>dot product</b>.</div>""",
        ],
        "exercises": [
            {"kind": "mc", "q": "In attention, a query is compared with a key. What does a large positive "
                                "query · key mean?",
             "options": ["The two tokens are far apart in the text", "The query and key point roughly the same way: "
                         "a good match", "The key vector is longer than the query vector", "The two tokens are the "
                         "same word"],
             "answer": "B.", "why": "A large positive dot product means strong agreement. It says nothing about "
                                   "distance in the text, and length alone doesn't decide it.", "key": {'choice': 1}},
            {"kind": "number", "q": "A tiny output layer knows 3 words: <i>cat</i> = (1, 2), <i>dog</i> = (2, 1), "
                                    "<i>mat</i> = (−1, 1). The final vector is h = (2, 3). Compute h · each word. "
                                    "Which word scores highest?",
             "answer": "cat 8, dog 7, mat 1: cat.", "why": "2 + 6 = 8, 4 + 3 = 7, −2 + 3 = 1. h points most nearly "
                                                          "the same way as cat.", "key": {'parts': [{'label': 'cat', 'value': 8, 'tol': 0.5, 'unit': None}, {'label': 'dog', 'value': 7, 'tol': 0.5, 'unit': None}, {'label': 'mat', 'value': 1, 'tol': 0.5, 'unit': None}]}},
            {"kind": "number", "q": "An MLP neuron has weights (0.5, −1, 2) and receives the input (4, 1, 0.5). "
                                    "What is its dot product?",
             "answer": "2.", "why": "0.5 × 4 − 1 × 1 + 2 × 0.5 = 2 − 1 + 1 = 2.", "key": {'parts': [{'label': None, 'value': 2, 'tol': 0.5, 'unit': None}]}},
            {"kind": "number", "q": "An output layer compares a 768-number vector with every word of a 50,000-word "
                                    "vocabulary. How many dot products is that, and how many multiplications in "
                                    "total?",
             "answer": "50,000 dot products; 38,400,000 multiplications.",
             "why": "One dot product per word, each with 768 multiplications: 50,000 × 768. F03 shows how matrices do "
                    "them all at once.", "key": {'parts': [{'label': 'dot products', 'value': 50000, 'tol': 0.5, 'unit': None}, {'label': 'multiplications', 'value': 38400000, 'tol': 0.5, 'unit': None}]}},
        ],
    },
    {
        "title": "The dot product in NumPy",
        "segment": (93, 109),
        "figures": [{"t": 99.8, "size": "small", "caption": "The code from the video: @ gives the dot product; "
                                                            "dividing by both lengths gives cosine similarity."}],
        "body": [
            """<p>In NumPy, the dot product is the <b>@</b> sign: <code>a @ b</code>. Divide by the two lengths
(<code>np.linalg.norm</code>, from F01) and you get cosine similarity.</p>""",
            "{fig0}",
            """<pre class="code">a = np.array([3, 1])
b = np.array([2, 2])
a @ b                                     # 8
cos = a @ b / (np.linalg.norm(a) * np.linalg.norm(b))
cos                                       # 0.894</pre>""",
            """<p>You'll meet the dot product in episodes 3, 5, 6, 8 and 10. Next, F03 does lots of dot products at
once, with matrices.</p>""",
            """<div class="box key"><b class="t">Key idea</b><code>a @ b</code> is the dot product;
<code>a @ b / (norm(a) * norm(b))</code> is cosine similarity.</div>""",
        ],
        "exercises": [
            {"kind": "mc", "q": "What does <code>a * b</code> (a star, not @) give for the a and b above?",
             "options": ["8", "array([6, 2])", "0.894", "An error"],
             "answer": "B.", "why": "<code>*</code> multiplies matching numbers but doesn't add them up. "
                                   "<code>np.sum(a * b)</code> is 8, the same as <code>a @ b</code>.", "key": {'choice': 1}},
            {"kind": "short", "q": "What does <code>np.array([1, 2, 3]) @ np.array([4, 5, 6])</code> return?",
             "answer": "32.", "why": "4 + 10 + 18 = 32."},
            {"kind": "code", "q": "<b>Try it yourself.</b> Below are toy 3-number “word vectors”. (a) Complete "
                                  "<code>cos_sim(a, b)</code>. (b) Before running: which word should come out more "
                                  "similar to <i>cat</i>? Then run it: what are the two numbers? (c) Replace "
                                  "<code>words[\"cat\"]</code> by <code>10 * words[\"cat\"]</code>. Which changes: "
                                  "the dot product, the cosine similarity, or both?",
             "code": """import numpy as np

words = {
    "cat": np.array([0.9, 0.8, 0.1]),
    "dog": np.array([0.8, 0.9, 0.2]),
    "car": np.array([0.1, 0.2, 0.9]),
}

def cos_sim(a, b):
    ...                        # (a) your code here

for w in ["dog", "car"]:
    print(w, round(cos_sim(words["cat"], words[w]), 3))""",
             "answer": "(b) dog 0.99, car 0.303 · (c) only the dot product",
             "why": """(a) <code>def cos_sim(a, b): return a @ b / (np.linalg.norm(a) * np.linalg.norm(b))</code>.
(b) cat and dog point almost the same way (0.99); car points elsewhere (0.303). (c) The dot product with dog grows
10 times (1.46 → 14.6), but the cosine similarity stays 0.99: dividing by the lengths removes the scaling."""},
        ],
    },
]
