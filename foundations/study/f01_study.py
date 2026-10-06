"""Study guide content for How LLMs Work · Foundations F01: Vectors: Lists of Numbers as Arrows.

Build:  python framework/study_guide.py foundations f01
Figures: {"t": seconds into the rendered video, "caption": ...}; placed in body blocks with {fig0}, {fig1}, ...
Exercise kinds: mc (options), tf, short, number, order, code. Every exercise has an answer and a why.
All numeric answers and code outputs were checked with NumPy.
"""

LESSON = {
    "series": "How LLMs Work · Foundations",
    "label": "F01",
    "title": "Vectors: Lists of Numbers as Arrows",
    "tagline": "The basic building block of every LLM",
    "duration": "1:56",
    "intro": """<p>This lesson has one big idea: a <b>vector</b> is a <b>list of numbers</b> that you can also picture as
an <b>arrow</b>. You can add vectors, stretch them and measure their length, and the rules are the same whether a vector
has 2 numbers or 768. Every token inside an LLM is represented by one.</p>""",
    "prereq": """<div class="box note"><b>Before you start:</b> this is the first Foundations video, so nothing earlier
is needed: adding, multiplying and a square root are enough. Vectors are used throughout the main series, especially in
episodes 1 (What is an LLM?), 3 (Embeddings), 4 (Position) and 9 (The Transformer Block). Concept 7 reads a few lines of
NumPy; F11 (NumPy in Three Minutes) covers more.</div>""",
}

CONCEPTS = [
    {
        "title": "A vector is a list of numbers",
        "segment": (8, 27),
        "figures": [{"t": 27.2, "caption": "Two vectors: [2, 1] has dimension 2, and [0.3, −1.2, 0.8, 2.0] "
                                           "(illustrative values) has dimension 4."}],
        "body": [
            """<p>Everything an LLM does, it does with vectors: every token, every hidden state and every prediction
starts as one. At its simplest, a <b>vector</b> is just a <b>list of numbers</b>, such as [2, 1]. The <b>order</b>
matters: each position has its own meaning, so [2, 1] and [1, 2] are different vectors.</p>""",
            """<p>How many numbers a vector has is called its <b>dimension</b>. [2, 1] has dimension 2;
[0.3, −1.2, 0.8, 2.0] has dimension 4.</p>""",
            "{fig0}",
            """<div class="box key"><b class="t">Key idea</b>A vector is an <b>ordered list of numbers</b>. Its
<b>dimension</b> is how many numbers it has.</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "What is the dimension of each vector? (a) [5, 0, −2] (b) [7] "
                                    "(c) [0, 0, 0, 0, 0, 0]",
             "answer": "(a) 3 (b) 1 (c) 6.", "why": "Count the numbers. Zeros and negative numbers count like any "
                                                    "other entry.", "key": {'parts': [{'label': '(a)', 'value': 3, 'tol': 0.5, 'unit': None}, {'label': '(b)', 'value': 1, 'tol': 0.5, 'unit': None}, {'label': '(c)', 'value': 6, 'tol': 0.5, 'unit': None}]}},
            {"kind": "tf", "q": "“[2, 1] and [1, 2] are the same vector, because they contain the same numbers.”",
             "answer": "False.", "why": "Order matters: the first position and the second position mean different "
                                        "things. As arrows, one goes 2 right and 1 up, the other 1 right and 2 up.", "key": {'value': False}},
            {"kind": "short", "q": "You describe a flat as [bedrooms, bathrooms, floor area in m², year built] = "
                                   "[3, 2, 95, 1987]. What is its dimension, and what does the second number mean?",
             "lines": 2,
             "answer": "Dimension 4; the second number is the number of bathrooms.",
             "why": "Four numbers, and each position has a fixed meaning. An LLM's vectors work the same way, except "
                    "that the meanings are learned, not chosen by hand."},
        ],
    },
    {
        "title": "Two numbers, one arrow",
        "segment": (27, 51),
        "figures": [{"t": 37.7, "caption": "[2, 1] as an arrow: 2 to the right, then 1 up, starting at the origin."},
                    {"t": 50.4, "caption": "Three numbers give an arrow in 3-D. GPT-2 uses 768 numbers per vector "
                                           "(values illustrative)."}],
        "body": [
            """<p>With two numbers we can draw a vector. The first number says how far to go <b>right</b>, the second
how far to go <b>up</b>. So [2, 1] is the <b>arrow from the origin</b> (the point [0, 0]) to the point 2 right, 1 up.
Negative numbers go left or down.</p>""",
            """<p>Three numbers give an arrow in three-dimensional space. Beyond that we can't draw it, but the
calculations work <b>exactly the same</b>: adding, scaling and length (the next concepts) don't care how many numbers
there are. GPT-2 uses vectors with <b>768</b> numbers.</p>""",
            '<div class="figrow">{fig0}{fig1}</div>',
            """<div class="box key"><b class="t">Key idea</b>A vector is both a <b>list of numbers</b> and an
<b>arrow from the origin</b>. The picture stops at 3 dimensions; the arithmetic doesn't.</div>""",
        ],
        "exercises": [
            {"kind": "short", "q": "Describe the arrow for [−3, 2]: starting at the origin, which way and how far do "
                                   "you go?",
             "answer": "3 to the left, then 2 up.",
             "why": "The first number is horizontal (negative means left), the second is vertical (positive means up)."},
            {"kind": "mc", "q": "An arrow starts at the origin and ends 1 to the right and 4 down. Which vector is it?",
             "options": ["[4, 1]", "[1, 4]", "[1, −4]", "[−1, 4]"],
             "answer": "C.", "why": "Right comes first (1), then up/down (4 down = −4). A swaps the order; B and D get "
                                   "the signs wrong.", "key": {'choice': 2}},
            {"kind": "tf", "q": "“Vectors with 768 numbers can't be drawn, so the rules for adding them are "
                                "different.”",
             "answer": "False.", "why": "The picture stops at 3 dimensions, but the rules are exactly the same for "
                                        "any number of entries.", "key": {'value': False}},
            {"kind": "number", "q": "A model uses one vector of dimension 768 per token. How many numbers does it need "
                                    "for the 5 tokens of <i>“The cat sat on the”</i>?",
             "answer": "3,840.", "why": "5 vectors × 768 numbers each = 3,840.", "key": {'parts': [{'label': None, 'value': 3840, 'tol': 0.5, 'unit': None}]}},
        ],
    },
    {
        "title": "Adding vectors",
        "segment": (51, 62),
        "figures": [{"t": 61.8, "caption": "[2, 1] + [1, 2] = [3, 3]. Put b at the tip of a; the sum a + b goes "
                                           "from the origin to where b now ends."}],
        "body": [
            """<p>To add two vectors, add their numbers <b>position by position</b>:
[2, 1] + [1, 2] = [2 + 1, 1 + 2] = [3, 3]. Both vectors need the same dimension, so that every number has a
partner.</p>""",
            """<p>As arrows, adding means <b>placing one arrow at the tip of the other</b>. The sum is the arrow from
the origin to the tip of the second one: walk along a, then along b, and a + b is the shortcut from start to finish.</p>""",
            "{fig0}",
            """<div class="box key"><b class="t">Key idea</b>Add vectors <b>position by position</b>. As arrows:
<b>tip to tail</b>, and the sum goes from the start of the first to the end of the second.</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "Compute [4, −1, 2] + [1, 3, −2].",
             "answer": "[5, 2, 0].", "why": "Position by position: 4 + 1, −1 + 3, 2 + (−2).", "key": {'parts': [{'label': '1st', 'value': 5, 'tol': 0.5, 'unit': None}, {'label': '2nd', 'value': 2, 'tol': 0.5, 'unit': None}, {'label': '3rd', 'value': 0, 'tol': 0.5, 'unit': None}]}},
            {"kind": "number", "q": "Start at the origin, walk along [2, 1], then along [−3, 2]. Where do you end up?",
             "answer": "[−1, 3].", "why": "Walking tip to tail is vector addition: [2 + (−3), 1 + 2] = [−1, 3].", "key": {'parts': [{'label': 'x', 'value': -1, 'tol': 0.5, 'unit': None}, {'label': 'y', 'value': 3, 'tol': 0.5, 'unit': None}]}},
            {"kind": "tf", "q": "“[1, 2] + [2, 1] gives a different result from [2, 1] + [1, 2].”",
             "answer": "False.", "why": "Both are [3, 3]. Adding numbers doesn't depend on order, so neither does "
                                        "adding vectors.", "key": {'value': False}},
            {"kind": "mc", "q": "Why can't you add [1, 2] and [1, 2, 3]?",
             "options": ["Because the result would be negative", "Because the third number has no partner: both "
                         "vectors must have the same dimension", "Because only vectors of dimension 2 can be added",
                         "You can: the answer is [2, 4, 3]"],
             "answer": "B.", "why": "Addition pairs up positions. With 2 and 3 numbers there is nothing to pair with the "
                                   "3 (D just invents a 0).", "key": {'choice': 1}},
        ],
    },
    {
        "title": "Scaling: stretch, shrink, flip",
        "segment": (62, 73),
        "figures": [{"t": 70.5, "caption": "2 × [2, 1] = [4, 2]: the same direction, twice as long."},
                    {"t": 73.0, "caption": "−1 × [2, 1] = [−2, −1]: a negative number flips the arrow around."}],
        "body": [
            """<p>Multiplying a vector by a single number, called a <b>scalar</b>, multiplies <b>every</b> number in
it: 2 × [2, 1] = [4, 2]. The arrow keeps its direction and becomes twice as long.</p>""",
            """<p>A scalar between 0 and 1 shrinks the arrow: 0.5 × [4, 2] = [2, 1]. A <b>negative</b> scalar flips it
around, so it points the opposite way: −1 × [2, 1] = [−2, −1].</p>""",
            '<div class="figrow">{fig0}{fig1}</div>',
            """<div class="box key"><b class="t">Key idea</b>A scalar multiplies <b>every</b> number of the vector.
The arrow stretches or shrinks along the same line; a <b>negative</b> scalar also reverses it.</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "Compute 3 × [1, −2, 0].",
             "answer": "[3, −6, 0].", "why": "Every entry is multiplied by 3, including the 0.", "key": {'parts': [{'label': '1st', 'value': 3, 'tol': 0.5, 'unit': None}, {'label': '2nd', 'value': -6, 'tol': 0.5, 'unit': None}, {'label': '3rd', 'value': 0, 'tol': 0.5, 'unit': None}]}},
            {"kind": "mc", "q": "Which vector points in exactly the opposite direction to [2, 1]?",
             "options": ["[1, 2]", "[−2, 1]", "[−4, −2]", "[2, −1]"],
             "answer": "C.", "why": "[−4, −2] = −2 × [2, 1]: flipped and twice as long. B and D flip only one number, "
                                   "which gives a different direction, not the opposite one.", "key": {'choice': 2}},
            {"kind": "tf", "q": "“0.5 × [4, 2] points the same way as [4, 2], but is half as long.”",
             "answer": "True.", "why": "It is [2, 1]: a positive scalar keeps the direction, and 0.5 halves the "
                                       "length.", "key": {'value': True}},
            {"kind": "number", "q": "Compute 2 × [1, 3] + [−1, 0].",
             "answer": "[1, 6].", "why": "Scale first: 2 × [1, 3] = [2, 6]. Then add: [2 − 1, 6 + 0] = [1, 6].", "key": {'parts': [{'label': 'x', 'value': 1, 'tol': 0.5, 'unit': None}, {'label': 'y', 'value': 6, 'tol': 0.5, 'unit': None}]}},
        ],
    },
    {
        "title": "Length, by Pythagoras",
        "segment": (73, 81),
        "figures": [{"t": 81.2, "caption": "The legs 3 and 4 form a right triangle, so the arrow [3, 4] has length "
                                           "√(3² + 4²) = √25 = 5."}],
        "body": [
            """<p>The <b>length</b> of a vector comes from Pythagoras: <b>square each number, add them up, and take the
square root</b>. For [3, 4]: √(3² + 4²) = √(9 + 16) = √25 = <b>5</b>. The length of a vector a is often written
‖a‖; you'll see this in F02.</p>""",
            "{fig0}",
            """<p>The same recipe works in any dimension, with more numbers to square. For [2, 3, 6]:
√(4 + 9 + 36) = √49 = 7. Squares are never negative, so neither is a length.</p>""",
            """<div class="box key"><b class="t">Key idea</b>Length = the <b>square root of the sum of the
squares</b> of the numbers. It works in any number of dimensions.</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "What is the length of [6, 8]?",
             "answer": "10.", "why": "√(36 + 64) = √100 = 10. It is 2 × [3, 4], so it is twice as long as [3, 4].", "key": {'parts': [{'label': None, 'value': 10, 'tol': 0.5, 'unit': None}]}},
            {"kind": "number", "q": "What is the length of [1, 1], to 3 decimal places?",
             "answer": "1.414.", "why": "√(1 + 1) = √2 ≈ 1.414. A length doesn't have to be a whole number.", "key": {'parts': [{'label': None, 'value': 1.414, 'tol': 0.0005, 'unit': None}]}},
            {"kind": "number", "q": "The length of [3, 4] is 5. Without squaring anything, what is the length of "
                                    "−2 × [3, 4]?",
             "answer": "10.", "why": "The −2 flips the arrow and doubles it. Flipping doesn't change length, so it "
                                     "is 2 × 5 = 10 (check: [−6, −8] has length √100 = 10).", "key": {'parts': [{'label': None, 'value': 10, 'tol': 0.5, 'unit': None}]}},
            {"kind": "mc", "q": "Which vector has length exactly 3?",
             "options": ["[1, 2]", "[3, 3]", "[2, 1, 2]", "[1, 1, 1]"],
             "answer": "C.", "why": "√(4 + 1 + 4) = √9 = 3. The others: √5 ≈ 2.24, √18 ≈ 4.24, √3 ≈ 1.73.", "key": {'choice': 2}},
        ],
    },
    {
        "title": "Directions that mean something",
        "segment": (81, 92),
        "figures": [{"t": 92.4, "caption": "An illustrative space of meaning: one direction leans toward animals, "
                                           "another toward plurals. The same “plural” arrow links cat → cats, "
                                           "dog → dogs and mat → mats."}],
        "body": [
            """<p>In an LLM, <b>directions</b> can come to mean something. One direction might lean toward animals,
another toward plurals. A token's vector is a <b>point in this space of meaning</b>, and moving along the “plural”
direction takes <i>cat</i> toward <i>cats</i>.</p>""",
            "{fig0}",
            """<p>The picture is illustrative. A real model learns its directions from data, in hundreds of
dimensions, and they are usually not as neat as one arrow per idea. Episode 3 (Embeddings) looks at this properly.</p>""",
            """<div class="box key"><b class="t">Key idea</b>A token's vector is a <b>point in a space of
meaning</b>. Directions in that space can stand for ideas such as “animal” or “plural”.</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "In a toy 2-D space, <i>cat</i> = [1, 3] and <i>cats</i> = [4, 3]. (a) Which "
                                    "vector must you add to <i>cat</i> to get <i>cats</i>? (b) <i>dog</i> = [2, 5]. If "
                                    "the same “plural” arrow applies, where do you expect <i>dogs</i>?",
             "lines": 2,
             "answer": "(a) [3, 0] (b) [5, 5].",
             "why": "[1, 3] + [3, 0] = [4, 3]. Adding the same plural arrow to dog: [2 + 3, 5 + 0] = [5, 5].", "key": {'parts': [{'label': '(a) 1st', 'value': 3, 'tol': 0.5, 'unit': None}, {'label': '(a) 2nd', 'value': 0, 'tol': 0.5, 'unit': None}, {'label': '(b) 1st', 'value': 5, 'tol': 0.5, 'unit': None}, {'label': '(b) 2nd', 'value': 5, 'tol': 0.5, 'unit': None}]}},
            {"kind": "tf", "q": "“Each direction in an LLM's vectors was given its meaning by hand, e.g. an engineer "
                                "decided which direction means ‘animal’.”",
             "answer": "False.", "why": "The directions come to mean something through training on data; nobody "
                                        "assigns them by hand.", "key": {'value': False}},
            {"kind": "mc", "q": "A token's vector is best thought of as…",
             "options": ["a point (or arrow) in a space where directions can carry meaning",
                         "the dictionary definition of the word, stored as text",
                         "the token's position in the sentence",
                         "a probability for every word in the vocabulary"],
             "answer": "A.", "why": "It is a point in a space of meaning. Position (C) is added separately in episode "
                                   "4; probabilities (D) are the model's output, not a token's vector.", "key": {'choice': 0}},
        ],
    },
    {
        "title": "Vectors in NumPy",
        "segment": (92, 114),
        "figures": [{"t": 97.6, "size": "small", "caption": "The NumPy code from the video: adding, scaling and "
                                                            "measuring length take one line each."}],
        "body": [
            """<p>In NumPy, a vector is an <b>array</b>: <code>np.array([2, 1])</code>. The operations of this lesson
take one line each: <code>a + b</code> adds position by position, <code>2 * a</code> scales, and
<code>np.linalg.norm(a)</code> gives the length.</p>""",
            "{fig0}",
            """<pre class="code">import numpy as np
a = np.array([2, 1])
b = np.array([1, 2])
a + b                               # array([3, 3])
2 * a                               # array([4, 2])
np.linalg.norm(np.array([3, 4]))    # 5.0</pre>""",
            """<p>You'll meet vectors in almost every episode: as <b>embeddings</b> in episode 3, <b>added to
positions</b> in episode 4, and flowing along the <b>residual stream</b> in episode 9.</p>""",
            """<div class="box key"><b class="t">Key idea</b>A vector is a NumPy array. <code>+</code>,
<code>*</code> and <code>np.linalg.norm</code> do the work, for 2 numbers or 768.</div>""",
        ],
        "exercises": [
            {"kind": "short", "q": "With <code>a</code> and <code>b</code> as above, what does <code>a + 3 * b</code> "
                                   "give?",
             "answer": "array([5, 7]).", "why": "3 * b = [3, 6], then [2 + 3, 1 + 6] = [5, 7]."},
            {"kind": "mc", "q": "What does <code>np.linalg.norm(np.array([2, 1, 2]))</code> return?",
             "options": ["5.0", "3.0", "9.0", "array([4, 1, 4])"],
             "answer": "B.", "why": "√(4 + 1 + 4) = √9 = 3. A just adds the numbers; C forgets the square root; D "
                                   "only squares them.", "key": {'choice': 1}},
            {"kind": "code", "q": "<b>Try it yourself.</b> (a) Complete <code>length(v)</code> without using "
                                  "<code>np.linalg.norm</code>: use <code>v ** 2</code> (square every entry), "
                                  "<code>np.sum</code> and <code>np.sqrt</code>. (b) Predict what the three "
                                  "<code>print</code> lines show, then run it. (c) Now try "
                                  "<code>a + np.array([1, 2, 3])</code>. What happens, and why?",
             "code": """import numpy as np

a = np.array([2, 1])
b = np.array([1, 2])

def length(v):
    ...                     # (a) your code here

print(a + b, 2 * a, -1 * a)
print(length(np.array([3, 4])), np.linalg.norm(np.array([3, 4])))
print(length(2 * a) / length(a))""",
             "answer": "(b) [3 3] [4 2] [-2 -1], then 5.0 5.0, then 2.0 · (c) a ValueError",
             "why": """(a) <code>def length(v): return np.sqrt(np.sum(v ** 2))</code>. (b) Line 1: add, scale, flip.
Line 2: your function and NumPy agree on 5.0. Line 3: doubling a vector doubles its length, so the ratio is 2.0.
(c) NumPy raises <code>ValueError: operands could not be broadcast together with shapes (2,) (3,)</code>: a has 2
numbers and the other array has 3, so the third position has no partner."""},
        ],
    },
]
