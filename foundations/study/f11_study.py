"""Study guide content for How LLMs Work · Foundations, F11: NumPy in Three Minutes.

Build:  python framework/study_guide.py foundations f11 --video <NumPyVideo.mp4>
Figures: {"t": seconds into the rendered video, "caption": ...}; placed in body blocks with {fig0}, {fig1}, ...
Exercise kinds: mc (options), tf, short, number, order, code. Every exercise has an answer and a why.
All numbers, shapes and code outputs were checked by running NumPy.
"""

LESSON = {
    "series": "How LLMs Work · Foundations",
    "label": "F11",
    "title": "NumPy in Three Minutes",
    "tagline": "Just enough to read the code in the series",
    "duration": "2:03",
    "intro": """<p>This lesson is a quick tour of <b>NumPy</b>, Python's library for fast math on arrays of numbers. It
covers just enough to read every code snippet in the series: <b>shapes</b>, element-wise arithmetic, the <code>@</code>
sign, <b>axis</b> and <b>keepdims</b>, <b>broadcasting</b>, <b>reshape</b> and <b>masks</b>. Most questions ask you to
<b>predict</b> what a line of code gives. Write your prediction down first, then run the code to check.</p>""",
    "prereq": """<div class="box note"><b>Before you start:</b> this lesson puts earlier Foundations into code: vectors
(F01), the dot product (F02), matrices (F03) and softmax (F07). You will meet this NumPy in the code of episodes 3, 6, 7,
9 and 13. To run the code questions you need Python with NumPy (<code>pip install numpy</code>).</div>""",
}

Y_CODE = """Y = np.array([[2, 1, 0, 3], [1, 4, 2, 1], [0, 2, 5, 1],
              [3, 3, 1, 2], [1, 0, 2, 2]])"""

E_CODE = """E = np.array([[1, 2, 4, 2, 1], [1, 1, 1, 1, 1], [2, 6, 4, 4, 4],
              [5, 5, 10, 3, 2], [6, 1, 1, 1, 1]])"""

CONCEPTS = [
    {
        "title": "Arrays and shapes",
        "segment": (8, 29),
        "figures": [{"t": 28.6, "caption": "A vector (1-D), and a matrix (2-D) of shape (5, 4)."}],
        "body": [
            """<p>NumPy works with <b>arrays</b>: numbers in a grid. A <b>vector</b> is a one-dimensional (1-D) array, a
<b>matrix</b> a two-dimensional (2-D) one. Every array has a <b>shape</b>: the matrix below, one row per token of
<i>“The cat sat on the”</i>, has shape <b>(5, 4)</b>, 5 rows of 4 numbers. A 1-D shape has a trailing comma: a vector
of 4 numbers has shape <code>(4,)</code>.</p>""",
            "{fig0}",
            """<div class="box key"><b class="t">Key idea</b>An array is a grid of numbers, and its <b>shape</b> lists the
size of each dimension, rows first. In the series, a matrix of shape (tokens, numbers per token) is everywhere.</div>""",
        ],
        "exercises": [
            {"kind": "mc", "q": "An array holds 3 rows of 2 numbers each. What is its shape?",
             "options": ["(2, 3)", "(3, 2)", "(6,)", "(3,)"],
             "answer": "B.", "why": "Rows first, then the numbers in each row. (6,) would be the same 6 numbers as one "
                                   "flat vector.", "key": {'choice': 1}},
            {"kind": "number", "q": "A sentence has 7 tokens, and each token is a vector of 768 numbers. Stacked as one "
                                    "matrix (one row per token, as in the picture), what is its shape, and how many "
                                    "numbers does it hold?",
             "answer": "(7, 768), 5,376 numbers.", "why": "One row per token, 768 numbers per row: 7 × 768 = 5,376.", "key": {'parts': [{'label': 'rows', 'value': 7, 'tol': 0.5, 'unit': None}, {'label': 'columns', 'value': 768, 'tol': 0.5, 'unit': None}, {'label': 'numbers', 'value': 5376, 'tol': 0.5, 'unit': None}]}},
            {"kind": "code", "q": "Predict the three shapes this prints. (<code>X[0]</code> is the first row.)",
             "code": """import numpy as np
v = np.array([0.3, -1.1, 2.4, 0.8])
X = np.zeros((5, 4))      # 5 rows of 4 zeros
print(v.shape, X.shape, X[0].shape)""",
             "answer": "(4,) (5, 4) (4,)",
             "why": "v is 1-D with 4 numbers. <code>X[0]</code> is one row, one token's vector of 4 numbers, so it is "
                    "1-D too: (4,)."},
        ],
    },
    {
        "title": "Arithmetic on every element at once",
        "segment": (29, 39),
        "figures": [{"t": 38.9, "caption": "A + B adds matching elements; A × 2 doubles every number."}],
        "body": [
            """<p>Arithmetic works on <b>every element at once</b>, with no loops. Add two arrays of the <b>same
shape</b> and matching elements are added: 1 + 10, 2 + 20, and so on. Multiply by 2 and every number doubles. The same
goes for <code>-</code>, <code>/</code> and <code>**</code>, and for <code>*</code> between two arrays: <code>A * B</code>
multiplies matching elements.</p>""",
            "{fig0}",
            """<div class="box key"><b class="t">Key idea</b>Arithmetic is <b>element-wise</b>: each cell of the result
uses only the matching cells of the inputs. A single number, like the 2 in A × 2, applies to every element.</div>""",
        ],
        "exercises": [
            {"kind": "code", "q": "Work out the three printed arrays by hand, then run it.",
             "code": """import numpy as np
A = np.array([[1, 2, 3], [4, 5, 6]])
B = np.array([[10, 20, 30], [40, 50, 60]])
print(B - A)
print(A * 2 + 1)
print(A * B)""",
             "answer": "[[9 18 27] [36 45 54]] · [[3 5 7] [9 11 13]] · [[10 40 90] [160 250 360]]",
             "why": "Each position on its own: 10 − 1 = 9, 20 − 2 = 18, …; 1 × 2 + 1 = 3, 2 × 2 + 1 = 5, …; and "
                    "<code>*</code> multiplies matching elements: 1 × 10, 2 × 20, …, 6 × 60 = 360."},
            {"kind": "tf", "q": "“A of shape (2, 3) plus B of shape (3, 2) works, because both hold 6 numbers.”",
             "answer": "False.", "why": "Element-wise needs the elements to match up. NumPy raises an error: "
                                        "“operands could not be broadcast together”.", "key": {'value': False}},
            {"kind": "mc", "q": "<code>x = np.array([1.0, 2.0, 3.0])</code>. Which expression gives [1, 4, 9]?",
             "options": ["<code>x ** 2</code>", "<code>x @ x</code>", "<code>x * 2</code>", "<code>np.sum(x) ** 2</code>"],
             "answer": "A.", "why": "<code>**</code> squares each element. <code>x @ x</code> is the dot product 14 "
                                   "(next concept), <code>x * 2</code> is [2, 4, 6], and D squares the total: 36.", "key": {'choice': 0}},
        ],
    },
    {
        "title": "The @ sign: matrix multiplication",
        "segment": (40, 49),
        "figures": [{"t": 48.9, "caption": "Each cell of X @ W: one row of X dotted with one column of W."}],
        "body": [
            """<p><code>@</code> is <b>matrix multiplication</b>: each cell of the result is the <b>dot product</b> of one
row of the left array with one column of the right array (F02, F03). For the shapes: <b>(5, 4) @ (4, 4) → (5, 4)</b>.
The inner sizes (the green 4s) must match; the result takes its rows from the left array and its columns from the
right one.</p>""",
            "{fig0}",
            """<div class="box key"><b class="t">Key idea</b><b>(n, k) @ (k, m) → (n, m)</b>, and the inner sizes must
match. <code>@</code> means dot products of rows with columns; <code>*</code> is element by element.</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "What is the shape of (a) (5, 4) @ (4, 8)? (b) (5, 64) @ (64, 5)? "
                                    "(c) (4, 5) @ (4, 4)?",
             "answer": "(a) (5, 8) · (b) (5, 5) · (c) an error.",
             "why": "Rows from the left, columns from the right. (b) gives one dot product for every pair of the 5 "
                    "rows, like the scores <code>Q @ K.T</code> in episode 6. (c) Rows of 5 numbers can't be dotted with "
                    "columns of 4: the inner sizes don't match.", "key": {'self': True}},
            {"kind": "tf", "q": "“In (5, 4) @ (4, 4), each of the 20 cells of the result is a dot product of 4 pairs "
                                "of numbers.”",
             "answer": "True.", "why": "The result is (5, 4), so 20 cells. Each is one row (4 numbers) dotted with one "
                                       "column (4 numbers).", "key": {'value': True}},
            {"kind": "code", "q": "Predict both results by hand, then run it. Why are they different?",
             "code": """import numpy as np
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
print(A * B)
print(A @ B)""",
             "answer": "[[5 12] [21 32]] · [[19 22] [43 50]]",
             "why": "<code>*</code> multiplies matching elements (1 × 5, 2 × 6, …). <code>@</code> takes dot products: "
                    "row 1 · column 1 = 1 × 5 + 2 × 7 = 19, row 1 · column 2 = 1 × 6 + 2 × 8 = 22, "
                    "row 2 · column 1 = 3 × 5 + 4 × 7 = 43, row 2 · column 2 = 3 × 6 + 4 × 8 = 50."},
        ],
    },
    {
        "title": "Summing along an axis",
        "segment": (50, 61),
        "figures": [{"t": 58.4, "caption": "One total per row: shape (5,)."},
                    {"t": 61.3, "caption": "keepdims=True: a column, (5, 1)."}],
        "body": [
            """<p>Many functions take an <b>axis</b>: the dimension to work along. <code>axis=-1</code> means the
<b>last</b> axis, which in a matrix runs along each row. So <code>Y.sum(axis=-1)</code> gives <b>one total per row</b>:
6, 8, 8, 9, 5, with shape (5,). The summed axis disappears. With <code>keepdims=True</code> it stays, with size 1:
shape <b>(5, 1)</b>, a column.</p>""",
            '<div class="figrow">{fig0}{fig1}</div>',
            """<div class="box key"><b class="t">Key idea</b>Summing along an axis removes it: (5, 4) → (5,).
<code>keepdims=True</code> keeps it with size 1: (5, 4) → (5, 1). Same numbers, different shape.</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "For the Y in the picture, work out <code>Y.sum(axis=0)</code>, which sums along "
                                    "the first axis: down each column. What is its shape?",
             "answer": "[7 10 10 9], shape (4,).", "why": "Column by column: 2 + 1 + 0 + 3 + 1 = 7, 1 + 4 + 2 + 3 + 0 = 10, "
                                                          "0 + 2 + 5 + 1 + 2 = 10, 3 + 1 + 1 + 2 + 2 = 9.", "key": {'parts': [{'label': 'col 1', 'value': 7, 'tol': 0.5, 'unit': None}, {'label': 'col 2', 'value': 10, 'tol': 0.5, 'unit': None}, {'label': 'col 3', 'value': 10, 'tol': 0.5, 'unit': None}, {'label': 'col 4', 'value': 9, 'tol': 0.5, 'unit': None}, {'label': 'shape (length)', 'value': 4, 'tol': 0.5, 'unit': None}]}},
            {"kind": "mc", "q": "Which call gives the row totals of Y as a column, shape (5, 1)?",
             "options": ["<code>Y.sum(axis=0)</code>", "<code>Y.sum(axis=-1)</code>",
                         "<code>Y.sum(axis=-1, keepdims=True)</code>", "<code>Y.sum(axis=0, keepdims=True)</code>"],
             "answer": "C.", "why": "A gives the column totals (4,), B the same row totals as C but flat, (5,), and "
                                   "D the column totals kept as a row, (1, 4).", "key": {'choice': 2}},
            {"kind": "code", "q": "Other functions take an axis too. Predict the three lines.",
             "code": "import numpy as np\n" + Y_CODE + """
print(Y.max(axis=-1))
print(Y.mean(axis=-1, keepdims=True).shape)
print(np.ones((12, 5, 64)).sum(axis=-1).shape)""",
             "answer": "[3 4 5 3 2] · (5, 1) · (12, 5)",
             "why": "The largest number in each row; the row means kept as a column; and for 3-D arrays the last axis "
                    "(64) disappears, leaving (12, 5)."},
        ],
    },
    {
        "title": "Broadcasting",
        "segment": (62, 75),
        "figures": [{"t": 69.6, "caption": "The (5, 1) column is stretched …"},
                    {"t": 75.2, "caption": "… so each row is divided by its total."}],
        "body": [
            """<p>That column lines up with the rows because of <b>broadcasting</b>: when an array has size 1 along an
axis, NumPy stretches it to match the other array, as if it were copied (the faint columns in the picture). Divide a
(5, 5) array by a (5, 1) column, and <b>each row is divided by its own number</b>. That is exactly how <b>softmax</b>
normalizes every row (F07): afterwards, every row adds up to 1.</p>""",
            '<div class="figrow">{fig0}{fig1}</div>',
            """<div class="box key"><b class="t">Key idea</b>A size-1 axis is stretched to match: (5, 5) / (5, 1) →
(5, 5), each row divided by its own total. A single number is stretched to every element.</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "Row 4 of E is [5, 5, 10, 3, 2]. Find its total, divide the row by it, and check "
                                    "that the result adds up to 1.",
             "answer": "Total 25 → [0.2, 0.2, 0.4, 0.12, 0.08].", "why": "5/25 = 0.2, 10/25 = 0.4, 3/25 = 0.12, "
                                                                        "2/25 = 0.08, and 0.2 + 0.2 + 0.4 + 0.12 + 0.08 "
                                                                        "= 1, as in the picture.", "key": {'parts': [{'label': 'total', 'value': 25, 'tol': 0.5, 'unit': None}, {'label': '1st', 'value': 0.2, 'tol': 0.005, 'unit': None}, {'label': '2nd', 'value': 0.2, 'tol': 0.005, 'unit': None}, {'label': '3rd', 'value': 0.4, 'tol': 0.005, 'unit': None}, {'label': '4th', 'value': 0.12, 'tol': 0.005, 'unit': None}, {'label': '5th', 'value': 0.08, 'tol': 0.005, 'unit': None}]}},
            {"kind": "mc", "q": "E has shape (5, 5). Which line raises an error?",
             "options": ["<code>E / E.sum(axis=-1, keepdims=True)</code>", "<code>E + 1</code>",
                         "<code>E * np.ones((1, 5))</code>", "<code>E + np.ones((5, 2))</code>"],
             "answer": "D.", "why": "A size of 2 can't be stretched to 5; only size 1 can. A stretches a column, B a "
                                   "single number, C a row (1, 5).", "key": {'choice': 3}},
            {"kind": "short", "q": "<b>Write it yourself.</b> S is a (5, 5) array of scores. In two lines, using "
                                   "<code>np.exp</code>, a sum with axis and keepdims, and a division, compute P, the "
                                   "softmax of every row.",
             "lines": 2,
             "answer": "<code>e = np.exp(S)</code> and <code>P = e / e.sum(axis=-1, keepdims=True)</code>.",
             "why": "Exponentiate every score, then divide each row by its own total (a (5, 1) column, broadcast). "
                    "<code>P.sum(axis=-1)</code> is then all 1s."},
            {"kind": "code", "q": "<b>Try it yourself.</b> The second division forgets <code>keepdims=True</code>. Predict the three lines. "
                                  "Why does <code>bad</code> run without an error, and what did it divide by?",
             "code": "import numpy as np\n" + E_CODE + """
good = E / E.sum(axis=-1, keepdims=True)
bad  = E / E.sum(axis=-1)
print(E.sum(axis=-1).shape)
print(good.sum(axis=-1))
print(bad[0])""",
             "answer": "(5,) · [1. 1. 1. 1. 1.] · [0.1 0.4 0.2 0.08 0.1]",
             "why": "A (5,) array is lined up with the <i>last</i> axis, like a row (1, 5), so it is stretched down the "
                    "columns: column j is divided by the total of row j. Row 0 becomes 1/10, 2/5, 4/20, 2/25, 1/10, "
                    "which adds up to 0.88, not 1. No error, just a wrong answer: this is why the series always writes "
                    "<code>keepdims=True</code>."},
        ],
    },
    {
        "title": "Reshape and transpose",
        "segment": (76, 85),
        "figures": [{"t": 85.2, "size": "small", "caption": "768 numbers as 12 heads of 64; top left, a transpose."}],
        "body": [
            """<p><code>reshape</code> regroups the <b>same numbers, in the same order</b>, into a new shape:
<code>np.arange(6)</code> is 0 1 2 3 4 5, and <code>.reshape(2, 3)</code> fills 2 rows of 3, [[0 1 2], [3 4 5]]. The
total must not change: 768 numbers can become <b>12 heads of 64</b> because 12 × 64 = 768 (episode 7).
<code>.T</code> <b>transposes</b>: it swaps the axes, so rows become columns and (2, 3) becomes (3, 2).</p>""",
            "{fig0}",
            """<div class="box key"><b class="t">Key idea</b><code>reshape</code> changes the grouping, never the numbers
or their order, so the sizes must multiply to the same total. <code>.T</code> swaps rows and columns: (n, m) →
(m, n).</div>""",
        ],
        "exercises": [
            {"kind": "mc", "q": "Which of these raises an error for <code>np.arange(768)</code>?",
             "options": ["<code>.reshape(12, 64)</code>", "<code>.reshape(8, 96)</code>",
                         "<code>.reshape(10, 76)</code>", "<code>.reshape(3, 256)</code>"],
             "answer": "C.", "why": "10 × 76 = 760, not 768. The others all multiply to 768.", "key": {'choice': 2}},
            {"kind": "code", "q": "Both results have shape (3, 2). Predict them. Are they the same array?",
             "code": """import numpy as np
a = np.arange(6)
print(a.reshape(3, 2))
print(a.reshape(2, 3).T)""",
             "answer": "[[0 1] [2 3] [4 5]] · [[0 3] [1 4] [2 5]]. No.",
             "why": "Reshape fills the new rows in order: 0 1, then 2 3, then 4 5. Transposing [[0 1 2], [3 4 5]] turns "
                    "its columns into rows. Same shape, different arrangement."},
            {"kind": "number", "q": "x has shape (5, 64). What is the shape of <code>x.T</code>? And of "
                                    "<code>x @ x.T</code>?",
             "answer": "(64, 5) and (5, 5).", "why": "(5, 64) @ (64, 5) → (5, 5): a dot product for every pair of rows. "
                                                     "This is how episode 6 computes <code>Q @ K.T</code>.", "key": {'parts': [{'label': 'x.T rows', 'value': 64, 'tol': 0.5, 'unit': None}, {'label': 'x.T columns', 'value': 5, 'tol': 0.5, 'unit': None}, {'label': 'x @ x.T rows', 'value': 5, 'tol': 0.5, 'unit': None}, {'label': 'x @ x.T columns', 'value': 5, 'tol': 0.5, 'unit': None}]}},
            {"kind": "tf", "q": "“Reshaping 768 numbers into (12, 64) changes some of the numbers.”",
             "answer": "False.", "why": "The same 768 numbers stay in the same order; they are only grouped into 12 rows "
                                        "of 64.", "key": {'value': False}},
        ],
    },
    {
        "title": "Masks, and it all together",
        "segment": (86, 106),
        "figures": [{"t": 99.1, "caption": "The causal mask: −∞ above the diagonal."},
                    {"t": 105.6, "caption": "The toolkit in eight lines."}],
        "body": [
            """<p><code>np.triu</code> builds an <b>upper triangle</b>: <code>np.triu(np.ones((5, 5)), k=1)</code> has
1s above the diagonal and 0s elsewhere, and <code>.astype(bool)</code> turns them into True/False: a <b>mask</b>.
<code>scores[mask] = -np.inf</code> picks out exactly the True cells and sets them to −∞. Row i is a token, column j a
token it could look at, and the True cells are the <b>later</b> tokens. That is the <b>causal mask</b> of episode 6:
after softmax, e<sup>−∞</sup>&nbsp;=&nbsp;0, so no token sees the future.</p>""",
            '<div class="figrow">{fig0}{fig1}</div>',
            """<div class="box key"><b class="t">Key idea</b>A True/False <b>mask</b> picks cells:
<code>scores[mask] = -np.inf</code> changes only the True ones. <code>np.triu(…, k=1)</code> marks everything above the
diagonal: the future.</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "How many cells are set to −∞ by the causal mask for 5 tokens? For 10 tokens?",
             "answer": "10 and 45.", "why": "The cells above the diagonal: 4 + 3 + 2 + 1 + 0 = 10 for 5 tokens, and "
                                           "9 + 8 + … + 1 = 45 for 10.", "key": {'parts': [{'label': '5 tokens', 'value': 10, 'tol': 0.5, 'unit': None}, {'label': '10 tokens', 'value': 45, 'tol': 0.5, 'unit': None}]}},
            {"kind": "mc", "q": "In the picture, which tokens can <i>sat</i> (the third row) still look at?",
             "options": ["Only <i>sat</i>", "<i>The</i>, <i>cat</i> and <i>sat</i>", "<i>sat</i>, <i>on</i> and "
                         "<i>the</i>", "All five"],
             "answer": "B.", "why": "Its row has 0 in the first three columns and −∞ for <i>on</i> and <i>the</i>, "
                                   "which come later. A token sees itself and everything before it.", "key": {'choice': 1}},
            {"kind": "tf", "q": "“<code>np.triu(np.ones((5, 5)), k=1)</code> already contains −∞ values.”",
             "answer": "False.", "why": "It holds only 1s and 0s. The −∞ comes from <code>scores[mask] = -np.inf</code>.", "key": {'value': False}},
            {"kind": "code", "q": "<b>Try it yourself.</b> This is the code from the video, with two prints added. "
                                  "(a) What does <code>mask.sum()</code> print? (b) What does <code>scores[1]</code> "
                                  "print? (c) With <code>k=0</code> instead of <code>k=1</code>, how many cells would be "
                                  "masked, and what would go wrong for the first token, <i>The</i>?",
             "code": """import numpy as np
X = np.random.randn(5, 4)               # (5, 4)
W = np.random.randn(4, 4)
Y = X @ W                               # (5, 4)
totals = Y.sum(axis=-1, keepdims=True)  # (5, 1)
heads = np.arange(768).reshape(12, 64)  # (12, 64)
mask = np.triu(np.ones((5, 5)), k=1).astype(bool)
scores = np.zeros((5, 5))
scores[mask] = -np.inf                  # hide future

print(mask.sum())                       # (a)
print(scores[1])                        # (b)""",
             "answer": "(a) 10 · (b) [0. 0. -inf -inf -inf] · (c) 15",
             "why": """(a) True counts as 1, so the sum counts the masked cells. (b) Row 1 is <i>cat</i>: it may look
at <i>The</i> and itself, and the three later tokens are −∞. (c) <code>k=0</code> includes the diagonal: 10 + 5 = 15
cells, so every token is hidden from itself, and <i>The</i> has nothing left to look at: its whole row is −∞."""},
        ],
    },
]
