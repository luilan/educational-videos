"""Study guide content for How LLMs Work · Foundations F03: Matrices: Many Dot Products at Once.

Build:  python framework/study_guide.py foundations f03
Figures: {"t": seconds into the rendered video, "caption": ...}; placed in body blocks with {fig0}, {fig1}, ...
Exercise kinds: mc (options), tf, short, number, order, code. Every exercise has an answer and a why.
All numeric answers and code outputs were checked with NumPy.
"""


def mat(*rows):
    """A small matrix drawn inline: rows of numbers between bracket lines (inline styles; the builder is unchanged)."""
    td = '<td style="padding:0 3pt; border:none; text-align:right; line-height:1.12">'
    cells = "".join("<tr>" + "".join(f"{td}{v}</td>" for v in r) + "</tr>" for r in rows)
    return ('<table style="display:inline-table; width:auto; margin:0 2pt; vertical-align:middle; font-size:9.2pt; '
            'border-collapse:separate; border-left:1.3px solid #1d2433; border-right:1.3px solid #1d2433; '
            f'border-radius:4px">{cells}</table>')


def col(*vals):
    """A column vector."""
    return mat(*[[v] for v in vals])


M = mat([1, 0, 2], [0, 1, "−1"])       # the video's M = [[1, 0, 2], [0, 1, −1]]
STRETCH, ROTATE, SHEAR = mat([2, 0], [0, 1]), mat([0, "−1"], [1, 0]), mat([1, 1], [0, 1])
CENTER = '<div style="text-align:center; margin:4pt 0">'

LESSON = {
    "series": "How LLMs Work · Foundations",
    "label": "F03",
    "title": "Matrices: Many Dot Products at Once",
    "tagline": "The table of numbers behind every layer",
    "duration": "2:02",
    "intro": """<p>This lesson has one big idea: a <b>matrix</b> is a table of numbers that does <b>many dot products
at once</b>. Matrix times vector dots every row with the vector; matrix times matrix dots every row with every column.
That is how one layer of an LLM transforms every token in a single step.</p>""",
    "prereq": """<div class="box note"><b>Before you start:</b> this builds on F01 (Vectors) and F02 (The Dot
Product): every matrix product here is a set of dot products. Matrices power main episodes 5, 6 and 7 (Attention),
8 (The MLP) and 10 (From Vectors Back to Words). Concept 7 reads a few lines of NumPy.</div>""",
}

CONCEPTS = [
    {
        "title": "A matrix is a table of numbers",
        "segment": (8, 25),
        "figures": [{"t": 25.15, "caption": "M has 2 rows and 3 columns: a 2 × 3 matrix. Shapes are always rows × "
                                            "columns."}],
        "body": [
            """<p>Language models compute billions of dot products; writing them one by one would be hopeless.
<b>Matrices</b> let us do many of them at once. A matrix is a <b>table of numbers</b>, with <b>rows</b> and
<b>columns</b>:</p>""",
            f"{CENTER}M = {M}</div>",
            """<p>This M has 2 rows and 3 columns, so we call it a <b>2 × 3 matrix</b> (“two by three”). The shape is
always <b>rows × columns</b>, in that order. Each row is a vector: row 1 of M is [1, 0, 2].</p>""",
            "{fig0}",
            """<div class="box key"><b class="t">Key idea</b>A matrix is a <b>table of numbers</b>. Its shape is
<b>rows × columns</b>: M is 2 × 3.</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": f"A = {mat([2, 7, 1, 0], [5, 3, 8, 4])}. What is its shape, and which number is in "
                                    "row 2, column 3?",
             "answer": "2 × 4; the number 8.", "why": "2 rows, 4 columns. Row 2 is [5, 3, 8, 4], and its third entry "
                                                      "is 8."},
            {"kind": "tf", "q": "“A 2 × 3 matrix and a 3 × 2 matrix have the same shape, since both hold 6 numbers.”",
             "answer": "False.", "why": "Shape is rows × columns, and the order matters: 2 rows of 3 is not 3 rows "
                                        "of 2."},
            {"kind": "short", "q": "Write down row 2 and column 3 of the matrix M above.",
             "answer": "Row 2 = [0, 1, −1]; column 3 = [2, −1].",
             "why": "A row goes across (3 numbers in a 2 × 3 matrix); a column goes down (2 numbers)."},
            {"kind": "number", "q": "GPT-2's vectors have 768 numbers. How many numbers does a 768 × 768 matrix "
                                    "hold?",
             "answer": "589,824.", "why": "768 rows × 768 columns = 589,824. Weight matrices get big fast."},
        ],
    },
    {
        "title": "Matrix times vector",
        "segment": (25, 36),
        "figures": [{"t": 35.7, "size": "small", "caption": "Each row of M dotted with x gives one number: 7 from row 1, −1 from "
                                           "row 2."}],
        "body": [
            """<p>To multiply a matrix by a vector, take the <b>dot product of each row with the vector</b>. Each row
gives one number, so the result is a new vector, with <b>one entry per row</b>.</p>""",
            f"{CENTER}M x = {M} {col(3, 1, 2)} = {col(7, '−1')}</div>",
            """<p>Row 1: 1·3 + 0·1 + 2·2 = 7. Row 2: 0·3 + 1·1 + (−1)·2 = −1. So M x = [7, −1].</p>""",
            "{fig0}",
            """<div class="box key"><b class="t">Key idea</b>M x = <b>one dot product per row</b> of M. The result
has one entry per row.</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": f"Compute {mat([2, 1], [0, 3])} {col(4, 5)}.",
             "answer": "[13, 15].", "why": "Row 1: 2·4 + 1·5 = 13. Row 2: 0·4 + 3·5 = 15."},
            {"kind": "number", "q": f"Compute M y for the video's M and y = [1, 1, 1].",
             "answer": "[3, 0].", "why": "Dotting with [1, 1, 1] just adds up each row: 1 + 0 + 2 = 3 and "
                                         "0 + 1 − 1 = 0."},
            {"kind": "mc", "q": f"What does {mat([0, 1], [1, 0])} do to any vector [p, q]?",
             "options": ["Leaves it unchanged: [p, q]", "Swaps the two numbers: [q, p]", "Doubles it: [2p, 2q]",
                         "Flips it: [−p, −q]"],
             "answer": "B.", "why": "Row 1 [0, 1] picks out q; row 2 [1, 0] picks out p."},
            {"kind": "tf", "q": "“Each entry of M x uses all the numbers of x, but only one row of M.”",
             "answer": "True.", "why": "Entry i is row i of M dotted with the whole of x."},
        ],
    },
    {
        "title": "A matrix transforms space",
        "segment": (36, 47),
        "figures": [{"t": 43.4, "caption": "Stretch: every x-coordinate doubles; y stays the same."},
                    {"t": 47.0, "caption": "Shear: space slides sideways, and every arrow moves along."}],
        "body": [
            """<p>So a matrix <b>turns one vector into another</b>. In two dimensions you can watch it happen: apply
the matrix to every point and the whole plane moves. A matrix can <b>stretch</b> space, <b>rotate</b> it, or
<b>shear</b> it, and every arrow moves along. The video's three matrices, applied to the yellow arrow [1, 2]:</p>""",
            f"""{CENTER}stretch {STRETCH} → [2, 2] &nbsp;&nbsp;·&nbsp;&nbsp; rotate 90° {ROTATE} → [−2, 1]
&nbsp;&nbsp;·&nbsp;&nbsp; shear {SHEAR} → [3, 2]</div>""",
            '<div class="figrow">{fig0}{fig1}</div>',
            """<div class="box key"><b class="t">Key idea</b>A matrix is a <b>function on vectors</b>: vector in,
vector out. It moves every arrow in the same systematic way, such as a stretch, a rotation or a shear.</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": f"Apply the rotate matrix {ROTATE} to [3, 1].",
             "answer": "[−1, 3].", "why": "Row 1: 0·3 + (−1)·1 = −1. Row 2: 1·3 + 0·1 = 3. As F02 showed, [−1, 3] is "
                                          "at right angles to [3, 1]."},
            {"kind": "number", "q": f"Apply the stretch matrix {STRETCH} to [−1, 3].",
             "answer": "[−2, 3].", "why": "The first number doubles, the second stays the same."},
            {"kind": "number", "q": "Write down a 2 × 2 matrix that makes every arrow 3 times as long, without "
                                    "changing its direction.",
             "lines": 2,
             "answer": "Rows [3, 0] and [0, 3].", "why": "It sends [p, q] to [3p, 3q] = 3 × [p, q]. The video's stretch "
                                                        "matrix changes only the first number, so it stretches "
                                                        "sideways only."},
            {"kind": "tf", "q": "“Applying the rotate matrix four times brings every arrow back to where it "
                                "started.”",
             "answer": "True.", "why": "4 × 90° = 360°, a full turn."},
        ],
    },
    {
        "title": "The shapes must line up",
        "segment": (47, 58),
        "figures": [{"t": 57.9, "size": "small", "caption": "A 2 × 3 matrix needs 3 numbers and gives back 2. In general, "
                                           "(m × n) · (n) → (m)."}],
        "body": [
            """<p>The <b>shapes must line up</b>. Each row is dotted with the vector, so the vector needs exactly as
many numbers as the matrix has <b>columns</b>. A 2 × 3 matrix needs a vector with 3 numbers, and gives back 2, one per
row.</p>""",
            """<p>In general, an <b>m × n</b> matrix turns <b>n</b> numbers into <b>m</b> numbers:
(m × n) · (n) → (m).</p>""",
            "{fig0}",
            """<div class="box key"><b class="t">Key idea</b><b>Columns</b> = how many numbers go in;
<b>rows</b> = how many come out.</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "A 4 × 6 matrix multiplies a vector. How many numbers must the vector have, and "
                                    "how many come out?",
             "answer": "6 in, 4 out.", "why": "Columns (6) = input size; rows (4) = output size."},
            {"kind": "mc", "q": "You want a matrix that turns a 768-number vector into a 3,072-number vector. What "
                                "shape must it have?",
             "options": ["768 × 3,072", "3,072 × 768", "768 × 768", "3,072 × 3,072"],
             "answer": "B.", "why": "m × n turns n numbers into m: 3,072 rows (one per output), 768 columns (one per "
                                   "input)."},
            {"kind": "tf", "q": "“A 3 × 2 matrix can multiply the vector [3, 1, 2].”",
             "answer": "False.", "why": "A 3 × 2 matrix has 2 columns, so it needs 2 numbers. The video's 2 × 3 M is "
                                        "the one that takes [3, 1, 2]."},
            {"kind": "number", "q": "A vector of 5 numbers is multiplied by A (shape 8 × 5), and the result by B "
                                    "(shape 2 × 8). How many numbers come out at the end?",
             "answer": "2.", "why": "A turns 5 numbers into 8, then B turns 8 into 2. The shapes line up at each "
                                   "step."},
        ],
    },
    {
        "title": "Matrix times matrix: every token at once",
        "segment": (58, 84),
        "figures": [{"t": 66.7, "caption": "Row 1 · column 1 = 6 (illustrative)."},
                    {"t": 83.7, "caption": "All five tokens in one product."}],
        "body": [
            """<p>Multiplying two matrices is the same idea, repeated: <b>every row of the first, dotted with every
column of the second</b>. The inner sizes must match and the outer sizes give the result: (5 × 4) · (4 × 4) →
<b>5 × 4</b>. In general, (m × n) · (n × p) → (m × p).</p>""",
            """<p>That's exactly how LLMs use it: stack the tokens' vectors as the <b>rows</b> of X, multiply by one
<b>weight matrix</b> W, and every token is transformed in a single step. Row i of X · W belongs to token i.</p>""",
            '<div class="figrow">{fig0}{fig1}</div>',
            """<div class="box key"><b class="t">Key idea</b>A · B = <b>every row of A dotted with every column of
B</b>. With tokens as rows, X · W transforms them all at once.</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": f"Compute {mat([1, 2], [3, 4])} · {mat([5, 6], [7, 8])}.",
             "answer": "Rows [19, 22] and [43, 50].",
             "why": "Top-left: row 1 · column 1 = 1·5 + 2·7 = 19. Likewise 1·6 + 2·8 = 22, 3·5 + 4·7 = 43, "
                    "3·6 + 4·8 = 50."},
            {"kind": "number", "q": "In the video, row 2 of X is [2, −1, 0, 2] and column 1 of W is [1, −2, −1, 2]. "
                                    "Which entry of X · W do they give, and what is it?",
             "answer": "Row 2, column 1: 8.", "why": "2·1 + (−1)·(−2) + 0·(−1) + 2·2 = 2 + 2 + 0 + 4 = 8, as in the "
                                                     "finished result in the video."},
            {"kind": "mc", "q": "Which product is <b>not</b> allowed?",
             "options": ["(5 × 4) · (4 × 4)", "(2 × 3) · (3 × 1)", "(4 × 5) · (4 × 5)", "(1 × 768) · (768 × 768)"],
             "answer": "C.", "why": "The inner sizes are 5 and 4, which don't match. The others give 5 × 4, 2 × 1 "
                                   "and 1 × 768."},
            {"kind": "number", "q": "10 tokens of 768 numbers form X; W is 768 × 768. (a) Shape of X · W? (b) How "
                                    "many dot products? (c) Which row holds the transformed 4th token?",
             "answer": "(a) 10 × 768 (b) 7,680 (c) row 4.",
             "why": "(10 × 768) · (768 × 768) → 10 × 768; one dot product per entry, 10 × 768 = 7,680; row i of "
                    "the result belongs to token i."},
        ],
    },
    {
        "title": "The transpose",
        "segment": (84, 96),
        "figures": [{"t": 90.6, "caption": "M flipped over its diagonal: Mᵀ is 3 × 2."},
                    {"t": 96.3, "caption": "Q · Kᵀ: every query against every key."}],
        "body": [
            """<p>One more operation: the <b>transpose</b>, written Mᵀ. It flips a matrix over its <b>diagonal</b>, so
<b>rows become columns</b>: row 1 of M, [1, 0, 2], becomes column 1 of Mᵀ, and the 2 × 3 M becomes 3 × 2.</p>""",
            """<p>Why it matters: in attention, Q holds a <i>query</i> vector per token and K a <i>key</i> vector per
token, both as rows. To dot every query with every key, the keys must become columns: <b>Q · Kᵀ</b>. With 5 tokens of
4 numbers each, (5 × 4) · (4 × 5) → 5 × 5: one score for every pair of tokens.</p>""",
            '<div class="figrow">{fig0}{fig1}</div>',
            """<div class="box key"><b class="t">Key idea</b>The transpose <b>swaps rows and columns</b>
(m × n → n × m). <b>Q · Kᵀ</b> compares every query with every key in one product.</div>""",
        ],
        "exercises": [
            {"kind": "short", "q": f"Transpose {mat([1, 2, 3], [4, 5, 6])}. What shape is the result?",
             "answer": "Rows [1, 4], [2, 5], [3, 6]; shape 3 × 2.",
             "why": "Row 1 [1, 2, 3] becomes column 1, row 2 [4, 5, 6] becomes column 2."},
            {"kind": "tf", "q": "“Transposing a matrix twice gives back the original matrix.”",
             "answer": "True.", "why": "Flipping over the diagonal twice puts every number back where it was."},
            {"kind": "mc", "q": "Q and K both have shape 5 × 4 (5 tokens, 4 numbers each). Why can't we just "
                                "compute Q · K?",
             "options": ["The inner sizes don't match (4 and 5)", "Only square matrices can be multiplied",
                         "Q · K gives the same result as Q · Kᵀ anyway",
                         "Two matrices of the same shape can never be multiplied"],
             "answer": "A.", "why": "Kᵀ is 4 × 5, so (5 × 4) · (4 × 5) lines up and gives 5 × 5. B and D are false: "
                                   "only the inner sizes matter."},
            {"kind": "number", "q": "A prompt has 12 tokens, with query and key vectors of 64 numbers. What shape is "
                                    "Q · Kᵀ, and how many scores does it hold?",
             "answer": "12 × 12; 144 scores.", "why": "(12 × 64) · (64 × 12) → 12 × 12: one score for every "
                                                      "(query, key) pair."},
        ],
    },
    {
        "title": "Matrices in NumPy",
        "segment": (96, 119),
        "figures": [{"t": 103.9, "size": "small", "caption": "The code from the video: @ multiplies matrices too, "
                                                             ".T transposes, and .shape lets you check shapes."}],
        "body": [
            """<p>In NumPy, the <b>@</b> sign multiplies matrices too (matrix × vector and matrix × matrix),
<code>.T</code> gives the transpose, and <code>.shape</code> tells you the shape. Always check the shapes.</p>""",
            "{fig0}",
            """<pre class="code">M = np.array([[1, 0, 2],
              [0, 1, -1]])          # 2 x 3
x = np.array([3, 1, 2])
M @ x                               # array([ 7, -1])
X = np.random.randn(5, 4)           # 5 tokens, 4 numbers each
W = np.random.randn(4, 4)
(X @ W).shape                       # (5, 4)
M.T.shape                           # (3, 2)</pre>""",
            """<p>Matrices power almost every step from episode 5 to episode 10: queries, keys and values, attention
scores, the MLP, and the final logits.</p>""",
            """<div class="box key"><b class="t">Key idea</b><code>@</code> multiplies, <code>.T</code> transposes,
<code>.shape</code> checks: (m × n) @ (n × p) → (m × p).</div>""",
        ],
        "exercises": [
            {"kind": "short", "q": "With M from the code above, what is <code>(M.T @ M).shape</code>?",
             "answer": "(3, 3).", "why": "M.T is 3 × 2 and M is 2 × 3: (3 × 2) @ (2 × 3) → 3 × 3."},
            {"kind": "mc", "q": "What happens when you run <code>M @ M</code>?",
             "options": ["You get an array of shape (2, 3)", "A ValueError: the shapes don't line up",
                         "Every number in M is squared", "You get an array of shape (2, 2)"],
             "answer": "B.", "why": "(2 × 3) @ (2 × 3): the inner sizes 3 and 2 don't match. Squaring every number "
                                   "(C) would be <code>M * M</code>."},
            {"kind": "code", "q": "<b>Try it yourself.</b> (a) Complete <code>matvec</code> so it gives the same "
                                  "result as <code>M @ x</code>, using one dot product per row. (b) X holds 4 "
                                  "tokens as rows. Why does <code>X @ M</code> fail, while <code>X @ M.T</code> "
                                  "works? What shape does <code>X @ M.T</code> have? (c) Compare the first row of "
                                  "<code>X @ M.T</code> with <code>matvec(M, X[0])</code>. What do you notice?",
             "code": """import numpy as np

M = np.array([[1, 0, 2],
              [0, 1, -1]])
x = np.array([3, 1, 2])

def matvec(M, x):
    out = []
    for row in M:
        out.append(...)          # (a) your code here
    return np.array(out)

print(matvec(M, x), M @ x)

X = np.array([[1, 2, 0],         # 4 tokens as rows,
              [0, 1, 1],         # 3 numbers each
              [2, 0, 1],
              [1, 1, 1]])
print(X @ M.T)
print(matvec(M, X[0]))""",
             "answer": "(a) [ 7 -1] [ 7 -1] · (b) shape (4, 2) · (c) both are [1 2]",
             "why": """(a) <code>out.append(row @ x)</code>. (b) <code>X @ M</code> is (4 × 3) @ (2 × 3): the inner
sizes 3 and 2 don't match, so NumPy raises a ValueError. <code>M.T</code> is 3 × 2, so (4 × 3) @ (3 × 2) → (4, 2):
two numbers per token. (c) Both are [1, 2]. Each row of <code>X @ M.T</code> is M applied to that token, so one
product applies M to all four tokens at once."""},
        ],
    },
]
