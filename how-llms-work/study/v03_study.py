"""Study guide content for How LLMs Work, episode 3: Embeddings: Words as Vectors.

Build:  python framework/study_guide.py how-llms-work v03
Figures: {"t": seconds into the rendered video, "caption": ...}; placed in body blocks with {fig0}, {fig1}, ...
Exercise kinds: mc (options), tf, short, number, order, code. Every exercise has an answer and a why.
"""

LESSON = {
    "series": "How LLMs Work",
    "label": "Episode 3",
    "title": "Embeddings: Words as Vectors",
    "tagline": "How numbers start to carry meaning",
    "duration": "2:28",
    "intro": """<p>This lesson has one big idea: every token gets a <b>vector</b>, a list of numbers that places it as a
<b>point in space</b>, and after training, <b>where a token sits carries meaning</b>. Similar tokens end up close
together, closeness can be measured with the dot product, and even directions can mean something.</p>""",
    "prereq": """<div class="box note"><b>Before you start:</b> this lesson builds on episode 2 (tokens and token IDs).
The maths is vectors and the dot product: Foundations F01 (Vectors) and F02 (The Dot Product) cover it. Concept 7
reads a few lines of NumPy; F11 (NumPy in Three Minutes) helps.</div>""",
}

CONCEPTS = [
    {
        "title": "IDs are labels, vectors are points",
        "segment": (8, 30),
        "figures": [{"t": 21.9, "caption": "Token 3797 is ␣cat, and its neighbours are rief and esc: the ID numbers "
                                           "mean nothing."},
                    {"t": 30.0, "caption": "Give each token a vector, like coordinates, and it becomes a point in "
                                           "space."}],
        "body": [
            """<p>After tokenizing, <i>“The cat sat on the”</i> is [464, 3797, 3332, 319, 262]. But an ID is <b>just a
label</b>. Token 3797 is <i>␣cat</i>, while its neighbours 3796 and 3798 are <i>rief</i> and <i>esc</i>. The numbers
say nothing about meaning.</p>""",
            """<p>The fix: give every token a <b>vector</b>, a list of numbers, like coordinates. Then each token
becomes a <b>point in space</b>, and where it sits can carry meaning.</p>""",
            '<div class="figrow">{fig0}{fig1}</div>',
            """<div class="box key"><b class="t">Key idea</b>An ID only says <b>which</b> token it is. A token's
<b>embedding</b> is its vector: a point in space whose position can say something about <b>what the token
means</b>.</div>""",
        ],
        "exercises": [
            {"kind": "tf", "q": "“IDs 3796 and 3797 are neighbours, so <i>rief</i> and <i>␣cat</i> must have similar "
                                "meanings.”",
             "answer": "False.", "why": "IDs are just labels; neighbouring numbers are unrelated tokens."},
            {"kind": "mc", "q": "A token's embedding is…",
             "options": ["its ID written in binary", "a list of numbers (a vector) that places it as a point in space",
                         "how often it appears in the training text", "the list of its letters"],
             "answer": "B.", "why": "The vector works like coordinates, so every token becomes a point."},
            {"kind": "mc", "q": "Why not feed the raw ID numbers into the network as they are?",
             "options": ["They are too big to store", "The network would treat 3797 and 3798 as almost equal, though "
                         "<i>␣cat</i> and <i>esc</i> are unrelated", "IDs change every time you tokenize",
                         "There are not enough IDs for every token"],
             "answer": "B.", "why": "The size and closeness of IDs mean nothing, so arithmetic on them would be "
                                   "misleading."},
        ],
    },
    {
        "title": "The embedding matrix: one row per token",
        "segment": (30, 58),
        "figures": [{"t": 44.9, "caption": "Embedding is a lookup: token 3797 grabs row 3797 of the embedding matrix "
                                           "(illustrative values)."},
                    {"t": 55.4, "caption": "Real vectors are long: 768 numbers in GPT-2 small, many thousands in the "
                                           "largest models."}],
        "body": [
            """<p>All the vectors are stored in one big table, the <b>embedding matrix</b> <code>E</code>, with <b>one
row for every token</b> in the vocabulary: 50,257 rows for GPT-2. Embedding a token is just a <b>lookup</b>: token
3797 grabs row 3797.</p>""",
            '<div class="figrow">{fig0}{fig1}</div>',
            """<p>Real vectors are long. The smallest GPT-2 uses <b>768 numbers per token</b>, and the largest models
use many thousands. Pictures show just two or three, so we can see them.</p>""",
            """<div class="box key"><b class="t">Key idea</b>Embedding = <b>row lookup</b>: token ID <i>i</i> → row
<i>i</i> of <code>E</code>. <code>E</code> has one row per token in the vocabulary and one column per number in the
vector (768 for GPT-2 small).</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "GPT-2 small's embedding matrix has 50,257 rows and 768 columns. How many numbers "
                                    "does it hold?",
             "answer": "38,597,376 (about 38.6 million).", "why": "50,257 × 768: one number per token per dimension."},
            {"kind": "mc", "q": "To embed <i>␣on</i> (ID 319), the model…",
             "options": ["multiplies 319 by every row of <code>E</code>", "takes row 319 of <code>E</code>",
                         "searches <code>E</code> for the row closest to 319", "averages all the rows of <code>E</code>"],
             "answer": "B.", "why": "Embedding is just a lookup: ID 319 grabs row 319."},
            {"kind": "short", "q": "A toy model has a vocabulary of 1,000 tokens and uses 64 numbers per token. What "
                                   "is the shape of its embedding matrix (rows × columns)?",
             "answer": "1,000 × 64.", "why": "One row per token, one column per number in each vector."},
            {"kind": "tf", "q": "“The same token, used in two different sentences, gets two different embedding "
                                "vectors.”",
             "answer": "False.", "why": "It's a lookup: the same ID always grabs the same row. This is the catch at the "
                                        "end of the video, and the topic of episode 4."},
        ],
    },
    {
        "title": "Similar tokens end up close together",
        "segment": (58, 72),
        "figures": [{"t": 71.7, "caption": "After training: animals, floor coverings, numbers and verbs each gather in "
                                           "their own region (illustrative 2-D picture)."}],
        "body": [
            """<p>Here's the key idea. After training, tokens that are <b>used in similar ways</b> end up <b>close
together</b>. <i>cat</i> lands near <i>dog</i> and <i>kitten</i>; <i>mat</i> near <i>rug</i> and <i>carpet</i>.
Numbers gather in one place, verbs in another.</p>""",
            "{fig0}",
            """<div class="box key"><b class="t">Key idea</b>In embedding space, <b>closeness means similarity</b>:
tokens used in similar ways get nearby points. Position reflects how a token is used, not how it is spelled.</div>""",
        ],
        "exercises": [
            {"kind": "mc", "q": "The word <i>puppy</i> is added and the model is trained. In the figure's picture, "
                                "which group would you expect it to join?",
             "options": ["cat, dog, kitten", "mat, rug, carpet", "one, two, three", "sat, ran, jumped"],
             "answer": "A.", "why": "<i>puppy</i> is used much like <i>dog</i> and <i>kitten</i>."},
            {"kind": "short", "q": "<i>rug</i> and <i>carpet</i> share no letters. Why do they still end up close "
                                   "together?",
             "answer": "They are used in similar ways.",
             "why": "Position reflects usage, not spelling: <i>on the rug</i> and <i>on the carpet</i> appear in "
                    "similar sentences."},
            {"kind": "tf", "q": "“After training, <i>sat</i> is closer to <i>ran</i> than to <i>rug</i>.”",
             "answer": "True.", "why": "<i>sat</i> and <i>ran</i> are both verbs and are used in similar ways; verbs "
                                       "gather in one region."},
        ],
    },
    {
        "title": "Measuring closeness: dot product and cosine",
        "segment": (72, 91),
        "figures": [{"t": 90.2, "size": "small", "caption": "a · b = 3·2 + 1·2 = 8, and cos θ ≈ 0.89."}],
        "body": [
            """<p>To measure “close”, use the <b>dot product</b>: multiply matching coordinates, then add them up.
For <i>a</i> = (3, 1) and <i>b</i> = (2, 2): <i>a · b</i> = 3·2 + 1·2 = <b>8</b>. Vectors pointing the <b>same
way</b> score high, unrelated (perpendicular) ones score <b>near zero</b>, and opposite ones go <b>negative</b>.</p>""",
            "{fig0}",
            """<p>The dot product also grows with length. <b>Divide by both lengths</b> (‖<i>a</i>‖ = √(3² + 1²) ≈
3.16, ‖<i>b</i>‖ ≈ 2.83) to get the <b>cosine similarity</b>, the cosine of the angle θ: 8 / (3.16 · 2.83) ≈
<b>0.89</b>. It runs from 1 (same direction) to −1 (opposite).</p>""",
            """<div class="box key"><b class="t">Key idea</b><b>a · b = a₁b₁ + a₂b₂ + …</b> High = same direction, ≈ 0 =
unrelated, negative = opposite. <b>cos θ = a · b / (‖a‖ ‖b‖)</b> ignores length and measures direction only.</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "Compute the dot product of (2, −1, 3) and (1, 4, 2).",
             "answer": "4.", "why": "2·1 + (−1)·4 + 3·2 = 2 − 4 + 6 = 4. Same rule in any number of dimensions."},
            {"kind": "mc", "q": "Which pair has a <b>negative</b> dot product?",
             "options": ["(1, 0) and (0, 1)", "(2, 1) and (1, 2)", "(3, 1) and (−3, −1)", "(1, 1) and (5, 5)"],
             "answer": "C.", "why": "They point in opposite directions: −9 − 1 = −10. A is 0 (perpendicular), B is 4, "
                                   "D is 10."},
            {"kind": "number", "q": "Compute the cosine similarity of (3, 4) and (6, 8). What does the result tell "
                                    "you?",
             "answer": "1: same direction.", "why": "50 / (5 · 10) = 1. (6, 8) is just (3, 4) stretched; cosine "
                                                    "ignores length."},
            {"kind": "tf", "q": "“<i>a</i> = (3, 1) scores 10 with (0, 10) but only 8 with <i>b</i> = (2, 2), so "
                                "(0, 10) points more nearly the same way as <i>a</i>.”",
             "answer": "False.", "why": "(0, 10) is just long. Its cosine with <i>a</i> is 10 / (3.16 · 10) ≈ 0.32, far "
                                        "below 0.89. That's why we divide by the lengths."},
        ],
    },
    {
        "title": "Directions carry meaning",
        "segment": (91, 105),
        "figures": [{"t": 104.7, "caption": "The step from man to woman, added to king, lands close to queen "
                                            "(illustrative 2-D picture)."}],
        "body": [
            """<p>Directions can carry meaning too. In classic word embeddings like <b>word2vec</b>, the step from
<i>man</i> to <i>woman</i> is roughly the same as the step from <i>king</i> to <i>queen</i>. So vector arithmetic
works: <i>king − man + woman</i> lands <b>close to</b> <i>queen</i>.</p>""",
            "{fig0}",
            """<div class="box key"><b class="t">Key idea</b>A <b>direction</b>, the step from one vector to another, can
stand for a relationship. Adding the <i>man</i> → <i>woman</i> step to <i>king</i> moves it to about
<i>queen</i>.</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "Toy 2-D embeddings: man = (1, 1), woman = (1, 3), king = (5, 1). Compute "
                                    "king − man + woman.",
             "answer": "(5, 3).", "why": "The step man → woman is (0, 2); adding it to king gives (5, 3), where queen "
                                         "would be."},
            {"kind": "short", "q": "If the step from <i>France</i> to <i>Paris</i> means “capital of”, which word would "
                                   "you expect near <i>Paris − France + Italy</i>?",
             "answer": "Rome.", "why": "Paris − France is the “capital of” step; adding it to Italy lands near Italy's "
                                       "capital."},
            {"kind": "tf", "q": "“king − man + woman gives exactly the vector of queen.”",
             "answer": "False.", "why": "It lands close to queen, not exactly on it: the two steps are only roughly "
                                        "the same."},
        ],
    },
    {
        "title": "Learned, not designed",
        "segment": (105, 116),
        "figures": [{"t": 108.8, "caption": "Before training: the vectors are random, so the words are scattered."},
                    {"t": 115.6, "caption": "After training: similar words have been nudged together."}],
        "body": [
            """<p>Nobody writes these numbers by hand. The embedding vectors <b>start out random</b>. Then
<b>training nudges them</b>, a tiny bit at a time, until the geometry reflects how words are actually used. The
clusters of concept 3 are a result of training, not a design.</p>""",
            '<div class="figrow">{fig0}{fig1}</div>',
            """<div class="box key"><b class="t">Key idea</b>The embedding matrix is part of the model's
<b>parameters</b>: learned from data like the rest of its numbers (episode 1), starting from random values.</div>""",
        ],
        "exercises": [
            {"kind": "tf", "q": "“Engineers place <i>cat</i> near <i>dog</i> by hand, using a dictionary.”",
             "answer": "False.", "why": "Nobody writes the numbers. Training moves the vectors until similar words "
                                        "are close."},
            {"kind": "mc", "q": "Before training, where are <i>cat</i> and <i>dog</i> in the embedding space?",
             "options": ["Already next to each other", "At random places, unrelated to their meaning",
                         "Both exactly at the origin", "At positions given by their IDs"],
             "answer": "B.", "why": "The vectors start random; meaning only shows up after training."},
            {"kind": "short", "q": "The same model is trained twice, starting from different random numbers. Would "
                                   "<i>cat</i> get exactly the same vector both times? Would it still end up near "
                                   "<i>dog</i>?",
             "lines": 2,
             "answer": "No; yes.", "why": "The exact numbers depend on the random start, but in both runs training "
                                         "pulls words that are used alike close together."},
        ],
    },
    {
        "title": "Embeddings in code",
        "segment": (116, 145),
        "figures": [{"t": 128.8, "caption": "The whole embedding step: one line, x = E[ids]."},
                    {"t": 143.7, "caption": "The catch: same tokens, same vectors. Nothing says which came first."}],
        "body": [
            """<p>In code, embedding is a single line. <code>E</code> has one row per token; indexing it with the list
of IDs gives one vector per token, a 5 × 768 array for our 5-token sentence.</p>""",
            """<pre class="code">import numpy as np

vocab_size, d_model = 50257, 768
E = np.random.randn(vocab_size, d_model) * 0.02   # learned in training

ids = [464, 3797, 3332, 319, 262]   # "The cat sat on the"
x = E[ids]                          # one row per token
print(x.shape)                      # (5, 768)</pre>""",
            '<div class="figrow">{fig0}{fig1}</div>',
            """<p>But there's a catch: <i>“The cat sat on the mat”</i> and <i>“The mat sat on the cat”</i> contain
exactly the same tokens, so they get exactly the same vectors. Nothing says which came first. Next: position.</p>""",
            """<div class="box key"><b class="t">Key idea</b><code>x = E[ids]</code>: one row of <code>E</code> per
token, shape (number of tokens, <code>d_model</code>). A token gets the same vector <b>wherever it is</b> in the
sentence.</div>""",
        ],
        "exercises": [
            {"kind": "short", "q": "What does <code>print(x.shape)</code> show for <i>“The cat sat on the mat”</i> "
                                   "(6 tokens)? And for the original 5 tokens if <code>d_model</code> were 4,096?",
             "answer": "(6, 768) and (5, 4096).", "why": "Rows = number of tokens; columns = <code>d_model</code>."},
            {"kind": "mc", "q": "Why does the code fill <code>E</code> with small random numbers?",
             "options": ["It's a bug; real models use zeros", "It's only the starting point: training then adjusts "
                         "the numbers", "Random vectors work as well as trained ones", "To hide the token IDs"],
             "answer": "B.", "why": "Embeddings start random and are learned (concept 6), hence the comment "
                                   "<code># learned in training</code>."},
            {"kind": "code", "q": "<b>Try it yourself.</b> The code below builds a toy <code>E</code> with 4 numbers "
                                  "per token and embeds the two sentences from the video. (a) Write "
                                  "<code>cosine(u, v)</code> (hint: <code>np.dot</code>, <code>np.linalg.norm</code>) "
                                  "and check it on (3, 1) and (2, 2). (b) Is <code>x1[1]</code> equal to "
                                  "<code>E[3797]</code>? What does <code>np.array_equal(x1, x2)</code> print, and why? "
                                  "(c) What does <code>np.allclose(x1.sum(axis=0), x2.sum(axis=0))</code> print? What "
                                  "does that mean for a model that only looks at the collection of vectors?",
             "code": """import numpy as np

def cosine(u, v):
    ...   # (a) your code here

rng = np.random.default_rng(0)
E = rng.normal(size=(50257, 4))            # toy: 4 numbers per token
ids1 = [464, 3797, 3332, 319, 262, 2603]   # "The cat sat on the mat"
ids2 = [464, 2603, 3332, 319, 262, 3797]   # "The mat sat on the cat"
x1, x2 = E[ids1], E[ids2]""",
             "answer": "(a) ≈ 0.894 · (b) True; False · (c) True",
             "why": """(a) <code>def cosine(u, v): return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))</code>
gives 8 / (3.16 · 2.83) ≈ 0.894, as in the video. (b) Row 1 of <code>x1</code> is the lookup of ID 3797 (␣cat), so
yes. The arrays are not equal because the rows come in a different order. (c) The sum of the vectors is identical: the
two sentences contain exactly the same six vectors. Anything that ignores order can't tell them apart, which is why
episode 4 adds position."""},
        ],
    },
]
