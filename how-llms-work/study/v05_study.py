"""Study guide content for How LLMs Work, episode 5: Attention I: Tokens Talking to Each Other.

Build:  python framework/study_guide.py how-llms-work v05
Figures: {"t": seconds into the rendered video, "caption": ...}; placed in body blocks with {fig0}, {fig1}, ...
Exercise kinds: mc (options), tf, short, number, order, code. Every exercise has an answer and a why.
All numbers are the video's own (v05_scene.py) and were checked with NumPy.
"""

LESSON = {
    "series": "How LLMs Work",
    "label": "Episode 5",
    "title": "Attention I: Tokens Talking to Each Other",
    "tagline": "How words borrow meaning from their neighbours",
    "duration": "2:22",
    "intro": """<p>This lesson has one big idea: <b>attention lets every token look at the other tokens and borrow
meaning from the ones that matter</b>. Each token asks a question (its query), compares it with what every token
offers (its key), and mixes in the information (the values) of the best matches.</p>""",
    "prereq": """<div class="box note"><b>Before you start:</b> this lesson builds on episodes 3 and 4: every token
enters attention as one vector that says what it is and where it is. Concept 4 uses the dot product (Foundations F02)
and concept 7 multiplies a vector by a matrix (F03); F07 (Softmax, Properly) prepares episode 6, where scores become
weights. The code exercise in concept 5 uses NumPy (F11).</div>""",
}

CONCEPTS = [
    {
        "title": "Context changes meaning",
        "segment": (8, 43),
        "figures": [{"t": 30.2, "caption": "Same token, same vector, two meanings."},
                    {"t": 42.6, "caption": "bank notices river and updates its vector."}],
        "body": [
            """<p>Each token's vector says <b>what it is</b> and <b>where it is</b>, but it was made <b>on its own</b>,
without looking at any other word. Take <i>bank</i>: in <i>“I sat on the river bank”</i> it is the edge of a river; in
<i>“I paid money into the bank”</i> it is a place that keeps money. Same token, same starting vector, two meanings.</p>""",
            '<div class="figrow">{fig0}{fig1}</div>',
            """<div class="box key"><b class="t">Key idea</b><b>Attention</b> lets every token <b>look at the other
tokens</b> and pull in information from the ones that matter. <i>bank</i> notices <i>river</i> and updates its own
vector to mean “riverbank”: context changes meaning.</div>""",
        ],
        "exercises": [
            {"kind": "short", "q": "In both sentences, <i>bank</i> is the 6th token. Before attention, are its two "
                                   "vectors the same or different? Why?",
             "answer": "The same.", "why": "The vector only encodes what the token is (<i>bank</i>) and where it is "
                                          "(position 6), and both are identical. <i>river</i> and <i>money</i> have "
                                          "not been looked at yet."},
            {"kind": "mc", "q": "What does attention let a token do?",
             "options": ["Swap itself for a different, less ambiguous token",
                         "Look at the other tokens and pull in information from the ones that matter",
                         "Look the word up in a built-in dictionary of meanings",
                         "Delete the words that don't matter from the text"],
             "answer": "B.", "why": "The token stays the same token (<i>bank</i> never becomes a new token); only its "
                                   "vector is updated with information from the other tokens.", "key": {'choice': 1}},
            {"kind": "short", "q": "In <i>“The baseball player swung the bat”</i>, which earlier word should "
                                   "<i>bat</i> pay most attention to, and what should its updated vector mean?",
             "lines": 2,
             "answer": "<i>baseball</i> (or <i>swung</i>): a baseball bat, not the flying animal.",
             "why": "Like <i>river</i> for <i>bank</i>, that word tells <i>bat</i> which meaning is meant."},
        ],
    },
    {
        "title": "Attention weights: a budget of 1",
        "segment": (43, 55),
        "figures": [{"t": 54.3, "caption": "How much bank listens to each token: 0.72 on river, very little on "
                                           "“the” and “on”. The weights add up to 1, like a budget."}],
        "body": [
            """<p>Each token decides <b>how much to listen</b> to every other token. These numbers are the
<b>attention weights</b>. They work like a <b>budget</b>: every weight is between 0 and 1, and together they <b>add
up to 1</b>. Here <i>bank</i> spends 0.72 of its attention on <i>river</i>, and only 0.04 each on <i>the</i> and
<i>on</i>. Notice that a token also gives some weight to <b>itself</b> (0.12).</p>""",
            "{fig0}",
            """<div class="box key"><b class="t">Key idea</b>For each token, the attention weights are all ≥ 0 and
<b>add up to 1</b>. A large weight means “listen closely to this token”; a tiny one means “mostly ignore it”.</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "From the chart, how much of <i>bank</i>'s attention goes to tokens other than "
                                    "<i>river</i>?",
             "answer": "0.28.", "why": "1 − 0.72 = 0.28 (check: 0.03 + 0.05 + 0.04 + 0.04 + 0.12 = 0.28).", "key": {'parts': [{'label': None, 'value': 0.28, 'tol': 0.005, 'unit': None}]}},
            {"kind": "mc", "q": "Which list could be one token's attention weights over four tokens?",
             "options": ["0.5, 0.5, 0.5, 0.5", "0.1, 0.2, 0.3, 0.4", "0.9, 0.2, −0.1, 0.0", "0.25, 0.25, 0.25, 0.2"],
             "answer": "B.", "why": "A adds up to 2; C adds up to 1 but has a negative weight; D adds up to only "
                                   "0.95. B is all ≥ 0 and adds up to exactly 1.", "key": {'choice': 1}},
            {"kind": "tf", "q": "“If <i>bank</i> starts paying more attention to <i>river</i>, it must pay less "
                                "attention to at least one other token.”",
             "answer": "True.", "why": "The total is fixed at 1, like a budget: spending more in one place means "
                                       "spending less somewhere else.", "key": {'value': True}},
            {"kind": "number", "q": "In <i>“I paid money into the bank”</i>, suppose <i>bank</i> gives 0.60 to "
                                    "<i>money</i>, 0.12 to itself, and splits the rest equally over <i>I</i>, "
                                    "<i>paid</i>, <i>into</i> and <i>the</i>. What weight does each of those four get?",
             "answer": "0.07.", "why": "1 − 0.60 − 0.12 = 0.28 is left, and 0.28 ÷ 4 = 0.07.", "key": {'parts': [{'label': None, 'value': 0.07, 'tol': 0.005, 'unit': None}]}},
        ],
    },
    {
        "title": "Query, key and value",
        "segment": (55, 81),
        "figures": [{"t": 66.2, "caption": "Every token makes a query, a key and a value."},
                    {"t": 81.1, "caption": "The library: read most from the best-matching title."}],
        "body": [
            """<p>How does a token decide whom to listen to? Every token produces <b>three new vectors</b>: a
<b>query</b> (“what am I looking for?”), a <b>key</b> (“what do I contain?”) and a <b>value</b> (“what will I share,
if someone listens to me?”).</p>""",
            """<p>Think of a library. Your <b>query</b> is the question you bring (<i>which kind of bank?</i>). Each
book has a <b>key</b>, like the title on its spine. You compare your question with every title, and the better the
match, the more you read from that book's contents: its <b>value</b>.</p>""",
            '<div class="figrow">{fig0}{fig1}</div>',
            """<div class="box key"><b class="t">Key idea</b>The <b>query and the key</b> are compared to decide
<b>how much</b> a token listens. The <b>value</b> is <b>what gets passed along</b> to the token that listens.</div>""",
        ],
        "exercises": [
            {"kind": "mc", "q": "In the library analogy, what plays the role of a token's <b>key</b>?",
             "options": ["The question you bring", "The title on a book's spine", "The text inside the book",
                         "The shelf the book stands on"],
             "answer": "B.", "why": "A is the query and C is the value. The key is what your question is compared "
                                   "with.", "key": {'choice': 1}},
            {"kind": "mc", "q": "To decide how much <i>bank</i> listens to <i>river</i>, which two vectors are "
                                "compared?",
             "options": ["bank's query and river's key", "bank's key and river's query",
                         "bank's value and river's value", "bank's query and river's value"],
             "answer": "A.", "why": "The token that asks uses its query; the token being considered offers its key. "
                                   "Values do not decide the amount; they are only used afterwards, for the mix.", "key": {'choice': 0}},
            {"kind": "short", "q": "The library's match scores were <i>Rivers &amp; Lakes</i> 0.9, <i>Banking "
                                   "101</i> 0.6, <i>Cooking</i> 0.1, <i>Poetry</i> 0.2. Which book do you read most "
                                   "from, and which of the three vectors is the text you read?",
             "answer": "<i>Rivers &amp; Lakes</i>; the value.",
             "why": "The highest match gets the most reading, and a book's contents (“land along a river's edge”) "
                    "are its value."},
        ],
    },
    {
        "title": "Matching with the dot product",
        "segment": (81, 93),
        "figures": [{"t": 93.2, "size": "small", "caption": "river's key points the same way as bank's query: top score."}],
        "body": [
            """<p>The match between a query and a key is measured with the <b>dot product</b> from the embeddings
episode (Foundations F02): multiply matching numbers, then add everything up. So <b>score = query · key</b>.</p>""",
            """<p>If <i>bank</i>'s query points in the <b>same direction</b> as <i>river</i>'s key, the score is high,
and bank pays attention to river. In the video, river's key turns from pointing away (score −1.18) to lining up with
the query (3.00), beating <i>the</i> (0.56) and <i>sat</i> (−0.60).</p>""",
            "{fig0}",
            """<div class="box key"><b class="t">Key idea</b><b>score = query · key</b>. Same direction → large
score → more attention. At a right angle the score is 0; pointing away, it is negative.</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "<i>bank</i>'s query is (1.6, 1.2) and <i>river</i>'s key is (1.2, 0.9). "
                                    "Compute the score.",
             "answer": "3.00.", "why": "1.6 × 1.2 + 1.2 × 0.9 = 1.92 + 1.08 = 3.00, the value in the video.", "key": {'parts': [{'label': None, 'value': 3, 'tol': 0.005, 'unit': None}]}},
            {"kind": "number", "q": "Two new keys: <i>water</i> (0.8, 0.6) and <i>money</i> (−0.6, 0.8). Compute each "
                                    "one's score with bank's query (1.6, 1.2). Which would bank listen to more?",
             "answer": "water 2.00, money 0.00; water.",
             "why": "1.28 + 0.72 = 2.00 and −0.96 + 0.96 = 0. <i>water</i> points exactly the same way as the query "
                    "(it is half of it); <i>money</i> is at a right angle to it.", "key": {'self': True}},
            {"kind": "mc", "q": "As river's key turns towards bank's query, its length stays 1.5. What happens to its "
                                "score?",
             "options": ["It goes down", "It goes up, from −1.18 to 3.00", "It stays the same, because the length "
                         "does not change", "It becomes exactly 1"],
             "answer": "B.", "why": "The dot product depends on direction as well as length: the smaller the angle to "
                                   "the query, the bigger the score.", "key": {'choice': 1}},
            {"kind": "tf", "q": "“<i>sat</i>'s score is −0.60. A negative score means the angle between the query "
                                "and the key is more than 90°.”",
             "answer": "True.", "why": "Less than 90° gives a positive score, exactly 90° gives 0, more than 90° "
                                       "(pointing away) gives a negative score.", "key": {'value': True}},
        ],
    },
    {
        "title": "Mixing in the values",
        "segment": (93, 104),
        "figures": [{"t": 104.0, "caption": "Each value times its weight, added up: a weighted mix, mostly river's. "
                                            "The mix is added to bank's own vector."}],
        "body": [
            """<p>Once the weights are known, <i>bank</i> collects a <b>weighted mix of all the values</b>: multiply
each token's value by its weight and add the results. River's weight is 0.72, so the mix is <b>mostly river's
value</b>. Then bank <b>adds</b> the mix to its own vector; it does not replace it. Now its vector doesn't just mean
<i>bank</i>: it means “bank, next to a river”.</p>""",
            "{fig0}",
            """<div class="box key"><b class="t">Key idea</b><b>new vector = old vector + the weighted mix of the
values</b> (weight × value, added up over all tokens). The weights decide how much of each value flows in.</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "River's value is (0.80, −0.50, 0.60) and its weight is 0.72. What does river "
                                    "alone contribute to the mix? Round to two decimals.",
             "answer": "(0.58, −0.36, 0.43).", "why": "0.72 × each entry = (0.576, −0.36, 0.432). That is most of the "
                                                     "full mix (0.61, −0.30, 0.40).", "key": {'parts': [{'label': '1st', 'value': 0.58, 'tol': 0.005, 'unit': None}, {'label': '2nd', 'value': -0.36, 'tol': 0.005, 'unit': None}, {'label': '3rd', 'value': 0.43, 'tol': 0.005, 'unit': None}]}},
            {"kind": "number", "q": "A token attends to just two tokens, with weights 0.25 and 0.75. Their values are "
                                    "(0.8, 0.2) and (0.0, 0.4). What is the weighted mix?",
             "answer": "(0.20, 0.35).", "why": "0.25 × (0.8, 0.2) + 0.75 × (0.0, 0.4) = (0.20, 0.05) + (0.00, 0.30) = "
                                              "(0.20, 0.35). It lies closer to the value with the bigger weight.", "key": {'parts': [{'label': '1st', 'value': 0.2, 'tol': 0.005, 'unit': None}, {'label': '2nd', 'value': 0.35, 'tol': 0.005, 'unit': None}]}},
            {"kind": "tf", "q": "“After attention, <i>bank</i>'s vector is replaced by the weighted mix.”",
             "answer": "False.", "why": "The mix is <b>added</b> to bank's own vector: (0.30, 0.50, −0.30) + (0.61, "
                                        "−0.30, 0.40) = (0.91, 0.20, 0.10). Bank keeps what it was and gains context.", "key": {'value': False}},
            {"kind": "code", "q": "<b>Try it yourself.</b> The code computes bank's update with NumPy, using the "
                                  "video's numbers. (a) What does it print? (b) In words, what does <code>w @ V</code> "
                                  "compute for each of the 3 columns? (c) Change <code>w</code> so that bank listens "
                                  "<i>only</i> to river (weight 1 on river, 0 elsewhere). What is the mix now, and why?",
             "code": """import numpy as np

# tokens:          I     sat   on    the   river bank
w = np.array([0.03, 0.05, 0.04, 0.04, 0.72, 0.12])  # bank's weights
V = np.array([[ 0.10,  0.40, -0.20],                 # one value per token
              [ 0.30, -0.10,  0.20],
              [-0.20,  0.10,  0.30],
              [ 0.00,  0.20,  0.10],
              [ 0.80, -0.50,  0.60],
              [ 0.20,  0.35, -0.40]])
x_bank = np.array([0.30, 0.50, -0.30])               # bank's own vector

mix = w @ V
print(mix.round(2))
print((x_bank + mix).round(2))""",
             "answer": "(a) [ 0.61 -0.3   0.4 ] and [0.91 0.2  0.1 ] · (c) [ 0.8 -0.5  0.6]",
             "why": """(a) The video's numbers (NumPy drops trailing zeros: −0.3 is −0.30). (b) For each column it
multiplies the six values in that column by the six weights and adds them up, e.g. 0.03 × 0.10 + 0.05 × 0.30 + … +
0.12 × 0.20 = 0.61. (c) <code>w = np.array([0, 0, 0, 0, 1.0, 0])</code> gives exactly river's value: with the whole
budget on one token, the mix is that token's value."""},
        ],
    },
    {
        "title": "Only looking backwards",
        "segment": (104, 115),
        "figures": [{"t": 114.4, "size": "small", "caption": "Causal attention: cat can look at The and at itself, never at sat."}],
        "body": [
            """<p>One rule for language models: a token can only <b>look backwards</b>, at itself and the tokens
before it. The model writes one word at a time (episode 1), so when it predicts the next word, the future isn't
written yet. In <i>“The cat sat on the”</i>, <i>cat</i> can look at <i>The</i> (and at itself), but never at
<i>sat</i>. This is called <b>causal attention</b>.</p>""",
            "{fig0}",
            """<div class="box key"><b class="t">Key idea</b>In causal attention each token attends only to
<b>itself and earlier tokens</b>. Later tokens get no attention at all.</div>""",
        ],
        "exercises": [
            {"kind": "short", "q": "In <i>“The cat sat on the”</i>, list every token that <i>on</i> is allowed to "
                                   "look at.",
             "answer": "The, cat, sat, on.", "why": "Itself and everything before it. The final <i>the</i> comes "
                                                    "later, so it is off limits."},
            {"kind": "number", "q": "In the 5-token text <i>“The cat sat on the”</i>, each token may look at some "
                                    "tokens. Count all allowed (looker, looked-at) pairs. How many of the 25 possible "
                                    "pairs are forbidden?",
             "lines": 2,
             "answer": "15 allowed, 10 forbidden.", "why": "The sees 1 token, cat 2, sat 3, on 4, the 5: "
                                                           "1 + 2 + 3 + 4 + 5 = 15, and 25 − 15 = 10.", "key": {'parts': [{'label': 'allowed', 'value': 15, 'tol': 0.5, 'unit': None}, {'label': 'forbidden', 'value': 10, 'tol': 0.5, 'unit': None}]}},
            {"kind": "mc", "q": "Why can a token not look at the tokens after it?",
             "options": ["Looking forward would be too slow to compute",
                         "When the model generates text, the later words have not been written yet",
                         "Later words never help with meaning", "The dot product only works in one direction"],
             "answer": "B.", "why": "Only the earlier words exist when each new word is predicted. C is false: later "
                                   "words can help, but they are not available yet.", "key": {'choice': 1}},
            {"kind": "tf", "q": "“In <i>“I sat on the river bank”</i>, <i>bank</i> is allowed to look at every token "
                                "of the sentence.”",
             "answer": "True.", "why": "<i>bank</i> is the last token, so every other token comes before it. That is "
                                       "why it can use <i>river</i>.", "key": {'value': True}},
        ],
    },
    {
        "title": "Learned, not hand-written",
        "segment": (115, 140),
        "figures": [{"t": 129.5, "size": "small", "caption": "q, k and v: the token's vector times three learned matrices."}],
        "body": [
            """<p>Each query, key and value is made by <b>multiplying the token's vector x by a matrix</b> (Foundations
F03): q = x · W<sub>Q</sub>, k = x · W<sub>K</sub>, v = x · W<sub>V</sub>. The matrices are <b>learned in
training</b>. Nobody tells the model that <i>river</i> explains <i>bank</i>; it figures that out from data.</p>""",
            "{fig0}",
            """<div class="box key"><b class="t">Key idea</b>q, k and v are the token's vector times <b>three learned
matrices</b>: what to look for, what to advertise and what to share are all <b>learned from data</b>.</div>""",
        ],
        "exercises": [
            {"kind": "mc", "q": "Where do W<sub>Q</sub>, W<sub>K</sub> and W<sub>V</sub> come from?",
             "options": ["Engineers write rules into them, like “river explains bank”",
                         "They are learned from data during training",
                         "They are copied from the embedding table", "They are random and never change"],
             "answer": "B.", "why": "They start random and training adjusts them; no rules are written by hand.", "key": {'choice': 1}},
            {"kind": "number", "q": "In the video's picture, the token vector x has 4 numbers and each matrix is a "
                                    "4 × 4 grid. (a) How many numbers are in the query q? (b) How many learned numbers "
                                    "are in W<sub>Q</sub>, W<sub>K</sub> and W<sub>V</sub> together?",
             "answer": "(a) 4. (b) 48.", "why": "A 4-number vector times a 4 × 4 matrix gives 4 numbers; each matrix "
                                               "holds 16, and 3 × 16 = 48.", "key": {'parts': [{'label': '(a)', 'value': 4, 'tol': 0.5, 'unit': None}, {'label': '(b)', 'value': 48, 'tol': 0.5, 'unit': None}]}},
            {"kind": "order", "q": "Put the steps of attention for <i>bank</i> in order: <i>add the mix to bank's "
                                   "vector · make q, k and v for every token · take the weighted mix of the values · "
                                   "compare bank's query with every key · turn the scores into weights that add up "
                                   "to 1</i>.",
             "answer": "make q, k, v → compare query with every key → weights → weighted mix of values → add to "
                       "bank's vector.",
             "why": "Matching (query · key) decides the weights; the weights decide the mix; the mix is added.", "key": {'items': ['make q, k and v for every token', "compare bank's query with every key", 'turn the scores into weights that add up to 1', 'take the weighted mix of the values', "add the mix to bank's vector"]}},
            {"kind": "tf", "q": "“Every token is multiplied by the same W<sub>Q</sub>, so <i>bank</i> and "
                                "<i>river</i> get different queries only because their vectors x are different.”",
             "answer": "True.", "why": "One set of learned matrices serves every token; each token's own vector makes "
                                       "its query, key and value different.", "key": {'value': True}},
        ],
    },
]
