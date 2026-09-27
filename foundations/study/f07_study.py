"""Study guide content for How LLMs Work · Foundations, F07: Softmax, Properly.

Build:  python framework/study_guide.py foundations f07 --video <rendered mp4>
Figures: {"t": seconds into the rendered video, "caption": ...}; placed in body blocks with {fig0}, {fig1}, ...
Exercise kinds: mc (options), tf, short, number, order, code. Every exercise has an answer and a why.
All numbers were checked with NumPy (softmax with subtract-the-max, as in the video).
"""

LESSON = {
    "series": "How LLMs Work · Foundations",
    "label": "F07",
    "title": "Softmax, Properly",
    "tagline": "From any scores to probabilities",
    "duration": "1:52",
    "intro": """<p>This lesson has one big idea: <b>softmax turns any list of scores into probabilities</b>, in two steps.
Exponentiate every score, then divide by the total. Everything else follows from those two steps: why it is “soft”, why
only the differences between scores matter, how temperature sharpens or flattens the result, and why code subtracts the
largest score first.</p>""",
    "prereq": """<div class="box note"><b>Before you start:</b> this lesson builds on F05 (Exponentials and Logarithms):
e<sup>x</sup> is always positive and exaggerates differences; and on F06 (Probability and Sampling): a distribution is a
list of probabilities that adds up to 1. A calculator helps for the Checks. Softmax is used in episodes 5 and 6
(attention scores become weights) and in episode 10 (logits become word probabilities).</div>""",
}

CONCEPTS = [
    {
        "title": "Scores in, probabilities out",
        "segment": (8, 36),
        "figures": [{"t": 20.3, "caption": "Logits can be any number; softmax makes them probabilities."},
                    {"t": 35.2, "caption": "The recipe on the illustrative logits (total 13.78)."}],
        "body": [
            """<p>A neural network ends with raw scores called <b>logits</b>, one per option. They can be any number:
large, small or negative (the video uses 2.1, −0.8, 0.4 and 1.3). But probabilities must be <b>≥ 0</b> and
<b>add up to 1</b>. <b>Softmax</b> converts in two steps. <b>Step 1: exponentiate</b> every score; e<sup>x</sup> is always
positive, so even −0.8 becomes 0.45. <b>Step 2: divide</b> each result by their total (13.78). Now they add up to
exactly 1.</p>""",
            '<div class="figrow">{fig0}{fig1}</div>',
            """<div class="box key"><b class="t">Key idea</b>softmax = <b>exponentiate, then divide by the total</b>.
Probability of option i = e<sup>score i</sup> ÷ (sum of all the e<sup>score</sup>). Step 1 makes every value
positive; step 2 makes them add up to 1.</div>""",
        ],
        "exercises": [
            {"kind": "mc", "q": "Why not skip step 1, and simply divide the raw scores by their total?",
             "options": ["It would give the same result, only more slowly",
                         "Raw scores can be negative, so some results would be negative",
                         "Division only works on whole numbers",
                         "The results would add up to more than 1"],
             "answer": "B.", "why": "For 2.1, −0.8, 0.4, 1.3 the total is 3.0, and −0.8 ÷ 3.0 ≈ −0.27: not a "
                                   "probability. e<sup>x</sup> makes every value positive first."},
            {"kind": "tf", "q": "“Logits must be between 0 and 1, just like probabilities.”",
             "answer": "False.", "why": "Logits can be any number, even negative. Only the output of softmax has to be "
                                        "≥ 0 and add up to 1."},
            {"kind": "number", "q": "Apply softmax by hand to the two scores <b>1</b> and <b>0</b>. (Use e ≈ 2.718.)",
             "answer": "73.1 % and 26.9 %.", "why": "e¹ ≈ 2.718 and e⁰ = 1, total 3.718. Then 2.718 ÷ 3.718 ≈ 0.731 and "
                                                   "1 ÷ 3.718 ≈ 0.269, which add up to 1."},
            {"kind": "order", "q": "Put in order: <i>divide by the total · raw scores (logits) · probabilities · "
                                   "exponentiate</i>.",
             "answer": "raw scores → exponentiate → divide by the total → probabilities.",
             "why": "Exponentiate first (everything positive), then normalize (everything adds up to 1)."},
        ],
    },
    {
        "title": "Working it through: 3, 2, 0",
        "segment": (36, 50),
        "figures": [{"t": 50.0, "caption": "The video's example: scores 3, 2, 0 become 70.5 %, 25.9 % and 3.5 %."}],
        "body": [
            """<p>The video's example: after <i>“The cat sat on the”</i>, the scores for <i>mat</i>, <i>floor</i> and
<i>sofa</i> are 3, 2 and 0. <b>Exponentiate:</b> e³ ≈ 20.1, e² ≈ 7.4, e⁰ = 1.0. <b>Add:</b> the total is about 28.5.
<b>Divide:</b> 20.1 ÷ 28.5 ≈ <b>70.5 %</b>, 7.4 ÷ 28.5 ≈ <b>25.9 %</b>, 1.0 ÷ 28.5 ≈ <b>3.5 %</b>.</p>""",
            "{fig0}",
            """<p>Two things to notice. The scores 3 and 2 differ by only 1, yet <i>mat</i> gets almost three times the
probability of <i>floor</i>: e<sup>x</sup> exaggerates differences. And a score of 0 is not “no chance”: e⁰ = 1, so
<i>sofa</i> still gets 3.5 %.</p>""",
            """<div class="box key"><b class="t">Key idea</b>By hand: <b>exponentiate each score, add them up, divide
each one by the total</b>. Check your work: the results must add up to 100 % (70.5 + 25.9 + 3.5 = 99.9, off only by
rounding).</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "Four words all get the same score, 2. What probability does each one get?",
             "answer": "25 %.", "why": "All four values are e² ≈ 7.39, so each is 7.39 ÷ (4 × 7.39) = ¼. Equal scores "
                                      "always give equal shares, whatever the score."},
            {"kind": "number", "q": "Compute the softmax of the scores <b>1, 0, 0</b>. (Use e ≈ 2.718.)",
             "answer": "57.6 %, 21.2 %, 21.2 %.", "why": "The values are 2.718, 1 and 1, total 4.718. "
                                                        "2.718 ÷ 4.718 ≈ 0.576 and 1 ÷ 4.718 ≈ 0.212."},
            {"kind": "number", "q": "Drop <i>sofa</i> and keep only <i>mat</i> (score 3) and <i>floor</i> (score 2). "
                                    "Using e³ ≈ 20.1 and e² ≈ 7.4, what are their new probabilities?",
             "answer": "About 73.1 % and 26.9 %.", "why": "The total is 20.1 + 7.4 = 27.5, and 20.1 ÷ 27.5 ≈ 0.731. "
                                                         "Notice: the same as the scores 1 and 0 in 1.3. Concept 4 "
                                                         "explains why."},
            {"kind": "tf", "q": "“A word with a score of 0 gets a probability of 0.”",
             "answer": "False.", "why": "e⁰ = 1, so the word keeps a share: <i>sofa</i> gets 3.5 %. Softmax never gives "
                                        "exactly 0."},
        ],
    },
    {
        "title": "A soft version of the max",
        "segment": (51, 61),
        "figures": [{"t": 61.3, "size": "small", "caption": "Hard max: all to the top score. Softmax: a share for "
                                                            "everyone."}],
        "body": [
            """<p>A <b>hard max</b> gives all the probability to the top score: <i>mat</i> 100 %, <i>floor</i> 0 %,
<i>sofa</i> 0 %. Softmax <b>leans toward</b> the top score (70.5 %) but still gives the others a share: a soft
version of the max.</p>""",
            "{fig0}",
            """<p>The order never changes: a higher score always gets a higher probability, because e<sup>x</sup> always
grows. The bigger the gaps between the scores, the closer softmax gets to a hard max.</p>""",
            """<div class="box key"><b class="t">Key idea</b>Hard max: the winner takes everything. Softmax: the winner
takes the most, and everyone else keeps a share, <b>in the same order as the scores</b>.</div>""",
        ],
        "exercises": [
            {"kind": "mc", "q": "For the illustrative logits 2.1, −0.8, 0.4, 1.3, what does a <b>hard max</b> output?",
             "options": ["100 %, 0 %, 0 %, 0 %", "59.3 %, 3.3 %, 10.8 %, 26.6 %", "25 %, 25 %, 25 %, 25 %",
                         "0 %, 100 %, 0 %, 0 %"],
             "answer": "A.", "why": "Everything goes to the top score, 2.1. B is what softmax gives."},
            {"kind": "number", "q": "Compute the softmax of the scores <b>4</b> and <b>0</b> (e⁴ ≈ 54.6). Is it closer "
                                    "to a hard max than 3, 2, 0 was?",
             "answer": "98.2 % and 1.8 %. Yes.", "why": "54.6 ÷ 55.6 ≈ 0.982. A gap of 4 is large, so softmax is "
                                                       "almost a hard max."},
            {"kind": "tf", "q": "“Softmax can give a lower score a higher probability than a higher score.”",
             "answer": "False.", "why": "e<sup>x</sup> always grows with x, so the ranking never changes."},
            {"kind": "mc", "q": "Which statement is <b>always</b> true for softmax?",
             "options": ["The top score gets more than 50 %",
                         "Every option gets more than 0 %, and higher scores get higher probabilities",
                         "The lowest score gets 0 %",
                         "Probabilities are proportional to the scores"],
             "answer": "B.", "why": "A fails for equal scores (three options get 33.3 % each); C fails because "
                                   "e<sup>x</sup> &gt; 0; D fails: scores 3 and 2 give 70.5 % and 25.9 %, 2.7 times as much, not 1.5 times."},
        ],
    },
    {
        "title": "Only the differences matter",
        "segment": (62, 68),
        "figures": [{"t": 68.0, "size": "small", "caption": "Add 10 to every score: the gaps stay 1 and 2, so nothing "
                                                            "changes."}],
        "body": [
            """<p>Add 10 to every score: 3, 2, 0 become 13, 12, 10. The probabilities do not change at all: still
70.5 %, 25.9 %, 3.5 %. The gaps between the scores (1 and 2) are the same, and <b>only the gaps matter</b>.</p>""",
            """<p>Why: e<sup>x + c</sup> = e<sup>x</sup> × e<sup>c</sup>, so adding c multiplies every value, and the
total, by the same factor, which cancels when you divide. So a gap of d always makes one option <b>e<sup>d</sup> times
as likely</b> as the other: <i>mat</i> vs <i>floor</i> (gap 1) is e ≈ 2.72 times (70.5 ÷ 25.9).</p>""",
            "{fig0}",
            """<div class="box key"><b class="t">Key idea</b>Adding the same number to every score changes nothing.
Only the differences matter: a gap of d makes one option e<sup>d</sup> times as likely as the other.</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "Compute the softmax of the scores <b>−1, −2, −4</b>. (No new calculation needed.)",
             "answer": "70.5 %, 25.9 %, 3.5 %.", "why": "They are 3, 2, 0 minus 4: the same gaps (1 and 2), so the same "
                                                       "probabilities."},
            {"kind": "mc", "q": "The scores 1 and 0 give 73.1 % and 26.9 %. Which scores give exactly the same result?",
             "options": ["2 and 0", "101 and 100", "10 and 0", "0.5 and 0"],
             "answer": "B.", "why": "The gap is still 1. The others change the gap (2, 10, 0.5), so they change the "
                                   "probabilities."},
            {"kind": "number", "q": "In the video's example, how many times as likely is <i>mat</i> (score 3) as "
                                    "<i>sofa</i> (score 0)? Use the gap, then check with the percentages.",
             "answer": "About 20 times.", "why": "The gap is 3, and e³ ≈ 20.1. Check: 70.5 ÷ 3.5 ≈ 20."},
            {"kind": "tf", "q": "“Multiplying every score by 2 also leaves the probabilities unchanged.”",
             "answer": "False.", "why": "Doubling changes the gaps (1 and 2 become 2 and 4), so the result is sharper: "
                                        "87.9 %, 11.9 %, 0.2 %. That is temperature, the next concept."},
        ],
    },
    {
        "title": "Temperature: sharper or flatter",
        "segment": (69, 79),
        "figures": [{"t": 76.3, "caption": "T = 0.5: gaps double, mat dominates."},
                    {"t": 78.8, "caption": "T = 2: gaps halve, the distribution flattens."}],
        "body": [
            """<p>Before the softmax, we can divide the scores by a <b>temperature</b> T: softmax(scores / T). With
<b>T = 0.5</b>, the scores 3, 2, 0 become 6, 4, 0: the gaps double and the top choice dominates (87.9 %, 11.9 %,
0.2 %). With <b>T = 2</b>, they become 1.5, 1, 0: the gaps halve and the distribution flattens (54.7 %, 33.1 %,
12.2 %).</p>""",
            '<div class="figrow">{fig0}{fig1}</div>',
            """<div class="box key"><b class="t">Key idea</b>Divide the scores by T, then apply softmax.
<b>T &lt; 1: the gaps grow → sharper</b> (toward a hard max). <b>T &gt; 1: the gaps shrink → flatter</b> (toward
equal shares). T = 1 is plain softmax.</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "Scores 3, 2, 0 with <b>T = 0.25</b>. (a) What are the scores after dividing by T? "
                                    "(b) What are the gaps? (c) Will <i>mat</i> get more or less than the 87.9 % it gets "
                                    "at T = 0.5?",
             "lines": 2,
             "answer": "(a) 12, 8, 0. (b) 4 and 8. (c) More: 98.2 %.",
             "why": "T = 0.25 multiplies the gaps by 4, even more than T = 0.5 does, so the result is even sharper."},
            {"kind": "order", "q": "Scores 3, 2, 0. Order these temperatures by the probability <i>mat</i> gets, "
                                   "lowest first: <i>T = 0.5 · T = 10 · T = 1 · T = 2</i>.",
             "answer": "T = 10 → T = 2 → T = 1 → T = 0.5.",
             "why": "37.8 %, 54.7 %, 70.5 %, 87.9 %: the higher the temperature, the flatter the distribution."},
            {"kind": "tf", "q": "“Dividing the scores by T = 0.5 gives the same probabilities as multiplying every "
                                "score by 2.”",
             "answer": "True.", "why": "Dividing by 0.5 is multiplying by 2. Both give 6, 4, 0 → 87.9 %, 11.9 %, 0.2 %."},
            {"kind": "mc", "q": "A chatbot keeps giving the same safe, predictable answer. You want more varied word "
                                "choices. What should you change?",
             "options": ["Lower the temperature to 0.3", "Raise the temperature to 1.3", "Add 5 to every score",
                         "Subtract the largest score"],
             "answer": "B.", "why": "T &gt; 1 flattens the distribution, so sampling (F06) picks less likely words more "
                                   "often. C and D change nothing (concept 4)."},
        ],
    },
    {
        "title": "Subtract the max: softmax in code",
        "segment": (80, 100),
        "figures": [{"t": 92.3, "size": "small", "caption": "Subtract the largest score: identical answer, no overflow."}],
        "body": [
            """<p>A computer cannot store e<sup>1000</sup>: the largest 64-bit float is about e<sup>709</sup> ≈
8.2 × 10<sup>307</sup>, so e<sup>1000</sup> <b>overflows</b> to ∞. The fix: <b>subtract the largest score first</b>.
For 3, 2, 0 that gives 0, −1, −3; e<sup>x</sup> ≈ 1.00, 0.37, 0.05; total ≈ 1.42; divided out: 70.5 %, 25.9 %,
3.5 %. Identical, because only differences matter.</p>""",
            "{fig0}",
            """<p>In code that is three lines: divide by the temperature, exponentiate after subtracting the max, and
normalize.</p>""",
            """<pre class="code">def softmax(scores, temperature=1.0):
    z = np.array(scores) / temperature
    e = np.exp(z - z.max())         # no overflow
    return e / e.sum()

softmax([3, 2, 0])                  # [0.705, 0.259, 0.035]
softmax([3, 2, 0], 0.5)             # [0.879, 0.119, 0.002]</pre>""",
            """<div class="box key"><b class="t">Key idea</b>Subtracting the max changes nothing mathematically
(concept 4), but every exponent becomes ≤ 0, so every e<sup>x</sup> lies between 0 and 1 and <b>nothing
overflows</b>.</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "Scores 1000, 999, 997. (a) What are they after subtracting the max? (b) What are "
                                    "the probabilities?",
             "lines": 2,
             "answer": "(a) 0, −1, −3. (b) 70.5 %, 25.9 %, 3.5 %.",
             "why": "The gaps are 1 and 2, as for 3, 2, 0, so these are exactly the video's numbers."},
            {"kind": "tf", "q": "“After subtracting the max, the top score always becomes e⁰ = 1, and every other "
                                "value lies between 0 and 1.”",
             "answer": "True.", "why": "The top score becomes 0, and all the others become negative, so their "
                                       "e<sup>x</sup> is below 1."},
            {"kind": "code", "q": "<b>Try it yourself.</b> Below are the video's <code>softmax</code> and a "
                                  "<code>naive_softmax</code> that skips the subtract-the-max step. Predict each output "
                                  "first, then run it. (a) <code>naive_softmax([1000, 999, 997])</code>. Why? "
                                  "(b) <code>softmax([1000, 999, 997])</code>. (c) <code>softmax([3, 2, 0], 0.01)</code> "
                                  "and <code>softmax([3, 2, 0], 100)</code>: which one looks like a hard max?",
             "code": """import numpy as np
np.set_printoptions(precision=3, suppress=True)

def naive_softmax(scores):
    e = np.exp(np.array(scores, dtype=float))
    return e / e.sum()

def softmax(scores, temperature=1.0):
    z = np.array(scores) / temperature
    e = np.exp(z - z.max())         # no overflow
    return e / e.sum()

print(softmax([3, 2, 0]))           # [0.705 0.259 0.035]""",
             "answer": "(a) [nan nan nan] · (b) [0.705 0.259 0.035] · (c) [1. 0. 0.] and [0.338 0.334 0.328]",
             "why": """(a) <code>np.exp(1000.0)</code> overflows to <code>inf</code> (NumPy prints a warning), and
inf ÷ inf is <code>nan</code>, “not a number”. (b) Subtracting 1000 gives 0, −1, −3, the video's example. (c) T = 0.01
makes the gaps 100 times bigger, so <i>mat</i> takes everything: a hard max. T = 100 shrinks the gaps to 0.01 and 0.02,
so the shares are almost equal."""},
        ],
    },
]
