"""Study guide content for How LLMs Work · Foundations, F05: Exponentials and Logarithms.

Build:  python framework/study_guide.py foundations f05
Figures: {"t": seconds into the rendered video, "caption": ...}; placed in body blocks with {fig0}, {fig1}, ...
Exercise kinds: mc (options), tf, short, number, order, code. Every exercise has an answer and a why.
All numbers verified with NumPy (the scene's own asserts plus the hand calculations below).
"""

LESSON = {
    "series": "How LLMs Work · Foundations",
    "label": "F05",
    "title": "Exponentials and Logarithms",
    "tagline": "Fast growth, and the function that undoes it",
    "duration": "1:59",
    "intro": """<p>This lesson is about two functions that appear all over LLMs. The <b>exponential</b> eˣ grows by
multiplying: it is always positive, and it turns small differences into big ones. The <b>logarithm</b> ln x undoes it:
it turns probabilities into negative numbers that are easy to compare, and it turns multiplication into addition.</p>""",
    "prereq": """<div class="box note"><b>Before you start:</b> no earlier Foundations video is needed, only powers such
as 2³ = 2 × 2 × 2 = 8. You will meet the exponential inside softmax (F07, and episodes 6 and 10), the logarithm in the
training loss of episode 11, and a log-scale chart in episode 12.</div>""",
}

CONCEPTS = [
    {
        "title": "Growth by multiplying",
        "segment": (8, 26),
        "figures": [{"t": 15.8, "caption": "The exponential eˣ and its opposite, the logarithm ln x: mirror images "
                                           "across the line y = x."},
                    {"t": 25.8, "caption": "Doubling every step: 2, 4, 8, 16, … After ten doublings, 2¹⁰ = 1,024."}],
        "body": [
            """<p>Two functions show up again and again in LLMs: the <b>exponential</b> and its opposite, the
<b>logarithm</b>. Their graphs are mirror images of each other.</p>""",
            """<p><b>Exponential growth</b> means <b>multiplying</b> by the same number at every step, not adding.
Doubling gives 2, 4, 8, 16, … and after just ten doublings you are past a thousand: <b>2¹⁰ = 1,024</b>. Adding 2 ten
times would only get you to 20.</p>""",
            '<div class="figrow">{fig0}{fig1}</div>',
            """<div class="box key"><b class="t">Key idea</b>Exponential = <b>repeated multiplication</b>. Every step
multiplies by the same factor, so it soon outruns anything that adds: 2¹⁰ = 1,024.</div>""",
        ],
        "exercises": [
            {"kind": "mc", "q": "Which sequence grows exponentially?",
             "options": ["2, 4, 6, 8, 10", "1, 4, 9, 16, 25", "3, 6, 12, 24, 48", "5, 10, 15, 20, 25"],
             "answer": "C.", "why": "Each term is the previous one × 2. A and D add a fixed amount; B are squares "
                                   "(1², 2², 3², …), whose step-to-step factor keeps shrinking."},
            {"kind": "number", "q": "Start from 1. (a) Double it 10 times. (b) Instead, add 2 ten times. "
                                    "(c) Double it 20 times.",
             "answer": "(a) 1,024 (b) 21 (c) 1,048,576.",
             "why": "2²⁰ = 2¹⁰ × 2¹⁰ = 1,024 × 1,024: past a million, while adding only reaches 21."},
            {"kind": "number", "q": "A number starts at 1 and is multiplied by 3 at every step. What is it after "
                                    "5 steps?",
             "answer": "243.", "why": "3⁵ = 3 × 3 × 3 × 3 × 3 = 243. Same idea as doubling, with factor 3."},
            {"kind": "tf", "q": "“In exponential growth, the amount added at each step stays the same.”",
             "answer": "False.", "why": "The <i>factor</i> stays the same (× 2); the amount added keeps growing: "
                                       "+2, +4, +8, +16, …"},
        ],
    },
    {
        "title": "The number e, and eˣ is always positive",
        "segment": (26, 46),
        "figures": [{"t": 35.2, "caption": "The usual base is e ≈ 2.718. The function eˣ has two properties we care "
                                           "about."},
                    {"t": 46.3, "caption": "Property 1: e⁻¹⁰ ≈ 0.0000454, less than 0.0001 but still positive."}],
        "body": [
            """<p>In machine learning the base is usually the number <b>e ≈ 2.718</b> (Euler's number), and the
function is <b>eˣ</b>, “e to the x”. Just like doubling, every +1 in x multiplies eˣ by e: e⁰ = 1, e¹ ≈ 2.718,
e² ≈ 7.39, e³ ≈ 20.1.</p>""",
            """<p>Property 1: eˣ is <b>always positive</b>. Negative inputs give small numbers, never negative ones:
e⁻¹ ≈ 0.368, and e⁻¹⁰ ≈ 0.0000454, less than one ten-thousandth. It never reaches zero.</p>""",
            '<div class="figrow">{fig0}{fig1}</div>',
            """<div class="box key"><b class="t">Key idea</b>eˣ &gt; 0 for every x. Very negative inputs give tiny
positive outputs, e⁰ = 1, and each +1 in x multiplies the output by e ≈ 2.718.</div>""",
        ],
        "exercises": [
            {"kind": "tf", "q": "“For a very negative input, such as x = −100, eˣ is exactly zero.”",
             "answer": "False.", "why": "It is extremely small, but still positive. eˣ never reaches zero."},
            {"kind": "mc", "q": "Which list could be the outputs of eˣ for three inputs?",
             "options": ["1, 0.37, 7.39", "−1, 1, 2.72", "0, 1, 2.72", "2.72, −7.39, 20.1"],
             "answer": "A.", "why": "These are e⁰, e⁻¹ and e². The others contain 0 or a negative number, which eˣ "
                                   "never produces."},
            {"kind": "number", "q": "Using e ≈ 2.718, find (a) e⁰, (b) e² and (c) e⁻¹ = 1/e.",
             "answer": "(a) 1 (b) ≈ 7.39 (c) ≈ 0.368.",
             "why": "e² = 2.718 × 2.718 ≈ 7.39; e⁻¹ = 1 ÷ 2.718 ≈ 0.368; zero steps of multiplying leaves 1."},
            {"kind": "number", "q": "The scores −2, 0 and 3 go through eˣ. Roughly what comes out? Is any output "
                                    "negative?",
             "answer": "≈ 0.135, 1 and 20.1. None is negative.",
             "why": "Even the negative score gives a positive number. This is what lets softmax (F07) turn any scores "
                    "into positive numbers."},
        ],
    },
    {
        "title": "eˣ exaggerates differences",
        "segment": (47, 60),
        "figures": [{"t": 59.7, "caption": "Inputs 1 and 3 differ by only 2, but e³ ≈ 20.1 is about 7.4 times "
                                           "e¹ ≈ 2.7."}],
        "body": [
            """<p>Property 2: eˣ <b>exaggerates differences</b>. The inputs 1 and 3 differ by just 2. But
e¹ ≈ 2.7 and e³ ≈ 20.1, so now one is <b>more than seven times</b> the other: 20.1 ÷ 2.7 ≈ 7.4 (exactly,
e³/e¹ = e² ≈ 7.39).</p>""",
            """<p>The reason is concept 2: every +1 in the input multiplies the output by e. A gap of 2 in the
inputs becomes a factor of e² ≈ 7.39 in the outputs, wherever the two inputs are.</p>""",
            "{fig0}",
            """<div class="box key"><b class="t">Key idea</b>eˣ turns <b>differences</b> into <b>ratios</b>: inputs
that differ by d give outputs whose ratio is e<sup>d</sup>. A small lead in the input becomes a big lead in the
output.</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "e¹ ≈ 2.718 and e³ ≈ 20.09. Compute e³ ÷ e¹ to two decimals. Which power of e "
                                    "is it?",
             "answer": "≈ 7.39 = e².", "why": "20.09 ÷ 2.718 ≈ 7.39: two extra factors of e."},
            {"kind": "number", "q": "The inputs 10 and 12 also differ by 2. How many times bigger is e¹² than e¹⁰?",
             "answer": "≈ 7.39 times, the same as before.",
             "why": "Only the difference matters: e¹² ÷ e¹⁰ = e² ≈ 7.39."},
            {"kind": "mc", "q": "Two scores are 2 and 5. After eˣ, how does the larger output compare with the smaller?",
             "options": ["2.5 times bigger (5 ÷ 2)", "Exactly 3 bigger", "About 20 times bigger", "About the same"],
             "answer": "C.", "why": "The inputs differ by 3, so the outputs differ by a factor e³ ≈ 20.1."},
            {"kind": "tf", "q": "“eˣ keeps ratios: if one input is 3 times another, its output is also 3 times "
                                "bigger.”",
             "answer": "False.", "why": "Inputs 1 and 3 (ratio 3) give 2.7 and 20.1 (ratio ≈ 7.4). eˣ exaggerates."},
        ],
    },
    {
        "title": "The logarithm undoes the exponential",
        "segment": (60, 73),
        "figures": [{"t": 72.7, "caption": "ln undoes exp: ln(eˣ) = x. ln 1 = 0, and as x shrinks toward 0, ln x "
                                           "dives toward −∞."}],
        "body": [
            """<p>The <b>natural logarithm</b>, ln, undoes the exponential. It answers the question “e to <i>what</i>
gives this number?”, so <b>ln(eˣ) = x</b>: go through exp and then ln, and you are back where you started. For
example ln 20.1 ≈ 3, because e³ ≈ 20.1.</p>""",
            """<p>Two landmarks: <b>ln 1 = 0</b> (because e⁰ = 1), and as the input shrinks toward zero, ln x
<b>dives toward minus infinity</b>. So numbers between 0 and 1 have <b>negative</b> logs.</p>""",
            "{fig0}",
            """<div class="box key"><b class="t">Key idea</b>ln(eˣ) = x. ln 1 = 0; ln x is negative for x between 0 and 1
and heads to −∞ as x approaches 0; it is positive for x &gt; 1.</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "Without a calculator: (a) ln(e⁵) (b) ln 1 (c) ln e (d) ln 7.39 (hint: concept 3).",
             "answer": "(a) 5 (b) 0 (c) 1 (d) ≈ 2.",
             "why": "ln asks “e to what?”: e⁵, e⁰, e¹ and e² ≈ 7.39."},
            {"kind": "order", "q": "Put in order from smallest to largest: <i>ln 2 · ln 0.01 · ln 1 · ln 0.5</i>.",
             "answer": "ln 0.01 → ln 0.5 → ln 1 → ln 2.",
             "why": "About −4.61, −0.69, 0 and 0.69. Bigger input, bigger log."},
            {"kind": "tf", "q": "“ln 0.5 is a positive number.”",
             "answer": "False.", "why": "ln 0.5 ≈ −0.693. Every number between 0 and 1 has a negative log."},
            {"kind": "mc", "q": "What happens to ln x as x gets closer and closer to 0 (0.1, 0.01, 0.001, …)?",
             "options": ["It approaches 0", "It approaches 1", "It dives toward −∞", "It becomes positive"],
             "answer": "C.", "why": "ln 0.1 ≈ −2.3, ln 0.01 ≈ −4.6, ln 0.001 ≈ −6.9, … with no lower limit."},
        ],
    },
    {
        "title": "Logs of probabilities",
        "segment": (73, 88),
        "figures": [{"t": 88.1, "caption": "ln p for probabilities: 0.9 gives about −0.1, 0.01 gives about −4.6. "
                                           "The less likely, the more negative."}],
        "body": [
            """<p>Probabilities live between 0 and 1, exactly where ln is <b>negative</b>. That makes logs perfect for
probabilities. A likely event has a log <b>close to zero</b>: ln 0.9 ≈ −0.105. An unlikely one is strongly
negative: ln 0.01 ≈ −4.605. A certain event, p = 1, has ln 1 = 0.</p>""",
            "{fig0}",
            """<div class="box key"><b class="t">Key idea</b>For a probability p, ln p ≤ 0. <b>The less likely, the more
negative</b>: likely events sit near 0 (ln 0.9 ≈ −0.105), unlikely ones far below (ln 0.01 ≈ −4.605).</div>""",
        ],
        "exercises": [
            {"kind": "mc", "q": "Model A gives the correct next word probability 0.9; model B gives it 0.01. "
                                "Which statement is right?",
             "options": ["A's log-probability (≈ −0.105) is closer to 0, so A did better",
                         "B's log-probability is closer to 0, so B did better",
                         "Both log-probabilities are positive", "Logs can't compare probabilities"],
             "answer": "A.", "why": "ln 0.9 ≈ −0.105 is near 0; ln 0.01 ≈ −4.605 is far below. Closer to 0 = more likely."},
            {"kind": "tf", "q": "“The log of a probability is never positive.”",
             "answer": "True.", "why": "A probability is at most 1, and ln 1 = 0; anything below 1 has a negative log."},
            {"kind": "number", "q": "Without a calculator, using the two landmarks above: (a) is ln 0.99 closer to 0 "
                                    "or to −1? (b) Is ln 0.001 above or below −4.605?",
             "answer": "(a) Very close to 0 (≈ −0.01). (b) Below (≈ −6.9).",
             "why": "0.99 is even more likely than 0.9, so its log is even nearer 0. 0.001 is less likely than 0.01, so "
                    "its log is more negative."},
            {"kind": "short", "q": "A model's log-probability for the correct word is −4.605. What probability did it "
                                   "give that word? Was it confident?",
             "answer": "0.01: not confident at all.",
             "why": "ln 0.01 ≈ −4.605, so p = e<sup>−4.605</sup> ≈ 0.01. It thought the correct word was very unlikely."},
        ],
    },
    {
        "title": "Multiplication becomes addition",
        "segment": (89, 100),
        "figures": [{"t": 99.9, "caption": "log(a × b) = log a + log b. On a log scale each step is × 10, so a "
                                           "thousand and a billion fit on one axis."}],
        "body": [
            """<p>Logs turn <b>multiplication into addition</b>: <b>log(a × b) = log a + log b</b>. For example,
ln 6 + ln 7 ≈ 1.792 + 1.946 = 3.738 ≈ ln 42. This matters for text: the probability of a whole sentence is the
<b>product</b> of its words' probabilities (each given the words before it), and its log is simply the <b>sum</b> of
the words' logs.</p>""",
            """<p>Logs also give <b>log-scale</b> charts: each tick is <b>ten times</b> the one before (1, 10, 100, …),
so a thousand and a billion fit on the same axis.</p>""",
            "{fig0}",
            """<div class="box key"><b class="t">Key idea</b>log(a × b) = log a + log b: products of many probabilities
become sums of logs. On a log scale, equal steps mean equal <b>factors</b> (× 10).</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "A three-word sentence has word probabilities 0.5, 0.2 and 0.1. (a) Multiply them. "
                                    "(b) Add their logs: ln 0.5 ≈ −0.693, ln 0.2 ≈ −1.609, ln 0.1 ≈ −2.303. "
                                    "(c) Compare (b) with ln 0.01 from concept 5.",
             "answer": "(a) 0.01 (b) −4.605 (c) the same.",
             "why": "The sum of the logs is the log of the product: ln 0.01 ≈ −4.605."},
            {"kind": "number", "q": "ln 0.2 ≈ −1.609 and ln 0.1 ≈ −2.303. By how much does the log drop when a "
                                    "probability halves? Use it to predict ln 0.005 from ln 0.01 ≈ −4.605.",
             "answer": "By ≈ 0.693 each time; ln 0.005 ≈ −5.298.",
             "why": "Halving is × 0.5, which adds ln 0.5 ≈ −0.693, wherever you start."},
            {"kind": "number", "q": "On the video's log-scale axis (1, 10, 100, 1K, …, 1B), how many × 10 steps "
                                    "is it from a thousand to a billion? Between which two ticks is 50,000?",
             "answer": "6 steps; between 10K and 100K.",
             "why": "1K → 10K → 100K → 1M → 10M → 100M → 1B. 50,000 is more than 10,000 and less than 100,000."},
        ],
    },
    {
        "title": "exp and log in NumPy",
        "segment": (101, 117),
        "figures": [{"t": 102.8, "size": "small", "caption": "The code from the video: np.exp and np.log."}],
        "body": [
            """<p>In NumPy it's <code>np.exp</code> and <code>np.log</code> (<code>np.log</code> is the natural log,
ln). Both work on a single number or on every number of an array at once:</p>""",
            "{fig0}",
            """<pre class="code">np.exp([1, 3])        # array([ 2.718, 20.086])
np.exp(-10)           # 4.54e-05
np.log(0.9)           # -0.105
np.log(0.01)          # -4.605</pre>""",
            """<p>Where you'll see them: the exponential inside <b>softmax</b> (episodes 6 and 10), the logarithm in
the <b>loss</b> of episode 11, and the <b>log-scale chart</b> of episode 12.</p>""",
            """<div class="box key"><b class="t">Key idea</b><code>np.exp</code> turns scores into positive,
exaggerated numbers; <code>np.log</code> turns probabilities into negative numbers you can add.</div>""",
        ],
        "exercises": [
            {"kind": "short", "q": "What does <code>np.log(np.exp(4.2))</code> return? Why?",
             "answer": "4.2.", "why": "ln undoes exp: ln(eˣ) = x."},
            {"kind": "mc", "q": "<code>p</code> is an array of the probabilities of a sentence's words. Which expression "
                                "gives the log of the whole sentence's probability as a sum?",
             "options": ["<code>np.log(p).sum()</code>", "<code>np.exp(p).sum()</code>",
                         "<code>np.log(p.sum())</code>", "<code>np.log(p).prod()</code>"],
             "answer": "A.", "why": "The log of a product is the sum of the logs. C takes the log of a sum (wrong rule); "
                                   "D multiplies the logs."},
            {"kind": "code", "q": "<b>Try it yourself.</b> Predict each printed line before running the code. "
                                  "(a) What do the first two lines print (2 decimals)? Which number from concept 3 is "
                                  "the second one? (b) What does the third line print, and why? (c) The fourth line "
                                  "prints two numbers: which, and how are they related? (d) <code>q.prod()</code> "
                                  "prints <code>0.0</code>. Is the true product zero? What does the sum of logs give "
                                  "instead?",
             "code": """import numpy as np

x = np.array([1.0, 2.0, 3.0])
print(np.exp(x))
print(np.exp(x[2]) / np.exp(x[0]))
print(np.log(np.exp(x)))

p = np.array([0.5, 0.2, 0.1])       # probabilities of 3 words
print(p.prod(), np.log(p).sum())

q = np.full(1000, 0.01)             # 1,000 words, each 0.01
print(q.prod(), np.log(q).sum())""",
             "answer": "(a) [2.72, 7.39, 20.09] and 7.39 · (b) [1. 2. 3.] · (c) 0.01 and −4.605 · (d) −4605.17",
             "why": """(a) e¹, e², e³; the ratio e³/e¹ = e² ≈ 7.39 is the one from concept 3.
(b) ln undoes exp, so you get x back.
(c) The product 0.01 (printed as 0.010000000000000002, a tiny rounding error) and the sum of logs −4.605 = ln 0.01:
the log of a product is the sum of the logs.
(d) Not zero: 0.01 multiplied 1,000 times is 10<sup>−2000</sup>, far too small for the computer to store, so it rounds
to 0.0. The sum of logs, 1,000 × (−4.605) ≈ −4605.17, is a perfectly ordinary number. This is one reason models work
with log-probabilities."""},
        ],
    },
]
