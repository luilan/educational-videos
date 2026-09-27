"""Study guide content for How LLMs Work · Foundations, F06: Probability and Sampling.

Build:  python framework/study_guide.py foundations f06
Figures: {"t": seconds into the rendered video, "caption": ...}; placed in body blocks with {fig0}, {fig1}, ...
Exercise kinds: mc (options), tf, short, number, order, code. Every exercise has an answer and a why.
All numbers verified with NumPy (the scene's own asserts plus the hand calculations below).
"""

LESSON = {
    "series": "How LLMs Work · Foundations",
    "label": "F06",
    "title": "Probability and Sampling",
    "tagline": "Distributions, weighted dice, and randomness",
    "duration": "1:45",
    "intro": """<p>This lesson is about what an LLM actually outputs: not a word, but a <b>probability
distribution</b> over words. You will see what a probability is, why a distribution adds up to one, how
<b>sampling</b> picks one word at random according to it (a weighted spinner), and how a <b>seed</b> makes that
randomness repeatable.</p>""",
    "prereq": """<div class="box note"><b>Before you start:</b> no earlier Foundations video is needed. The whole
lesson uses the video's illustrative distribution for the word after <i>“The cat sat on the”</i>: <i>mat</i> 0.40,
<i>floor</i> 0.25, <i>sofa</i> 0.15, <i>bed</i> 0.10, <i>roof</i> 0.10. You will see these ideas in episode 1 (the
model predicts a distribution), episode 10 (sampling from it) and episode 14.</div>""",
}

CONCEPTS = [
    {
        "title": "A probability is a number from 0 to 1",
        "segment": (8, 26),
        "figures": [{"t": 14.8, "caption": "An LLM never simply knows the next word: it produces a probability for "
                                           "each candidate."},
                    {"t": 25.6, "caption": "0 means impossible, 1 means certain, 0.5 means about half the time."}],
        "body": [
            """<p>An LLM never simply knows the next word. It produces <b>probabilities</b>. A probability is a
number <b>between 0 and 1</b> that says how likely something is: <b>0</b> means impossible, <b>1</b> means certain,
and <b>0.5</b> means it happens about half the time. As a percentage, 0.25 is 25&nbsp;%.</p>""",
            '<div class="figrow">{fig0}{fig1}</div>',
            """<div class="box key"><b class="t">Key idea</b>0 ≤ p ≤ 1. The bigger p, the more often the event happens:
p = 0.5 about half the time, p = 0.1 about one time in ten.</div>""",
        ],
        "exercises": [
            {"kind": "mc", "q": "Which of these can be a probability?",
             "options": ["1.2", "−0.1", "0.35", "3/2"],
             "answer": "C.", "why": "A probability must lie between 0 and 1. 1.2 and 3/2 are above 1; −0.1 is below 0."},
            {"kind": "number", "q": "An event has probability 0.1. In 1,000 tries, about how many times does it happen? "
                                    "And an event with probability 0.5?",
             "answer": "About 100; about 500.", "why": "0.1 × 1,000 = 100 and 0.5 × 1,000 = 500. “About”, because "
                                                     "chance makes the exact count vary."},
            {"kind": "tf", "q": "“An event with probability 0.001 is impossible.”",
             "answer": "False.", "why": "It is very unlikely (about once in 1,000 tries), but possible. Only 0 means "
                                       "impossible."},
            {"kind": "order", "q": "Put in order from least to most likely: <i>p = 0.5 · certain · p = 0.05 · "
                                   "impossible · p = 0.9</i>.",
             "answer": "impossible → 0.05 → 0.5 → 0.9 → certain.",
             "why": "Impossible is p = 0 and certain is p = 1; everything else sits in between."},
        ],
    },
    {
        "title": "A distribution adds up to one",
        "segment": (26, 43),
        "figures": [{"t": 42.5, "caption": "One whole unit, shared out across the options: sum = 1.00."}],
        "body": [
            """<p>A <b>probability distribution</b> spreads <b>one whole unit of belief</b> across all the options.
For the word after <i>“The cat sat on the”</i>, the video's illustrative distribution is <i>mat</i> 0.40,
<i>floor</i> 0.25, <i>sofa</i> 0.15, <i>bed</i> 0.10 and <i>roof</i> 0.10: together <b>exactly 1</b>. So the chance
of “bed or roof” is the sum of their shares, and “not mat” gets whatever is left, 1 − 0.40 = 0.60.</p>""",
            "{fig0}",
            """<div class="box key"><b class="t">Key idea</b>Every value is between 0 and 1, and together they <b>add up
to exactly 1</b>. More for one option means less for the others.</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "Using the video's distribution: (a) check that it adds up to 1. What is the "
                                    "probability that the next word is (b) not <i>mat</i>, (c) <i>bed</i> or "
                                    "<i>roof</i>?",
             "answer": "(a) 1.00 (b) 0.60 (c) 0.20.",
             "why": "0.40 + 0.25 + 0.15 + 0.10 + 0.10 = 1.00; 1 − 0.40 = 0.60; 0.10 + 0.10 = 0.20."},
            {"kind": "mc", "q": "Which could be a distribution over the words <i>yes</i>, <i>no</i>, <i>maybe</i>?",
             "options": ["0.6, 0.3, 0.3", "0.8, 0.2, 0.0", "0.9, 0.2, −0.1", "0.3, 0.3, 0.3"],
             "answer": "B.", "why": "A adds up to 1.2 and D to only 0.9; C has a negative value. B is fine: 0 is "
                                   "allowed and the total is exactly 1."},
            {"kind": "tf", "q": "“A model raises <i>mat</i> from 0.40 to 0.60. The other four words can keep "
                                "exactly the same probabilities.”",
             "answer": "False.", "why": "The total would become 1.20. The other words must lose 0.20 between them."},
            {"kind": "number", "q": "For the word after <i>“I drank a cup of”</i>, a model gives <i>tea</i> 0.55, "
                                    "<i>coffee</i> 0.30 and <i>water</i> 0.10. How much probability is left for all the "
                                    "other words together?",
             "answer": "0.05.", "why": "0.55 + 0.30 + 0.10 = 0.95, and the whole distribution adds up to 1."},
        ],
    },
    {
        "title": "Sampling: a weighted spinner",
        "segment": (43, 52),
        "figures": [{"t": 52.1, "caption": "A spinner with one slice per word. floor's slice is 0.25 × 360° = 90°; "
                                           "this spin landed on floor."}],
        "body": [
            """<p><b>Sampling</b> means picking one option <b>at random, according to the probabilities</b>. Think of
a spinner where each word gets a <b>slice sized by its probability</b>: <i>floor</i> gets 0.25 of the circle,
0.25 × 360° = 90°. Spin it and it lands on one word. This time it landed on <i>floor</i>, not on the favourite,
<i>mat</i>.</p>""",
            "{fig0}",
            """<div class="box key"><b class="t">Key idea</b>Slice size = probability. One sample = one spin: a random
pick in which every option can win, each with its own probability.</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "Work out the slice angle of every word: <i>mat</i> 0.40, <i>floor</i> 0.25, "
                                    "<i>sofa</i> 0.15, <i>bed</i> 0.10, <i>roof</i> 0.10. Do they fill the circle?",
             "answer": "144°, 90°, 54°, 36°, 36°. Yes: 360°.",
             "why": "Each angle is probability × 360°, and since the probabilities add up to 1, the angles add up to 360°."},
            {"kind": "number", "q": "On another spinner, a slice covers 72°. What probability does it stand for?",
             "answer": "0.2.", "why": "72 ÷ 360 = 0.2: turn the rule around."},
            {"kind": "mc", "q": "The spinner landed on <i>floor</i>. What does that tell you?",
             "options": ["floor is the most likely word", "This spin happened to land on floor, which has a 25&nbsp;% chance "
                         "on every spin", "The next spin will also give floor", "mat's probability went down"],
             "answer": "B.", "why": "One spin is one random pick. Spins don't change the probabilities, and the next "
                                   "spin is a fresh random pick."},
            {"kind": "tf", "q": "“Because <i>mat</i> has the biggest slice, sampling always picks <i>mat</i>.”",
             "answer": "False.", "why": "mat is picked most often (40&nbsp;% of spins), but every word with a slice can "
                                       "come up."},
        ],
    },
    {
        "title": "Many spins match the probabilities",
        "segment": (53, 62),
        "figures": [{"t": 61.9, "caption": "10,000 simulated spins: how often each word came up (filled) against its "
                                           "probability (outline)."}],
        "body": [
            """<p>One spin is unpredictable, but many spins are not. Spin 10,000 times and each word comes up
<b>about as often as its probability says</b>: in the video's simulation, <i>mat</i> 40.0&nbsp;%, <i>floor</i> 24.7&nbsp;%,
<i>sofa</i> 15.2&nbsp;%, <i>bed</i> 10.4&nbsp;% and <i>roof</i> 9.8&nbsp;%. The match is close but not exact, and it gets better the
more times you spin. With only a few spins, the counts can be far off.</p>""",
            "{fig0}",
            """<div class="box key"><b class="t">Key idea</b>Over many samples, <b>how often ≈ probability</b>. Expected
count = probability × number of samples.</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "You sample 200 times from the video's distribution. About how many times do you "
                                    "expect each word?",
             "answer": "mat 80, floor 50, sofa 30, bed 20, roof 20.",
             "why": "Probability × 200: 0.40 × 200 = 80, and so on. The five counts add up to 200."},
            {"kind": "number", "q": "In the 10,000 spins, <i>floor</i> came up 24.7&nbsp;% of the time. About how many times "
                                    "is that? How many would its probability predict?",
             "answer": "About 2,470; 2,500 predicted.",
             "why": "0.247 × 10,000 ≈ 2,470 (the exact count was 2,466); 0.25 × 10,000 = 2,500. Close, not exact."},
            {"kind": "short", "q": "Ten samples gave: <i>bed, sofa, mat, mat, bed, mat, bed, sofa, floor, mat</i>. "
                                   "Count each word. Which word is furthest from its probability? Is the sampler broken?",
             "lines": 2,
             "answer": "mat 4, bed 3, sofa 2, floor 1, roof 0. bed: 30&nbsp;% instead of 10&nbsp;%. Not broken.",
             "why": "Ten samples are far too few for the counts to settle; with 10,000 they match closely. (These are "
                    "the ten samples from the video's code, concept 7.)"},
            {"kind": "tf", "q": "“After 10,000 spins, every word came up exactly as often as its probability says.”",
             "answer": "False.", "why": "Close, but not exact: floor 24.7&nbsp;% instead of 25&nbsp;%, roof 9.8&nbsp;% instead of 10&nbsp;%."},
        ],
    },
    {
        "title": "Why sample? Variety",
        "segment": (62, 70),
        "figures": [{"t": 69.9, "caption": "Always taking the biggest slice repeats itself; sampling varies, so the same "
                                           "prompt can give different answers."}],
        "body": [
            """<p>Always choosing the biggest slice (<b>greedy</b> picking) is <b>predictable</b>: <i>mat</i>,
<i>mat</i>, <i>mat</i>, every time. <b>Sampling</b> adds <b>variety</b>. That is why the same prompt can give different
answers: one run says <i>“The cat sat on the mat.”</i>, the next <i>“The cat sat on the sofa.”</i></p>""",
            "{fig0}",
            """<div class="box key"><b class="t">Key idea</b>Greedy = always the most likely option: the same answer every
time. Sampling = a random pick by probability: varied answers, with likely words still the most common.</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "With the video's distribution, what is the probability that one sample is "
                                    "<b>not</b> the greedy choice?",
             "answer": "0.60.", "why": "Greedy always picks mat (0.40), so every other word counts: 1 − 0.40 = 0.60."},
            {"kind": "number", "q": "You run the prompt 20 times, sampling the next word each time. About how many runs "
                                    "end in <i>sofa</i>? And with greedy picking?",
             "answer": "About 3; with greedy, 0.",
             "why": "0.15 × 20 = 3. Greedy picks mat every single time, so sofa never appears."},
            {"kind": "mc", "q": "Why can a chatbot give two different answers to the same prompt?",
             "options": ["The model's probabilities changed between the runs", "It samples from its probabilities",
                         "It forgot the prompt", "Greedy picking is random"],
             "answer": "B.", "why": "Same prompt, same probabilities, but each run makes its own random picks."},
            {"kind": "tf", "q": "“With greedy picking, the same prompt always gives the same next word.”",
             "answer": "True.", "why": "Greedy has no randomness: it always takes the biggest probability."},
        ],
    },
    {
        "title": "Seeds make randomness repeatable",
        "segment": (70, 82),
        "figures": [{"t": 82.2, "caption": "Two generators started with seed 7 give identical choices: floor, bed, "
                                           "sofa, mat, …"}],
        "body": [
            """<p>Computers make randomness with <b>pseudo-random number generators</b>: a fixed recipe that produces
numbers that look random. The recipe starts from a number called the <b>seed</b>. Start one with the <b>same seed</b>
and you get the <b>same sequence</b> of random choices every time. With seed 7 and the video's distribution, it always
begins <i>floor, bed, sofa, mat</i>.</p>""",
            """<p>That keeps experiments <b>reproducible</b>: you, or anyone else, can rerun the code and get exactly
the same “random” results.</p>""",
            "{fig0}",
            """<div class="box key"><b class="t">Key idea</b><b>Same seed → same results.</b> Pseudo-random choices look
random, but the seed fixes the whole sequence.</div>""",
        ],
        "exercises": [
            {"kind": "tf", "q": "“Two generators started with the same seed can give different sequences.”",
             "answer": "False.", "why": "Same seed, same recipe, same sequence, every time."},
            {"kind": "mc", "q": "Why do researchers fix the seed of their experiments?",
             "options": ["To make the model more accurate", "So the experiment can be repeated exactly",
                         "To change the probabilities", "To make sampling faster"],
             "answer": "B.", "why": "The seed doesn't change the distribution; it makes the random picks repeatable."},
            {"kind": "short", "q": "You sample with seed 7; a colleague uses seed 8. Must your first sampled words be "
                                   "the same? Over 10,000 samples, should your word frequencies be similar?",
             "lines": 2,
             "answer": "No; yes.",
             "why": "Different seeds give different sequences (seed 8 starts with mat, not floor). Both still sample the "
                    "same distribution, so over many samples both land near 40&nbsp;%, 25&nbsp;%, 15&nbsp;%, 10&nbsp;%, 10&nbsp;%."},
        ],
    },
    {
        "title": "Sampling in NumPy",
        "segment": (83, 103),
        "figures": [{"t": 89.2, "size": "small", "caption": "The code from the video: a seeded generator samples one "
                                                            "word, then ten."}],
        "body": [
            """<p>In NumPy, create a generator with a seed, then call <code>choice</code> with the options and their
probabilities <code>p</code>. Without <code>size</code> it returns one word; with <code>size=10</code>, ten:</p>""",
            "{fig0}",
            """<pre class="code">words = ["mat", "floor", "sofa", "bed", "roof"]
p = [0.4, 0.25, 0.15, 0.1, 0.1]           # adds up to 1
rng = np.random.default_rng(seed=7)
rng.choice(words, p=p)                    # 'floor'
rng.choice(words, size=10, p=p)           # ten samples</pre>""",
            """<p>You'll see this in episode 1, where the model predicts a distribution, in episode 10, where we sample
from it, and in episode 14.</p>""",
            """<div class="box key"><b class="t">Key idea</b><code>rng = np.random.default_rng(seed)</code>, then
<code>rng.choice(options, size=n, p=probs)</code>. Same seed and same calls → same samples.</div>""",
        ],
        "exercises": [
            {"kind": "mc", "q": "What does <code>rng.choice(words, size=10, p=p)</code> return?",
             "options": ["The 10 most likely words", "Ten words, each picked at random according to p",
                         "Ten copies of 'mat'", "The probabilities of ten words"],
             "answer": "B.", "why": "Ten independent spins of the spinner."},
            {"kind": "tf", "q": "“With <code>p = [0.4, 0.25, 0.15, 0.1, 0.2]</code>, NumPy refuses to sample.”",
             "answer": "True.", "why": "Those add up to 1.1, not 1, so <code>choice</code> raises “ValueError: "
                                      "Probabilities do not sum to 1”."},
            {"kind": "code", "q": "<b>Try it yourself.</b> (a) Run the code. Do the first two printed lines match the "
                                  "video? (b) The third print starts seed 7 again but asks for ten words at once. "
                                  "<b>Predict</b> it before you run it, then explain. (c) What do the last five lines "
                                  "print, and which frame of the video do they match?",
             "code": """import numpy as np

words = ["mat", "floor", "sofa", "bed", "roof"]
p = [0.4, 0.25, 0.15, 0.1, 0.1]

rng = np.random.default_rng(seed=7)
print(rng.choice(words, p=p))              # one word
print(rng.choice(words, size=10, p=p))     # ten more

rng = np.random.default_rng(seed=7)        # start again
print(rng.choice(words, size=10, p=p))     # ten at once

s = np.random.default_rng(seed=0).choice(words, size=10000, p=p)
for w in words:
    print(w, (s == w).mean())""",
             "answer": "(a) floor, then ['bed' 'sofa' 'mat' 'mat' 'bed' 'mat' 'bed' 'sofa' 'floor' 'mat'] · "
                       "(b) ['floor' 'bed' 'sofa' 'mat' 'mat' 'bed' 'mat' 'bed' 'sofa' 'floor'] · (c) 0.3996, "
                       "0.2466, 0.1517, 0.1041, 0.098",
             "why": """(a) Yes: exactly the one word and the ten words shown in the video.
(b) The same seed restarts the same sequence of random picks, so the ten words are the video's sequence from the
start: <i>floor</i> followed by the first nine of the ten. Here it makes no difference whether you draw them one at a
time or ten at once.
(c) mat 0.3996, floor 0.2466, sofa 0.1517, bed 0.1041, roof 0.098: close to p, not exact. These are the video's
10,000-spin frequencies (40.0&nbsp;%, 24.7&nbsp;%, 15.2&nbsp;%, 10.4&nbsp;%, 9.8&nbsp;%), which were simulated with this same seed."""},
        ],
    },
]
