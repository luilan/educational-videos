"""Study guide content for Reinforcement Learning for LLMs, episode 1: Learning From a Grade.

Build:  python framework/study_guide.py rl-llms r01
Every number comes from code/r01_learning_from_a_grade (grade.py, 2026-11-27): a 102,413-parameter tiny GPT
pre-trained on 500 worked additions, then 64 graded samples for each of 1,000 new questions.
"""

LESSON = {
    "series": "Reinforcement Learning for LLMs",
    "label": "Episode 1",
    "title": "Learning From a Grade",
    "tagline": "Reinforcement learning needs a score, not an answer key",
    "duration": "2:25",
    "intro": """<p>There are two ways to train a language model on a task: show it the right answer and make it
<b>copy</b>, or let it <b>try</b> and only tell it how well it did. The second is <b>reinforcement learning</b> (RL).
This lesson sets up the task used through the whole series (two-digit addition with an automatic checker) and a tiny
model that has seen only 500 worked examples. It introduces the RL vocabulary (policy, action, episode, reward), then
looks at what the model's own graded tries contain, which is everything RL has to work with.</p>""",
    "prereq": """<div class="box note"><b>Before you start:</b> How LLMs Work episodes 1 (predict, pick, append),
10 (probabilities), 11 (training) and 12 (building a tiny GPT); Foundations episode 6 (probability and sampling). Code:
<code>code/r01_learning_from_a_grade</code> (Python with PyTorch; runs in about 30 seconds on a CPU).</div>""",
}

CONCEPTS = [
    {
        "title": "Copying versus learning from a grade",
        "segment": (7.5, 37.7),
        "figures": [{"t": 22.8, "caption": "Supervised learning copies a shown answer; reinforcement learning only "
                                           "gets a score back for its own try."},
                    {"t": 37.4, "caption": "Checking is often easier than answering: tests for code, comparing a "
                                           "final number for maths."}],
        "body": [
            """<p>In <b>supervised</b> training (pre-training and supervised fine-tuning), every example comes with the
answer, and the model is trained to make that exact text more likely: it learns to <b>copy</b>. In
<b>reinforcement learning</b> the model writes its own answer and receives only a <b>score</b>, the reward. Nobody
shows it the right text.</p>""",
            "{fig0}",
            """<p>Why give up the answer key? Because for many tasks <b>checking is easier than answering</b>. A test
suite can tell whether code works without being able to write it. A grader can compare a final number without showing
the steps. And since the score only judges the result, the model can be rewarded for solutions that nobody wrote
down: a shorter proof, a new strategy, a better style. This is how today's reasoning models are trained on maths and
code.</p>""",
            "{fig1}",
            """<div class="box key"><b class="t">Key idea</b>Supervised learning needs the right answer; reinforcement
learning needs only a way to score the model's own answers.</div>""",
        ],
        "exercises": [
            {"kind": "mc", "q": "Which training signal is reinforcement learning?",
             "options": ["The text of a correct answer to copy", "A score for an answer the model wrote itself",
                         "A larger training set", "A longer context window"],
             "answer": "B.", "why": "RL learns from rewards for its own outputs, not from shown answers.",
             "key": {"choice": 1}},
            {"kind": "tf", "q": "“Reinforcement learning can only teach a model answers that appear in its training "
                                "data.”",
             "answer": "False.", "why": "The reward judges the result, so any answer that scores well is reinforced, "
                                        "even one nobody wrote down.",
             "key": {"value": False}},
            {"kind": "short", "q": "Give one task, other than arithmetic, where checking an answer is much easier "
                                   "than producing it.",
             "answer": "For example: code (run the tests), a Sudoku (check rows, columns and boxes), a maths problem "
                       "with a known final number, a proof checked by a proof assistant.",
             "why": "Any task with a cheap, reliable checker can provide RL rewards."},
        ],
    },
    {
        "title": "The task, the checker and the starting model",
        "segment": (38.2, 66.3),
        "figures": [{"t": 54.6, "caption": "Two-digit addition, one character per token, ending with a period. The "
                                           "checker: reward 1 if exactly right, else 0."},
                    {"t": 66.0, "caption": "After pre-training on 500 worked examples: all 500 right, but only 14% of "
                                           "1,000 new questions."}],
        "body": [
            """<p>The task for the whole series is adding two numbers between 10 and 99. A question is
<code>47+38=</code>, and the model writes the answer one character at a time, ending with a period:
<code>85.</code>. Every character is a token (13 tokens: the digits, <code>+</code>, <code>=</code> and
<code>.</code>). There are 90 × 90 = 8,100 questions; 1,000 are kept aside as a test set the model never trains on.</p>
<p>The <b>checker</b> is one line of code: the reward is 1 if the answer is exactly the right sum followed by a period,
and 0 otherwise. A missing digit, a wrong digit or a missing period all score 0.</p>""",
            "{fig0}",
            """<p>The model is a tiny GPT (2 transformer blocks, 64-dimensional embeddings, 102,413 parameters). It is
pre-trained on only <b>500 worked examples</b>, for 2,000 steps. It learns all 500 by heart (100% right), but on the
1,000 test questions its first choice is right only <b>14%</b> of the time (139 questions). Five hundred examples are
not enough to learn addition from copying alone.</p>""",
            "{fig1}",
            """<div class="box key"><b class="t">Key idea</b>A reward needs a checker, not a teacher: here it is a single
comparison with the right sum.</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "What reward does the checker give the answer <code>85</code> (no period) to "
                                    "<code>47+38=</code>?",
             "answer": "0.", "why": "The answer must be exactly “85.”; without the period it scores 0.",
             "key": {"parts": [{"label": None, "value": 0, "tol": 0, "unit": None}]}},
            {"kind": "number", "q": "How many tokens does the model write for the answer to <code>63+58=</code>, "
                                    "including the period?",
             "answer": "4.", "why": "63 + 58 = 121, written as “121.”: 3 digits and a period.",
             "key": {"parts": [{"label": None, "value": 4, "tol": 0, "unit": None}]}},
            {"kind": "short", "q": "The model gets 100% of its 500 training examples right and 14% of new questions. "
                                   "What does that gap tell you?",
             "answer": "It memorised the 500 examples instead of learning how to add, so it does not generalise to "
                       "questions it has not seen.",
             "why": "500 of 8,100 questions is too little data to learn the rule by copying."},
        ],
    },
    {
        "title": "The model as a policy",
        "segment": (66.8, 99.1),
        "figures": [{"t": 84.2, "caption": "For 47+38=, the policy puts 83% on a 7 as the first digit and 17% on the "
                                           "right 8: it usually forgets to carry."},
                    {"t": 98.8, "caption": "64 tries at 27+15=: 48 (33 times, reward 0), 42 (24 times, reward 1), 49 "
                                           "(7 times, reward 0)."}],
        "body": [
            """<p>Reinforcement learning has its own vocabulary, and for language models it maps neatly:</p>
<ul><li>the <b>policy</b> is the model: given the text so far, a probability for every next token;</li>
<li>an <b>action</b> is one token it writes (here, one character);</li>
<li>an <b>episode</b> is one whole answer, from the question to the period;</li>
<li>the <b>reward</b> is the score, and it arrives only at the end of the episode.</li></ul>
<p>For <code>47+38=</code>, the policy puts 83% on <code>7</code> as the first digit and 17% on <code>8</code>.
The right answer is 85: the 8 needs the 1 carried from 7 + 8 = 15. The model has learned to add the tens digits but
usually forgets the carry.</p>""",
            "{fig0}",
            """<p>Because the policy gives probabilities, it can be <b>sampled</b> many times. Sixty-four tries at
<code>27+15=</code> gave 48 thirty-three times, the right answer 42 twenty-four times and 49 seven times. Each try is
an episode with reward 0 or 1. The right answer is not the model's favourite, but it is in there.</p>""",
            "{fig1}",
            """<div class="box key"><b class="t">Key idea</b>A language model is a policy: each token is an action, an
answer is an episode, and the reward comes once, at the end.</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "In the episode <code>27+15=42.</code>, how many actions did the model take?",
             "answer": "3.", "why": "It wrote “4”, “2” and “.”; the question is given, not written by the model.",
             "key": {"parts": [{"label": None, "value": 3, "tol": 0, "unit": None}]}},
            {"kind": "number", "q": "Of the 64 tries at <code>27+15=</code>, what percentage were right? (Round to "
                                    "a whole number.)",
             "answer": "38%.", "why": "24 / 64 = 0.375.",
             "key": {"parts": [{"label": None, "value": 38, "tol": 1, "unit": "%"}]}},
            {"kind": "mc", "q": "When does the model receive its reward in this task?",
             "options": ["After every character", "After the first digit", "Once, after the whole answer",
                         "Never; it is shown the answer instead"],
             "answer": "C.", "why": "The checker can only grade a finished answer, so the reward arrives at the end "
                                    "of the episode.",
             "key": {"choice": 2}},
        ],
    },
    {
        "title": "What reinforcement learning works with",
        "segment": (99.6, 137.7),
        "figures": [{"t": 115.1, "caption": "1,000 questions × 64 tries: 14% of all answers right; 644 questions never "
                                            "right; 217 where the first choice is wrong but some tries are right."},
                    {"t": 129.2, "caption": "The idea of RL: make answers that scored 1 more likely, those that scored "
                                            "0 less likely, and repeat (illustration)."},
                    {"t": 137.4, "caption": "The reward is one line; sampling writes up to four characters, one at a "
                                            "time."}],
        "body": [
            """<p>Sampling 64 answers for each of the 1,000 test questions and grading all 64,000 shows what the model
can already do. Overall, <b>14%</b> of the answers are right. The questions are very uneven: 644 are never right in 64
tries, 184 are right in 1–25% of tries, and only 19 are right every time. In total 356 questions get at least one right
answer. On <b>217</b> of them the model's first choice (greedy decoding) is wrong, yet some of its tries are right.</p>""",
            "{fig0}",
            """<p>That is the raw material of reinforcement learning. The good answers are sometimes already among the
samples. RL makes the answers that scored 1 more likely and the ones that scored 0 less likely, samples again and
repeats. It never needs the answer key, only the grades. In the next episode we do this for real with
<b>REINFORCE</b>, and also see what happens to the 644 questions the model never gets right.</p>""",
            '<div class="figrow">{fig1}{fig2}</div>',
            """<div class="box key"><b class="t">Key idea</b>RL can only reinforce what the model sometimes does;
the graded samples show how much it has to work with.</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "How many of the 1,000 test questions got at least one right answer in 64 tries?",
             "answer": "356.", "why": "1,000 − 644 questions that were never right.",
             "key": {"parts": [{"label": None, "value": 356, "tol": 0, "unit": None}]}},
            {"kind": "tf", "q": "“If a model never samples the right answer to a question, the reward for that "
                                "question is always 0 and gives no direct signal toward the right answer.”",
             "answer": "True.", "why": "With every try scoring 0 there is nothing better to reinforce on that "
                                       "question; progress there can only come from what is learned on others.",
             "key": {"value": True}},
            {"kind": "code", "q": "<b>Try it yourself.</b> Grade 64 sampled answers to one question.",
             "code": """_, answers, _, qs = sample(model, [(27, 15)], n=64)
rewards = [reward(a, b, ans) for (a, b), ans in zip(qs, answers)]
print(collections.Counter(answers), sum(rewards) / 64)""",
             "answer": "grade.py does this for all 1,000 test questions and writes results.json.",
             "why": "Sampling gives many episodes per question; the checker turns each into a 0 or 1."},
        ],
    },
]
