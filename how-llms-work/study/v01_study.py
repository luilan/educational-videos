"""Study guide content for How LLMs Work, episode 1: What is an LLM?

Build:  python framework/study_guide.py how-llms-work v01
Figures: {"t": seconds into the rendered video, "caption": ...}; placed in body blocks with {fig0}, {fig1}, ...
Exercise kinds: mc (options), tf, short, number, order, code. Every exercise has an answer and a why.
"""

LESSON = {
    "series": "How LLMs Work",
    "label": "Episode 1",
    "title": "What is an LLM?",
    "tagline": "How a language model writes, one word at a time",
    "duration": "2:05",
    "intro": """<p>This lesson has one big idea: a large language model does <b>one simple thing, over and over</b>.
It predicts what comes next, picks something, adds it to the text, and repeats. Everything else in the series is about
how the prediction works on the inside.</p>""",
    "prereq": """<div class="box note"><b>Before you start:</b> no maths is needed for this lesson. Concept 5 reads a short
Python function; if Python is new to you, Foundations F11 (NumPy in Three Minutes) helps. A note on words: the video
says the model predicts the next <i>word</i>. Strictly it predicts the next <i>token</i>, a piece of a word, which is
the topic of episode 2. In this guide, "word" and "token" mean the same thing.</div>""",
}

CONCEPTS = [
    {
        "title": "One job: predict what comes next",
        "segment": (8, 30),
        "figures": [{"t": 29.6, "caption": "The text so far goes into the model. Its only job is to fill the empty slot: "
                                           "what comes next?"}],
        "body": [
            """<p>You type a question and a chatbot writes an answer. It can look like the model plans the whole reply,
or like someone is typing. What actually happens is simpler: the model is given <b>all the text so far</b> and answers
a single question, <b>what comes next?</b></p>""",
            "{fig0}",
            """<div class="box key"><b class="t">Key idea</b>An LLM (large language model) maps <b>the text so far</b> to
<b>a guess about the next word</b>. Answering questions, writing code and translating are all this one operation,
repeated.</div>""",
        ],
        "exercises": [
            {"kind": "mc", "q": "You give an LLM the text <i>“The cat sat on the”</i>. What does it produce in one step?",
             "options": ["A complete paragraph continuing the story", "A guess about the single next word",
                         "A summary of the sentence", "A list of grammar corrections"],
             "answer": "B.", "why": "One step = one prediction of the next word. Longer text comes from repeating the step.", "key": {'choice': 1}},
            {"kind": "tf", "q": "“An LLM writes its whole reply at once, as a single output.”",
             "answer": "False.", "why": "The reply is produced one word at a time, each word predicted from the text so far.", "key": {'value': False}},
            {"kind": "short", "q": "The input is <i>“Once upon a”</i>. In one sentence, describe the model's job. "
                                   "Which word would you expect it to rate as most likely?",
             "lines": 2,
             "answer": "Predict the next word.", "why": "A well-trained model would rate <i>“time”</i> as by far the most likely."},
        ],
    },
    {
        "title": "A probability for every word",
        "segment": (30, 42),
        "figures": [{"t": 41.4, "caption": "The model's output for “The cat sat on the”: a probability for every word it "
                                           "knows. Only the top few (and banana) are shown."}],
        "body": [
            """<p>The model does not output just one word. It outputs a <b>probability for every word in its vocabulary</b>,
tens of thousands of numbers. A probability is a number between 0 and 1 (0 % to 100 %) saying how likely each word is
to come next.</p>""",
            "{fig0}",
            """<div class="box key"><b class="t">Key idea</b>The output is a <b>probability distribution</b>: every value is
between 0 and 1, and all of them together <b>add up to exactly 1</b> (100 %). Likely words get large values (<i>mat</i>
41 %); unlikely words get tiny values (<i>banana</i> 0.01 %), but not zero.</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "In the chart, <i>mat</i> 41 %, <i>floor</i> 22 %, <i>sofa</i> 12 %, <i>bed</i> 9 % and "
                                    "<i>roof</i> 5 %. How much probability is left for all the other words together?",
             "answer": "11 %.", "why": "41 + 22 + 12 + 9 + 5 = 89 %, and the whole distribution adds up to 100 %.", "key": {'parts': [{'label': None, 'value': 11, 'tol': 0.5, 'unit': '%'}]}},
            {"kind": "mc", "q": "Which list could be a probability distribution over three words?",
             "options": ["0.5, 0.3, 0.3", "0.6, 0.4, 0.0", "0.7, −0.1, 0.4", "0.2, 0.2, 0.2"],
             "answer": "B.", "why": "A adds up to 1.1; C has a negative value; D adds up to only 0.6. "
                                   "B is fine: a 0 is allowed, and the total is exactly 1.", "key": {'choice': 1}},
            {"kind": "tf", "q": "“Because <i>banana</i> is such an unlikely next word, the model gives it a probability "
                                "of exactly 0.”",
             "answer": "False.", "why": "It gets a tiny probability (0.01 % in the video), not zero. Every word in the "
                                        "vocabulary gets some probability.", "key": {'value': False}},
            {"kind": "number", "q": "A model's vocabulary has 50,000 words. How many probabilities does it produce each "
                                    "time it predicts the next word?",
             "answer": "50,000.", "why": "One probability per word in the vocabulary, every single step.", "key": {'parts': [{'label': None, 'value': 50000, 'tol': 0.5, 'unit': None}]}},
        ],
    },
    {
        "title": "Pick one word and append it",
        "segment": (42, 48),
        "figures": [{"t": 47.0, "caption": "“mat” has been picked and added to the end of the text."}],
        "body": [
            """<p>A distribution is not yet text. We <b>pick one word</b> from it, usually one of the likely ones, and
<b>append</b> it to the end of the text.</p>""",
            "{fig0}",
            """<p>There are two simple ways to pick. <b>Greedy</b>: always take the most likely word. <b>Sampling</b>:
pick at random, so that each word is chosen as often as its probability says (41 % of the time <i>mat</i>, 22 % of the
time <i>floor</i>, …). Sampling gives variety; episode 10 covers the details.</p>""",
            """<div class="box key"><b class="t">Key idea</b>Prediction gives <b>probabilities</b>; <b>picking</b> turns them
into one actual word, which is then <b>appended</b> to the text.</div>""",
        ],
        "exercises": [
            {"kind": "short", "q": "Using the chart from concept 2, which word does <b>greedy</b> picking choose?",
             "answer": "mat.", "why": "Greedy always takes the highest probability, and mat has 41 %."},
            {"kind": "number", "q": "You <b>sample</b> from that same distribution 100 separate times. About how many "
                                    "times would you expect <i>floor</i> to be picked?",
             "answer": "About 22.", "why": "Sampling picks each word in proportion to its probability: 22 % of 100 ≈ 22 "
                                            "(the exact count varies from run to run).", "key": {'parts': [{'label': None, 'value': 22, 'tol': 0.5, 'unit': None}]}},
            {"kind": "short", "q": "The text is <i>“The cat sat on the”</i> and the picked word is <i>mat</i>. "
                                   "Write the new text.",
             "answer": "“The cat sat on the mat”.", "why": "Appending adds the word at the end; nothing else changes."},
            {"kind": "mc", "q": "Why pick a word that is not the single most likely one?",
             "options": ["Because the model is not accurate enough", "To add variety, so the text is not always the same "
                         "or repetitive", "Because the probabilities must add up to 1", "To make generation faster"],
             "answer": "B.", "why": "Sampling makes the output varied. The model's probabilities are the same either way.", "key": {'choice': 1}},
        ],
    },
    {
        "title": "The loop: predict, pick, append, repeat",
        "segment": (48, 63),
        "figures": [{"t": 62.7, "caption": "The whole text goes back in for every new word. The loop that generates "
                                           "all text: predict → pick → append → repeat."}],
        "body": [
            """<p>After appending, we <b>feed the whole text back in</b> and predict again: <i>“The cat sat on the mat”</i> →
maybe a period. Then another word, and another. Each new word costs <b>one full run of the model</b>, and the model reads
its own earlier words as part of its input.</p>""",
            "{fig0}",
            """<p>The loop stops when it reaches a <b>length limit</b>, or when the model predicts a special
<b>END</b> token that means “I'm done”.</p>""",
            """<div class="box key"><b class="t">Key idea</b>All text generation is one loop:
<b>predict → pick → append → repeat</b>, until a length limit or an END token.</div>""",
        ],
        "exercises": [
            {"kind": "order", "q": "Put these steps in the right order: <i>append · repeat · predict · pick</i>.",
             "answer": "predict → pick → append → repeat.", "why": "Predict probabilities, pick a word, append it, go again.", "key": {'items': ['predict', 'pick', 'append', 'repeat']}},
            {"kind": "number", "q": "Starting from <i>“The cat sat on the”</i>, the model generates <i>mat</i>, <i>.</i>, "
                                    "<i>It</i>, <i>purred</i>. (a) How many times was the model run? (b) What text was its "
                                    "input on the third run?",
             "lines": 2,
             "answer": "(a) 4 times. (b) “The cat sat on the mat .”",
             "why": "One run per new word. The third run sees the original text plus the first two generated words.", "key": {'parts': [{'label': None, 'value': 4, 'tol': 0.5, 'unit': None}]}},
            {"kind": "tf", "q": "“On later steps, the model reads the words it generated itself as part of its input.”",
             "answer": "True.", "why": "The whole text, including everything generated so far, goes back in each time.", "key": {'value': True}},
            {"kind": "short", "q": "Name the two ways the loop can stop.",
             "answer": "A length limit, or the model producing the END token.",
             "why": "In the code of concept 5 these are <code>max_new</code> and <code>next_tok == END</code>."},
        ],
    },
    {
        "title": "The loop in code",
        "segment": (63, 76),
        "figures": [{"t": 69.6, "size": "small", "caption": "The generate function from the video: the whole loop in eight lines."}],
        "body": [
            """<p>The loop fits in a few lines of Python. <code>tokens</code> is the text so far (a list of words),
<code>model</code> returns the probabilities, and <code>sample</code> picks one word.</p>""",
            "{fig0}",
            """<pre class="code">def generate(model, tokens, max_new=50):
    for _ in range(max_new):
        probs = model(tokens)      # a probability per word
        next_tok = sample(probs)   # pick one
        tokens.append(next_tok)    # add it to the text
        if next_tok == END:        # model says it's done
            break
    return tokens</pre>""",
            """<div class="box key"><b class="t">Key idea</b>One loop iteration = one new word. <code>max_new</code> is the
length limit; <code>END</code> is the model's own stop signal.</div>""",
        ],
        "exercises": [
            {"kind": "short", "q": "What does <code>max_new</code> control?",
             "answer": "The maximum number of new words the loop can add.",
             "why": "The <code>for</code> loop runs at most <code>max_new</code> times, adding one word each time."},
            {"kind": "mc", "q": "Suppose the model's very first pick is <code>END</code>. Read the code carefully: what "
                                "does <code>generate</code> return?",
             "options": ["The original tokens, unchanged", "The original tokens with END added at the end",
                         "An empty list", "An error"],
             "answer": "B.", "why": "<code>tokens.append(next_tok)</code> runs <i>before</i> the END check, so END is "
                                   "appended and then the loop breaks. Real code often strips it off afterwards.", "key": {'choice': 1}},
            {"kind": "code", "q": "<b>Try it yourself.</b> Below is a toy “model” that looks only at the last word, and a "
                                  "version of <code>generate</code> that takes the picking function as an argument. "
                                  "(a) Write <code>greedy(probs)</code>, which returns the most likely word, and "
                                  "<code>sample(probs)</code>, which picks at random using the probabilities (hint: "
                                  "<code>random.choices</code>). (b) What does <code>generate(model, [\"the\"], greedy, "
                                  "max_new=8)</code> return? Why does it never reach the period? (c) With "
                                  "<code>sample</code>, what is the probability that the output is exactly "
                                  "<i>the mat . END</i>?",
             "code": """import random

END = "END"
NEXT = {
    "the": {"cat": 0.6, "mat": 0.4},
    "cat": {"sat": 1.0},
    "sat": {"on": 1.0},
    "on":  {"the": 1.0},
    "mat": {".": 1.0},
    ".":   {END: 1.0},
}

def model(tokens):
    return NEXT[tokens[-1]]          # probabilities for the next word

def generate(model, tokens, pick, max_new=50):
    tokens = list(tokens)
    for _ in range(max_new):
        next_tok = pick(model(tokens))
        tokens.append(next_tok)
        if next_tok == END:
            break
    return tokens""",
             "answer": "(b) ['the', 'cat', 'sat', 'on', 'the', 'cat', 'sat', 'on', 'the'] · (c) 0.4",
             "why": """(a) <code>def greedy(probs): return max(probs, key=probs.get)</code> and
<code>def sample(probs): words = list(probs); return random.choices(words, weights=[probs[w] for w in words])[0]</code>.
(b) After <i>the</i>, greedy always picks <i>cat</i> (0.6 beats 0.4), so it cycles <i>the cat sat on the …</i> and only
the <code>max_new</code> limit stops it. This is why pure greedy picking can get stuck repeating itself.
(c) The first pick must be <i>mat</i> (0.4); after that <i>.</i> and END are certain (1.0), so 0.4 × 1 × 1 = 0.4."""},
        ],
    },
    {
        "title": "Inside the model: a function made of numbers",
        "segment": (76, 117),
        "figures": [{"t": 90.3, "caption": "Everything interesting hides inside model: a huge pile of numbers that "
                                           "somehow knows “mat” is likely."},
                    {"t": 116.3, "caption": "The road ahead: how the series opens up the model, episode by episode."}],
        "body": [
            """<p>In the code, all the intelligence is hidden in one call: <code>model(tokens)</code>. Inside it there are no
hand-written rules like “after <i>sat on the</i>, say <i>mat</i>”. There is a huge collection of numbers, called
<b>parameters</b> or <b>weights</b>, and the model computes the probabilities from them. The numbers are
<b>learned from data</b> during training (episode 11).</p>""",
            """<p>The rest of the series opens the function up, in the order the data flows through it: text is split into
<b>tokens</b> (ep 2), each token becomes a vector, an <b>embedding</b> (ep 3), plus its <b>position</b> (ep 4).
<b>Attention</b> lets words look at each other (eps 5–7) and the <b>MLP</b> processes each word (ep 8). Together these form a
<b>transformer block</b>, stacked many times (ep 9). Finally vectors become <b>probabilities</b> again (ep 10).</p>""",
            '<div class="figrow">{fig0}{fig1}</div>',
            """<div class="box key"><b class="t">Key idea</b><code>model</code> is a function built from millions (or
billions) of learned numbers. Data flows: <b>tokens → embeddings (+ position) → transformer blocks (attention + MLP)
→ probabilities</b>.</div>""",
        ],
        "exercises": [
            {"kind": "tf", "q": "“Inside an LLM there is a list of hand-written rules, such as ‘after <i>sat on the</i>, "
                                "say <i>mat</i>’.”",
             "answer": "False.", "why": "It is a function of learned numbers (parameters). Nobody writes the rules.", "key": {'value': False}},
            {"kind": "order", "q": "Put these stages in the order data flows through the model: <i>probabilities · "
                                   "tokens · transformer blocks (attention + MLP) · embeddings</i>.",
             "answer": "tokens → embeddings → transformer blocks → probabilities.",
             "why": "Text is split into tokens, tokens become vectors, blocks process them, and the result is turned back "
                    "into a probability for every word.", "key": {'items': ['tokens', 'embeddings', 'transformer blocks (attention + MLP)', 'probabilities']}},
            {"kind": "mc", "q": "Where do the model's numbers come from?",
             "options": ["Engineers write them by hand", "They are random and never change",
                         "They are learned from lots of text during training", "They are copied from a dictionary"],
             "answer": "C.", "why": "They start random and training adjusts them to predict text better (episode 11).", "key": {'choice': 2}},
        ],
    },
]
