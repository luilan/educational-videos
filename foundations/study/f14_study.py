"""Study guide content for How LLMs Work · Foundations, F14: Train vs Validation Data.

Build:  python framework/study_guide.py foundations f14 --video <TrainValVideo.mp4>
Figures: {"t": seconds into the rendered video, "caption": ...}; placed in body blocks with {fig0}, {fig1}, ...
Exercise kinds: mc (options), tf, short, number, order, code. Every exercise has an answer and a why.
All numbers and code outputs were checked by running Python; the tiny GPT numbers come from
how-llms-work/tiny_gpt/input.txt and training_log.json.
"""

LESSON = {
    "series": "How LLMs Work · Foundations",
    "label": "F14",
    "title": "Train vs Validation Data",
    "tagline": "How we know a model really learned",
    "duration": "1:53",
    "intro": """<p>This lesson answers one question: how do we know a model has really <b>learned</b>, and not just
memorized? Before training we split the data, never train on one slice, the <b>validation set</b>, and measure the loss
there too. Comparing the two losses tells us whether the model is learning or <b>overfitting</b>.</p>""",
    "prereq": """<div class="box note"><b>Before you start:</b> this lesson builds on F12 (training nudges the parameters)
and F13 (the training loop). All you need about the <b>loss</b>: it measures how wrong the model's predictions are, so
lower is better. The two loss curves appear in episode 11 (Training: Learning from Mistakes), and this exact split in
the tiny GPT of episode 12 (Build a Tiny GPT).</div>""",
}

CONCEPTS = [
    {
        "title": "A low training loss can fool us",
        "segment": (8, 26),
        "figures": [{"t": 16.7, "caption": "Episode 11's real run. Why two curves?"},
                    {"t": 25.4, "caption": "A memorizer (illustrative)."}],
        "body": [
            """<p>Episode 11 showed two loss curves, one for <b>training</b> (teal) and one for <b>validation</b>
(yellow). Why two? Because a low training loss, on its own, can fool us. A model could simply <b>memorize</b> its
training text, word for word. It would score perfectly on that text, a training loss of about 0, and be useless on
anything new.</p>""",
            '<div class="figrow">{fig0}{fig1}</div>',
            """<div class="box key"><b class="t">Key idea</b>The loss on the training data shows how well the model
<b>fits what it has seen</b>, not whether it has <b>learned</b>. To judge learning, measure on text it has never
seen.</div>""",
        ],
        "exercises": [
            {"kind": "mc", "q": "A student scores 100 % on a practice exam after studying the answer key of that exact "
                                "exam. What does the score tell you?",
             "options": ["The student has mastered the subject", "Very little: they may just have memorized the "
                         "answers, so test them on new questions", "The exam was too easy for everyone",
                         "They will score 100 % on any exam"],
             "answer": "B.", "why": "The same trap as a low training loss: a perfect score on seen material can be "
                                   "pure memory.", "key": {'choice': 1}},
            {"kind": "tf", "q": "“A training loss close to 0 proves that the model will do well on new text.”",
             "answer": "False.", "why": "A memorizer also gets a training loss near 0 and is useless on new text.", "key": {'value': False}},
            {"kind": "mc", "q": "Which measurement would reveal that the video's memorizer has not really learned?",
             "options": ["Its loss on the training text", "Its loss on text it was never trained on",
                         "How long training took", "How many parameters it has"],
             "answer": "B.", "why": "On unseen text the memorizer's answers are gibberish, so that loss would be high.", "key": {'choice': 1}},
            {"kind": "short", "q": "In one sentence: what is the difference between memorizing and learning?",
             "answer": "Memorizing reproduces the seen text; learning finds patterns that also work on new text.",
             "why": "Only learning makes the model useful on text it has never seen, which is what we want."},
        ],
    },
    {
        "title": "Split the data before training",
        "segment": (26, 50),
        "figures": [{"t": 35.9, "caption": "Most of the data is the training set; a slice we never train on is "
                                           "the validation set."},
                    {"t": 49.1, "caption": "The tiny GPT's split: 1,003,854 characters for training, 111,540 only "
                                           "for measuring."}],
        "body": [
            """<p>So <b>before</b> training, we split the data. Most of it, the <b>training set</b>, is used for
learning. A slice we never train on, the <b>validation set</b>, is kept aside to test with.</p>""",
            '<div class="figrow">{fig0}{fig1}</div>',
            """<p>For the tiny GPT, the data is the Shakespeare text: 1,115,394 characters. The first 90 %,
<b>1,003,854</b> characters (about a million), was for training. The last 10 %, <b>111,540</b> characters, was only ever
used to measure.</p>""",
            """<div class="box key"><b class="t">Key idea</b>Split first. The <b>training set</b> is for learning; the
<b>validation set</b> is never trained on, only measured.</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "Check the tiny GPT's split. What is 90 % of 1,115,394, rounded down to a whole "
                                    "character? How many characters are left for validation?",
             "answer": "1,003,854 and 111,540.",
             "why": "0.9 × 1,115,394 = 1,003,854.6, rounded down to 1,003,854; then 1,115,394 − 1,003,854 = 111,540.", "key": {'parts': [{'label': 'training', 'value': 1003854, 'tol': 0.5, 'unit': None}, {'label': 'validation', 'value': 111540, 'tol': 0.5, 'unit': None}]}},
            {"kind": "number", "q": "A dataset has 50,000 sentences. With the same 90 / 10 split, how many are for "
                                    "training and how many for validation?",
             "answer": "45,000 and 5,000.", "why": "0.9 × 50,000 = 45,000; the remaining 10 % is 5,000.", "key": {'parts': [{'label': 'training', 'value': 45000, 'tol': 0.5, 'unit': None}, {'label': 'validation', 'value': 5000, 'tol': 0.5, 'unit': None}]}},
            {"kind": "tf", "q": "“It is fine to train on the validation set as well, as long as you also measure on "
                                "it.”",
             "answer": "False.", "why": "Then it is no longer text the model has never seen, and its loss can be fooled "
                                        "by memorizing, just like the training loss.", "key": {'value': False}},
        ],
    },
    {
        "title": "Two losses, and the gap between them",
        "segment": (50, 73),
        "figures": [{"t": 60.2, "caption": "Both losses, measured during the whole run. (The training curve is a "
                                           "50-step average.)"},
                    {"t": 72.4, "caption": "The last 2,500 steps: training ends at about 1.33, validation at "
                                           "about 1.59."}],
        "body": [
            """<p>During training we measure the loss on <b>both</b> sets. The <b>training loss</b> says how well the
model fits what it has seen. The <b>validation loss</b> says how well it does on text it has <b>never seen</b>, which
is what we really care about.</p>""",
            '<div class="figrow">{fig0}{fig1}</div>',
            """<p>The tiny GPT ended at about <b>1.33</b> on training text and about <b>1.59</b> on validation text.
That <b>gap</b> is normal: a model usually does a little better on what it studied.</p>""",
            """<div class="box key"><b class="t">Key idea</b>Training loss = fit to seen text. Validation loss =
performance on unseen text. A small, steady gap between them is normal.</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "How big is the gap between the tiny GPT's final losses?",
             "answer": "0.26.", "why": "1.59 − 1.33 = 0.26.", "key": {'parts': [{'label': None, 'value': 0.26, 'tol': 0.005, 'unit': None}]}},
            {"kind": "mc", "q": "Which number best predicts how the tiny GPT will do on a new piece of Shakespeare?",
             "options": ["The training loss, 1.33", "The validation loss, 1.59", "Their average, 1.46",
                         "Neither: only a loss of 0 means anything"],
             "answer": "B.", "why": "The validation loss is measured on text the model never trained on, like the new "
                                   "piece.", "key": {'choice': 1}},
            {"kind": "tf", "q": "“The validation loss is a little higher than the training loss, so the tiny GPT is "
                                "broken.”",
             "answer": "False.", "why": "A small gap is normal: a model usually does a little better on what it "
                                        "studied.", "key": {'value': False}},
            {"kind": "number", "q": "The tiny GPT's validation loss was 1.79 at step 1,000 and 1.59 at step 5,000. By "
                                    "how much did it fall? What does a falling validation loss tell you?",
             "lines": 2,
             "answer": "0.20; the model was still getting better on unseen text.",
             "why": "1.79 − 1.59 = 0.20. Memorizing cannot lower the loss on unseen text, so the improvement is real "
                    "learning.", "key": {'parts': [{'label': None, 'value': 0.2, 'tol': 0.005, 'unit': None}]}},
        ],
    },
    {
        "title": "Overfitting",
        "segment": (73, 85),
        "figures": [{"t": 84.6, "caption": "Overfitting (illustrative): the gap keeps growing, and the validation "
                                           "loss stops falling and rises. Stop at its lowest point."}],
        "body": [
            """<p>Watch the gap over time. If it <b>keeps growing</b> while the validation loss <b>stops falling</b>, or
even <b>rises</b>, the model is <b>overfitting</b>: memorizing instead of learning. The training loss still improves,
but the improvement no longer carries over to new text. That is the time to <b>stop</b>, or to find <b>more
data</b>.</p>""",
            "{fig0}",
            """<div class="box key"><b class="t">Key idea</b>Overfitting: training loss still falling, validation loss
flat or rising, gap growing. Stop at the lowest validation loss, or get more data.</div>""",
        ],
        "exercises": [
            {"kind": "mc", "q": "Which pattern shows overfitting?",
             "options": ["Both losses falling together, with a small, steady gap",
                         "Training loss falling, validation loss rising",
                         "Both losses flat and high from the start",
                         "Validation a little above training, both still falling"],
             "answer": "B.", "why": "The model keeps fitting its training text better while getting worse on unseen "
                                   "text: memorizing, not learning.", "key": {'choice': 1}},
            {"kind": "number", "q": "A run logs these losses. Step 1,000: train 2.05, val 2.10 · 2,000: 1.70, 1.85 · "
                                    "3,000: 1.50, 1.78 · 4,000: 1.35, 1.81 · 5,000: 1.20, 1.90. (a) Which checkpoint "
                                    "would you keep? (b) What is the gap at step 1,000 and at step 5,000?",
             "lines": 2,
             "answer": "(a) Step 3,000. (b) 0.05 and 0.70.",
             "why": "The validation loss is lowest (1.78) at step 3,000 and rises after it, while the gap grows from "
                    "2.10 − 2.05 = 0.05 to 1.90 − 1.20 = 0.70: overfitting.", "key": {'parts': [{'label': '(a) step', 'value': 3000, 'tol': 0.5, 'unit': None}, {'label': '(b) gap at 1,000', 'value': 0.05, 'tol': 0.005, 'unit': None}, {'label': '(b) gap at 5,000', 'value': 0.7, 'tol': 0.005, 'unit': None}]}},
            {"kind": "tf", "q": "“When a model overfits, its training loss goes up.”",
             "answer": "False.", "why": "The training loss keeps going down. It is the validation loss that stalls "
                                        "or rises.", "key": {'value': False}},
            {"kind": "short", "q": "Name the two remedies the video gives for overfitting.",
             "answer": "Stop training (at the lowest validation loss), or find more data.",
             "why": "Stopping keeps the best model; more data makes memorizing harder than learning."},
        ],
    },
    {
        "title": "The test set: one final, honest score",
        "segment": (85, 92),
        "figures": [{"t": 91.9, "caption": "A third slice, the test set, stays locked until the very end."}],
        "body": [
            """<p>Careful projects keep a <b>third slice</b>, the <b>test set</b>, untouched until the very end. Why,
if there is already a validation set? Because we <i>use</i> the validation loss to make decisions, such as when to
stop. Every decision tunes the model a little to that slice. The test set, used once at the end, gives <b>one final,
honest score</b>.</p>""",
            "{fig0}",
            """<div class="box key"><b class="t">Key idea</b>Train on the <b>training set</b>, steer with the
<b>validation set</b>, and score once on the <b>test set</b>, at the very end.</div>""",
        ],
        "exercises": [
            {"kind": "mc", "q": "Which set do you use to decide when to stop training?",
             "options": ["The training set", "The validation set", "The test set", "All three, averaged"],
             "answer": "B.", "why": "Stopping is a decision made during the project; the test set must stay untouched "
                                   "until the end.", "key": {'choice': 1}},
            {"kind": "tf", "q": "“It is fine to check the test score after every training step, to see how things are "
                                "going.”",
             "answer": "False.", "why": "Then the test set steers your decisions like a validation set, and its score "
                                        "is no longer an honest final check.", "key": {'value': False}},
            {"kind": "order", "q": "Put in order: <i>report the test score · split the data three ways · keep the "
                                   "checkpoint with the lowest validation loss · train while watching the validation "
                                   "loss</i>.",
             "answer": "split → train and watch → keep the best checkpoint → report the test score.",
             "why": "The test set is touched only once, at the very end.", "key": {'items': ['split the data three ways', 'train while watching the validation loss', 'keep the checkpoint with the lowest validation loss', 'report the test score']}},
        ],
    },
    {
        "title": "The split in code",
        "segment": (92, 99),
        "figures": [{"t": 98.1, "size": "small", "caption": "The split from the tiny GPT: the first 90 % for "
                                                           "training, the rest for validation."}],
        "body": [
            """<p>In the tiny GPT, the split is two lines. <code>split</code> is the position 90 % of the way through
the data; <code>data[:split]</code> is everything before it (training), <code>data[split:]</code> everything from it on
(validation). So the validation set is the <b>last</b> 10 % of the text, one continuous piece.</p>""",
            "{fig0}",
            """<pre class="code">data = torch.tensor(encode(text))
split = int(0.9 * len(data))
train_data = data[:split]       # 1,003,854 characters
val_data = data[split:]         # 111,540 characters</pre>""",
            """<div class="box key"><b class="t">Key idea</b><code>data[:split]</code> and <code>data[split:]</code>: every
character lands in exactly one of the two sets, and none is shared.</div>""",
        ],
        "exercises": [
            {"kind": "mc", "q": "Why is <code>int(…)</code> needed in <code>split = int(0.9 * len(data))</code>?",
             "options": ["To round 0.9 up to 1", "A slice position must be a whole number, and 0.9 × 1,115,394 = "
                         "1,003,854.6 is not", "To shuffle the data first", "To turn characters into numbers"],
             "answer": "B.", "why": "<code>int</code> drops the .6, giving 1,003,854. Turning characters into numbers "
                                   "is <code>encode</code>'s job.", "key": {'choice': 1}},
            {"kind": "code", "q": "<b>Try it yourself.</b> The code below does the same split on the first line of "
                                  "the Shakespeare text, one token per character. (a) What does it print? (b) Change "
                                  "it to a three-way split: the first 80 % for training, the next 10 % for validation, "
                                  "the last 10 % for testing. How long is each part? (c) With the same 80 / 10 / 10 "
                                  "code, how many characters would each part get from the full 1,115,394?",
             "code": """text = "First Citizen: Before we proceed any further, hear me speak."
data = list(text)                    # one token per character, like the tiny GPT
split = int(0.9 * len(data))
train_data = data[:split]
val_data = data[split:]
print(len(data), split, len(val_data))
print("".join(val_data))""",
             "answer": "(a) 60 54 6, then speak. · (b) 48, 6, 6 · (c) 892,315 / 111,539 / 111,540",
             "why": """(a) 60 characters, 0.9 × 60 = 54, so the last 6 characters, <i>speak.</i>, are the validation set.
(b) <code>a, b = int(0.8 * len(data)), int(0.9 * len(data))</code>, then <code>train_data, val_data, test_data =
data[:a], data[a:b], data[b:]</code>. Lengths 48, 6 and 6.
(c) a = int(0.8 × 1,115,394) = 892,315 and b = 1,003,854, so 892,315 for training, 1,003,854 − 892,315 = 111,539 for
validation, and 111,540 for testing. Rounding down makes the middle part one character shorter; all three still add up
to 1,115,394."""},
        ],
    },
]
