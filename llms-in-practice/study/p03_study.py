"""Study guide content for LLMs in Practice, episode 3: Sampling.

Build:  python framework/study_guide.py llms-in-practice p03 --video llms-in-practice/media/videos/p03_scene/1080p60/SamplingVideo.mp4
Figures: {"t": seconds into the rendered video, "caption": ...}; placed in body blocks with {fig0}, {fig1}, ...
Exercise kinds: mc (options), tf, short, number, order, code. Every exercise has an answer and a why.
Real numbers come from code/p03_sampling (Qwen2.5-0.5B-Instruct, transformers 4.57.1, torch 2.14.0, CPU);
toy calculations were checked in Python.
"""

LESSON = {
    "series": "LLMs in Practice",
    "label": "Episode 3",
    "title": "Sampling",
    "tagline": "Temperature, top-p, and why answers vary",
    "duration": "2:24",
    "intro": """<p>This lesson answers one question: why can the same prompt give different answers? At every step
the model gives a <b>probability to every token</b>, and a <b>decoding rule</b> picks one. <b>Greedy</b> always takes
the top token; <b>sampling</b> draws at random by probability. <b>Temperature</b> sharpens or flattens the
probabilities, and <b>top-p</b> cuts the long tail of unlikely tokens.</p>""",
    "prereq": """<div class="box note"><b>Before you start:</b> How LLMs Work episode 10 (logits, softmax, choosing the
next token) and Foundations F06–F07 (probability, sampling and softmax). Code: <code>code/p03_sampling</code> (downloads
the 0.5B model, about 1 GB, and runs on a CPU).</div>""",
}

CONCEPTS = [
    {
        "title": "Real next-token probabilities",
        "segment": (8, 40),
        "figures": [{"t": 39.0, "caption": "Real probabilities from Qwen2.5-0.5B-Instruct after “The cat sat on the”: "
                                           "couch 8.1%, and “mat” only 2.0%, in 9th place."}],
        "body": [
            """<p>Ask a model the same question twice and you can get two different answers. That is not a bug: it is
a choice made at every token, called <b>sampling</b>.</p>""",
            """<p>At each step the model scores every token in its vocabulary (<b>151,936</b> for Qwen2.5), and
<b>softmax</b> turns the scores into probabilities. The real top choices after <i>“The cat sat on the”</i> are
<b>couch 8.1%</b>, bed 5.7%, window 5.2%; <b>mat</b> gets only <b>2.0%</b>, in ninth place. Real models are rarely as
sure as tidy textbook examples.</p>""",
            "{fig0}",
            """<div class="box key"><b class="t">Key idea</b>The model outputs a <b>distribution</b>, not a word.
Choosing the word is a separate step, and you control it.</div>""",
        ],
        "exercises": [
            {"kind": "tf", "q": "“The model itself decides the next word; the app only displays it.”",
             "answer": "False.", "why": "The model outputs probabilities; a decoding rule (greedy, sampling, …) set by "
                                       "the app picks the token."},
            {"kind": "number", "q": "After “The cat sat on the”, the top 9 tokens have probabilities 8.1, 5.7, 5.2, 4.8, "
                                    "4.8, 2.8, 2.7, 2.5 and 2.0 percent. What percentage is left for the other "
                                    "151,927 tokens?",
             "answer": "About 61.4%.", "why": "100 − 38.6 = 61.4: most of the probability is spread over the long "
                                             "tail."},
            {"kind": "short", "q": "Why does “mat” come only ninth, although it is the classic example?",
             "answer": "Many continuations are plausible (couch, bed, window, …); the model spreads probability over "
                       "all of them.",
             "why": "Without more context, “mat” is just one of many reasonable endings."},
        ],
    },
    {
        "title": "Greedy decoding and sampling",
        "segment": (40, 64),
        "figures": [{"t": 51.5, "caption": "Greedy: always the top token, so every run is identical."},
                    {"t": 63.0, "caption": "Sampling: draw by probability, so three runs give three different "
                                           "stories (real outputs, seeds 1–3)."}],
        "body": [
            """<p>The simplest rule is to always pick the top token: <b>greedy decoding</b>. Same input, same output,
every time: <i>“The cat sat on the couch, and the dog was sitting next”</i>. But greedy text tends to be dull, and it
can get stuck repeating itself.</p>""",
            '<div class="figrow">{fig0}{fig1}</div>',
            """<p><b>Sampling</b> draws a token at random, weighted by its probability: couch 8.1% of the time, bed
5.7%, and so on. Each step's choice changes everything after it, so three runs give three different stories.</p>""",
            """<div class="box key"><b class="t">Key idea</b>Greedy = deterministic and safe but flat. Sampling =
varied, because every step is a weighted random draw.</div>""",
        ],
        "exercises": [
            {"kind": "mc", "q": "You run greedy decoding on the same prompt three times. What do you get?",
             "options": ["Three different texts", "The same text three times", "An error", "Only the first token"],
             "answer": "B.", "why": "Greedy always picks the top token, so nothing is random."},
            {"kind": "number", "q": "With plain sampling (temperature 1), how many times out of 1,000 would you expect "
                                    "the first token to be “couch” (8.1%)?",
             "answer": "About 81.", "why": "1,000 × 0.081 = 81."},
            {"kind": "short", "q": "Why does one different early token lead to a completely different story?",
             "answer": "Each new token is appended and becomes part of the input for every later step.",
             "why": "That is the generation loop of How LLMs Work episode 1: predict, pick, append, repeat."},
        ],
    },
    {
        "title": "Temperature",
        "segment": (64, 87),
        "figures": [{"t": 85.5, "caption": "At temperature 2 the distribution is almost flat: couch falls to 0.6%."}],
        "body": [
            """<p><b>Temperature</b> reshapes the odds: divide the scores by the temperature before the softmax. At
<b>0.5</b> the distribution gets sharper: couch jumps from 8.1% to <b>29.2%</b>. At <b>2</b> it flattens: couch falls
to <b>0.6%</b>, and every token is a long shot.</p>""",
            "{fig0}",
            """<p>Low temperature is focused and predictable; high temperature is creative, and then chaotic.
Temperature 0 is usually treated as greedy.</p>""",
            """<div class="box key"><b class="t">Key idea</b><code>softmax(scores / T)</code>: T &lt; 1 sharpens,
T &gt; 1 flattens. The ranking of tokens never changes, only how peaked it is.</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "Three tokens have scores 2, 1 and 0. At temperature 1 the softmax gives 0.665, "
                                    "0.245 and 0.090. What is the top token's probability at temperature 0.5? "
                                    "(Hint: the scores become 4, 2, 0.)",
             "answer": "0.867.", "why": "e⁴ / (e⁴ + e² + e⁰) = 54.6 / 63.0 ≈ 0.867: lower temperature sharpens."},
            {"kind": "mc", "q": "Same scores, temperature 2. Which is closest to the top token's probability?",
             "options": ["0.87", "0.67", "0.51", "0.33"],
             "answer": "C.", "why": "Scores 1, 0.5, 0 give 0.506: flatter, moving toward 1/3 each."},
            {"kind": "tf", "q": "“Raising the temperature can make a token that was ranked 5th become ranked 1st.”",
             "answer": "False.", "why": "Dividing all scores by the same positive number keeps their order."},
        ],
    },
    {
        "title": "Top-p: cutting the long tail",
        "segment": (87, 111),
        "figures": [{"t": 109.5, "caption": "Cumulative probability of the top tokens (log scale): top-p 0.5 keeps 19 "
                                            "tokens, top-p 0.9 keeps 378 of 151,936."}],
        "body": [
            """<p>The <b>long tail</b>: thousands of unlikely tokens, each tiny, but together they add up, and some of
them are nonsense. <b>Top-p</b> (nucleus) sampling keeps only the smallest set of top tokens whose probabilities add
up to <b>p</b>, renormalises them, and samples from those.</p>""",
            "{fig0}",
            """<p>For our prompt, top-p 0.9 keeps <b>378</b> tokens and top-p 0.5 keeps just <b>19</b>. Everything
else is cut. How many tokens survive depends on how sure the model is: a confident model may keep only one or two.</p>""",
            """<div class="box key"><b class="t">Key idea</b>Top-p adapts to the model's certainty: it keeps few tokens
when the model is sure, and many when it is not.</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "Four tokens have probabilities 0.5, 0.3, 0.15 and 0.05. With top-p = 0.8 (keep a "
                                    "token while the mass before it is below p), which tokens are kept, and what are "
                                    "their probabilities after renormalising?",
             "answer": "The first two: 0.625 and 0.375.",
             "why": "Mass before the 3rd token is 0.8, not below 0.8, so it is cut. 0.5 / 0.8 = 0.625; 0.3 / 0.8 = "
                    "0.375."},
            {"kind": "tf", "q": "“Top-p 0.9 always keeps the same number of tokens.”",
             "answer": "False.", "why": "It keeps as many as needed to reach 90% of the probability, which depends on "
                                       "the distribution at that step."},
            {"kind": "order", "q": "Put the top-p steps in order: <i>sample · sort by probability · renormalise · add up "
                                   "until p</i>.",
             "answer": "sort by probability → add up until p → renormalise → sample.",
             "why": "This is what the episode's code does."},
        ],
    },
    {
        "title": "Choosing settings",
        "segment": (111, 142),
        "figures": [{"t": 120.5, "caption": "Temperature, then sort, cumulative sum, cut at p, and draw."},
                    {"t": 136.0, "caption": "Typical starting points: low temperature for precise tasks, higher with "
                                            "top-p for creative ones."}],
        "body": [
            """<p>In code: divide the logits by the temperature and take the softmax; sort the probabilities, add them
up, keep tokens until you reach p; then draw one at random.</p>""",
            '<div class="figrow">{fig0}{fig1}</div>',
            """<p>For <b>facts, code and extracting data</b>: low temperature, close to greedy. For <b>brainstorming and
stories</b>: a higher temperature, with top-p to trim the nonsense. For <b>repeatable tests</b>: fix the random
seed.</p>""",
            """<div class="box key"><b class="t">Key idea</b>Pick decoding settings per task: precision wants low
randomness, creativity wants more, and testing wants a fixed seed.</div>""",
        ],
        "exercises": [
            {"kind": "mc", "q": "You extract invoice totals from emails into a spreadsheet. Which setting?",
             "options": ["Temperature 1.5, top-p 1.0", "Temperature 0 (greedy)", "Temperature 1.2, top-p 0.9",
                         "A different random seed for each email"],
             "answer": "B.", "why": "You want the single most likely, consistent answer, not variety."},
            {"kind": "short", "q": "Why fix the random seed when testing a prompt that uses sampling?",
             "answer": "So the same input gives the same output, and changes in results come from your prompt changes, "
                       "not from chance.",
             "why": "With a fixed seed the random draws repeat exactly."},
            {"kind": "code", "q": "<b>Try it yourself.</b> In <code>code/p03_sampling/sampling.py</code>, change "
                                  "<code>PROMPT</code> to <code>\"The capital of France is\"</code> and run it. "
                                  "(a) What is the top token and its probability at T = 1? (b) How many tokens does "
                                  "top-p 0.9 keep now? (c) Explain the difference from the cat prompt.",
             "code": """PROMPT = "The capital of France is"
# then run: python sampling.py""",
             "answer": "(a) “ Paris”, about 0.30 (next come quiz-style blanks such as “ ______”, 0.12). (b) 25 tokens. "
                       "(c) The model is much surer about a fact than about where a cat sat, so probability is "
                       "concentrated on a few tokens and top-p 0.9 needs 25 instead of 378.",
             "why": "Top-p adapts to certainty. The blanks show the model also considers that the text might be a quiz, "
                    "like a base model would (How LLMs Work, episode 14)."},
        ],
    },
]
