"""Study guide content for How LLMs Work: Deep Dive, episode 3: Tokenizer Quirks.

Build:  python framework/study_guide.py deep-dive d03 --video deep-dive/media/videos/d03_scene/1080p60/TokenizerQuirksVideo.mp4
Figures: {"t": seconds into the rendered video, "caption": ...}; placed in body blocks with {fig0}, {fig1}, ...
Exercise kinds: mc (options), tf, short, number, order, code. Every exercise has an answer and a why.
Every token id, split and probability comes from code/d03_tokenizer_quirks/quirks.py or the same tokenizers and model
run in Python (GPT-2, Qwen2.5-0.5B-Instruct; transformers 4.57.1, torch 2.14.0, CPU).
"""

LESSON = {
    "series": "How LLMs Work: Deep Dive",
    "label": "Episode 3",
    "title": "Tokenizer Quirks",
    "tagline": "Small details in tokenization, big effects on the model",
    "duration": "2:40",
    "intro": """<p>This lesson answers one question: how do small details of tokenization show up as strange model
behaviour? Four quirks, each tested for real: the same word as <b>many different tokens</b>; <b>numbers</b> cut into
irregular pieces; a <b>trailing space</b> that derails a prediction; and <b>glitch tokens</b> that were in the
vocabulary but almost never in the training text.</p>""",
    "prereq": """<div class="box note"><b>Before you start:</b> Deep Dive episodes 1–2 (BPE, bytes and Unicode) and
How LLMs Work episode 3 (embeddings). Code: <code>code/d03_tokenizer_quirks</code> (downloads Qwen2.5-0.5B and GPT-2,
about 1.5 GB).</div>""",
}

CONCEPTS = [
    {
        "title": "The same word, many tokens",
        "segment": (8, 38),
        "figures": [{"t": 37.2, "caption": "hello, ␣hello, Hello and ␣Hello: four GPT-2 tokens with unrelated ids; HELLO "
                                           "is three pieces."}],
        "body": [
            """<p>In GPT-2, <i>hello</i>, <i>␣hello</i> (with a space in front), <i>Hello</i> and <i>␣Hello</i> are four
different tokens with unrelated numbers: 31,373, 23,748, 15,496 and 18,435. The model has to learn, separately, that
they mean the same thing. And <i>HELLO</i> is not even one token: it is three pieces, HE | LL | O.</p>""",
            "{fig0}",
            """<div class="box key"><b class="t">Key idea</b>Token ids carry no notion of “the same word”: every variant is
learned from its own examples.</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "GPT-2 ids: “␣World” 2159, “World” 10603, “␣world” 995, “world” 6894. How many different "
                                    "tokens is that for one word?",
             "answer": "4.", "why": "Case and the leading space each produce a separate vocabulary entry."},
            {"kind": "tf", "q": "“Token ids that are numerically close (like 995 and 996) have similar meanings.”",
             "answer": "False.", "why": "Ids are just positions in the vocabulary, in merge order; meaning lives in the "
                                       "learned embeddings."},
        ],
    },
    {
        "title": "Numbers",
        "segment": (38, 61),
        "figures": [{"t": 60.6, "caption": "GPT-2 cuts numbers into irregular chunks; Qwen2.5 uses one token per digit."}],
        "body": [
            """<p>GPT-2 cuts 1234567 into <b>123 | 45 | 67</b>, 2024 into <b>20 | 24</b> and 3.14159 into 3 | . | 14 |
159. The pieces do not line up with place value, which makes arithmetic harder to learn. <b>Qwen2.5</b> splits every
number into <b>single digits</b>, so each digit keeps its place; many newer models do the same.</p>""",
            "{fig0}",
            """<div class="box key"><b class="t">Key idea</b>Consistent digit tokens let the model learn arithmetic as
operations on digits, the way we do on paper.</div>""",
        ],
        "exercises": [
            {"kind": "short", "q": "GPT-2 splits 12345 into 123 | 45, and 1000000 into 1 | 000000. Why does this make "
                                   "adding numbers hard to learn?",
             "answer": "The same digit position falls in different tokens depending on the number, so the model cannot line "
                       "up ones, tens, hundreds consistently.",
             "why": "Column addition needs aligned digits; irregular chunks hide the columns."},
            {"kind": "number", "q": "How many tokens is 1000000 for Qwen2.5 (one token per digit)?",
             "answer": "7.", "why": "Seven digits, seven tokens (GPT-2: 2)."},
        ],
    },
    {
        "title": "The trailing space",
        "segment": (61, 84),
        "figures": [{"t": 83.5, "caption": "One trailing space: “ Paris” falls from 0.302 to rank 28 (0.008)."}],
        "body": [
            """<p>Given <i>“The capital of France is”</i>, Qwen2.5-0.5B's top guess is <b>“ Paris”, 0.302</b>. Add one
space at the end, and “ Paris” drops to <b>rank 28</b> (0.008); the top guess becomes the digit 1. Words carry their
space at the start, so a prompt ending in a lone space is unusual text, and the model is thrown off.</p>""",
            "{fig0}",
            """<div class="box key"><b class="t">Key idea</b>Prompts should end where a natural token boundary is: never
with a trailing space.</div>""",
        ],
        "exercises": [
            {"kind": "mc", "q": "For “The sky is”, Qwen2.5-0.5B's top guesses are “ a” (0.129) and “ blue” (0.107). With a "
                                "trailing space, “The sky is ”, what happens?",
             "options": ["“ blue” jumps to the top", "Nothing changes", "Digits take over: “1” (0.31), “3”, “5”",
                         "The model refuses"],
             "answer": "C.", "why": "Checked by running the model: the trailing space makes the text unusual, and digits "
                                   "become the likeliest continuation."},
            {"kind": "short", "q": "Why does “ Paris” (with a leading space) become unlikely after a trailing space?",
             "answer": "The space is already in the text, so a second space token is unlikely; and “Paris” without a space "
                       "is a rare token in that position.",
             "why": "The model learned that a word's space comes attached to the word, not before it."},
        ],
    },
    {
        "title": "Glitch tokens",
        "segment": (84, 157),
        "figures": [{"t": 106.9, "caption": "Glitch tokens: single GPT-2 tokens whose strings almost never appear in the "
                                            "training text."},
                    {"t": 131.9, "caption": "Embeddings closest to the average: control characters, broken bytes, "
                                            "externalToEVA, quickShip; SolidGoldMagikarp is not flagged."}],
        "body": [
            """<p>GPT-2's vocabulary contains strange single tokens like <b>“␣SolidGoldMagikarp”</b> (id 43,453), a
username from a Reddit forum. The vocabulary was built from one collection of text, and the model was trained on
another, where these strings almost never appear. In early models of that family, asking about them famously produced
bizarre answers.</p>""",
            '<div class="figrow">{fig0}{fig1}</div>',
            """<p>A token that is never seen in training gets almost no updates, so its embedding stays generic. A rough
test, the embeddings closest to the average, finds control characters, broken byte pieces and odd strings like
“externalToEVA” and “quickShip” (themselves known glitch tokens), but “␣SolidGoldMagikarp” ranks 14,357 of 50,257: not
flagged. No single check finds them all. Lessons: no trailing spaces, consistent spelling and capitalization,
digit-splitting tokenizers for math, and look at how your own data is tokenized.</p>""",
            """<div class="box key"><b class="t">Key idea</b>Every token needs training data. Vocabulary built from
different text than the model was trained on leaves under-trained tokens.</div>""",
        ],
        "exercises": [
            {"kind": "short", "q": "Why does a token that never appears in the training text keep a “generic” embedding?",
             "answer": "Its embedding row only receives gradient updates when the token appears in a training example; if it "
                       "never appears, the row stays close to where it started.",
             "why": "Embeddings are learned from use (How LLMs Work, episodes 3 and 11)."},
            {"kind": "code", "q": "<b>Try it yourself.</b> Run this with the GPT-2 tokenizer from "
                                  "<code>quirks.py</code>. How many tokens is each string, and which ones are single "
                                  "tokens?",
             "code": """for w in [" SolidGoldMagikarp", " TheNitromeFan", " davidjl", " cat", " HELLO"]:
    print(repr(w), len(gpt2(w)["input_ids"]))""",
             "answer": "“␣SolidGoldMagikarp”, “␣TheNitromeFan”, “␣davidjl” and “␣cat” are 1 token each; “␣HELLO” is 2 "
                       "(␣HELL | O).",
             "why": "Rare usernames got their own tokens because they were frequent in the text the vocabulary was built "
                    "from."},
        ],
    },
]
