"""Study guide content for How LLMs Work: Deep Dive, episode 2: Bytes, Unicode, and Strawberry.

Build:  python framework/study_guide.py deep-dive d02 --video deep-dive/media/videos/d02_scene/1080p60/BytesUnicodeVideo.mp4
Figures: {"t": seconds into the rendered video, "caption": ...}; placed in body blocks with {fig0}, {fig1}, ...
Exercise kinds: mc (options), tf, short, number, order, code. Every exercise has an answer and a why.
Every number and answer comes from code/d02_bytes_unicode (GPT-2 and Qwen2.5 tokenizers, Qwen2.5-1.5B-Instruct greedy;
transformers 4.57.1, torch 2.14.0, CPU); byte counts were checked in Python.
"""

LESSON = {
    "series": "How LLMs Work: Deep Dive",
    "label": "Episode 2",
    "title": "Bytes, Unicode, and Strawberry",
    "tagline": "Why a model can't see the letters it writes",
    "duration": "2:59",
    "intro": """<p>This lesson answers one question: why is counting the letters in “strawberry” hard for a language
model? Text reaches the model as <b>tokens</b> built from <b>UTF-8 bytes</b>, not as letters. That explains why every
language can be read, why some languages cost several times more tokens, and why spelling and letter counting are
hard: the letters are hidden inside tokens.</p>""",
    "prereq": """<div class="box note"><b>Before you start:</b> Deep Dive episode 1 (byte-pair encoding). Code:
<code>code/d02_bytes_unicode</code> (downloads two tokenizers and Qwen2.5-1.5B-Instruct, about 3 GB).</div>""",
}

CONCEPTS = [
    {
        "title": "Code points and UTF-8",
        "segment": (8, 59),
        "figures": [{"t": 35.6, "caption": "Unicode gives every character a number: a code point."},
                    {"t": 58.3, "caption": "UTF-8 writes code points as 1 to 4 bytes; the leading bits (yellow) say how "
                                           "many bytes belong together."}],
        "body": [
            """<p><b>Unicode</b> gives every character a number, its <b>code point</b>: “a” is 97 (U+0061), “é” U+00E9,
“ж” U+0436, “中” U+4E2D, 🍓 U+1F353; more than a million are possible. <b>UTF-8</b> writes each code point as one to
four bytes: plain English letters take 1 byte, accented and Russian letters 2, Chinese characters 3, emoji 4.</p>""",
            '<div class="figrow">{fig0}{fig1}</div>',
            """<p>The first bits of each byte say how many bytes belong together (<code>0…</code> one byte,
<code>110…</code> the start of two, <code>1110…</code> of three, <code>11110…</code> of four, and <code>10…</code> a
continuation). That is why a byte-level tokenizer, with just 256 starting tokens, can read any language.</p>""",
            """<div class="box key"><b class="t">Key idea</b>Characters → code points → UTF-8 bytes → tokens. The model
only ever sees the last step.</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "How many UTF-8 bytes is the word “café”?",
             "answer": "5.", "why": "c, a, f are 1 byte each; é is 2 bytes."},
            {"kind": "mc", "q": "A byte starts with the bits 1110. What does it tell a UTF-8 reader?",
             "options": ["It is a plain ASCII character", "It starts a 3-byte character", "It continues the previous "
                         "character", "It starts a 4-byte character"],
             "answer": "B.", "why": "1110 = three leading ones: this byte and two continuation bytes (10…) form one "
                                   "character, like 中."},
            {"kind": "number", "q": "The euro sign € is U+20AC. How many UTF-8 bytes does it need?",
             "answer": "3.", "why": "Code points from U+0800 to U+FFFF use 3 bytes (checked: '€'.encode() has length 3)."},
        ],
    },
    {
        "title": "Languages don't cost the same",
        "segment": (59, 86),
        "figures": [{"t": 85.2, "caption": "One sentence in five languages: bytes and tokens for GPT-2 and Qwen2.5."}],
        "body": [
            """<p>One sentence about a cat, in five languages. For <b>GPT-2</b>, English is <b>10</b> tokens; Russian is
<b>38</b> and Chinese <b>25</b>, for the same meaning, because GPT-2 learned its merges mostly from English text.
<b>Qwen2.5</b>, trained on far more languages, needs <b>16</b> tokens for the Russian and only <b>8</b> for the
Chinese.</p>""",
            "{fig0}",
            """<div class="box key"><b class="t">Key idea</b>The tokenizer's training data sets the price of each language:
in cost, speed and how much fits in the context window.</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "For GPT-2, how many times more tokens does the Russian sentence need than the English "
                                    "one?",
             "answer": "3.8 times.", "why": "38 / 10 = 3.8."},
            {"kind": "tf", "q": "“The Chinese sentence is 11 characters but 33 bytes.”",
             "answer": "True.", "why": "Each Chinese character (and the full stop 。) takes 3 bytes: 11 × 3 = 33."},
            {"kind": "short", "q": "A product charges per token. Why might Russian-speaking users pay more with a "
                                   "GPT-2-style tokenizer?",
             "answer": "The same text needs far more tokens (38 vs 10 for our sentence), because the tokenizer has few "
                       "merges for Cyrillic.",
             "why": "Fewer merges learned for a script means shorter tokens and longer sequences."},
        ],
    },
    {
        "title": "Strawberry: letters hidden in tokens",
        "segment": (86, 126),
        "figures": [{"t": 103.5, "caption": "“ strawberry” is one token; without the space, str | aw | berry. The ten "
                                            "letters are never seen."},
                    {"t": 125.0, "caption": "Counting letters: three of four right; nevertheless, just two tokens, gets "
                                            "3 instead of 4."}],
        "body": [
            """<p>With a space in front, as it usually appears in a sentence, “strawberry” is a <b>single token</b>;
without the space, three: <b>str | aw | berry</b>. Either way, the model never sees the ten letters. It sees one or
three token numbers, and has to have learned how each one is spelled.</p>""",
            '<div class="figrow">{fig0}{fig1}</div>',
            """<p>Asked directly, Qwen2.5-1.5B says strawberry has 3 r's: correct, but the question is famous and may
simply have been learned. Bookkeeper and Mississippi were right too; but “nevertheless” got 3 e's instead of 4. To the
model, nevertheless is just two tokens: never, theless.</p>""",
            """<div class="box key"><b class="t">Key idea</b>A right answer to a famous question is not proof of a skill:
test on cases the model cannot have memorised.</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "How many letters are hidden inside the single token “␣strawberry” (not counting the "
                                    "space)?",
             "answer": "10.", "why": "s-t-r-a-w-b-e-r-r-y."},
            {"kind": "short", "q": "Why test “nevertheless” and “bookkeeper” instead of only “strawberry”?",
             "answer": "Strawberry is a famous test the model may have memorised; less famous words check whether it can "
                       "actually count.",
             "why": "A memorised answer and a real skill look the same on a famous question."},
        ],
    },
    {
        "title": "Spelling it out, and what to do",
        "segment": (126, 177),
        "figures": [{"t": 144.5, "caption": "Asked to spell first, the model misspelled strawberry and counted one r; with "
                                            "one token per letter, it said four."},
                    {"t": 161.8, "caption": "Tasks that tokens make hard, and the fixes."}],
        "body": [
            """<p>Asked to spell strawberry first, the model wrote <i>s-t-r-o-w-a-b-e</i>: it misspelled the word and
counted one r. Given the letters with spaces, so every letter is its own token, it said <b>four</b>. Seeing the letters
is necessary, but a small model still struggles to count them.</p>""",
            '<div class="figrow">{fig0}{fig1}</div>',
            """<p>Because tokens hide letters, <b>spelling</b>, <b>counting characters</b>, <b>rhyming</b> and
<b>reversing</b> a word are surprisingly hard, and numbers can be split into odd chunks too. The fixes: let the model
<b>call a tool</b> (a short piece of code, LLMs in Practice episode 7), use a <b>larger model</b>, and <b>check</b>.</p>""",
            """<div class="box key"><b class="t">Key idea</b>For exact character-level work, give the model code to run
instead of asking it to look.</div>""",
        ],
        "exercises": [
            {"kind": "mc", "q": "Which task is easiest for a token-based model?",
             "options": ["Reverse the letters of “strawberry”", "Count the e's in “nevertheless”",
                         "Translate “strawberry” into Italian", "List the letters of “bookkeeper” in order"],
             "answer": "C.", "why": "Translation works at the level of words and meaning; the others need the hidden "
                                   "letters."},
            {"kind": "code", "q": "<b>Try it yourself.</b> Add this to <code>bytes_unicode.py</code>: how does Qwen2.5 "
                                  "tokenize these words, and which do you expect the model to find hardest to spell?",
             "code": """for w in [" strawberry", " unbelievable", " Mississippi", " hippopotamus"]:
    print(w, [qwen.decode([i]) for i in qwen(w)["input_ids"]])""",
             "answer": "“ strawberry”, “ unbelievable” and “ Mississippi” are each one token; “ hippopotamus” is "
                       "“ hipp | opot | amus”. The single-token words hide all their letters, so expect them to be the "
                       "hardest to spell or count letter by letter.",
             "why": "The fewer and longer the tokens, the more the model must have memorised about their spelling."},
        ],
    },
]
