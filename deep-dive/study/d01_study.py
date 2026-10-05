"""Study guide content for How LLMs Work: Deep Dive, episode 1: Byte-Pair Encoding.

Build:  python framework/study_guide.py deep-dive d01 --video deep-dive/media/videos/d01_scene/1080p60/BPEVideo.mp4
Figures: {"t": seconds into the rendered video, "caption": ...}; placed in body blocks with {fig0}, {fig1}, ...
Exercise kinds: mc (options), tf, short, number, order, code. Every exercise has an answer and a why.
Every number comes from code/d01_bpe/bpe.py (Tiny Shakespeare) and GPT-2's tokenizer (transformers 4.57.1); the toy
example was run in Python.
"""

LESSON = {
    "series": "How LLMs Work: Deep Dive",
    "label": "Episode 1",
    "title": "Byte-Pair Encoding",
    "tagline": "How a tokenizer learns its vocabulary, step by step",
    "duration": "3:04",
    "intro": """<p>This lesson answers one question: where does a tokenizer's vocabulary come from? <b>Byte-pair
encoding</b> starts from the 256 possible <b>bytes</b> and repeatedly <b>merges the most frequent neighbouring
pair</b> into a new token. Trained from scratch on Shakespeare, it learns sensible pieces, but also glues names,
punctuation and new lines together, until <b>pre-tokenization</b> restricts merges to inside words, as GPT-2 does.</p>""",
    "prereq": """<div class="box note"><b>Before you start:</b> How LLMs Work episode 2 (tokens). Code:
<code>code/d01_bpe/bpe.py</code> (pure Python; <code>transformers</code> only for the GPT-2 comparison).</div>""",
}

CONCEPTS = [
    {
        "title": "Start from bytes",
        "segment": (8, 41),
        "figures": [{"t": 39.8, "caption": "“To be” as UTF-8 bytes; 256 starting tokens; 200,000 characters = 200,000 "
                                           "byte tokens."}],
        "body": [
            """<p>Any text, in any language, can be written as <b>bytes</b> (with UTF-8), and a byte has only <b>256</b>
possible values. So BPE begins with a vocabulary of exactly 256 tokens: nothing is ever unknown. But sequences are long:
200,000 characters of Shakespeare are 200,000 tokens.</p>""",
            "{fig0}",
            """<div class="box key"><b class="t">Key idea</b>Bytes guarantee coverage; merges buy back shorter
sequences.</div>""",
        ],
        "exercises": [
            {"kind": "short", "q": "“To be” becomes the bytes 84, 111, 32, 98, 101. Which byte is the space?",
             "answer": "32.", "why": "In UTF-8 (and ASCII) the space character is byte 32."},
            {"kind": "tf", "q": "“A byte-level tokenizer can still meet a word it has no tokens for.”",
             "answer": "False.", "why": "Any text can be written with the 256 byte tokens, so the worst case is just more "
                                       "tokens."},
        ],
    },
    {
        "title": "Merge the most frequent pair, again and again",
        "segment": (41, 88),
        "figures": [{"t": 69.4, "caption": "The real first merges on Tiny Shakespeare: e + space, t + h, t + space, s + "
                                           "space, o + u."},
                    {"t": 87.1, "caption": "After 500 merges: 81,132 tokens; the Hamlet line is 16 tokens."}],
        "body": [
            """<p>The algorithm: count every pair of neighbouring tokens, take the most frequent pair, give it a new token
number, replace it everywhere, and repeat. On Tiny Shakespeare, merge 1 is <b>“e” + space</b>, seen 5,249 times; it
becomes token 256. Then t + h, t + space, s + space, o + u. Each merge makes the text a little shorter: 200,000 →
194,751 → 190,686 …</p>""",
            '<div class="figrow">{fig0}{fig1}</div>',
            """<p>After 500 merges the text is <b>81,132</b> tokens. To encode new text, replay the merges in the order
they were learned: <i>“To be, or not to be, that is the question.”</i> becomes 16 tokens, with “question” split into
“qu”, “es”, “tion”.</p>""",
            """<div class="box key"><b class="t">Key idea</b>The vocabulary is just the history of merges: frequent
sequences in the training text become single tokens.</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "Run BPE by hand on <i>aaabdaaabac</i> (11 characters). After each of the first three "
                                    "merges, how many tokens are left?",
             "answer": "9, 7, 5.", "why": "Merge “aa” → [aa]abd[aa]abac (9); merge [aa]+a → (7); merge [[aa]a]+b → (5). "
                                         "(Counting overlapping pairs, “aa” appears 4 times, but only 2 non-overlapping "
                                         "copies can be merged.)"},
            {"kind": "number", "q": "Merge 1 (“e” + space, seen 5,249 times) shrank the text from 200,000 to 194,751 "
                                    "tokens. By how many tokens?",
             "answer": "5,249.", "why": "Every merged pair turns two tokens into one."},
            {"kind": "order", "q": "Order the steps of one BPE iteration: <i>replace everywhere · count neighbouring pairs "
                                   "· add a new token id · pick the most frequent pair</i>.",
             "answer": "count neighbouring pairs → pick the most frequent pair → add a new token id → replace "
                       "everywhere.",
             "why": "Then the loop starts again on the shorter sequence."},
        ],
    },
    {
        "title": "Pre-tokenization: merge only inside words",
        "segment": (88, 124),
        "figures": [{"t": 102.7, "caption": "Naive BPE glues punctuation, new lines and a name into one token."},
                    {"t": 123.4, "caption": "Split into words first: tokens like “␣be”, whole words like “␣Senator”; 15 "
                                            "tokens."}],
        "body": [
            """<p>The longest tokens the naive version learned: a period, two new lines, the name <i>CORIOLANUS</i>, a
colon and another new line, all glued into one token. The merges ignore word boundaries, so punctuation, spacing and
names fuse together.</p>""",
            '<div class="figrow">{fig0}{fig1}</div>',
            """<p>The fix: split the text into <b>words first</b>, keeping each space at the start of the word, and only
merge inside a word. The first merges become “␣t”, “he”, “ou”; the longest tokens are whole words like “␣Senator”;
and the Hamlet line becomes 15 tokens: “␣be”, “␣not”, “␣the”, just like a real tokenizer.</p>""",
            """<div class="box key"><b class="t">Key idea</b>Pre-tokenization decides which merges are allowed; it shapes
the vocabulary as much as the data does.</div>""",
        ],
        "exercises": [
            {"kind": "short", "q": "Why is a token like “.↵↵CORIOLANUS:↵” a bad use of the vocabulary?",
             "answer": "It only fits one exact pattern in this play; it wastes a vocabulary slot and teaches the model "
                       "nothing reusable about words or punctuation.",
             "why": "Good tokens are pieces that recur in many contexts."},
            {"kind": "mc", "q": "With word pre-splitting, where does the space go?",
             "options": ["At the end of each word", "At the start of each word (“␣be”)", "It is deleted",
                         "It is always a separate token"],
             "answer": "B.", "why": "GPT-2-style splitting attaches the space to the following word."},
        ],
    },
    {
        "title": "GPT-2, and how big a vocabulary should be",
        "segment": (124, 182),
        "figures": [{"t": 147.8, "caption": "GPT-2: 256 + 50,000 + 1 = 50,257 tokens; the Hamlet line is 13 tokens."},
                    {"t": 164.3, "caption": "A bigger vocabulary: shorter sequences, but more embeddings and rarer "
                                            "tokens."}],
        "body": [
            """<p><b>GPT-2</b> uses exactly this recipe: <b>50,000 merges</b> learned from 40 GB of web text. Its
vocabulary is 256 bytes + 50,000 merges + 1 end-of-text token = <b>50,257</b>. The Hamlet line is 13 tokens, with
“␣question” a single token. On 100,000 characters of Shakespeare it has not seen, GPT-2 needs 32,324 tokens, about a
third fewer than our 500-merge tokenizer (48,811).</p>""",
            '<div class="figrow">{fig0}{fig1}</div>',
            """<p>Why not merge forever? A bigger vocabulary makes sequences <b>shorter</b> (less attention work,
episode 2 of LLMs in Practice), but every token needs its own <b>embedding</b>, and <b>rare</b> tokens get little
training. Modern models settle on about 100,000–250,000 tokens (Qwen2.5: 151,936).</p>""",
            """<div class="box key"><b class="t">Key idea</b>Vocabulary size trades sequence length against embedding size
and how well each token is trained.</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "GPT-2's embedding vectors have 768 numbers. How many numbers are in its token "
                                    "embedding table?",
             "answer": "38,597,376.", "why": "50,257 × 768 = 38,597,376: one row per token."},
            {"kind": "code", "q": "<b>Try it yourself.</b> In <code>code/d01_bpe/bpe.py</code>, set "
                                  "<code>N_MERGES = 100</code> and run it. (a) How many tokens is the Hamlet line for the "
                                  "naive tokenizer, and how many for unseen text? (b) Compare with 500 merges.",
             "code": """N_MERGES = 100""",
             "answer": "(a) 21 tokens; 62,593 tokens for the 100,000 unseen characters (1.60 bytes per token). "
                       "(b) With 500 merges: 16 tokens and 46,714 (2.14 bytes per token). More merges, shorter sequences.",
             "why": "Each merge is a new vocabulary entry that can replace a frequent pair."},
        ],
    },
]
