"""Study guide content for How LLMs Work, episode 2: Tokens: Chopping Text into Pieces.

Build:  python framework/study_guide.py how-llms-work v02
Figures: {"t": seconds into the rendered video, "caption": ...}; placed in body blocks with {fig0}, {fig1}, ...
Exercise kinds: mc (options), tf, short, number, order, code. Every exercise has an answer and a why.
"""

LESSON = {
    "series": "How LLMs Work",
    "label": "Episode 2",
    "title": "Tokens: Chopping Text into Pieces",
    "tagline": "How text becomes numbers a model can read",
    "duration": "2:53",
    "intro": """<p>This lesson answers a question episode 1 skipped: what exactly goes into the model? A neural network
only works with numbers, so text is first chopped into <b>tokens</b>, usually pieces of words, and each token is
replaced by a number, its <b>ID</b>. The pieces are not chosen by hand: an algorithm called <b>byte pair encoding</b>
learns them by counting which letters appear side by side.</p>""",
    "prereq": """<div class="box note"><b>Before you start:</b> this lesson builds on episode 1 (the predict → pick →
append loop). No maths is needed. Concept 5 reads a short Python loop; if Python is new to you, Foundations F11 (NumPy
in Three Minutes) helps. A note on notation: <b>␣</b> marks a space, so <i>␣cat</i> is “cat” with a space in
front.</div>""",
}

CONCEPTS = [
    {
        "title": "Text in, numbers out: the tokenizer",
        "segment": (8, 29),
        "figures": [{"t": 28.4, "caption": "The tokenizer turns text into a list of numbers, and numbers back into "
                                           "text (the curved arrow)."}],
        "body": [
            """<p>Episode 1 said an LLM predicts the next word. That was a small lie: it predicts the next
<b>token</b>. The reason is that a neural network only understands <b>numbers</b>. So before anything else, text must
become a list of numbers, and the model's output must be turned back into text.</p>""",
            "{fig0}",
            """<div class="box key"><b class="t">Key idea</b>The <b>tokenizer</b> converts text into a list of token
<b>IDs</b> (<i>“The cat sat on the”</i> → [464, 3797, 3332, 319, 262]) and converts IDs back into text. The model
itself only ever sees the numbers.</div>""",
        ],
        "exercises": [
            {"kind": "tf", "q": "“An LLM reads the letters of your text directly.”",
             "answer": "False.", "why": "It reads a list of numbers, the token IDs produced by the tokenizer.", "key": {'value': False}},
            {"kind": "mc", "q": "What does a tokenizer do?",
             "options": ["Corrects spelling before the model sees the text",
                         "Turns text into a list of numbers, and numbers back into text",
                         "Predicts the next token", "Picks the most likely word from the probabilities"],
             "answer": "B.", "why": "Predicting and picking are the model's loop from episode 1. The tokenizer only "
                                   "converts between text and numbers, in both directions.", "key": {'choice': 1}},
            {"kind": "short", "q": "The model predicts that the next token is ID 262. What has to happen before you "
                                   "see anything on screen? What appears?",
             "lines": 2,
             "answer": "The tokenizer turns 262 back into text: <i>the</i>.",
             "why": "Output goes through the tokenizer too, in the reverse direction. In the video's list, 262 is the "
                    "last token, <i>the</i>."},
        ],
    },
    {
        "title": "Two extremes: characters or words",
        "segment": (29, 58),
        "figures": [{"t": 42.6, "caption": "Characters: 18 tokens for 5 words."},
                    {"t": 57.3, "caption": "Words: a new word becomes [UNKNOWN]."}],
        "body": [
            """<p>The simplest tokenizer uses <b>one token per character</b>: a tiny vocabulary (a few hundred
symbols), but long sequences, and a single letter like <i>t</i> carries almost no meaning. The other extreme, <b>one
token per word</b>, gives short sequences, but the vocabulary explodes: every name, typo and bit of slang needs an
entry, and a word never seen in training, like <i>catfluencer</i>, can't be read at all.</p>""",
            '<div class="figrow">{fig0}{fig1}</div>',
            """<div class="box key"><b class="t">Key idea</b>Characters: <b>small vocabulary, long sequences</b>.
Words: <b>short sequences, huge vocabulary</b>, and unknown words are lost. Neither extreme works well.</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "How many tokens is <i>“The cat sat on the mat”</i> with one token per character "
                                    "(spaces count)? Per word?",
             "answer": "22 and 6.", "why": "17 letters + 5 spaces = 22 characters, against just 6 words.", "key": {'parts': [{'label': 'per character', 'value': 22, 'tol': 0.5, 'unit': None}, {'label': 'per word', 'value': 6, 'tol': 0.5, 'unit': None}]}},
            {"kind": "mc", "q": "A word-level tokenizer meets <i>rizzmaster</i>, a word that was not in its training "
                                "text. What happens?",
             "options": ["It splits it into <i>rizz</i> + <i>master</i>", "It becomes [UNKNOWN]: the model can't read it",
                         "It adds the word to its vocabulary on the fly", "It spells it out letter by letter"],
             "answer": "B.", "why": "A word tokenizer only knows the whole words in its fixed vocabulary. Splitting a "
                                   "word into pieces is exactly what it can't do.", "key": {'choice': 1}},
            {"kind": "tf", "q": "“Character tokens can spell any text, even words never seen in training.”",
             "answer": "True.", "why": "Every word is built from characters the vocabulary already has. The price is "
                                       "long sequences with little meaning per token.", "key": {'value': True}},
            {"kind": "mc", "q": "Which problem belongs to character tokens, not to word tokens?",
             "options": ["The vocabulary needs hundreds of thousands of entries", "Typos and new slang become unknown",
                         "Sequences are long and each token carries almost no meaning",
                         "Every name needs its own token"],
             "answer": "C.", "why": "A, B and D are all word-tokenizer problems.", "key": {'choice': 2}},
        ],
    },
    {
        "title": "Subword tokens: the middle road",
        "segment": (58, 76),
        "figures": [{"t": 75.8, "caption": "Subword tokens in GPT-2: a common word is one token, rarer words are "
                                           "built from pieces."}],
        "body": [
            """<p>Modern LLMs use <b>subword tokens</b>. Common words get a single token; rarer words are built from
pieces. In GPT-2's tokenizer, <i>the</i> is one token, <i>tokenization</i> is two (<i>token</i> + <i>ization</i>) and
<i>catnap</i> is three (<i>cat</i> + <i>n</i> + <i>ap</i>).</p>""",
            "{fig0}",
            """<div class="box key"><b class="t">Key idea</b><b>Subword tokens</b> get the best of both extremes: common
words = one token, rare words = several pieces. Sequences stay fairly short, the vocabulary stays manageable, and a
brand-new word can still be read as a sequence of pieces.</div>""",
        ],
        "exercises": [
            {"kind": "short", "q": "Why does GPT-2 split <i>tokenization</i> into two pieces but keep <i>the</i> "
                                   "as one token?",
             "answer": "<i>the</i> is very common; <i>tokenization</i> is rarer.",
             "why": "Frequent words earn their own token. Rarer words are built from pieces that are themselves "
                    "common."},
            {"kind": "mc", "q": "GPT-2 has no token for the made-up word <i>catfluencer</i>. What does its tokenizer "
                                "do with it?",
             "options": ["Outputs [UNKNOWN]", "Builds it from smaller pieces", "Refuses the whole input",
                         "Replaces it with the closest known word"],
             "answer": "B.", "why": "It becomes <i>cat</i> + <i>flu</i> + <i>encer</i>. With subwords, no word is "
                                   "unreadable.", "key": {'choice': 1}},
            {"kind": "number", "q": "Tokenized one at a time, how many GPT-2 tokens do <i>the</i>, <i>tokenization</i> "
                                    "and <i>catnap</i> take in total? How many tokens would a character tokenizer "
                                    "need for the same three words?",
             "answer": "6 and 21.", "why": "1 + 2 + 3 = 6 subword tokens against 3 + 12 + 6 = 21 characters.", "key": {'parts': [{'label': 'GPT-2 total', 'value': 6, 'tol': 0.5, 'unit': None}, {'label': 'character tokenizer', 'value': 21, 'tol': 0.5, 'unit': None}]}},
            {"kind": "tf", "q": "“With subword tokens, every English word is exactly one token.”",
             "answer": "False.", "why": "Only common words are one token. <i>tokenization</i> takes 2 and "
                                        "<i>catnap</i> takes 3.", "key": {'value': False}},
        ],
    },
    {
        "title": "Byte pair encoding: merge the most frequent pair",
        "segment": (76, 110),
        "figures": [{"t": 88.6, "caption": "Start with single characters and count every pair of neighbours. e+s is "
                                           "a top pair (3 times)."},
                    {"t": 109.6, "caption": "After four merges the tokenizer has found the word low and the ending "
                                            "est."}],
        "body": [
            """<p>How are the pieces chosen? The most popular method is <b>byte pair encoding (BPE)</b>. Start with
single characters. Count every pair of neighbours in a pile of training text. <b>Merge the most frequent pair</b>
into a brand-new token. Then count again, and repeat.</p>""",
            '<div class="figrow">{fig0}{fig1}</div>',
            """<p>In the toy text <i>low, lowest, newest, widest</i>, <i>e</i>+<i>s</i> appears 3 times, as often as
any pair, so it merges first into <i>es</i>. Next <i>es</i>+<i>t</i> → <i>est</i>, then <i>l</i>+<i>o</i> →
<i>lo</i>, then <i>lo</i>+<i>w</i> → <i>low</i>. Nobody told it about words: it found <i>low</i> and <i>est</i>
just by counting.</p>""",
            """<div class="box key"><b class="t">Key idea</b>BPE repeats one step: <b>count neighbouring pairs, merge the
most frequent one into a new token</b>. Each merge adds one token to the vocabulary.</div>""",
        ],
        "exercises": [
            {"kind": "order", "q": "Put the steps of BPE training in order: <i>merge the most frequent pair · start "
                                   "with single characters · repeat · count every pair of neighbours</i>.",
             "answer": "start with characters → count pairs → merge the most frequent → repeat.",
             "why": "Each round counts the pairs in the current text and merges the winner.", "key": {'items': ['start with single characters', 'count every pair of neighbours', 'merge the most frequent pair', 'repeat']}},
            {"kind": "short", "q": "New training text: <i>hug, pug, hugs, pun</i>. Which pair is merged first, and "
                                   "how many times does it appear?",
             "answer": "u + g, 3 times.", "why": "It appears in hug, pug and hugs. h+u and p+u appear twice; g+s and "
                                                 "u+n once."},
            {"kind": "short", "q": "Apply the video's four merges, in order (e+s, es+t, l+o, lo+w), to the new word "
                                   "<i>slowest</i>. Which tokens do you get?",
             "answer": "s + low + est.", "why": "s l o w e s t → s l o w es t → s l o w est → s lo w est → "
                                                "s low est."},
            {"kind": "tf", "q": "“Once <i>e</i> and <i>s</i> are merged into <i>es</i>, the token <i>e</i> is no "
                                "longer needed.”",
             "answer": "False.", "why": "A merge adds a token; nothing is removed. <i>e</i> is still needed, e.g. in "
                                        "<i>n e w est</i>, where it is not followed by <i>s</i>.", "key": {'value': False}},
        ],
    },
    {
        "title": "BPE training in code",
        "segment": (110, 121),
        "figures": [{"t": 120.3, "size": "small", "caption": "Training a tokenizer: a short loop, repeated tens of "
                                                             "thousands of times."}],
        "body": [
            """<p>In code, training a tokenizer is a short loop. <code>words</code> is the training text split into
characters. Each round counts the pairs, picks the most frequent one, merges it everywhere and records the merge.</p>""",
            "{fig0}",
            """<pre class="code">def train_bpe(words, num_merges):
    merges = []
    for _ in range(num_merges):
        pairs = count_pairs(words)              # neighbour pairs -> count
        best = max(pairs, key=pairs.get)        # most frequent pair
        words = merge_pair(words, best)         # glue it everywhere
        merges.append(best)
    return merges</pre>""",
            """<div class="box key"><b class="t">Key idea</b>One loop iteration = <b>one merge = one new token</b>.
<code>num_merges</code> sets how many tokens are added to the starting characters: tens of thousands for a real
tokenizer.</div>""",
        ],
        "exercises": [
            {"kind": "mc", "q": "Why does the loop call <code>count_pairs</code> again in every round, instead of "
                                "counting once at the start?",
             "options": ["To make training slower but safer", "Because each merge changes the words, so the pair "
                         "counts change", "Because <code>max</code> can only be used once per list",
                         "It doesn't need to; counting once gives the same result"],
             "answer": "B.", "why": "After <i>e</i>+<i>s</i> merges, the pair <i>es</i>+<i>t</i> exists for the first "
                                   "time, and it wins the next round.", "key": {'choice': 1}},
            {"kind": "number", "q": "The toy text <i>low, lowest, newest, widest</i> uses 9 different letters. After "
                                    "the 4 merges of concept 4, how many different tokens are in the vocabulary?",
             "answer": "13.", "why": "9 starting characters + 1 new token per merge = 9 + 4.", "key": {'parts': [{'label': None, 'value': 13, 'tol': 0.5, 'unit': None}]}},
            {"kind": "code", "q": "<b>Try it yourself.</b> Below are <code>merge_pair</code> and "
                                  "<code>train_bpe</code> for the toy text. (a) Write <code>count_pairs(words)</code>, "
                                  "which returns a <code>Counter</code> of every neighbouring pair (hint: "
                                  "<code>zip(w, w[1:])</code>). How many different pairs does it find, and what is the "
                                  "count of <code>('e', 's')</code>? (b) What does <code>train_bpe(words, 4)</code> "
                                  "return? (c) Run <code>train_bpe(words, 5)</code>. What is the fifth merge, and why "
                                  "is that choice somewhat arbitrary?",
             "code": """from collections import Counter

def merge_pair(words, pair):
    out = []
    for w in words:
        new, i = [], 0
        while i < len(w):
            if i + 1 < len(w) and (w[i], w[i + 1]) == pair:
                new.append(w[i] + w[i + 1])   # glue the pair
                i += 2
            else:
                new.append(w[i])
                i += 1
        out.append(new)
    return out

def train_bpe(words, num_merges):
    merges = []
    for _ in range(num_merges):
        pairs = count_pairs(words)
        best = max(pairs, key=pairs.get)
        words = merge_pair(words, best)
        merges.append(best)
    return merges

words = [list(w) for w in ["low", "lowest", "newest", "widest"]]""",
             "answer": "(a) 10 pairs; ('e', 's') → 3 · (b) [('e', 's'), ('es', 't'), ('l', 'o'), ('lo', 'w')] · "
                       "(c) ('low', 'est')",
             "why": """(a) <code>def count_pairs(words): return Counter(p for w in words for p in zip(w, w[1:]))</code>.
The 10 pairs are the 5 in the video's table plus the 5 “seen once”. (b) The same four merges as the video. When counts tie (e+s and s+t both 3), <code>max</code> returns
the pair it counted first. (c) After four merges every remaining pair appears only once, so it is a 7-way tie and
<code>max</code> takes the first pair counted, <i>low</i>+<i>est</i> (from <i>lowest</i>). With a huge training
text, exact ties like this are rare."""},
        ],
    },
    {
        "title": "Vocabulary and token IDs",
        "segment": (121, 142),
        "figures": [{"t": 131.0, "caption": "The merges give a vocabulary; each token gets an ID. GPT-2 has 50,257 "
                                            "tokens."},
                    {"t": 141.2, "caption": "Spaces are part of the tokens: ␣cat (3797) and cat (9246) are "
                                            "different tokens."}],
        "body": [
            """<p>All the merges together give a <b>vocabulary</b>. GPT-2 has about fifty thousand tokens (50,257);
newer models use 100,000 or more. Each token gets an <b>ID</b>, and a sentence becomes a short list of numbers.</p>""",
            '<div class="figrow">{fig0}{fig1}</div>',
            """<p>Notice that <b>spaces are part of the tokens</b>. <i>␣cat</i> (with a space in front, ID 3797) and
<i>cat</i> (no space, ID 9246) are different tokens, so <i>“The cat sat on the”</i> is really <i>The · ␣cat · ␣sat ·
␣on · ␣the</i>.</p>""",
            """<div class="box key"><b class="t">Key idea</b>Text → tokens → <b>IDs</b>, one number per token from a
fixed vocabulary. A leading space is part of a token, so the same word can have two IDs.</div>""",
        ],
        "exercises": [
            {"kind": "short", "q": "Use the vocabulary table in the left figure to decode [464, 3332, 319, 262, 9246]. "
                                   "Watch the spaces!",
             "answer": "“The sat on thecat”.",
             "why": "9246 is <i>cat</i> with no space in front, so it sticks to <i>the</i>. With 3797 (<i>␣cat</i>) "
                    "it would read “… the cat”."},
            {"kind": "number", "q": "GPT-2's tokenizer starts from 256 basic symbols (one per possible byte), makes "
                                    "50,000 merges, and adds one special end-of-text token. How big is its "
                                    "vocabulary?",
             "answer": "50,257.", "why": "256 + 50,000 + 1 = 50,257: one token per starting symbol, one per merge, "
                                         "plus the special token.", "key": {'parts': [{'label': None, 'value': 50257, 'tol': 0.5, 'unit': None}]}},
            {"kind": "mc", "q": "In GPT-2, why does <i>cat</i> get a different ID in <i>“cat food”</i> than in "
                                "<i>“my cat”</i>?",
             "options": ["IDs are assigned at random each time", "In <i>“my cat”</i> the token is <i>␣cat</i>, with a "
                         "space in front", "IDs depend on the word's position in the sentence",
                         "The tokenizer ignores the second word of every text"],
             "answer": "B.", "why": "At the start of <i>“cat food”</i> there is no space, so it is <i>cat</i> (9246); "
                                   "in <i>“my cat”</i> it is <i>␣cat</i> (3797).", "key": {'choice': 1}},
        ],
    },
    {
        "title": "The model never sees letters",
        "segment": (142, 170),
        "figures": [{"t": 155.3, "caption": "To GPT-2, ␣strawberry is one single token, ID 41236."}],
        "body": [
            """<p>Tokens explain some odd behaviour. Ask a model how many r's are in <i>strawberry</i> and it may
stumble. That's because it never sees the letters: to GPT-2, <i>␣strawberry</i> is <b>one single token</b>, ID 41236.</p>""",
            "{fig0}",
            """<div class="box key"><b class="t">Key idea</b>The model sees <b>token IDs, not letters</b>: the letters
inside a token are hidden from it. And an ID like 3797 is <b>just a label</b> that says nothing about what a cat is;
episode 3 turns each token into something richer, a vector.</div>""",
        ],
        "exercises": [
            {"kind": "mc", "q": "Why can a model miscount the r's in <i>strawberry</i>?",
             "options": ["It never saw the word during training", "It sees <i>␣strawberry</i> as one ID, not as "
                         "separate letters", "Its vocabulary has no letter r", "The question has too many tokens"],
             "answer": "B.", "why": "The letters are inside a single token; the model only gets the number 41236.", "key": {'choice': 1}},
            {"kind": "number", "q": "<i>“How many r's are in strawberry?”</i> has 31 characters. How many GPT-2 "
                                    "tokens is it? (Count them in the figure.)",
             "answer": "8.", "why": "How · ␣many · ␣r · 's · ␣are · ␣in · ␣strawberry · ?: about 4 characters per "
                                    "token.", "key": {'parts': [{'label': None, 'value': 8, 'tol': 0.5, 'unit': None}]}},
            {"kind": "tf", "q": "Without a space in front, GPT-2 splits <i>strawberry</i> into <i>st</i> + "
                                "<i>raw</i> + <i>berry</i>. “Now the model can see each letter.”",
             "answer": "False.", "why": "It sees three IDs instead of one. The r's are still hidden, inside <i>raw</i> "
                                        "and <i>berry</i>.", "key": {'value': False}},
            {"kind": "tf", "q": "“Token 3798 must mean something close to <i>␣cat</i> (3797), because the IDs are "
                                "neighbours.”",
             "answer": "False.", "why": "An ID is just a label. (Episode 3 shows that 3798 is <i>esc</i>.)", "key": {'value': False}},
        ],
    },
]
