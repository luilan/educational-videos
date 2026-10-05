"""Study guide content for LLMs in Practice, episode 2: Context Windows.

Build:  python framework/study_guide.py llms-in-practice p02 --video llms-in-practice/media/videos/p02_scene/1080p60/ContextWindowVideo.mp4
Figures: {"t": seconds into the rendered video, "caption": ...}; placed in body blocks with {fig0}, {fig1}, ...
Exercise kinds: mc (options), tf, short, number, order, code. Every exercise has an answer and a why.
Every number was produced by code/p02_context_window (Qwen2.5-0.5B-Instruct config and tokenizer, transformers 4.57.1).
"""

LESSON = {
    "series": "LLMs in Practice",
    "label": "Episode 2",
    "title": "Context Windows",
    "tagline": "Why a model can only read so much at once",
    "duration": "2:21",
    "intro": """<p>This lesson answers one question: why can a model only read so much at once, and what happens
when a chat outgrows it? The <b>context window</b> is the maximum number of tokens per request, prompt plus reply. It
exists because of <b>position</b> (trained lengths), <b>attention</b> (work grows with the square of the length) and
<b>memory</b> (the KV cache). When a chat overflows, apps <b>drop the oldest turns</b>, <b>summarise</b> them or
<b>retrieve</b> only what matters.</p>""",
    "prereq": """<div class="box note"><b>Before you start:</b> episode 1 of this series (the whole chat is resent on
every turn). How LLMs Work episodes 4 (position), 5–6 (attention) and 13 (the KV cache) explain the three reasons for
the limit. Code: <code>code/p02_context_window</code> in the repository.</div>""",
}

CONCEPTS = [
    {
        "title": "The context window",
        "segment": (8, 46),
        "figures": [{"t": 34.5, "caption": "The window holds the prompt plus the reply: 32,768 tokens for "
                                           "Qwen2.5-0.5B-Instruct."},
                    {"t": 45.5, "caption": "Tiny Shakespeare is 301,829 tokens: 9.2 full windows."}],
        "body": [
            """<p>Every message resends the whole chat (episode 1), so the text the model reads keeps growing. But a
model can only read so much at once. The <b>context window</b> is the maximum number of tokens the model can handle in
<b>one request</b>: the <b>prompt plus the reply</b> it writes. For Qwen2.5-0.5B-Instruct, the model in our code, it is
<b>32,768 tokens</b>.</p>""",
            '<div class="figrow">{fig0}{fig1}</div>',
            """<p>That sounds like a lot, but Tiny Shakespeare, the collection of plays the tiny GPT of How LLMs Work
learned from, is <b>301,829 tokens</b>: more than nine full windows.</p>""",
            """<div class="box key"><b class="t">Key idea</b>The window is a hard limit per request, and the reply has
to fit in it too.</div>""",
        ],
        "exercises": [
            {"kind": "mc", "q": "What has to fit inside the context window?",
             "options": ["Only your newest message", "Only the model's reply",
                         "The whole prompt (system, earlier turns, documents) plus the reply",
                         "The model's training data"],
             "answer": "C.", "why": "The window counts everything in one request, including the tokens the model "
                                   "writes."},
            {"kind": "number", "q": "A prompt is 30,000 tokens long. With a 32,768-token window, what is the longest "
                                    "reply the model can write?",
             "answer": "2,768 tokens.", "why": "32,768 − 30,000 = 2,768: the reply shares the window with the "
                                              "prompt."},
            {"kind": "number", "q": "Tiny Shakespeare is 301,829 tokens. How many 32,768-token windows is that, to one "
                                    "decimal place?",
             "answer": "9.2.", "why": "301,829 / 32,768 ≈ 9.21."},
        ],
    },
    {
        "title": "Why there is a limit",
        "segment": (46, 81),
        "figures": [{"t": 63.5, "caption": "Position: lengths beyond training are unfamiliar. Attention: double the "
                                           "length, about four times the work."},
                    {"t": 79.5, "caption": "Memory: the KV cache keeps keys and values for every token: 12,288 bytes "
                                           "per token, 384 MiB for a full window."}],
        "body": [
            """<p>Three reasons. <b>Position</b>: the model was trained on sequences up to a certain length, and
positions beyond that are unfamiliar. <b>Attention</b>: every new token looks back at every token before it, so
<b>doubling the length roughly quadruples</b> the attention work (the triangle of token pairs grows with the square
of the length).</p>""",
            '<div class="figrow">{fig0}{fig1}</div>',
            """<p><b>Memory</b>: for each token in the window the model keeps its keys and values, the <b>KV cache</b>
(How LLMs Work, episode 13). In this small model that is 2 (K and V) × 24 layers × 2 KV heads × 64 numbers × 2 bytes
= <b>12,288 bytes per token</b>, so a full window takes <b>384 MiB</b>, for a single conversation.</p>""",
            """<div class="box key"><b class="t">Key idea</b>Longer contexts cost more in every way: unfamiliar
positions, quadratic attention work and linear KV-cache memory.</div>""",
        ],
        "exercises": [
            {"kind": "mc", "q": "A request grows from 1,000 to 2,000 tokens. Roughly how does the attention work "
                                "change?",
             "options": ["It stays the same", "It doubles", "It roughly quadruples", "It grows tenfold"],
             "answer": "C.", "why": "Every token attends to every earlier token, so the work grows with the square of "
                                   "the length: 2² = 4."},
            {"kind": "number", "q": "The KV cache of this model takes 12,288 bytes per token. How many MiB for a "
                                    "8,192-token conversation? (1 MiB = 1,048,576 bytes)",
             "answer": "96 MiB.", "why": "12,288 × 8,192 = 100,663,296 bytes = 96 MiB."},
            {"kind": "tf", "q": "“The KV cache memory grows with the square of the context length.”",
             "answer": "False.", "why": "It grows linearly: a fixed amount per token. It is the attention work that "
                                       "grows with the square."},
            {"kind": "short", "q": "A server holds 10 full-window conversations of this model at once. How much KV-cache "
                                   "memory is that?",
             "answer": "3,840 MiB (3.75 GiB).", "why": "10 × 384 MiB. This is why long contexts limit how many users "
                                                       "a server can handle."},
        ],
    },
    {
        "title": "When the chat overflows",
        "segment": (81, 104),
        "figures": [{"t": 92.0, "caption": "The usual fix: keep the system prompt and drop the oldest turns."},
                    {"t": 103.0, "caption": "Smarter apps summarise old turns, or store them and retrieve only what "
                                            "matters."}],
        "body": [
            """<p>When a chat outgrows the window, something has to go. The usual fix: <b>keep the system prompt</b>
and <b>drop the oldest turns</b>. The model simply forgets how the conversation started.</p>""",
            '<div class="figrow">{fig0}{fig1}</div>',
            """<p>Smarter apps <b>summarise</b> the old turns into a short note, or <b>store</b> them and bring back
only the pieces that matter: that is <b>retrieval</b>, which we build in episode 5.</p>""",
            """<div class="box key"><b class="t">Key idea</b>Dropped text is gone for the model. Summaries and
retrieval keep the important parts inside the window.</div>""",
        ],
        "exercises": [
            {"kind": "mc", "q": "Why does the trimming code never drop the system prompt?",
             "options": ["It cannot be tokenized", "It sets the model's role and rules for every reply",
                         "It is always the shortest message", "The window does not count it"],
             "answer": "B.", "why": "Losing it would change how the model behaves, not only what it remembers."},
            {"kind": "short", "q": "In a long chat, the oldest turns (where you told the assistant you are vegetarian) "
                                   "were dropped. You ask for a dinner idea. What can go wrong, and name one fix?",
             "answer": "It may suggest meat, because that fact is no longer in the context. Fix: summarise old turns "
                       "(keeping “vegetarian”), or retrieve relevant facts.",
             "why": "If it is not in the context, the model cannot see it."},
            {"kind": "order", "q": "Put the trimming loop in order: <i>drop the oldest exchange · count the tokens · "
                                   "stop when under budget · keep the system prompt aside</i>.",
             "answer": "keep the system prompt aside → count the tokens → drop the oldest exchange → stop when under "
                       "budget (repeat counting and dropping until it fits).",
             "why": "This is the <code>fit</code> function of the episode's code."},
        ],
    },
    {
        "title": "Trimming in code, and practical lessons",
        "segment": (104, 139),
        "figures": [{"t": 116.0, "caption": "Trimming a 133-token chat to a 100-token budget leaves 82 tokens."},
                    {"t": 132.0, "caption": "Models tend to use the start and end of a long context better than the "
                                            "middle (illustrative shape)."}],
        "body": [
            """<p>In code: count the tokens; while the chat is over budget, drop the oldest exchange, but never the
system prompt. In the episode's code a 10-message chat of <b>133 tokens</b> is trimmed to fit a budget of
<b>100</b>: it keeps <b>82 tokens</b>, 6 of the 10 messages.</p>""",
            '<div class="figrow">{fig0}{fig1}</div>',
            """<p>Bigger windows help, but they are not free: long requests are <b>slower</b> and <b>cost more</b>,
and models tend to use information at the <b>start and end</b> of a long context better than in the
<b>middle</b>. So keep the context <b>lean</b>, and put what matters most where the model will notice it.</p>""",
            """<div class="box key"><b class="t">Key idea</b>Fit the budget deliberately, and spend the window on what
matters.</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "The trimmed chat keeps 82 of the original 133 tokens. How many tokens were "
                                    "dropped?",
             "answer": "51.", "why": "133 − 82 = 51: the two oldest user + assistant exchanges."},
            {"kind": "tf", "q": "“With a 1-million-token window, it no longer matters where in the prompt you put the "
                                "key instruction.”",
             "answer": "False.", "why": "Models tend to use the start and end of long contexts better than the middle, "
                                       "and long prompts are slower and cost more."},
            {"kind": "code", "q": "<b>Try it yourself.</b> In <code>code/p02_context_window/fit_the_window.py</code>, "
                                  "change <code>budget = 100</code> to <code>budget = 60</code> and run it. (a) How many "
                                  "messages are kept? (b) Which user question is the first one kept? (c) What happens "
                                  "with <code>budget = 30</code>?",
             "code": """budget = 60
kept = fit(chat, budget)
print(count(kept), len(kept))
for m in kept:
    print(m["role"], m["content"])""",
             "answer": "(a) 4 messages, 53 tokens: the system prompt, one exchange and the last question. "
                       "(b) “What about poached eggs?”. (c) Only the system prompt and the last question remain: "
                       "2 messages, 28 tokens.",
             "why": "The loop drops whole exchanges from the oldest end until the chat fits. It never drops the system "
                    "prompt or the newest question, so if even those two did not fit, a real app would have to shorten "
                    "them or reject the request."},
        ],
    },
]
