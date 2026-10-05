"""Study guide content for LLMs in Practice, episode 6: Chunking and Retrieval.

Build:  python framework/study_guide.py llms-in-practice p06 --video llms-in-practice/media/videos/p06_scene/1080p60/ChunkingVideo.mp4
Figures: {"t": seconds into the rendered video, "caption": ...}; placed in body blocks with {fig0}, {fig1}, ...
Exercise kinds: mc (options), tf, short, number, order, code. Every exercise has an answer and a why.
Every result comes from code/p06_chunking (three made-up handbooks, all-MiniLM-L6-v2, transformers 4.57.1,
torch 2.14.0, CPU).
"""

LESSON = {
    "series": "LLMs in Practice",
    "label": "Episode 6",
    "title": "Chunking and Retrieval",
    "tagline": "Why RAG fails, and how to fix it",
    "duration": "2:39",
    "intro": """<p>This lesson answers one question: why does RAG fail on real documents, and how do you fix it?
Long documents must be cut into <b>chunks</b> before retrieval, and the way you cut decides whether the right fact is
found. In a measured test, <b>whole documents</b> waste the prompt and silently overflow the embedding model,
<b>fixed-size</b> chunks cut facts in half, and chunks that follow the text's own <b>structure</b> (sections, or
sentences with their neighbours) find every answer with few words.</p>""",
    "prereq": """<div class="box note"><b>Before you start:</b> episodes 2 (context windows), 4 (embedding search) and
5 (RAG) of this series. Code: <code>code/p06_chunking</code> (90 MB model, CPU).</div>""",
}

CONCEPTS = [
    {
        "title": "The experiment",
        "segment": (8, 41),
        "figures": [{"t": 39.5, "caption": "Three made-up handbooks, ten questions about the bakery with known answers, "
                                           "and two measurements."}],
        "body": [
            """<p>Real documents are long: handbooks, manuals, contracts. You cannot paste a whole library into the
prompt, so you <b>retrieve pieces</b>, and how you cut those pieces decides whether RAG works at all.</p>""",
            """<p>The test: a small library of three made-up handbooks (a bakery, a bike shop, a gym; 467 words) and ten
questions about the bakery, each with a known answer. For each way of cutting the text, we check two things: is the
answer in the <b>top chunk</b>, and how many <b>words go into the prompt</b>.</p>""",
            "{fig0}",
            """<div class="box key"><b class="t">Key idea</b>Judge a retrieval setup with questions whose answers you
know: is the answer retrieved, and at what cost?</div>""",
        ],
        "exercises": [
            {"kind": "short", "q": "Why are the questions “about the bakery” worded carefully, e.g. “Do students get a "
                                   "discount <b>at the bakery</b>?”",
             "answer": "The gym also has student prices; without “at the bakery” the question is ambiguous and a gym "
                       "chunk would be a fair answer.",
             "why": "A test only measures retrieval if each question has one right answer."},
            {"kind": "tf", "q": "“Only the first measurement (answer found) matters; the number of words sent is "
                                "irrelevant.”",
             "answer": "False.", "why": "Every extra word costs time and money and can bury the key fact (episode 2)."},
        ],
    },
    {
        "title": "Whole documents: found, but wasteful and truncated",
        "segment": (41, 65),
        "figures": [{"t": 64.0, "caption": "The bakery handbook is 327 tokens; the embedding model reads 256. The end "
                                           "never reaches the vector."}],
        "body": [
            """<p>With no cutting (one vector per handbook) retrieval found the right handbook every time: 10 / 10.
But it sends all <b>255 words</b> for every question, about nine times more than needed.</p>""",
            "{fig0}",
            """<p>And there is a hidden problem: this embedding model reads at most <b>256 tokens</b>. The bakery
handbook is <b>327</b>. Everything after <i>“we accept cash, cards and”</i> (the student discount, the BELLA10 code,
the job ad) never makes it into the vector, and nothing warns you.</p>""",
            """<div class="box key"><b class="t">Key idea</b>Embedding models have their own input limit. Text beyond it is
silently ignored, so long chunks hide their endings.</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "The handbook is 327 tokens and the model reads 256. What fraction of the handbook is "
                                    "ignored, to the nearest percent?",
             "answer": "22%.", "why": "71 / 327 ≈ 0.217."},
            {"kind": "mc", "q": "Whole-document retrieval still scored 10 / 10, even for questions about the ignored end. "
                                "Why?",
             "options": ["The ignored text is read later by the LLM", "The rest of the handbook is clearly about the "
                         "bakery, so the bakery document still wins against the bike shop and the gym",
                         "The model memorised the handbook", "Truncation never happens with sentence models"],
             "answer": "B.", "why": "With only three very different documents, topic alone picks the right one. In a "
                                   "library of many bakery documents it would not."},
        ],
    },
    {
        "title": "Fixed-size chunks cut facts in half",
        "segment": (65, 86),
        "figures": [{"t": 84.5, "caption": "Real 30-word cuts: “Students get a 10 | percent discount” and “a morning "
                                           "baker. The | shift starts at 4:00”."}],
        "body": [
            """<p>Cutting every <b>30 words</b>, wherever the cut falls, gives small chunks but only <b>8 / 10</b>. Look
at the cuts: one chunk ends <i>“Students get a 10”</i> and the word <i>percent</i> is in the next chunk; another ends
<i>“We are hiring a morning baker. The”</i> and the start time, 4:00, is in the next one. The facts were cut in
half.</p>""",
            "{fig0}",
            """<div class="box key"><b class="t">Key idea</b>Cuts that ignore the text's structure separate a fact from
the words that make it findable.</div>""",
        ],
        "exercises": [
            {"kind": "short", "q": "For “What time does the morning baker start?”, which chunk did fixed 30-word "
                                   "chunking rank first, and why is that a failure?",
             "answer": "The bakery's opening-hours chunk. The chunk with “starts at 4:00” lost the words “morning baker”, "
                       "so it no longer looked like an answer to the question.",
             "why": "The question's key words and the answer ended up in different chunks."},
            {"kind": "tf", "q": "“Smaller fixed chunks are always more precise, so 15 words beats 30.”",
             "answer": "False.", "why": "In the same test, 15-word chunks found only 4 / 10 answers: more cuts, more "
                                       "facts split."},
        ],
    },
    {
        "title": "Cut along the text's own structure",
        "segment": (86, 131),
        "figures": [{"t": 99.5, "caption": "One chunk per section: 10 / 10 with 38 words. Sentence + neighbours: 10 / 10 "
                                           "with 28 words."},
                    {"t": 130.5, "caption": "Other failures and their fixes."}],
        "body": [
            """<p>Cutting along the text's own structure works: <b>one chunk per section</b>, with its heading,
finds <b>10 / 10</b> with about <b>38 words</b> each; <b>each sentence with its neighbours</b>, so the chunks overlap,
finds <b>10 / 10</b> with <b>28 words</b>.</p>""",
            '<div class="figrow">{fig0}{fig1}</div>',
            """<p>So: cut at <b>natural boundaries</b>; keep the <b>headings</b>, they carry the topic; <b>overlap</b> a
little, so no fact falls between two chunks; size chunks to hold <b>one answer</b>. And remember the other failures:
the question may use different words than the document (rewrite it, or add keyword search); the answer may need two
chunks (retrieve more); retrieve too much and the key fact gets lost in the middle.</p>""",
            """<div class="box key"><b class="t">Key idea</b>Good chunks are self-contained: one topic, its heading, and
enough neighbouring text that no fact is split.</div>""",
        ],
        "exercises": [
            {"kind": "order", "q": "Order the four strategies from most to fewest words sent per question: <i>sentence + "
                                   "neighbours · whole document · one per section · fixed 30 words</i>.",
             "answer": "whole document (255) → one per section (38) → fixed 30 words (30) → sentence + neighbours (28).",
             "why": "Values from the episode's table."},
            {"kind": "mc", "q": "A user asks “refund rules?” but the policy says “returns are accepted within 14 days”. "
                                "Which fix targets this failure?",
             "options": ["Bigger chunks", "Rewrite the question (or add keyword search with synonyms)",
                         "Fewer chunks", "A higher temperature"],
             "answer": "B.", "why": "The problem is different words for the same thing, not chunk size."},
        ],
    },
    {
        "title": "Measure what works",
        "segment": (131, 156),
        "figures": [{"t": 139.0, "caption": "The two good chunkers are a few lines each."}],
        "body": [
            """<p>In code, the good chunkers are short: split on <b>blank lines</b> for sections, or split into
<b>sentences</b> and join each with its <b>neighbours</b>.</p>""",
            "{fig0}",
            """<p>Most of all, <b>measure</b>: write a few questions with known answers, check what retrieval returns,
change one thing at a time, and keep what works.</p>""",
            """<div class="box key"><b class="t">Key idea</b>Chunking choices are empirical: a ten-question test catches
problems that are invisible by inspection.</div>""",
        ],
        "exercises": [
            {"kind": "code", "q": "<b>Try it yourself.</b> Run this at the end of <code>chunking.py</code>. (a) What scores "
                                  "do 60-word chunks, 15-word chunks and single sentences (no neighbours) get? (b) Which "
                                  "question do single sentences miss, and why do neighbours fix it?",
             "code": """for label, fn in [("fixed 60", lambda t: fixed(t, 60)),
                  ("fixed 15", lambda t: fixed(t, 15)),
                  ("single sentences", sentences_of)]:
    chunks = chunk_library(fn)
    print(label, evaluate(chunks, show_misses=True))""",
             "answer": "(a) Fixed 60: 9 / 10 (60 words sent); fixed 15: 4 / 10 (15 words); single sentences: 9 / 10 "
                       "(about 12 words). (b) “What time does the morning baker start?”: “We are hiring a morning baker.” "
                       "and “The shift starts at 4:00, …” are separate sentences, so the chunk that matches the question "
                       "does not contain the time. A sentence + neighbours chunk contains both.",
             "why": "The output also lists each miss with the chunk that was ranked first."},
        ],
    },
]
