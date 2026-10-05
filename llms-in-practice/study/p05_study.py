"""Study guide content for LLMs in Practice, episode 5: RAG.

Build:  python framework/study_guide.py llms-in-practice p05 --video llms-in-practice/media/videos/p05_scene/1080p60/RagVideo.mp4
Figures: {"t": seconds into the rendered video, "caption": ...}; placed in body blocks with {fig0}, {fig1}, ...
Exercise kinds: mc (options), tf, short, number, order, code. Every exercise has an answer and a why.
Real outputs come from code/p05_rag (all-MiniLM-L6-v2 retrieval; Qwen2.5-1.5B/0.5B-Instruct, greedy; transformers
4.57.1, torch 2.14.0, CPU). Bella's Bakery is fictional.
"""

LESSON = {
    "series": "LLMs in Practice",
    "label": "Episode 5",
    "title": "RAG: Giving the Model a Library",
    "tagline": "Search first, then answer",
    "duration": "2:26",
    "intro": """<p>This lesson answers one question: how can a model answer from information it never saw in
training? <b>RAG</b>, retrieval-augmented generation, works in two steps: <b>search</b> a library for the passages that
match the question (episode 4), then <b>paste them into the prompt</b> and let the model answer from them. It works
without retraining, but only if retrieval finds the right pieces and the model reads them carefully.</p>""",
    "prereq": """<div class="box note"><b>Before you start:</b> episode 1 (everything is context) and episode 4
(embedding search) of this series. Code: <code>code/p05_rag</code> (downloads two small models, about 4 GB in total;
runs on a CPU).</div>""",
}

CONCEPTS = [
    {
        "title": "The gap RAG fills",
        "segment": (8, 32),
        "figures": [{"t": 31.3, "caption": "Without RAG, the real Qwen2.5-1.5B answer: it cannot know about a bakery it "
                                           "never saw."}],
        "body": [
            """<p>A model only knows what was in its <b>training data</b>, up to a <b>cutoff date</b>. It has never seen
your company's documents, your notes or today's news. Asked <i>“Can I buy gluten-free bread at Bella's Bakery on
Wednesday?”</i> about a made-up bakery, the model can only say it does not know, or worse, <b>invent</b> something
that sounds right.</p>""",
            "{fig0}",
            """<div class="box key"><b class="t">Key idea</b>A model cannot answer from facts that are neither in its
training data nor in its context.</div>""",
        ],
        "exercises": [
            {"kind": "mc", "q": "Which question can a model answer reliably <b>without</b> RAG?",
             "options": ["What did my manager email me this morning?", "What is the capital of France?",
                         "What are today's opening hours of my local bakery?", "What is in our company's new policy?"],
             "answer": "B.", "why": "Only B is common knowledge likely to be in the training data."},
            {"kind": "tf", "q": "“Saying ‘I don't know’ is the worst thing a model can do when it lacks the facts.”",
             "answer": "False.", "why": "Inventing a plausible but false answer is worse: it can mislead you."},
        ],
    },
    {
        "title": "Retrieve, then generate",
        "segment": (32, 80),
        "figures": [{"t": 63.4, "caption": "Retrieval: the real similarity scores of the six facts; the top two go into "
                                           "the prompt."},
                    {"t": 79.0, "caption": "The prompt: instructions, the retrieved facts, the question. 101 tokens."}],
        "body": [
            """<p><b>RAG</b> has two steps. First, <b>search</b> a library of documents for the parts that match the
question. Then <b>paste them into the prompt</b>, and let the model answer from them. The search is episode 4's
embedding search: the bakery library has six facts, and the closest two are <i>“Gluten-free bread is only available on
Fridays”</i> (<b>0.70</b>) and the opening hours (<b>0.62</b>).</p>""",
            '<div class="figrow">{fig0}{fig1}</div>',
            """<p>The prompt has an <b>instruction</b> (answer only from the information below; say so if the answer is
not there), the <b>retrieved facts</b>, and the <b>question</b>: 101 tokens, all of it context, as in episode 1.</p>""",
            """<div class="box key"><b class="t">Key idea</b>RAG = retrieval puts the right text into the context;
generation reads it.</div>""",
        ],
        "exercises": [
            {"kind": "order", "q": "Put the RAG steps in order: <i>generate the answer · embed the question · build the "
                                   "prompt · embed the library (once) · take the top-k matches</i>.",
             "answer": "embed the library (once) → embed the question → take the top-k matches → build the prompt → "
                       "generate the answer.",
             "why": "Library vectors are computed ahead of time; the rest happens per question."},
            {"kind": "short", "q": "Why does the prompt say “If the answer is not there, say you don't know”?",
             "answer": "To stop the model from guessing when retrieval did not find the answer.",
             "why": "Without it, the model may fill the gap with something invented."},
            {"kind": "number", "q": "Retrieval scores for six facts are 0.62, 0.60, 0.41, 0.70, 0.17 and 0.18. With "
                                    "k = 3, what is the lowest score that still makes it into the prompt?",
             "answer": "0.60.", "why": "The top three are 0.70, 0.62 and 0.60."},
        ],
    },
    {
        "title": "Right fact, wrong answer",
        "segment": (80, 114),
        "figures": [{"t": 93.5, "caption": "Qwen2.5-1.5B reads the retrieved fact and answers correctly."},
                    {"t": 113.0, "caption": "With the exact same prompt, the 0.5B model answers wrongly."}],
        "body": [
            """<p>With the retrieved facts, <b>Qwen2.5-1.5B-Instruct</b> answers: <i>“No, you cannot buy gluten-free
bread at Bella's Bakery on Wednesday because it is only available on Fridays.”</i> Correct, from a fact it never saw in
training.</p>""",
            '<div class="figrow">{fig0}{fig1}</div>',
            """<p>But the even smaller <b>0.5B</b> model, given the exact same prompt, said <i>yes</i>, because the
bakery is open on weekdays. It had the right fact and still got it wrong. Retrieval only helps if the model
<b>reads carefully</b>: check answers, and ask for <b>sources</b>.</p>""",
            """<div class="box key"><b class="t">Key idea</b>RAG can fail at either step: retrieving the wrong text, or
reading the right text wrongly.</div>""",
        ],
        "exercises": [
            {"kind": "mc", "q": "In the 0.5B example, which step of RAG failed?",
             "options": ["Retrieval: it found the wrong facts", "Generation: it misread the right facts",
                         "Tokenization", "The chat template"],
             "answer": "B.", "why": "The top retrieved fact was exactly the right one (0.70)."},
            {"kind": "short", "q": "Name two ways to catch answers like the 0.5B model's in a real product.",
             "answer": "For example: ask the model to quote the source sentence it used and check it; use a more capable "
                       "model; test on questions with known answers.",
             "why": "A quoted source makes a contradiction like “only on Fridays” → “yes on Wednesday” easy to spot."},
        ],
    },
    {
        "title": "Why RAG is everywhere",
        "segment": (114, 143),
        "figures": [{"t": 122.0, "caption": "RAG in a few lines: retrieve the top matches, join them into the prompt with "
                                            "the instructions, generate."},
                    {"t": 135.0, "caption": "Update a document and the next answer uses it, with no retraining "
                                            "(illustrative)."}],
        "body": [
            """<p>In code: embed the library once. For each question, retrieve the top matches, join them into the
prompt with the instructions, and generate.</p>""",
            '<div class="figrow">{fig0}{fig1}</div>',
            """<p>RAG is how most assistants answer from <b>private or fresh</b> information <b>without retraining</b>
the model: update a document, and the next answer uses it. But everything depends on <b>retrieving the right
pieces</b>, which is harder than it looks (next episode).</p>""",
            """<div class="box key"><b class="t">Key idea</b>To change what a RAG system knows, change the library, not
the model.</div>""",
        ],
        "exercises": [
            {"kind": "tf", "q": "“To teach a RAG assistant a new opening time, you must fine-tune the model.”",
             "answer": "False.", "why": "Update the document in the library; the next retrieval returns the new text."},
            {"kind": "code", "q": "<b>Try it yourself.</b> In <code>code/p05_rag/rag.py</code>, set <code>QUESTION</code> "
                                  "to each question below and run it. (a) Which two facts are retrieved for each, and what "
                                  "does the 1.5B model answer? (b) The second answer is not wrong, but it is unhelpful. "
                                  "Why, and how could you improve it?",
             "code": """QUESTION = "Is Bella's Bakery open on Monday at 8:00?"
QUESTION = "Do you deliver 3 km away for a 20 euro order?\"""",
             "answer": "(a) Monday: the weekday hours (0.88) and “closed on Mondays” (0.82); answer: “No, Bella's Bakery "
                       "is closed on Mondays.” Delivery: the delivery rule (0.80) and the sourdough price (0.28); answer: "
                       "“I don't know…”, although it quotes the rule. (b) The answer follows from the rule (20 euros is "
                       "under the 30-euro minimum, so no), but the model did not apply it. Fixes: a more capable model, or "
                       "an instruction such as “apply the rules to the question and explain your reasoning”.",
             "why": "Retrieval was right both times; the difference is in how well the model reasons over the text."},
        ],
    },
]
