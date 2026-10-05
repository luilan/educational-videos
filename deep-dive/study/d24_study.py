"""Study guide content for How LLMs Work: Deep Dive, episode 24: Where Training Data Comes From.

Build:  python framework/study_guide.py deep-dive d24 --video deep-dive/media/videos/d24_scene/1080p60/TrainingDataVideo.mp4
Figures: {"t": seconds into the rendered video, "caption": ...}; placed in body blocks with {fig0}, {fig1}, ...
Exercise kinds: mc (options), tf, short, number, order, code. Every exercise has an answer and a why.
Every number comes from code/d24_training_data/training_data.py (Tiny Shakespeare; GPT-2's tokenizer; a 4-layer tiny GPT
trained 2,000 steps; transformers 4.57.1, torch 2.14.0, CPU), except Meta's public figure for Llama 3.
"""

LESSON = {
    "series": "How LLMs Work: Deep Dive",
    "label": "Episode 24",
    "title": "Where Training Data Comes From",
    "tagline": "Quantity, quality, and copies",
    "duration": "2:18",
    "intro": """<p>This lesson answers one question: where does training text come from, and does its quality matter?
Large models train on trillions of tokens, mostly from the filtered public web. Two experiments show why filtering
matters: mixing junk into the training data raises the loss on clean text (1.644 → 1.776 with half junk), and a passage
repeated 1,000 times is memorized word for word, the main reason training data is deduplicated.</p>""",
    "prereq": """<div class="box note"><b>Before you start:</b> Deep Dive episodes 1 (tokenization) and 17 (the loss).
Code: <code>code/d24_training_data</code> (GPT-2's tokenizer plus seven tiny models, about 35 minutes on a CPU).</div>""",
}

CONCEPTS = [
    {
        "title": "Quantity and sources",
        "segment": (8, 51),
        "figures": [{"t": 36.0, "caption": "Sources (web, books, code, papers) and filters (language, quality, personal "
                                           "information, duplicates)."},
                    {"t": 50.0, "caption": "Even Tiny Shakespeare repeats itself: 23% of its lines appear more than once."}],
        "body": [
            """<p>Our Tiny Shakespeare is 1,115,394 characters, <b>338,025</b> GPT-2 tokens. Meta says Llama 3 was trained
on more than <b>15 trillion</b> tokens. That much text comes mostly from the public web, crawled at enormous scale, plus
books, code and papers; then it is filtered hard: by language, by quality, by removing personal information, and by
removing duplicates.</p>""",
            '<div class="figrow">{fig0}{fig1}</div>',
            """<p>Duplicates are everywhere. Even in our small file, 7,388 of the 32,777 non-empty lines (<b>23%</b>)
appear more than once, mostly speaker names: “GLOUCESTER:” 229 times, “DUKE VINCENTIO:” 193. On the web it is
boilerplate, mirrors, and copies of copies.</p>""",
            """<div class="box key"><b class="t">Key idea</b>Raw text is plentiful; good, unique text is what filtering is
for.</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "How many times larger than Tiny Shakespeare (in GPT-2 tokens) is a 15-trillion-token "
                                    "dataset?",
             "answer": "About 44 million times.", "why": "15 × 10¹² / 338,025 ≈ 4.4 × 10⁷."},
            {"kind": "number", "q": "About how many characters per GPT-2 token does Tiny Shakespeare have?",
             "answer": "About 3.3.", "why": "1,115,394 / 338,025."},
        ],
    },
    {
        "title": "Quality: junk in the data",
        "segment": (51, 70),
        "figures": [{"t": 69.2, "caption": "Clean 1.644; 25% junk 1.719; 50% junk 1.776 (loss on clean validation text)."}],
        "body": [
            """<p>Part of every training batch is replaced by junk (random characters), and the loss is measured on clean
validation text after 2,000 steps: clean data <b>1.644</b>, a quarter junk <b>1.719</b>, half junk <b>1.776</b>. Junk
wastes compute (those sequences teach nothing useful) and it hurts the model.</p>""",
            "{fig0}",
            """<div class="box key"><b class="t">Key idea</b>Every bad token costs training time and pulls the model away
from the text you care about.</div>""",
        ],
        "exercises": [
            {"kind": "short", "q": "With 50% junk, the model sees half as much real text in the same number of steps. How "
                                   "much of the damage could that alone explain?",
             "answer": "Most of it: in episode 23, halving a similar model's data (4.1M → 2.0M tokens) raised its loss by "
                       "0.11 (1.684 → 1.798); here half junk cost 0.13 (1.644 → 1.776).",
             "why": "At this scale, junk mainly wastes the steps it occupies; the rest of the gap may be the junk's own "
                    "effect."},
        ],
    },
    {
        "title": "Copies and memorization",
        "segment": (70, 129),
        "figures": [{"t": 97.2, "caption": "Never seen: 48% of next characters right; 200 copies: 94%, but generation drifts "
                                           "after one mistake."},
                    {"t": 111.7, "caption": "1,000 copies: 100%, and the next 45 characters written word for word; other text "
                                            "barely changes."}],
        "body": [
            """<p>One 65-character passage from the validation text, which the model normally never sees, is slipped into
training again and again. Never seen: loss <b>1.558</b> on the passage, next character right <b>48%</b> of the time.
200 copies: 0.467, 94% right, but started from its first 20 characters the model drifts off after one mistake (2 of 45
characters right).</p>""",
            """<p><b>1,000 copies</b>: loss 0.062, 100% right, and from the first 20 characters it writes the next
<b>45 exactly</b>. Meanwhile the loss on everything else barely moves (1.644 → 1.657). The model has memorized one
passage word for word: repeated text wastes training, and what a model memorizes it can repeat, private details or
copyrighted pages. That is why data is deduplicated.</p>""",
            '<div class="figrow">{fig0}{fig1}</div>',
            """<div class="box key"><b class="t">Key idea</b>Repetition turns prediction into memorization; deduplication
prevents it.</div>""",
        ],
        "exercises": [
            {"kind": "order", "q": "Order by how well the passage was memorized: <i>200 copies · never · 1,000 copies</i>.",
             "answer": "1,000 copies (100%, 45/45) → 200 copies (94%, 2/45) → never (48%).", "why": "From the episode's run."},
            {"kind": "tf", "q": "“A 94% next-character accuracy guarantees the model can reproduce the passage.”",
             "answer": "False.", "why": "One early mistake sends generation elsewhere: at 200 copies it wrote only 2 of 45 "
                                       "characters right."},
            {"kind": "code", "q": "<b>Try it yourself.</b> In <code>training_data.py</code>, change the repetition list to "
                                  "<code>(5,)</code> (400 copies). Is the passage reproduced?",
             "code": """for every in (5,):
    ...""",
             "answer": "Yes: loss 0.134, 98% of next characters right, and 45/45 written word for word. The threshold "
                       "lies between 200 and 400 copies for this model and passage.",
             "why": "Memorization grows with repetition; the exact threshold depends on the model and the passage."},
        ],
    },
]
