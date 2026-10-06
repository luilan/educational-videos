"""Study guide content for LLMs in Practice, episode 9: Fine-Tuning and LoRA.

Build:  python framework/study_guide.py llms-in-practice p09 --video llms-in-practice/media/videos/p09_scene/1080p60/LoraVideo.mp4
Figures: {"t": seconds into the rendered video, "caption": ...}; placed in body blocks with {fig0}, {fig1}, ...
Exercise kinds: mc (options), tf, short, number, order, code. Every exercise has an answer and a why.
Every number comes from code/p09_lora (the tiny GPT trained 3,000 steps on Shakespeare, then 600 steps on made-up
recipes; torch 2.14.0, CPU).
"""

LESSON = {
    "series": "LLMs in Practice",
    "label": "Episode 9",
    "title": "Fine-Tuning and LoRA",
    "tagline": "When prompting isn't enough",
    "duration": "2:37",
    "intro": """<p>This lesson answers one question: when should you change the model itself, and how can you do it
cheaply? <b>Fine-tuning</b> keeps training a model on examples of the behaviour you want. On the tiny GPT, full
fine-tuning learns a new style but <b>forgets</b> the old one and produces a whole new model. <b>LoRA</b> freezes every
original weight and trains two thin matrices per layer: <b>4% of the parameters</b>, almost the same result, and an
adapter you can switch off or swap.</p>""",
    "prereq": """<div class="box note"><b>Before you start:</b> How LLMs Work episodes 11–12 (training and the tiny GPT)
and Foundations F03 (matrices). Code: <code>code/p09_lora</code> (CPU only; the first run trains the base model in
about 25 minutes).</div>""",
}

CONCEPTS = [
    {
        "title": "Prompting, RAG or fine-tuning?",
        "segment": (8, 38),
        "figures": [{"t": 37.0, "caption": "Prompting is cheap and instant; RAG adds knowledge; fine-tuning changes the "
                                           "weights."}],
        "body": [
            """<p>Prompts, documents and tool results all change what goes <b>into</b> the model; the model itself never
changes. <b>Fine-tuning</b> changes the <b>weights</b>, for when you want different behaviour every time without
saying so in every prompt.</p>""",
            "{fig0}",
            """<p><b>Prompting</b> is cheap and instant, and is the first thing to try. <b>RAG</b> adds knowledge.
<b>Fine-tuning</b> is for a style, a format or a skill that is hard to describe in words; it costs training time,
and you need examples.</p>""",
            """<div class="box key"><b class="t">Key idea</b>Change the input first; change the model only when the
behaviour cannot be prompted or retrieved.</div>""",
        ],
        "exercises": [
            {"kind": "mc", "q": "Your assistant must know this week's prices, which change every Monday. Best tool?",
             "options": ["Fine-tuning every week", "RAG over the current price list", "A higher temperature",
                         "Training a new model"],
             "answer": "B.", "why": "Changing facts belong in the library, not in the weights.", "key": {'choice': 1}},
            {"kind": "mc", "q": "You need every reply in a strict house style that is hard to describe, and you have "
                                "2,000 example replies. Best tool?",
             "options": ["Prompting alone", "RAG", "Fine-tuning (e.g. LoRA)", "Sampling at temperature 0"],
             "answer": "C.", "why": "A consistent style learned from many examples is what fine-tuning is for.", "key": {'choice': 2}},
        ],
    },
    {
        "title": "Full fine-tuning: learns, and forgets",
        "segment": (38, 72),
        "figures": [{"t": 50.0, "caption": "The Shakespeare-trained tiny GPT, asked for a recipe, writes Shakespeare "
                                           "(recipe loss 2.77)."},
                    {"t": 71.0, "caption": "Full fine-tuning: recipe loss 0.25, but Shakespeare loss 1.60 → 4.40."}],
        "body": [
            """<p>The experiment uses the tiny GPT from How LLMs Work, trained on Shakespeare. Asked for a recipe, it
writes Shakespeare; on the made-up recipes its loss is <b>2.77</b>.</p>""",
            '<div class="figrow">{fig0}{fig1}</div>',
            """<p><b>Full fine-tuning</b>: keep training on 600 steps of recipes, updating all <b>818,241</b> weights.
Recipe loss drops to <b>0.25</b> and it writes a perfect recipe. But its Shakespeare loss jumps from <b>1.60 to
4.40</b>: it <b>forgot</b> (this is called catastrophic forgetting). And you now have a whole new copy of the
model.</p>""",
            """<div class="box key"><b class="t">Key idea</b>Updating every weight for a new task can overwrite what the
model knew, and every task needs its own full copy.</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "By how much did the Shakespeare loss rise after full fine-tuning?",
             "answer": "2.80.", "why": "4.40 − 1.60 = 2.80.", "key": {'parts': [{'label': None, 'value': 2.8, 'tol': 0.005, 'unit': None}]}},
            {"kind": "tf", "q": "“After full fine-tuning on recipes, the model is better at both recipes and "
                                "Shakespeare.”",
             "answer": "False.", "why": "Recipes improved (2.77 → 0.25) but Shakespeare got much worse (1.60 → 4.40).", "key": {'value': False}},
        ],
    },
    {
        "title": "LoRA in one picture",
        "segment": (72, 111),
        "figures": [{"t": 92.0, "caption": "LoRA: W stays frozen; B × A, two thin matrices, adds a small correction."},
                    {"t": 109.5, "caption": "32,768 trainable numbers (4%), a 128 KB adapter, and almost the same "
                                            "recipe loss."}],
        "body": [
            """<p><b>LoRA</b> (low-rank adaptation): freeze every original weight. Next to each weight matrix W, add two
thin matrices, <b>A</b> (r × in) and <b>B</b> (out × r). Their product B × A is a small correction added to the frozen
output: <code>output = W x + (B A x) · scale</code>. B starts at zero, so at first nothing changes. Only A and B are
trained.</p>""",
            '<div class="figrow">{fig0}{fig1}</div>',
            """<p>With rank r = 4 on every attention and MLP layer, that is <b>32,768</b> trainable numbers, <b>4.0%</b>
of the model; the adapter is a <b>128 KB</b> file (the model is 3,196 KB). Recipe loss: <b>0.27</b>, almost the same
as full fine-tuning (0.25).</p>""",
            """<div class="box key"><b class="t">Key idea</b>Many useful changes are low-rank: two thin matrices can
express them with a small fraction of the parameters.</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "A frozen layer maps 128 inputs to 384 outputs (the attention's qkv). How many "
                                    "trainable numbers do A and B add with rank r = 4?",
             "answer": "2,048.", "why": "A is 4 × 128 = 512, B is 384 × 4 = 1,536; 512 + 1,536 = 2,048.", "key": {'parts': [{'label': None, 'value': 2048, 'tol': 0.5, 'unit': None}]}},
            {"kind": "short", "q": "Why does B start at zero?",
             "answer": "So that B × A is zero at the start: the model behaves exactly like the base model until training "
                       "begins.",
             "why": "The code confirms it: before training, the recipe loss is 2.77, identical to the base model."},
            {"kind": "number", "q": "A full 128 → 384 weight matrix has how many numbers, and what fraction of that is "
                                    "the rank-4 LoRA pair?",
             "answer": "49,152; about 4.2%.", "why": "128 × 384 = 49,152; 2,048 / 49,152 ≈ 0.042.", "key": {'parts': [{'label': 'numbers in full matrix', 'value': 49152, 'tol': 0.5, 'unit': None}, {'label': 'LoRA fraction', 'value': 4.2, 'tol': 0.084, 'unit': '%'}]}},
        ],
    },
    {
        "title": "Adapters you can switch and swap",
        "segment": (111, 157),
        "figures": [{"t": 125.0, "caption": "Adapter off: Shakespeare loss is exactly 1.60 again. One base model, many "
                                            "small adapters."},
                    {"t": 151.5, "caption": "When to use what."}],
        "body": [
            """<p>Switch the adapter off, and the Shakespeare loss is back to exactly <b>1.60</b>: the original weights
never changed. Keep one base model and <b>swap</b> small adapters: one for recipes, one for legal letters, one per
customer. In code, LoRA is a few lines: wrap a linear layer, freeze it, add A (random) and B (zero); the output is the
frozen layer plus x times A, times B, times a scale.</p>""",
            '<div class="figrow">{fig0}{fig1}</div>',
            """<p>So: start with <b>prompting</b>. Add <b>RAG</b> for knowledge, especially facts that change.
<b>Fine-tune with LoRA</b> for style, format and narrow skills, when you have hundreds of good examples.</p>""",
            """<div class="box key"><b class="t">Key idea</b>A LoRA adapter is a small, removable add-on: the base model
stays intact and shareable.</div>""",
        ],
        "exercises": [
            {"kind": "tf", "q": "“With the LoRA adapter switched on, the Shakespeare loss is also 1.60.”",
             "answer": "False.", "why": "With the adapter on it is 3.92: the adapter steers the model toward recipes. "
                                       "Only switched off is it exactly 1.60.", "key": {'value': False}},
            {"kind": "code", "q": "<b>Try it yourself.</b> In <code>code/p09_lora/lora.py</code>, change "
                                  "<code>add_lora(tuned, r=4)</code> to <code>r=1</code> and run it. (a) How many "
                                  "trainable parameters, and what recipe loss? (b) Compare a sample with rank 4's.",
             "code": """adapters = add_lora(tuned, r=1)""",
             "answer": "(a) 8,192 trainable parameters (1.0%, a 32 KB adapter); recipe loss 0.39 (rank 4: 0.27). "
                       "(b) The format is learned but sloppier, e.g. “RECIPE: GERO AND TAR: ELY AND PLUQ STEW … half a "
                       "minut or a warm pan.”",
             "why": "Rank sets the adapter's capacity: rank 1 is cheaper but cannot capture the change as well. (Exact "
                    "samples depend on the random state; the r = 1 numbers come from a run without the full "
                    "fine-tuning step before it.)"},
        ],
    },
]
