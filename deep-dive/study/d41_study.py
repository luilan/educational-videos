"""Study guide content for How LLMs Work: Deep Dive, episode 41: The Logit Lens.

Build:  python framework/study_guide.py deep-dive d41 --video deep-dive/media/videos/d41_scene/1080p60/LogitLensVideo.mp4
Figures: {"t": seconds into the rendered video, "caption": ...}; placed in body blocks with {fig0}, {fig1}, ...
Exercise kinds: mc (options), tf, short, number, order, code. Every exercise has an answer and a why.
Every number comes from code/d41_logit_lens/logit_lens.py (GPT-2 small and Qwen2.5-0.5B, float32; layer 0 = the
embeddings; "out" = the model's real output).
"""

LESSON = {
    "series": "How LLMs Work: Deep Dive",
    "label": "Episode 41",
    "title": "The Logit Lens",
    "tagline": "Watching a prediction form, layer by layer",
    "duration": "2:12",
    "intro": """<p>This lesson answers one question: what does a transformer “think” halfway through? The <b>logit lens</b>
decodes the residual stream after every layer with the model's own final norm and unembedding. In GPT-2 you can watch
“the” turn into Rome, London, then Paris, and see the last layer pull a 100%-sure “Shakespeare” back to 21%. Across real
text, most decisions settle late. In Qwen2.5-0.5B the lens shows junk until the last few layers: a reminder that it is a
quick, imperfect window.</p>""",
    "prereq": """<div class="box note"><b>Before you start:</b> Deep Dive episodes 13 (the residual stream) and 14
(LayerNorm vs RMSNorm). Code: <code>code/d41_logit_lens</code> (GPT-2 small and Qwen2.5-0.5B).</div>""",
}

CONCEPTS = [
    {
        "title": "Reading every layer",
        "segment": (8, 34),
        "figures": [{"t": 32.5, "caption": "The final norm and unembedding, normally applied only after the last layer, applied "
                                           "after every layer."}],
        "body": [
            """<p>After each layer the residual stream holds one vector per token, and only the last one is normally
turned into word probabilities, by the final norm and the unembedding matrix. The logit lens applies the same two steps
to the vector after <i>any</i> layer, as if the model stopped there. No training is involved.</p>""",
            "{fig0}",
            """<div class="box key"><b class="t">Key idea</b>Because every layer adds to the same residual stream, its
intermediate states live in roughly the same space as the output, so they can be decoded.</div>""",
        ],
        "exercises": [
            {"kind": "tf", "q": "“The logit lens needs to be trained on the model's activations.”",
             "answer": "False.", "why": "It reuses the model's own final norm and unembedding; the tuned lens is the trained "
                                       "variant.", "key": {'value': False}},
        ],
    },
    {
        "title": "Watching GPT-2 decide",
        "segment": (34, 82),
        "figures": [{"t": 49.9, "caption": "“…in the city of”: “the” for layers 1–6, then Rome (layer 9), London (10), Paris "
                                           "(11)."},
                    {"t": 66.5, "caption": "“…written by William”: “William” echoed until layer 7; Shakespeare 0.36 → 0.91 → "
                                           "1.00, then 0.21 at the output."},
                    {"t": 80.7, "caption": "Over 1,024 tokens: lens top-1 = final prediction 7% at layer 5, 30% at 8, 56% at "
                                           "11."}],
        "body": [
            """<p>On “The Eiffel Tower is in the city of”, GPT-2's lens guess is filler (“the”) for most early layers,
then Rome at layer 9, London at 10, and Paris from layer 11. On “Romeo and Juliet was written by William”, the early
layers echo the last word; “Shakespeare” takes over at layer 8 and reaches 1.00 at layer 10, but the final layer pulls it
back to 0.21. On “The capital of Japan is”, layer 11 puts “Tokyo” at 0.46 while the real output prefers “the”, with
Tokyo at 0.07. The last layer hedges.</p>""",
            '<div class="figrow">{fig0}{fig1}</div>',
            """<p>Across 1,024 tokens of Tiny Shakespeare, the lens's top token agrees with the final prediction 7% of the
time at layer 5, 30% at layer 8 and 56% at layer 11. Most of the decision is made late.</p>""",
            "{fig2}",
            """<div class="box key"><b class="t">Key idea</b>Predictions sharpen in the later layers; the very last layer
often spreads probability out again (calibration).</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "In GPT-2, by how many percentage points does the lens's agreement with the final "
                                    "prediction grow from layer 5 to layer 11?",
             "answer": "About 49 points (6.9% → 56.3%).", "why": "Read off the agreement curve.", "key": {'parts': [{'label': None, 'value': 49, 'tol': 0.98, 'unit': None}]}},
            {"kind": "short", "q": "GPT-2's layer 11 gives “Tokyo” 0.46, but the output picks “the”. Give a plausible reason.",
             "answer": "The last layer hedges toward continuations like “the capital … is the city of Tokyo”, spreading "
                       "probability instead of committing.", "why": "The output must be calibrated over all possible "
                                                                    "continuations, not just the fact."},
        ],
    },
    {
        "title": "Where the lens fails",
        "segment": (82, 121),
        "figures": [{"t": 97.5, "caption": "Qwen2.5-0.5B: agreement below 3% until layer 16; “Paris” first on top at layer 22 of "
                                           "24."},
                    {"t": 112.4, "caption": "The tuned lens trains a small translator per layer."}],
        "body": [
            """<p>Qwen2.5-0.5B tells a different story: for most of its 24 layers the lens shows code fragments and
pieces of words (“().'/”, “PushMatrix”), agreeing with the final answer under 3% of the time until layer 16; Paris only
appears at layer 22. That doesn't mean Qwen knows nothing until then: its middle layers likely use directions the
unembedding can't read. The <b>tuned lens</b> fixes this by training a small affine translator per layer.</p>""",
            '<div class="figrow">{fig0}{fig1}</div>',
            """<p>In code the lens is one line: <code>model.lm_head(model.transformer.ln_f(h))</code> for any hidden state
h.</p>""",
            """<div class="box key"><b class="t">Key idea</b>A readable lens shows what a layer represents; an unreadable
one doesn't prove the layer is empty.</div>""",
        ],
        "exercises": [
            {"kind": "mc", "q": "Which conclusion does Qwen's junk lens output support?",
             "options": ["Qwen's middle layers do nothing", "The unembedding can't decode Qwen's middle-layer directions",
                         "Qwen is a worse model than GPT-2", "The lens has a bug"],
             "answer": "B.", "why": "Qwen still answers correctly; the information is there but not in output-readable "
                                   "directions.", "key": {'choice': 1}},
            {"kind": "code", "q": "<b>Try it yourself.</b> Run the lens on your own prompt.",
             "code": """from transformers import GPT2LMHeadModel, GPT2TokenizerFast
tok = GPT2TokenizerFast.from_pretrained("gpt2"); model = GPT2LMHeadModel.from_pretrained("gpt2")
ids = tok("The capital of Italy is", return_tensors="pt").input_ids
hs = model(ids, output_hidden_states=True).hidden_states
for i, h in enumerate(hs[:-1]):
    print(i, tok.decode(model.lm_head(model.transformer.ln_f(h[0, -1])).argmax()))""",
             "answer": "A list of 12 guesses, mostly filler early on, with the answer appearing in the last layers.",
             "why": "Note hidden_states[-1] already has the final norm applied, so it is skipped here."},
        ],
    },
]
