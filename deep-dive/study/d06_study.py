"""Study guide content for How LLMs Work: Deep Dive, episode 6: Long Context.

Build:  python framework/study_guide.py deep-dive d06 --video deep-dive/media/videos/d06_scene/1080p60/LongContextVideo.mp4
Figures: {"t": seconds into the rendered video, "caption": ...}; placed in body blocks with {fig0}, {fig1}, ...
Exercise kinds: mc (options), tf, short, number, order, code. Every exercise has an answer and a why.
Every number comes from code/d06_long_context/long_context.py (an 810,049-parameter RoPE GPT trained on Tiny
Shakespeare with a 64-character context; torch 2.14.0, CPU) or the same functions run with the exercise settings.
"""

LESSON = {
    "series": "How LLMs Work: Deep Dive",
    "label": "Episode 6",
    "title": "Long Context",
    "tagline": "Stretching RoPE to read past the training length",
    "duration": "2:56",
    "intro": """<p>This lesson answers one question: how can a model read texts longer than the ones it was trained on?
A tiny RoPE model trained on 64 characters is asked to read 256. Used as is, it falls apart past about position 96.
Three fixes are tested: <b>position interpolation</b> (squeeze the positions), <b>NTK-aware scaling</b> (raise RoPE's
base) and <b>interpolation plus a short fine-tune</b>. Each fix changes one number in the RoPE code; real models such as
Qwen2.5 use refined versions of the same ideas.</p>""",
    "prereq": """<div class="box note"><b>Before you start:</b> Deep Dive episode 5 (RoPE) and LLMs in Practice episode 2
(context window, KV cache). Code: <code>code/d06_long_context</code> (trains in a few minutes on a laptop CPU, or loads
the included <code>rope_gpt.pt</code>).</div>""",
}

CONCEPTS = [
    {
        "title": "The experiment",
        "segment": (8, 46),
        "figures": [{"t": 27.5, "caption": "Models are trained up to a fixed length (32,768 tokens for Qwen2.5) but used "
                                           "on longer texts."},
                    {"t": 45.4, "caption": "The test: a RoPE model trained on 64 characters reads 256, and its loss is "
                                           "measured at every position."}],
        "body": [
            """<p>Every model is trained up to some length; for Qwen2.5 that is <b>32,768</b> tokens. Yet models are used
on far longer texts. With RoPE, this is done by stretching the rotations. To see it at laptop scale, the episode trains a
tiny GPT on Shakespeare: <b>810,049</b> parameters, RoPE and no position table, trained on only <b>64</b> characters.
Then it reads <b>256</b>, four times longer, and the loss is measured at every position.</p>""",
            '<div class="figrow">{fig0}{fig1}</div>',
            """<div class="box key"><b class="t">Key idea</b>Loss by position shows exactly where a model stops
understanding a long text: low and flat is good, a climb means trouble.</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "The model is trained on 64 characters and tested on 256. By what factor is the test "
                                    "longer?",
             "answer": "4.", "why": "256 / 64 = 4; every method below is tuned for this factor.", "key": {'parts': [{'label': None, 'value': 4, 'tol': 0.5, 'unit': None}]}},
            {"kind": "tf", "q": "“The tiny model has a learned table with one vector per position.”",
             "answer": "False.", "why": "It has no position table at all: position comes only from RoPE rotating the "
                                       "queries and keys.", "key": {'value': False}},
        ],
    },
    {
        "title": "No change, then squeezing",
        "segment": (46, 84),
        "figures": [{"t": 61.5, "caption": "No change: loss 1.60 on the trained positions, then a climb to about 3.4."},
                    {"t": 83.5, "caption": "Position interpolation without training: about 3.4–3.6 everywhere, even "
                                           "in the first 64 positions."}],
        "body": [
            """<p><b>No change.</b> In the first 64 positions the loss is <b>1.60</b>; from 64 to 127 it averages
<b>2.15</b>; from 128 to 255, <b>3.23</b>. Past about position 96, the model falls apart: those rotation angles never
appeared in training.</p>""",
            """<p><b>Position interpolation</b> multiplies every position by 64/256 = ¼, so the 256 positions are squeezed
into the trained range. Without retraining, the result is bad: about <b>3.4</b> everywhere (3.43, 3.58, 3.59), even in
the first 64 positions. Neighbouring tokens are now only a quarter of a step apart, and the fast pairs can no longer tell
them apart.</p>""",
            '<div class="figrow">{fig0}{fig1}</div>',
            """<div class="box key"><b class="t">Key idea</b>Extrapolating shows the model angles it has never seen;
interpolating keeps the angles familiar but crowds neighbouring tokens together.</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "With position interpolation at scale ¼, which position's angles does token 200 get?",
             "answer": "50.", "why": "200 × ¼ = 50, inside the trained range 0–63.", "key": {'parts': [{'label': None, 'value': 50, 'tol': 0.5, 'unit': None}]}},
            {"kind": "mc", "q": "Why does interpolation without training hurt even the first 64 positions?",
             "options": ["The model's weights change", "Neighbouring tokens become ¼ step apart, which the fast pairs never "
                         "saw", "The text is longer", "The vocabulary changes"],
             "answer": "B.", "why": "Every position is scaled, short texts included; the model was trained on steps of 1.", "key": {'choice': 1}},
            {"kind": "code", "q": "<b>Try it yourself.</b> Load the trained model from <code>long_context.py</code> and "
                                  "try a milder squeeze: scale ½ on 128 characters. Does halving the squeeze save "
                                  "interpolation without training?",
             "code": """m = copy.deepcopy(base)
m.rope = Rope(scale=0.5)
lp = loss_by_position(m, T=128)
print(lp[:64].mean(), lp[64:].mean())""",
             "answer": "No: 2.66 and 2.71, still far worse than the untouched model's 1.60 on the first 64.",
             "why": "Even a half-step spacing is unfamiliar to the fast pairs; interpolation needs some training."},
        ],
    },
    {
        "title": "Raising the base, and a short fine-tune",
        "segment": (84, 123),
        "figures": [{"t": 103.5, "caption": "NTK-aware scaling, no training: about 1.6 up to position 127, then a rise."},
                    {"t": 122.0, "caption": "Interpolation + 200 training steps at 256: flat at 1.62, 1.56, 1.58."}],
        "body": [
            """<p><b>NTK-aware scaling</b> raises RoPE's base instead, from 10,000 to about <b>43,873</b> (10,000 × 4<sup>
32/30</sup> for head size 32). The slowest pair now turns exactly 4× slower, pair 1 only 1.10× slower, and the fastest
pair not at all. With no training, the loss stays near <b>1.6</b> up to position 127, twice the training length (1.63,
1.62), and only then rises (2.51 for 128–255).</p>""",
            """<p><b>Interpolation, then a short fine-tune:</b> just 200 steps on 256-character texts. Now the loss is flat
across the whole text: <b>1.62, 1.56, 1.58</b>. The model quickly learns the finer grid of angles; longer texts even
help a little, because each prediction has more context.</p>""",
            '<div class="figrow">{fig0}{fig1}</div>',
            """<div class="box key"><b class="t">Key idea</b>Change the slow pairs, keep the fast ones (NTK), or squeeze
everything and briefly retrain (interpolation + fine-tune): both beat doing nothing.</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "In the code, how much slower does the slowest RoPE pair turn after NTK scaling for a "
                                    "factor of 4?",
             "answer": "4× (exactly).", "why": "Its frequency drops from 0.000178 to 0.0000445 rad per token; the "
                                              "exponent HD/(HD − 2) is chosen so the slowest pair stretches by the full "
                                              "factor.", "key": {'parts': [{'label': None, 'value': 4, 'tol': 0.5, 'unit': None}]}},
            {"kind": "tf", "q": "“NTK-aware scaling slows every pair down by 4×.”",
             "answer": "False.", "why": "The fastest pair is unchanged (1.0 rad per token) and pair 1 slows by only 1.10×; "
                                       "only the slowest pairs stretch by the full factor.", "key": {'value': False}},
            {"kind": "code", "q": "<b>Try it yourself.</b> Use NTK scaling for a factor of 2 and read 128 characters. "
                                  "What is the loss on positions 64–127, compared with 2.17 with no change?",
             "code": """m = copy.deepcopy(base)
m.rope = Rope(base=10_000 * 2 ** (HD / (HD - 2)))
lp = loss_by_position(m, T=128)
print(lp[:64].mean(), lp[64:].mean())""",
             "answer": "1.61 and 1.73.", "why": "Raising the base keeps the first 64 positions intact and makes the second "
                                               "half usable without any training."},
        ],
    },
    {
        "title": "At scale, the costs, and the code",
        "segment": (123, 167),
        "figures": [{"t": 141.5, "caption": "Qwen2.5's documentation describes YaRN to stretch 32,768 tokens to 131,072."},
                    {"t": 166.0, "caption": "Each method is one number: scale multiplies the positions, base sets the "
                                            "rotation speeds."}],
        "body": [
            """<p>Real models use the same recipe at scale. Qwen2.5's documentation describes <b>YaRN</b>, a refined
version of NTK scaling, to stretch <b>32,768</b> tokens to <b>131,072</b>. Other model families use their own variants,
usually with some long-text training.</p>""",
            """<p>A longer window isn't free: the <b>KV cache</b> grows with every token, and models still tend to use the
<b>middle</b> of a long text less well than its start and end. Stretching the window is one thing; using it well is
another.</p>""",
            '<div class="figrow">{fig0}{fig1}</div>',
            """<p>In code, each method is one number: position interpolation passes <code>scale</code> to multiply the
positions; NTK scaling raises <code>base</code>. Everything else in the model stays the same.</p>""",
            """<div class="box key"><b class="t">Key idea</b>Because RoPE is a formula, not a learned table, a model's
context can be extended after training by changing its scale or base, plus a little long-text training.</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "By what factor does YaRN stretch Qwen2.5's context, according to its documentation?",
             "answer": "4.", "why": "131,072 / 32,768 = 4, the same factor as the episode's 64 → 256.", "key": {'parts': [{'label': None, 'value': 4, 'tol': 0.5, 'unit': None}]}},
            {"kind": "short", "q": "A model's window is stretched from 32,768 to 131,072 tokens. Name two costs or limits "
                                   "that remain.",
             "answer": "The KV cache (memory) grows with every token, and the model may still use the middle of a long "
                       "text poorly.",
             "why": "A longer window makes long inputs possible, not cheap or well used."},
            {"kind": "order", "q": "Order from worst to best average loss on positions 128–255: <i>NTK-aware scaling · "
                                   "interpolation + 200 steps · no change · interpolation only</i>.",
             "answer": "Interpolation only (3.59), no change (3.23), NTK (2.51), interpolation + 200 steps (1.58).",
             "why": "From the episode's run; note that interpolation without training is worst even here.", "key": {'items': ['interpolation only', 'no change', 'NTK-aware scaling', 'interpolation + 200 steps']}},
        ],
    },
]
