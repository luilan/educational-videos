"""Study guide content for How LLMs Work: Deep Dive, episode 30: Multimodal: Images as Tokens.

Build:  python framework/study_guide.py deep-dive d30 --video deep-dive/media/videos/d30_scene/1080p60/MultimodalVideo.mp4
Figures: {"t": seconds into the rendered video, "caption": ...}; placed in body blocks with {fig0}, {fig1}, ...
Exercise kinds: mc (options), tf, short, number, order, code. Every exercise has an answer and a why.
Every number comes from code/d30_multimodal/multimodal.py (a tiny vision-language model on synthetic shape images; torch
2.14.0, CPU; seeded, so the run reproduces) or simple patch arithmetic.
"""

LESSON = {
    "series": "How LLMs Work: Deep Dive",
    "label": "Episode 30",
    "title": "Multimodal: Images as Tokens",
    "tagline": "Teaching a language model to read pictures",
    "duration": "2:18",
    "intro": """<p>This lesson answers one question: how does a language model read an image? By turning it into
tokens. A tiny model built from scratch cuts a 32 × 32 image into 16 patches, projects each with one linear layer into
the same space as the text tokens, and reads image and caption as one sequence. After 1,500 steps it describes new
images with the right color and position every time, and the right shape 91% of the time; real vision-language models
follow the same plan at scale.</p>""",
    "prereq": """<div class="box note"><b>Before you start:</b> How LLMs Work episodes 3–4 (tokens and embeddings) and
Deep Dive episode 7 (the causal mask). Code: <code>code/d30_multimodal</code> (PyTorch only; no downloads; about 10
minutes on a CPU).</div>""",
}

CONCEPTS = [
    {
        "title": "From pixels to tokens",
        "segment": (8, 55),
        "figures": [{"t": 40.4, "caption": "16 patches of 8 × 8 × 3 = 192 numbers; one linear layer maps each to a 128-number "
                                           "token."},
                    {"t": 54.3, "caption": "One sequence: 16 image tokens, then the caption one character at a time."}],
        "body": [
            """<p>A transformer only reads sequences of vectors, so the image is turned into tokens. A 32 × 32 color image
is cut into <b>16 patches</b> of 8 × 8 pixels; each patch is 8 × 8 × 3 = <b>192 numbers</b>. One linear layer turns each
patch into a vector of <b>128</b> numbers, exactly the size of a text token. Here that layer, 24,704 parameters, is the
whole vision encoder.</p>""",
            '<div class="figrow">{fig0}{fig1}</div>',
            """<p>Image and text then become <b>one sequence</b>: 16 image tokens followed by the characters of a caption.
One causal transformer (830,103 parameters) reads it all and learns to write the caption, e.g. “a green circle, top
left.”</p>""",
            """<div class="box key"><b class="t">Key idea</b>To a transformer, an image is just more tokens: patches
projected into the same vector space as words.</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "How many parameters does a linear layer from 192 to 128 numbers have (with biases)?",
             "answer": "24,704.", "why": "192 × 128 weights + 128 biases."},
            {"kind": "number", "q": "How many 8 × 8 patch tokens does a 64 × 64 image make?",
             "answer": "64.", "why": "(64 / 8)² = 8 × 8."},
        ],
    },
    {
        "title": "Training and testing",
        "segment": (55, 89),
        "figures": [{"t": 68.8, "caption": "Training images: 4 shapes × 3 colors × 4 positions; caption loss 3.164 → 0.040 by "
                                           "step 250."},
                    {"t": 87.6, "caption": "Most mistakes: squares called circles (10) and circles called squares (4)."}],
        "body": [
            """<p>The model trains on random pictures of circles, squares, triangles and crosses, in three colors and four
positions, with random size and offset. The caption loss falls from <b>3.164</b> to <b>0.040</b> in 250 steps (0.009 at
step 1,500).</p>""",
            """<p>On 200 new images: color right <b>100%</b> of the time, position <b>100%</b>, shape <b>91%</b>. Most shape
mistakes are squares called circles (10) and circles called squares (4), with two circles called triangles and two
other swaps. These shapes are only 8 to 12 pixels wide, and at that size a corner is a subtle thing.</p>""",
            '<div class="figrow">{fig0}{fig1}</div>',
            """<div class="box key"><b class="t">Key idea</b>Coarse properties (color, position) are easy from patches;
fine shape details need enough resolution.</div>""",
        ],
        "exercises": [
            {"kind": "mc", "q": "Which property did the model get wrong most often?",
             "options": ["Color", "Position", "Shape", "None"],
             "answer": "C.", "why": "Shape 91% vs 100% for color and position."},
            {"kind": "short", "q": "Why might color be easier than shape for this model?",
             "answer": "Color is visible in any single pixel of the shape, so one patch token carries it; telling a square "
                       "from a circle depends on a few corner pixels.",
             "why": "The mistakes were almost all between the two most similar shapes."},
        ],
    },
    {
        "title": "Real models, and the cost",
        "segment": (89, 128),
        "figures": [{"t": 107.6, "caption": "Real models: ~14–16-pixel patches, a vision transformer, projection into the "
                                            "language model; 224 × 224 → 196 tokens."},
                    {"t": 126.8, "caption": "Three steps in code: cut into patches, project, concatenate with the text."}],
        "body": [
            """<p>Real vision-language models follow the same plan at scale: the image is cut into patches of around 14 or
16 pixels, passed through a vision transformer, and projected into the language model's space. A 224 × 224 image in
16 × 16 patches is 14 × 14 = <b>196 tokens</b>. The catch: images cost tokens. A detailed, high-resolution image can
take hundreds or thousands, competing with text for the context window.</p>""",
            '<div class="figrow">{fig0}{fig1}</div>',
            """<div class="box key"><b class="t">Key idea</b>The number of image tokens grows with the square of the
resolution divided by the patch size.</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "How many tokens does a 336 × 336 image make with 14 × 14 patches? And 448 × 448?",
             "answer": "576 and 1,024.", "why": "(336 / 14)² = 24² and (448 / 14)² = 32²."},
            {"kind": "code", "q": "<b>Try it yourself.</b> Check the patch shapes with the episode's code.",
             "code": """img = draw("cross", "red", "bottom left")
print(img.shape, patches(img).shape)""",
             "answer": "torch.Size([3, 32, 32]) and torch.Size([16, 192]).",
             "why": "3 color channels of 32 × 32 pixels become 16 tokens of 192 numbers."},
        ],
    },
]
