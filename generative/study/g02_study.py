"""Study guide content for How Generative Models Work: Images and Sound, episode 2: Autoencoders.

Build:  python framework/study_guide.py generative g02 --video generative/media/videos/g02_scene/1080p60/AutoencoderVideo.mp4
Every number comes from code/g02_autoencoders (train_autoencoder.py, 2026-11-21): one convolutional autoencoder
trained for 15 epochs on 20,000 Shapes pictures with codes of 2, 4, 8, 16 and 32 numbers, tested on 2,000 new ones.
"""

LESSON = {
    "series": "How Generative Models Work: Images and Sound",
    "label": "Episode 2",
    "title": "Autoencoders",
    "tagline": "Squeeze a picture into a few numbers, then rebuild it",
    "duration": "2:26",
    "intro": """<p>An <b>autoencoder</b> learns to squeeze a picture into a short <b>code</b> and rebuild it. Nothing
tells it what the code should mean. To get through the bottleneck, it has to learn how pixels depend on each other.
This lesson trains one on Shapes with codes of 2 to 32 numbers. Reconstruction is excellent, but generation fails:
random codes decode to mush, because the codes form islands with empty space between them.</p>""",
    "prereq": """<div class="box note"><b>Before you start:</b> episode 1 and Foundations episodes 12–13 (neural
networks, PyTorch). Code: <code>code/g02_autoencoders</code> (PyTorch on a CPU, about 5 minutes for all five
models).</div>""",
}

CONCEPTS = [
    {
        "title": "Encoder, code, decoder",
        "segment": (7.5, 53.4),
        "figures": [{"t": 35.8, "caption": "Encoder → an 8-number code → decoder. The loss only asks the output to "
                                           "match the input; no labels."},
                    {"t": 52.9, "caption": "Three convolutions halve the picture (32 → 16 → 8 → 4), a linear layer makes "
                                           "the code; 366,539 parameters, 63 s of training."}],
        "body": [
            """<p>An autoencoder is two networks trained together. The <b>encoder</b> maps a picture (3,072 numbers)
to a short <b>code</b>; the <b>decoder</b> maps the code back to a picture. The loss is the mean squared difference
between the output and the input, so the training data needs no labels: the picture is its own target. This is
<b>self-supervised</b> learning.</p>""",
            "{fig0}",
            """<p>Our encoder has three convolutions with stride 2, each halving the width and height (3×32×32 →
32×16×16 → 64×8×8 → 128×4×4), then a linear layer down to the code. The decoder mirrors it with transposed
convolutions and ends with a sigmoid, so pixel values stay between 0 and 1. With an 8-number code it has 366,539
parameters and trains in about a minute (15 passes over 20,000 pictures) on a CPU. Our first version without batch
normalization got stuck predicting an empty background for most code sizes. Adding it fixed training.</p>""",
            "{fig1}",
            """<div class="box key"><b class="t">Key idea</b>An autoencoder learns a compressed code for its data
without labels, because the only way to rebuild the input is to keep what matters.</div>""",
        ],
        "exercises": [
            {"kind": "tf", "q": "“Training an autoencoder needs a label for every picture.”",
             "answer": "False.", "why": "The target is the input picture itself.", "key": {"value": False}},
            {"kind": "number", "q": "A stride-2 convolution halves width and height. Starting from 32×32, how many "
                                    "such layers give 4×4?",
             "answer": "3.", "why": "32 → 16 → 8 → 4.",
             "key": {"parts": [{"label": None, "value": 3, "tol": 0, "unit": None}]}},
            {"kind": "order", "q": "Put the data flow in order.",
             "answer": "picture → encoder → code → decoder → rebuilt picture.",
             "why": "The loss compares the last item with the first.",
             "key": {"items": ["picture", "encoder", "code", "decoder", "rebuilt picture"]}},
        ],
    },
    {
        "title": "How small can the code be?",
        "segment": (53.9, 85.8),
        "figures": [{"t": 70.3, "caption": "Rebuilt test pictures: 2 numbers 3.9% of pixels clearly wrong, 8 numbers "
                                           "0.9%, 32 numbers 0.5%."},
                    {"t": 85.3, "caption": "The pictures vary in only a few ways; the network finds them on its own."}],
        "body": [
            """<p>We trained the same network with codes of 2, 4, 8, 16 and 32 numbers and tested on 2,000 new
pictures. Counting a pixel as <b>clearly wrong</b> when any channel is off by more than 0.25:</p>
<table><tr><th>code size</th><th>test MSE</th><th>pixels clearly wrong</th></tr>
<tr><td>2</td><td>0.0044</td><td>3.9%</td></tr><tr><td>4</td><td>0.0022</td><td>2.0%</td></tr>
<tr><td>8</td><td>0.0011</td><td>0.9%</td></tr><tr><td>16</td><td>0.0010</td><td>0.7%</td></tr>
<tr><td>32</td><td>0.0007</td><td>0.5%</td></tr></table>
<p>With 2 numbers the pictures are blurry blobs, often with the wrong shape or a slightly wrong color. With 32 they are
almost perfect, from a code 96 times smaller than the picture.</p>""",
            "{fig0}",
            """<p>This works because Shapes pictures vary in only a few ways: shape (3), color (5), position (9) and
size. A handful of numbers can hold all of that. The network was never told these factors. It found a way to store
them because that was the only way through the bottleneck. Real photos vary in far more ways, so they need bigger
codes.</p>""",
            "{fig1}",
            """<div class="box key"><b class="t">Key idea</b>Data that varies in only a few ways can be compressed into a
few numbers; the code size trades detail for compression.</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "How many times smaller than the picture is an 8-number code?",
             "answer": "384.", "why": "3,072 / 8.",
             "key": {"parts": [{"label": None, "value": 384, "tol": 0, "unit": None}]}},
            {"kind": "mc", "q": "Going from a 16-number to a 32-number code, the share of clearly wrong pixels went "
                                "from 0.7% to 0.5%. What does this suggest?",
             "options": ["The 32-number code is broken", "Most of what matters already fits in 16 numbers",
                         "Bigger codes always make reconstructions worse", "The test set is too small"],
             "answer": "B.", "why": "Doubling the code gained little: the remaining detail is small.",
             "key": {"choice": 1}},
        ],
    },
    {
        "title": "Why an autoencoder can't generate",
        "segment": (86.3, 135.1),
        "figures": [{"t": 101.5, "caption": "Random codes: plausible blobs with 2 numbers, mush with 8 or 32; with 8 "
                                            "numbers they land 7.6× further from any training picture than a "
                                            "reconstruction."},
                    {"t": 112.2, "caption": "Halfway between a yellow square and a purple circle: green and blue smudges."},
                    {"t": 127.9, "caption": "2-number codes of 2,000 test pictures, colored by position: islands with "
                                            "gaps."}],
        "body": [
            """<p>To generate, we need codes to decode. Draw random codes with the same mean and spread as the real
ones, number by number. With 2 numbers the results are plausible blobs. With 8 or 32 they are mush. We can measure this:
the squared error from a decoded random code to the <b>nearest</b> of the 20,000 training pictures is 0.0086 (8
numbers), against 0.0011 for a rebuilt test picture, 7.6 times further. With 32 numbers it is 12 times further.</p>""",
            "{fig0}",
            """<p>Averaging two real codes does not help. Halfway between a yellow square and a purple circle, the
decoder draws green and blue smudges in both places, colors neither picture has.</p>""",
            "{fig1}",
            """<p>The 2-number codes show why. Plotted on a plane, the 2,000 test pictures form islands, roughly one
per position, with gaps between them. Training only ever asked the decoder about codes on the islands. Anywhere else,
its output is unconstrained. The more numbers in the code, the more empty space there is, so random codes almost never
land on an island.</p>""",
            "{fig2}",
            """<div class="box key"><b class="t">Key idea</b>An autoencoder's code space has gaps: the decoder only
learned the codes that real pictures produce, so random codes are not pictures.</div>""",
        ],
        "exercises": [
            {"kind": "short", "q": "Why do random codes decode to mush with an 8-number code but to plausible blobs "
                                   "with a 2-number code?",
             "answer": "In 2 dimensions the real codes cover much of the space, so a random code is often near an "
                       "island; in 8 or 32 dimensions almost all of the space is empty gaps the decoder never saw.",
             "why": "Empty space grows quickly with dimension, as in episode 1."},
            {"kind": "number", "q": "With a 32-number code, random codes were 0.0085 from the nearest training "
                                    "picture and rebuilt pictures 0.0007. About how many times further, to the nearest "
                                    "whole number?",
             "answer": "12.", "why": "0.0085 / 0.0007 ≈ 12.",
             "key": {"parts": [{"label": None, "value": 12, "tol": 0.5, "unit": None}]}},
            {"kind": "code", "q": "<b>Try it yourself.</b> One training step of an autoencoder.",
             "code": """code = model.encoder(x)               # (batch, 8)
x_hat = model.decoder(code)           # (batch, 3, 32, 32)
loss = ((x_hat - x) ** 2).mean()
opt.zero_grad(); loss.backward(); opt.step()""",
             "answer": "train_autoencoder.py runs this for 15 epochs per code size.",
             "why": "The picture is both the input and the target."},
        ],
    },
]
