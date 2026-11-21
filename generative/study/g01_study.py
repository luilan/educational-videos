"""Study guide content for How Generative Models Work: Images and Sound, episode 1: What It Means to Generate.

Build:  python framework/study_guide.py generative g01 --video generative/media/videos/g01_scene/1080p60/GenerateVideo.mp4
Every number comes from code/g01_what_it_means_to_generate (generate_basics.py, 2026-11-21): a 10,000-point spiral with
a 30×30 counting generator, and 20,000 Shapes images with a pixel-by-pixel generator.
"""

LESSON = {
    "series": "How Generative Models Work: Images and Sound",
    "label": "Episode 1",
    "title": "What It Means to Generate",
    "tagline": "Learn where the data lives, then sample from it",
    "duration": "2:30",
    "intro": """<p>A generative model learns the <b>distribution</b> of its training data and <b>samples</b> new
examples from it. This lesson introduces <b>Shapes</b>, the small image dataset used in this series. It then tries two
simple generators. Counting works for 2D points but cannot scale to images. Sampling each pixel on its own gets the colors
right but loses the shapes.</p>""",
    "prereq": """<div class="box note"><b>Before you start:</b> Foundations episodes 6 (probability and sampling) and
1 (vectors). Code: <code>code/g01_what_it_means_to_generate</code> (Python with NumPy and Pillow; runs in a few
seconds).</div>""",
}

CONCEPTS = [
    {
        "title": "Generating means sampling",
        "segment": (7.5, 49.3),
        "figures": [{"t": 21.8, "caption": "A language model picks one token at a time; a 32×32 color picture is 3,072 "
                                           "numbers that must fit together."},
                    {"t": 36.8, "caption": "Learn the distribution, then sample from it. A language model samples from "
                                           "its distribution over the next token."},
                    {"t": 48.8, "caption": "Shapes: one colored shape per 32×32 picture, with a caption. shapes.py makes "
                                           "20,000 in a few seconds."}],
        "body": [
            """<p>To <b>generate</b> means to draw a new example that looks as if it came from the same place as
the training data. Formally, a generative model learns a probability distribution <i>p(x)</i> over examples
<i>x</i>, then <b>samples</b> from it. You have seen this before: a language model learns <i>p(next token |
text so far)</i> and samples one token at a time.</p>""",
            '<div class="figrow">{fig0}{fig1}</div>',
            """<p>For pictures the problem is harder. A tiny 32×32 color picture is 32 × 32 × 3 = 3,072 numbers, and
the numbers are not independent: neighboring pixels must agree to form an edge, a shape, an object.</p>
<p>This series uses <b>Shapes</b>, a dataset made by our own code (<code>shapes.py</code>). Each picture has one
circle, square or triangle in one of 5 colors at one of 9 positions, with small random changes in position and size, and
a caption such as “a red circle at the top left”. That gives 3 × 5 × 9 = 135 different captions. It is small enough
to train on a laptop CPU and has no licensing issues.</p>""",
            "{fig2}",
            """<div class="box key"><b class="t">Key idea</b>A generative model is a learned distribution you can sample
from; the hard part for images is that thousands of numbers must be sampled together.</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "How many numbers describe a 64×64 color (RGB) picture?",
             "answer": "12,288.", "why": "64 × 64 × 3.",
             "key": {"parts": [{"label": None, "value": 12288, "tol": 0, "unit": None}]}},
            {"kind": "tf", "q": "“A language model is a generative model: it samples from a distribution over the next "
                                "token.”",
             "answer": "True.", "why": "It learns p(next token | context) and samples from it, one token at a time.",
             "key": {"value": True}},
            {"kind": "mc", "q": "Which of these is <b>not</b> generation?",
             "options": ["Drawing a new Shapes picture", "Writing the next word of a sentence",
                         "Labeling a picture as “circle”", "Producing a new spoken sentence"],
             "answer": "C.", "why": "Labeling picks a class for a given input; it does not sample a new example.",
             "key": {"choice": 2}},
        ],
    },
    {
        "title": "Counting, and why it fails",
        "segment": (49.8, 80.9),
        "figures": [{"t": 64.3, "caption": "A counting generator in 2D: 900 cells, 200 of them used; samples follow the "
                                           "spiral."},
                    {"t": 80.4, "caption": "Each extra number multiplies the cells by 30. Even with two levels per number, "
                                           "a 32×32 picture has about 10^924 cells."}],
        "body": [
            """<p>The simplest generator counts. Take 10,000 points on a 2D spiral, split the square into 30 × 30 =
900 cells and count the points in each. Only 200 cells contain any. To sample, pick a cell with probability
proportional to its count, then a random point inside it. The 2,000 samples trace the spiral.</p>""",
            "{fig0}",
            """<p>Each extra dimension multiplies the number of cells by 30: 30, 900, 27,000, and about 10<sup>15</sup>
for 10 numbers. A 32×32 picture has 3,072 numbers. Even if each number could only be dark or bright, there would be
2<sup>3072</sup> ≈ 10<sup>924</sup> cells. Our 20,000 pictures would leave almost every cell empty, and a counting
model can only give back pictures it has already seen. This is the <b>curse of dimensionality</b>.</p>""",
            "{fig1}",
            """<div class="box key"><b class="t">Key idea</b>Counting works in low dimensions but the number of cells
grows exponentially; real generators must generalize from far fewer examples than cells.</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "With 30 bins per number, how many cells does a 4-number space have?",
             "answer": "810,000.", "why": "30<sup>4</sup> = 810,000.",
             "key": {"parts": [{"label": None, "value": 810000, "tol": 0, "unit": None}]}},
            {"kind": "number", "q": "The spiral's 10,000 points fell into 200 of the 900 cells. What is the average "
                                    "number of points per <b>used</b> cell?",
             "answer": "50.", "why": "10,000 / 200.",
             "key": {"parts": [{"label": None, "value": 50, "tol": 0, "unit": None}]}},
            {"kind": "short", "q": "Why can't the counting generator produce a new Shapes picture it has never seen?",
             "answer": "Every cell it can sample from was filled by a training picture; with ~10^924 cells and 20,000 "
                       "pictures, all other cells have count zero and probability zero.",
             "why": "Counting memorizes; it does not generalize."},
        ],
    },
    {
        "title": "Pixels must fit together",
        "segment": (81.4, 142.2),
        "figures": [{"t": 102.8, "caption": "Sampling every pixel on its own: about the right amount of color (68 vs 65 "
                                            "shape pixels) but the biggest connected piece is 4.7 pixels, against 68."},
                    {"t": 119.6, "caption": "Two other failures: the average picture (a smudge) and copying a training "
                                            "picture (nothing new)."},
                    {"t": 141.6, "caption": "The counting generator in three lines."}],
        "body": [
            """<p>A cheaper model learns each pixel on its own: at every position, how often each of the 6 colors
(background plus 5) appears. The busiest pixel is part of a shape only 11% of the time. Sampling every pixel
independently gives pictures with the right colors and about the right amount of shape: 68 shape pixels on average,
against 65 in real pictures. But the biggest connected piece averages 4.7 pixels, against 68 in a real picture, and
every sample uses all 6 colors where a real picture has 2. The model knows <i>where</i> color tends to be, but not that
the pixels must agree with each other.</p>""",
            "{fig0}",
            """<p>Two other extremes fail too. The <b>average</b> of all 20,000 pictures is a dark smudge (with contrast
raised 4× you can see the 9 positions). <b>Copying</b> a random training picture gives a perfect sample but nothing new.
A useful generator sits in between: it learns how pixels depend on each other and makes new combinations. This series
covers the main ways to do that: codes (autoencoders, VAEs, GANs), diffusion, text conditioning, and then sound.</p>""",
            '<div class="figrow">{fig1}{fig2}</div>',
            """<div class="box key"><b class="t">Key idea</b>Matching each pixel's statistics is not enough; a
generator must capture how pixels depend on each other.</div>""",
        ],
        "exercises": [
            {"kind": "short", "q": "The pixel-by-pixel samples have about as many shape pixels as real pictures. Why "
                                   "don't they look like shapes?",
             "answer": "Each pixel is sampled independently, so the colored pixels are scattered (biggest piece 4.7 "
                       "pixels) and mix all colors; real shapes are one connected piece of one color.",
             "why": "The model ignores how pixels depend on each other."},
            {"kind": "number", "q": "In one real Shapes picture, how many different colors appear?",
             "answer": "2.", "why": "The background and the shape's color (shapes have hard edges).",
             "key": {"parts": [{"label": None, "value": 2, "tol": 0, "unit": None}]}},
            {"kind": "code", "q": "<b>Try it yourself.</b> Sample 2,000 points from a 2D counting generator.",
             "code": """counts = np.histogram2d(x, y, bins=[edges, edges])[0]
p = counts.ravel() / counts.sum()
cells = rng.choice(30 * 30, size=2000, p=p)
w = edges[1] - edges[0]
xs = edges[0] + (cells // 30 + rng.uniform(0, 1, 2000)) * w
ys = edges[0] + (cells % 30 + rng.uniform(0, 1, 2000)) * w""",
             "answer": "generate_basics.py does this for the spiral; the samples follow it.",
             "why": "Pick a cell in proportion to its count, then a uniform point inside it."},
        ],
    },
]
