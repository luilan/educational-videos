"""Study guide content for How Generative Models Work: Images and Sound, episode 12: CLIP: Matching Pictures and Words.

Build:  python framework/study_guide.py generative g12 --video generative/media/videos/g12_scene/1080p60/ClipVideo.mp4
Every number comes from code/g12_clip (train_clip.py, 2026-11-22): a tiny CLIP trained on Shapes pictures and
captions, with episode 10's eight combinations held out of training.
"""

LESSON = {
    "series": "How Generative Models Work: Images and Sound",
    "label": "Episode 12",
    "title": "CLIP: Matching Pictures and Words",
    "tagline": "Two encoders that learn to agree.",
    "duration": "2:27",
    "intro": """<p>To draw from a sentence, a model needs numbers that capture what the words mean <i>visually</i>.
<b>CLIP</b> learns them by contrastive training: an image encoder and a text encoder are trained together until
matching pictures and captions land close to each other. This lesson trains a tiny CLIP on Shapes captions. It
looks at what is learned first, tests it on combinations it never saw, and explains why its loss can't reach
zero.</p>""",
    "prereq": """<div class="box note"><b>Before you start:</b> episode 10; vectors and dot products; cross-entropy
and attention (How LLMs Work). Code: <code>code/g12_clip</code> (PyTorch; about 5 minutes on a CPU).</div>""",
}

CONCEPTS = [
    {
        "title": "Two encoders and a table",
        "segment": (7.5, 60.8),
        "figures": [{"t": 31.9, "caption": "A picture and a caption each become 64 numbers; when they match, the two "
                                           "vectors should point the same way."},
                    {"t": 49.4, "caption": "Similarities of 8 test pictures (columns) and their 8 captions (rows). "
                                           "The true pairs are on the diagonal."},
                    {"t": 60.5, "caption": "Our tiny CLIP: a small CNN, a 2-layer transformer, 17 words."}],
        "body": [
            """<p>CLIP (Radford et al. 2021) has two encoders. The image encoder maps a picture to a vector, the text
encoder maps a caption to a vector of the same length (64 here). Both are scaled to length 1, so their dot product
is the <b>cosine similarity</b>: 1 when they point the same way, 0 when unrelated.</p>""",
            "{fig0}",
            """<p>Training takes a batch of N picture-caption pairs and computes all N × N similarities. The N true
pairs sit on the diagonal. Multiply the table by a learned scale (1/temperature) and apply cross-entropy twice:
along each row (this picture: which of the N captions?) and along each column (this caption: which picture?). This
is <b>contrastive learning</b>. Matching pairs are pulled together and every mismatched pair in the batch is pushed
apart, with no labels beyond the pairing itself.</p>""",
            "{fig1}",
            """<p>Ours: a small CNN (588,992 parameters) and a 2-layer, 4-head transformer over the caption's words,
averaged into one vector (72,768 parameters). The vocabulary has 17 words. Batch 128, Adam 0.001, 4,000 steps,
about 5 minutes on a CPU.</p>""",
            "{fig2}",
            """<div class="box key"><b class="t">Key idea</b>Contrastive training needs only matching pairs: in each
batch, every picture must pick out its caption and every caption its picture.</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "In a batch of 128 pairs, how many entries of the similarity table should be "
                                    "pushed down?",
             "answer": "16,256.", "why": "128 × 128 = 16,384 entries, minus the 128 on the diagonal.",
             "key": {"parts": [{"label": None, "value": 16256, "tol": 0, "unit": None}]}},
            {"kind": "mc", "q": "Why are both vectors normalized to length 1?",
             "options": ["To make training faster", "So the dot product measures direction only (cosine similarity)",
                         "Because the captions are short", "To save memory"],
             "answer": "B.", "why": "Without it, a vector could score high just by being long.", "key": {"choice": 1}},
        ],
    },
    {
        "title": "What it learns, and in what order",
        "segment": (61.2, 91.0),
        "figures": [{"t": 77.1, "caption": "Reading pictures by choosing the closest of all 135 captions, during "
                                           "training: position first, then color, then shape."},
                    {"t": 90.7, "caption": "Eight combinations never seen in training are all read correctly."}],
        "body": [
            """<p>To “read” a picture, compare it with all 135 possible captions and take the most similar one. The
model was never trained on this task directly (zero-shot). On 1,877 test pictures:</p>
<table><tr><th>step</th><th>0</th><th>10</th><th>20</th><th>30</th><th>50</th><th>100</th><th>150</th><th>250</th></tr>
<tr><td>whole caption</td><td>0.2%</td><td>11.6%</td><td>37.3%</td><td>41.6%</td><td>58.4%</td><td>96.4%</td><td>99.9%</td><td>100%</td></tr>
<tr><td>position</td><td>7%</td><td>100%</td><td>100%</td><td>100%</td><td>100%</td><td>100%</td><td>100%</td><td>100%</td></tr>
<tr><td>color</td><td>19%</td><td>31%</td><td>93%</td><td>100%</td><td>100%</td><td>100%</td><td>100%</td><td>100%</td></tr>
<tr><td>shape</td><td>30%</td><td>35%</td><td>41%</td><td>42%</td><td>58%</td><td>96%</td><td>100%</td><td>100%</td></tr></table>
<p>Position is learned within 10 steps, color by 30, shape last, around step 150: the same coarse-to-fine order
in which episode 7's samples settled.</p>""",
            "{fig0}",
            """<p>Eight combinations (episode 10's) were left out of training completely, both as pictures and as
captions, for example “a purple triangle at the top left”. From step 150 on, the model reads all 200 test
pictures of them correctly. It learned “purple”, “triangle” and “top left” separately, and the text encoder
combines them into a vector that lands near the right pictures.</p>""",
            "{fig1}",
            """<div class="box key"><b class="t">Key idea</b>Because words are learned separately, a contrastive model
can match combinations it has never seen.</div>""",
        ],
        "exercises": [
            {"kind": "tf", "q": "“Our CLIP could only read combinations of shape, color and position that it saw in "
                                "training.”",
             "answer": "False.", "why": "All 200 pictures of the 8 held-out combinations were read correctly.",
             "key": {"value": False}},
        ],
    },
    {
        "title": "The loss floor, search, and real CLIP",
        "segment": (91.5, 139.8),
        "figures": [{"t": 107.9, "caption": "The loss levels off at 0.57, exactly the best possible value when "
                                            "batches contain repeated captions."},
                    {"t": 118.1, "caption": "Caption → picture search, including a combination never trained on."},
                    {"t": 139.5, "caption": "Contrastive loss in five lines."}],
        "body": [
            """<p>The loss levels off at 0.574 and never gets lower. A batch of 128 drawn from 127 captions often holds
the same caption twice or more. Identical captions give identical text vectors, so for a caption that appears k
times the best possible probability for the right partner is 1/k, and the loss is at least log k. For this batch
size, k − 1 follows a binomial distribution with 127 trials and probability 1/127. The expected log k is 0.574,
exactly where our loss stops. Real CLIP has the same effect in milder form: near-duplicate captions in a
web-scraped batch.</p>""",
            "{fig0}",
            """<p>The same table works in both directions. Encode a caption, rank pictures by similarity, and you have
a search engine. It works even for “a purple triangle at the top left”, which was never in training.</p>""",
            "{fig1}",
            """<p>OpenAI's CLIP was trained on 400 million picture-caption pairs from the web. Its text encoder is the
one Stable Diffusion 1.x uses to turn a prompt into numbers. A text encoder trained against pictures already
knows what words look like, which is exactly what a text-to-image model needs (episode 13).</p>""",
            "{fig2}",
            """<div class="box key"><b class="t">Key idea</b>CLIP turns words and pictures into one shared space,
useful both for search and as the text input of an image generator.</div>""",
        ],
        "exercises": [
            {"kind": "short", "q": "Why can't our CLIP's loss reach zero, even though it reads every test picture "
                                   "correctly?",
             "answer": "Batches often contain the same caption twice; identical captions can't be told apart, so the "
                       "right pair can get at most probability 1/k.",
             "why": "The expected log k for batch 128 and 127 captions is 0.574, where the loss stops."},
            {"kind": "code", "q": "<b>Try it yourself.</b> The contrastive loss.",
             "code": """img = F.normalize(image_encoder(pictures), dim=-1)
txt = F.normalize(text_encoder(captions), dim=-1)
logits = scale * img @ txt.T
target = torch.arange(len(img))
loss = (F.cross_entropy(logits, target) + F.cross_entropy(logits.T, target)) / 2""",
             "answer": "train_clip.py's CLIP.forward and training loop compute exactly this.",
             "why": "Rows: picture → caption; columns (logits.T): caption → picture."},
        ],
    },
]
