"""Study guide content for How Generative Models Work: Images and Sound, episode 10: Conditioning: Telling the Model What to Draw.

Build:  python framework/study_guide.py generative g10 --video generative/media/videos/g10_scene/1080p60/ConditioningVideo.mp4
Every number comes from code/g10_conditioning (train_cond.py, evaluate_cond.py, 2026-11-22): episode 6's diffusion
model with shape, color and position labels, eight combinations held out of training.
"""

LESSON = {
    "series": "How Generative Models Work: Images and Sound",
    "label": "Episode 10",
    "title": "Conditioning: Telling the Model What to Draw",
    "tagline": "Same model, one extra input.",
    "duration": "2:13",
    "intro": """<p>An unconditional diffusion model draws whatever it likes. To choose what comes out, give the
denoiser one extra input: a description of the picture. This lesson adds shape, color and position labels to our
Shapes diffusion model, tests it on combinations it never saw, and shows how <b>label dropout</b> lets the same
network draw from part of a description, or from none.</p>""",
    "prereq": """<div class="box note"><b>Before you start:</b> episodes 6–8 (the noise-guessing loss, the
fresh-noise sampler, the U-Net and its time embedding). Code: <code>code/g10_conditioning</code> (PyTorch; training
takes about 3.5 hours on a CPU, evaluation 15 minutes).</div>""",
}

CONCEPTS = [
    {
        "title": "One extra input",
        "segment": (7.5, 58.1),
        "figures": [{"t": 32.6, "caption": "Each label gets a learned vector; the vectors are added to the time "
                                           "embedding, which every block of the U-Net already reads."},
                    {"t": 44.6, "caption": "Training is unchanged: guess the noise, now with the labels as a hint."},
                    {"t": 57.9, "caption": "Label dropout: labels are hidden at random during training."}],
        "body": [
            """<p>Episode 8 fed the time step to every block of the U-Net by turning it into a vector (the time
embedding) and adding that vector inside each block. A description goes in the same way. Each of the three labels,
shape (3 values), color (5) and position (9), has its own small table of learned vectors (<code>nn.Embedding</code>),
and the three looked-up vectors are simply added to the time embedding. Every block now reads both the time and
the request. The network grows from 1,182,371 to 1,186,211 parameters.</p>""",
            "{fig0}",
            """<p>Training is episode 6's: noise a picture, ask the network to guess the noise, minimize the squared
error. The only difference is that the network also gets the picture's labels. They tell it what the clean picture
should look like, so the guess gets easier: the final loss is 0.0026, against 0.0028 for the unconditional model of
episode 8.</p>""",
            "{fig1}",
            """<p><b>Label dropout.</b> For 10% of training pictures, all three labels are replaced by a special
“not said” value. For the rest, each label is replaced on its own with probability 10%. So the network also
learns to denoise from part of a description, or from no description at all.</p>""",
            "{fig2}",
            """<div class="box key"><b class="t">Key idea</b>Conditioning is an extra input to the denoiser. Labels
become vectors that are added where the time already goes in; the training loss does not change.</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "During training, what share of pictures has its color label hidden? (10% lose "
                                    "all labels; the rest lose each label with probability 10%.)",
             "answer": "19%.", "why": "0.1 + 0.9 × 0.1 = 0.19.",
             "key": {"parts": [{"label": None, "value": 19, "tol": 0.5, "unit": "%"}]}},
            {"kind": "mc", "q": "How do the labels reach the U-Net's blocks?",
             "options": ["As extra color channels of the input picture",
                         "Their vectors are added to the time embedding, which every block reads",
                         "Through a separate network that edits the output", "Only through the loss"],
             "answer": "B.", "why": "The same path the time step uses (episode 8).", "key": {"choice": 1}},
        ],
    },
    {
        "title": "Following requests, even new ones",
        "segment": (58.6, 95.1),
        "figures": [{"t": 71.7, "caption": "Three requests, eight samples each."},
                    {"t": 85.5, "caption": "Four of the eight combinations that never appeared in training."},
                    {"t": 95.0, "caption": "Only “red” said: shape and position spread over all options."}],
        "body": [
            """<p>Sampling uses episode 7's fresh-noise sampler with 50 steps, the labels given at every step. Episode
3's judge reads each picture; a picture “follows” the request when it is recognizable and its shape, color and
position are all the requested ones.</p>
<table><tr><th>request</th><th>pictures</th><th>follows</th><th>shape</th><th>color</th><th>position</th></tr>
<tr><td>127 trained combinations</td><td>508</td><td>96.7%</td><td>97.2%</td><td>98.2%</td><td>97.6%</td></tr>
<tr><td>8 never-seen combinations</td><td>400</td><td>97.0%</td><td>98.0%</td><td>98.5%</td><td>98.0%</td></tr></table>""",
            "{fig0}",
            """<p>Eight combinations, for example “purple triangle, top left”, were removed from the training data.
The model still draws them as well as the trained ones (92% to 100% each). It has learned what “purple”,
“triangle” and “top left” mean separately, and combines them. This is called <b>compositional
generalization</b>.</p>""",
            "{fig1}",
            """<p>Thanks to label dropout, a request can leave things out. Asking only for “red” gives 86% red,
recognizable pictures, with the shape (51 circles, 50 squares, 71 triangles) and the position (15 to 25 per place)
spread over all options. Asking only for “triangle” or only for “top right” behaves the same way.</p>""",
            "{fig2}",
            """<div class="box key"><b class="t">Key idea</b>What is said is followed; what is not said stays
random. Labels learned separately combine into pictures never seen in training.</div>""",
        ],
        "exercises": [
            {"kind": "tf", "q": "“The model could only draw shape-color-position combinations that appeared in its "
                                "training data.”",
             "answer": "False.", "why": "The eight held-out combinations were drawn correctly 97.0% of the time.",
             "key": {"value": False}},
            {"kind": "short", "q": "Why does asking for only “red” give different shapes in different places?",
             "answer": "Shape and position are “not said”, so the model samples them, as an unconditional model "
                       "would.",
             "why": "Label dropout trained it to draw from partial descriptions."},
        ],
    },
    {
        "title": "One network, two models",
        "segment": (95.6, 125.2),
        "figures": [{"t": 106.5, "caption": "No labels at all: the same network as an unconditional model."},
                    {"t": 117.4, "caption": "Real models are conditioned on classes or on whole sentences."},
                    {"t": 124.7, "caption": "Conditioning in code."}],
        "body": [
            """<p>With every label set to “not said”, the network is an unconditional model: 90.0% of 500 pictures
are recognizable, covering 131 of the 135 combinations (episode 8's dedicated unconditional model: 95.2%). One set
of weights holds both a conditional and an unconditional model. Episode 11's classifier-free guidance compares the
two at every step.</p>""",
            "{fig0}",
            """<p>Real systems work the same way. Class-conditional ImageNet models (for example Dhariwal and Nichol,
2021) are told one of 1,000 classes through a learned embedding, exactly as here. Text-to-image models replace the
label table with a text encoder that turns a whole sentence into numbers (episodes 12 and 13).</p>""",
            "{fig1}",
            "{fig2}",
            """<div class="box key"><b class="t">Key idea</b>With label dropout, one network is both a conditional
and an unconditional model.</div>""",
        ],
        "exercises": [
            {"kind": "code", "q": "<b>Try it yourself.</b> The conditioning and the label dropout.",
             "code": """labels = [nn.Embedding(10, t_dim) for _ in range(3)]   # +1 entry: "not said"
temb = t_mlp(timestep_embedding(t))
temb = temb + sum(e(y[:, k]) for k, e in enumerate(labels))
y[torch.rand(len(y)) < 0.1] = NULL        # 10%: nothing said
y[torch.rand(y.shape) < 0.1] = NULL       # each label: 10%""",
             "answer": "train_cond.py's CondUNet.forward and drop_labels do exactly this.",
             "why": "NULL = 9 is the extra “not said” index of each table."},
        ],
    },
]
