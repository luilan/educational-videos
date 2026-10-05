"""Study guide content for How LLMs Work: Deep Dive, episode 14: LayerNorm vs RMSNorm.

Build:  python framework/study_guide.py deep-dive d14 --video deep-dive/media/videos/d14_scene/1080p60/NormsVideo.mp4
Figures: {"t": seconds into the rendered video, "caption": ...}; placed in body blocks with {fig0}, {fig1}, ...
Exercise kinds: mc (options), tf, short, number, order, code. Every exercise has an answer and a why.
Every number comes from code/d14_norms/norms.py (GPT-2 small and Qwen2.5-0.5B-Instruct, real weights; an 8-layer tiny
GPT trained 1,500 steps; transformers 4.57.1, torch 2.14.0, CPU) or the same formulas on the exercise vectors.
"""

LESSON = {
    "series": "How LLMs Work: Deep Dive",
    "label": "Episode 14",
    "title": "LayerNorm vs RMSNorm",
    "tagline": "Reading the stream at the right scale",
    "duration": "2:38",
    "intro": """<p>This lesson answers one question: how does each block read a residual stream that keeps growing?
Through a <b>normalization</b>. GPT-2 uses <b>LayerNorm</b> (subtract the mean, divide by the spread, scale and
shift); Llama, Mistral and Qwen use <b>RMSNorm</b> (divide by the root mean square, scale). Both are computed by hand
and match the real models. In a tiny GPT they give the same loss; without any normalization, training blows up at a
higher learning rate.</p>""",
    "prereq": """<div class="box note"><b>Before you start:</b> Deep Dive episode 13 (the residual stream) and Foundations
(mean, standard deviation). Code: <code>code/d14_norms</code> (GPT-2 small and Qwen2.5-0.5B, about 1.5 GB, plus six
tiny models, about 25 minutes on a CPU).</div>""",
}

CONCEPTS = [
    {
        "title": "Why normalize",
        "segment": (8, 40),
        "figures": [{"t": 19.0, "caption": "Each block reads the stream through a norm: LayerNorm in GPT-2, RMSNorm in "
                                           "Llama, Mistral and Qwen."},
                    {"t": 38.7, "caption": "The stream grows from 5.2 to 253.3; what each block reads is always √768 = "
                                           "27.7."}],
        "body": [
            """<p>Every block reads the residual stream through a normalization. In GPT-2 the stream grows from an average
size of <b>5.2</b> (layer 0) to <b>253.3</b> (layer 11). After normalization, every block reads a vector of exactly the
same size, <b>27.7 = √768</b>, at every layer: the block's weights see inputs on a fixed scale, however big the stream
has become.</p>""",
            '<div class="figrow">{fig0}{fig1}</div>',
            """<div class="box key"><b class="t">Key idea</b>Normalizing to unit spread makes a d-number vector's length
√d, at every layer.</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "Qwen2.5-0.5B's stream has 896 numbers. What is the length of a normalized vector "
                                    "(before the learned scale)?",
             "answer": "About 29.93.", "why": "√896 ≈ 29.93, as √768 = 27.7 for GPT-2."},
            {"kind": "tf", "q": "“Normalization changes the residual stream itself.”",
             "answer": "False.", "why": "Only the block's input is normalized; the stream keeps growing (5.2 → 253.3) and "
                                       "the block's output is added to it unnormalized."},
        ],
    },
    {
        "title": "LayerNorm and RMSNorm, by hand",
        "segment": (40, 84),
        "figures": [{"t": 64.7, "caption": "LayerNorm on “ sat” entering GPT-2 layer 6: mean 0.088, std 3.465; normalize, "
                                           "then learned scale and shift."},
                    {"t": 82.7, "caption": "RMSNorm: divide by the root mean square and scale; in Qwen2.5 the mean is "
                                           "0.022 anyway."}],
        "body": [
            """<p><b>LayerNorm.</b> The token “ sat” entering GPT-2's layer 6 is 768 numbers with mean <b>0.088</b> and
standard deviation <b>3.465</b>. Subtract the mean, divide by the standard deviation, then multiply by a learned scale
(γ) and add a learned shift (β), 768 numbers each. The hand computation matches GPT-2's LayerNorm to within 4.8 ×
10<sup>−7</sup>.</p>""",
            """<p><b>RMSNorm</b> skips the mean: divide by √(mean(x²) + ε), multiply by a learned scale. No centering, no
shift. In Qwen2.5-0.5B (layer 6, 896 numbers) the stream's mean is 0.022, almost zero anyway; its root mean square is
0.445. Again the hand version matches the model's to 4.8 × 10<sup>−7</sup>.</p>""",
            '<div class="figrow">{fig0}{fig1}</div>',
            """<div class="box key"><b class="t">Key idea</b>LayerNorm = center + rescale + scale + shift; RMSNorm =
rescale + scale.</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "LayerNorm (no learned scale or shift) of [1, 2, 3, 6]?",
             "answer": "About [−1.069, −0.535, 0, 1.604].", "why": "Mean 3, variance (4 + 1 + 0 + 9) / 4 = 3.5, standard "
                                                                  "deviation 1.871."},
            {"kind": "number", "q": "RMSNorm (no learned scale) of [1, 2, 3, 6]?",
             "answer": "About [0.283, 0.566, 0.849, 1.697].", "why": "RMS = √((1 + 4 + 9 + 36) / 4) = √12.5 ≈ 3.536."},
            {"kind": "short", "q": "Add 10 to every number: [11, 12, 13, 16]. Which normalization gives the same output as "
                                   "before, and why?",
             "answer": "LayerNorm: it subtracts the mean, so a constant shift disappears ([−1.069, −0.535, 0, 1.604] "
                       "again). RMSNorm changes: [0.838, 0.914, 0.990, 1.218].",
             "why": "Centering is exactly what RMSNorm leaves out."},
            {"kind": "number", "q": "How many learned numbers do GPT-2 small's 25 LayerNorms have in total?",
             "answer": "38,400.", "why": "2 per layer × 12 + the final one = 25, each with 768 + 768."},
        ],
    },
    {
        "title": "Does it matter?",
        "segment": (84, 115),
        "figures": [{"t": 99.6, "caption": "Learning rate 0.001: LayerNorm 1.69, RMSNorm 1.69, no norm 1.71."},
                    {"t": 114.3, "caption": "Learning rate 0.01: LayerNorm and RMSNorm 1.81; without a norm, the loss "
                                            "becomes NaN."}],
        "body": [
            """<p>The same 8-layer tiny GPT, trained 1,500 steps three ways. At learning rate 0.001: LayerNorm
<b>1.69</b>, RMSNorm <b>1.69</b>, no normalization <b>1.71</b>. Barely any difference.</p>""",
            """<p>Raise the learning rate ten times (0.01): LayerNorm and RMSNorm both reach <b>1.81</b>. Without
normalization, the loss becomes NaN (“not a number”): the training blows up. Normalization is what keeps it stable.</p>""",
            '<div class="figrow">{fig0}{fig1}</div>',
            """<div class="box key"><b class="t">Key idea</b>The choice of norm barely matters; having one matters a lot,
especially for stability.</div>""",
        ],
        "exercises": [
            {"kind": "mc", "q": "What happened without normalization at learning rate 0.01?",
             "options": ["Loss 1.71", "Same as LayerNorm", "The loss became NaN", "Training was faster"],
             "answer": "C.", "why": "The activations grew without bound and the loss became “not a number”."},
            {"kind": "tf", "q": "“RMSNorm gave a worse loss than LayerNorm in the episode's experiment.”",
             "answer": "False.", "why": "1.69 vs 1.69 at 0.001, and 1.81 vs 1.81 at 0.01."},
        ],
    },
    {
        "title": "Why modern models use RMSNorm",
        "segment": (115, 148),
        "figures": [{"t": 136.8, "caption": "LayerNorm: two statistics, scale and shift. RMSNorm: one statistic, scale only. "
                                            "Speed depends on the implementation."},
                    {"t": 146.6, "caption": "RMSNorm in code: multiply by the reciprocal root mean square, then by the "
                                            "learned weight."}],
        "body": [
            """<p>RMSNorm is simpler: one statistic instead of two, and no shift. Centering turned out to matter little.
Speed, though, depends on the implementation: on this CPU, PyTorch's built-in LayerNorm took <b>15 ms</b> and its
RMSNorm <b>46 ms</b> for 4,096 vectors of 4,096 numbers (a second run: 12 and 35 ms; times vary, the ratio of about
3 stays). With equally optimized code, RMSNorm does slightly less work.</p>""",
            '<div class="figrow">{fig0}{fig1}</div>',
            """<div class="box key"><b class="t">Key idea</b>RMSNorm keeps what matters (a fixed scale) and drops what
doesn't (centering and shift); measure speed rather than assume it.</div>""",
        ],
        "exercises": [
            {"kind": "code", "q": "<b>Try it yourself.</b> Check that the episode's <code>RMSNorm</code> class matches "
                                  "PyTorch's built-in one.",
             "code": """x = torch.randn(8, 128)
mine, theirs = RMSNorm(128), nn.RMSNorm(128, eps=1e-6)
print((mine(x) - theirs(x)).abs().max())""",
             "answer": "0 (or about 10⁻⁷): the same formula.",
             "why": "Both compute x / √(mean(x²) + ε) × weight, with weights initialized to 1."},
        ],
    },
]
