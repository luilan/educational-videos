"""Study guide content for How LLMs Work: Deep Dive, episode 45: Circuits.

Build:  python framework/study_guide.py deep-dive d45 --video deep-dive/media/videos/d45_scene/1080p60/CircuitsVideo.mp4
Figures: {"t": seconds into the rendered video, "caption": ...}; placed in body blocks with {fig0}, {fig1}, ...
Exercise kinds: mc (options), tf, short, number, order, code. Every exercise has an answer and a why.
Every number comes from code/d45_circuits/circuits.py (GPT-2 small, 90 IOI prompts: 3 templates × 30 seeded name pairs)
or from the arithmetic shown. The early/middle/late story of the circuit is from Wang et al., 2022.
"""

LESSON = {
    "series": "How LLMs Work: Deep Dive",
    "label": "Episode 45",
    "title": "Circuits",
    "tagline": "Tracing an algorithm inside GPT-2",
    "duration": "2:16",
    "intro": """<p>This lesson answers one question: can we find <i>which parts</i> of a model carry out a task, and in what
order? The test case is <b>indirect object identification</b> (IOI): “When Kate and Peter went to the store, Peter gave a
drink to” → Kate. With <b>activation patching</b> we watch the information move from the repeated name to the last
position at layers 7–8, find a handful of attention heads that write the answer, and see that knocking them out hurts
but does not break the behaviour: real circuits have backups.</p>""",
    "prereq": """<div class="box note"><b>Before you start:</b> Deep Dive episodes 13 (the residual stream), 8 (the attention
matrix) and 42 (induction heads). Code: <code>code/d45_circuits</code> (PyTorch + transformers, a few minutes on a
CPU).</div>""",
}

CONCEPTS = [
    {
        "title": "The task and the metric",
        "segment": (8, 34),
        "figures": [{"t": 17.5, "caption": "The task: indirect object identification (IOI)."},
                    {"t": 32.5, "caption": "90 prompts: GPT-2 prefers the right name 100% of the time; mean logit "
                                           "difference 2.68."}],
        "body": [
            """<p>In “When Kate and Peter went to the store, Peter gave a drink to”, the right next word is the name that
was <i>not</i> repeated: Kate, the indirect object. We test GPT-2 small on 90 such prompts (3 sentence templates × 30
name pairs). It prefers the right name on all 90.</p>""",
            '<div class="figrow">{fig0}{fig1}</div>',
            """<p>Instead of accuracy we track a continuous score, the <b>logit difference</b>: logit(right name) −
logit(repeated name), at the last position. Its mean is 2.68: the right name is e^2.68 ≈ 15 times likelier than the
wrong one. A continuous metric shows partial effects that accuracy would hide.</p>""",
            """<div class="box key"><b class="t">Key idea</b>Pick a narrow task the model does reliably, and a continuous
metric for it: here, logit(IO) − logit(S).</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "A logit difference of 2.68 between two names: how many times likelier is the right "
                                    "name than the repeated one?",
             "answer": "About 14.6.", "why": "The ratio of two softmax probabilities is e^(difference): e^2.68 ≈ 14.6.", "key": {'parts': [{'label': None, 'value': 14.6, 'tol': 0.292, 'unit': None}]}},
            {"kind": "short", "q": "Why use the logit difference instead of “right name or not”?",
             "answer": "It is continuous, so it shows partial damage or partial recovery; accuracy only changes when "
                       "the top answer flips.",
             "why": "In part 4, removing six heads cuts the difference by 39% but accuracy only from 100% to 93%."},
        ],
    },
    {
        "title": "Activation patching through the residual stream",
        "segment": (34, 67),
        "figures": [{"t": 49.5, "caption": "Corrupted prompt: the second name swapped, so the answer flips (−3.30); copy in "
                                           "one activation from the clean run."},
                    {"t": 66.5, "caption": "Patching the residual stream: through layer 6 what matters sits at the repeated "
                                           "name; at layers 7–8 it moves to the last position."}],
        "body": [
            """<p>Make a <b>corrupted</b> prompt: “… Kate gave a drink to”. Now the right answer is Peter, and the mean
logit difference (Kate − Peter) is −3.30. Run the corrupted prompt but overwrite one activation with its value from the
clean run, and measure how much of the clean answer comes back:</p>
<p style="text-align:center"><code>recovered = (patched − corrupted) / (clean − corrupted)</code></p>
<p>1 means that activation alone carries everything that distinguishes the two runs; 0 means it carries nothing.</p>""",
            '<div class="figrow">{fig0}{fig1}</div>',
            """<table><tr><th>layer</th><th>Kate</th><th>Peter</th><th>Peter (2nd)</th><th>“to” (last)</th></tr>
<tr><td>0–3</td><td>0.00</td><td>0.00</td><td>1.00–0.99</td><td>0.00</td></tr>
<tr><td>6</td><td>0.00</td><td>0.00</td><td>0.91</td><td>0.03</td></tr>
<tr><td>7</td><td>0.00</td><td>0.00</td><td>0.65</td><td>0.41</td></tr>
<tr><td>8</td><td>0.00</td><td>0.00</td><td>0.03</td><td>0.98</td></tr>
<tr><td>9–11</td><td>0.00</td><td>0.00</td><td>≈ 0</td><td>0.97–1.00</td></tr></table>""",
            """<p>(Template 1, 30 prompts.) The first two names are identical in both runs, so patching them does nothing.
The difference starts at the second name; layers 7 and 8 read it and move the result to the last position, where the
answer is written. Patching shows <i>where</i> and <i>at what depth</i> the information lives.</p>""",
            """<div class="box key"><b class="t">Key idea</b>Activation patching asks: if I restore just this piece, does the
behaviour come back? It locates information in position and depth.</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "Clean 2.68, corrupted −3.30. A patch gives a logit difference of 0. What share is "
                                    "recovered?",
             "answer": "About 0.55.", "why": "(0 − (−3.30)) / (2.68 − (−3.30)) = 3.30 / 5.98 ≈ 0.55.", "key": {'parts': [{'label': None, 'value': 0.55, 'tol': 0.011, 'unit': None}]}},
            {"kind": "tf", "q": "“Patching the first ‘Kate’ recovers 0.00, so GPT-2 ignores the first name.”",
             "answer": "False.", "why": "The first name is the same in the clean and corrupted prompts; patching only "
                                         "measures what differs between the two runs.", "key": {'value': False}},
            {"kind": "short", "q": "Why does the recovered share at the repeated name fall from 0.91 at layer 6 to 0.03 "
                                   "at layer 8, while at the last position it rises to 0.98?",
             "answer": "Attention heads at layers 7–8 copy the relevant information from the repeated-name position to the "
                       "last position; after that, the last position already holds it.",
             "why": "Information moves between positions only through attention."},
        ],
    },
    {
        "title": "Heads and the circuit",
        "segment": (67, 97),
        "figures": [{"t": 82.0, "caption": "Patching single heads at the last position: 8.10, 8.6, 7.9, 9.9 recover about a "
                                           "quarter each; 10.7 pushes the other way."},
                    {"t": 96.0, "caption": "The story from Wang et al., 2022: early heads, middle heads, name movers."}],
        "body": [
            """<p>Patch each of the 144 heads (layer.head) at the last position. A handful matter: 8.10 and 8.6 recover
0.29 each, 7.9 0.23, 9.9 0.22, 7.3 0.14, 10.0 0.08. One head, 10.7, gives −0.35: restoring it makes the answer
<i>worse</i>; it argues against the answer the rest of the circuit writes.</p>""",
            '<div class="figrow">{fig0}{fig1}</div>',
            """<p>The published analysis (Wang et al., 2022) explains the roles. <b>Early heads</b> notice that a name
appears twice. <b>Middle heads</b> (S-inhibition: 7.3, 7.9, 8.6, 8.10) pass on “not the repeated name” to the last
position. <b>Name movers</b> (9.9, 10.0 and others) attend to the remaining name and copy it to the output. Our patching
finds the same middle and late heads; the full role analysis is in the paper.</p>""",
            """<div class="box key"><b class="t">Key idea</b>A circuit is a small set of components, each with a role,
composed in order: detect the duplicate, inhibit it, copy the other name.</div>""",
        ],
        "exercises": [
            {"kind": "order", "q": "Put the circuit's steps in order.",
             "items": ["name movers copy the remaining name to the output", "early heads detect the repeated name",
                       "middle heads pass on “not the repeated name” to the last position"],
             "answer": "early heads detect the repeated name → middle heads pass on “not the repeated name” → name movers "
                       "copy the remaining name.",
             "why": "Each step needs the previous one's output, which is why they sit at increasing depth.", "key": {'items': ['early heads detect the repeated name', 'middle heads pass on “not the repeated name”', 'name movers copy the remaining name']}},
            {"kind": "short", "q": "What does a negative recovered share (10.7: −0.35) mean?",
             "answer": "Restoring that head's clean output moves the result further from the clean answer: the head "
                       "pushes against the answer the circuit is writing.",
             "why": "Wang et al. call such heads negative name movers; they reduce overconfidence."},
        ],
    },
    {
        "title": "Knockout and redundancy",
        "segment": (97, 121),
        "figures": [{"t": 112.5, "caption": "Mean-ablating heads at the last position: top 6 → 1.64, still right 93%; 6 "
                                            "random heads → 2.63."},
                    {"t": 120.0, "caption": "Patching in code: a forward hook that overwrites one activation."}],
        "body": [
            """<p>Patching shows what is <i>sufficient</i> to restore the answer; <b>knockout</b> tests what is
<i>necessary</i>. Replace a head's output at the last position by its average over all 90 prompts (mean ablation), so it
carries no prompt-specific information.</p>""",
            """<table><tr><th>heads removed</th><th>logit difference</th><th>right name</th></tr>
<tr><td>none</td><td>2.68</td><td>100%</td></tr>
<tr><td>top 3 (8.10, 8.6, 7.9)</td><td>2.08</td><td>100%</td></tr>
<tr><td>top 6 (+ 9.9, 7.3, 10.0)</td><td>1.64</td><td>93%</td></tr>
<tr><td>6 random heads (mean of 5 draws)</td><td>2.63</td><td></td></tr></table>""",
            '<div class="figrow">{fig0}{fig1}</div>',
            """<p>The chosen heads matter far more than random ones, yet GPT-2 still answers right 93% of the time:
<b>backup heads</b> take over when the main ones are gone. Circuits found by patching are real but rarely the whole
story; the behaviour is spread over redundant paths.</p>""",
            """<div class="box key"><b class="t">Key idea</b>Patching finds what is sufficient; knockout tests what is
necessary. Real circuits are redundant.</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "What fraction of the logit difference is lost when the top 6 heads are removed?",
             "answer": "About 39%.", "why": "(2.68 − 1.64) / 2.68 ≈ 0.39.", "key": {'parts': [{'label': None, 'value': 39, 'tol': 0.78, 'unit': '%'}]}},
            {"kind": "mc", "q": "Why compare with removing 6 random heads?",
             "options": ["To check the effect is about these heads, not about removing any 6 heads",
                         "Random heads are the backup heads", "To make the numbers smaller"],
             "answer": "To check the effect is about these heads, not about removing any 6 heads.",
             "why": "Random heads drop the difference only from 2.68 to 2.63.", "key": {'choice': 0}},
            {"kind": "code", "q": "<b>Try it yourself.</b> Patch one block's output at one position with a forward hook.",
             "code": """def patch(layer, pos, clean_acts):
    def hook(module, inp, out):
        h = out[0].clone()
        h[:, pos] = clean_acts[layer][:, pos]
        return (h,) + out[1:]
    return model.transformer.h[layer].register_forward_hook(hook)

handle = patch(8, -1, clean_acts)        # clean_acts saved by a hook during the clean run
logits = model(corrupted_ids).logits
handle.remove()""",
             "answer": "With layer 8 and the last position, about 98% of the clean logit difference comes back "
                       "(template 1).",
             "why": "This is part 2 of the episode's code: by layer 8 the answer sits at the last position."},
        ],
    },
]
