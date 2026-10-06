"""Study guide content for How LLMs Work · Foundations, F13: PyTorch and Autograd.

Build:  python framework/study_guide.py foundations f13 --video <AutogradVideo.mp4>
Figures: {"t": seconds into the rendered video, "caption": ...}; placed in body blocks with {fig0}, {fig1}, ...
Exercise kinds: mc (options), tf, short, number, order, code. Every exercise has an answer and a why.
All numbers and code outputs were checked by running Python (torch).
"""

LESSON = {
    "series": "How LLMs Work · Foundations",
    "label": "F13",
    "title": "PyTorch and Autograd",
    "tagline": "How one line computes every gradient",
    "duration": "1:51",
    "intro": """<p>This lesson opens up the one line that does all the calculus in training: <code>loss.backward()</code>.
PyTorch <b>records</b> every operation on a tensor as a graph, then walks that graph <b>in reverse</b> with the chain
rule. One call fills in the gradient of every weight, and an optimizer uses those gradients to update the weights.</p>""",
    "prereq": """<div class="box note"><b>Before you start:</b> this lesson builds on F10 (Slopes and Gradients: the
derivative, the chain rule, gradient descent), F11 (NumPy arrays) and F12 (weights and training). It is the machinery
behind the training loop of episode 11 (Training: Learning from Mistakes) and the tiny GPT of episode 12 (Build a Tiny
GPT).</div>""",
}

CONCEPTS = [
    {
        "title": "Tensors that remember",
        "segment": (8, 31),
        "figures": [{"t": 23.4, "caption": "A tensor is an array of numbers, like a NumPy array, that can remember "
                                           "its history."},
                    {"t": 30.5, "caption": "With requires_grad=True, PyTorch records every operation that uses the "
                                           "tensor."}],
        "body": [
            """<p>In episode 11, one line did all the calculus: <code>loss.backward()</code>. To see how, start with
PyTorch's basic object, the <b>tensor</b>: an array of numbers, just like a NumPy array. The difference is that a tensor
can <b>remember how it was computed</b>.</p>""",
            '<div class="figrow">{fig0}{fig1}</div>',
            """<p>Create a tensor with <code>requires_grad=True</code> and PyTorch starts <b>recording</b> every operation
that uses it, like a tape. Later, that record is what lets PyTorch find gradients for you.</p>""",
            """<div class="box key"><b class="t">Key idea</b>A tensor is a NumPy-like array that can keep a history.
<code>requires_grad=True</code> switches the recording on.</div>""",
        ],
        "exercises": [
            {"kind": "mc", "q": "For this lesson, what is the important difference between a PyTorch tensor and a "
                                "NumPy array?",
             "options": ["A tensor can hold only one number", "A tensor can remember how it was computed, so its "
                         "gradients can be found later", "A tensor is always two-dimensional",
                         "NumPy arrays cannot do arithmetic"],
             "answer": "B.", "why": "Both are arrays of numbers; only the tensor can record its history for autograd.", "key": {'choice': 1}},
            {"kind": "tf", "q": "“PyTorch records every operation on every tensor, whether or not "
                                "<code>requires_grad=True</code> was set.”",
             "answer": "False.", "why": "Only tensors that require gradients (and results computed from them) are "
                                        "recorded. Plain tensors are just numbers.", "key": {'value': False}},
            {"kind": "short", "q": "You create <code>a = torch.tensor(2.0)</code> and <code>b = torch.tensor(5.0, "
                                   "requires_grad=True)</code>, compute <code>c = a * b</code> and call "
                                   "<code>c.backward()</code>. (a) Which of a and b gets a gradient? (b) What is it?",
             "lines": 2,
             "answer": "(a) Only b. (b) b.grad = 2.",
             "why": "a was not recorded, so <code>a.grad</code> stays <code>None</code>. The slope of a × b with respect "
                    "to b is a = 2."},
        ],
    },
    {
        "title": "The forward pass builds a graph",
        "segment": (31, 45),
        "figures": [{"t": 45.6, "caption": "y = x² + 2x at x = 3: x goes into a square (9) and a times two (6); "
                                           "the add gives y = 15."}],
        "body": [
            """<p>Take x = 3 and compute y = x² + 2x. As PyTorch computes, it builds a <b>graph</b>: x goes into a
<i>square</i> (giving 9) and into a <i>times two</i> (giving 6), and the two results go into an <i>add</i>:
y = 9 + 6 = <b>15</b>.</p>""",
            "{fig0}",
            """<p>This is the <b>forward pass</b>. It gives the answer, 15, and as a side effect a record of how the
answer was reached: one node per operation, with arrows showing which values feed which.</p>""",
            """<div class="box key"><b class="t">Key idea</b>The forward pass computes the value <b>and</b> records a
graph of the operations. Nothing about slopes has been computed yet.</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "Same function y = x² + 2x, now with x = 5. What are the two middle values in the "
                                    "graph, and what is y?",
             "answer": "25 and 10, so y = 35.", "why": "The square gives 5² = 25, the times two gives 10, the add "
                                                       "gives 35.", "key": {'parts': [{'label': 'x²', 'value': 25, 'tol': 0.5, 'unit': None}, {'label': '2x', 'value': 10, 'tol': 0.5, 'unit': None}, {'label': 'y', 'value': 35, 'tol': 0.5, 'unit': None}]}},
            {"kind": "short", "q": "Describe the graph PyTorch builds for z = (x + 1)², and the values in it for "
                                   "x = 4.",
             "lines": 2,
             "answer": "x → add 1 → square → z; values 5, then 25.",
             "why": "Two operations in a chain this time: 4 + 1 = 5, then 5² = 25. Only one path runs from x to z."},
            {"kind": "tf", "q": "“The graph is built when you call <code>backward()</code>.”",
             "answer": "False.", "why": "It is recorded during the forward computation. <code>backward()</code> "
                                        "walks the graph that is already there.", "key": {'value': False}},
        ],
    },
    {
        "title": "backward(): the chain rule in reverse",
        "segment": (45, 64),
        "figures": [{"t": 54.9, "caption": "y.backward() walks the graph in reverse. The two paths from x contribute "
                                           "6 and 2: x.grad = 8."},
                    {"t": 63.8, "caption": "The check: the slope of x² + 2x is 2x + 2, which is 8 at x = 3."}],
        "body": [
            """<p>Call <code>y.backward()</code> and PyTorch walks the graph <b>in reverse</b>, applying the <b>chain
rule</b> (F10) at every step. Each node knows its own slope: at x = 3 the square contributes d(x²)/dx = 2x = 6, the times
two contributes 2. Since x reaches y along two paths, the contributions <b>add up</b>: 6 + 2 = 8. The result lands in
<code>x.grad</code>.</p>""",
            '<div class="figrow">{fig0}{fig1}</div>',
            """<p>Check by hand: the derivative of x² + 2x is 2x + 2, and 2 · 3 + 2 = <b>8</b>. Autograd gives exactly
the slope you would find with calculus, without you writing any derivative.</p>""",
            """<div class="box key"><b class="t">Key idea</b><code>y.backward()</code> applies the chain rule
automatically, from the output back to the inputs. The slope dy/dx is stored in <code>x.grad</code>.</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "y = x² + 2x at x = 5. What will <code>x.grad</code> be after "
                                    "<code>y.backward()</code>?",
             "answer": "12.", "why": "The paths contribute 2 · 5 = 10 and 2, so 10 + 2 = 12. Check: 2x + 2 = 12.", "key": {'parts': [{'label': None, 'value': 12, 'tol': 0.5, 'unit': None}]}},
            {"kind": "number", "q": "z = (x + 1)² at x = 4, as in question 2.2. What will <code>x.grad</code> be after "
                                    "<code>z.backward()</code>?",
             "answer": "10.", "why": "Chain rule along the single path: the square contributes 2 · 5 = 10, the add 1 "
                                    "contributes 1, and 10 × 1 = 10.", "key": {'parts': [{'label': None, 'value': 10, 'tol': 0.5, 'unit': None}]}},
            {"kind": "mc", "q": "In the video, why are 6 and 2 added to give <code>x.grad</code>?",
             "options": ["Because the order of operations says so", "Because x reaches y along two paths, and the "
                         "effects of both paths add up", "Because PyTorch always adds 2 to a gradient",
                         "Because backward() was called twice"],
             "answer": "B.", "why": "Nudging x changes both x² and 2x, and y feels both changes.", "key": {'choice': 1}},
            {"kind": "tf", "q": "“To use autograd, you first have to work out the derivative formula (like 2x + 2) "
                                "yourself.”",
             "answer": "False.", "why": "That is the point of autograd: the formula was only used to check the answer.", "key": {'value': False}},
        ],
    },
    {
        "title": "One backward call, every gradient",
        "segment": (64, 75),
        "figures": [{"t": 74.4, "caption": "A real model: millions of steps up to the loss. One backward pass fills in "
                                           "a gradient for every weight (illustrative values)."}],
        "body": [
            """<p>A real model is the same thing, just bigger. The forward pass from the inputs to the <b>loss</b> has
millions of steps, and the loss depends on millions of weights. PyTorch records all of it, and <b>one</b> call to
<code>loss.backward()</code> fills in the gradient of <b>every</b> weight: how the loss changes when that weight is
nudged.</p>""",
            "{fig0}",
            """<div class="box key"><b class="t">Key idea</b>However big the model, a single backward call gives one
gradient per weight. This is backpropagation (F10), done for you.</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "The tiny GPT of episode 12 has 818,241 parameters. After one call to "
                                    "<code>loss.backward()</code>, how many gradient numbers have been filled in?",
             "answer": "818,241.", "why": "One gradient per parameter, all from the same single backward call.", "key": {'parts': [{'label': None, 'value': 818241, 'tol': 0.5, 'unit': None}]}},
            {"kind": "mc", "q": "What does the gradient of one weight tell you?",
             "options": ["The weight's current value", "How the loss changes when that weight is nudged a little, "
                         "and in which direction", "How many times the weight was used", "The learning rate"],
             "answer": "B.", "why": "It is the slope of the loss with respect to that weight.", "key": {'choice': 1}},
            {"kind": "short", "q": "In the figure, w₂ has gradient −0.40 and w₅ has gradient 0.35. To lower the loss, "
                                   "should each weight go up or down?",
             "answer": "w₂ up, w₅ down.",
             "why": "Step against the slope (F10). A negative gradient means the loss falls as the weight rises."},
            {"kind": "tf", "q": "“A model with a million weights needs a million backward calls, one per weight.”",
             "answer": "False.", "why": "One backward pass walks the whole graph once and gives every gradient.", "key": {'value': False}},
        ],
    },
    {
        "title": "The optimizer, and zero_grad",
        "segment": (75, 90),
        "figures": [{"t": 89.6, "caption": "Adam reads the gradients and updates the weights; zero_grad clears the "
                                           "gradients. Forget it, and 8 + 8 = 16."}],
        "body": [
            """<p>Gradients alone change nothing. An <b>optimizer</b>, such as <b>Adam</b>, reads them and
<b>updates the weights</b> (in the video 0.50 → 0.49, −1.20 → −1.19, …). Then <code>optimizer.zero_grad()</code> clears
the gradients before the next backward pass. Why? PyTorch <b>adds</b> new gradients to whatever is already in
<code>.grad</code>: call backward twice on y = x² + 2x without clearing, and <code>x.grad</code> is 8 + 8 = <b>16</b>.</p>""",
            "{fig0}",
            """<div class="box key"><b class="t">Key idea</b><code>backward()</code> only fills in gradients;
the optimizer's <code>step()</code> changes the weights. Clear old gradients with <code>zero_grad()</code> every step,
or they pile up.</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "y = x² + 2x at x = 3. You compute y and call <code>y.backward()</code> three times "
                                    "in a row, never clearing. What is <code>x.grad</code>?",
             "answer": "24.", "why": "Each call adds another 8: 8 + 8 + 8 = 24.", "key": {'parts': [{'label': None, 'value': 24, 'tol': 0.5, 'unit': None}]}},
            {"kind": "order", "q": "Put one training step in order: <i>optimizer.step() · loss.backward() · "
                                   "optimizer.zero_grad() · compute the loss (forward pass)</i>.",
             "answer": "zero_grad → forward → backward → step.",
             "why": "zero_grad only has to come before backward (the tiny GPT calls it right after the forward pass). "
                    "step needs the fresh gradients, so it comes last.", "key": {'items': ['optimizer.zero_grad()', 'compute the loss (forward pass)', 'loss.backward()', 'optimizer.step()']}},
            {"kind": "number", "q": "Use plain gradient descent instead of Adam: new x = x − 0.1 × x.grad, starting at "
                                    "x = 3. What is the new x with the correct gradient 8, and with the doubled 16?",
             "answer": "2.2 with 8; 1.4 with 16.",
             "why": "3 − 0.8 = 2.2 and 3 − 1.6 = 1.4. A forgotten zero_grad makes the step twice as big as it should be.", "key": {'parts': [{'label': 'gradient 8', 'value': 2.2, 'tol': 0.05, 'unit': None}, {'label': 'gradient 16', 'value': 1.4, 'tol': 0.05, 'unit': None}]}},
            {"kind": "tf", "q": "“<code>loss.backward()</code> changes the weights.”",
             "answer": "False.", "why": "It only fills in <code>.grad</code>. The optimizer's step changes the weights.", "key": {'value': False}},
        ],
    },
    {
        "title": "Autograd in code",
        "segment": (90, 97),
        "figures": [{"t": 96.3, "size": "small", "caption": "The code from the video: create, compute, backward, "
                                                           "read."}],
        "body": [
            """<p>The whole idea fits in four lines: create a tensor that requires gradients, compute with it, call
<code>backward()</code>, and read the gradient.</p>""",
            "{fig0}",
            """<pre class="code">import torch

x = torch.tensor(3.0, requires_grad=True)
y = x ** 2 + 2 * x          # PyTorch records the graph
y.backward()                # chain rule, in reverse
x.grad                      # tensor(8.)</pre>""",
            """<div class="box key"><b class="t">Key idea</b>Create with <code>requires_grad=True</code> → compute →
<code>backward()</code> → read <code>.grad</code>.</div>""",
        ],
        "exercises": [
            {"kind": "mc", "q": "You leave out <code>requires_grad=True</code>: <code>x = torch.tensor(3.0)</code>, "
                                "then the same two lines. What happens at <code>y.backward()</code>?",
             "options": ["x.grad becomes 8", "x.grad becomes 0", "An error: nothing was recorded, so there is no "
                         "graph to walk back", "x.grad becomes 15"],
             "answer": "C.", "why": "PyTorch raises a RuntimeError (“element 0 of tensors does not require grad and "
                                   "does not have a grad_fn”).", "key": {'choice': 2}},
            {"kind": "code", "q": "<b>Try it yourself.</b> Now there are two tensors that require gradients, and the "
                                  "loop runs the same computation twice. (a) Predict the two printed lines, then run it. "
                                  "(b) Add one line so that both lines print the same gradients. (c) Change x to "
                                  "<code>-1.0</code>: what does the first line print?",
             "code": """import torch

x = torch.tensor(3.0, requires_grad=True)
w = torch.tensor(2.0, requires_grad=True)

for step in range(2):
    y = w * x ** 2 + 2 * x
    y.backward()
    print(step, y.item(), x.grad.item(), w.grad.item())""",
             "answer": "(a) 0 24.0 14.0 9.0 and 1 24.0 28.0 18.0 · (c) 0 0.0 -2.0 1.0",
             "why": """(a) y = 2 · 9 + 6 = 24. The slope with respect to x is 2wx + 2 = 14; with respect to w it is
x² = 9. The second backward adds to the first, so the gradients double: 28 and 18.
(b) At the end of the loop body add <code>x.grad.zero_(); w.grad.zero_()</code> (the job
<code>optimizer.zero_grad()</code> does for every weight); both lines then print 24.0 14.0 9.0.
(c) y = 2 · 1 − 2 = 0, x.grad = 2 · 2 · (−1) + 2 = −2, w.grad = (−1)² = 1."""},
        ],
    },
]
