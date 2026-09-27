"""Study guide content for How LLMs Work · Foundations, F10: Slopes and Gradients.

Build:  python framework/study_guide.py foundations f10 --video <GradientsVideo.mp4>
Figures: {"t": seconds into the rendered video, "caption": ...}; placed in body blocks with {fig0}, {fig1}, ...
Exercise kinds: mc (options), tf, short, number, order, code. Every exercise has an answer and a why.
All numbers and code outputs were checked by running Python.
"""

LESSON = {
    "series": "How LLMs Work · Foundations",
    "label": "F10",
    "title": "Slopes and Gradients",
    "tagline": "How a model knows which way to adjust",
    "duration": "2:00",
    "intro": """<p>This lesson answers one question: when a model is wrong, <b>which way should each weight move</b>? The
answer is the <b>slope</b>: how fast the output changes when you nudge the input. Take a small step against the slope,
repeat, and you have <b>gradient descent</b>, the way LLMs are trained.</p>""",
    "prereq": """<div class="box note"><b>Before you start:</b> this lesson builds on functions and their graphs from
F04 (Straight Lines and Bends) and on F01 (Vectors), because the gradient is a vector. It is the math behind training
in episode 11 (Training: Learning from Mistakes) and episode 12 (Build a Tiny GPT).</div>""",
}

CONCEPTS = [
    {
        "title": "What a slope measures",
        "segment": (8, 26),
        "figures": [{"t": 16.1, "caption": "Which way should the weight move to lower the loss?"},
                    {"t": 23.4, "caption": "Steep: a small nudge Δx gives a big change Δy."}],
        "body": [
            """<p>Training adjusts a model's weights to reduce the <b>loss</b>, the number that says how badly the model
is doing. To know <b>which way</b> to move each weight, we need one idea from calculus: the <b>slope</b>. It says how
much the output changes when you nudge the input a little: <b>slope = Δy / Δx</b>, where Δx is the nudge and Δy the
change it causes. Where the curve is <b>steep</b>, a small nudge gives a big change; where it is <b>flat</b>, almost
none.</p>""",
            '<div class="figrow">{fig0}{fig1}</div>',
            """<div class="box key"><b class="t">Key idea</b><b>slope = Δy / Δx</b>: the change in the output per unit
of nudge. Steep means a big slope, flat a slope near 0. Turned around: a nudge Δx changes the output by about
<b>slope × Δx</b>.</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "You nudge the input from 2 to 2.5, and the output goes from 4 to 5.5. What is the "
                                    "slope?",
             "answer": "3.", "why": "Δx = 0.5 and Δy = 1.5, so Δy / Δx = 1.5 / 0.5 = 3: the output changes three times "
                                   "as fast as the input."},
            {"kind": "mc", "q": "Look at the curve in the second picture (steep on the left, flat on the right). Where "
                                "does the same small nudge of the input change the output the most?",
             "options": ["On the right, where the curve is flat", "On the left, where the curve is steep",
                         "Everywhere the same, because the nudge is the same size",
                         "Nowhere: nudging the input never changes the output"],
             "answer": "B.", "why": "The steeper the curve, the bigger Δy for the same Δx. On the flat part, Δy is "
                                   "tiny."},
            {"kind": "tf", "q": "“Where the slope is close to 0, nudging the input a little barely changes the "
                                "output.”",
             "answer": "True.", "why": "A slope near 0 is a flat curve: Δy ≈ slope × Δx is almost nothing."},
            {"kind": "number", "q": "At the current weight, the loss curve has slope 2. You nudge the weight up by "
                                    "0.01. About how much does the loss change, and does it go up or down?",
             "answer": "About 0.02, up.", "why": "Change ≈ slope × nudge = 2 × 0.01 = 0.02. The slope is positive, so "
                                                 "moving the weight up moves the loss up."},
        ],
    },
    {
        "title": "The derivative: the slope at one point",
        "segment": (26, 40),
        "figures": [{"t": 36.5, "caption": "Zoomed in 1000 times around (3, 9): nudging x by 0.001 raises y by "
                                           "0.006001, so Δy / Δx ≈ 6."},
                    {"t": 39.9, "caption": "The slope at x = 3 is 6. In general the derivative of x² is 2x."}],
        "body": [
            """<p>Take y = x². At x = 3 the output is 9. Nudge x by a tiny amount, from 3 to 3.001, and the output
becomes 9.006001. So Δy = 0.006001 and Δy / Δx = 6.001 ≈ 6: right there, the output grows about <b>six times as fast</b>
as the input. The tinier the nudge, the closer the ratio gets to exactly 6.</p>""",
            '<div class="figrow">{fig0}{fig1}</div>',
            """<p>That rate is the <b>derivative</b>: the slope at a single point (the green line that just touches the
curve). You don't have to nudge every time: for x², the derivative at any x is <b>2x</b>. At x = 3 that is
2 × 3 = 6.</p>""",
            """<div class="box key"><b class="t">Key idea</b>The <b>derivative</b> is the slope at one point: Δy / Δx for
a tiny nudge. <b>The derivative of x² is 2x</b>, so the slope changes along the curve: 6 at x = 3, 0 at the bottom.</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "Use the rule 2x. What is the slope of y = x² at x = 5? At x = 0? At x = −2?",
             "answer": "10, 0 and −4.", "why": "2 × 5 = 10; at 0 the curve is flat (the bottom of the bowl); at −2 the "
                                               "slope is negative: moving right, the curve goes down."},
            {"kind": "number", "q": "Repeat the video's nudge at x = 2: compute 2.001², then Δy and Δy / Δx. Which "
                                    "value does the rule 2x predict?",
             "lines": 2,
             "answer": "2.001² = 4.004001, Δy = 0.004001, Δy / Δx = 4.001. The rule predicts 4.",
             "why": "2 × 2 = 4. The tiny leftover 0.001 is there because the nudge is small but not zero."},
            {"kind": "number", "q": "Now use a bigger nudge at x = 3: from 3 to 3.1, so the output is 3.1² = 9.61. "
                                    "What is Δy / Δx? Is it closer to 6 than 6.001, or further away?",
             "answer": "6.1, further away.", "why": "Δy = 0.61 and 0.61 / 0.1 = 6.1. The curve bends, so a big nudge "
                                                    "measures an average over a wide stretch. Tiny nudges give the slope "
                                                    "at the point."},
            {"kind": "tf", "q": "“The slope of y = x² is 6 everywhere on the curve.”",
             "answer": "False.", "why": "The slope is 2x, so it depends on where you are: 6 at x = 3, 10 at x = 5, 0 at "
                                        "x = 0. Only a straight line has the same slope everywhere."},
        ],
    },
    {
        "title": "Step against the slope: gradient descent",
        "segment": (41, 66),
        "figures": [{"t": 50.6, "caption": "Slope > 0 at x = 3, so step left: against the slope."},
                    {"t": 65.6, "caption": "Learning rate 0.1: x slides toward the minimum at 0."}],
        "body": [
            """<p>The slope also tells you <b>which way is downhill</b>. At x = 3 on y = x² the slope is +6, so to make
the output smaller we step left, <b>against the slope</b> (where the slope is negative, that means stepping right).
That is <b>gradient descent</b>: <b>new x = old x − learning rate × slope</b>, where the <b>learning rate</b> sets the
step size. With 0.1: 3 − 0.1 × 6 = <b>2.4</b>. Repeat: 1.92, 1.536, 1.229, 0.983, toward the minimum at 0.</p>""",
            '<div class="figrow">{fig0}{fig1}</div>',
            """<div class="box key"><b class="t">Key idea</b><b>x ← x − learning rate × slope.</b> Subtracting the slope
always moves downhill, whatever its sign. The learning rate controls how big each step is.</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "Do the third step yourself: x = 1.92 on y = x², learning rate 0.1. What is the "
                                    "slope there, and what is the new x?",
             "answer": "Slope 3.84, new x = 1.536.", "why": "Slope = 2 × 1.92 = 3.84; 1.92 − 0.1 × 3.84 = 1.92 − 0.384 "
                                                            "= 1.536, the next number in the video."},
            {"kind": "number", "q": "Start instead at x = −2 (same curve, learning rate 0.1). What is the slope, where "
                                    "does one step take you, and which way did x move?",
             "answer": "Slope −4, new x = −1.6: it moved right.",
             "why": "−2 − 0.1 × (−4) = −2 + 0.4 = −1.6. Minus a negative slope is a step to the right, again toward the "
                    "minimum at 0."},
            {"kind": "mc", "q": "At the current weight, the slope of the loss is −5. To reduce the loss, you should:",
             "options": ["Decrease the weight", "Increase the weight",
                         "Leave it alone: a negative slope means the loss is already at its minimum",
                         "Set the weight to −5"],
             "answer": "B.", "why": "A negative slope means the loss goes down to the right. The update "
                                   "w − lr × (−5) = w + 5 × lr increases the weight."},
            {"kind": "order", "q": "Put one round of gradient descent in order: <i>subtract the result from x · repeat "
                                   "from the new x · compute the slope at x · multiply the slope by the learning "
                                   "rate</i>.",
             "answer": "compute the slope → multiply by the learning rate → subtract from x → repeat.",
             "why": "Exactly the rule x ← x − learning rate × slope, applied again and again."},
        ],
    },
    {
        "title": "The gradient: one slope per weight",
        "segment": (66, 77),
        "figures": [{"t": 76.3, "size": "small", "caption": "Loss over two weights, seen from above. ∇L points "
                                                            "uphill; the steps go the opposite way."}],
        "body": [
            """<p>A real model has millions of weights, not one, and each has its own slope: how much the loss changes
when you nudge <i>that</i> weight alone, written ∂L/∂w₁, ∂L/∂w₂, … The <b>gradient</b> ∇L collects one slope for each
weight into a single <b>vector</b>. It points <b>uphill</b>, so we step the <b>opposite way</b>: every weight moves
against its own slope, all at once.</p>""",
            "{fig0}",
            """<div class="box key"><b class="t">Key idea</b>The <b>gradient</b> is a vector with <b>one slope per
weight</b>. It points uphill, so each step goes the other way:
<b>w&nbsp;←&nbsp;w&nbsp;−&nbsp;learning&nbsp;rate&nbsp;×&nbsp;∇L</b>.</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "How many numbers are in the gradient of a model with 3 weights? With 1,000,000 "
                                    "weights?",
             "answer": "3 and 1,000,000.", "why": "One slope per weight, so the gradient is as long as the list of "
                                                  "weights."},
            {"kind": "number", "q": "Two weights are at (2.4, 1.0). The slopes there are ∂L/∂w₁ = 2.4 and "
                                    "∂L/∂w₂ = 4.0. Take one step with learning rate 0.2. Where do the weights go?",
             "answer": "(1.92, 0.2).", "why": "Each weight steps against its own slope: 2.4 − 0.2 × 2.4 = 1.92 and "
                                              "1.0 − 0.2 × 4.0 = 0.2. This is the first yellow step in the picture."},
            {"kind": "tf", "q": "“The gradient points downhill, so we add it to the weights.”",
             "answer": "False.", "why": "The gradient points uphill (toward more loss). We subtract it, stepping the "
                                        "opposite way."},
            {"kind": "mc", "q": "The gradient for three weights is (0.5, −2.0, 0.0). With learning rate 0.1, which "
                                "statement is right?",
             "options": ["w₃ changes the most, because its slope is 0",
                         "w₂ is increased, and it moves the most of the three",
                         "All three weights are decreased by the same amount",
                         "w₁ is increased"],
             "answer": "B.", "why": "The changes are −0.1 × (0.5, −2.0, 0.0) = (−0.05, +0.2, 0): w₂ goes up by 0.2, "
                                   "w₁ goes down a little, and w₃ (slope 0) does not move."},
        ],
    },
    {
        "title": "Slopes multiply: the chain rule and backprop",
        "segment": (77, 100),
        "figures": [{"t": 91.1, "caption": "Slopes multiply: dz/dx = 3 × 2 = 6."},
                    {"t": 99.6, "caption": "Backward from the loss, layer by layer."}],
        "body": [
            """<p>Layers are <b>functions inside functions</b>: y = f(x), then z = g(y). The <b>chain rule</b> says their
slopes <b>multiply</b>: if y changes 3 times as fast as x (dy/dx = 3) and z twice as fast as y (dz/dy = 2), then z
changes <b>3 × 2 = 6</b> times as fast as x. <b>Backpropagation</b> (backprop) applies this layer by layer, from the
loss <b>backward</b> to every weight, so <b>one backward pass</b> gives the whole gradient.</p>""",
            '<div class="figrow">{fig0}{fig1}</div>',
            """<div class="box key"><b class="t">Key idea</b>Chain rule: <b>dz/dx = dy/dx × dz/dy</b>. Backprop runs it
from the loss backward: <b>one backward pass → the whole gradient</b>.</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "Three layers in a row have slopes 2, 0.5 and 4. How many times as fast as the "
                                    "input does the final output change?",
             "answer": "4 times.", "why": "Slopes multiply: 2 × 0.5 × 4 = 4."},
            {"kind": "number", "q": "Now y = x² and then z = 5y. At x = 3, find dy/dx, dz/dy and dz/dx.",
             "answer": "6, 5 and 30.", "why": "dy/dx = 2x = 6; z = 5y changes 5 times as fast as y; 6 × 5 = 30. Check "
                                              "by nudging: x = 3.001 gives z = 45.030005, a change of 0.030005, "
                                              "which is ≈ 30 × 0.001."},
            {"kind": "tf", "q": "“Backpropagation needs one separate backward pass for each weight.”",
             "answer": "False.", "why": "One backward pass from the loss gives the slopes of all the weights: the whole "
                                        "gradient."},
            {"kind": "mc", "q": "A network runs input → layer A → layer B → loss. Which layer does backprop reach first?",
             "options": ["Layer A, following the data", "Layer B: it starts at the loss and works backward",
                         "Both at once, each on its own", "Neither: backprop only looks at the loss itself"],
             "answer": "B.", "why": "Backprop starts at the loss and multiplies on each layer's slope as it moves back "
                                   "toward the input, so B (next to the loss) comes before A."},
        ],
    },
    {
        "title": "Gradient descent in code",
        "segment": (100, 105),
        "figures": [{"t": 104.9, "size": "small", "caption": "The code from the video: compute the slope, step against "
                                                             "it."}],
        "body": [
            """<p>In code, gradient descent is a short loop: <b>compute the slope, step against it</b>. Here the slope
of x² is written as a function, and five steps from x = 3 print the numbers from concept 3.</p>""",
            "{fig0}",
            """<pre class="code">slope = lambda x: 2 * x          # derivative of x ** 2

x, lr = 3.0, 0.1
for step in range(5):
    x = x - lr * slope(x)        # step against the slope
    print(round(x, 3))           # 2.4, 1.92, 1.536, 1.229, 0.983</pre>""",
            """<div class="box key"><b class="t">Key idea</b>The whole method is one line inside a loop:
<code>x = x - lr * slope(x)</code>. Training a real model does the same, with the gradient in place of the slope.</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "Show that each step of this loop turns x into exactly 0.8 × x. Then use it: what is "
                                    "x after 10 steps (3 decimals)?",
             "lines": 2,
             "answer": "x − 0.1 × 2x = 0.8x; after 10 steps x = 3 × 0.8¹⁰ ≈ 0.322.",
             "why": "Every step keeps 80 % of x, so the path is 3, 2.4, 1.92, … and 3 × 0.8¹⁰ = 0.322. It gets close "
                    "to 0 but never quite reaches it."},
            {"kind": "tf", "q": "“Each step in the printout moves x by the same amount.”",
             "answer": "False.", "why": "The steps are 0.6, 0.48, 0.384, …: they shrink because the slope 2x shrinks as "
                                        "x nears the minimum."},
            {"kind": "code", "q": "<b>Try it yourself.</b> This version measures a slope by nudging, like the video's "
                                  "zoom, and lets you change the learning rate. Predict each output, then run it. "
                                  "(a) What does line (a) print? (b) What does <code>descend(3.0, 0.5)</code> print, and "
                                  "why? (c) What does <code>descend(3.0, 1.1)</code> print? What goes wrong?",
             "code": """def f(x):
    return x ** 2

def slope_at(f, x, h=0.001):
    return (f(x + h) - f(x)) / h     # nudge x, measure the change

def descend(x, lr, steps=5):
    for step in range(steps):
        x = x - lr * 2 * x           # the slope of x ** 2 is 2x
        print(round(x, 3), end="  ")
    print()

print(round(slope_at(f, 3), 3))      # (a)
descend(3.0, 0.5)                    # (b)
descend(3.0, 1.1)                    # (c)""",
             "answer": "(a) 6.001 · (b) 0.0  0.0  0.0  0.0  0.0 · (c) −3.6  4.32  −5.184  6.221  −7.465",
             "why": """(a) The same nudge as the video: (9.006001 − 9) / 0.001 = 6.001. (b) 3 − 0.5 × 6 = 0: this
learning rate jumps straight to the minimum, and at 0 the slope is 0, so x stays put. (c) Each step is
x − 2.2x = −1.2x: x jumps over the minimum and lands further away every time. A learning rate that is too big makes
gradient descent blow up instead of settling down."""},
        ],
    },
]
