"""Study guide content for How LLMs Work · Foundations, F09: Waves and Rotations.

Build:  python framework/study_guide.py foundations f09 --video <rendered mp4>
Figures: {"t": seconds into the rendered video, "caption": ...}; placed in body blocks with {fig0}, {fig1}, ...
Exercise kinds: mc (options), tf, short, number, order, code. Every exercise has an answer and a why.
All numbers were checked with NumPy (angles in radians, as in the video's code).
"""

LESSON = {
    "series": "How LLMs Work · Foundations",
    "label": "F09",
    "title": "Waves and Rotations",
    "tagline": "Sine, cosine, and turning arrows",
    "duration": "1:48",
    "intro": """<p>This lesson has one big idea: <b>sine and cosine come from a point moving around a circle</b>. Its
height traces a wave that repeats every full turn, 2π, and the same two functions can turn an arrow by any angle without
changing its length. Episode 4 uses both: waves of different frequencies to mark positions, and rotations that turn
relative position into a relative angle.</p>""",
    "prereq": """<div class="box note"><b>Before you start:</b> concepts 5 and 6 use 2-D vectors and their length from
F01 (Vectors) and matrix-times-vector from F03 (Matrices). The code uses NumPy; if it is new to you, F11 (NumPy in Three
Minutes) helps. These ideas are used in episode 4 (Where Am I? Position): sine waves mark positions, and rotations
appear as RoPE. Set your calculator to <b>radians</b> for the Checks, unless a question uses degrees (°).</div>""",
}

CONCEPTS = [
    {
        "title": "Sine and cosine on the unit circle",
        "segment": (8, 25),
        "figures": [{"t": 24.2, "caption": "A point on a circle of radius 1: its height is sin θ, its horizontal "
                                           "position is cos θ."}],
        "body": [
            """<p>Picture a point moving around a circle of <b>radius 1</b>. At any angle θ, its <b>height</b> is the
<b>sine</b> of the angle, sin θ, and its <b>horizontal position</b> is the <b>cosine</b>, cos θ. So the point sits at
<b>(cos θ, sin θ)</b>.</p>""",
            "{fig0}",
            """<p>Some angles to know: at 0° the point is at (1, 0), at 90° at the top, (0, 1), and at 30° at
(0.866, 0.5). Because the radius is 1, Pythagoras (F01) gives cos²θ + sin²θ = 1 at every angle: for 30°,
0.75 + 0.25 = 1.</p>""",
            """<div class="box key"><b class="t">Key idea</b>On a circle of radius 1, the point at angle θ is
<b>(cos θ, sin θ)</b>: cosine is the horizontal position, sine is the height. Both always stay between −1 and 1.</div>""",
        ],
        "exercises": [
            {"kind": "mc", "q": "The point is at the very top of the circle, at 90°. What are cos 90° and sin 90°?",
             "options": ["cos = 1, sin = 0", "cos = 0, sin = 1", "cos = 1, sin = 1", "cos = 0.5, sin = 0.5"],
             "answer": "B.", "why": "At the top the point is (0, 1): no horizontal offset, full height."},
            {"kind": "number", "q": "Where is the point at 180°? Give (cos 180°, sin 180°).",
             "answer": "(−1, 0).", "why": "Half a turn puts the point on the far left, at height 0."},
            {"kind": "tf", "q": "“For some angle θ, sin θ = 1.5.”",
             "answer": "False.", "why": "sin θ is the height of a point on a circle of radius 1, so it is always between "
                                        "−1 and 1."},
            {"kind": "number", "q": "At 60° the height is sin 60° ≈ 0.866. Use cos²θ + sin²θ = 1 to find cos 60°.",
             "answer": "0.5.", "why": "cos² 60° = 1 − 0.75 = 0.25, and √0.25 = 0.5 (positive, because at 60° the point is "
                                     "right of the center)."},
        ],
    },
    {
        "title": "The sine wave, and a full turn of 2π",
        "segment": (25, 46),
        "figures": [{"t": 36.4, "caption": "Plot the height: the sine wave. Cosine is the same wave, ¼ turn shifted."},
                    {"t": 46.1, "caption": "A full turn is 2π ≈ 6.28, so the wave repeats every 6.28."}],
        "body": [
            """<p>Let the angle keep growing and plot the height as you go: that is the <b>sine wave</b>. It goes up to 1,
down to −1, and repeats forever, because after a full turn the point is back where it started. The <b>cosine</b> is
the same wave, shifted by a quarter turn.</p>""",
            """<p>Angles are usually measured in <b>radians</b>: a full turn (360°) is <b>2π ≈ 6.28</b>, so the wave
repeats every 6.28 units, one <b>period</b>. Half a turn (180°) is π ≈ 3.14; a quarter turn (90°) is π/2 ≈ 1.57.</p>""",
            '<div class="figrow">{fig0}{fig1}</div>',
            """<div class="box key"><b class="t">Key idea</b>The sine wave is the moving point's height, plotted against
the angle. It repeats every full turn: <b>360° = 2π ≈ 6.28 radians</b>.</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "Convert to radians: (a) 180° (b) 90° (c) 30°.",
             "answer": "(a) π ≈ 3.14 (b) π/2 ≈ 1.57 (c) π/6 ≈ 0.52.",
             "why": "360° = 2π, so 180° = π and 90° = π/2. 30° is a twelfth of a turn: 2π ÷ 12 = π/6 (the video's code "
                    "writes <code>np.pi / 6</code>)."},
            {"kind": "tf", "q": "“sin(1) and sin(1 + 2π) are equal.”",
             "answer": "True.", "why": "Adding a full turn brings the point back to the same place (both ≈ 0.841)."},
            {"kind": "mc", "q": "The cosine wave is…",
             "options": ["the sine wave flipped upside down", "the sine wave shifted by a quarter turn",
                         "the sine wave, twice as fast", "a straight line"],
             "answer": "B.", "why": "It has the same shape, shifted by a quarter turn (π/2)."},
            {"kind": "number", "q": "How many full periods does sin θ complete between θ = 0 and θ = 4π ≈ 12.57?",
             "answer": "2.", "why": "One period is 2π, and 4π ÷ 2π = 2. The video draws exactly these two."},
        ],
    },
    {
        "title": "Frequency: how fast it repeats",
        "segment": (47, 57),
        "figures": [{"t": 57.2, "size": "small", "caption": "×2: twice as fast. ×½: half as fast."}],
        "body": [
            """<p><b>Frequency</b> is how fast a wave repeats. Multiply the input by a bigger number and the wave
oscillates faster: sin(2x) repeats twice as often as sin(x), every π ≈ 3.14 instead of every 6.28. Multiply by a smaller
number and it slows down: sin(x/2) repeats only every 4π ≈ 12.57. The height still goes from −1 to 1.</p>""",
            "{fig0}",
            """<div class="box key"><b class="t">Key idea</b>sin(w·x) repeats every <b>2π ÷ w</b>. A bigger w means a
faster wave (a higher frequency); a smaller w, a slower one.</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "How long is one period of sin(3x)?",
             "answer": "2π/3 ≈ 2.09.", "why": "Three times as fast, so a third of 6.28."},
            {"kind": "order", "q": "Order these waves by the length of their period, shortest first: "
                                   "<i>sin(x/2) · sin(4x) · sin(x) · sin(2x)</i>.",
             "answer": "sin(4x) → sin(2x) → sin(x) → sin(x/2).",
             "why": "Periods 1.57, 3.14, 6.28 and 12.57: the bigger the multiplier, the shorter the period."},
            {"kind": "tf", "q": "“Multiplying the input by 2 makes the wave twice as tall.”",
             "answer": "False.", "why": "It makes the wave repeat twice as fast. The height still only goes from −1 to 1."},
            {"kind": "number", "q": "You want a wave that repeats exactly every 100 units. What should w be in "
                                    "sin(w·x)?",
             "answer": "w = 2π ÷ 100 ≈ 0.0628.", "why": "The period is 2π ÷ w, so w = 2π ÷ period."},
        ],
    },
    {
        "title": "Many waves: a fingerprint for every position",
        "segment": (58, 66),
        "figures": [{"t": 65.4, "size": "small", "caption": "At position 10 the four waves read −0.54, −0.96, 0.60, "
                                                            "0.95."}],
        "body": [
            """<p>Combine several waves with different frequencies, here sin(p), sin(p/2), sin(p/4) and sin(p/8), and
read them all at the same position p. Every position gets its <b>own combination of values</b>: at p = 3 it is
0.14, 1.00, 0.68, 0.37; at p = 10 it is −0.54, −0.96, 0.60, 0.95.</p>""",
            "{fig0}",
            """<p>It works like the hands of a clock. The fast wave changes a lot from one position to the next but soon
repeats; the slow waves change little but take long to repeat. Together they tell positions apart.</p>""",
            """<div class="box key"><b class="t">Key idea</b>One wave repeats, so on its own it cannot tell far-apart
positions apart. Several waves with different frequencies give every position its own list of values.</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "With a calculator in radian mode, compute sin(p/4) at p = 10. Which of the four "
                                    "values in the picture is it?",
             "answer": "sin(2.5) ≈ 0.60: the third value.", "why": "10 ÷ 4 = 2.5, and sin(2.5) ≈ 0.598."},
            {"kind": "mc", "q": "Why not mark positions with the fast wave sin(p) alone?",
             "options": ["It repeats every 6.28 positions, so far-apart positions can get almost the same value",
                         "It is always 0 at whole-number positions",
                         "It changes too slowly to tell neighbors apart",
                         "Sine cannot take whole numbers as input"],
             "answer": "A.", "why": "For example sin(0) = 0 and sin(44) ≈ 0.02, but the slower sin(p/4) tells them apart: "
                                   "0 vs −1.00."},
            {"kind": "number", "q": "How many positions does the slowest wave, sin(p/8), take to repeat?",
             "answer": "About 50.3.", "why": "2π ÷ (1/8) = 16π ≈ 50.3."},
            {"kind": "tf", "q": "“From one position to the next, sin(p) changes more than sin(p/8) does.”",
             "answer": "True.", "why": "sin(p) is the fast wave, like the second hand; sin(p/8) moves slowly, like the "
                                       "hour hand."},
        ],
    },
    {
        "title": "Rotating an arrow",
        "segment": (66, 78),
        "figures": [{"t": 77.4, "size": "small", "caption": "(1, 0) turned by 30°: (0.866, 0.5), length still 1."}],
        "body": [
            """<p>Sine and cosine also <b>rotate</b> arrows. To turn a 2-D vector (x, y) by an angle θ, mix its two
numbers with cos θ and sin θ: <b>(x cos θ − y sin θ, x sin θ + y cos θ)</b>. The <b>length never changes</b>, only the
direction. The video turns v = (1, 0) by 30°: (1 × 0.866 − 0 × 0.5, 1 × 0.5 + 0 × 0.866) = <b>(0.866, 0.5)</b>. Its
length is still √(0.75 + 0.25) = 1; only the direction went from 0° to 30°.</p>""",
            "{fig0}",
            """<div class="box key"><b class="t">Key idea</b>Rotate (x, y) by θ: <b>(x cos θ − y sin θ,
x sin θ + y cos θ)</b>. The direction turns by θ; the length stays the same.</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "Rotate (0, 1) by 30°. (cos 30° ≈ 0.866, sin 30° = 0.5.)",
             "answer": "(−0.5, 0.866).", "why": "x = 0, y = 1: (0 − 1 × 0.5, 0 + 1 × 0.866). The arrow pointed at 90° "
                                               "and now points at 120°."},
            {"kind": "number", "q": "Rotate (3, 4) by 90° (cos 90° = 0, sin 90° = 1). Check that the length is "
                                    "unchanged.",
             "lines": 2,
             "answer": "(−4, 3); length 5 before and after.",
             "why": "(3 × 0 − 4 × 1, 3 × 1 + 4 × 0) = (−4, 3). √(9 + 16) = √(16 + 9) = 5."},
            {"kind": "tf", "q": "“Rotating a vector by a large enough angle makes it longer.”",
             "answer": "False.", "why": "A rotation only changes the direction, never the length."},
            {"kind": "mc", "q": "Rotating (2, 0) by 30° gives…",
             "options": ["(1.732, 1)", "(0.866, 0.5)", "(2, 0.5)", "(1, 1.732)"],
             "answer": "A.", "why": "Twice the video's result, 2 × (0.866, 0.5): the length stays 2. D is a turn of 60°."},
        ],
    },
    {
        "title": "Relative angles, and rotation in code",
        "segment": (78, 97),
        "figures": [{"t": 88.6, "size": "small", "caption": "Turn by 2θ and by 5θ: they are 3θ apart, from any "
                                                            "start."}],
        "body": [
            """<p>Rotate one arrow by two steps (2θ) and another by five (5θ). The angle between them is
<b>5θ − 2θ = 3θ</b>, no matter where both started. So if each position turns an arrow by its own number of steps,
<b>relative position becomes a relative angle</b>: positions 5 and 2 end up 3 steps apart, and so do 12 and 9.</p>""",
            "{fig0}",
            """<p>In code, a rotation is a 2 × 2 matrix built from the cosine and the sine. Multiplying it with a vector
(F03) computes exactly the formula of concept 5, and the length stays exactly 1.</p>""",
            """<pre class="code">theta = np.pi / 6                        # 30 degrees
R = np.array([[np.cos(theta), -np.sin(theta)],
              [np.sin(theta),  np.cos(theta)]])
v = np.array([1.0, 0.0])
R @ v                                    # array([0.866, 0.5])
np.linalg.norm(R @ v)                    # 1.0</pre>""",
            """<div class="box key"><b class="t">Key idea</b>Turning by 2θ and by 5θ leaves two arrows <b>3θ apart</b>,
from any starting direction: only the difference in steps shows. A rotation is a 2 × 2 matrix of cos and sin.</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "Use the video's step, θ = 20°. (a) At which angles do the 2θ and 5θ arrows point? "
                                    "(b) What is the angle between them? (c) Now start both 50° further round. "
                                    "Answer (a) and (b) again.",
             "lines": 2,
             "answer": "(a) 40° and 100°. (b) 60°. (c) 90° and 150°; still 60°.",
             "why": "The shared start adds the same 50° to both, so the difference 3θ = 60° does not change."},
            {"kind": "mc", "q": "Which pair of positions gives the same relative angle as positions 5 and 2?",
             "options": ["7 and 1", "10 and 7", "5 and 3", "6 and 2"],
             "answer": "B.", "why": "10 − 7 = 3 steps, like 5 − 2. The others are 6, 2 and 4 steps apart."},
            {"kind": "code", "q": "<b>Try it yourself.</b> The code wraps the video's matrix in a function "
                                  "<code>rot</code>. Predict, then run: (a) <code>rot(theta) @ np.array([0.0, 1.0])</code>. "
                                  "(b) <code>rot(theta) @ rot(theta) @ v</code>: by what angle has <code>v</code> turned? "
                                  "(c) <code>np.linalg.norm(rot(theta) @ np.array([3.0, 4.0]))</code>.",
             "code": """import numpy as np
np.set_printoptions(precision=3, suppress=True)

def rot(theta):
    return np.array([[np.cos(theta), -np.sin(theta)],
                     [np.sin(theta),  np.cos(theta)]])

theta = np.pi / 6                        # 30 degrees
v = np.array([1.0, 0.0])
print(rot(theta) @ v)                    # [0.866 0.5  ]""",
             "answer": "(a) [-0.5 0.866] · (b) [0.5 0.866], 60° · (c) 5.0",
             "why": """(a) The same as question 5.1: (0, 1) turned from 90° to 120°. (b) Two turns of 30° make one turn of
60°, and (cos 60°, sin 60°) = (0.5, 0.866): rotations add up, which is exactly why 5θ − 2θ = 3θ works. (c) (3, 4) has
length 5, and rotating never changes the length."""},
        ],
    },
]
