"""Foundations F9 — Waves and Rotations."""
TITLE = "Waves and Rotations"
TAGLINE = "Sine, cosine, and turning arrows"
LABEL = "HOW LLMs WORK  ·  FOUNDATIONS F9"
NEXT = "F10 · Slopes and Gradients"
USED_IN = [(4, "Where Am I? Position")]
SECTIONS = [
    "Episode four encodes positions with waves, and with rotations. Let's see where sine and cosine come from.",
    "Picture a point moving around a circle of radius one. At any angle, its height is the sine of that angle, "
    "and its horizontal position is the cosine.",
    "Let the angle keep growing, and plot the height as it goes. You get the sine wave: up to one, down to minus one, "
    "repeating forever. The cosine is the same wave, shifted by a quarter turn.",
    "Angles are usually measured in radians. A full turn is two pi, about six point two eight. "
    "So the wave repeats every six point two eight units.",
    "Frequency is how fast the wave repeats. Multiply the input by a bigger number, and the wave oscillates faster. "
    "Multiply by a smaller number, and it slows down.",
    "Combine several waves with different frequencies, and every position gets its own combination of values, "
    "like the hands of a clock.",
    "Sine and cosine also rotate arrows. To turn a 2-D vector by an angle, mix its two numbers "
    "with the cosine and sine of that angle. The length never changes, only the direction.",
    "Rotate one arrow by two steps, and another by five, and the angle between them is three steps, "
    "no matter where you started. Relative position becomes a relative angle.",
    "In code, a rotation is a two by two matrix built from the cosine and the sine. The length stays exactly one.",
    "You'll see sine waves marking positions, and rotations as rope, in episode four. "
    "Next: slopes and gradients, the math of learning.",
]
