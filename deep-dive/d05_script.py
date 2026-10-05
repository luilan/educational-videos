"""How LLMs Work: Deep Dive, episode 5 — RoPE: Rotating Vectors to Encode Order. One entry per narration section."""
TITLE = "RoPE"
TAGLINE = "Rotating vectors to encode order"
NEXT = "Long Context: Stretching RoPE"
SECTIONS = [
    # 1. Hook
    "Last time, GPT-2's learned position table stopped at one thousand and twenty-four. Most modern models, like "
    "Llama, Mistral and Kwen, use a different idea instead: rotary position embedding, or Rope. "
    "It encodes position by rotating vectors.",
    # 2. Rotation in 2-D
    "Picture a query as an arrow, with just two numbers. At position zero, leave it alone. At position one, rotate it "
    "by a small angle. At position two, by twice that angle. And so on. The arrow keeps its length. "
    "Only its direction carries the position.",
    # 3. Why it works
    "Here's why this is clever. Attention scores are dot products, and a dot product depends on the angle between two "
    "vectors. Rotate the query by its position, m, and the key by its position, n. The angle between them changes by "
    "m minus n. So the score depends only on how far apart the two tokens are, not on where they are.",
    # 4. Real check
    "Let's check, with a random query and key of a real head size, sixty-four. Positions five and two: a score of "
    "five point nine two four seven two five. Positions one hundred and five and one hundred and two: exactly the same. "
    "Ten thousand and five and ten thousand and two: the same, to six decimal places. A distance of one gives a "
    "different score, and again, it's the same everywhere.",
    # 5. Many frequencies
    "Sixty-four numbers make thirty-two pairs, and each pair rotates at its own speed, like the hands of a clock. "
    "In Kwen two point five, the fastest pair makes a full turn every six tokens. Another, every two hundred. Another, "
    "every six thousand. The slowest, only every four million tokens. Fast pairs track nearby order. Slow pairs "
    "measure long distances.",
    # 6. Where it's applied
    "Rope is applied inside every attention layer, to the queries and the keys only. Nothing is added to the token "
    "embeddings, and the values are not rotated. Position only shapes who attends to whom.",
    # 7. Stretching
    "And because position is just an angle, it can be rescaled. Slow the rotations down, and the same angles cover "
    "a longer text. That's how models stretch their context window.",
    # 8. Code
    "In code, Rope is four lines: compute the angles, split the vector in two halves, and rotate each pair with a "
    "cosine and a sine. Our version matches the one in Hugging Face's Kwen code to within two ten-millionths.",
    # 9. Outro
    "Next up: long context, and how models stretch Rope to read far more than they were trained on.",
]
