"""Foundations F8 — Averages and Spread."""
TITLE = "Averages and Spread"
TAGLINE = "Mean, standard deviation, and keeping numbers tame"
LABEL = "HOW LLMs WORK  ·  FOUNDATIONS F8"
NEXT = "F9 · Waves and Rotations"
USED_IN = [(6, "Attention II: The Math"), (9, "The Transformer Block")]
SECTIONS = [
    "Deep networks pass numbers through dozens of layers. If they drift too big or too small, training breaks. "
    "Two simple statistics help keep them in check: the mean, and the standard deviation.",
    "The mean is the average: add the numbers up, and divide by how many there are. "
    "For two, four, six, and eight, the mean is five.",
    "The standard deviation measures spread: how far the numbers typically are from the mean. "
    "Take each distance from the mean, square it, average those squares, and take the square root.",
    "For two, four, six, and eight, the distances are minus three, minus one, one, and three. "
    "The squares average to five, so the standard deviation is the square root of five: about two point two four.",
    "To normalize, subtract the mean, and divide by the standard deviation. Our numbers become about minus one point three four, "
    "minus zero point four five, zero point four five, and one point three four. The mean is now zero, and the spread is one.",
    "That's exactly what layer norm does to every token's vector, followed by a learned scale and shift.",
    "Spread also explains a detail in attention. Add up many random products, as a dot product does, "
    "and the spread of the total grows with the square root of how many you added.",
    "So with vectors of sixty-four numbers, dot products spread about eight times wider. Dividing by the square root "
    "of the key size brings them back to a spread of about one, which keeps the softmax well behaved.",
    "In NumPy, mean and std are built in, and normalizing is one line.",
    "You'll see normalizing in the layer norm of episode nine, and the square root of d "
    "in the attention formula of episode six. Next: waves and rotations.",
]
