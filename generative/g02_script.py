"""How Generative Models Work: Images and Sound, episode 2 — Autoencoders. One entry per narration section.

Every number comes from code/g02_autoencoders (train_autoencoder.py, 2026-11-21): the same convolutional autoencoder
trained for 15 epochs on 20,000 Shapes pictures with codes of 2, 4, 8, 16 and 32 numbers, tested on 2,000 new ones.
"""
TITLE = "Autoencoders"
TAGLINE = "Squeeze a picture into a few numbers, then rebuild it"
NEXT = "Variational Autoencoders"
SECTIONS = [
    # 1. Hook
    "Last time, pixels sampled one by one never formed a shape. A model has to learn how pixels depend on each "
    "other. One way to force that: squeeze the picture through a bottleneck.",
    # 2. The idea
    "An autoencoder has two halves. The encoder turns three thousand and seventy-two numbers into a short code, say "
    "eight numbers. The decoder turns the code back into a picture. Training only asks for one thing: the output "
    "should match the input. No labels, no captions.",
    # 3. The network
    "Ours is small. Three convolution layers halve the picture each time, thirty-two, sixteen, eight, four, and a "
    "linear layer makes the code. The decoder mirrors it. About three hundred and seventy thousand parameters, and "
    "about a minute of training on a laptop CPU.",
    # 4. Code size
    "How small can the code be? With two numbers, the rebuilt pictures are blurry blobs, often the wrong shape. Four "
    "percent of pixels come back clearly wrong. With eight numbers, under one percent. With thirty-two, the "
    "pictures are almost perfect, from a code ninety-six times smaller.",
    # 5. Why it works
    "It works because these pictures only vary in a few ways: shape, color, position and size. Eight numbers are "
    "enough room for all of them. Nobody told the network what they were. It found them, because that was the only "
    "way through the bottleneck.",
    # 6. Random codes
    "So, can we generate? Pick random code numbers, with the same average and spread as real codes, and decode them. "
    "With two numbers, we get plausible blobs. With eight or thirty-two, we get mush, eight times further from any "
    "real picture than a rebuilt one.",
    # 7. Halfway
    "Halfway between the codes of a yellow square and a purple circle, the decoder draws faint smudges in both "
    "places, in green and blue, colors that neither picture has.",
    # 8. Why: islands
    "Here is why. With a two-number code, every test picture is a point on a plane. The points form islands, one "
    "for each position. Training only ever visited the islands. Between them, nothing told the decoder what to "
    "draw, so it draws anything.",
    # 9. Code
    "In code, training is one line of loss: the squared difference between the decoded picture and the original.",
    # 10. Next
    "Next: variational autoencoders. Fill the gaps between the islands, so that any code you pick decodes to a "
    "picture.",
]
