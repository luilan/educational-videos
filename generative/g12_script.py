"""How Generative Models Work: Images and Sound, episode 12 — CLIP: Matching Pictures and Words. One entry per narration section.

Every number comes from code/g12_clip (train_clip.py, 2026-11-22): a tiny CLIP trained on Shapes pictures and
captions, with episode 10's eight combinations held out of training.
"""
TITLE = "CLIP: Matching Pictures and Words"
TAGLINE = "Two encoders that learn to agree."
NEXT = "Latent Diffusion"
SECTIONS = [
    # 1. Hook
    "Labels are fine when there are three shapes and five colors. But people type sentences. To draw from words, a "
    "model first has to know which words go with which pictures. That's what CLIP learns.",
    # 2. Two encoders
    "CLIP has two encoders. One turns a picture into a list of sixty-four numbers, the other turns a caption into "
    "sixty-four numbers. When the caption fits the picture, the two lists should point the same way.",
    # 3. Contrastive
    "Training uses a batch of pairs. Compare every picture with every caption, and you get a table of similarities. "
    "The true pairs sit on the diagonal: push them up, and everything else down. Each picture has to pick out its "
    "own caption from the whole batch, and each caption its picture.",
    # 4. Ours
    "Ours is tiny: a small convolutional network for pictures, and a two-layer transformer that reads the caption's "
    "words. Its whole vocabulary is seventeen words.",
    # 5. Learning order
    "To read a picture, pick the closest of all a hundred and thirty-five captions. Before training, that's right "
    "almost never. After ten steps, every position is right. Color by step thirty, and shape last, around step a "
    "hundred and fifty. Then every caption is right.",
    # 6. Held out
    "The real test: eight combinations it never saw, not as pictures and not as captions, like a purple triangle at "
    "the top left. It reads every one of them correctly. It learned each word on its own, and combines them.",
    # 7. Loss floor
    "One oddity: the loss stops at about point five seven. A batch of a hundred and twenty-eight pairs often holds "
    "the same caption twice, and two identical captions can't be told apart. Work out the best possible loss for "
    "that, and you get exactly point five seven.",
    # 8. Search
    "The table works in both directions. Type a caption, rank the pictures by similarity, and you have a search "
    "engine, even for a combination it never trained on.",
    # 9. Real CLIP
    "OpenAI's CLIP did this with four hundred million pictures and captions from the web. And a text encoder that "
    "already knows what words look like is just what a text-to-image model needs: Stable Diffusion uses CLIP's.",
    # 10. Code
    "In code: normalize both, multiply them into the table, and apply cross-entropy along the rows and the columns.",
    # 11. Next
    "Next: latent diffusion, making pictures in a compressed space.",
]
