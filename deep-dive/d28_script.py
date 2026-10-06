"""How LLMs Work: Deep Dive, episode 28 — Mixture of Experts. One entry per narration section."""
TITLE = "Mixture of Experts"
TAGLINE = "Many experts, only a few awake per token"
NEXT = "State-Space Models"
SECTIONS = [
    # 1. Hook
    "What if a model could have many times more parameters, but use only a few of them for each token? That's a "
    "mixture of experts. Mixtral 8x7B, for example, has eight experts per layer and uses two for each token.",
    # 2. Idea
    "The M L P in each block is replaced by several expert M L Ps, plus a small router. For every token, the router "
    "scores the experts, keeps the best one or two, and mixes their outputs by those scores. The other experts don't "
    "run at all.",
    # 3. Experiment
    "We build it: eight experts per layer, in our tiny GPT. That's five and a half times the parameters of the dense "
    "model. But with top-one routing, each token still uses about the same number as before.",
    # 4. Top-1
    "The dense model reaches one point six four four. The mixture with top-one routing: one point six eight seven. "
    "Worse. At this small scale and short training, each expert sees only an eighth of the tokens, and it doesn't pay "
    "off.",
    # 5. Balance
    "There's another problem. Without help, the router plays favorites: one expert gets twenty percent of the tokens, "
    "another only four. A small extra loss that rewards even use fixes it: every expert then gets between eleven and "
    "fourteen percent.",
    # 6. Top-2
    "Now let every token use two experts. The loss drops to one point five nine six, the best so far. But that uses "
    "more compute per token, so the fair comparison is a dense model just as big: twice as wide. It reaches one point "
    "six two eight. The mixture still wins.",
    # 7. Why
    "That's the deal: at the same compute per token, a mixture of experts can hold far more parameters, more knowledge, "
    "and it can come out ahead. The price is memory for all the experts, and some care to keep them all busy.",
    # 8. Code
    "In code, the router is one linear layer and a softmax; top k picks the experts, and each expert runs only on the "
    "tokens sent to it.",
    # 9. Outro
    "That's mixture of experts. Next: state-space models, which replace attention with a running memory.",
]
