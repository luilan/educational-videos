"""Foundations F12 (extra) — Neural Networks in Three Minutes."""
TITLE = "Neural Networks in Three Minutes"
TAGLINE = "Neurons, layers, weights, and learning"
LABEL = "HOW LLMs WORK  ·  FOUNDATIONS F12  ·  EXTRA"
NEXT = "F13 · PyTorch and Autograd"
USED_IN = [(8, "The MLP: Where Facts Live"), (11, "Training: Learning from Mistakes"), (12, "Build a Tiny GPT")]
SECTIONS = [
    "The main series keeps saying: neural network. So what is one, really? Strip away the hype, "
    "and it's a function, built from very simple parts.",
    "The basic part is a neuron. It takes some numbers in, multiplies each one by a weight, adds them up with a bias, "
    "and passes the result through a bend, like the ReLU we met earlier.",
    "Say the inputs are two and three, the weights are zero point five and one, and the bias is minus one. "
    "That's one, plus three, minus one: three. ReLU keeps it, so this neuron fires, with a value of three.",
    "Put many neurons side by side, all reading the same inputs, and you have a layer. "
    "Their weights together form a matrix, so a whole layer is one matrix multiplication, plus the bend.",
    "Stack layers, and the outputs of one become the inputs of the next. "
    "Early layers find simple patterns. Later layers combine them into more complex ones.",
    "The weights and biases are called parameters. They're the only thing that changes when a network learns. "
    "Our tiny GPT had about eight hundred thousand of them. Big models have billions.",
    "A network is used in two ways. Inference means running it: inputs flow forward, and out comes a prediction. "
    "Training means adjusting it: compare the prediction with the right answer, "
    "and nudge every parameter to do a little better.",
    "Nobody writes the rules inside. We choose the shape: how many layers, and how many neurons. "
    "The training data does the rest.",
    "In code, a neuron is a dot product, a bias, and a ReLU. A layer does many neurons at once.",
    "You'll see neural networks throughout the series: the MLP in episode eight, training in episode eleven, "
    "and a complete network in episode twelve. Next: PyTorch, and how it computes gradients for us.",
]
