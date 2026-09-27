"""Video 14 — From GPT to Chatbot. One entry per narration section."""
TITLE = "From GPT to Chatbot"
TAGLINE = "How a text predictor learns to be an assistant"
NEXT = None  # series finale
SECTIONS = [
    # 1. The base model
    "A model trained only to predict the next token of internet text is called a base model. "
    "It's a powerful autocomplete. Ask it a question, and it might answer. "
    "Or it might continue with three more questions, as if it were writing a quiz.",
    # 2. Pretraining
    "This first stage is called pretraining. It's where almost all the computing goes, "
    "and where the knowledge comes from: trillions of tokens of text.",
    # 3. Supervised fine-tuning
    "Stage two is supervised fine-tuning. We keep training the same model, with the same next-token loss, "
    "but now on examples of conversations: a user asks, and an assistant answers helpfully. "
    "The model learns the format, and the style.",
    # 4. Chat templates
    "Conversations are turned into text, with special tokens that mark who is speaking. "
    "The model is trained to predict the assistant's turns. "
    "To the model, a chat is still just one long document to continue.",
    # 5. Preferences
    "Stage three teaches judgment. People compare two answers from the model, and pick the better one. "
    "From many thousands of these comparisons, the model learns what better means.",
    # 6. RLHF and DPO
    "One way to use them is RLHF: reinforcement learning from human feedback. "
    "Train a reward model to predict which answers people prefer, "
    "then nudge the chatbot toward answers that score higher. "
    "A simpler method, called DPO, learns from the preference pairs directly.",
    # 7. Other signals
    "Many labs also use AI feedback, guided by written principles, "
    "and reinforcement learning on problems with checkable answers, like math and code.",
    # 8. Same machine
    "Through all of this, the machinery doesn't change. It's the same transformer: the same embeddings, "
    "attention, and next-token prediction. Only the data and the training signal change.",
    # 9. Series recap
    "So here's the whole journey. Text becomes tokens. Tokens become vectors, with positions. "
    "Attention lets them talk, and MLPs let them think, stacked in blocks. The final vector becomes probabilities, "
    "we pick a token, and repeat. Training shapes every number, and fine-tuning turns it into an assistant.",
    # 10. Farewell
    "That's how an LLM works, from scratch. Thanks for watching.",
]
