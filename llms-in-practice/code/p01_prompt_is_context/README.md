# Episode 1 · Prompts Are Just Context

What the model really sees when you chat: not bubbles, but one document of tokens.

```bash
pip install -r requirements.txt
python what_the_model_sees.py
```

It downloads only the tokenizer of `Qwen/Qwen2.5-0.5B-Instruct` (about 10 MB, not the model), and prints:

1. the conversation flattened by the model's real **chat template**, with special tokens `<|im_start|>` / `<|im_end|>`
   marking who speaks, ending with an open assistant turn;
2. the same document as **token IDs** (55 tokens);
3. how many tokens are sent on each turn (28, then 55): the model has no memory, so the app resends the whole chat.

Try it: change the messages, add a long pasted document, or swap `MODEL` for another instruct model and compare
its template.
