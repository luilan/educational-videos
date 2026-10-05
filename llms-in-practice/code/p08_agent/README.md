# Episode 8 · Agents: Think, Act, Observe

The tool loop of episode 7, run until a goal is reached, with three made-up tools (`get_today`, `find_shops`,
`get_walking_time`) and a hard step limit.

```bash
pip install -r requirements.txt
python agent.py
```

Downloads `Qwen/Qwen2.5-3B-Instruct` (about 6 GB, loaded in bfloat16: about 7 GB of RAM; a minute or two on a CPU,
greedy decoding). For *"I want to buy gluten-free bread today. Where can I get it, and how long is the walk?"* it
prints every step: the agent makes three mistakes, reads the tools' error messages, recovers, and answers correctly in
six steps (Pane Vivo, 18 minutes), with the prompt growing from 422 to 814 tokens.

Experiments from the episode and the study guide:

- `VALIDATE = False`: the tools accept bad arguments, and the agent confidently answers that no shop sells gluten-free
  bread today (wrong).
- `MAX_STEPS = 3`: the agent is stopped before it can recover.
- `MODEL = "Qwen/Qwen2.5-1.5B-Instruct"` (with `dtype=torch.float32`): the smaller model invents shops.
