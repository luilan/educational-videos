# Episode 7 · Tool Use: How a Model Calls a Function

The model never runs code. It writes a request in an agreed text format; this script runs the function and pastes
the result back into the context. The weather service is made up.

```bash
pip install -r requirements.txt
python tool_use.py
```

Downloads `Qwen/Qwen2.5-1.5B-Instruct` (about 3 GB; CPU, greedy decoding). It prints:

1. for *"What is the weather like in Milan right now? Do I need an umbrella?"*: the model's real tool call
   (`<tool_call> {"name": "get_weather", "arguments": {"city": "Milan"}} </tool_call>`), the app running it, and the
   final answer (204 → 264 prompt tokens);
2. the full document the model read on its last turn: the tool description in the system prompt, the call and the
   `<tool_response>` are all just text in the context (episode 1);
3. a vaguer question, *"Do I need an umbrella in Milan right now?"*, for which this small model does not call the tool.

The `chat` function is the whole tool loop: generate → if there is a tool call, parse it, run it, append the result →
generate again. Add your own functions to `TOOLS` (type hints and a docstring become the tool description).
