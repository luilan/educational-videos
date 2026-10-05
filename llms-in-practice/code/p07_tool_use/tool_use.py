"""LLMs in Practice, episode 7: Tool Use — how a model calls a function.

The model never runs code. It writes a request in an agreed text format; your program runs the function and
pastes the result back into the context. The weather here is made up.

    pip install -r requirements.txt
    python tool_use.py
Downloads Qwen/Qwen2.5-1.5B-Instruct (about 3 GB); runs on a CPU, greedy decoding.
"""
import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
tok = AutoTokenizer.from_pretrained(MODEL)
llm = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).eval()


def get_weather(city: str) -> dict:
    """Get the current weather in a city.

    Args:
        city: The name of the city, e.g. Milan
    """
    return {"city": city, "temperature_c": 18, "sky": "light rain"}     # a real app would call a weather API


TOOLS = {"get_weather": get_weather}


def generate(messages):
    ids = tok.apply_chat_template(messages, tools=list(TOOLS.values()), add_generation_prompt=True,
                                  return_tensors="pt")
    out = llm.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=80, do_sample=False,
                       pad_token_id=tok.eos_token_id)
    return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip(), ids.shape[1]


def chat(question):
    print(f"\n=== {question}")
    messages = [{"role": "user", "content": question}]
    while True:
        reply, n = generate(messages)
        print(f"[model, {n} prompt tokens]\n{reply}")
        if "<tool_call>" not in reply:                  # a normal answer: done
            return messages
        # 1. Parse the request the model wrote.
        call = json.loads(reply.split("<tool_call>")[1].split("</tool_call>")[0])
        # 2. Run the real function: this is the only place where something happens in the world.
        result = TOOLS[call["name"]](**call["arguments"])
        print(f"[app runs {call['name']}({call['arguments']}) -> {result}]")
        # 3. Paste the result back into the context and let the model continue.
        messages += [{"role": "assistant", "content": "", "tool_calls": [{"type": "function", "function": call}]},
                     {"role": "tool", "name": call["name"], "content": json.dumps(result)}]


messages = chat("What is the weather like in Milan right now? Do I need an umbrella?")
print("\n=== What the model actually read on its last turn (episode 1: it's all one document):")
print(tok.apply_chat_template(messages, tools=list(TOOLS.values()), tokenize=False, add_generation_prompt=True))

chat("Do I need an umbrella in Milan right now?")   # a vaguer question: the small model does not call the tool
