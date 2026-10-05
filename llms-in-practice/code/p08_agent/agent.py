"""LLMs in Practice, episode 8: Agents — think, act, observe.

An agent is the tool loop from episode 7, run until a goal is reached: the model decides the next action, the
program runs it, the result goes back into the context, repeat. The shops, the calendar and the map are made up.

    pip install -r requirements.txt
    python agent.py
Downloads Qwen/Qwen2.5-3B-Instruct (about 6 GB, loaded in bfloat16: about 7 GB of RAM); runs on a CPU in a
minute or two, greedy decoding. Smaller models fail at multi-step tasks (see the episode).
"""
import json
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen2.5-3B-Instruct"
MAX_STEPS = 8                                     # a hard limit: agents can loop forever
VALIDATE = True                                   # set to False to see the agent fail (episode 8)
SYSTEM = ("You are a helpful assistant. Use the tools to gather every fact you need, one step at a time. "
          "Never say what you are going to do: call the tool. "
          "Reply without a tool call only when you can give the complete final answer.")
tok = AutoTokenizer.from_pretrained(MODEL)
llm = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).eval()


def get_today() -> dict:
    """Get today's day of the week."""
    return {"today": "Wednesday"}


def find_shops(product: str, day: str) -> dict:
    """Find shops in Milan that sell a product on a given day of the week.

    Args:
        product: What to buy, e.g. sourdough bread
        day: The day of the week, e.g. Monday
    """
    shops = {"Bella's Bakery": ["Friday"], "Pane Vivo": ["Monday", "Wednesday", "Saturday"]}
    weekdays = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    if VALIDATE and day.capitalize() not in weekdays:          # a helpful error lets the agent recover
        return {"error": f"day must be a weekday name such as Monday, not {day!r}. Call get_today first."}
    if "gluten" not in product.lower():
        return {"shops": list(shops)}
    return {"product": product, "day": day, "shops": [s for s, days in shops.items() if day.capitalize() in days]}


def get_walking_time(destination: str) -> dict:
    """Get the walking time from the user's home to a place.

    Args:
        destination: Where the walk ends, e.g. Pane Vivo
    """
    minutes = {"Pane Vivo": 18, "Bella's Bakery": 25}
    if VALIDATE and destination not in minutes:
        return {"error": f"unknown place {destination!r}. Use a shop name returned by find_shops."}
    return {"destination": destination, "minutes": minutes.get(destination, 30)}


TOOLS = {f.__name__: f for f in (get_today, find_shops, get_walking_time)}


def generate(messages):
    ids = tok.apply_chat_template(messages, tools=list(TOOLS.values()), add_generation_prompt=True,
                                  return_tensors="pt")
    out = llm.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=200, do_sample=False,
                       pad_token_id=tok.eos_token_id)
    return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip(), ids.shape[1]


def run_agent(goal):
    print(f"GOAL: {goal}\n")
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": goal}]
    for step in range(1, MAX_STEPS + 1):
        reply, n = generate(messages)                                     # THINK: decide the next action
        calls = [json.loads(c) for c in re.findall(r"<tool_call>\s*(.*?)\s*</tool_call>", reply, re.S)]
        if not calls:
            print(f"step {step} ({n} tokens) FINAL ANSWER:\n{reply}")
            return
        messages.append({"role": "assistant", "content": "",
                         "tool_calls": [{"type": "function", "function": c} for c in calls]})
        for call in calls:
            result = TOOLS[call["name"]](**call["arguments"])            # ACT: the program runs the tool
            print(f"step {step} ({n} tokens) {call['name']}({call['arguments']}) -> {result}")
            messages.append({"role": "tool", "name": call["name"],      # OBSERVE: the result joins the context
                             "content": json.dumps(result)})
    print(f"stopped after {MAX_STEPS} steps without a final answer")


run_agent("I want to buy gluten-free bread today. Where can I get it, and how long is the walk?")
