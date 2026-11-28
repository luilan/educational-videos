"""Episode 7, step 1: sample answers to word problems from Qwen2.5-0.5B-Instruct.

Each question gets 4 answers at temperature 0.7: 2 with the question alone (the model then explains step by step)
and 2 with "Answer in one short sentence." appended, so answers vary in length as well as in correctness. The reward
model later sees only the question and the answer, never the instruction.
Run as 4 shards in parallel (4 threads each), then merge:
    for k in 0 1 2 3; do python make_data.py $k 4 & done; wait; python make_data.py merge 4
Writes answers_shard<k>.json, then answers.json. About 35 minutes on 16 threads.
"""
import json
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from word_problems import correct, problems

NAME = "Qwen/Qwen2.5-0.5B-Instruct"
N_QUESTIONS, BATCH = 400, 32
SHORT = " Answer in one short sentence."


def sample(tok, model, texts, max_new_tokens):
    prompts = [tok.apply_chat_template([{"role": "user", "content": t}], add_generation_prompt=True, tokenize=False)
               for t in texts]
    enc = tok(prompts, return_tensors="pt", padding=True)
    with torch.no_grad():
        g = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=True, temperature=0.7, top_p=1.0, top_k=0,
                           pad_token_id=tok.pad_token_id)
    return tok.batch_decode(g[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)


def main(shard=0, shards=1):
    torch.set_num_threads(4)
    torch.manual_seed(shard)
    tok = AutoTokenizer.from_pretrained(NAME)
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(NAME, dtype=torch.float32).eval()
    allq = problems(N_QUESTIONS, seed=7)
    keep = list(range(shard, len(allq), shards))
    qs = [allq[k] for k in keep]
    out = [{"id": k, "question": q, "answer": a, "samples": []} for k, (q, a) in zip(keep, qs)]
    t0 = time.time()
    for style, suffix, max_new in (("short", SHORT, 64), ("long", "", 256)):
        for i in range(0, len(qs), BATCH // 2):
            chunk = list(range(i, min(i + BATCH // 2, len(qs))))
            texts = sample(tok, model, [qs[k][0] + suffix for k in chunk for _ in range(2)], max_new)
            for j, k in enumerate(chunk):
                for t in texts[2 * j:2 * j + 2]:
                    out[k]["samples"].append({"text": t.strip(), "style": style, "words": len(t.split()),
                                              "correct": correct(t, qs[k][1])})
            if i % 160 == 0:
                s = [x for o in out for x in o["samples"] if x["style"] == style]
                print(f"{style}: {len(s)} answers, accuracy {sum(x['correct'] for x in s) / len(s):.2f}, "
                      f"{time.time() - t0:.0f}s", flush=True)
        json.dump(out, open(f"answers_shard{shard}.json", "w"), indent=1)


def merge(shards):
    out = []
    for k in range(shards):
        out += json.load(open(f"answers_shard{k}.json"))
    out.sort(key=lambda o: o["id"])
    assert all(len(o["samples"]) == 4 for o in out), "a shard is incomplete"
    json.dump(out, open("answers.json", "w"), indent=1)
    s = [x for o in out for x in o["samples"]]
    for style in ("short", "long"):
        t = [x for x in s if x["style"] == style]
        print(style, len(t), "accuracy", sum(x["correct"] for x in t) / len(t), "mean words", sum(x["words"] for x in t) / len(t))


if __name__ == "__main__":
    import sys
    if sys.argv[1:2] == ["merge"]:
        merge(int(sys.argv[2]))
    else:
        main(int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) > 2 else main()
