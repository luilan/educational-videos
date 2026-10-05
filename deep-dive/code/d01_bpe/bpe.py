"""How LLMs Work: Deep Dive, episode 1 — Byte-Pair Encoding, step by step.

Train a byte-level BPE tokenizer from scratch on Tiny Shakespeare: start from the 256 possible bytes, and repeatedly
merge the most frequent adjacent pair into a new token. Then compare with GPT-2's real tokenizer (50,000 merges).

    pip install -r requirements.txt
    python bpe.py
"""
from collections import Counter
from pathlib import Path

TEXT = (Path(__file__).resolve().parents[3] / "how-llms-work" / "tiny_gpt" / "input.txt").read_text()
train = TEXT[:200_000]                       # learn merges on the first 200,000 characters
test = TEXT[-100_000:]                        # measure on text the tokenizer never saw
N_MERGES = 500


def pair_counts(ids):
    return Counter(zip(ids, ids[1:]))


def merge(ids, pair, new_id):
    out, i = [], 0
    while i < len(ids):
        if i + 1 < len(ids) and (ids[i], ids[i + 1]) == pair:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


# 1. Start: every byte is a token (256 possible values). UTF-8 turns text into bytes.
ids = list(train.encode("utf-8"))
vocab = {i: bytes([i]) for i in range(256)}
merges = {}
print(f"start: {len(train):,} characters -> {len(ids):,} byte tokens")

# 2. Repeat: count adjacent pairs, merge the most frequent one into a new token.
for step in range(N_MERGES):
    counts = pair_counts(ids)
    pair, n = counts.most_common(1)[0]
    new_id = 256 + step
    ids = merge(ids, pair, new_id)
    merges[pair] = new_id
    vocab[new_id] = vocab[pair[0]] + vocab[pair[1]]
    if step < 12 or step + 1 in (50, 100, 200, 500):
        print(f"merge {step + 1:>3}: {vocab[pair[0]]!r:>8} + {vocab[pair[1]]!r:<8} -> {vocab[new_id]!r:<12}"
              f" seen {n:>5,} times; text is now {len(ids):,} tokens")


# 3. Encoding new text: apply the learned merges in the order they were learned.
def encode(text):
    ids = list(text.encode("utf-8"))
    while len(ids) > 1:
        counts = pair_counts(ids)
        pair = min(counts, key=lambda p: merges.get(p, float("inf")))   # earliest-learned merge present
        if pair not in merges:
            break
        ids = merge(ids, pair, merges[pair])
    return ids


def decode(ids):
    return b"".join(vocab[i] for i in ids).decode("utf-8", errors="replace")


sample = "To be, or not to be, that is the question."
toks = encode(sample)
print(f"\n{sample!r} -> {len(toks)} tokens:")
print(" | ".join(repr(vocab[t].decode('utf-8', 'replace')) for t in toks))
assert decode(toks) == sample
held = encode(test)
print(f"\nunseen text: {len(test):,} characters -> {len(held):,} tokens "
      f"({len(test.encode('utf-8')) / len(held):.2f} bytes per token after {N_MERGES} merges)")
longest = sorted(vocab.values(), key=len)[-5:]
print("longest tokens learned:", [t.decode("utf-8", "replace") for t in longest])

# 4. GPT-2's trick: split the text into words first (a space stays at the start of the word), and only merge
#    inside a word. Merges can no longer glue words, punctuation and names together.
import re
SPLIT = re.compile(r" ?[A-Za-z]+| ?[0-9]+| ?[^\sA-Za-z0-9]+|\s+")


def train_words(text, n_merges):
    words = Counter(tuple(w.encode("utf-8")) for w in SPLIT.findall(text))     # each distinct word, with its count
    vocab2 = {i: bytes([i]) for i in range(256)}
    merges2 = {}
    for step in range(n_merges):
        counts = Counter()
        for w, c in words.items():
            for pair in zip(w, w[1:]):
                counts[pair] += c
        pair = counts.most_common(1)[0][0]
        new_id = 256 + step
        merges2[pair] = new_id
        vocab2[new_id] = vocab2[pair[0]] + vocab2[pair[1]]
        words = Counter({tuple(merge(list(w), pair, new_id)): c for w, c in words.items()})
    return merges2, vocab2


merges, vocab = train_words(train, N_MERGES)                 # encode() and decode() now use these
first = [vocab[256 + i].decode() for i in range(8)]
print(f"\nword-split BPE, first merges: {first}")
toks = [t for w in SPLIT.findall(sample) for t in encode(w)]
print(f"{sample!r} -> {len(toks)} tokens:", " | ".join(repr(vocab[t].decode()) for t in toks))
held = [t for w in SPLIT.findall(test) for t in encode(w)]
print(f"unseen text: {len(held):,} tokens ({len(test.encode('utf-8')) / len(held):.2f} bytes per token)")
print("longest tokens learned:", [t.decode() for t in sorted(vocab.values(), key=len)[-5:]])

# 5. The real thing: GPT-2's tokenizer is the same algorithm with 50,000 merges on 40 GB of web text.
try:
    from transformers import AutoTokenizer
    gpt2 = AutoTokenizer.from_pretrained("openai-community/gpt2")
    g = gpt2(sample)["input_ids"]
    print(f"\nGPT-2 (vocabulary {gpt2.vocab_size:,}): {len(g)} tokens:",
          " | ".join(repr(gpt2.decode([t])) for t in g))
    print(f"GPT-2 on the unseen text: {len(gpt2(test, verbose=False)['input_ids']):,} tokens")
except ImportError:
    print("\n(pip install transformers to compare with GPT-2)")
