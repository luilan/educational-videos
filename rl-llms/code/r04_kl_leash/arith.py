"""The task used through Reinforcement Learning for LLMs: two-digit addition, with a checker.

A question is "47+38=" and a full answer is "85." (the period ends the answer). Every character is a token.
The checker is the reward: 1 if the answer is exactly right, else 0. Copy this file into each episode's code folder.
"""
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

CHARS = "0123456789+=."
STOI = {c: i for i, c in enumerate(CHARS)}
VOCAB = len(CHARS)
END = STOI["."]
Q_LEN, MAX_ANS = 6, 4                     # "47+38=" ; at most "185." (3 digits + end)
BLOCK = Q_LEN + MAX_ANS


def encode(s):
    return [STOI[c] for c in s]


def decode(ids):
    return "".join(CHARS[i] for i in ids)


def all_questions():
    return [(a, b) for a in range(10, 100) for b in range(10, 100)]   # 8,100 questions


def split(seed=0, n_test=1000):
    qs = all_questions()
    random.Random(seed).shuffle(qs)
    return qs[n_test:], qs[:n_test]                  # train, test


def prompt(a, b):
    return f"{a}+{b}="


def reward(a, b, answer):
    """The grade: 1.0 if the generated answer text is exactly a+b followed by '.', else 0.0."""
    return 1.0 if answer == f"{a + b}." else 0.0


# ----------------------------------------------------------------------------- the model (How LLMs Work, ep. 11)
class Block(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        self.register_buffer("mask", torch.triu(torch.ones(BLOCK, BLOCK, dtype=torch.bool), 1))

    def forward(self, x):
        T = x.shape[1]
        h = self.ln1(x)
        x = x + self.attn(h, h, h, attn_mask=self.mask[:T, :T], need_weights=False)[0]
        return x + self.mlp(self.ln2(x))


class TinyGPT(nn.Module):
    def __init__(self, d=64, heads=4, layers=2):
        super().__init__()
        self.tok = nn.Embedding(VOCAB, d)
        self.pos = nn.Embedding(BLOCK, d)
        self.blocks = nn.Sequential(*[Block(d, heads) for _ in range(layers)])
        self.ln = nn.LayerNorm(d)
        self.head = nn.Linear(d, VOCAB)

    def forward(self, idx):
        x = self.tok(idx) + self.pos(torch.arange(idx.shape[1]))
        return self.head(self.ln(self.blocks(x)))


@torch.no_grad()
def sample(model, questions, n=1, temperature=1.0, greedy=False, generator=None):
    """Answer each question n times. Returns (token ids of prompt+answer [B, BLOCK], answer strings, answer lengths)."""
    qs = [q for q in questions for _ in range(n)]
    idx = torch.tensor([encode(prompt(a, b)) for a, b in qs])
    done = torch.zeros(len(qs), dtype=torch.bool)
    lengths = torch.full((len(qs),), MAX_ANS)
    for t in range(MAX_ANS):
        logits = model(idx)[:, -1] / (temperature if not greedy else 1.0)
        nxt = logits.argmax(-1) if greedy else torch.multinomial(F.softmax(logits, -1), 1, generator=generator)[:, 0]
        nxt = torch.where(done, torch.full_like(nxt, END), nxt)       # pad finished answers with '.'
        idx = torch.cat([idx, nxt[:, None]], 1)
        newly = (nxt == END) & ~done
        lengths[newly] = t + 1
        done |= nxt == END
    answers = []
    for row, L in zip(idx[:, Q_LEN:].tolist(), lengths.tolist()):
        answers.append(decode(row[:L]))
    return idx, answers, lengths, qs


def answer_logprob(model, idx, lengths):
    """Sum of log-probabilities of each answer's tokens (prompt excluded, padding after '.' excluded)."""
    logits = model(idx[:, :-1])
    lp = F.log_softmax(logits, -1).gather(-1, idx[:, 1:, None])[..., 0]          # [B, BLOCK-1]
    pos = torch.arange(BLOCK - 1)[None]
    mask = (pos >= Q_LEN - 1) & (pos < Q_LEN - 1 + lengths[:, None])
    return (lp * mask).sum(-1)


def accuracy(model, questions, greedy=True):
    _, answers, _, qs = sample(model, questions, greedy=greedy)
    return sum(reward(a, b, ans) for (a, b), ans in zip(qs, answers)) / len(qs)
