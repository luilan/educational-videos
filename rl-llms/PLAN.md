# Reinforcement Learning for LLMs

The seventh series. Deep Dive gave reinforcement learning five overview episodes (d36 SFT, d37 reward models, d38 PPO,
d39 DPO, d40 GRPO for reasoning). This series is the hands-on version: every algorithm is written from scratch, trained
on a language model, and measured. There are no gridworlds and no robots. Every idea is shown on the thing it is used
for today, a model that writes text and is graded on it. Same format as the other series: ~2–2.5 minute narrated Manim
episodes, a study-guide PDF, and runnable code for every number on screen. Ids `rNN`, code in `code/rNN_<topic>/`.

Everything runs on our CPU-only server (16 threads, 62 GB, no GPU). Two models carry the series:

- **tiny_gpt** (from How LLMs Work and Deep Dive), small enough to run thousands of RL updates in minutes. It is used
  for every algorithm episode, where we need many runs and clean curves.
- **Qwen2.5-0.5B-Instruct** (as in d40), a real instruction-tuned model, for the episodes where a real model's
  behaviour matters: reward hacking, verifiable rewards, reasoning length and agents. Heavy runs take a
  `systemd-run --user --scope -p MemoryMax=…` scope and run one at a time.

The running task is **arithmetic with a checker**: questions like "What is 37 × 2 + 48?" whose answers code can grade
exactly. It makes the reward honest and cheap, so the series can show what RL really changes.

Prerequisites: How LLMs Work (tokens, training, sampling) and Foundations (probability, logs, gradients). The Deep
Dive post-training episodes are optional background; we refer back to them rather than repeat them.

## Part 1 · Learning from a score
| Id | Title | Idea | Experiment |
|---|---|---|---|
| r01 | Learning From a Grade | Supervised training copies answers; RL only gets a score, so it can learn things nobody wrote down. The model is a policy, an answer is a sequence of actions, the reward arrives at the end | tiny_gpt pre-trained on arithmetic text: sample 64 answers per question, grade them, plot the reward spread. The good answers are already in there |
| r02 | REINFORCE | Make rewarded answers more likely: weight each answer's log-probability by its reward. Why it works (the log trick), in one picture | REINFORCE on tiny_gpt; reward curve over 500 updates; the answer distribution before and after |
| r03 | Baselines and Advantages | Rewards of 0 or 1 give noisy gradients; subtract the average to learn "better than usual". Variance, explained by coin flips | Same run with no baseline, a moving-average baseline and a per-question baseline: learning speed and gradient variance |
| r04 | The KL Leash | Without a leash the model drifts into strange text that still scores. A reference model and a KL penalty keep it close | Turn β up and down: reward vs KL vs how the text reads; a run that collapses to one repeated answer |
| r05 | PPO: Small, Safe Steps | Reuse each batch for several updates without breaking anything: the probability ratio and clipping; the value head as a learned baseline | PPO on tiny_gpt vs REINFORCE at equal compute; what happens without the clip |
| r06 | GRPO: Grading on a Curve | Drop the value model: compare each answer with the others in its group. Why this made RL on big models cheap (DeepSeek) | GRPO vs PPO on tiny_gpt: memory, speed, final reward; group size 2, 4, 8, 16 |

## Part 2 · What to reward
| Id | Title | Idea | Experiment |
|---|---|---|---|
| r07 | Rewards From Preferences | When there is no checker, people compare two answers; a reward model learns those comparisons (Bradley–Terry) | Train a small reward model on preference pairs (one style preferred over another); its accuracy on held-out pairs |
| r08 | Reward Hacking | Optimise hard against a learned reward and the model finds its blind spots: longer answers, flattery, format tricks. Goodhart's law | Qwen 0.5B with GRPO against the r07 reward model: proxy reward rises while true quality falls; the exact moment it turns |
| r09 | Verifiable Rewards | Grade with code instead of a model: exact answers, unit tests, format checks. Why labs moved to this for maths and code | Qwen 0.5B on multi-step arithmetic: verifiable reward vs reward model, same budget. Picks up where d40 stopped (d40 taught format; here we target accuracy) |
| r10 | Partial Credit: Process vs Outcome | Reward only the final answer, or each step? Credit assignment over a long answer | Outcome-only vs step-checked rewards on multi-step problems: which mistakes each one fixes |

## Part 3 · Reasoning and beyond
| Id | Title | Idea | Experiment |
|---|---|---|---|
| r11 | Learning to Think Longer | Under RL, answers grow longer and accuracy rises: test-time compute learned, not prompted. Length penalties | Answer length and accuracy over training; the same model with a token budget |
| r12 | Does RL Teach New Skills? | The pass@k debate: does RL find new solutions or only make the model pick the ones it already had more often? | pass@1 and pass@64 before and after RL on the same questions; problems no sample solved before |
| r13 | RL for Agents | Multi-turn episodes: the model calls tools, the reward arrives at the end of the whole episode. Credit across turns | A tiny calculator-tool environment from the Agents series' harness; GRPO on whole episodes; tool calls that become useful |
| r14 | Self-Play | The model writes its own problems and solves them; one side gets harder as the other improves. From AlphaZero to LLMs that make their own curriculum (series finale) | Proposer/solver on arithmetic: difficulty and solve rate over rounds; what stops it running away |

## Production notes
- Release order is strict (r01 first), interleaved with The Frontier (`frontier/PLAN.md`); the lab runs one heavy job
  at a time, so each series releases at about half the pace of Generative.
- Kokoro pronunciation: write "Kwen" for Qwen, "P P O"/"G R P O" if it stumbles, "K L" for KL.
- Every number on screen comes from a seeded run whose log is committed next to the code.
