# The Frontier: Self-Improvement and Memory

The eighth series, made alongside Reinforcement Learning for LLMs. It follows research that is weeks old, not
textbook material, so every episode names its papers with dates and says clearly when a result is a single preprint
that nobody has replicated yet. Same format as the other series: ~2–2.5 minute narrated Manim episodes, a study-guide
PDF, and runnable code for every number on screen. Ids `xNN` (f is taken by Foundations), code in `code/xNN_<topic>/`.

Two arcs:

1. **AI that improves itself.** A trained model is frozen; improving it means another training run, or a prompt or
   harness that is gone when the task ends. In November 2026 a burst of papers showed a third way: a language model
   reads the weights of a network, edits them like code and makes it better, with no gradients ("weight-space
   programming", WSP), and then does the same to *itself*. We rebuild these experiments at a size our CPU can run and
   test them as sceptically as the best of the papers do.
2. **Memory, as one project we keep building.** A model forgets everything between sessions. We take one agent and
   upgrade its memory episode by episode (retrieval, notes it manages itself, a memory that learns while it runs,
   memory in the weights, a memory design found by another model), and score every version on the same benchmark.
   The question throughout is whether memory changes what the agent *does next*, not only what it can recall.

Everything runs on our CPU-only server (16 threads, 62 GB, no GPU). Models: tiny_gpt (our own), **Qwen3.5-0.8B** (the
model the Self-WSP paper used, small enough to read and edit itself on a CPU), qwen3:4b in the Agents-series lab
(Ollama on `hx-net`, one job at a time), and, where a paper's "frontier editor" is a large model, a frontier model run
through its CLI, shown as such on screen. The 27B result of the self-rewriting paper is too heavy to rerun here; we
reproduce the method on a small model and say so.

Prerequisites: How LLMs Work; Agents (the harness) for Part 3; LLMs in Practice (RAG, embeddings) for Part 4.

## Key papers
- Language Models Can Rewrite Their Own Weights (arXiv 2611.21885, 16 Nov 2026): a 27B model edits its own weights by
  reasoning; +6.6 points on unseen problems; keeps improving over 8 sessions; repairs a damaged copy of itself.
- Self-WSP: Can Language Models Improve Themselves Through Weight-Space Programming? (arXiv 2611.28833, 21 Nov 2026):
  Qwen3.5-0.8B edits itself; GSM8K 11.8% → 62.2%, sometimes changing 16 parameters; little understanding of its own
  edits; random-search WSP sometimes comparable.
- The wider WSP burst of Nov 2026 (frontier models repairing, auditing and improving small networks), including the
  sceptical "Can an LLM actually reason about weights or is it memorisation?" (2611.08070) and "Do we really need an
  LLM for WSP?" (2611.34612). The original WSP report is to be pinned down when x02 is written.
- SEAL, Self-Adapting Language Models (2506.10943, Jun 2025); Darwin Gödel Machine (2505.22954, May 2025); SIA,
  harness and weight updates (2605.27276, May 2026).
- MemGPT (2310.08560); Titans: Learning to Memorize at Test Time (2501.00663); Continual Learning via Sparse Memory
  Finetuning (2510.15103); ALMA, meta-learning agentic memory designs (2602.07755, Feb 2026); Memory-Centric
  Continual Learning for AI Agents (2610.29357, Oct 2026).

## Part 1 · Weights as code
| Id | Title | Idea | Experiment |
|---|---|---|---|
| x01 | The Frozen Model | Models are trained once and frozen; the three usual ways to improve one, and the new fourth: the model changes itself. A network's weights are just a file of numbers | Open tiny_gpt's weight file; change one number by hand and watch an answer change; how many numbers matter |
| x02 | Weight-Space Programming | A large model reads a small network's weights, finds what is wrong and writes an edit, no gradients. The loop: inspect, propose, test, keep | A frontier model repairs a small network we broke on purpose (a flipped sign, a dead neuron); how many parameters it touches |
| x03 | The Random Baseline | How much of the gain is the loop itself? The same propose-test-keep loop with random edits | Random search vs the model editor at equal trials; when random search wins, and when it cannot |
| x04 | A Model That Edits Itself | Self-WSP: the editor and the edited are the same model; after each edit it is a different editor | Qwen3.5-0.8B reads and edits its own weights on arithmetic; accuracy over rounds on held-out problems |

## Part 2 · Self-rewriting
| Id | Title | Idea | Experiment |
|---|---|---|---|
| x05 | Diagnosing Its Own Mistakes | Self-rewriting (2611.21885): graded answers → a diagnosis → the weights behind the failure → an edit. Practice set vs unseen set | Our small-scale version: failures it diagnoses, edits it submits, gain on unseen problems |
| x06 | Repairing a Broken Copy | A damaged copy that solves almost nothing fixes itself without retraining or seeing the original | Damage the model (noise in one layer, a zeroed block); self-repair rounds; how much comes back |
| x07 | Does It Understand Its Edits? | The sceptical half: edits that help for the wrong reasons, overfitting the practice set, gains that random search matches. How to tell real improvement from luck | Ablate each accepted edit; held-out vs practice gains; the edits the model explains vs what they actually do |

## Part 3 · Improving the harness
| Id | Title | Idea | Experiment |
|---|---|---|---|
| x08 | Rewriting Its Own Harness | The other lever: keep the weights, let the agent rewrite its own prompts, tools and retry logic (Darwin Gödel Machine, SIA). Harness plus weights together | The Agents-series harness, rewritten by the model over rounds and scored on a fixed task set; what it changes first |
| x09 | Can It Keep Going? | Recursive self-improvement: plateaus, the cost of each round, gaming the score, and why a held-out test and a human sign-off matter (end of arc 1) | Long runs from x04 and x08: where gains stop, what the failures look like |

## Part 4 · Memory (one agent, upgraded each episode)
| Id | Title | Idea | Experiment |
|---|---|---|---|
| x10 | The Agent That Forgets | Each session starts blank; the context window is the only memory. The benchmark for the arc: a stream of tasks where lessons from earlier tasks help later ones, with interference and a mid-stream change of rules (MCCL's retention, adaptation, transfer) | Our agent on the task stream with no memory: the baseline scores |
| x11 | Memory by Retrieval | Store past episodes, embed them, retrieve the closest. What helps, and how stale or wrong memories hurt | Retrieval memory on the stream: retention up, adaptation down when the rules change |
| x12 | Memory the Agent Manages | MemGPT-style: the agent writes, edits and deletes its own notes through tools; memory as a tool, not a cache | Agent-managed notes vs plain retrieval on the same stream |
| x13 | A Memory That Learns While It Runs | Titans-style neural memory: a small network inside the model, updated at test time by how surprising each input is | A neural memory module on tiny_gpt: recall over long streams vs a fixed context window |
| x14 | Memory in the Weights | Keep fine-tuning after deployment: catastrophic forgetting, and sparse memory fine-tuning that updates only the memory slots a new fact uses | Teach new facts by full fine-tuning vs sparse updates; how much old knowledge each destroys |
| x15 | A Model That Designs Its Own Memory | ALMA: a meta-agent writes memory designs as code, tests them and keeps the best | Meta-agent search over memory designs for our agent; the design it finds vs ours |
| x16 | Does Memory Make It Better? | Score every version of the arc on the same benchmark; memory judged by future actions, not recall. Where self-improvement and memory meet: continual learning (series finale) | The scoreboard: x10–x15 on retention, adaptation and transfer |

## Production notes
- Release order is strict (x01 first), interleaved with Reinforcement Learning for LLMs (`rl-llms/PLAN.md`).
- Every paper claim on screen carries the paper's name and date; preprint results are labelled as such.
- Model-in-the-loop experiments are seeded where possible, and every accepted edit is logged with a before/after score
  so the scepticism episodes (x03, x07) can be built from the same logs.
- Kokoro pronunciation: "Kwen", "W S P" (or "weight-space programming"), "Mem G P T", "Titans", "Alma".
