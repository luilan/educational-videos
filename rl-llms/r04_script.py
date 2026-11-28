"""Reinforcement Learning for LLMs, episode 4 — The KL Leash. One entry per narration section.

Every number comes from code/r04_kl_leash (kl_leash.py, 2026-11-28): REINFORCE with the per-question baseline from
episode 1's model, KL penalty beta in {0, 0.01, 0.03, 0.1, 0.3, 1}, 2 seeds, 2,000 steps at lr 0.0003; then beta 0
and 0.1 at lr 0.003.
"""
TITLE = "The KL Leash"
TAGLINE = "Chase the reward, but stay close to where you started"
NEXT = "PPO: Small, Safe Steps"
SECTIONS = [
    # 1. Hook
    "In episode two, a small learning rate took our model from fourteen percent to seventy-three. Make the steps ten "
    "times bigger, and within a hundred and fifty steps, the model answers one hundred and eleven to every question. "
    "One and a half percent right.",
    # 2. Why it can't recover
    "A few big updates pushed one answer up so hard that the model became certain of it. Now all eight tries at a "
    "question are the same answer, with the same reward. Every advantage is zero. There's nothing left to learn "
    "from, so it stays stuck forever.",
    # 3. The leash
    "The fix is a leash. Keep a frozen copy of the model from before training: the reference. For every sampled "
    "answer, measure how much more likely the model makes it than the reference did, as a difference of log "
    "probabilities, and subtract beta times that from the reward. Averaged over the model's answers, that "
    "difference is the K L divergence: how far the model has moved.",
    # 4. Rescue
    "Same big steps, now with beta at point one. The model wobbles, with strange answers for a while, then "
    "recovers: sixty-one percent right. Its K L stays near seven, instead of twenty-three.",
    # 5. Sweep
    "But a leash has a cost. At the normal learning rate, we tried six values of beta. Up to point zero three, "
    "accuracy stays near seventy percent. At point one, it drops to twenty-three, and at one, the model barely "
    "moves from fifteen. Too tight, and it can't learn. Too loose, and nothing protects it.",
    # 6. How the text reads
    "Look at the answers to forty-seven plus thirty-eight. The reference says seventy-four, seventy-five, eighty-one. "
    "Without a leash, eighty-five, every time. With the tightest leash, the same mix as before. Here a single right "
    "answer is what we want. For a chatbot, that variety is its style, its phrasing and its knowledge.",
    # 7. Why language models use it
    "That's why R L for language models always keeps a leash. The reward model is imperfect, and a model free to "
    "drift will find strange text that scores well. The reference is the model before R L. P P O puts the penalty "
    "in the reward, as we did. G R P O adds it to the loss.",
    # 8. Code
    "In code, it's two lines: the log-probability difference against the frozen reference, and beta times it "
    "subtracted from the reward, before the baseline.",
    # 9. Next
    "Next: P P O. Small, safe steps, so one batch can be used many times.",
]
