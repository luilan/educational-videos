# Deep Dive, episode 37 · Reward Models

A reward model on a frozen Qwen2.5-0.5B: the layer-12 hidden state of the conversation's last token goes through one
linear layer to give the reward; trained on preference pairs with the Bradley–Terry loss −log σ(r_chosen − r_rejected).

```bash
pip install -r requirements.txt
python reward_model.py
```

Results (deterministic, CPU):

1. Capitals, 90 pairs from 30 countries: right answer preferred 100% on training pairs, 98% on 60 pairs about 20 unseen
   countries.
2. Sums, 300 pairs: 79% on training pairs, 55% on 40 unseen sums (the frozen features don't encode arithmetic).
3. Shortcut, preferred answers always ending with "I hope this helps!": 75% right on unseen countries without the
   phrase, 0% when the phrase is on the wrong answer (Budapest −3.46 vs "Vienna. I hope this helps!" +4.11). Trained with
   the phrase on both answers or neither: 98% and 82%.
