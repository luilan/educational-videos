# Deep Dive, episode 43 · Superposition

The toy model of "Toy Models of Superposition" (Elhage et al., 2022): x' = ReLU(Wᵀ W x + b), with n sparse features
squeezed into m < n dimensions.

```bash
pip install -r requirements.txt
python superposition.py
```

Results (seeded; a feature is "represented" when its direction has norm > 0.5):

1. 5 features → 2 dimensions, importance 0.8^i: 2 features at sparsity 0, 4 at 0.7 and 0.9, all 5 at 0.97 (angles
   −149°, −81°, −8°, 67°, 140°: a pentagon). Saves `toy_5x2.json`.
2. 100 equally important features → 20 dimensions: 14, 40, 58, 95, 100, 100 represented at sparsity 0, 0.5, 0.8, 0.9,
   0.95, 0.99. At 0.5, features come in exactly opposite pairs (signed cosine −1.00).
