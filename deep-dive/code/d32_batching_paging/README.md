# Deep Dive, episode 32 · Batching and Paged Attention

Batched generation with GPT-2, padding waste in fixed batches, and contiguous vs paged KV-cache allocation.

```bash
pip install -r requirements.txt
python batching_paging.py
```

Downloads GPT-2 small (about 500 MB). On an 8-thread CPU (two runs; times vary):

1. 64 new tokens per request: 1 request 26–28 tokens/s; 4 at once 148–153; 16 at once 404–433 (about 15x);
2. 16 requests of 20 to 1,000 tokens in one fixed batch: 16,000 token slots, 5,570 useful (65% wasted);
3. 100 requests of 50 to 1,500 tokens (average 810), GPT-2 float16 cache: reserving 2,048 tokens each takes 7.03 GiB
   (40% used); pages of 16 tokens take 2.81 GiB (99.0% used): 2.5x more requests in the same memory.
