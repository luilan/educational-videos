# Deep Dive, episode 31 · The KV Cache, Deeper

GPT-2 small, generating with and without the KV cache; the cache's contents; prefill vs decode.

```bash
pip install -r requirements.txt
python kv_cache.py
```

Downloads GPT-2 small (about 500 MB). On an 8-thread CPU it printed (times depend on the machine):

1. 200 new tokens, greedy: identical output with and without the cache; 7.43 s vs 18.03 s;
2. the cache: per layer, keys and values of shape (1, 12, tokens, 64); 73,728 bytes per token (float32); 72 MiB at
   1,024 tokens;
3. time per new token at 7 / 56 / 106 / 206 tokens: with the cache 36.6 / 35.8 / 36.3 / 36.6 ms, without 42.9 / 57.5 /
   88.2 / 138.3 ms;
4. a 512-token prompt: 0.29 s in one pass (prefill), 18.76 s one token at a time; prefill 1,738 tokens/s, decoding
   27 tokens/s.
