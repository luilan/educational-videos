"""How LLMs Work: Deep Dive, episode 32 — Batching and Paged Attention. One entry per narration section."""
TITLE = "Batching and Paged Attention"
TAGLINE = "Serving many users from one model"
NEXT = "Quantization, Deeper"
SECTIONS = [
    # 1. Hook
    "A model answering one user at a time wastes most of its hardware. Serving systems answer many users at once. "
    "Three ideas make that work: batching, continuous batching, and paged attention.",
    # 2. Batching
    "Batching first. GPT-2 writes sixty-four tokens for one request: twenty-six tokens per second on this CPU. Four "
    "requests at once: a hundred and fifty-three. Sixteen: over four hundred. Fifteen times the throughput.",
    # 3. Why
    "Why? Decoding one token reads all the weights for very little math. With a batch, the same weights serve many "
    "requests in one pass. On this CPU, a batch of one is so inefficient that four requests even finish sooner than "
    "one.",
    # 4. Padding
    "But real requests have different lengths. Take sixteen requests wanting between twenty and a thousand tokens. A "
    "fixed batch runs until the longest finishes, and sixty-five percent of the work is wasted on requests that are "
    "already done.",
    # 5. Continuous
    "Continuous batching fixes that: as soon as one request finishes, it leaves, and a waiting request joins at the "
    "next step. The batch stays full of useful work.",
    # 6. Memory
    "The other limit is memory: every request needs its K V cache. The simple approach reserves room for the maximum "
    "length, two thousand and forty-eight tokens, for every request. For a hundred requests of mixed lengths, that's "
    "seven gigabytes, and only forty percent of it ever gets used.",
    # 7. Paging
    "Paged attention borrows an idea from operating systems: store the cache in small pages of sixteen tokens, "
    "allocated only as a request grows. Now the same hundred requests need two point eight gigabytes, ninety-nine "
    "percent used. The same memory holds two and a half times as many requests.",
    # 8. Code
    "In code, paging is bookkeeping: round each request up to whole pages, and keep a table of where its pages live.",
    # 9. Outro
    "That's serving at scale. Next: quantization, a deeper look at squeezing weights into fewer bits.",
]
