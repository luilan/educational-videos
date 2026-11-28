"""Two-step word problems with exact answers, and a checker. Shared by episodes 7 and 8."""
import random
import re

NAMES = ["Tom", "Ana", "Lena", "Omar", "Mia", "Raj", "Sofia", "Ben", "Yuki", "Leo"]
TEMPLATES = [
    ("{n} has {a} apples and buys {b} bags with {c} apples each. How many apples does {n} have now?", lambda a, b, c: a + b * c),
    ("{n} had {a} euros, spent {b} euros on lunch and then earned {c} euros. How many euros does {n} have?", lambda a, b, c: a - b + c),
    ("A shelf holds {a} books. {n} adds {b} boxes of {c} books. How many books are on the shelf?", lambda a, b, c: a + b * c),
    ("{n} reads {a} pages a day for {b} days, then {c} more pages. How many pages did {n} read?", lambda a, b, c: a * b + c),
    ("A bus has {a} people. At a stop {b} get off and {c} get on. How many people are on the bus?", lambda a, b, c: a - b + c),
]


def problems(n, seed):
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        t, f = rng.choice(TEMPLATES)
        a, b, c = rng.randint(12, 60), rng.randint(2, 9), rng.randint(3, 12)
        out.append((t.format(n=rng.choice(NAMES), a=a, b=b, c=c), f(a, b, c)))
    return out


def final_number(text):
    """The answer a reader would take: the last number in the text."""
    nums = re.findall(r"-?\d[\d,]*(?:\.\d+)?", text)
    if not nums:
        return None
    try:
        return float(nums[-1].replace(",", ""))
    except ValueError:
        return None


def correct(text, answer):
    v = final_number(text)
    return v is not None and abs(v - answer) < 1e-6
