"""Write recipes.txt: a small corpus in a new style for the tiny GPT (made-up recipes, Shakespeare's characters only)."""
import random

rng = random.Random(9)
dishes = ["soup", "pie", "stew", "bread", "cake", "salad", "pasta", "tart", "broth", "porridge"]
mains = ["onion", "leek", "apple", "pear", "carrot", "bean", "mushroom", "lemon", "honey", "cheese", "barley", "plum"]
amounts = ["one cup of", "two cups of", "half a cup of", "a spoon of", "a pinch of", "three handfuls of"]
extras = ["salt", "butter", "flour", "milk", "water", "sugar", "oil", "pepper", "cream", "eggs"]
verbs = ["Chop", "Stir", "Boil", "Bake", "Mix", "Fry", "Simmer", "Whisk", "Pour", "Roast"]
tails = ["until golden.", "for ten minutes.", "until soft.", "gently.", "in a warm pan.", "until thick.",
         "and let it rest.", "over a low fire."]
out = []
for _ in range(900):
    a, b = rng.sample(mains, 2)
    title = f"RECIPE: {a.upper()} AND {b.upper()} {rng.choice(dishes).upper()}"
    ingredients = [f"- {rng.choice(amounts)} {x}" for x in [a, b] + rng.sample(extras, 2)]
    steps = [f"{i}, {rng.choice(verbs).lower()} the {x} {rng.choice(tails)}" for i, x in
             zip(["First", "Then", "Last"], [a, b, rng.choice(extras)])]
    out.append("\n".join([title, "Ingredients:"] + ingredients + ["Steps:"] + steps) + "\n")
open("recipes.txt", "w").write("\n".join(out))
print(len("\n".join(out)), "characters")
