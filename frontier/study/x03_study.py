"""Study guide content for The Frontier: Self-Improvement and Memory, episode 3: The Random Baseline.

Build:  python framework/study_guide.py frontier x03
Every number comes from code/x03_the_random_baseline (random_search.py, 2026-11-28); Claude's numbers are episode 2's.
"""

LESSON = {
    "series": "The Frontier: Self-Improvement and Memory",
    "label": "Episode 3",
    "title": "The Random Baseline",
    "tagline": "Is it the reasoning, or just the loop?",
    "duration": "2:23",
    "intro": """<p>In episode 2 Claude repaired our damaged addition model by reading its weights. But the harness
only kept edits that raised the practice score, so part of the success could belong to the loop rather than the
reasoning. This lesson replaces Claude with chance: random edits in the same edit language, and blind random nudges,
under the same keep-if-better rule. Random search fixes small damage if it gets thousands of tries, at the price of a
messier model that does worse on unseen questions. It never finds the structural fixes Claude found at once.</p>""",
    "prereq": """<div class="box note"><b>Before you start:</b> episodes 1 and 2 of this series (the model, the damage
cases and the repair loop); Foundations on probability (multiplying independent chances). Code:
<code>code/x03_the_random_baseline</code> (Python with PyTorch, no language model needed; about 13 minutes on 16
threads).</div>""",
}

CONCEPTS = [
    {
        "title": "A random editor in the same loop",
        "segment": (7.8, 41.8),
        "figures": [{"t": 21.2, "caption": "The loop from episode 2: propose an edit, test it on the practice "
                                           "questions, keep it only if the score goes up. Here the proposer is a die."},
                    {"t": 41.4, "caption": "Random editor 1 uses Claude's four edits with every choice random; random "
                                           "editor 2 nudges every weight by a little noise. Up to 10,000 tries."}],
        "body": [
            """<p>Episode 2's loop has two parts: an <b>editor</b> that proposes changes to the weights, and a
<b>harness</b> that tests each change on 300 practice questions and keeps it only if the score rises. The harness is
itself a search algorithm (hill climbing). So when the loop succeeds, how much credit goes to the editor's reasoning?
The way to find out is a <b>baseline</b>: keep the harness and replace the editor with chance.</p>""",
            "{fig0}",
            """<p>We used two random editors:</p>
<ul><li><b>Random edits</b>: exactly Claude's edit language (set one number, scale a row, swap two rows, replace a
row) with every choice random: which of the 30 weight tensors, which row or entry, the new value (drawn from that
tensor's own mean and spread), the factor (between −2 and 2).</li>
<li><b>Random nudges</b>: add a little Gaussian noise to every one of the 102,413 weights at once (1% of each
tensor's spread).</li></ul>
<p>Same rule as Claude: keep an edit only if the practice score goes up, stop at 99%. Up to 10,000 tries; 3 runs
(seeds) for random edits, 1 for nudges. The 1,000 test questions are never used to choose edits.</p>""",
            "{fig1}",
            """<div class="box key"><b class="t">Key idea</b>To credit an editor, compare it with a random editor inside
the same keep-if-better loop.</div>""",
        ],
        "exercises": [
            {"kind": "short", "q": "Why must the random editor use the same edit language and the same keep rule as "
                                   "Claude?",
             "answer": "So that the only difference is who chooses the edits; anything else would mix up the effect of "
                       "the loop, the edit types and the reasoning.",
             "why": "A baseline changes one thing at a time."},
            {"kind": "tf", "q": "“The harness that keeps an edit only if the score improves is itself a search "
                                "algorithm.”",
             "answer": "True.", "why": "It is hill climbing: any proposer, even a random one, can make progress through "
                                       "it.",
             "key": {"value": True}},
        ],
    },
    {
        "title": "Equal tries, and many tries",
        "segment": (42.3, 86.6),
        "figures": [{"t": 53.5, "caption": "With 4 tries, random edits leave all three damaged models where they were; "
                                           "each successful Claude repair took one try."},
                    {"t": 73.9, "caption": "One number zeroed: random edits reach 99% on practice after 774 and 1,567 "
                                           "tries (2 of 3 runs), changing 331 and 1,803 numbers; Claude changed 1."},
                    {"t": 86.2, "caption": "Practice vs hidden test: Claude 99.7% / 99.1%; random runs 99.0% / 98.6% "
                                           "and 99.0% / 96.8%; nudges 98.7% / 92.5%."}],
        "body": [
            """<p><b>Equal tries.</b> Each of Claude's successful repairs took a single proposal (in one of three
sign-flip runs it failed after four). With four tries, random edits changed nothing in any of the three cases: test
accuracy stayed at 77.7%, 39.3% and 58.7%.</p>""",
            "{fig0}",
            """<p><b>Many tries.</b> Random tries are cheap: about 45 ms of CPU each, against 8–34 seconds and about
$0.27 for a Claude round. For the easiest damage (one LayerNorm gain set to 0), random edits reached the 99% practice
target in 2 of 3 runs, after <b>774</b> and <b>1,567</b> tries; the third run was at 95% after 10,000. Nudges reached
98.7%. But the repairs look nothing alike. Claude set the one zeroed number back to 0.79 (it was 0.81). The random
runs piled up dozens of unrelated edits and changed <b>331</b> and <b>1,803</b> numbers; nudges changed all
102,413.</p>""",
            "{fig1}",
            """<p><b>Overfitting.</b> Each random edit was kept because it helped on the same 300 practice questions,
so together they fit those questions, not addition. On the 1,000 hidden questions the two solved random runs scored
98.6% and 96.8%, and the nudges 92.5%, though all were at 99% or nearly on practice. Claude's single edit scored 99.1%,
almost exactly the intact model's 99.2%.</p>""",
            "{fig2}",
            """<div class="box key"><b class="t">Key idea</b>With enough cheap tries, blind search can repair small
damage, but with many more changes and a bigger gap between practice and unseen questions.</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "A random try costs 45 ms. How many seconds did the 1,567 tries of the slower "
                                    "solved run take (nearest second)?",
             "answer": "About 71 s.", "why": "1,567 × 0.045 s ≈ 70.5 s (the log says 75.7 s including test checks).",
             "key": {"parts": [{"label": None, "value": 71, "tol": 5, "unit": "s"}]}},
            {"kind": "mc", "q": "A repair scores 99.0% on the practice questions and 96.8% on the hidden test. What "
                                "does the gap suggest?",
             "options": ["The test is broken", "The edits fit the practice questions more than the task",
                         "The model is now better than the original", "Random search always fails"],
             "answer": "B.", "why": "The edits were chosen using the practice questions only; the hidden test measures "
                                    "the task itself.",
             "key": {"choice": 1}},
        ],
    },
    {
        "title": "Where random search cannot follow",
        "segment": (87.1, 106.5),
        "figures": [{"t": 106.1, "caption": "Swapped digits and flipped sign: after 10,000 tries random edits reach "
                                            "60–93% and 66–69% on the test with piles of patches; nudges never keep an "
                                            "edit."}],
        "body": [
            """<p>For the two structural damages, random edits ran the full 10,000 tries in all three runs. The swapped
digits ended at 60.2%, 67.4% and 92.7% on the test, the flipped sign at 68.6%, 65.9% and 68.2%. None found the real
fix; each run stacked up 14–29 small patches (for example scaling the embedding of 8 by −1.57). Random nudges never
kept a single edit: no small noise improves a model whose 3 and 8 are exchanged.</p>
<p>Why so hard? The exact fix for the swap needs the right tensor (1 of 30), the swap operation (1 of 4) and the
right pair of rows (2/13 × 1/12): about <b>1 chance in 9,400</b> per try. The sign flip needs row 7 of the embedding
and a factor close to −1, which is rarer still. Claude read the wrong answers (“every 3 is read as 8”) and went
straight to the edit.</p>""",
            "{fig0}",
            """<div class="box key"><b class="t">Key idea</b>When the fix is one specific edit in a huge space, chance
almost never finds it; understanding the failure does.</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "With a 1 in 9,400 chance per try, how many tries are expected before the exact swap "
                                    "appears (nearest hundred)?",
             "answer": "About 9,400.", "why": "The expected wait for an event with probability p is 1/p.",
             "key": {"parts": [{"label": None, "value": 9400, "tol": 100, "unit": None}]}},
            {"kind": "tf", "q": "“Random nudges failed on the swap because 10,000 tries were too few.”",
             "answer": "False.", "why": "No nudge was ever kept: small noise cannot move toward exchanging two rows, so "
                                        "more tries would not help.",
             "key": {"value": False}},
        ],
    },
    {
        "title": "What this means for self-improvement claims",
        "segment": (107.0, 134.2),
        "figures": [{"t": 125.9, "caption": "The loop alone fixes small damage at a price; structural damage needs the "
                                            "right edit. Random search is a strong rival (Neural Thickets, arXiv "
                                            "2603.12228)."},
                    {"t": 133.8, "caption": "The random editor: pick a tensor, an operation and its numbers, then hand "
                                            "the edit to the same loop."}],
        "body": [
            """<p>The loop alone can fix small damage, if you can afford thousands of tries and accept a messier model.
Structural damage needs the right edit in a huge space, and that is where reading the weights pays: one proposal
against 10,000 failed tries.</p>
<p>Random search is not a straw man. The Neural Thickets paper (Gan &amp; Isola, arXiv 2603.12228, March 2026)
argues that near pretrained weights, random perturbations plus selection are a serious rival to gradient methods.
So any claim that a model improves a network, or itself, by editing weights should be compared with random edits
under the same loop and the same budget, and judged on held-out questions. The next episodes keep this
baseline.</p>""",
            '<div class="figrow">{fig0}{fig1}</div>',
            """<div class="box key"><b class="t">Key idea</b>A weight-editing result counts only if it beats random
search under the same loop and budget, on questions the search never saw.</div>""",
        ],
        "exercises": [
            {"kind": "short", "q": "A paper reports that a model improved itself by +5 points through weight edits. "
                                   "Name two checks this episode suggests.",
             "answer": "Run random edits under the same loop and budget, and measure on held-out questions not used "
                       "to accept edits.",
             "why": "The loop and overfitting to the selection set can both produce gains without understanding."},
            {"kind": "code", "q": "<b>Try it yourself.</b> The random editor.",
             "code": """def random_edit(rng, weights):
    name = rng.choice(tensor_names)              # which weights
    op = rng.choice(["set", "scale_row", "swap_rows", "set_row"])
    edit = random_numbers_for(op, weights[name], rng)
    return edit      # same loop: apply, test, keep only if better""",
             "answer": "random_search.py runs both random editors on the three damages; try a budget of 50,000 for the "
                       "swap.",
             "why": "A few lines give the baseline every weight-editing result should beat."},
        ],
    },
]
