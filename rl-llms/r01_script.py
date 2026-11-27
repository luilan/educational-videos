"""Reinforcement Learning for LLMs, episode 1 — Learning From a Grade. One entry per narration section.

Every number comes from code/r01_learning_from_a_grade (grade.py, 2026-11-27): a 102,413-parameter tiny GPT pre-trained
on 500 worked additions, then 64 graded samples for each of 1,000 new questions.
"""
TITLE = "Learning From a Grade"
TAGLINE = "Reinforcement learning needs a score, not an answer key"
NEXT = "REINFORCE"
SECTIONS = [
    # 1. Hook
    "There are two ways to teach a language model. Show it the right answer and make it copy. Or let it try, and only "
    "tell it how well it did. The second way is reinforcement learning, and this series builds it from scratch, the "
    "way language models use it today.",
    # 2. Why a grade
    "Why learn from a grade? Because checking is often easier than answering. A program can test code it could never "
    "write. A grader can say right or wrong without showing the work. And a grade can teach things that nobody ever "
    "wrote down.",
    # 3. Task and checker
    "Our task for the whole series: adding two numbers of two digits, one character at a time, ending with a period. "
    "The checker gives a reward of one if the answer is exactly right, and zero otherwise. Our model is a tiny GPT "
    "with about a hundred thousand parameters.",
    # 4. Pretraining
    "First we pre-train it on just five hundred worked examples. It learns all five hundred by heart. But on a "
    "thousand questions it has never seen, it gets only fourteen percent right.",
    # 5. The policy
    "In reinforcement learning, the model is called a policy. Each character it writes is an action, a whole answer "
    "is an episode, and the reward only comes at the end. For forty-seven plus thirty-eight, the policy puts "
    "eighty-three percent on a seven first. It usually forgets to carry the one.",
    # 6. Sample and grade
    "Now let it try. Sixty-four answers to twenty-seven plus fifteen: forty-eight, thirty-three times. The right "
    "answer, forty-two, twenty-four times. Forty-nine, seven times. Each one gets its grade: one or zero.",
    # 7. The spread
    "Do this for all thousand questions. Overall, fourteen percent of the answers are right. Six hundred and "
    "forty-four questions are never right in sixty-four tries. But on two hundred and seventeen, its first choice is "
    "wrong while some of its tries are right.",
    # 8. The idea
    "That is what reinforcement learning works with. The good answers are sometimes already in the samples. Make the "
    "answers that scored one more likely and the others less likely, and repeat. No answer key, only grades.",
    # 9. Code
    "In code, the whole grade is one line: compare the answer with the right sum. Sampling writes up to four "
    "characters, one at a time.",
    # 10. Next
    "Next: REINFORCE. The simplest way to turn grades into learning.",
]
