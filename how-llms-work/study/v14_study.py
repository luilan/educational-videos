"""Study guide content for How LLMs Work, episode 14 (bonus, series finale): From GPT to Chatbot.

Build:  python framework/study_guide.py how-llms-work v14
Figures: {"t": seconds into the rendered video, "caption": ...}; placed in body blocks with {fig0}, {fig1}, ...
Exercise kinds: mc (options), tf, short, number, order, code. Every exercise has an answer and a why.
"""

LESSON = {
    "series": "How LLMs Work",
    "label": "Episode 14",
    "title": "From GPT to Chatbot",
    "tagline": "How a text predictor learns to be an assistant",
    "duration": "2:18",
    "intro": """<p>This finale answers one question: how does a next-token predictor become a helpful assistant? It is
trained in stages: <b>pretraining</b> on internet text, <b>supervised fine-tuning</b> on example conversations, then
learning from <b>preferences</b> and other feedback. The machine never changes; only the <b>data</b> and the
<b>training signal</b> do.</p>""",
    "prereq": """<div class="box note"><b>Before you start:</b> this is the series finale, so it builds on the whole
series: the generation loop (episode 1), the parts of the transformer (episodes 2–10) and training with the next-token
loss (episode 11). Foundations F06 (Probability and Sampling) helps with the idea of a model's output as
probabilities.</div>""",
}

CONCEPTS = [
    {
        "title": "Pretraining makes a base model",
        "segment": (8, 31),
        "figures": [{"t": 22.3, "caption": "A base model is an autocomplete: asked a question, it may answer, or carry "
                                           "on with more questions, like a quiz (illustrative)."},
                    {"t": 30.9, "caption": "Stage 1, pretraining: trillions of tokens of text, almost all the compute, "
                                           "and the model's knowledge."}],
        "body": [
            """<p>A model trained only to predict the next token of internet text is called a <b>base model</b>. It is
a powerful <b>autocomplete</b>. Ask it a question and it might answer. Or it might continue with three more questions,
as if it were writing a quiz: to a base model, your question is just text to continue.</p>""",
            '<div class="figrow">{fig0}{fig1}</div>',
            """<p>This first stage is called <b>pretraining</b>. It is where almost all the computing goes, and where
the <b>knowledge</b> comes from: <b>trillions of tokens</b> of text.</p>""",
            """<div class="box key"><b class="t">Key idea</b>Pretraining gives a <b>base model</b>: lots of knowledge,
but its only habit is to continue text. It is not yet a helpful assistant.</div>""",
        ],
        "exercises": [
            {"kind": "mc", "q": "You type <i>“What is the capital of France?”</i> into a base model. What can happen?",
             "options": ["It always answers “Paris.”", "It might answer, or it might continue with more questions, "
                         "like a quiz", "It gives an error: base models cannot read questions",
                         "It refuses, because it was never trained on questions"],
             "answer": "B.", "why": "A base model only continues text, and a list of questions is a perfectly "
                                   "plausible continuation.", "key": {'choice': 1}},
            {"kind": "tf", "q": "“Most of the training compute goes into the later stages that turn a base model into a "
                                "chatbot.”",
             "answer": "False.", "why": "Almost all the computing goes into pretraining.", "key": {'value': False}},
            {"kind": "short", "q": "Where does a chatbot's knowledge mostly come from?",
             "answer": "Pretraining, on trillions of tokens of text.",
             "why": "The later stages teach format, style and judgment; the knowledge was learned in stage 1."},
            {"kind": "short", "q": "A new case: you want a base model to tell you the capital of France. How could you "
                                   "write the input so that simply continuing it gives the answer?",
             "answer": "Write the start of the answer, e.g. <i>“The capital of France is”</i>.",
             "why": "A base model continues text, and the most likely continuation of that sentence is "
                    "<i>“Paris”</i>."},
        ],
    },
    {
        "title": "Supervised fine-tuning",
        "segment": (31, 47),
        "figures": [{"t": 47.2, "caption": "Stage 2: same model, same loss, example conversations."}],
        "body": [
            """<p>Stage two is <b>supervised fine-tuning</b> (SFT). We keep training the <b>same model</b>, with the
<b>same next-token loss</b> (episode 11), but now on examples of <b>conversations</b>: a user asks, and an assistant
answers helpfully. From them the model learns the <b>format</b> (who speaks, and when) and the <b>style</b> (helpful,
clear, friendly).</p>""",
            "{fig0}",
            """<div class="box key"><b class="t">Key idea</b>SFT = same model, same loss, <b>new data</b>: example
conversations teach format and style.</div>""",
        ],
        "exercises": [
            {"kind": "mc", "q": "What changes when we move from pretraining to supervised fine-tuning?",
             "options": ["The model's architecture", "The loss function", "The training data",
                         "The number of layers"],
             "answer": "C.", "why": "Same model, same next-token loss; only the data changes, to example "
                                   "conversations.", "key": {'choice': 2}},
            {"kind": "tf", "q": "“Supervised fine-tuning starts again from a new, random model.”",
             "answer": "False.", "why": "We keep training the same model, so everything it learned in pretraining is "
                                        "kept.", "key": {'value': False}},
            {"kind": "mc", "q": "Which of these is an SFT training example?",
             "options": ["A web page about cats", "A user's question followed by a helpful assistant answer",
                         "Two answers, with a person's pick of the better one",
                         "A math problem whose answer can be checked"],
             "answer": "B.", "why": "A is pretraining data, C is preference data (stage 3), and D is for reinforcement "
                                   "learning on checkable answers (concept 6).", "key": {'choice': 1}},
            {"kind": "short", "q": "Name the two things the model learns from SFT examples, with a few words on each.",
             "answer": "Format and style.", "why": "Format: who speaks, and when. Style: helpful, clear, friendly."},
        ],
    },
    {
        "title": "A chat is one long document",
        "segment": (47, 60),
        "figures": [{"t": 56.0, "caption": "A chat template: special tokens mark who is speaking. The model is trained "
                                           "to predict the assistant's part."},
                    {"t": 60.0, "caption": "To the model, the whole chat is one long sequence of tokens to "
                                           "continue."}],
        "body": [
            """<p>Conversations are turned into text, with <b>special tokens</b> that mark who is speaking, like
<code>&lt;|user|&gt;</code> and <code>&lt;|assistant|&gt;</code> in the video (the exact format varies by model). The
model is trained to predict the <b>assistant's turns</b>.</p>""",
            '<div class="figrow">{fig0}{fig1}</div>',
            """<p>To the model, a chat is still <b>just one long document to continue</b>, with the same next-token
prediction as always.</p>""",
            """<div class="box key"><b class="t">Key idea</b>A chat template turns a conversation into one sequence of
tokens, with special tokens marking the speakers. The model learns to predict the <b>assistant's part</b>.</div>""",
        ],
        "exercises": [
            {"kind": "mc", "q": "In the video's template, what are <code>&lt;|user|&gt;</code> and "
                                "<code>&lt;|assistant|&gt;</code>?",
             "options": ["Ordinary words the model read on the internet", "Special tokens that mark who is speaking",
                         "Comments that are deleted before training", "The names of two separate models"],
             "answer": "B.", "why": "They are special tokens added to the text so the model can tell the turns apart.", "key": {'choice': 1}},
            {"kind": "tf", "q": "“A chatbot reads a conversation in a completely different way from how a base model "
                                "reads text.”",
             "answer": "False.", "why": "To the model, a chat is just one long document, and it predicts the next token "
                                        "as always.", "key": {'value': False}},
            {"kind": "code", "q": "<b>Try it yourself.</b> The code below turns the video's chat into tokens with a toy "
                                  "template (one token per word) and marks which tokens the model is trained to "
                                  "predict. (a) What do the two <code>print</code>s show? (b) Add a second exchange to "
                                  "<code>chat</code>: the user asks <i>“And dogs ?”</i> and the assistant answers "
                                  "<i>“Dogs prefer a warm bed .”</i> What are the two numbers now? (c) After training, a "
                                  "user asks a new question. With which token should the text given to the model end, "
                                  "so that its continuation is the assistant's answer?",
             "code": """chat = [("user", "Why do cats love boxes ?"),
        ("assistant", "Boxes feel safe and warm , so a cat can relax .")]

def to_tokens(chat):
    tokens, train = [], []
    for role, text in chat:
        tokens.append(f"<|{role}|>")            # special token: who speaks
        train.append(False)
        for word in text.split():               # toy tokenizer: one token per word
            tokens.append(word)
            train.append(role == "assistant")   # predict only the assistant's turn
    return tokens, train

tokens, train = to_tokens(chat)
print(len(tokens), sum(train))
print([t for t, keep in zip(tokens, train) if keep][:3])""",
             "answer": "(a) 20 12 and ['Boxes', 'feel', 'safe'] · (b) 31 18 · (c) <code>&lt;|assistant|&gt;</code>",
             "why": """(a) 20 tokens: 7 for the user's turn (<code>&lt;|user|&gt;</code> + 6 words) and 13 for the
assistant's (<code>&lt;|assistant|&gt;</code> + 12). Only the assistant's 12 words are trained, starting
<i>Boxes feel safe</i>. (b) The new exchange adds 1 + 3 user tokens and 1 + 6 assistant tokens: 20 + 11 = 31 tokens, and
12 + 6 = 18 trained. (c) Give <code>&lt;|user|&gt;</code>, the question, then <code>&lt;|assistant|&gt;</code>. The
chat is one document, and after that token the natural continuation is an assistant's answer: exactly what the model
was trained to predict."""},
        ],
    },
    {
        "title": "Learning from preferences",
        "segment": (60, 71),
        "figures": [{"t": 66.0, "caption": "Two answers from the model to the same prompt. A person picks the better "
                                           "one."},
                    {"t": 70.6, "caption": "Many thousands of comparisons teach the model what “better” means."}],
        "body": [
            """<p>Stage three teaches <b>judgment</b>. People compare <b>two answers from the model</b> to the same
prompt, and <b>pick the better one</b>. In the video both answers have the right format, but only one is really
helpful.</p>""",
            '<div class="figrow">{fig0}{fig1}</div>',
            """<p>From many thousands of these comparisons, the model learns what <b>“better”</b> means.</p>""",
            """<div class="box key"><b class="t">Key idea</b>A preference example is a prompt, <b>two of the model's
answers</b>, and <b>which one a person preferred</b>. Many thousands of them teach judgment.</div>""",
        ],
        "exercises": [
            {"kind": "mc", "q": "What does one preference example contain?",
             "options": ["A prompt and one ideal answer written by a person",
                         "A prompt, two answers from the model, and which one a person preferred",
                         "A single answer with a grammar check", "A list of words the model must never use"],
             "answer": "B.", "why": "People compare two of the model's own answers and pick the better one.", "key": {'choice': 1}},
            {"kind": "tf", "q": "“In the preference stage, people write the ideal answers themselves, and the model "
                                "learns to copy them.”",
             "answer": "False.", "why": "People only compare and pick. Learning from written example answers is closer "
                                        "to SFT.", "key": {'value': False}},
            {"kind": "order", "q": "Put the three training stages in order: <i>preferences · supervised fine-tuning · "
                                   "pretraining</i>.",
             "answer": "pretraining → supervised fine-tuning → preferences.",
             "why": "Knowledge first, then the format and style of an assistant, then judgment.", "key": {'items': ['pretraining', 'supervised fine-tuning', 'preferences']}},
            {"kind": "short", "q": "Both answers to <i>“Why do cats love boxes?”</i> are in the right format. So what "
                                   "does a person's pick of answer B teach the model?",
             "answer": "Judgment: which of two well-formed answers is better.",
             "why": "SFT already taught the format; comparisons teach what “better” means."},
        ],
    },
    {
        "title": "RLHF and DPO",
        "segment": (71, 89),
        "figures": [{"t": 83.5, "caption": "RLHF: a reward model learns to score answers the way people would; the "
                                           "chatbot is nudged toward higher scores."},
                    {"t": 89.3, "caption": "DPO: skip the reward model and learn from the preference pairs "
                                           "directly."}],
        "body": [
            """<p>One way to use the comparisons is <b>RLHF</b>: reinforcement learning from human feedback. First,
train a <b>reward model</b> to predict which answers people prefer; it gives each answer a score. Then <b>nudge the
chatbot</b> toward answers that score higher. A simpler method, called <b>DPO</b>, learns from the preference pairs
<b>directly</b>, without the reward model in between.</p>""",
            '<div class="figrow">{fig0}{fig1}</div>',
            """<div class="box key"><b class="t">Key idea</b>Same data, two routes. <b>RLHF</b>: pairs → reward model →
nudge the chatbot toward higher scores. <b>DPO</b>: pairs → chatbot, directly.</div>""",
        ],
        "exercises": [
            {"kind": "order", "q": "Put the steps of RLHF in order: <i>nudge the chatbot toward higher scores · collect "
                                   "preference pairs · the reward model scores the chatbot's answers · train a reward "
                                   "model</i>.",
             "answer": "collect preference pairs → train a reward model → the reward model scores the chatbot's "
                       "answers → nudge the chatbot toward higher scores.",
             "why": "The reward model must learn from people's picks before it can score new answers.", "key": {'items': ['collect preference pairs', 'train a reward model', "the reward model scores the chatbot's answers", 'nudge the chatbot toward higher scores']}},
            {"kind": "mc", "q": "The reward model scores two of the chatbot's answers 0.2 and 0.9. What does RLHF do "
                                "with this?",
             "options": ["Nothing: the scores are only for people to read", "Nudges the chatbot toward answers like the "
                         "0.9 one", "Nudges the chatbot toward the 0.2 one", "Uses the reward model as the chatbot "
                         "from now on"],
             "answer": "B.", "why": "The chatbot is nudged toward answers that score higher, i.e. that people would "
                                   "probably prefer.", "key": {'choice': 1}},
            {"kind": "tf", "q": "“DPO first trains a reward model, then uses it to nudge the chatbot.”",
             "answer": "False.", "why": "That is RLHF. DPO learns from the preference pairs directly.", "key": {'value': False}},
            {"kind": "short", "q": "What do RLHF and DPO have in common, and what is the main difference?",
             "answer": "Both learn from the same preference pairs.",
             "why": "RLHF goes through a reward model and nudges the chatbot toward higher scores; DPO learns from the "
                    "pairs directly, which is simpler."},
        ],
    },
    {
        "title": "AI feedback and checkable answers",
        "segment": (89, 100),
        "figures": [{"t": 99.3, "caption": "Other signals: AI feedback, and RL on checkable answers."}],
        "body": [
            """<p>People are not the only source of feedback. Many labs also use <b>AI feedback</b>, guided by
<b>written principles</b> (in the video: be helpful, be honest, avoid harm), and <b>reinforcement learning on problems
with checkable answers</b>, like math and code: <i>17 × 24 = 408</i> can be checked, and code can be run against its
tests.</p>""",
            "{fig0}",
            """<div class="box key"><b class="t">Key idea</b>The training signal can come from people, from an AI guided
by written principles, or, for problems like math and code, from <b>checking the answer</b>.</div>""",
        ],
        "exercises": [
            {"kind": "mc", "q": "Which task is best suited to reinforcement learning on checkable answers?",
             "options": ["Writing a poem about autumn", "Working out 17 × 24",
                         "Choosing a friendly tone for an email", "Describing the mood of a painting"],
             "answer": "B.", "why": "A calculation has one right answer that can be checked. The others are matters "
                                   "of judgment.", "key": {'choice': 1}},
            {"kind": "tf", "q": "“With AI feedback, an AI gives the feedback, guided by written principles.”",
             "answer": "True.", "why": "That is how the video describes it (one example is Constitutional AI).", "key": {'value': True}},
            {"kind": "short", "q": "A model writes a function <code>add(a, b)</code>. How can its answer be checked "
                                   "without a person?",
             "answer": "Run tests on it, like <code>assert add(2, 3) == 5</code>.",
             "why": "If the tests pass, the answer counts as good, as in the video's “tests pass ✓”."},
            {"kind": "short", "q": "A new case: a model's <code>add</code> always returns 5. Does it pass the video's "
                                   "test? What does that tell you?",
             "answer": "Yes, it passes, although it is wrong.",
             "why": "<code>add(2, 3)</code> happens to be 5. A check is only as good as its tests; more tests (like "
                    "<code>add(1, 1) == 2</code>) would catch it."},
        ],
    },
    {
        "title": "Same machine, different data",
        "segment": (100, 131),
        "figures": [{"t": 110.8, "caption": "The same transformer at every stage."},
                    {"t": 130.8, "caption": "The whole journey, from tokens to a chatbot."}],
        "body": [
            """<p>Through all of this, the <b>machinery doesn't change</b>: the same transformer, with the same
embeddings, attention and next-token prediction. Only the <b>data</b> and the <b>training signal</b> change: internet
text, then conversations, then feedback. And that is the whole series: text becomes <b>tokens</b>, tokens become
<b>vectors with positions</b>, <b>attention</b> lets them talk and <b>MLPs</b> let them think, stacked in <b>blocks</b>;
the final vector becomes <b>probabilities</b>, we pick a token and repeat. Training shapes every number, and fine-tuning
turns it into an assistant.</p>""",
            '<div class="figrow">{fig0}{fig1}</div>',
            """<div class="box key"><b class="t">Key idea</b>A chatbot is the same next-token predictor you met in
every episode. What makes it an assistant is <b>the data and the training signal</b>.</div>""",
        ],
        "exercises": [
            {"kind": "mc", "q": "What is different between a base model and the chatbot made from it?",
             "options": ["A new kind of network architecture", "The data and the training signal it was trained with",
                         "Attention is replaced by a reward model", "It no longer predicts the next token"],
             "answer": "B.", "why": "Same transformer, same next-token prediction; only the training changes.", "key": {'choice': 1}},
            {"kind": "short", "q": "For each stage, name the data it trains on: (a) pretraining, (b) supervised "
                                   "fine-tuning, (c) preferences and RL.",
             "answer": "(a) Internet text. (b) Conversations. (c) Feedback.",
             "why": "Feedback from people's comparisons, from an AI guided by principles, or from checking answers."},
            {"kind": "order", "q": "Put in order: <i>probabilities · tokens · transformer blocks · pick a token · embeddings with positions</i>.",
             "answer": "tokens → embeddings with positions → transformer blocks → probabilities → pick a token "
                       "(then repeat).",
             "why": "This is the data flow of episodes 2–10, closed into the loop of episode 1.", "key": {'items': ['tokens', 'embeddings with positions', 'transformer blocks', 'probabilities', 'pick a token']}},
            {"kind": "tf", "q": "“A chatbot still writes its reply one token at a time, with episode 1's loop.”",
             "answer": "True.", "why": "Nothing about generation changes: the reply is still produced token by token, as "
                                       "a continuation of the chat document.", "key": {'value': True}},
        ],
    },
]
