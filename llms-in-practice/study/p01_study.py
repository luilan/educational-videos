"""Study guide content for LLMs in Practice, episode 1: Prompts Are Just Context.

Build:  python framework/study_guide.py llms-in-practice p01 --video llms-in-practice/media/videos/p01_scene/1080p60/PromptIsContextVideo.mp4
Figures: {"t": seconds into the rendered video, "caption": ...}; placed in body blocks with {fig0}, {fig1}, ...
Exercise kinds: mc (options), tf, short, number, order, code. Every exercise has an answer and a why.
Every number was produced by code/p01_prompt_is_context (Qwen2.5-0.5B-Instruct tokenizer, transformers 4.57.1).
"""

LESSON = {
    "series": "LLMs in Practice",
    "label": "Episode 1",
    "title": "Prompts Are Just Context",
    "tagline": "What the model really sees when you chat",
    "duration": "2:06",
    "intro": """<p>This lesson answers one question: what does a language model actually receive when you chat with
it? Not speech bubbles, and not a memory of you, but <b>one document of tokens</b>, built by a <b>chat template</b>.
The app <b>resends the whole conversation</b> on every turn, so everything the model knows right now is in that
<b>context</b>. Prompting is <b>writing the beginning of a document</b> whose likely continuation is the answer you
want.</p>""",
    "prereq": """<div class="box note"><b>Before you start:</b> this series builds on How LLMs Work, especially
episode 1 (the generation loop), episode 2 (tokens) and episode 14 (chat templates and fine-tuning). The code uses
Python and the <code>transformers</code> library; see <code>code/p01_prompt_is_context</code> in the repository.</div>""",
}

CONCEPTS = [
    {
        "title": "A chat is one document",
        "segment": (8, 49),
        "figures": [{"t": 34.5, "caption": "The four messages, flattened by Qwen2.5's real chat template. Orange: "
                                           "special tokens that mark who is speaking."},
                    {"t": 47.5, "caption": "The document ends with an open assistant turn, so the most natural "
                                           "continuation is the reply (illustrative reply)."}],
        "body": [
            """<p>A chat app shows <b>speech bubbles</b>, turns and what looks like a memory. The model never sees
any of that. Before each request, the app uses the model's <b>chat template</b> to flatten the system message, your
messages and the assistant's earlier replies into <b>one document</b>. <b>Special tokens</b> mark who is speaking; for
Qwen2.5 they are <code>&lt;|im_start|&gt;</code> and <code>&lt;|im_end|&gt;</code>, and each model family has its
own.</p>""",
            '<div class="figrow">{fig0}{fig1}</div>',
            """<p>Look at the very end: the document stops right after a new <code>&lt;|im_start|&gt;assistant</code>
marker. The most natural way to continue this document is the assistant's reply. <b>Answering is continuing the
text</b>, with the same next-token loop as always.</p>""",
            """<div class="box key"><b class="t">Key idea</b>A conversation reaches the model as <b>one document</b>,
built by a chat template, ending with an <b>open assistant turn</b> that the model continues.</div>""",
        ],
        "exercises": [
            {"kind": "mc", "q": "What does the model receive when you send the fourth message of a chat?",
             "options": ["Only your newest message", "Your newest message plus a summary the model wrote earlier",
                         "One document containing the whole conversation, built by a chat template",
                         "A list of speech bubbles, each processed separately"],
             "answer": "C.", "why": "The app flattens every message into one templated document.", "key": {'choice': 2}},
            {"kind": "short", "q": "Why does the templated document end with <code>&lt;|im_start|&gt;assistant</code> "
                                   "and nothing after it?",
             "answer": "So that the most likely continuation is the assistant's reply.",
             "why": "The model only ever continues text; opening the assistant's turn tells it what to write next."},
            {"kind": "tf", "q": "“Every model family uses the same special tokens, <code>&lt;|im_start|&gt;</code> and "
                                "<code>&lt;|im_end|&gt;</code>.”",
             "answer": "False.", "why": "Templates differ between model families; that is why the code calls the "
                                       "model's own <code>apply_chat_template</code>.", "key": {'value': False}},
            {"kind": "mc", "q": "A user sends one message and no system message. Qwen2.5's template still produces a "
                                "system turn. What does it contain?",
             "options": ["Nothing: an empty system turn",
                         "“You are Qwen, created by Alibaba Cloud. You are a helpful assistant.”",
                         "The user's message, repeated", "The name of the app"],
             "answer": "B.", "why": "This template inserts a default system prompt when none is given (check it with "
                                   "the hands-on code in concept 4). Other models do not.", "key": {'choice': 1}},
        ],
    },
    {
        "title": "Tokens are the model's whole world",
        "segment": (49, 58),
        "figures": [{"t": 56.5, "caption": "The first 12 of the 55 real tokens, with their IDs. Orange: special "
                                           "tokens."}],
        "body": [
            """<p>Then the document becomes <b>tokens</b> (How LLMs Work, episode 2). The four-message chat is
<b>55 tokens</b>, and each one is just a number: <code>&lt;|im_start|&gt;</code> is 151644, <code>system</code> is
8948, a line break is 198. Special tokens are tokens like any other, with their own IDs.</p>""",
            "{fig0}",
            """<div class="box key"><b class="t">Key idea</b>That list of numbers is the model's <b>entire world</b>:
anything that is not in it does not exist for the model.</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "The templated four-message chat is 55 tokens. Its first user turn alone (with the "
                                    "system message) is 28 tokens. How many tokens do the assistant's reply and the "
                                    "follow-up question add, including their template tokens?",
             "answer": "27.", "why": "55 − 28 = 27.", "key": {'parts': [{'label': None, 'value': 27, 'tol': 0.5, 'unit': None}]}},
            {"kind": "tf", "q": "“<code>&lt;|im_end|&gt;</code> is split into several ordinary tokens such as "
                                "<code>&lt;</code>, <code>|</code> and <code>im</code>.”",
             "answer": "False.", "why": "It is a single special token, ID 151645 in Qwen2.5.", "key": {'value': False}},
            {"kind": "short", "q": "Your question <i>“How long should I boil an egg?”</i> is 8 tokens on its own. Why "
                                   "is the first request bigger (28 tokens)?",
             "answer": "The system message and the template's special tokens and role names are added.",
             "why": "Everything in the document counts: roles, markers, line breaks and the system prompt."},
        ],
    },
    {
        "title": "No memory: the whole chat is sent again",
        "segment": (58, 75),
        "figures": [{"t": 74.0, "caption": "The second request contains the first one again: 28 tokens, then 55."}],
        "body": [
            """<p>The model has <b>no memory between messages</b>. Every time you send one, the app sends the
<b>whole conversation again, from the top</b>. In our chat, the first question is a 28-token request; the follow-up is
a 55-token request that contains the first one again.</p>""",
            "{fig0}",
            """<p>So a chat gets longer to read on every turn: a third exchange would make the request 79 tokens. This
is why long chats get slower and cost more, and why they eventually hit a limit (next episode).</p>""",
            """<div class="box key"><b class="t">Key idea</b>The “memory” of a chat is the app <b>resending the
transcript</b>. The model itself starts from scratch on every request.</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "The requests on three turns are 28, 55 and 79 tokens. How many tokens does the "
                                    "model read in total over the three requests?",
             "answer": "162.", "why": "28 + 55 + 79 = 162: each request is read in full.", "key": {'parts': [{'label': None, 'value': 162, 'tol': 0.5, 'unit': None}]}},
            {"kind": "tf", "q": "“After you tell an assistant your name in message 1, the model stores it in its "
                                "weights for message 2.”",
             "answer": "False.", "why": "The weights do not change while chatting. Your name is known in message 2 only "
                                       "because the app resends message 1.", "key": {'value': False}},
            {"kind": "mc", "q": "An app deletes the oldest messages of a long chat to save money. What happens?",
             "options": ["Nothing: the model remembers them anyway", "The model can no longer use what they said",
                         "The model's weights are reset", "The chat template stops working"],
             "answer": "B.", "why": "If it is not in the context, the model cannot see it.", "key": {'choice': 1}},
        ],
    },
    {
        "title": "Everything is context",
        "segment": (75, 101),
        "figures": [{"t": 87.0, "caption": "Everything the model knows about you right now is in the context."},
                    {"t": 100.0, "caption": "The episode's code: messages → chat template → tokens → count."}],
        "body": [
            """<p>Everything the model knows about you, right now, lives in that text: the <b>system prompt</b>,
<b>earlier turns</b>, <b>documents you pasted</b> and <b>results from tools</b> (episode 7). Yesterday's chat or a
file you did not send are not there, so the model cannot use them.</p>""",
            '<div class="figrow">{fig0}{fig1}</div>',
            """<p>In code it is three steps: a list of messages, <code>apply_chat_template</code> to get one string,
then tokenize and count. Run <code>code/p01_prompt_is_context/what_the_model_sees.py</code> to see the full document,
the token IDs and the size of each request.</p>""",
            """<div class="box key"><b class="t">Key idea</b><b>If it isn't in the context, the model can't see
it.</b> Apps that seem to “remember” or “search” are putting text into the context for you.</div>""",
        ],
        "exercises": [
            {"kind": "mc", "q": "Which of these can the model use when answering your next message?",
             "options": ["A PDF you pasted earlier in this chat", "A chat you had with it yesterday, in another window",
                         "A file on your computer you never uploaded", "Your browser history"],
             "answer": "A.", "why": "Only the pasted PDF is part of the context that is sent.", "key": {'choice': 0}},
            {"kind": "order", "q": "Put in order what happens to your message: <i>tokens · chat template · the model "
                                   "continues · list of messages</i>.",
             "answer": "list of messages → chat template → tokens → the model continues.",
             "why": "The messages are flattened into one document, tokenized, then continued.", "key": {'items': ['list of messages', 'chat template', 'tokens', 'the model continues']}},
            {"kind": "short", "q": "An assistant answers a question about today's news correctly, although its "
                                   "training data is old. Using this lesson, how is that possible?",
             "answer": "A tool (such as web search) put today's news into the context.",
             "why": "Tool results are text added to the context, like anything else."},
            {"kind": "code", "q": "<b>Try it yourself.</b> Run the code below. (a) What system prompt appears, although "
                                  "you did not write one? (b) How many tokens is the request? (c) Add a system message "
                                  "<i>“You are a friendly cooking assistant.”</i> at the start of <code>messages</code>: "
                                  "how many tokens now?",
             "code": """from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
messages = [{"role": "user", "content": "How long should I boil an egg?"}]
text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
print(text)
print(len(tok(text)["input_ids"]))""",
             "answer": "(a) “You are Qwen, created by Alibaba Cloud. You are a helpful assistant.” (b) 37. (c) 28.",
             "why": "Qwen2.5's template inserts a default system prompt when none is given. Your shorter system "
                    "message replaces it, so the request shrinks from 37 to 28 tokens."},
        ],
    },
    {
        "title": "Prompting is writing the start of a document",
        "segment": (101, 123),
        "figures": [{"t": 116.5, "caption": "A prompt is the beginning of a document; a good example makes the "
                                            "continuation easy to predict."}],
        "body": [
            """<p>This is why prompting works the way it does. You are not giving orders to a mind. You are
<b>writing the beginning of a document</b>, so that the answer you want is its <b>most likely continuation</b>.</p>""",
            "{fig0}",
            """<p><b>Clear instructions</b> and a <b>few good examples</b> (called <i>few-shot</i> prompting) make
that continuation easy to predict: after a soft-boiled question answered in the format <i>“A: 6 minutes.”</i>, the
likely continuation of the hard-boiled question is an answer in the same format.</p>""",
            """<div class="box key"><b class="t">Key idea</b>A good prompt sets up a document whose natural next
words are the answer you want, in the form you want.</div>""",
        ],
        "exercises": [
            {"kind": "mc", "q": "You want answers as a single number of minutes. Which prompt makes that most likely?",
             "options": ["“Eggs?”", "“Tell me about boiling eggs.”",
                         "“Q: Soft-boiled egg? A: 6\\nQ: Hard-boiled egg? A:”", "“Please be smart.”"],
             "answer": "C.", "why": "The example sets the format, so the likely continuation is a single number.", "key": {'choice': 2}},
            {"kind": "tf", "q": "“Few-shot prompting means retraining the model on a few examples.”",
             "answer": "False.", "why": "The examples are only text in the context; the weights do not change.", "key": {'value': False}},
            {"kind": "short", "q": "Rewrite <i>“Translate: cat”</i> as the start of a document whose most likely "
                                   "continuation is the Italian word.",
             "answer": "For example: <i>“English: dog → Italian: cane\\nEnglish: cat → Italian:”</i>",
             "why": "An example sets the pattern, and the open slot at the end is what the model continues."},
        ],
    },
]
