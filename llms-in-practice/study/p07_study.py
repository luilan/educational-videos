"""Study guide content for LLMs in Practice, episode 7: Tool Use.

Build:  python framework/study_guide.py llms-in-practice p07 --video llms-in-practice/media/videos/p07_scene/1080p60/ToolUseVideo.mp4
Figures: {"t": seconds into the rendered video, "caption": ...}; placed in body blocks with {fig0}, {fig1}, ...
Exercise kinds: mc (options), tf, short, number, order, code. Every exercise has an answer and a why.
Real outputs come from code/p07_tool_use (Qwen2.5-1.5B-Instruct and its chat template, greedy; transformers 4.57.1,
torch 2.14.0, CPU). The weather service is made up.
"""

LESSON = {
    "series": "LLMs in Practice",
    "label": "Episode 7",
    "title": "Tool Use",
    "tagline": "How a model calls a function",
    "duration": "2:17",
    "intro": """<p>This lesson answers one question: how can a model that only reads and writes text check the weather,
search the web or run code? It never runs anything itself. The app <b>describes the tools</b> in the context, the
model <b>writes a request</b> in a trained format, the app <b>runs the function</b>, and the <b>result goes back into
the context</b> for the model to continue. You control which tools exist, so you control what the model can do.</p>""",
    "prereq": """<div class="box note"><b>Before you start:</b> episode 1 of this series (chat templates and context).
Code: <code>code/p07_tool_use</code> (downloads Qwen2.5-1.5B-Instruct, about 3 GB; runs on a CPU).</div>""",
}

CONCEPTS = [
    {
        "title": "The model only writes requests",
        "segment": (8, 28),
        "figures": [{"t": 26.5, "caption": "The model writes a request in a trained format; your program does the "
                                           "work."}],
        "body": [
            """<p>Ask a model about today's weather and it cannot know: no window, no internet, no clock, just
<b>text in and text out</b>. Yet assistants check the weather, search the web and run code. The trick: the model
<b>never runs anything itself</b>. It writes a <b>request</b>, in a format it was trained on, and <b>your program</b>
does the work.</p>""",
            "{fig0}",
            """<div class="box key"><b class="t">Key idea</b>Tool use = the model writes structured text asking for an
action; ordinary code decides whether and how to perform it.</div>""",
        ],
        "exercises": [
            {"kind": "tf", "q": "“When a model calls a weather tool, the model itself connects to the weather service.”",
             "answer": "False.", "why": "The model only writes the request; the app's code calls the service."},
            {"kind": "mc", "q": "Where does the model learn the format for writing tool calls?",
             "options": ["It invents a new format each time", "From training (fine-tuning on examples) plus the "
                         "instructions in the system prompt", "From the weather service", "From the user"],
             "answer": "B.", "why": "Tool-capable models are fine-tuned on tool-call examples, and the template restates "
                                   "the format."},
        ],
    },
    {
        "title": "The four steps",
        "segment": (28, 85),
        "figures": [{"t": 44.0, "caption": "Step 1: the real Qwen2.5 template pastes the tool's JSON description into the "
                                           "system prompt."},
                    {"t": 55.3, "caption": "Step 2: instead of an answer, the model writes a tool call."},
                    {"t": 69.0, "caption": "Step 3: the program parses the JSON and runs the real function."},
                    {"t": 84.3, "caption": "Step 4: the result goes back into the context, and the model answers."}],
        "body": [
            """<p><b>1. Describe the tools.</b> A name, what it does and its parameters, written as JSON. The chat
template pastes this into the <b>system prompt</b> with instructions: you may call functions, here are their
signatures, here is how to write a call. It is just more text in the context.</p>""",
            '<div class="figrow">{fig0}{fig1}</div>',
            """<p><b>2. The model writes a call.</b> Asked <i>“What is the weather like in Milan right now? Do I need
an umbrella?”</i>, the model's reply is not an answer but
<code>&lt;tool_call&gt; {"name": "get_weather", "arguments": {"city": "Milan"}} &lt;/tool_call&gt;</code>.
<b>3. The app runs it:</b> it spots the tool call, reads the JSON and runs the real function (here a made-up
service: 18 °C, light rain). This is the only moment anything happens in the world. <b>4. The result goes back</b>
into the context as a tool response; the model now reads <b>264 tokens</b>, including the weather, and answers.</p>""",
            '<div class="figrow">{fig2}{fig3}</div>',
            """<div class="box key"><b class="t">Key idea</b>Describe → call → run → return. Everything except “run” is
text in the context.</div>""",
        ],
        "exercises": [
            {"kind": "order", "q": "Put in order: <i>the app runs the function · the model writes a tool call · the "
                                   "tool descriptions are added to the prompt · the model answers · the result is "
                                   "appended to the context</i>.",
             "answer": "tool descriptions added → model writes a tool call → app runs the function → result appended → "
                       "model answers.",
             "why": "These are the four steps, plus the final answer."},
            {"kind": "number", "q": "The first request was 204 tokens; after the tool call and its result were added, "
                                    "264. How many tokens did the call and the response add?",
             "answer": "60.", "why": "264 − 204 = 60, including the template's tags and markers."},
            {"kind": "short", "q": "Why does the function's docstring matter?",
             "answer": "It becomes the tool's description in the prompt, which is how the model decides when and how to "
                       "call it.",
             "why": "The template turns the signature and docstring into the JSON the model reads."},
        ],
    },
    {
        "title": "When the model does not call",
        "segment": (85, 103),
        "figures": [{"t": 102.4, "caption": "A vaguer question: the 1.5B model does not call the tool and asks for the "
                                            "location it was given."}],
        "body": [
            """<p>Ask more vaguely, <i>“Do I need an umbrella in Milan right now?”</i>, and the same small model did not
call the tool at all: it asked for your location, even though you had said Milan. Larger models call tools more
reliably, and clear descriptions help. Your code must handle <b>no call</b>, and <b>broken JSON</b>.</p>""",
            "{fig0}",
            """<div class="box key"><b class="t">Key idea</b>Whether to call a tool is the model's guess. Robust apps
expect missing, wrong or malformed calls.</div>""",
        ],
        "exercises": [
            {"kind": "mc", "q": "The model's reply contains <code>&lt;tool_call&gt;</code> but the JSON inside is "
                                "invalid. What should the app do?",
             "options": ["Crash", "Run the function with empty arguments", "Not run anything; report the error or ask "
                         "the model to try again", "Guess the arguments"],
             "answer": "C.", "why": "Never act on a request you could not parse."},
            {"kind": "short", "q": "Give one change to the tool description that might make the model call "
                                   "<code>get_weather</code> for umbrella questions.",
             "answer": "For example: “Get the current weather in a city, including rain. Use it for questions about "
                       "rain, umbrellas or what to wear.”",
             "why": "The model decides from the description; mentioning the use cases makes the match easier."},
        ],
    },
    {
        "title": "The loop, and who is in control",
        "segment": (103, 134),
        "figures": [{"t": 111.7, "caption": "Tool use is a loop: generate, run any call, append the result, generate "
                                            "again."},
                    {"t": 126.5, "caption": "Read-only tools are harmless; actions need confirmation, checked arguments "
                                            "and limits."}],
        "body": [
            """<p>In code it is a <b>loop</b>: generate; if there is no tool call, you are done; otherwise parse it,
run the function, append the result, and generate again.</p>""",
            '<div class="figrow">{fig0}{fig1}</div>',
            """<p>The model chooses what to call, but <b>you choose what it can do</b>. Reading the weather is harmless.
Sending an email or making a payment is not: <b>ask the user to confirm</b>, <b>check the arguments</b>, and never
run code from the model without limits. Give a model several tools and a goal, run this loop until the job is done,
and you have an <b>agent</b> (next episode).</p>""",
            """<div class="box key"><b class="t">Key idea</b>The tool list is the model's set of powers. Grant only what
is needed, and guard every action with consequences.</div>""",
        ],
        "exercises": [
            {"kind": "mc", "q": "Which tool most needs a confirmation step before it runs?",
             "options": ["get_weather(city)", "search_docs(query)", "transfer_money(to, amount)", "get_time()"],
             "answer": "C.", "why": "It changes the world irreversibly; the others only read information."},
            {"kind": "code", "q": "<b>Try it yourself.</b> Replace the two <code>chat(…)</code> calls at the end of "
                                  "<code>tool_use.py</code> with the lines below and run it. (a) Which question triggers "
                                  "a tool call, with what arguments? (b) What does the model do with the other one, and "
                                  "why is that correct?",
             "code": """chat("Is it raining in Paris right now?")
chat("Write a haiku about rain.")""",
             "answer": "(a) “Is it raining in Paris right now?” → get_weather with {\"city\": \"Paris\"}; the fake service "
                       "says light rain, and the model answers “Yes, it is currently raining in Paris…”. (b) For the "
                       "haiku it answers directly, with no tool call: writing a poem needs no outside information.",
             "why": "The model decides per question whether a tool is useful. (The fake service returns the same weather "
                    "for every city.)"},
        ],
    },
]
