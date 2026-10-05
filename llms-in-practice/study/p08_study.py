"""Study guide content for LLMs in Practice, episode 8: Agents.

Build:  python framework/study_guide.py llms-in-practice p08 --video llms-in-practice/media/videos/p08_scene/1080p60/AgentVideo.mp4
Figures: {"t": seconds into the rendered video, "caption": ...}; placed in body blocks with {fig0}, {fig1}, ...
Exercise kinds: mc (options), tf, short, number, order, code. Every exercise has an answer and a why.
Real traces come from code/p08_agent (Qwen2.5-3B-Instruct in bfloat16, greedy; and Qwen2.5-1.5B-Instruct for the
invented shops; transformers 4.57.1, torch 2.14.0, CPU). The shops, calendar and map are made up.
"""

LESSON = {
    "series": "LLMs in Practice",
    "label": "Episode 8",
    "title": "Agents",
    "tagline": "Think, act, observe, repeat",
    "duration": "2:46",
    "intro": """<p>This lesson answers one question: how does a model carry out a task that takes several steps? An
<b>agent</b> is the tool loop of episode 7, run until a goal is reached: <b>think</b> (the model picks the next action),
<b>act</b> (the program runs a tool), <b>observe</b> (the result joins the context). Two real runs show how agents fail
(confident wrong answers, invented facts) and how <b>tools that check their inputs and explain errors</b> let the same
model recover.</p>""",
    "prereq": """<div class="box note"><b>Before you start:</b> episode 7 (tool use) and episodes 1–2 (context and
its limits). Code: <code>code/p08_agent</code> (downloads Qwen2.5-3B-Instruct, about 6 GB; needs about 7 GB of RAM;
runs on a CPU in a minute or two).</div>""",
}

CONCEPTS = [
    {
        "title": "The agent loop",
        "segment": (8, 36),
        "figures": [{"t": 35.4, "caption": "Think, act, observe, repeat, until a final answer or a step limit."}],
        "body": [
            """<p>Real tasks take several steps, and each step depends on the last. A model that keeps choosing tools,
in a loop, until a goal is reached is called an <b>agent</b>. The loop has three parts: <b>think</b>, the model decides
the next action; <b>act</b>, the program runs the tool; <b>observe</b>, the result goes back into the context. Repeat
until the model gives a <b>final answer</b>, or a <b>step limit</b> is hit.</p>""",
            "{fig0}",
            """<div class="box key"><b class="t">Key idea</b>An agent is not a new kind of model: it is a loop around a
model that can call tools.</div>""",
        ],
        "exercises": [
            {"kind": "order", "q": "Put one turn of the loop in order: <i>the result is appended to the context · the "
                                   "model decides the next action · the program runs the tool</i>.",
             "answer": "the model decides (think) → the program runs the tool (act) → the result is appended (observe).",
             "why": "Then the loop repeats with the longer context."},
            {"kind": "short", "q": "Why does an agent need a step limit?",
             "answer": "A model can keep calling tools without ever finishing (looping, repeating mistakes); the limit "
                       "guarantees the program stops.",
             "why": "Every step also costs time and tokens."},
        ],
    },
    {
        "title": "Confident, and wrong",
        "segment": (36, 79),
        "figures": [{"t": 51.0, "caption": "The task needs three tools in order: the day, then the shops, then the walk."},
                    {"t": 71.4, "caption": "First try (3B, tools without checks): a bad argument, an empty result, and a "
                                           "confident wrong answer."},
                    {"t": 78.0, "caption": "The 1.5B model, after getting the day, invented two shops with "
                                           "addresses."}],
        "body": [
            """<p>The task: <i>“I want to buy gluten-free bread today. Where can I get it, and how long is the
walk?”</i> The agent has three made-up tools (get today's date, find shops, get the walking time) and must use them in
order: first the day, then the shops, then the walk.</p>""",
            "{fig0}",
            """<p>On the first try, Qwen2.5-3B called <code>find_shops</code> with the day set to <i>“today”</i>, not a
weekday name, so the tool found nothing; then it asked for the walk to <i>“shop for gluten-free bread”</i>. Its final
answer: no shop sells it today. <b>Wrong</b>: Pane Vivo does, on Wednesdays. The even smaller 1.5B model did worse:
after getting the day, it simply <b>invented</b> two shops, with addresses.</p>""",
            '<div class="figrow">{fig1}{fig2}</div>',
            """<div class="box key"><b class="t">Key idea</b>A tool that silently accepts a bad argument returns a
misleading result, and the agent builds a confident answer on it.</div>""",
        ],
        "exercises": [
            {"kind": "mc", "q": "In the first try, which step caused the wrong final answer?",
             "options": ["get_today() returned the wrong day", "find_shops was called with day='today' and returned an "
                         "empty list", "The walking time was too long", "The step limit was too low"],
             "answer": "B.", "why": "The tool's empty result looked like a real answer (“no shops”)."},
            {"kind": "tf", "q": "“If the model's final answer sounds confident and detailed, it was probably based on "
                                "tool results.”",
             "answer": "False.", "why": "The 1.5B model invented detailed shop names and addresses without any tool "
                                       "returning them."},
        ],
    },
    {
        "title": "Tools that explain errors",
        "segment": (79, 117),
        "figures": [{"t": 92.0, "caption": "The fix is in the tools: check every input and say what to do instead."},
                    {"t": 115.8, "caption": "Second try, same model: six steps, three errors, each fixed by reading the "
                                            "error message."}],
        "body": [
            """<p>The fix is not in the model: it is in the <b>tools</b>. Check every input, and when it is wrong, say
<b>why</b> and <b>what to do instead</b>: <i>“day must be a weekday name such as Monday, not 'today'. Call get_today
first.”</i>, <i>“unknown place … Use a shop name returned by find_shops.”</i></p>""",
            "{fig0}",
            """<p>Second try, same model. Step 1: the same two mistakes, now returned as clear errors. Step 2: it gets
the day, Wednesday, but repeats a mistake. Step 3: it checks the day again. Step 4: Wednesday, and the shop list says
Pane Vivo. Step 5: 18 minutes. Step 6: the right answer. Six steps, three errors, each fixed by reading an error
message.</p>""",
            "{fig1}",
            """<div class="box key"><b class="t">Key idea</b>Error messages are part of the agent's prompt. Write them for
the model: what was wrong, and what to do next.</div>""",
        ],
        "exercises": [
            {"kind": "short", "q": "Rewrite this unhelpful tool error for an agent: <code>{\"error\": \"invalid "
                                   "input\"}</code> (the problem: a date was passed as “next Friday”).",
             "answer": "For example: <code>{\"error\": \"date must be YYYY-MM-DD, not 'next Friday'. Call get_today and "
                       "compute the date first.\"}</code>",
             "why": "It names the expected format, echoes the bad value, and suggests the next action."},
            {"kind": "number", "q": "In the second try, how many tool calls did the agent make in total, and how many of "
                                    "them returned errors?",
             "answer": "7 calls, 3 errors.", "why": "Step 1: 2 calls (2 errors); step 2: 2 calls (1 error); steps 3, "
                                                   "4, 5: 1 call each."},
        ],
    },
    {
        "title": "Cost, code and good practice",
        "segment": (117, 164),
        "figures": [{"t": 130.0, "caption": "Every step resends everything so far: 422 tokens at step 1, 814 at step 6."},
                    {"t": 157.5, "caption": "What makes agents work in practice."}],
        "body": [
            """<p>Every step resends everything so far: <b>422 tokens</b> at the start, <b>814</b> at the end. Agents
are slow and costly, and long runs can overflow the window (episode 2). In code it is the tool loop with a limit: for
each step, generate; no tool call means the final answer; otherwise run every call, append the results, and go around
again.</p>""",
            '<div class="figrow">{fig0}{fig1}</div>',
            """<p>To build agents that work: set a <b>step limit</b>; make tools <b>check their inputs</b>, with helpful
errors; keep the toolset <b>small and clear</b>; <b>log</b> every step; ask a <b>human</b> before any action with
consequences; and use a model that is <b>capable enough</b> for the job.</p>""",
            """<div class="box key"><b class="t">Key idea</b>Reliable agents come from the loop around the model:
limits, checks, logs and human approval.</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "The six requests were 422, 551, 652, 691, 760 and 814 tokens. How many tokens did the "
                                    "model read in total?",
             "answer": "3,890.", "why": "422 + 551 + 652 + 691 + 760 + 814 = 3,890: the whole context is read again at "
                                       "every step."},
            {"kind": "code", "q": "<b>Try it yourself.</b> In <code>code/p08_agent/agent.py</code>: (a) set "
                                  "<code>MAX_STEPS = 3</code> and run it. What happens? (b) Set it back to 8 and set "
                                  "<code>VALIDATE = False</code>. What is the final answer, and why is it wrong?",
             "code": """MAX_STEPS = 3        # (a)
VALIDATE = False     # (b), with MAX_STEPS = 8""",
             "answer": "(a) The same first three steps (two errors, get_today, a repeated error, get_today again), then "
                       "“stopped after 3 steps without a final answer”. (b) “Today there are no shops in Milan that sell "
                       "gluten-free bread. The walk … would take approximately 30 minutes.” Wrong, because find_shops "
                       "accepted day='today' and returned an empty list, and the walking tool accepted a made-up place.",
             "why": "Too few steps stop a recovering agent; tools without checks let it finish confidently with bad "
                    "data."},
        ],
    },
]
