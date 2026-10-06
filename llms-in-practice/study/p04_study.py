"""Study guide content for LLMs in Practice, episode 4: Embeddings for Search.

Build:  python framework/study_guide.py llms-in-practice p04 --video llms-in-practice/media/videos/p04_scene/1080p60/EmbeddingSearchVideo.mp4
Figures: {"t": seconds into the rendered video, "caption": ...}; placed in body blocks with {fig0}, {fig1}, ...
Exercise kinds: mc (options), tf, short, number, order, code. Every exercise has an answer and a why.
Real numbers come from code/p04_embedding_search (all-MiniLM-L6-v2, transformers 4.57.1, torch 2.14.0, CPU);
toy calculations were checked in Python.
"""

LESSON = {
    "series": "LLMs in Practice",
    "label": "Episode 4",
    "title": "Embeddings for Search",
    "tagline": "Meaning as distance",
    "duration": "2:14",
    "intro": """<p>This lesson answers one question: how can a computer find text by <b>meaning</b>, not by matching
words? An <b>embedding model</b> turns a whole text into one vector (here 384 numbers), so that similar meanings point
in similar directions. A question becomes a vector too, and <b>cosine similarity</b> ranks the documents. This is
the search step behind RAG (next episode).</p>""",
    "prereq": """<div class="box note"><b>Before you start:</b> How LLMs Work episode 3 (embeddings) and Foundations
F01–F02 (vectors, the dot product and cosine similarity). Code: <code>code/p04_embedding_search</code> (downloads a
90 MB model, runs on a CPU).</div>""",
}

CONCEPTS = [
    {
        "title": "One vector per text",
        "segment": (8, 47),
        "figures": [{"t": 34.5, "caption": "An embedding model turns a sentence into 384 numbers (the first five are "
                                           "real). Similar meanings point in similar directions."},
                    {"t": 46.5, "caption": "Inside: a small transformer, the token vectors averaged into one, then "
                                           "scaled to length 1."}],
        "body": [
            """<p>Keyword search matches <b>letters</b>, not meaning: searching your notes for <i>“kitten”</i> finds
nothing, although you wrote about your cat twice. An <b>embedding model</b> reads a whole piece of text and returns a
single <b>vector</b>; for all-MiniLM-L6-v2, <b>384 numbers</b>. Texts with similar meanings get vectors that point in
similar directions: the same idea as token embeddings (How LLMs Work, episode 3), for whole sentences.</p>""",
            '<div class="figrow">{fig0}{fig1}</div>',
            """<p>Under the hood it is a small <b>transformer</b>: each token comes out as a vector, the vectors are
<b>averaged</b> into one, and the result is <b>scaled to length 1</b>, so only its direction matters.</p>""",
            """<div class="box key"><b class="t">Key idea</b>An embedding is a text's <b>meaning as a direction</b> in a
high-dimensional space.</div>""",
        ],
        "exercises": [
            {"kind": "mc", "q": "What does all-MiniLM-L6-v2 return for the sentence “Cats love sleeping in cardboard "
                                "boxes.”?",
             "options": ["One vector of 384 numbers", "One vector per word", "A list of keywords",
                         "A probability for the next token"],
             "answer": "A.", "why": "Token vectors are averaged into a single sentence vector.", "key": {'choice': 0}},
            {"kind": "tf", "q": "“After scaling to length 1, two sentence vectors can only differ in direction.”",
             "answer": "True.", "why": "All vectors have the same length, so only the direction carries meaning.", "key": {'value': True}},
            {"kind": "short", "q": "Why does keyword search find nothing for “kitten” in notes about a cat?",
             "answer": "The notes share no word with the query; keyword search compares letters, not meaning.",
             "why": "“kitten”, “cat” and “cats” are different strings."},
        ],
    },
    {
        "title": "A map of meaning",
        "segment": (47, 63),
        "figures": [{"t": 62.0, "caption": "The 8 sentences squashed from 384 dimensions to 2 (real PCA positions): "
                                           "eggs together, cats together, the train apart."}],
        "body": [
            """<p>To look at the vectors we can squash 384 dimensions down to two (with PCA). These are the real
positions of our eight sentences: the three about <b>eggs</b> land together, the two about <b>cats</b> sit side by side,
and the <b>train to Milan</b> is off on its own.</p>""",
            "{fig0}",
            """<div class="box key"><b class="t">Key idea</b>Similar topics form <b>clusters</b>. A 2-D picture loses
most of the information, but it shows the idea.</div>""",
        ],
        "exercises": [
            {"kind": "tf", "q": "“In the 2-D map, the two cat sentences are close because they share many words.”",
             "answer": "False.", "why": "“Cats love sleeping in cardboard boxes” and “Our cat naps on the sofa all "
                                       "afternoon” share almost no words; they are close because of meaning.", "key": {'value': False}},
            {"kind": "short", "q": "Why can the 2-D map be misleading?",
             "answer": "Squashing 384 dimensions into 2 throws most of the information away; points close in 2-D are "
                       "not always close in 384-D.",
             "why": "Search always uses the full vectors, never the 2-D picture."},
        ],
    },
    {
        "title": "Searching with cosine similarity",
        "segment": (63, 98),
        "figures": [{"t": 84.5, "caption": "The kitten question lands next to the cats; the top two scores are the cat "
                                           "sentences, with zero words in common."},
                    {"t": 97.5, "caption": "Two more real queries: the right document wins each time."}],
        "body": [
            """<p>A question becomes a vector too. <i>“Where does my kitten like to sleep?”</i> lands right next to
the cats. To rank the documents we take the <b>cosine similarity</b>; for length-1 vectors it is just a <b>dot
product</b>. Cat on the sofa: <b>0.65</b>. Cats in boxes: <b>0.60</b>. Dogs: only <b>0.21</b>. And not one word in
common.</p>""",
            '<div class="figrow">{fig0}{fig1}</div>',
            """<p>Ask how long to cook a hard-boiled egg, and the best match is <i>“Boil eggs for 10 minutes”</i>
(0.65). Ask about the first departure to Milan, and the train wins easily (0.57); everything else scores close to
zero (0.10).</p>""",
            """<div class="box key"><b class="t">Key idea</b>Search = embed the question, take its dot product with every
document vector, return the highest.</div>""",
        ],
        "exercises": [
            {"kind": "number", "q": "Two length-1 vectors in 3-D are a = (0.6, 0.8, 0) and b = (0.8, 0.6, 0). What is "
                                    "their cosine similarity?",
             "answer": "0.96.", "why": "0.6 × 0.8 + 0.8 × 0.6 + 0 × 0 = 0.48 + 0.48 = 0.96.", "key": {'parts': [{'label': None, 'value': 0.96, 'tol': 0.005, 'unit': None}]}},
            {"kind": "mc", "q": "Cosine similarity of 0 between two length-1 vectors means:",
             "options": ["They are identical", "They point at right angles: unrelated directions",
                         "They point in opposite directions", "One of them is zero"],
             "answer": "B.", "why": "The dot product of perpendicular vectors is 0 (Foundations F02).", "key": {'choice': 1}},
            {"kind": "short", "q": "The Milan question scores 0.57 with the train sentence and 0.10 with the next best. "
                                   "What does that big gap tell you?",
             "answer": "The search is confident: one document clearly matches, and the rest are unrelated.",
             "why": "When the top scores are close together, several documents are similarly relevant."},
        ],
    },
    {
        "title": "At scale, and the limits",
        "segment": (98, 131),
        "figures": [{"t": 108.0, "caption": "The whole search in a few lines: embed once, then a dot product per "
                                            "document."},
                    {"t": 125.5, "caption": "Vector databases find nearest neighbours fast; keyword search still wins for "
                                            "exact names, codes and numbers."}],
        "body": [
            """<p>In code: embed your documents <b>once</b> and store the vectors. When a question comes in, embed it,
take its dot product with every stored vector, and return the top few.</p>""",
            '<div class="figrow">{fig0}{fig1}</div>',
            """<p>With millions of documents, a <b>vector database</b> finds the nearest neighbours fast by searching
<b>approximately</b>. But close in meaning is not the same as <b>correct</b>, and for exact things like names, codes or
numbers, plain <b>keyword search</b> still wins. Many systems combine both (<b>hybrid search</b>).</p>""",
            """<div class="box key"><b class="t">Key idea</b>Embeddings find what is <b>about</b> the same thing;
keywords find what <b>says</b> the same thing. Good search often needs both.</div>""",
        ],
        "exercises": [
            {"kind": "mc", "q": "A user searches a parts catalogue for “part number 7741-B”. Which search is most "
                                "reliable?",
             "options": ["Embedding search only", "Keyword (exact) search", "Sampling at high temperature",
                         "A 2-D PCA map"],
             "answer": "B.", "why": "Exact codes have no “meaning” for an embedding model; matching the string is what "
                                   "you want.", "key": {'choice': 1}},
            {"kind": "tf", "q": "“The top result of an embedding search is always correct.”",
             "answer": "False.", "why": "It is the closest in meaning among your documents, which may still be wrong or "
                                       "irrelevant if no document answers the question.", "key": {'value': False}},
            {"kind": "code", "q": "<b>Try it yourself.</b> Add these lines at the end of <code>search.py</code> and run it. "
                                  "(a) Which document is the top match for each question, with what score? "
                                  "(b) The top egg score is lower than for the hard-boiled question (0.65). Why might "
                                  "that be? (c) Compute the similarity of “I love my cat” and “I love my car” and "
                                  "explain the result.",
             "code": """search("Can I still eat a boiled egg I made five days ago?")
search("My puppy is full of energy")
a, b = embed(["I love my cat", "I love my car"])
print(round(float(a @ b), 2))""",
             "answer": "(a) “Store cooked eggs in the fridge for up to a week.” (0.56); “Dogs need a walk at least twice "
                       "a day.” (0.47). (b) The question is about safety and time, which no document states directly, "
                       "so even the best match is only partly related. (c) 0.60: the sentences share their structure and "
                       "most words, so they score high although a cat and a car mean very different things.",
             "why": "Similarity mixes topic, wording and structure; it is a useful signal, not a guarantee of meaning."},
        ],
    },
]
