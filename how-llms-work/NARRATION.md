# How LLMs Work: narration

Narration text for every video, generated from the `*_script.py` files (one paragraph per section).

## V01 · What is an LLM?

*How a language model writes, one word at a time* · 2:05

1. You type a question, and a large language model writes back an answer, word by word. It can feel like magic, or like someone is in there typing. But at its core, an LLM does one surprisingly simple thing, over and over.

2. It predicts the next word. Give it the words: the cat sat on the... and it answers just one question. What comes next?

3. Actually, it doesn't pick a single word. It gives a probability to every word it knows. Mat gets a high score. Floor, a little less. Sofa, less still. And banana? Almost nothing.

4. Then we pick one of them, usually one of the likely ones, and add it to the end of the text.

5. And now, the trick. We feed the whole thing back in. The model reads: the cat sat on the mat, and predicts the next word again. Maybe a period. Then another word, and another. That's all text generation is: predict, pick, append, repeat.

6. In code, the whole loop fits in a few lines. We start with some tokens. We ask the model for probabilities, sample one, and append it. Then we go around again, until we reach a limit, or the model says it's done.

7. So everything interesting hides inside this one function: model. Underneath, it's just a huge pile of numbers. So how can a pile of numbers read: the cat sat on the, and know that mat is likely? That's what this series is about.

8. First, we'll chop text into tokens. Each token becomes a vector, called an embedding, and we'll add information about its position. Then comes the heart of it all: attention, where words look at each other. Next, the MLP, where facts are stored. Together, they form a transformer block, stacked many times. Finally, we turn vectors back into probabilities, see how training sets all those numbers, and build a tiny GPT of our own.

9. Next up: tokens. How do you turn text into something a computer can actually work with? See you there.

## V02 · Tokens: Chopping Text into Pieces

*How text becomes numbers a model can read* · 2:53

1. Last time, we said an LLM predicts the next word. That was a small lie. It actually predicts the next token. So what is a token? And why not just use words?

2. A neural network only understands numbers. So before anything else, we need a way to turn text into a list of numbers, and back again. That's the tokenizer's job.

3. The simplest idea: one number per character. The vocabulary is tiny, just a few hundred symbols. But sequences get long, and a single letter carries almost no meaning. The model would spend its effort just spelling.

4. The other extreme: one number per word. Now sequences are short, but the vocabulary explodes. Think of every name, every typo, every new bit of slang. And a word the model never saw during training? It simply can't read it.

5. Modern LLMs take the middle road: subword tokens. Common words get a single token. Rarer words are built from pieces. In GPT two's tokenizer, tokenization becomes two pieces: token, and ization. And catnap becomes three: cat... N... and ap.

6. The most popular way to choose these pieces is called byte pair, encoding. Start with single characters. Count every pair of neighbours in a pile of text. Then merge the most frequent pair into a brand new token.

7. Here's a tiny example. E and S appear side by side three times, as often as any pair. So we merge them into a single token. Now that new token and T are the top pair. Merge again. Then L and O. And then that pair and W. Already, the tokenizer has discovered the word low, and the ending E S T.

8. In code, training a tokenizer is a short loop. Count the pairs. Pick the most frequent one. Merge it everywhere. And repeat, tens of thousands of times.

9. All those merges give us a vocabulary. GPT two has about fifty thousand tokens; newer models use a hundred thousand or more. Each token gets an ID, and our sentence becomes a short list of numbers. Notice that spaces are part of the tokens: cat with a space in front is a different token from cat without one.

10. Tokens explain some odd behaviour. Ask a model how many R's are in strawberry, and it may stumble. That's because it never sees the letters. To GPT two, strawberry, with a space in front, is one single token.

11. So now our text is a list of numbers. But an ID like thirty-seven ninety-seven is just a label. It says nothing about what a cat is. Next time, we'll turn each token into something much richer: a vector. That's embeddings.

## V03 · Embeddings: Words as Vectors

*How numbers start to carry meaning* · 2:28

1. Our sentence is now a list of token IDs. But an ID is just a label. Cat is token thirty-seven ninety-seven, and its neighbours are rief and esc. The numbers mean nothing. We need numbers that actually carry meaning.

2. The idea: give every token a vector, a list of numbers, like coordinates. Then each token becomes a point in space.

3. These vectors are stored in a big table called the embedding matrix, with one row for every token in the vocabulary. Embedding a token is just a lookup: token thirty-seven ninety-seven grabs row thirty-seven ninety-seven.

4. Real vectors are long. The smallest GPT two uses seven hundred sixty-eight numbers per token, and the largest models use many thousands. We'll draw just two or three, so we can see them.

5. Here's the key idea. After training, tokens that are used in similar ways end up close together. Cat lands near dog and kitten. Mat, near rug and carpet. Numbers gather in one place, verbs in another.

6. How do we measure close? With the dot product: multiply matching coordinates, then add them up. Vectors pointing the same way score high. Unrelated ones score near zero, and opposite ones go negative. Divide by their lengths, and you get cosine similarity: the cosine of the angle between them.

7. Directions can carry meaning too. In classic word embeddings like word two vec, the step from man to woman is roughly the same as the step from king to queen. So king, minus man, plus woman, lands close to queen.

8. Nobody writes these numbers by hand. They start out random. Then training nudges them, a tiny bit at a time, until the geometry reflects how words are actually used.

9. In code, it's a single line. The embedding matrix has one row per token. Index it with our token IDs, and we get one vector per token: a five by seven sixty-eight array for our sentence.

10. So every token is now a point in a space of meaning. But there's a catch. The cat sat on the mat, and the mat sat on the cat, contain exactly the same tokens, so they get exactly the same vectors. Nothing says which came first. Next time: position.

## V04 · Where Am I? Position

*How a model knows the order of words* · 2:26

1. The cat sat on the mat. The mat sat on the cat. Same tokens, completely different meaning. So far, our token vectors can't tell these apart.

2. And it matters. Attention, the heart of the transformer that we'll meet next, compares every token with every other token. Nothing in that comparison says which one came first. We have to put the order into the vectors themselves.

3. The most obvious idea: just add the position number. One, two, three, and so on. But these numbers grow without limit. By token five thousand, the position would drown out the meaning.

4. Another option, used by GPT two, is to learn it. Keep a second table, with one vector for each position, and add it to the token's embedding. Simple, but the model can't handle positions it never saw in training.

5. The original transformer paper used a neat trick instead: waves. Each pair of dimensions follows a sine and a cosine wave, and every pair has its own frequency. To encode a position, you read off where each wave is at that point.

6. It works like a clock. The second hand moves fast, the minute hand slower, the hour hand slowest. Each hand alone is ambiguous, but read them together and you know the exact time. Fast waves tell neighbours apart; slow waves track where you are in the long run.

7. This position vector is simply added to the token's embedding. So cat at position two, and cat at position six, start out as different vectors: the same meaning, in a different place.

8. In code, it's a few lines. For each position and each pair of dimensions, compute an angle. Put the sine in the even dimensions, and the cosine in the odd ones. Then add the result to the embeddings.

9. Most modern models, like Llama, use a newer method called rotary position embedding, or rope. Instead of adding a vector, it rotates pairs of numbers by an angle that grows with position. When two tokens are compared, only the difference between their angles matters, so the model directly sees how far apart they are.

10. Now every token vector carries two things: what it is, and where it is. We're finally ready for the heart of the transformer. Next time: attention.

## V05 · Attention I: Tokens Talking to Each Other

*How words borrow meaning from their neighbours* · 2:22

1. Each token now has a vector that says what it is, and where it is. But each vector was made on its own, without looking at any other word. And words don't work like that.

2. Take the word bank. In: I sat on the river bank, it means the edge of a river. In: I paid money into the bank, it's a place that keeps your money. Same token, same starting vector, but two very different meanings.

3. Attention lets every token look at the other tokens, and pull in information from the ones that matter. Bank looks around, notices river, and updates its own vector to mean: riverbank.

4. Each token decides how much to listen to every other token. These attention weights add up to one, like a budget. Bank might spend most of its attention on river, and very little on the, or on.

5. How does a token decide? Every token produces three new vectors. A query: what am I looking for? A key: what do I contain? And a value: what will I share, if someone listens to me?

6. Think of a library. Your query is the question you bring. Each book has a key, like the title on its spine. You compare your question with every title, and the better the match, the more you read from that book's contents: its value.

7. The match is measured with the dot product we met in the embeddings video. If bank's query points in the same direction as river's key, the score is high, and bank pays attention to river.

8. Then bank collects a weighted mix of all the values, mostly river's, and adds it to its own vector. Now its vector doesn't just mean bank. It means bank, next to a river.

9. One rule for language models: a token can only look backwards. When predicting the next word, the future isn't written yet. So cat can look at the, but never at sat.

10. Where do the queries, keys and values come from? Each one is made by multiplying the token's vector by a matrix, and those matrices are learned in training. Nobody tells the model that river explains bank. It figures that out from data.

11. That's the idea of attention: tokens asking questions, and borrowing meaning from each other. Next time, we'll open it up and do the actual math, step by step.

## V06 · Attention II: The Math

*Queries, keys and values, one matrix at a time* · 2:18

1. Last time, every token asked a question and borrowed meaning from the others. Now let's do it for real, with actual numbers. We'll follow our five tokens: the cat sat on the.

2. Start with the input: one row per token. That's a matrix called X, with five rows, and d columns. We'll use d equals four, so every number fits on screen.

3. Multiply X by three learned matrices: W Q, W K, and W V. That gives us Q, K, and V: a query, a key, and a value for every token, all computed at once.

4. Next, compare every query with every key. That's a single matrix multiplication: Q times K transposed. The result is a five by five grid of scores. Row i, column j, says how well token i's question matches token j's key.

5. Then divide every score by the square root of the key size. Without this, scores grow with the vector length, the softmax gets too extreme, and training struggles.

6. Now the causal mask. A token must not see the future, so every score above the diagonal is set to minus infinity. After the next step, those become exactly zero.

7. Softmax turns each row into weights. Exponentiate every score, then divide by the row's total. Every row is now positive, and adds up to one.

8. Finally, multiply the weights by V. Each token's output is a weighted blend of the values it attends to. The first token can only see itself, so it simply gets its own value back.

9. Put together, it's one line: softmax of Q times K transposed, over the square root of d, times V. Every token, in parallel, in a handful of matrix multiplications. That's a big reason transformers run so well on GPUs.

10. In NumPy, it's just as short. Project. Score and scale. Mask the future. Softmax. And mix the values.

11. One attention layer asks one kind of question. But a token might want to know several things at once. Next time: multi-head attention.

## V07 · Multi-Head Attention

*Many questions, asked at the same time* · 2:00

1. One attention head asks one kind of question. But when the model reads the word sat, it might want to know several things at once. Who is sitting? Where? What just happened before?

2. So instead of one big attention, we run several smaller ones, side by side. Each one is called a head, and each has its own query, key and value matrices.

3. Here's the trick: the heads share out the vector. GPT two small uses twelve heads on its seven hundred sixty-eight numbers, so each head works with sixty-four of them.

4. Because each head learns its own matrices, each can learn a different pattern. In trained models, researchers have found heads that look at the previous token, heads that connect a verb to its object, and heads that spot a repeated phrase and predict how it continues. Many others are much harder to interpret.

5. Here's an illustration. On the word sat, one head looks at cat, the one doing the sitting. Another looks just one step back. A third spreads its attention widely, over the whole sentence.

6. Each head produces its own sixty-four numbers. We glue them back together, into one vector of seven hundred sixty-eight.

7. Then one more learned matrix, the output projection, mixes what all the heads found, and the result is added back to the token's vector.

8. The nice part: twelve heads of size sixty-four cost about the same as one head of size seven sixty-eight. We get many points of view, for the price of one.

9. In code, we split the vectors into heads, run attention on every head at once, then merge the heads and apply the output projection.

10. Attention lets tokens share information. But once a token has gathered its context, it needs to work with it on its own. That's the job of the other half of every layer: the MLP. That's next.

## V08 · The MLP: Where Facts Live

*The part of the model that thinks on its own* · 2:16

1. Attention moves information between tokens. The MLP works on each token by itself, one at a time, with the same weights at every position. And it's where much of the model's factual knowledge seems to live.

2. MLP stands for multi-layer perceptron. Here, it's just three steps. Expand the vector to four times its size. Apply a simple nonlinear function. Then project it back down.

3. In GPT two small, that's seven hundred sixty-eight numbers, expanded to three thousand and seventy-two, and back down to seven hundred sixty-eight.

4. Each of those three thousand hidden numbers is a neuron. A neuron takes a dot product with the token's vector, so it measures how much the vector points in one particular direction. You can think of it as a question. Is this about animals? Is this the end of a sentence?

5. The nonlinear function is usually GELU. Large positive inputs pass through almost unchanged, and negative ones are squashed close to zero. So a neuron stays quiet, unless its pattern really shows up.

6. Without that bend, the two matrix multiplications would collapse into a single one, and stacking layers would add nothing. The nonlinearity is what lets the network learn more than straight lines.

7. Then the down projection turns each active neuron into a push in some direction. A simplified picture: if a neuron that detects Eiffel Tower fires, it might add a direction that means Paris. Researchers have found that facts like this are largely recalled in the MLP layers.

8. The MLP also holds most of each layer's weights. In GPT two small, every layer's MLP has about four point seven million weights, twice as many as its attention.

9. In code, it's two matrix multiplications with GELU in between: expand and bend, then project back.

10. So each layer has two halves: attention, where tokens talk, and the MLP, where each token thinks. Next time, we'll wire them together into a transformer block, and stack it.

## V09 · The Transformer Block

*Attention and MLP, wired together and stacked* · 1:58

1. We now have both halves: attention, where tokens share information, and the MLP, where each token processes it. Let's wire them together.

2. The central idea is the residual stream. Picture each token's vector flowing along a stream, from its embedding at the start, to the prediction at the end.

3. Attention and the MLP don't replace the vector. They read from the stream, compute something, and add their result back. X plus attention of X. Then, X plus MLP of X.

4. This matters. Adding keeps the original information around, and it gives training a direct path back through the network. It's a big part of what makes deep stacks of layers trainable.

5. Before each half, there's a layer norm. It rescales the vector so its numbers have a mean of zero and a standard deviation of one, then applies a learned scale and shift. It keeps the numbers in a healthy range as they flow through many layers.

6. And that's a transformer block. Layer norm, attention, add. Layer norm, MLP, add.

7. Now stack it. GPT two small has twelve blocks. The largest GPT two has forty-eight. Every block has its own weights, and each one refines the vectors a little more.

8. Researchers see a rough pattern. Earlier layers tend to handle local things, like grammar and nearby words. Later layers deal with more abstract meaning, and with predicting the next token.

9. In code, a block is two lines: add attention of the normalized input, then add the MLP. The layer norm is a few more lines. And the whole model is just a loop over the blocks.

10. After the last block, each token's vector holds the model's view of what comes next. But it's still a vector. Next time: turning it back into words.

## V10 · From Vectors Back to Words

*Logits, softmax, and choosing the next token* · 2:14

1. After the last transformer block, we have one vector per token. To predict what comes after: the cat sat on the, we only need the last one, the vector sitting on the word: the.

2. First, one last layer norm. Then we multiply by the unembedding matrix, which has one column for every token in the vocabulary. The result is one score per token: fifty thousand two hundred fifty-seven numbers, called logits. In GPT two, this matrix is simply the embedding matrix again, reused.

3. Each logit is a dot product: how well the final vector lines up with that token's direction. The better the match, the higher the score.

4. Logits can be any number, positive or negative. Softmax turns them into probabilities. Exponentiate each one, then divide by the total. Now every token has a probability, and they add up to one.

5. The simplest choice is greedy: always take the most likely token. But that tends to be repetitive and dull.

6. Instead, we sample, and we control how adventurous that is with temperature: divide the logits by a number before the softmax. A low temperature sharpens the distribution toward the top choice. A high temperature flattens it, and rarer words get a chance.

7. Two more tricks keep sampling sensible. Top k keeps only the k most likely tokens. Top p keeps the smallest set whose probabilities add up to p, say ninety percent. Everything else is cut before we sample.

8. During training, every position predicts its own next token at the same time, and each prediction is checked against the real next token. So one sentence gives the model many lessons at once.

9. In code: compute the logits and divide by the temperature. Keep the top k. Softmax. And sample one token.

10. And that closes the loop from video one: predict, pick, append, repeat. We've now seen the whole forward pass. But all those matrices started out as random noise. How do they learn? Next time: training.

## V11 · Training: Learning from Mistakes

*Loss, gradients, and millions of tiny nudges* · 2:38

1. A freshly created model is pure noise. Every weight is random, and its predictions are gibberish. Training is how those numbers become useful.

2. The recipe is simple. Take a huge amount of text. Hide the next token, ask the model to predict it, and compare its guess with the truth. The text itself provides the answers, so nobody has to label anything.

3. We score each prediction with cross-entropy loss: the negative log of the probability the model gave to the correct token. If it gave mat a probability of ninety percent, the loss is about zero point one. If it gave it just one percent, the loss is about four point six.

4. We average this loss over every position in a batch of text. Lower is better. Training is simply the search for weights that make the loss small.

5. Picture the loss as a landscape, where every point is one setting of all the weights. We want to find a deep valley. But there are millions of dimensions, so we can't just look around. We have to feel our way downhill.

6. The gradient tells us, for every single weight, which direction makes the loss go up, and how steeply. So we step the other way. Every weight moves a tiny bit against its gradient. That's gradient descent.

7. The size of that step is the learning rate. Too small, and training takes forever. Too large, and we overshoot the valley and bounce around.

8. But how do we get the gradient for millions of weights at once? With backpropagation. It applies the chain rule from calculus, starting at the loss and working backwards through every layer, reusing results along the way. A backward pass costs only about twice as much as a forward pass.

9. Then we repeat. A batch of text, a forward pass, the loss, a backward pass, a small step. Over and over, across billions of tokens. In practice, we use a smarter update rule called Adam, which adapts the step size for each weight.

10. In code, a library like PyTorch does the calculus for us. Forward pass. Loss. Backward pass. Step.

11. Here's a real loss curve, from the tiny model we'll build next time. It starts at about four point four: no better than guessing. It falls fast at first, then slower, and ends near one point six, measured on text it never trained on.

12. We now have every piece. Next time, we'll put them all together, write a tiny GPT from scratch, train it, and watch it learn to write.

## V12 · Build a Tiny GPT

*Every piece, in about a hundred lines of code* · 2:19

1. Over eleven videos, we've built every piece of a GPT. Now let's put them all together, in about a hundred lines of Python, and train one on an ordinary computer.

2. Our training text is about a million characters of Shakespeare's plays. To keep things tiny, every character is a token. The whole vocabulary is just sixty-five symbols.

3. So the tokenizer is two small dictionaries: from characters to IDs, and back again.

4. The model starts with two embedding tables, one for tokens and one for positions, added together. Just like videos three and four.

5. Then attention: queries, keys and values from one linear layer, split across four heads, with the causal mask, and the softmax.

6. The MLP expands to four times the width, applies GELU, and projects back. And the block wires them onto the residual stream, with layer norms. We stack four blocks.

7. Finally, one last layer norm, and a linear layer that turns each vector into sixty-five logits, one per character.

8. That's the whole model: eight hundred eighteen thousand parameters. GPT two small has a hundred and twenty-four million. The largest models today have hundreds of billions.

9. Training is the loop from last time. Grab random chunks of text, predict every next character, compute the loss, backpropagate, and step with Adam. Five thousand steps took under an hour, on an ordinary CPU.

10. Before training, here's what it writes. Random characters. Pure noise.

11. After just two hundred fifty steps, it has learned which letters are common, where the spaces go, and that lines are short. The words are still made up.

12. And after five thousand steps: real words, character names, and the shape of a play. It isn't Shakespeare, but it learned all of this from nothing but predicting the next character.

13. That's a GPT, built from scratch. Real models use the same recipe, with far more data, far more layers, and far more compute. In two bonus videos, we'll see how they generate text quickly, and how a raw GPT becomes a helpful chatbot.

## V13 · Making It Fast: the KV Cache

*Why generation doesn't start over for every token* · 2:07

1. Generating text means running the model once for every new token. Done naively, each step processes the whole text again, from the very first token. For a long answer, that's a lot of repeated work.

2. But look at attention with the causal mask. Earlier tokens never look at later ones. So when a new token arrives, nothing about the old tokens changes. Their keys and values are exactly the same as last time.

3. So we save them. That's the KV cache: every time we compute a token's key and value, in every layer, we store them.

4. When the next token arrives, we only compute its own query, key and value. Its query is compared with all the cached keys, and the weights mix the cached values. One new row, instead of redoing the whole grid.

5. Each step now processes one token, instead of the entire text. That's a big part of why chatbots can stream their answers to you so quickly.

6. The price is memory. The cache holds a key and a value for every token, in every layer. For GPT two small, that's about eighteen thousand numbers per token. For big models and long conversations, the cache can take up gigabytes.

7. That's why generation has two phases. Pre-fill: read the whole prompt at once, and fill the cache. Then decode: one token at a time, reusing the cache.

8. Because the cache gets so large, modern models shrink it. One common trick lets several query heads share the same keys and values. It's called grouped-query attention.

9. In code: compute the new token's query, key and value. Append the key and value to the cache. Then attend over everything cached so far. No mask is needed, because the cache only holds the past.

10. One more bonus video to go. A GPT trained on internet text is a brilliant autocomplete, but it isn't a helpful assistant. Next time: from GPT to chatbot.

## V14 · From GPT to Chatbot

*How a text predictor learns to be an assistant* · 2:18

1. A model trained only to predict the next token of internet text is called a base model. It's a powerful autocomplete. Ask it a question, and it might answer. Or it might continue with three more questions, as if it were writing a quiz.

2. This first stage is called pretraining. It's where almost all the computing goes, and where the knowledge comes from: trillions of tokens of text.

3. Stage two is supervised fine-tuning. We keep training the same model, with the same next-token loss, but now on examples of conversations: a user asks, and an assistant answers helpfully. The model learns the format, and the style.

4. Conversations are turned into text, with special tokens that mark who is speaking. The model is trained to predict the assistant's turns. To the model, a chat is still just one long document to continue.

5. Stage three teaches judgment. People compare two answers from the model, and pick the better one. From many thousands of these comparisons, the model learns what better means.

6. One way to use them is RLHF: reinforcement learning from human feedback. Train a reward model to predict which answers people prefer, then nudge the chatbot toward answers that score higher. A simpler method, called DPO, learns from the preference pairs directly.

7. Many labs also use AI feedback, guided by written principles, and reinforcement learning on problems with checkable answers, like math and code.

8. Through all of this, the machinery doesn't change. It's the same transformer: the same embeddings, attention, and next-token prediction. Only the data and the training signal change.

9. So here's the whole journey. Text becomes tokens. Tokens become vectors, with positions. Attention lets them talk, and MLPs let them think, stacked in blocks. The final vector becomes probabilities, we pick a token, and repeat. Training shapes every number, and fine-tuning turns it into an assistant.

10. That's how an LLM works, from scratch. Thanks for watching.
