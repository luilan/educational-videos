# Foundations: narration

Narration text for every video, generated from the `*_script.py` files (one paragraph per section).

## F01 · Vectors: Lists of Numbers as Arrows

*The basic building block of every LLM* · 1:56

1. Everything an LLM does, it does with vectors. Every token, every hidden state, every prediction starts as one. So what exactly is a vector?

2. At its simplest, a vector is just a list of numbers. Like this one: two, one. Or this one, with four numbers. How many numbers it has is called its dimension.

3. With two numbers, we can draw it. The first number says how far to go right, the second, how far to go up. So the vector two, one is an arrow from the origin to that point.

4. Three numbers give an arrow in three-dimensional space. Beyond that, we can't draw it, but the math works exactly the same. GPT two uses vectors with seven hundred sixty-eight numbers.

5. To add two vectors, add their numbers, position by position. Two, one, plus one, two, is three, three. As arrows, that means placing one arrow at the tip of the other.

6. Multiplying by a single number, called a scalar, stretches the arrow. Two times two, one, is four, two: the same direction, twice as long. A negative number flips it around.

7. The length of a vector comes from Pythagoras: square each number, add them up, and take the square root. For three, four, that's five.

8. In an LLM, directions can come to mean something. One direction might lean toward animals, another toward plurals. A token's vector is a point in this space of meaning.

9. In NumPy, a vector is an array. Adding, scaling, and measuring length take one line each.

10. You'll see vectors in almost every episode: as embeddings in episode three, added to positions in episode four, and flowing along the residual stream in episode nine. Next: how to compare two vectors, with the dot product.

## F02 · The Dot Product

*One number that says how much two vectors agree* · 1:51

1. Given two vectors, one of the most useful questions is: how much do they point the same way? The dot product answers that, with a single number.

2. The recipe: multiply matching numbers, then add everything up. For three, one, and two, two: three times two is six, one times two is two, and six plus two is eight.

3. It works in any number of dimensions, as long as both vectors have the same length. Seven hundred sixty-eight multiplications, then one big sum.

4. Geometrically, it measures agreement. Arrows pointing the same way give a large positive number. At right angles, the dot product is exactly zero. Pointing in opposite directions, it goes negative.

5. Here's the picture: project one arrow onto the other, like a shadow. The dot product is the length of that shadow, times the length of the other arrow.

6. That gives a second formula: the length of a, times the length of b, times the cosine of the angle between them.

7. If we only care about direction, we divide by both lengths. What's left is the cosine of the angle, called cosine similarity. One means the same direction, zero means unrelated, and minus one means opposite.

8. Language models use dot products everywhere. Attention compares a query with a key. The output layer compares a vector with every word. And each MLP neuron is a dot product too.

9. In NumPy, the dot product is the at sign. Divide by the two lengths, and you get cosine similarity.

10. You'll meet the dot product in episodes three, five, six, eight, and ten. Next: doing lots of dot products at once, with matrices.

## F03 · Matrices: Many Dot Products at Once

*The table of numbers behind every layer* · 2:01

1. Language models compute billions of dot products. Writing them one by one would be hopeless. Matrices let us do many of them at once.

2. A matrix is a table of numbers, with rows and columns. This one has two rows and three columns, so we call it a two by three matrix.

3. To multiply a matrix by a vector, take the dot product of each row with the vector. Each row gives one number, so the result is a new vector, with one entry per row.

4. So a matrix turns one vector into another. In two dimensions you can watch it happen: a matrix can stretch space, rotate it, or shear it, and every arrow moves along.

5. The shapes must line up. A two by three matrix needs a vector with three numbers, and gives back two. In general, an m by n matrix turns n numbers into m numbers.

6. Multiplying two matrices is the same idea, repeated: every row of the first, dotted with every column of the second. A five by four matrix, times a four by four matrix, gives a five by four matrix.

7. That's exactly how LLMs use it. Stack our five tokens as the rows of a matrix, multiply by one weight matrix, and every token is transformed in a single step.

8. One more operation: the transpose. It flips a matrix over its diagonal, so rows become columns. Q times K transposed compares every query with every key, all at once.

9. In NumPy, the at sign multiplies matrices too, and dot T gives the transpose. Always check the shapes.

10. Matrices power almost every step from episode five to episode ten: queries, keys and values, attention scores, the MLP, and the final logits. Next: why stacking matrices isn't enough, and the bend that fixes it.

## F04 · Straight Lines and Bends

*Linear functions, and why networks need a bend* · 1:49

1. A function takes an input and gives an output. The simplest kind is linear: double the input, and the output doubles too. Its graph is a straight line through the origin.

2. Multiplying by a matrix is linear too. Scale the input vector, and the output scales the same way. Add two inputs, and their outputs add.

3. Here's the catch. Apply one matrix, then another, and the result is just a single matrix: their product. Stack a hundred linear layers, and they still collapse into one.

4. And a single linear map can only do so much. It can stretch and rotate, but it can't bend. Try to separate the points inside this circle from the points outside it with a straight line, and you can't.

5. The fix is to add a bend between the layers: a nonlinear function, applied to every number separately.

6. The simplest one is called ReLU. It keeps positive numbers, and replaces negative ones with zero. A tiny change, but now stacked layers can't collapse.

7. Transformers usually use GELU, a smooth version of ReLU. Large positive inputs pass through, large negative ones fade to zero, with a gentle curve in between.

8. With bends between the layers, a network can build curved shapes out of many simple pieces, and approximate almost any function.

9. In code, ReLU is one line. A layer is a matrix multiplication, then the bend, then the next matrix.

10. You'll see this in episode eight, the MLP: expand, bend with GELU, and project back. And in episode nine, where the blocks are stacked. Next: exponentials and logarithms.

## F05 · Exponentials and Logarithms

*Fast growth, and the function that undoes it* · 1:59

1. Two functions show up again and again in LLMs: the exponential, and its opposite, the logarithm. Let's build an intuition for both.

2. Exponential growth means multiplying, not adding. Two, four, eight, sixteen: doubling every step. After just ten doublings, you're past a thousand.

3. In machine learning, the base is usually the number e, about two point seven one eight. The function e to the x has two properties we care about.

4. First, it's always positive. Even e to the minus ten is a tiny positive number, less than one ten-thousandth. It never reaches zero, and it never goes negative.

5. Second, it exaggerates differences. The inputs one and three differ by just two. But e to the one is about two point seven, and e to the three is about twenty. Now one is more than seven times the other.

6. The logarithm undoes the exponential. The natural log of e to the x is just x. The log of one is zero, and as numbers shrink toward zero, the log dives toward minus infinity.

7. That makes logs perfect for probabilities. A probability of zero point nine has a log close to zero. A probability of zero point zero one has a log of about minus four point six. The less likely, the more negative.

8. Logs also turn multiplication into addition. And they give us log scale charts, where each step is ten times bigger than the last, so a thousand and a billion fit on the same axis.

9. In NumPy, it's np dot exp and np dot log.

10. You'll see the exponential inside softmax, in episodes six and ten, and the logarithm in the loss of episode eleven, and the log scale chart of episode twelve. Next: probability and sampling.

## F06 · Probability and Sampling

*Distributions, weighted dice, and randomness* · 1:45

1. An LLM never simply knows the next word. It produces probabilities. So let's be clear about what a probability is.

2. A probability is a number between zero and one: how likely something is. Zero means impossible, one means certain, and a half means it happens about half the time.

3. A probability distribution spreads one whole unit of belief across all the options. For the word after: the cat sat on the, maybe mat gets zero point four, and floor, zero point two five. Together, all the probabilities must add up to exactly one.

4. Sampling means picking one option at random, according to those probabilities. Think of a spinner, where each word gets a slice sized by its probability.

5. Spin it many times, and each word comes up about as often as its probability says. Mat about forty percent of the time, floor about a quarter of the time.

6. Always choosing the biggest slice is predictable. Sampling adds variety, which is why the same prompt can give different answers.

7. Computers make randomness with pseudo-random number generators. Start one with the same seed, and you get the same sequence of random choices every time. That keeps experiments reproducible.

8. In NumPy, a random generator with a seed can sample one word, or many, from a list of probabilities.

9. You'll see this in episode one, where the model predicts a distribution, in episode ten, where we sample from it, and in episode fourteen. Next: softmax, the function that turns scores into probabilities.

## F07 · Softmax, Properly

*From any scores to probabilities* · 1:52

1. A neural network outputs raw scores, called logits. They can be any number: large, small, or negative. But we need probabilities: positive, and adding up to one. Softmax does that conversion.

2. Step one: exponentiate every score. Thanks to e to the x, every result is positive, even for negative scores.

3. Step two: divide each result by their total. Now they add up to exactly one. That's the whole recipe.

4. Take the scores three, two, and zero. Exponentiated, they're about twenty, seven point four, and one. The total is about twenty-eight and a half. Divided out, that's about seventy percent, twenty-six percent, and three and a half percent.

5. Why the name? A hard max would give all the probability to the top score. Softmax leans toward the top score, but still gives the others a share: a soft version of the max.

6. Only the differences between scores matter. Add ten to every score, and the probabilities don't change at all.

7. We can also divide the scores by a temperature before the softmax. Below one, the differences grow, and the top choice dominates. Above one, the distribution flattens out.

8. One practical trick. E to the power of a thousand is too big for a computer. So code subtracts the largest score first. Since only differences matter, the answer is identical, and nothing overflows.

9. In code, that's three lines: divide by the temperature, exponentiate after subtracting the max, and normalize.

10. Softmax turns attention scores into weights in episodes five and six, and logits into word probabilities in episode ten. Next: averages and spread.

## F08 · Averages and Spread

*Mean, standard deviation, and keeping numbers tame* · 1:56

1. Deep networks pass numbers through dozens of layers. If they drift too big or too small, training breaks. Two simple statistics help keep them in check: the mean, and the standard deviation.

2. The mean is the average: add the numbers up, and divide by how many there are. For two, four, six, and eight, the mean is five.

3. The standard deviation measures spread: how far the numbers typically are from the mean. Take each distance from the mean, square it, average those squares, and take the square root.

4. For two, four, six, and eight, the distances are minus three, minus one, one, and three. The squares average to five, so the standard deviation is the square root of five: about two point two four.

5. To normalize, subtract the mean, and divide by the standard deviation. Our numbers become about minus one point three four, minus zero point four five, zero point four five, and one point three four. The mean is now zero, and the spread is one.

6. That's exactly what layer norm does to every token's vector, followed by a learned scale and shift.

7. Spread also explains a detail in attention. Add up many random products, as a dot product does, and the spread of the total grows with the square root of how many you added.

8. So with vectors of sixty-four numbers, dot products spread about eight times wider. Dividing by the square root of the key size brings them back to a spread of about one, which keeps the softmax well behaved.

9. In NumPy, mean and std are built in, and normalizing is one line.

10. You'll see normalizing in the layer norm of episode nine, and the square root of d in the attention formula of episode six. Next: waves and rotations.

## F09 · Waves and Rotations

*Sine, cosine, and turning arrows* · 1:48

1. Episode four encodes positions with waves, and with rotations. Let's see where sine and cosine come from.

2. Picture a point moving around a circle of radius one. At any angle, its height is the sine of that angle, and its horizontal position is the cosine.

3. Let the angle keep growing, and plot the height as it goes. You get the sine wave: up to one, down to minus one, repeating forever. The cosine is the same wave, shifted by a quarter turn.

4. Angles are usually measured in radians. A full turn is two pi, about six point two eight. So the wave repeats every six point two eight units.

5. Frequency is how fast the wave repeats. Multiply the input by a bigger number, and the wave oscillates faster. Multiply by a smaller number, and it slows down.

6. Combine several waves with different frequencies, and every position gets its own combination of values, like the hands of a clock.

7. Sine and cosine also rotate arrows. To turn a 2-D vector by an angle, mix its two numbers with the cosine and sine of that angle. The length never changes, only the direction.

8. Rotate one arrow by two steps, and another by five, and the angle between them is three steps, no matter where you started. Relative position becomes a relative angle.

9. In code, a rotation is a two by two matrix built from the cosine and the sine. The length stays exactly one.

10. You'll see sine waves marking positions, and rotations as rope, in episode four. Next: slopes and gradients, the math of learning.

## F10 · Slopes and Gradients

*How a model knows which way to adjust* · 2:00

1. Training means adjusting weights to reduce a loss. To know which way to adjust them, we need one idea from calculus: the slope.

2. The slope of a curve tells you how much the output changes when you nudge the input a little. A steep slope means a big change. A flat one, almost no change.

3. Take the function x squared. At x equals three, nudge x by a tiny amount, and the output grows about six times as fast. That rate, six, is the derivative. In general, the derivative of x squared is two x.

4. The slope also tells us which way is downhill. At x equals three, the slope is positive, so to make the output smaller, we step to the left: against the slope.

5. That's gradient descent, in one dimension. The new x is the old x, minus the learning rate times the slope. With a learning rate of zero point one, three becomes two point four. Repeat, and x slides toward the minimum at zero.

6. A real model has millions of weights, not one. The gradient collects one slope for each weight, into a single vector. It points uphill, so we step the opposite way.

7. Layers are functions inside functions. The chain rule says their slopes multiply. If y changes three times as fast as x, and z changes twice as fast as y, then z changes six times as fast as x.

8. Backpropagation applies the chain rule layer by layer, from the loss back to every weight, so one backward pass gives the whole gradient.

9. In code, gradient descent is a short loop: compute the slope, and step against it.

10. You'll see all of this in episode eleven: the landscape of the loss, gradients, the learning rate, and backpropagation. Next: a quick tour of NumPy, so you can read every snippet in the series.

## F11 · NumPy in Three Minutes

*Just enough to read the code in the series* · 2:03

1. The code in this series uses NumPy, Python's library for fast math on arrays of numbers. Here's just enough to read every snippet.

2. An array holds numbers in a grid. A vector is a one-dimensional array. A matrix is two-dimensional. Every array has a shape: this one is five by four, five rows of four numbers.

3. Arithmetic works on every element at once. Add two arrays of the same shape, and matching elements are added. Multiply by two, and every number doubles.

4. The at sign is matrix multiplication: dot products of rows with columns. A five by four array, at a four by four array, gives five by four.

5. Many functions take an axis. Summing with axis equals minus one adds along the last dimension, giving one total per row. Keep dims equals true keeps that result as a column.

6. That column lines up with the rows because of broadcasting. Divide a five by five array by a five by one column, and each row is divided by its own number. That's exactly how softmax normalizes every row.

7. Reshape regroups the same numbers into a new shape. Seven hundred sixty-eight numbers can become twelve heads of sixty-four. Transpose swaps the axes.

8. Finally, masks. NumPy's upper-triangle function builds a triangle of ones above the diagonal. Use it to pick out elements, and set them to minus infinity. That's the causal mask from episode six.

9. Here it all is in a few lines: shapes, the at sign, a sum along an axis, a reshape, and a mask.

10. That's the toolkit: vectors, dot products, matrices, bends, exponentials, probability, softmax, statistics, waves, gradients, and NumPy. You're ready for the main series. Next up: episode one, what is an LLM?

## F12 · Neural Networks in Three Minutes

*Neurons, layers, weights, and learning* · 2:00

1. The main series keeps saying: neural network. So what is one, really? Strip away the hype, and it's a function, built from very simple parts.

2. The basic part is a neuron. It takes some numbers in, multiplies each one by a weight, adds them up with a bias, and passes the result through a bend, like the ReLU we met earlier.

3. Say the inputs are two and three, the weights are zero point five and one, and the bias is minus one. That's one, plus three, minus one: three. ReLU keeps it, so this neuron fires, with a value of three.

4. Put many neurons side by side, all reading the same inputs, and you have a layer. Their weights together form a matrix, so a whole layer is one matrix multiplication, plus the bend.

5. Stack layers, and the outputs of one become the inputs of the next. Early layers find simple patterns. Later layers combine them into more complex ones.

6. The weights and biases are called parameters. They're the only thing that changes when a network learns. Our tiny GPT had about eight hundred thousand of them. Big models have billions.

7. A network is used in two ways. Inference means running it: inputs flow forward, and out comes a prediction. Training means adjusting it: compare the prediction with the right answer, and nudge every parameter to do a little better.

8. Nobody writes the rules inside. We choose the shape: how many layers, and how many neurons. The training data does the rest.

9. In code, a neuron is a dot product, a bias, and a ReLU. A layer does many neurons at once.

10. You'll see neural networks throughout the series: the MLP in episode eight, training in episode eleven, and a complete network in episode twelve. Next: PyTorch, and how it computes gradients for us.

## F13 · PyTorch and Autograd

*How one line computes every gradient* · 1:51

1. In episode eleven, one line did all the calculus: loss, dot backward. Let's see what that line actually does.

2. PyTorch works with tensors: arrays of numbers, just like NumPy arrays. The difference is that a tensor can remember how it was computed.

3. Mark a tensor with requires grad equals true, and PyTorch starts recording every operation that uses it.

4. Take x equals three, and compute y equals x squared, plus two x. As it computes, PyTorch builds a graph: x goes into a square, and into a times two, and the two results are added. Y comes out as fifteen.

5. Call y dot backward, and PyTorch walks that graph in reverse, applying the chain rule at every step. The result lands in x dot grad.

6. The slope of x squared plus two x is two x plus two. At x equals three, that's eight. And x dot grad says exactly eight.

7. A real model is the same thing, just bigger. The graph has millions of steps, and the loss depends on millions of weights, but one backward call fills in the gradient of every weight.

8. Then an optimizer, like Adam, reads those gradients and updates the weights. Zero grad clears the old gradients first, because PyTorch adds new gradients to whatever is already there. Forget it, and eight plus eight becomes sixteen.

9. In code: create a tensor that requires gradients, compute with it, call backward, and read the gradient.

10. You'll see this in the training loop of episode eleven, and in the tiny GPT of episode twelve. Next: why we always keep some data aside, and what the validation curve tells us.

## F14 · Train vs Validation Data

*How we know a model really learned* · 1:53

1. Episode eleven showed two loss curves: one for training, and one for validation. Why two? Because a low training loss, on its own, can fool us.

2. A model could simply memorize its training text, word for word. It would score perfectly on that text, and be useless on anything new.

3. So before training, we split the data. Most of it, the training set, is used for learning. A slice we never train on, the validation set, is kept aside to test with.

4. For our tiny GPT, ninety percent of the Shakespeare text, about a million characters, was for training. The last ten percent, about a hundred and eleven thousand characters, was only ever used to measure.

5. During training, we measure the loss on both. The training loss says how well the model fits what it has seen. The validation loss says how well it does on text it has never seen.

6. Our tiny GPT ended at about one point three on training text, but about one point six on validation text. That gap is normal: a model usually does a little better on what it studied.

7. But if the gap keeps growing, while the validation loss stops falling, or even rises, the model is overfitting: memorizing instead of learning. That's the time to stop, or to find more data.

8. Careful projects keep a third slice, the test set, untouched until the very end, for one final, honest score.

9. In code, the split is two lines: the first ninety percent for training, the rest for validation.

10. You'll see both curves in episode eleven, and this exact split in the tiny GPT of episode twelve. That completes the foundations. Next up: episode one, what is an LLM?
