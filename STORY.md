# Inside the Machine: A Flight Through a Neural Network

A guided, scroll-driven tour through a small neural network as it recognizes a handwritten **7**.

**The network we fly through** (used consistently in narration and scene):
784 inputs (28×28 pixels) → 32 hidden neurons → 16 hidden neurons → 10 outputs.
ReLU activations in the hidden layers, softmax at the output. About 26,000 weights and biases in total.

**Format:** each beat has an ID, a stage direction in `[brackets]` (what the 3D scene does), and 1–3 sentences of narration. Beats appear one at a time as the viewer scrolls.

---

## Chapter 1: The Dark

### 1.1
[Total black. Far ahead, a single faint point of cool blue light pulses slowly. The camera is still.]

Look closely. There, in the dark: one faint light. That's where we're going.

### 1.2
[The camera begins a slow drift forward. A sparse, very dim starfield fades in at the edges of the screen.]

Somewhere, someone wrote the number seven on a scrap of paper. A quick stroke across the top, then a slash down and to the left. If you saw it, you'd know it instantly.

### 1.3
[The light ahead flickers once, as if listening.]

But try to explain *how* you knew. Not "it looks like a seven," but the actual rule. What, exactly, makes a seven a seven?

### 1.4
[The camera keeps drifting. The light pulses patiently.]

Try a thought experiment. Describe a seven over the phone to someone who has never seen a written number. You'd run out of words long before the world ran out of ways to write a seven.

### 1.5
[Three faint ghost sevens drift past in the dark, each written differently: one crossed, one with a curled top, one lazy and nearly vertical. They dissolve.]

Maybe: a flat line on top, a diagonal coming down. But some people cross their sevens. Some curl the top. Some write it so lazily it's nearly a one.

### 1.6
[The ghosts are gone. The camera keeps drifting; the light grows slightly.]

Every rule you write has exceptions, and every exception needs another rule. People spent years trying to hand-write rules like these, and it was brutally hard.

### 1.7
[The light brightens into a soft glow. The starfield thickens a little.]

So here's a stranger idea. What if, instead of giving a machine the rules, we showed it tens of thousands of examples and let it work out its own?

### 1.8
[The camera accelerates gently. Behind the glow, faint structure appears: rows of tiny dim points, and beyond them, scattered dark spheres.]

That's what a neural network does. And tonight we're going inside one: a small one, already trained, in the instant it looks at a handwritten seven.

### 1.9
[A barely visible web of threads catches the light for a moment, then fades.]

Despite the name, it isn't a brain. It's arithmetic: lots of simple calculations, wired together. The surprise is what that arithmetic can do.

### 1.10
[The glow fills the center of the screen. The camera holds for a beat, then pushes forward.]

You're the camera. I'm your guide. Keep scrolling, and we'll fly in.

---

## Chapter 2: The Input Layer

### 2.1
[The glow resolves into a flat square grid of 784 dim points, 28 across and 28 down, floating face-on. The camera approaches from far away. Palette: pale cyan on near-black.]

Here's the front door: a grid, 28 points across and 28 down. That's 784 points, and each one is a single pixel of the image.

### 2.2
[The camera slows. The grid hangs silent, every point equally dim.]

This is all the network will ever see of that seven. No paper, no ink, no pen. Just 784 numbers. Your eye, by comparison, has over a hundred million light-sensitive cells.

### 2.3
[One pixel near the center brightens to full white, then fades to a mid-grey, then goes dark, demonstrating the range.]

Each pixel holds a brightness between 0 and 1. Zero is blank paper. One is solid ink. The values in between are the soft grey edges of a stroke.

### 2.4
[Pixels light up in drawing order: first a horizontal bar across the top, then a long diagonal sweeping down to the lower left. The digit 7 appears in cyan-white, with softer, dimmer pixels along its edges.]

Watch. Here comes the seven: the top bar, then the long diagonal. The edges glow softer, because a pen stroke never lands neatly on a pixel grid.

### 2.5
[The camera glides in until the 7 overfills the screen. Individual pixels become separate glowing points with dark gaps between them.]

Up close, the seven dissolves. From here you can't see a seven at all, only individual lights: a few bright, most dark.

### 2.6
[The camera hovers among the pixels. Each lit pixel pulses independently, out of sync with its neighbors.]

That's exactly the network's situation. It never gets "the shape." Each input knows only its own brightness, and nothing about its neighbors.

### 2.7
[The grid unrolls row by row into one long line of 784 points stretching off into the fog, holds for a moment, then rolls back into a square.]

Here's a strange fact. To this kind of network, the grid is really one long list of 784 numbers. Shuffle the pixel order, the same way for every image, and it could learn just as well.

### 2.8
[A few random pixel pairs are linked by faint dashed lines, then the lines fade.]

So "nearby" means nothing to it at the start. Whatever it figures out about how pixels relate, it has to learn from examples.

### 2.9
[The camera pivots around the edge of the grid. Behind it, in the distance, a loose cluster of unlit spheres waits in the fog.]

So how do you get from 784 separate lights to a single idea, "seven"? You pass them along to something that can combine them. Let's follow.

---

## Chapter 3: Neurons Wake Up

### 3.1
[The camera passes through the plane of the grid. Ahead, 32 spheres hang in a gently curved formation, dark blue and barely visible. Lighting shifts to a deeper indigo.]

This is the first hidden layer, "hidden" simply because its values aren't the input or the final answer. Thirty-two neurons, waiting.

### 3.2
[The camera closes in on a single neuron. The others blur into the fog.]

Let's pick one and look closely. An artificial neuron is surprisingly simple. It does three things.

### 3.3
[Faint threads appear, running from all 784 pixels behind us to this one neuron, like a cone of fine wire.]

First, it listens to every pixel, all 784 of them. Each connection has its own number, called a weight, that says how much that pixel matters to this neuron.

### 3.4
[Threads from the lit pixels of the 7 brighten. Most threads, from dark pixels, stay faint.]

It multiplies each pixel's brightness by its weight and adds everything up. That's the weighted sum. Dark pixels contribute nothing, since zero times anything is zero.

### 3.5
[Some of the bright threads are cyan (positive), some violet (negative). The neuron's inner glow wavers up, then down, as contributions arrive.]

Think of it as a vote. Bright pixels on positive weights push the total up. Bright pixels on negative weights pull it down.

### 3.6
[A small ring around the neuron rotates slightly, like a dial being set.]

Second, it adds one more number: the bias. The bias works like a threshold dial, deciding how easy or hard this neuron is to excite before any pixel has its say.

### 3.7
[The neuron's glow hesitates, then flares from dark blue to warm amber.]

Third, the activation function. This network uses one called ReLU, a very simple rule: if the total is negative, output zero. If it's positive, pass it along unchanged.

### 3.8
[A single amber pulse leaves the neuron and travels forward into the dark.]

That's the whole recipe: weighted sum, plus bias, through the activation. This neuron's total came out positive, so it fired.

### 3.9
[A brief diagram-like flash: a straight line, then the same line bent at zero into a hinge.]

Why bother with that last step? Without it, the whole network, however many layers deep, would collapse into one big weighted sum. That bend at zero is what lets layers build on each other.

### 3.10
[The camera pulls back. Across the layer, about a third of the neurons flare amber at different intensities; the rest stay dark blue.]

Now zoom out. All thirty-two did the same thing, each with its own weights. A few fired strongly, some weakly, many not at all.

### 3.11
[The layer settles into a quiet shimmer.]

About that word "neuron": real brain cells inspired the idea, but they're vastly more complicated. Think of this as a cartoon of a neuron, useful but not literal.

---

## Chapter 4: The Connections

### 4.1
[The camera rises above the gap between the input grid and the first hidden layer. The full web comes into view: over 25,000 threads, most faint, some bright. Palette shifts to cyan and teal.]

Pull back and you can see what joins these layers. Every pixel connects to every neuron in the first hidden layer. That's 784 times 32, more than twenty-five thousand threads.

### 4.2
[Brightness and thickness of threads vary visibly. Cyan threads and dimmer violet threads interweave.]

Each thread is one weight. Brighter, thicker threads carry strong weights; faint ones barely matter. Cyan threads are positive; violet ones are negative.

### 4.3
[The whole web brightens slightly, as if breathing in.]

Here's the heart of it. When people say a network "learned" something, this is what they mean. Everything it knows lives in these numbers: the weights, plus the biases.

### 4.4
[The camera follows one bright thread from a top-bar pixel to its neuron, then pans across hundreds of others.]

But not in any single one. No thread here means "seven," and no thread means "top bar." The knowledge is spread across thousands of weights at once, the way a picture is spread across pixels.

### 4.5
[Small amber pulses begin traveling along the brightest threads, from lit pixels toward the neurons, arriving at different times.]

Now watch the signal move. Each pulse is a pixel's brightness multiplied by a weight, flowing toward the neuron that adds them up.

### 4.6
[Beside one neuron, its 784 incoming weights are laid out as a floating 28×28 heat map: a smudgy patch of cyan surrounded by violet.]

Here's a trick to see what one neuron is looking for. Lay its 784 weights back out on the 28 by 28 grid, and you get a picture of the pattern it responds to most.

### 4.7
[The heat map holds. Its edges are noisy and irregular.]

Often it's a smudgy blob, positive in one region and negative around it: a detector for "ink here, but not there." Often it's hard to put into words at all.

### 4.8
[The camera pulls back to show the input web, and faintly, the smaller webs deeper in.]

Count every weight and bias in this network and you get about 26,000 numbers. That's the entire "mind" of the machine. It would fit in a spreadsheet.

### 4.9
[The web flickers: for a moment the thread brightnesses rearrange completely, then snap back.]

A thought experiment: keep the wiring exactly the same but swap in different weights, and this same structure could recognize letters, or sketches of shoes. The wiring is the stage; the weights are the script.

### 4.10
[The camera turns toward the first hidden layer and starts moving again.]

Nobody chose these values by hand. They started out random. Where they came from is the last stop on our tour. For now, let's go deeper.

---

## Chapter 5: Deeper Layers

### 5.1
[The camera dives through the first hidden layer. Fog thickens. Ahead, 16 neurons sit in a tighter cluster. Palette deepens to blue-violet.]

Through the first hidden layer, and deeper. Sixteen neurons wait here, and each one listens not to pixels but to the thirty-two neurons behind us.

### 5.2
[A web of 512 connections links the two hidden layers, denser-looking but far fewer threads than before.]

Same three steps as before: weighted sum, bias, activation. Only the inputs have changed. Now they're the outputs of the previous layer.

### 5.3
[Amber activity from the first hidden layer pulses forward along the connections. A handful of second-layer neurons ignite.]

This is where stacking layers starts to pay off. Each layer can combine whatever the last one found.

### 5.4
[Ghostly overlays appear: small stroke fragments float near first-layer neurons, larger shapes near second-layer neurons.]

Here's the story people love to tell. The first layer finds small strokes: a short flat edge, a slanted line. The next layer combines strokes into parts, like a long top bar or a diagonal sweep.

### 5.5
[The fragments drift together and assemble into a faint, perfect 7.]

Then the final layer combines parts into whole digits. Top bar, plus diagonal, plus no closed loop anywhere: that's a seven.

### 5.6
[The perfect 7 flickers and breaks apart into noisy, abstract speckle patterns hovering near each neuron.]

I should be honest with you, though. That tidy story is the hope, not a guarantee. In a small network like this one, if you look at what each neuron responds to, you mostly find messy, hard-to-name patterns.

### 5.7
[The camera slowly circles the second layer, passing close to the speckled patterns.]

Nobody told the network to find edges or loops. It found whatever patterns happened to reduce its mistakes. Sometimes those resemble our ideas. Often they don't.

### 5.8
[A brief inset of clean, striped edge-detector patterns appears, then fades.]

Larger networks built specially for images do tend to learn clear edge detectors in their first layers. And the core principle holds here too: each layer builds new features out of the previous layer's features.

### 5.9
[Light from the second layer gathers into streams flowing forward.]

It's a bit like twenty questions, where each round of questions is built from the answers to the last. Except here the answers aren't yes or no; they come in every shade in between.

### 5.10
[Far ahead, ten larger spheres emerge from the fog in a row, still unlit.]

We're nearly at the end. Ahead: ten final neurons, one for each digit, zero through nine.

---

## Chapter 6: The Verdict

### 6.1
[The camera emerges into open space. Ten large spheres stand in a gentle arc, each faintly labeled with a digit, 0 to 9. The lighting is cool and still.]

The output layer. Ten neurons, labeled zero through nine. Each one produces a score: how strongly does the evidence point to this digit?

### 6.2
[Signals from the 16 neurons behind us flow into all ten spheres at once along 160 threads.]

Each listens to all sixteen neurons behind it, using the same weighted sum plus bias. But this time, there's no ReLU at the end.

### 6.3
[Ten thin vertical bars rise beneath the spheres, uneven, then rescale so their heights clearly add up to one full bar.]

Instead, the ten raw scores go through one final step called softmax. It turns them into ten positive numbers that add up to exactly one: each digit's share of the confidence.

### 6.4
[The tallest bar grows sharply while the shorter ones shrink.]

Softmax exaggerates differences. A score that's only somewhat higher than the rest gets a much bigger share. Think less of a fair vote and more of a spotlight swinging toward the leader.

### 6.5
[The spheres brighten in proportion to their probability. Most stay nearly dark. "1" glows faintly. "7" swells into brilliant warm amber; the bloom intensifies across the whole scene.]

And there it is. The seven blazes: ninety-eight percent, for this particular image.

### 6.6
[The camera glides along the arc, past the faint "1" and the dark others.]

Look at the others. The one gets a sliver, about one percent; that makes sense, since a seven is a bit like a one with a hat. The other eight share what's left.

### 6.7
[The camera turns to look back down the whole path: glowing pixels, the two hidden layers, the amber 7, connected by threads of light.]

Look back the way we came: 784 pixels, two hidden layers, about twenty-six thousand multiplications, all done in well under a millisecond. That is what "seeing" means for this network.

### 6.8
[The amber of the 7 pulses once, slightly less steady.]

One warning about that word "confidence." Ninety-eight percent isn't a guarantee. It doesn't promise the network is right 98 times out of 100 when it says this.

### 6.9
[A random scribble flashes onto the input grid; the output layer still lights up, one digit glowing confidently.]

Networks can be confidently wrong. Show this one a random scribble and it will still hand you ten numbers that add up to one, because that's all it can do. It has no answer for "none of the above."

### 6.10
[The amber fades back to blue. Every connection in the network dims to a uniform faint grey.]

So how did it get good enough to be right? Let's rewind, to before it knew anything at all.

---

## Chapter 7: How It Learned

### 7.1
[Scene reset: every connection flickers to a random brightness, a chaotic, evenly dim tangle. The palette desaturates.]

Before training, every weight was random. Same wiring, same seven going in, but what comes out is noise.

### 7.2
[The same 7 flows through. The output layer lights up as a muddle: "3" is brightest at 14%, the rest close behind, "7" at 9%.]

The network says: maybe a three? Fourteen percent. It's guessing, barely better than rolling a ten-sided die.

### 7.3
[A single number appears above the output layer, large and pulsing red-orange: the loss.]

To improve, it first needs to measure how wrong it was. That measure is called the loss: a single number that's big when the network gives the right answer low confidence, and small when it gives it high confidence.

### 7.4
[The "7" sphere, dim at 9%, is highlighted; the loss number swells.]

The right answer was seven, and it gave seven just nine percent. Big loss.

### 7.5
[The scene abstracts: a vast, foggy landscape of hills and valleys forms beneath the camera. A single glowing marker sits high on a slope.]

Now picture every possible setting of the weights as a spot on a landscape, where height is the loss. Training means finding low ground.

### 7.6
[Fog rolls in thick around the marker; only the ground right beneath it is visible.]

But you're standing in thick fog. You can't see the valley. You can only feel which way the ground slopes under your feet, so you take a small step downhill, then feel again.

### 7.7
[The marker steps downhill, pauses, steps again, each step a little shorter as the slope eases.]

That's gradient descent. The gradient tells you, for each weight, which way and how steeply the loss changes if you nudge it. Then you step the opposite way.

### 7.8
[The landscape ripples, as if hinting at far more dimensions than it can show.]

Here's where the picture breaks down: this landscape doesn't have two directions to walk in. It has about 26,000, one per weight and bias. Nobody can picture that; the math doesn't mind.

### 7.9
[The landscape dissolves back into the network. A wave of red-orange light starts at the output layer and travels backward, toward the input.]

But how do you find the slope for 26,000 weights at once? That's backpropagation. You start at the output, where the error is, and work backward.

### 7.10
[The backward wave splits at each neuron and spreads across the connections, brighter on some threads than others.]

Each weight gets its share of the blame, according to how much it contributed to the mistake. Under the hood, this is the chain rule from calculus, applied layer by layer.

### 7.11
[Thousands of connections shift their brightness very slightly. The "7" output brightens a hair.]

Then every weight moves a tiny step in the direction that would have made the mistake smaller. (In practice, the nudges are averaged over a small batch of examples first.)

### 7.12
[Time-lapse: thousands of different handwritten digits stream through the input grid in a blur. The tangle of connections steadily reorganizes into structured patterns. The palette warms back toward cyan and blue.]

Now repeat with sixty thousand handwritten digits, again and again. No single step teaches it what a seven is, but thousands of small corrections add up.

### 7.13
[The output for the 7 climbs steadily: 9%, 30%, 70%, 98%.]

Slowly, the weights settle into the patterns we flew through tonight.

### 7.14
[New digits, never seen before, flow through; small checkmarks appear for most, a few red marks for misses.]

Then the real test: digits it has never seen. A network that merely memorized its examples would stumble here. This one gets about 97 out of 100 right.

### 7.15
[The camera pulls back further and further. The whole network shrinks into a single faint point of light in the dark: the same light from the opening.]

So how does a machine learn to see a seven? Nobody tells it. It guesses, measures how wrong it was, adjusts, and repeats, thousands of times over.

### 7.16
[Black. The single light pulses once, slowly.]

That faint light we started with? It was this whole network, seen from far away. Everything you passed tonight was just numbers, shaped by examples.

### 7.17
[The light fades. An end card appears.]

Thanks for flying with me.
