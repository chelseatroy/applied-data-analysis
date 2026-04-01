# Language Generator Exercise

This exercise walks through how a Markov chain language model works — training it on
real blog post text and then stepping through generation interactively in the browser.

## Scripts

### `demo_voice_generation.py` — Train and generate

Trains a Markov model on blog post text, saves it to `my_model.pkl`, and prints a
sample of generated text.

```bash
python demo_voice_generation.py          # order 4 (default)
python demo_voice_generation.py 2        # order 2
python demo_voice_generation.py 1        # order 1
```

The `order` argument controls how many previous tokens the model looks at when
predicting the next one. Higher order = more coherent text, but fewer branching
choices per step. Lower order = more surprising output, more visible probability
distributions.

### `generate_markov_viz.py` — Visualize the model

Loads `my_model.pkl` and generates `markov_visualization.html` — an interactive
step-through visualization of how the model generates text.

```bash
python generate_markov_viz.py
```

Then open `markov_visualization.html` in any browser.

## Typical workflow

```bash
# 1. Train at whatever order you want
python demo_voice_generation.py 2

# 2. Visualize that model
python generate_markov_viz.py

# 3. Open the result
open markov_visualization.html
```

## What the visualization shows

- **Generated text** — tokens produced so far, with the current state highlighted in gold
- **Current state** — the last N tokens the model is looking at right now
- **Possible next tokens** — every token the model could choose next, colored by
  relative probability using jewel tones: sapphire/aquamarine for low-probability
  options, through amethyst, to topaz and ruby for the most likely choices
- **Step** — advance one token (the chosen card flashes)
- **Auto-play** — watch generation run continuously; adjust speed with the slider
- **Reset** — start over from a new random sentence beginning
