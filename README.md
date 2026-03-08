# JackPT

A minimal GPT implementation based on the ideas in Andrej Karpathy's blog post:
[https://karpathy.github.io/2026/02/12/microgpt/](https://karpathy.github.io/2026/02/12/microgpt/)

Built as a personal learning exercise. The main difference from the original is a large number of inline comments explaining every concept covered — attention mechanisms, residual connections, autograd, the Adam optimizer, KV caching, and more.

These comments were naturally generated through conversation with AI and are not guaranteed to be correct.

## Structure

- `value.py` — scalar autograd engine (`Value` class with backprop)
- `helpers.py` — math primitives: `dot`, `linear`, `softmax`, `rmsnorm`, `matrix`
- `model.py` — model definition: `build_model()` and `gpt()` forward pass
- `jackpt.py` — entry point: corpus loading, training loop, persistence, inference

## Usage

Train from scratch:

```sh
python jackpt.py --retrain
```

Load saved weights and generate names (default if a saved model exists):

```sh
python jackpt.py
```

Options:

```sh
python jackpt.py --retrain              # force retraining even if a saved model exists
python jackpt.py --num_steps 3000       # number of training steps (default: 1000)
python jackpt.py --n_embed 32           # embedding dimensionality (default: 16)
```

Trained models are saved to `models/` with a filename encoding the hyperparameters, e.g.:

```
models/jackpt_embed16_heads4_layers1_block16_steps1000.json
```

Different hyperparameter configurations are saved separately so they never overwrite each other.

## Requirements

Pure Python — no dependencies beyond the standard library.