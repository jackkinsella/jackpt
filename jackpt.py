# See https://karpathy.github.io/2026/02/12/microgpt/
import os
import sys
import json
import random
import argparse

from value import Value
from helpers import Vector, Matrix, softmax
from model import KVCache, build_model, gpt

random.seed(42)

# Command-line arguments
# ======================
parser = argparse.ArgumentParser(description='JackPT — a micro GPT implementation')
parser.add_argument('--retrain',   action='store_true', default=False, help='Force retraining even if a saved model exists')
parser.add_argument('--num_steps', type=int,            default=1000,  help='Number of training steps (default: 1000)')
parser.add_argument('--n_embed',   type=int,            default=16,    help='Embedding dimensionality (default: 16)')
args = parser.parse_args()

# Corpus Grabbing and Preparation
# ================================

def get_corpus_and_persist_locally() -> list[str]:
    if not os.path.exists('input.txt'):
        import urllib.request
        names_url = 'https://raw.githubusercontent.com/karpathy/makemore/refs/heads/master/names.txt'
        urllib.request.urlretrieve(names_url, 'input.txt')
    docs = [l.strip() for l in open('input.txt').read().strip().split('\n') if l.strip()]
    random.shuffle(docs)
    return docs

# Tokenizer
# =========

docs = get_corpus_and_persist_locally()
# Lesson 1: We want unique set of tokens here
uchars = sorted(set(''.join(docs)))
# Lesson 2: We want a BOS token to mark the beginning and end of a sequence
BOS = len(uchars)
# vocab_size includes all unique characters plus the BOS token
vocab_size = len(uchars) + 1

# Hyperparameters
# ===============
n_embed    = args.n_embed
n_head     = 4
n_layer    = 1
block_size = 16

# Build model — initialises all weight matrices as Value objects and returns
# the state_dict (named matrices), params (flat list of all Values), and
# the hyperparameters needed by gpt() at call time.
state_dict, params, n_embed, n_head, n_layer, head_dim = build_model(
    vocab_size=vocab_size,
    n_embed=n_embed,
    n_head=n_head,
    n_layer=n_layer,
    block_size=block_size,
)
print(f"num params: {len(params)}")

# Model Persistence
# =================

def model_filename(n_embed: int, n_head: int, n_layer: int, block_size: int, num_steps: int) -> str:
    # Filename encodes the hyperparameters so different runs don't overwrite each other.
    # e.g. "models/jackpt_embed16_heads4_layers1_block16_steps1000.json"
    return f"models/jackpt_embed{n_embed}_heads{n_head}_layers{n_layer}_block{block_size}_steps{num_steps}.json"

def save_model(state_dict: dict, filename: str) -> None:
    # Serialise every weight's .data value to JSON.
    # We only save .data (the float), not the full Value graph — the graph is
    # rebuilt fresh each run and we just overwrite .data on load.
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    serialised = {
        key: [[cell.data for cell in row] for row in mat]
        for key, mat in state_dict.items()
    }
    with open(filename, 'w') as f:
        json.dump(serialised, f)
    print(f"model saved to {filename}")

def load_model(state_dict: dict, filename: str) -> None:
    # Read saved float values back and overwrite .data on each Value in place.
    # The Value graph structure (children, local_grads) is left untouched so
    # backprop still works correctly if you continue training after loading.
    with open(filename, 'r') as f:
        serialised = json.load(f)
    for key, mat in serialised.items():
        for row_idx, row in enumerate(mat):
            for col_idx, value in enumerate(row):
                state_dict[key][row_idx][col_idx].data = value
    print(f"model loaded from {filename}")

# Training Loop
# =============

num_steps     = args.num_steps
retrain       = args.retrain
saved_filename = model_filename(n_embed, n_head, n_layer, block_size, num_steps)

first_moment  = [0.0] * len(params) # exponential moving average of gradients
second_moment = [0.0] * len(params) # exponential moving average of squared gradients

if not retrain and os.path.exists(saved_filename):
    load_model(state_dict, saved_filename)
else:
    for step in range(num_steps):
        # (Yes this doesn't actually look at the whole corpus. With num_steps equal
        # to 1,000 it's only going through about 3% of the total of 32,000 names.
        # The standard approach, not depicted here, is to cycle through the data
        # multiple times using epochs as an outer loop and shuffling the docs each time)
        doc = docs[step % len(docs)]

        # Take single document (a name here), tokenize it, surround it with BOS special token on both sides
        tokens = [BOS] + [uchars.index(ch) for ch in doc] + [BOS]
        # Why min?
        # ===
        # A **defensive guard** against edge cases in the data. `names.txt` could theoretically contain an unusually long entry — a hyphenated name, a data corruption, a stray sentence — that exceeds 16 characters. Without the `min`, `position_idx` would march past 16 and the KV cache would grow beyond what the model was designed for, likely producing nonsense or crashing.
        num_predictions_needed = min(block_size, len(tokens) - 1)

        # This is a deliberate reset between documents because each document is an
        # independent training example. The KV cache accumulates context as the model
        # processes tokens within a document, but it must not bleed into the next
        # document. i.e. "emma" attending to tokens from "olivia" would be nonsensical.
        keys: KVCache   = [[] for _ in range(n_layer)]
        values: KVCache = [[] for _ in range(n_layer)]
        losses: list[Value] = []
        for position_idx in range(num_predictions_needed):
            token_idx  = tokens[position_idx]
            target_idx = tokens[position_idx + 1]
            # How come gpt() is used here in the training loop?
            # ====
            # It's the same function for both training and generation — the forward
            # pass is identical in both cases. What changes is what you *do* with
            # the logits afterwards:
            # - Training: compare logits against the known correct next token (`target_idx`) to compute a loss, then run backprop to adjust the weights
            # - Generation: sample from the logits to pick the next token, then feed that token back in as the next input
            #
            # What are logits in this context?
            # ====
            # Each of the 27 logits is the model's **raw, unnormalized confidence
            # score** for "given everything I've seen so far in this name, how
            # likely is each character to come next?"
            logits: Vector = gpt(token_idx, position_idx, keys, values, state_dict, n_head, n_layer, head_dim)
            # Make sure the probabilities sum to one.
            probs: Vector = softmax(logits)
            # Why -log(p)? The betting analogy.
            # ===
            # Think of the model as a gambler spreading £1 across 27 horses (characters). After the
            # race, we only care about how much it bet on the winning horse — probs[target_idx].
            #
            # log() is a function that grows slowly: log(1) = 0, log(0.5) ≈ -0.69, log(0.01) ≈ -4.6.
            # As p approaches 0 it shoots toward -infinity; as p approaches 1 it flattens toward 0.
            # The negative sign flips it so that low confidence = high penalty:
            #
            #   bet £0.90 on winner → -log(0.90) ≈ 0.10  tiny penalty   (confident and right)
            #   bet £0.10 on winner → -log(0.10) ≈ 2.30  bigger penalty  (uncertain)
            #   bet £0.01 on winner → -log(0.01) ≈ 4.60  harsh penalty   (almost certain about the wrong horse)
            #
            # Crucially, we never look at the wrong horses at all. It doesn't matter how the
            # remaining probability was spread — only the winner's share counts. This is called
            # cross-entropy loss and is the standard loss function for classification problems.
            loss_at_position: Value = -probs[target_idx].log()
            losses.append(loss_at_position)

        # `losses` at this point is a list of `num_predictions_needed` individual
        # loss values — one per character position in the name. For `"emma"` that
        # might be something like `[3.2, 2.1, 0.8, 1.4, 2.0]`.
        #
        # We take the mean rather than the sum because names have different lengths —
        # otherwise a 10-character name would produce a bigger loss than a 3-character
        # name just by having more positions, not because the model was worse at it.
        loss: Value = (1 / num_predictions_needed) * sum(losses)

        # `loss.backward()` walks the computation graph in reverse — from `loss` back
        # through every operation to every parameter — and computes the gradient of each
        # parameter: "if I nudge this weight slightly, how much does the loss change?"
        # That gradient gets stored in `param.grad` on each parameter.
        loss.backward()

        # Adam optimizer
        # ====
        # Used instead of plain gradient descent (param.data -= lr * param.grad).
        # Adam keeps a smoothed running average of gradients (first moment) and a
        # running average of squared gradients (second moment), giving each parameter
        # its own personalised, adaptive learning rate.
        learning_rate       = 0.01  # base learning rate before decay
        first_moment_decay  = 0.85  # how much to weight past gradients (vs new signal) in the running average
        second_moment_decay = 0.99  # same but for squared gradients
        epsilon             = 1e-8  # tiny number added to denominator to prevent division by zero
        # Controls how big a step we take when updating each weight. Decays linearly
        # over time — bold early updates, careful nudges later when weights are converging.
        learning_rate_current_step = learning_rate * (1 - step / num_steps)
        for param_idx, param in enumerate(params):
            # The first moment smooths gradients by keeping a memory of recent ones.
            # It's like a rolling average that fades the past:
            #
            # Imagine gradients over 3 steps were: 0.8, 0.1, 0.9
            # first_moment_decay = 0.85 means "weight the past at 85%, new signal at 15%"
            #
            # step 1: first_moment = 0.85 * 0.0   + 0.15 * 0.8  = 0.12   # mostly nothing yet
            # step 2: first_moment = 0.85 * 0.12  + 0.15 * 0.1  = 0.117  # smoothed down
            # step 3: first_moment = 0.85 * 0.117 + 0.15 * 0.9  = 0.234  # trending upward
            #
            # The 0.85 * first_moment[param_idx] part is "remember most of what I knew before".
            # The (1 - 0.85) * param.grad part is "but let the new gradient nudge me slightly".
            # Together they produce a smooth trend rather than a jittery signal.
            first_moment[param_idx]  = first_moment_decay  * first_moment[param_idx]  + (1 - first_moment_decay)  * param.grad
            second_moment[param_idx] = second_moment_decay * second_moment[param_idx] + (1 - second_moment_decay) * param.grad ** 2
            # Why bias correction?
            # ===
            # Both moment buffers start at 0.0. At step 1, the running average is heavily dragged
            # toward zero just because of that initialisation — not because the gradient is small.
            # Dividing by (1 - decay ** (step + 1)) undoes that drag:
            #
            #   step 1:  divide by (1 - 0.85^1)  = 0.15   → rescales strongly  (buffer is cold)
            #   step 10: divide by (1 - 0.85^10) = 0.803  → rescales barely    (buffer is warming)
            #   step 50: divide by (1 - 0.85^50) ≈ 1.0    → no effect at all   (buffer is fully warm)
            first_moment_bias_corrected  = first_moment[param_idx]  / (1 - first_moment_decay  ** (step + 1))
            second_moment_bias_corrected = second_moment[param_idx] / (1 - second_moment_decay ** (step + 1))
            # The actual weight update: move in the smoothed gradient direction, scaled by
            # each parameter's own volatility history and the current learning rate.
            # -= because we want to move against the gradient to reduce loss.
            param.data -= learning_rate_current_step * first_moment_bias_corrected / (second_moment_bias_corrected ** 0.5 + epsilon)
            # Zero the gradient so the next step's backward() doesn't accumulate on top of this one.
            param.grad = 0

        print(f"step {step+1:4d} / {num_steps:4d} | loss {loss.data:.4f}")

    save_model(state_dict, saved_filename)

# Inference
# =========

# in (0, 1] — controls the "creativity" of generated text.
# Low temperature → model picks the most likely characters more often → names sound more normal but repetitive.
# High temperature → probability is spread more evenly → more creative but potentially nonsensical.
temperature = 0.5
number_of_items_to_generate = 20
print("\n--- inference (new, hallucinated names) ---")
for sample_number in range(number_of_items_to_generate):
    keys: KVCache   = [[] for _ in range(n_layer)]
    values: KVCache = [[] for _ in range(n_layer)]
    # Seed generation with BOS — the same "start of name" signal the model was trained on.
    # The model learned that BOS means "a name is about to begin". It predicts the first
    # character, which becomes the next token_idx, and so on until it predicts BOS again (end of name).
    token_idx: int = BOS
    generated_chars: list[str] = []
    # Why block_size? Safety cap — forces a stop if the model never predicts BOS.
    for position_idx in range(block_size):
        logits: Vector = gpt(token_idx, position_idx, keys, values, state_dict, n_head, n_layer, head_dim)
        # Divide logits by temperature before softmax to reshape the probability distribution.
        # Say the model's raw logits for three characters are [2.0, 1.5, 1.0]:
        #
        # temperature = 1.0  →  [2.0/1.0, 1.5/1.0, 1.0/1.0] = [2.0, 1.5, 1.0]
        # softmax([2.0, 1.5, 1.0]) → [0.51, 0.31, 0.18]  # spread out, uncertain
        #
        # temperature = 0.5  →  [2.0/0.5, 1.5/0.5, 1.0/0.5] = [4.0, 3.0, 2.0]
        # softmax([4.0, 3.0, 2.0]) → [0.71, 0.26, 0.10]  # more confident
        #
        # temperature = 0.1  →  [2.0/0.1, 1.5/0.1, 1.0/0.1] = [20.0, 15.0, 10.0]
        # softmax([20.0, 15.0, 10.0]) → [0.99, 0.007, 0.00003]  # almost deterministic
        probs: Vector = softmax([l / temperature for l in logits])
        # Sample the next token weighted by the probability distribution.
        # range(vocab_size) is the full set of token indices; weights steers the random choice.
        # [0] unwraps the single-item list that random.choices always returns.
        token_idx = random.choices(range(vocab_size), weights=[p.data for p in probs])[0]
        if token_idx == BOS:
            break
        generated_chars.append(uchars[token_idx])
    print(f"sample {sample_number+1:2d}: {''.join(generated_chars)}")
