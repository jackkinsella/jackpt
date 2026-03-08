import random
from value import Value

# Type aliases — used throughout for readability
# Vector:      a 1D list of Values, e.g. a token embedding or hidden state
# Matrix:      a 2D list of Values, e.g. a weight matrix [nout × nin]
Vector = list[Value]
Matrix = list[list[Value]]

# nout x nin is a convention - this means rows by columns - so "rows are outputs"
# A weight matrix `W` transforms an input vector `x` into an output vector `z` via:
# z = W @ x
#
# Concretely, for W [2 x 3] and x [3]:
#
#   W = [[w11, w12, w13],    x = [x1,
#        [w21, w22, w23]]         x2,
#                                 x3]
#
#   z = W @ x = [w11*x1 + w12*x2 + w13*x3,   <- row 1 dotted with x -> z1 (output neuron 1)
#                w21*x1 + w22*x2 + w23*x3]    <- row 2 dotted with x -> z2 (output neuron 2)
#
# Each row gets dotted with the entire input vector, producing one scalar output.
# So z has shape [z1, z2] — one value per row of W, one value per output neuron.
#
# NB: This logic doesn't really apply for "lookup matrices" - e.g. wte and wpe
# - So we are doing `wpe[t]` to grab a whole VECTOR `t` directly
# - In this context, the first param (nout), means "number of rows in the table".
#
def matrix(nout: int, nin: int) -> Matrix:
    std = 0.08
    # random.gauss - mean of 0, std of 0.08 -- so very close to 0 - even 0.3 or -0.3 will be rare
    # ==
    # Why random?
    # ===
    # The symmetry problem: If all weights start equal, then:

    # 1. Every neuron gets the same input
    # 2. Every neuron computes the same output
    # 3. Every neuron gets the same gradient
    # 4. Every neuron updates identically

    # So you'd effectively have a network with just **one neuron** repeated
    #
    # (Background for next question) What does a neuron do?
    # ===
    # 1. Linear step**: `z = w1*x1 + w2*x2 + b` — weighted sum of inputs plus bias
    # 2. **Activation step**: `a = tanh(z)` (or some other non linear function, like relu) — squash through a non-linear function
    # The result `a` is the **activation** — what gets passed to the next layer. It is called activation based on analogy with neuroscience where nerves above a certain threshold fire due to being above a threshold
    # Without the activation function, step 1 is just a linear transformation. And stacking linear transformations on top of each other is still just... a linear transformation. No matter how many layers you add, you could collapse it all into one matrix multiply. The network couldn't learn anything a single layer couldn't.

    # The activation function introduces **non-linearity**, which is what lets deep networks learn complex patterns like curves, boundaries, and hierarchical structure.
    #
    # ===
    # Why near 0 with low std
    # ===
    #
    # We keep them near 0 for three reasons:
    # 1. No single weight dominates early on — all neurons get a chance to learn from all inputs equally.
    # 2. Healthy gradients at init — tanh has its steepest slope near 0 (gradient ≈ 1), so keeping z = w1*x1 + w2*x2 + ...
    #    near 0 means gradients are large and learning starts fast. If weights were large, z would be large,
    #    tanh would saturate (flat region), and gradients would be near 0 — the network would barely learn at all.
    #    This is the ACTIVATION part causing the problem.
    # 3. Avoid vanishing/exploding gradients through layers — this is the LINEAR part (chain rule):
    #    The chain rule multiplies gradients through every layer. So weight magnitude compounds across layers:
    #
    #    Too small (e.g. 0.001):
    #      gradient at input = upstream × 0.001 × 0.001 × 0.001 = effectively 0  ← vanishing
    #
    #    Too large (e.g. 10.0):
    #      gradient at input = upstream × 10 × 10 × 10 = explodes to infinity     ← exploding
    #
    #    ~0.08 is the sweet spot: not so small that gradients vanish, not so large that they explode
    #    or saturate tanh. (More principled choices: Xavier = 1/sqrt(nin), He = sqrt(2/nin) for relu)

    #
    # tanh activation function — maps any input to the range (-1, 1):
    #
    #    1.0 |                                    .........
    #        |                              ......
    #        |                          ....
    #    0.5 |                        ..
    #        |                       .
    #        |                      .         <- steep centre: gradient ≈ 1, healthy for backprop ✅
    #    0.0 |---------------------.---------------------
    #        |                    .
    #        |                   .
    #   -0.5 |                 ..
    #        |             ....
    #        |        ......
    #   -1.0 |.........                       <- flat edges: gradient ≈ 0, saturated, learning stalls ✗
    #        |
    #         -4    -3    -2    -1     0     1     2     3     4
    #
    return [[Value(random.gauss(0, std)) for _ in range(nin)] for _ in range(nout)]


# dot: the dot product of two equal-length (num of elements) vectors — multiply element-by-element, then sum.
# Geometrically, it measures how much two vectors point in the same direction.
# High value = similar direction = high relevance (used in attention scoring).
# Zero = perpendicular = no relationship. Negative = opposite directions.
#
# a:       Vector  — first vector
# b:       Vector  — second vector, same length as a
# returns: Value   — single scalar
def dot(a: Vector, b: Vector) -> Value:
    return sum(a_i * b_i for a_i, b_i in zip(a, b))


# x:       Vector  — input vector, length nin
# w:       Matrix  — weight matrix, shape [nout × nin]
# returns: Vector  — output vector, length nout (one dot product per row)
# This is matrix-VECTOR multiplication (not matrix-matrix): equivalent to W @ x in numpy.
# x is a 1D vector, not a matrix. Each row of w is dot-producted with x,
# producing one scalar per row, giving a 1D output vector of length nout.
def linear(x: Vector, w: Matrix) -> Vector:
    return [dot(row, x) for row in w]


# softmax: converts raw scores (logits) into a probability distribution summing to 1.
#
# logits:  Vector  — raw scores, any real number, one per token in vocab
# returns: Vector  — probabilities in (0, 1), same length, summing to 1.0
#
# How it works: exponentiate every logit (making all positive), then divide each
# by the total (normalising to sum to 1). Exponentiation also amplifies differences —
# the highest logit gets disproportionately more mass (i.e. more probability), making the output peaky:
#
#   logits:  [-1.0,  0.5,  2.0,  0.1]
#   exp:     [ 0.37, 1.65, 7.39, 1.11]   ← all positive, large gap at 2.0
#   sum:     10.52
#   output:  [ 0.04, 0.16, 0.70, 0.11]   ← sums to 1.0, token 2 dominates
#
# Shape of softmax output as a function of input spread:
#
#   narrow spread (logits close together) → flat, uncertain distribution:
#     [0.26, 0.28, 0.24, 0.22]   ← model is unsure
#
#   wide spread (one logit much larger) → peaked, confident distribution:
#     [0.02, 0.05, 0.91, 0.02]   ← model is confident
#
#   visualised on a number line (4 tokens, width = probability mass):
#
#   uncertain:  |████|█████|████|████|
#   confident:  |█|██|████████████████████████████████|█|
#
# Peaky means the model is confident. During training, distributions start flat (random weights → similar logits)
# and should become peaky as the model learns to score the correct token much higher.. If the trained model is flat, it has learned nothing.
# During generation (as opposed to Training), you sometimes want it _less_ peaky
# (i.e.not to always pick the most probable term) so you deliberately flatten
# the distribution (temperature) to produce more varied output — always picking
# the top token produces repetitive, boring names.
def softmax(logits: Vector) -> Vector:
    # Numerical stability: "numerically stable" is a general term meaning the computation
    # gives accurate results under floating point's limited precision. An algorithm is
    # unstable if large or extreme inputs cause overflow (→ inf), underflow (→ 0), or
    # cancellation errors — not because the math is wrong, but because floats break at extremes.
    #
    # The problem here: if any logit is large (e.g. 500.0), exp(500) overflows to inf in
    # 32-bit float. Then inf/inf = nan — the entire computation silently produces garbage:
    #
    #   naive:   exp(500) / (exp(500) + exp(499))  →  inf / inf  →  nan  ✗
    #
    # The fix: subtract the max logit from all logits before exponentiating.
    # This is mathematically identical (the constant cancels in the division)
    # but keeps all exponent inputs <= 0, so exp() never overflows:
    #
    #   stable:  exp(500-500) / (exp(500-500) + exp(499-500))
    #          = exp(0)       / (exp(0)       + exp(-1))
    #          = 1.0          / (1.0          + 0.37)      = 0.73  ✓
    #
    # max_val is extracted as a plain float (.data) — intentionally outside the autograd
    # graph, since it is just a numerical stabiliser, not part of the learned computation.
    max_val = max(val.data for val in logits)
    exps = [(val - max_val).exp() for val in logits]
    total = sum(exps)
    return [e / total for e in exps]


# RMSNorm — Root Mean Square Normalisation.
# x:       Vector  — input vector, length n_embed
# returns: Vector  — same shape, rescaled so its RMS equals 1
#
# As vectors flow through layer after layer of matrix multiplies and additions,
# their magnitudes drift — growing large or shrinking small. This destabilises
# training (gradients explode or vanish). RMSNorm keeps every vector at a
# controlled scale before it enters each sublayer.
#
# Steps:
#   ms    = mean of squared elements  (the "mean square" in RMS)
#   scale = 1 / sqrt(ms)              (** -0.5 is the same as 1/sqrt)
#   output = every element × scale    (rescale so RMS = 1)
#
# RMSNorm vs softmax — they are often confused because both "normalise", but
# they guarantee completely different things:
#
#   softmax  → elements are all in (0,1) AND sum to exactly 1.0  (probability distribution)
#   RMSNorm  → elements are unbounded, can be negative, sum to anything  (magnitude control)
#
# Concrete example — same input vector through each:
#
#   input:           [0.50, -1.20,  0.80,  2.00]
#
#   after softmax:   [0.13,  0.02,  0.17,  0.57]   sum = 1.00  ✓  all positive  ✓
#   after RMSNorm:   [0.40, -0.95,  0.63,  1.58]   sum = 1.66  ✗  negatives ok  ✓
#
#   RMS check:  sqrt((0.16 + 0.90 + 0.40 + 2.50) / 4) = sqrt(0.99) ≈ 1.0  ✓
#
# softmax is used at the OUTPUT of the model to pick the next token (needs probabilities).
# RMSNorm is used INSIDE the model between layers to stop magnitudes drifting.
#
# Why softmax cannot replace RMSNorm here:
# Softmax has two effects: (1) forces all elements positive, (2) forces them to sum to 1.
# Effect 2 is the killer. If elements must sum to 1, raising one element forces all others
# down — every dimension is defined relative to all the others. It's a competition, not a
# representation. The downstream matrix multiplies and relu activations need dimensions to
# vary freely and independently:
#   - negatives matter: relu gates on sign — kill negatives and relu has nothing to gate on
#   - values > 1 matter: matrix multiply depends on actual magnitude, not just ratios
#   - independent dimensions matter: softmax makes every element relative to every other,
#     so the weight matrices can never learn clean independent features
#
# RMSNorm's much lighter touch: it only controls collective magnitude — how loud the vector
# is overall. Signs, ratios, relative structure all preserved. Like turning down the volume
# on a song without changing the mix.
#
# The 1e-5 epsilon is numerical stability: if x is all zeros, ms=0 and
# 1/sqrt(0) = inf. Adding 0.00001 prevents that without affecting normal inputs.
def rmsnorm(x: Vector) -> Vector:
    ms = sum(xi * xi for xi in x) / len(x)
    scale = (ms + 1e-5) ** -0.5
    return [xi * scale for xi in x]
