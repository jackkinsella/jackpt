# See https://karpathy.github.io/2026/02/12/microgpt/
import os
import sys
import json
import math
import random
import argparse

random.seed(42)

# Command-line arguments
# ======================
parser = argparse.ArgumentParser(description='JackPT — a micro GPT implementation')
parser.add_argument('--retrain',   action='store_true', default=False, help='Force retraining even if a saved model exists')
parser.add_argument('--num_steps', type=int,            default=1000,  help='Number of training steps (default: 1000)')
parser.add_argument('--n_embed',   type=int,            default=16,    help='Embedding dimensionality (default: 16)')
args = parser.parse_args()

# Corpus Grabbing and Preparation
# ======

def get_corpus_and_persist_locally() -> list[str]:
    if not os.path.exists('input.txt'):
        import urllib.request
        names_url = 'https://raw.githubusercontent.com/karpathy/makemore/refs/heads/master/names.txt'
        urllib.request.urlretrieve(names_url, 'input.txt')
    docs = [l.strip() for l in open('input.txt').read().strip().split('\n') if l.strip()] # list[str] of documents
    random.shuffle(docs)
    return docs

# Tokenizer
# =======

docs = get_corpus_and_persist_locally()
# Lesson 1: We want unique set of tokens here
uchars = sorted(set(''.join(docs)))
# Lesson 2: We want a BOS token to mark the beginning of a sequence
# We get to next available ID - it says start/end
BOS = len(uchars)
# vocab_size includes all unique characters plus the BOS token
vocab_size = len(uchars) + 1

# Autograd

class Value():
    """
     Briefly, a Value wraps a single scalar number (.data) and tracks how it was
     computed. Think of each operation as a little lego block: it takes some
     inputs, produces an output (the forward pass), and it knows how its output
     would change with respect to each of its inputs (the local gradient).
     That’s all the information autograd needs from each block. Everything else
     is just the chain rule from calc, stringing the blocks together.

     The chain rule is a calculus formula for finding the derivative of a composite function (a function inside another function,
     ). It states that the derivative is the derivative of the outer function applied to the inner function, multiplied by the derivative of the inner function:

     This allows us to work with massive mathematical expressions.
    """

    def __init__(self, data: float, children: tuple['Value', ...] = (), local_grads: dict['Value', float] | None = None) -> None:
        self.data = data
        # Initialize all values to starting gradient.
        # It's the **partial derivative of the loss with respect to that specific node**. So every `Value` in the network stores: *"if I nudge this value slightly, how much does the final loss change?"
        # i.e. if I create the value's data by 1, how much will lose change by
        self.grad = 0
        # Tracks which Value nodes were inputs to the operation that produced this node.
        # Used to traverse the computation graph in backpropagation.
        # For example, `c = a + b` creates a new `Value` for `c`, and `c._children = {a, b}`.
        # When you later call `backward()`, you walk back through `_children` to know which nodes to propagate gradients to.
        # Stored as a set because we don't want duplicates in the _traversal_ graph because it creates slowness (if the input counts double, we track that in grad)
        self.children = set(children)
        # Stores the local gradient of this node's output w.r.t. each of its inputs.
        # i.e. "if this input changes by 1, how much does this node's output change?"
        self.local_grads: dict['Value', float] = local_grads if local_grads is not None else {}


    # We want inputs -> middle values -> loss ordering so that mathematical dependencies for grad are available
    def backward(self) -> None:
        # out and visited are initialized here so we can avoid resetting them each recursive call
        topo: list['Value'] = []
        visited: set['Value'] = set()
        def build_topological_graph(node: 'Value') -> None:
            visited.add(node)
            for child_node in node.children:
                if child_node not in visited:
                    build_topological_graph(child_node)

            # We do this finally in order to ensure that loss node (/output node) is the lost
            topo.append(node)

        build_topological_graph(self)
        # We kick things off by setting self.grad = 1 at the loss node, because ∂L/∂L = 1
        # the loss's rate of change with respect to itself is trivially 1.
        # In reality, this function (backward) is only ever called on the loss node .
        self.grad = 1
        # Now topo() has the graph
        # Now we want to go from loss back to inpts
        for node in reversed(topo):
            for child, local_grad in node.local_grads.items():
                child.grad += local_grad * node.grad

    # Math (binary operations)
    # ===
    def __mul__(self, other: 'Value | float') -> 'Value':
        # c = a * b
        # ∂c/∂a = b  (b is constant w.r.t. a)
        # ∂c/∂b = a  (a is constant w.r.t. b)
        other = other if isinstance(other, Value) else Value(other)
        return Value(data=self.data * other.data, children=(self, other), local_grads={self: other.data, other: self.data})

    def __add__(self, other: 'Value | float') -> 'Value':
        # c = a + b
        # ∂c/∂a = 1  (a nudge of 1 in a causes a nudge of 1 in c)
        # ∂c/∂b = 1  (same for b)
        other = other if isinstance(other, Value) else Value(other)
        return Value(data=self.data + other.data, children=(self, other), local_grads={self: 1.0, other: 1.0})

    def __pow__(self, other: 'Value | float') -> 'Value':
        # c = a ** n
        # ∂c/∂a = n * a^(n-1)  (power rule)
        # Note: we only track gradient w.r.t. the base (self), not the exponent
        other = other if isinstance(other, Value) else Value(other)
        return Value(data=self.data ** other.data, children=(self, other), local_grads={self: other.data * self.data ** (other.data - 1)})

    # Called "true" div because it always does floating point division
    def __truediv__(self, other: 'Value | float') -> 'Value':
        return self * other ** -1

    def __sub__(self, other: 'Value | float') -> 'Value':
        return self + (-other)

    # Reflected (right-hand) operations
    # ===
    # When Python evaluates `2 * x`, it first tries `int.__mul__(x)`, which fails
    # because int doesn't know about Value. Python then falls back to `x.__rmul__(2)`.
    # Without these, expressions with a scalar on the left would crash.
    def __rmul__(self, other: 'Value | float') -> 'Value':
        return self * other

    def __radd__(self, other: 'Value | float') -> 'Value':
        return self + other

    def __rsub__(self, other: 'Value | float') -> 'Value':
        return Value(other) - self

    def __rtruediv__(self, other: 'Value | float') -> 'Value':
        return Value(other) / self

    # Math (unary operations)
    # ===
    def log(self) -> 'Value':
        # c = log(a)
        # Derivation:
        #   log and e^x are inverses, so if c = log(a) then a = e^c
        #   Differentiate both sides w.r.t. c: da/dc = e^c  (because d/dx(e^x) = e^x — e^x is the unique function that is its own derivative)
        #   We want dc/da, which is the reciprocal: dc/da = 1/e^c
        #   Substitute back a = e^c: dc/da = 1/a
        return Value(data=math.log(self.data), children=(self,), local_grads={self: 1.0 / self.data})

    def exp(self) -> 'Value':
        # c = e^a
        # ∂c/∂a = e^a  (e^x is the unique function that is its own derivative)
        result = math.exp(self.data)
        return Value(data=result, children=(self,), local_grads={self: result})

    def neg(self) -> 'Value':
        return self * -1

    def __neg__(self) -> 'Value':
        return self.neg()

    def relu(self) -> 'Value':
        # c = max(a, 0)
        # ∂c/∂a = 1 if a > 0 else 0
        # Intuitively: when a > 0, relu is just the identity function (slope 1).
        # When a <= 0, the output is a flat 0 regardless of a (slope 0), so no gradient flows back.
        return Value(data=max(self.data, 0), children=(self,), local_grads={self: 1.0 if self.data > 0 else 0.0})

# ===
# SaaS Profit Model — Example Usage of Autograd
# ===
# profit = (subscribers × monthly_price) - (subscribers × churn_rate × refund_value) - server_cost
#
# We want to know: if I nudge any one variable slightly, how much does profit change?
# That's exactly what .grad tells us after backward().
#
# subscribers    = Value(1000.0)   # number of active subscribers
# monthly_price  = Value(49.0)     # $ per subscriber per month
# churn_rate     = Value(0.05)     # 5% of subscribers churn each month
# refund_value   = Value(49.0)     # full refund given to churned subscribers
# server_cost    = Value(8000.0)   # fixed monthly server cost
#
# revenue        = subscribers * monthly_price
# churn_loss     = subscribers * churn_rate * refund_value
# profit         = revenue - churn_loss - server_cost
#
# profit.backward()
#
# print(f"profit:                  {profit.data}")
# print(f"∂profit/∂subscribers:   {subscribers.grad:.4f}  — each new subscriber adds this much profit")
# print(f"∂profit/∂monthly_price: {monthly_price.grad:.4f}  — raising price $1 adds this much profit")
# print(f"∂profit/∂churn_rate:    {churn_rate.grad:.4f}  — reducing churn by 1% saves this much")
# print(f"∂profit/∂server_cost:   {server_cost.grad:.4f}   — every $1 of server cost costs exactly $1 of profit")
#
# Output:
# profit:                  38550.0
# ∂profit/∂subscribers:   46.5500  — each new subscriber adds this much profit
# ∂profit/∂monthly_price: 1000.0000  — raising price $1 adds this much profit
# ∂profit/∂churn_rate:    -49000.0000  — reducing churn by 1% saves this much
# ∂profit/∂server_cost:   -1.0000   — every $1 of server cost costs exactly $1 of profit
# ===

# Type aliases — used throughout for readability
# Vector:      a 1D list of Values, e.g. a token embedding or hidden state
# Matrix:      a 2D list of Values, e.g. a weight matrix [nout × nin]
# Rank3Tensor: a 3D array of Values — "tensor" is the general term for arrays of any number
#              of dimensions: rank-1 = vector, rank-2 = matrix, rank-3+ = tensor
# KVCache:     a rank-3 tensor used to store per-layer key (or value) vectors, one vector
#              per token seen so far. shape: [n_layer][n_tokens_so_far][n_embed]
#              keys and values each have their own separate KVCache instance.
Vector      = list[Value]
Matrix      = list[list[Value]]
Rank3Tensor = list[list[Vector]]
KVCache     = Rank3Tensor

# Every token (character) gets converted into a vector of this size. So each token is represented as 16 numbers. Bigger = the model can encode more nuanced meaning per token, but more parameters to train.
n_embed = args.n_embed
# In self-attention, instead of one big attention computation, you split `n_embed` into `n_head` parallel "heads" — each one attends to different aspects of the sequence (e.g. one head might learn grammar, another learns position). Each head works on `n_embed / n_head = 4` dimensions. They all run in parallel then get concatenated back to n_embed:
#
#   input:         [n_embed = 16]
#   split:         [4] [4] [4] [4]   ← 4 heads, each size 4
#   each head:     [4] [4] [4] [4]   ← attend independently
#   concatenate:   [n_embed = 16]     ← back to original size
n_head = 4
# Number of transformer blocks (layers) stacked on top of each other.
# Each layer (technically a "TransformerBlock" containing a "MultiHeadAttention" + "FeedForward" sublayer)
# takes the output of the previous one as input — same shape [n_embed] in and out, so they stack cleanly.
# Each layer learns increasingly abstract patterns:
#
#   Layer 1 (shallow)  — raw token patterns:        "'i' often follows 'l'"
#   Layer 2            — higher order patterns:      "this looks like a name ending"
#   Layer 3 (deep)     — abstract representations:  "this is a feminine name pattern"
#
# Analogy: editing a document in passes — first pass fix spelling, second fix grammar, third fix flow.
# Each pass builds on the previous one.
n_layer = 1
head_dim = n_embed // n_head
# The context window — the maximum number of tokens the model can look back at when predicting the next token.
# So when predicting the 5th character of a name, the model can see at most the previous 16 characters.
# It also determines the shape of the attention matrix: each token attends to every other token in the window,
# so attention is [block_size × block_size] = [16 × 16]. Making it bigger gets expensive fast — it scales quadratically
# (double the block size = 4× the compute).
block_size = 16


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
# - So we are doing `wpe[t]` to grab a whole VECTOR `t directly
# - In this context, the first param (nout), means "number of rows in the table".
#
def matrix(nout: int, nin: int) -> Matrix:
    std=0.08
    # random.gauss - mean of 0, std of 0.08 -- so very close to 0 - even 0.3 or -0.3 will be are
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

#  state_dict = the model's learnable parameters  - every matrix of weights
state_dict = {
    #  Weight matrix, Token Embeddings
    # It is a lookup table. When model sees a token, e.g. 'a' with index 0, it looks up what that character is as a learned vector (16 dims hear due to n_embed 16)
    # Token 'a' (id=0)  →  wte[0]  =  [0.02, -0.11, 0.07, ...]   # 16 numbers
    # Token 'b' (id=1)  →  wte[1]  =  [-0.05, 0.13, -0.03, ...]  # 16 numbers
    # These 16 numbers are not hand-crafted — they start random and get adjusted by backprop. After training, similar characters end up with similar vectors.
    #
    # This one is context-free -- wte['a']` is the same vector whether `'a'` appears at the start of a name, the end, or the middle. It encodes *"what kind of thing is this token"*, nothing more. It has no idea what came before or after.
    #
    # Why this shape? Because it is a lookup matrix, we want one row for each thing we want to look up (vocab vectors). Each of these is a (in this case) 16D row
    'wte': matrix(vocab_size, n_embed),
    # Weight matrix, Positional Embeddings
    #  the token's slot in the sequence, independent of what's in it
    #  encode things like:
    # - *"I am near the start of a name"* (positions 0–2 tend to be opening consonants/vowels in English names)
    # - *"I am near the end"* (positions 5–8 are where names tend to finish)
    #
    # wpe[3] is the same vector whether position 3 contains 'a' or 'z' or BOS. It encodes *"what does it mean to be in slot 3"*, nothing more.
    #
    # Why does wpe end up encoding position and wte end up encoding token identity?
    # Nobody tells them to — it falls out of the indexing structure automatically.
    # wte is indexed by token id, so wte[0] is the same vector every time
    #
    # The lookup here is added to the one in wte:
    # position 0:  wte[26] + wpe[0]  →  x[0]  (16 numbers)
    # position 1:  wte[0]  + wpe[1]  →  x[1]  (16 numbers)
    # position 2:  wte[11] + wpe[2]  →  x[2]  (16 numbers)
    'wpe': matrix(block_size, n_embed),
    # Language Model Head
    # This is the output layer and the inverse of `wte` conceptually
    # When this is reached, the transformed will have `h`,  the transformer's internal summary of "given everything I've seen so far, what should come next?" But it's 16 opaque numbers — not a probability distribution over characters. You can't sample from it yet.
    #  You THEN do a matrix multiply: `lm_head @ h`, giving a scalar per row.
    # lm_head[0]  · h  =  0.71   ← score for 'a'
    # lm_head[1]  · h  = -0.32   ← score for 'b'
    # lm_head[2]  · h  =  1.44   ← score for 'c'
    # lm_head[3]  · h  =  0.22   ← score for 'd'
    #
    # A dot product is high when two vectors **point in the same direction**. So what `lm_head` has learned during training is: row `i` points in the direction that `h` points when character `i` is the right answer. If the transformer's internal state `h` is saying "this looks like a name ending in a vowel", then the rows for `'a'`, `'e'`, `'i'` etc will have high dot products with it.
    # This allows the system to give output in terms of the tokens that the user expects
    #
    # lm_head` is essentially asking 27 questions simultaneously — one per token — each of the form: **"does the hidden state `h` look like the situation where this token should come next?"** T
    'lm_head': matrix(vocab_size, n_embed)
}

# Transformer layer weights — one set per layer (n_layer = 1 here, but could be more)
# Each layer has two sublayers: attention and MLP (feedforward)
for layer_idx in range(n_layer):
    # ===
    # ATTENTION WEIGHTS
    # ===
    # x` carries both token identity and position — it's a general-purpose representation. But attention needs three **specialised** signals from each token:

    # - Q: A **query** — a vector optimised for *searching*
    # - K: A **key** — a vector optimised for *being found*
    # - V: A **value** — a vector optimised for *transferring content
    #
    # All three projections are [n_embed × n_embed] = [16 × 16].
    # They take the 16D input vector and produce a new 16D vector in a different "space".
    # The Q and K vectors are compared (dot product) to produce attention scores.
    # The V vectors are then mixed according to those scores.
    #
    # Concretely for a sequence of tokens [t0, t1, t2, t3]:
    #   Q[t2] · K[t0] = how much should t2 attend to t0?
    #   Q[t2] · K[t1] = how much should t2 attend to t1?
    #   Q[t2] · K[t2] = how much should t2 attend to itself?
    # → softmax those scores → weighted sum of V vectors → new representation of t2
    #
    # Why three separate matrices instead of one?
    # Because "what you're looking for" (Q) and "what you advertise" (K) are usefully
    # different things. A token can broadcast one signal to others while privately
    # searching for something else. W and V are separate for the same reason —
    # the matching signal (K) and the content actually passed along (V) can differ.

    # Head concept introduced in the math at this stage BTW
    # In the `wq`/`wk`/`wv` multiply. That's where the 16D vector gets projected and then **sliced** into 4×4D chunks. Before that point — in `wte`, `wpe`, and the residual stream — everything is just plain 16D vectors with no head concept at all.
    # This system works with multiple heads... however, for mathematical efficiency, the mtraxi is `[16×16]` matrix implicitly contains all 4 heads' query projections packed together. After the multiply you slice the 16D result into 4 chunks of 4D
    # x[1] (16D) → wq [16×16] → q (16D) → split → [4D] [4D] [4D] [4D]
    #                                                 head0 head1 head2 head3
    #
    # wo** — goes the other direction. It takes the concatenated 4×4D back to 16D, and here heads are *dissolving* rather than being created. The whole point is that after `wo` there are no more heads — just one unified 16D vector. So you could say heads are a consideration in `wo` too, but in the opposite sense: it's where they stop existing.
    # attn_wq = Weight matrix, Query
    # Query projection: input embedding → query vector
    # "Given my current representation, what pattern am I searching for in the context?"
    #
    state_dict[f'layer{layer_idx}.attn_wq'] = matrix(n_embed, n_embed)
    # attn_wk = Weight matrix, Key
    # Key projection: input embedding → key vector
    # "Given my current representation, what signal do I broadcast to queries?"
    state_dict[f'layer{layer_idx}.attn_wk'] = matrix(n_embed, n_embed)
    # attn_wv = Weight matrix, Value
    # Value projection: input embedding → value vector
    # "If someone attends to me, what content do I actually send them?"
    state_dict[f'layer{layer_idx}.attn_wv'] = matrix(n_embed, n_embed)
    # attn_wo = Weight matrix, Output
    # Output projection: after all attention heads are concatenated back to [n_embed],
    # this final [n_embed × n_embed] matrix mixes the heads together into one coherent
    # updated representation. Without it, each head's output would just sit in its own
    # slice and never interact with the others.
    state_dict[f'layer{layer_idx}.attn_wo'] = matrix(n_embed, n_embed)

    # ===
    # MLP (FEEDFORWARD) WEIGHTS
    # MLP = Multi-Layer Perceptron. "Perceptron" is the 1950s name for a single artificial neuron
    # (weighted sum of inputs + threshold). "Multi-Layer" means multiple layers of them stacked.
    # In transformer literature it's also called FFN (FeedForward Network) — same thing.
    # ===
    # After attention has let tokens communicate with each other, the MLP processes
    # each token's updated representation independently — no cross-token communication here.
    # Think of attention as the "reading the room" step, and MLP as the "thinking it over" step.
    #
    # The MLP expands the representation to 4×n_embed then squeezes it back:
    #
    #   [n_embed=16]  →  fc1  →  [4×n_embed=64]  →  relu  →  fc2  →  [n_embed=16]
    #
    # Why expand to 4× first?
    # The wider middle layer gives the network more "working space" to compute non-linear
    # functions. With only 16 dimensions in and out, a single linear layer couldn't do much.
    # Expanding to 64, applying relu (which zeroes out negatives, introducing non-linearity),
    # then compressing back to 16 lets the network learn much richer transformations.
    # The 4× ratio is an empirical GPT convention — wide enough to be expressive, not so
    # wide that it dominates parameter count.

    # fc1: expand from n_embed → 4*n_embed
    # Each of the 64 output neurons learns a different linear combination of the 16 inputs,
    # then relu kills the negative ones — different neurons fire for different input patterns.
    state_dict[f'layer{layer_idx}.mlp_fc1'] = matrix(4 * n_embed, n_embed)
    # fc2: compress from 4*n_embed → n_embed
    # Reads the pattern of which fc1 neurons fired and distils it back into a 16D update
    # that gets added to the token's representation going into the next layer.
    state_dict[f'layer{layer_idx}.mlp_fc2'] = matrix(n_embed, 4 * n_embed)

# Params
# 1. The number of weights in the whole model. It's its memory. When you say GPT-4 "knows" that Paris is the capital of France, that fact is somehow encoded as specific float values across millions of parameters.
# 2. Parameters are knobs that backprop tunes (before they are just random noise) - but the forward pass runs, computes loss (how wrong was it), then backprop computes grad for every parameter ("nudge up and down") and graident descence adjusts them slightly. . Repeat millions of times until loss is low.
params = [p for mat in state_dict.values() for row in mat for p in row]
print(f"num params: {len(params)}")


# Helper mathematical functions
# =====

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

# Architecture
# ====
#
# The model architecture is a stateless function: it takes a token, a position,
# the parameters, and the cached keys/values from previous positions, and
# returns logits (scores) over what token the model thinks should come next in
# the sequence.
#
# Architecture defines the **hypothesis space** — the complete set of
# input/output mappings the model could ever produce. Training searches that
# space. Data determines which point in the space gets selected.
#
# Architecture also encodes **inductive bias** — a thumb on the scale toward
# certain solutions. The transformer doesn't just *allow* attending to context,
# it *structurally enforces* that every token looks at every other token. That's
# not just capability, it's a built-in assumption that context matters. So
# architecture is also about built-in assumptions about what the solution looks
# like.


# Full pipeline (WIP - check at the end)
# =====
# token ids
#     ↓
# wte + wpe  →  x  (16D per token)
#     ↓
# wq, wk, wv  →  q, k, v  (project x into three roles)
#     ↓
# attention scores (q · k) → softmax → weighted sum of v
#     ↓
# wo  →  mix heads back together  (still 16D)
#     ↓
# mlp_fc1  →  64D  →  relu
#     ↓
# mlp_fc2  →  16D
#     ↓
# lm_head  →  27 logits  →  softmax  →  probabilities
#
#  gpt function
# ====
# gpt` is an **autoregressive** function — each call produces one token, and the
# output of one call becomes the input to the next. The "auto" means
# self-referential. Specifically, it is a single autogressive step.
#
# It is stateless - it has no memory of its own between calls (strictly speaking,
# it mutates the keys and values caches passed in, so it is not purely stateless —
# but the caller owns those caches and is responsible for managing them). The caller owns
# the cache (keys and values together count as the whole cache) and passes it in
# each time. This makes the function pure and reusable — you could run multiple
# independent generation sequences simultaneously just by keeping separate
# `keys`/`values` for each.
#
# Aside: The reason keys and values are cached but **not queries** is that queries
# are never reused — each new token only needs its own fresh query to attend
# over the past. Past queries served their purpose at the time and are never
# looked up again.
#
# Computational complexity:
# Each call to gpt() at position t does t dot products in the attention step
# (the new token's query against all t accumulated keys). Summing across a full
# generation of T tokens:
#
#   call 1:  1 dot product
#   call 2:  2 dot products
#   call 3:  3 dot products
#   ...
#   call T:  T dot products
#   total:   1 + 2 + 3 + ... + T
#
# This is the sum of an arithmetic series. Gauss's proof — write it forwards
# and backwards, add them together:
#
#   forwards:   1   +   2   +   3   + ... +   T
#   backwards:  T   + (T-1) + (T-2) + ... +   1
#               ───────────────────────────────────
#   sum:       (T+1)+ (T+1) + (T+1) + ... + (T+1)   ← T copies of (T+1)
#
#   so: 2 × total = T(T+1)  →  total = T(T+1)/2
#
# For large T, T(T+1)/2 ≈ T²/2. In Big O notation constant factors are dropped,
# so this is O(T²) — quadratic. Double the sequence length, quadruple the compute.
# This is the central bottleneck of the transformer architecture.
# block_size (=16) is the context window — the hard cap on how long keys and values
# can grow, and therefore the maximum value of T. Without it, a sequence of 1000
# tokens would require 1000²/2 = 500,000 dot products just for attention. With
# block_size=16, attention never costs more than 16 dot products per call regardless
# of how many tokens have been generated — the quadratic is bounded by a small constant.
# The matrix multiplies (wq, wk, wv, wo, mlp, lm_head) are all O(1) per call —
# fixed-size vectors regardless of sequence length. For example:
#   wq, wk, wv, wo — always [16×16] multiplied by a length-16 vector.
#   Same operation on call 1 as on call 1000. The matrix never grows.

# token_idx: int      — index into wte (which character)
# pos_idx:   int      — index into wpe (which position)
# keys:      KVCache  — shape [n_layer][n_tokens_so_far][n_embed]
#            outer list: one slot per layer
#            middle list: grows by 1 each time a new token is processed (the KV cache)
#            inner list: the key vector for that token at that layer, length n_embed (16)
# values:    KVCache  — same shape as keys, holds value vectors
# returns:   Vector   — raw logits, length vocab_size (27)
def gpt(token_idx: int, pos_idx: int, keys: KVCache, values: KVCache) -> Vector:
    tok_emb: Vector = state_dict['wte'][token_idx]             # Vector length n_embed (16)
    pos_emb: Vector = state_dict['wpe'][pos_idx]               # Vector length n_embed (16)
    # What is x?
    # ====
    # x is the current token being processed (the one at `token_idx`/`pos_idx`). It starts as the raw embedding of that one token, and gets progressively refined as it passes through each transformer block.
    # It is represented as a 16D vector here.
    # The model only processes one token per call to `gpt()`. The "knowledge" of past tokens doesn't come from having them all in `x` simultaneously — it comes from the KV cache.
    #
    # What do the 16 dimensions mean?
    # ====
    # - the 16 dimensions have no fixed, pre-assigned meaning. There is no "dimension 3 always means vowel-ness"
    # - they are entangled — a single concept like "this is a name-ending pattern" is typically spread across many dimensions simultaneously, and a single dimension participates in encoding many different concepts.
    # - meanings differ across transformer block layers - x` after transformer
    # block 0 and `x` after transformer block 1 are both 16D vectors, but they
    # live in different "spaces". The weight matrices in each transformer block
    # co-evolve during training to speak a shared language with their
    # neighbours. Transformer block 0 outputs `x` in a format that transformer
    # block 1 expects as input, but that format is not human-interpretable — the
    # meaning is in the geometry, and each transformer block's geometry is
    # private to that block in the sense that it is not human-interpretable.
    # - the concept is encoded as the DIRECTION in 16D space - so many more concepts can be represented than dimensions. This is called superposition.
    # - but then why not just use 2d? Because the directions will be too close
    # and the concepts will interfere with each other too much (and the model
    # will not be able to reliably tell them apart). More dimensions give you
    # more **orthogonal** directions — directions that are at 90° to each other
    # and therefore have zero interference. In 2D you only have 2 perfectly
    # orthogonal directions (the x and y axes). In 16D you have 16 perfectly orthogonal directions.
    #
    #
    # Why addition
    # ====
    # Combine info about token meaning and position via simple addition!
    # Mathematically: information is lost — the sum is not invertible (i.e. many possible inputs lead to that outpt)
    # Functionally: no information that the model needs is lost — because the model learns to work with the sum directly, and `wte`/`wpe` co-evolve during training to make the sum as informative as possible
    x: Vector      = [t + p for t, p in zip(tok_emb, pos_emb)]  # Vector length n_embed (16)
    x              = rmsnorm(x)                                  # Vector length n_embed (16)

    # What is a transformer block?
    # ===
    # One round of "look around at context, then think" — after which the model's
    # understanding of the current token is hopefully a little sharper.
    # The "look around" part is attention: the token gathers relevant information
    # from all past tokens. The "think" part is the MLP: the token processes that
    # gathered information independently, refining its own representation.
    #
    # layer_idx: int    — layer index, selects this layer's weights from state_dict and its KV cache slot
    # x_in:      Vector — token representation coming in,  shape [n_embed] (16)
    # returns    Vector — token representation going out, shape [n_embed] (16)
    def transformer_block(x_in: Vector, layer_idx: int) -> Vector:
       # 1) Attention block
       # =======
       #
       # What is the purpose of residual?
       # ======
       #
       # This is used as `output = f(x) + x` where X is the residual
       #
       # We save this at the top of the attention block because X will get
       # modified later on but we want the original. The idea is that attention
       # only needs to compute a *correction* — what new information should be
       # added — rather than recomputing the entire representation from scratch.
       # The original `x_in` carries through unchanged and the attention output
       # is folded in on top of it - And we don't want to throw it out.
       #
       # Math explanation: The chain rule would otherwise cause the gradients to
       # shrink towards zero as things go through each layer so we need some
       # other way to preserve the healthy gradients.
       # For most operations in a neural network, that number is less than
       # 0.9 × 0.9 × 0.9 × 0.9 × 0.9 × 0.9 × 0.9 × 0.9 × 0.9 × 0.9
       # = 0.9^10
       # = 0.35   ← gradient at layer 10 is already less than half
       # = 0.9^50
       # = 0.005  ← gradient at layer 50 is nearly zero
       #
       # The residual connection sidesteps this because addition has a local gradient of exactly 1 — it passes gradients through unchanged.
       #
       # ∂output/∂x = ∂f(x)/∂x + 1
       # That `+ 1` is the key — even if `∂f(x)/∂x` shrinks toward zero through
       # many layers, there's always that constant 1 being added, so the
       # gradient can never fully vanish. It's a guaranteed highway for
       # gradients straight back to the input regardless of what `f` does.
        x_residual: Vector = x_in
        x_in = rmsnorm(x_in)
        # x_in` is a general-purpose 16D vector — it encodes "what token I am and where I am", but it's not specialised for any particular role. Before the token can participate in attention, it needs to be projected into three specialised roles:

        # - **query** — "what am I looking for in my past context?"
        # - **key** — "what do I advertise to other tokens searching for something?"
        # - **value** — "if someone attends to me, what content do I actually send them?"

        # Each of those three projections is just a `linear()` call — a `[16×16]` matrix multiply that rotates and stretches `x_in` into a new 16D vector optimised for that specific role.
        query: Vector = linear(x_in, state_dict[f'layer{layer_idx}.attn_wq'])
        key: Vector   = linear(x_in, state_dict[f'layer{layer_idx}.attn_wk'])
        value: Vector = linear(x_in, state_dict[f'layer{layer_idx}.attn_wv'])

        # These are the KV cache writes — the current token leaving its key and value vectors behind for future tokens to use.
        # Without this cache, on call 100 you'd have to reprocess all 99 previous tokens from scratch just to reconstruct their keys and values.
        keys[layer_idx].append(key)
        values[layer_idx].append(value)
        layer_keys: Matrix   = keys[layer_idx]    # Matrix [n_tokens_so_far × n_embed] — all key vectors for this layer
        layer_values: Matrix = values[layer_idx]  # Matrix [n_tokens_so_far × n_embed] — all value vectors for this layer

        attn_output: Vector = []         # Vector length n_embed (16) — built up head by head via extend
        # Multi-head: instead of attending once with the full 16D vector, we do it
        # n_head=4 times in parallel on 4D slices. Each head independently asks a
        # different question of the context — one might focus on which character tends
        # to follow, another on what position pattern is forming. The results are
        # concatenated back into 16D at the end via attn_output.extend(head_out).
        #
        # Even though having multiple heads works out the same amount of computation (16D vs 4x4D), it has more expressiveness.
        #
        # Without it  the entire 16D query has to encode *all* the things the current token is simultaneously searching for, collapsed into a single number per past token.
        # Mathematically, this is because a dot product is a symmetric, linear
        # operation. It can only measure one kind of similarity at a time. If
        # your query is pulling in two different directions at once (e.g. "I
        # want vowels AND I want name-starters"), those two signals interfere
        # with each other in the same dot product. The score for any given past
        # token is one blended number that can't cleanly separate the two
        # questions.
        #
        # By separating out into 4 heads in 4D each,  each head has its own independent query, key, and value projection, Head 1's query asks a completely different clean question. They never interfere because they operate in separate 4D subspaces and produce separate weighted sums
        for head_idx in range(n_head):
            # This is necessary due to how data is represented (four heads concatenated onto a single 16D vector)
            head_start: int       = head_idx * head_dim                          # int — start index of this head's slice
            head_slice: slice     = slice(head_start, head_start + head_dim)     # slice [head_start:head_start+head_dim] — reused for query, keys, values
            query_head: Vector    = query[head_slice]                                                                # Vector length head_dim (4)
            # Why are keys_heads and value_heads matrices whereas query_head is just a vector?
            # =====
            # Because `query_head` is just this one token's question, but `key_heads` is the answer catalogue for every token the model has ever seen in this sequence.
            #
            # `layer_keys` is the KV cache — it has grown by one entry every
            # time `gpt()` was called. So on call 10, `layer_keys` has 10 key
            # vectors in it, one per past token. `key_heads` slices out this
            # head's 4D chunk from each of those 10 vectors, giving a matrix of
            # 10 × 4D vectors — one row per past token.
            #
            # Basically,  the keys/values are always plural because attention is the act of comparing that one token against the entire histor
            key_heads: Matrix     = [key_vec[head_slice] for key_vec in layer_keys]    # Matrix shape [n_tokens][head_dim]
            value_heads: Matrix   = [val_vec[head_slice] for val_vec in layer_values]  # Matrix shape [n_tokens][head_dim]
            # The attn_logits bit computes one attention score per past token — "how relevant is each past token to what I'm currently searching for?"
            #
            # # Why do we divide by `head_dim ** 0.5`? (i.e. sqrt of head_size)
            # ====
            # This is needed to prevent the scale of the numbers getting too m
            # arge and causing problems for softmax below (specifically, one
            # score becomes close to 1.0 and everything else collapses to 0, and
            # gradient is nearly 0 so backprop stalls). Dot products
            # naturally grow larger as the number of dimensions increases — with
            # 4 dimensions you're summing 4 products. So a larger `head_dim`
            # produces larger raw scores, purely as a side effect of having more
            # terms in the sum, not because the vectors are actually more
            # similar - therefore it needs some correction.
            #
            # Why divide by sqrt of head_size instead of head size?
            # =====
            # When you sum `head_dim` independent random terms, each with
            # variance 1, the total variance is `head_dim` — but the **standard
            # deviation** (which is what actually determines the spread of
            # values on the number line) is √`head_dim`. Standard deviation is
            # the square root of variance.
            attn_logits: Vector   = [dot(query_head, key_heads[token_idx]) / head_dim**0.5 for token_idx in range(len(key_heads))]                                    # Vector length n_tokens — raw attention scores
            # attn_logits` at this point is a vector of raw scores — one per
            # past token — saying "how relevant is each past token to what I'm
            # searching for?" But they're unbounded numbers, could be anything
            # like `[0.3, -1.2, 2.1, 0.8]`. You can't use them directly as
            # mixing weights because they don't sum to 1 and could be negative.
            #
            # softmax` converts them into a proper probability distribution — all positive, all summing to 1:
            attn_weights: Vector  = softmax(attn_logits)                                                                                                              # Vector length n_tokens — probabilities summing to 1
            head_out: Vector      = [dot(attn_weights, [value_heads[token_idx][dim_idx] for token_idx in range(len(value_heads))]) for dim_idx in range(head_dim)]    # Vector length head_dim (4) — weighted sum of value vectors
            # This is result of all this attention stuff (calculated per head and accumlated in this structure)
            attn_output.extend(head_out)  # concatenate this head's 4D output into attn_output

        # The job of `attn_wo` is to mix the 4 heads' outputs together. Each
        # head independently produced a 4D chunk representing what it found, and
        # those chunks are just concatenated end-to-end in `attn_output`.
        # Without `attn_wo`, the heads would never interact — head 0's findings
        # would stay in dimensions 0-3 and never influence how head 1's findings
        # in dimensions 4-7 are interpreted. `attn_wo` is what learned, during
        # training, how to blend those four independent signals into one
        # coherent updated representation.
        #
        # Aside: In general weights matrices are what are built up during the training stage.
        x_in = linear(attn_output, state_dict[f'layer{layer_idx}.attn_wo'])
        # This step ensures that the residual freeway is in place in the mathematics.
        x_in = [attn_out + residual for attn_out, residual in zip(x_in, x_residual)]

        # 2) MLP block
        # =====
        #
        # Another residual?
        # ======
        # This is a separate one to win the attention block. By the time we get to this line, Xin already has the attention block's output folded into it. It just saves that new one as a baseline.
        x_residual = x_in                                                              # Vector length n_embed (16) — saved for residual addition
        # This is needed to control the scale before the next round of computation, especially since we added a residual.
        x_in = rmsnorm(x_in)
        # This is the expand phase of MLP - we scale from 16D to 64D to give more working space.
        #
        # This is especially important due to ReLU on the next line. With only
        # 16 dimensions in and 16 out, a single matrix multiply is a linear
        # transformation — it can rotate and scale the vector but can't learn
        # anything that a simpler model couldn't. The non-linearity (relu on the
        # next line) is what actually buys expressiveness, but relu needs room
        # to work — it zeros out negative values, and if you only have 16 values
        # to work with, zeroing some out leaves very little signal.
        #
        # Think of it like doing mental arithmetic — it's easier to solve a hard
        # problem if you're allowed to write out intermediate steps on a big
        # sheet of paper rather than keeping everything in your head. T
        x_in = linear(x_in, state_dict[f'layer{layer_idx}.mlp_fc1'])                  # Vector length 4*n_embed (64) — expand
        # without it, stacking linear transformations (matrix multiplies) on top
        # of each other is mathematically equivalent to just one matrix
        # multiply. No matter how many `linear()` calls you chain together, you
        # could always collapse them into a single matrix. The network would
        # have no more expressive power than a single layer. . This is what lets
        # the MLP learn genuinely non-linear functions.
        #
        # After relu, the 64D vector contains a sparse pattern of activations —
        # some neurons fired, most are zero. `
        x_in = [element.relu() for element in x_in]                                   # Vector length 4*n_embed (64) — non-linearity
        x_in = linear(x_in, state_dict[f'layer{layer_idx}.mlp_fc2'])                  # Vector length n_embed (16)   — compress back
        # This is the second half of the expand-then-compress pattern — the mirror image of `fc1`. It takes the 64D vector back down to 16D using a `[16×64]` weight matrix.
        #
        # fc2` is not just throwing away 48 dimensions — it's learned to read the *pattern* of which neurons fired as a code. For example it might have learned "if neurons 3, 17, and 42 all fired together, that means this is a name-ending pattern, so push dimension 7 of x_in in this direction." The specific combination matters, not just the individual neurons.
        x_in = [mlp_out + residual for mlp_out, residual in zip(x_in, x_residual)]    # Vector length n_embed (16) — residual connection
        return x_in


    # Why layers?
    # ===
    # Each block has its own independent set of weights (selected by layer_idx).
    # If every block shared the same weights, each pass would apply the identical
    # transformation to x — like running the same edit pass on a document repeatedly.
    # Nothing new could be learned on the second pass that wasn't already done on the first.
    # With separate weights, each block is free to specialise at a different level of
    # abstraction: layer 0 might learn "which characters follow which", layer 1 might learn
    # "what kind of name pattern is forming". They develop distinct competencies because
    # backprop trains them independently — each layer's gradients update only that layer's
    # own weights.
    for layer_idx in range(n_layer):
        x = transformer_block(x, layer_idx)

    # What is LM head all about?
    # ===
    # By this point `x` is a 16D vector that has been refined by every transformer block — it's the model's best summary of "given this token, at this position, having seen this context, what should come next?"
    #  you can't sample from 16 numbers. You need one score per possible next token — 27 scores (one per character in the vocabulary). That's what `lm_head` does.
    #
    #  Each of its 27 rows is a 16D vector that has learned, during training, to point in the direction that `x` points when a particular token is the right answer.
    #  dot product is high when two vectors point in similar directions. So if `x` is saying "this looks like a name about to end in a vowel", the rows for `'a'`, `'e'`, `'i'` will have high dot products with it, producing high logits for those tokens.
    logits: Vector = linear(x, state_dict['lm_head'])           # Vector length vocab_size (27)
    return logits

# Model Persistence
# =================

def model_filename(n_embed: int, n_head: int, n_layer: int, block_size: int, num_steps: int) -> str:
    # Filename encodes the hyperparameters so different runs don't overwrite each other.
    # e.g. "jackpt_embed16_heads4_layers1_block16_steps1000.json"
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

num_steps = args.num_steps
retrain   = args.retrain
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
        # multiple times using epochs as an outer loop and shuffling the docs each time  )
        doc = docs[step % len(docs)]

        # Take single document (a name here), tokenize it, surround it with BOS special token on both sides
        tokens = [BOS] + [uchars.index(ch) for ch in doc] + [BOS]
        # Why min?
        # ===
        # A **defensive guard** against edge cases in the data. `names.txt` could theoretically contain an unusually long entry — a hyphenated name, a data corruption, a stray sentence — that exceeds 16 characters. Without the `min`, `pos_id` would march past 16 and the KV cache would grow beyond what the model was designed for, likely producing nonsense or crashing.
        num_predictions_needed = min(block_size, len(tokens) - 1)

        # This is a deliberate reset between documents because each document is an
        # independent training example. Instead the KV cache accumulates context as
        # the model processes tokens within a document. But it must not bleed into
        # the next document. I.e. Emma attending to tokens from Olivia will be
        # nonsensical since they have no relationship.
        keys, values = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
        losses = []
        for position_idx in range(num_predictions_needed):
            token_idx  = tokens[position_idx]
            target_idx = tokens[position_idx + 1]
            # How come GPT is used here in the training loop?
            # ====
            # It's the same function for both training and generation — the forward
            # pass is identical in both cases. What changes is what you *do* with
            # the logits afterwards:
            # - Training: you compare the logits against the known correct next token (`target_idx`) to compute a loss, then run backprop to adjust the weights
            # - Generation: sample from the logits to pick the next token, then feed that token back in as the next input
            #
            # What are logits in this context?
            # ====
            # Each of the 27 logits is the model's **raw, unnormalized confidence
            # score** for "given everything I've seen so far in this name, how
            # likely is each character to come next?"
            logits: Vector = gpt(token_idx, position_idx, keys, values)
            # Make sure the probability is summed to one.
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
            #
            loss_at_position: Value = -probs[target_idx].log()
            losses.append(loss_at_position)

        # losses` at this point is a list of `num_predictions_needed` individual
        # loss values — one per character position in the name. For `"emma"` that
        # might be something like `[3.2, 2.1, 0.8, 1.4, 2.0]` — the model was more
        # or less wrong at each position
        #
        # We take the mean rather than the sum because names have different lenghts and otherwise a 10 char name would produce a bigger loss.
        loss: Value = (1 / num_predictions_needed) * sum(losses) # final average loss over the document sequence. May yours be low.

        # Up until this point, every operation in the forward pass — the matrix multiplies, the softmax, the log, the averaging — wasn't just computing numbers. Because this is a `Value`-based autograd engine (like Karpathy's micrograd), every operation also secretly recorded *how it was computed* and *what its inputs were*, building up a **computation graph** — a chain of operations from the raw weights all the way to the final `loss` scalar.

        # `loss.backward()` walks that graph in reverse — from `loss` back through
        # every operation to every parameter — and computes the **gradient** of each
        # parameter: "if I nudge this weight slightly, how much does the loss go up
        # or down?" That gradient gets stored in `p.grad` on each parameter.
        #
        # Where do these contagious Value objects enter?
        # ===
        # In the `matrix` function above (its weights are initialized with Value objects) then the `gpt` functino does math on them.
        loss.backward()

        # Adam optimizer
        # ====
        # This is used instead of gradient descent (p.data -= lr * p.grad)
        #
        # What is the learning rate?
        # ===
        # Controls how big a step we take when updating each weight. This line decays it linearly over time — at step 0 it's the full `learning_rate` (0.01), at the final step it's nearly 0.
        #
        # Why do we change it over time as training goes on?
        # ====
        # Early in training the model knows nothing so you want bold, aggressive updates. Later, as weights converge toward good values, you want small, careful nudges so you don't overshoot and undo what you've learned.
        learning_rate       = 0.01  # base learning rate before decay
        first_moment_decay  = 0.85  # how much to weight past gradients (vs new signal) in the running average
        second_moment_decay = 0.99  # same but for squared gradients
        epsilon             = 1e-8  # tiny number added to denominator to prevent division by zero
        learning_rate_current_step = learning_rate * (1 - step / num_steps) # linear learning rate decay
        # What is params here?
        # ====
        # It's a flat list of every value weight in the whole model. Enumerate gives
        # us both its index and its weight. The reason we can index here is that we
        # create a sort of super structure that takes into account all the attention
        # weight matrices , etc. Just refer to the definition of it above.
        #
        for param_idx, param in enumerate(params):
            # What's the overall idea in this optimizer?
            # ====
            # After `loss.backward()`, every parameter has a `param.grad` — a raw gradient saying "nudge me in this direction by this much." But raw gradients are noisy. One step the gradient might say "go left hard", the next step "go right a bit", the next "go left hard" again. If you follow each raw gradient blindly you zigzag and make slow progress.

            # The first moment smooths that out by keeping a **memory of recent
            # gradients**. It's like a rolling average that fades the past:
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
            #
            # The big idea is that each parameter gets its own personalized learning rate.
            #
            # What is the difference between the first and second moment measure?
            # ====
            # The first moment tracks the **average direction** — is this parameter's gradient consistently pointing left, or right, or all over the place? It's a smoothed version of the gradient itself, so it retains sign (+/-).

            # The second moment tracks the **average magnitude of movement** — has
            # this parameter been getting large gradients or small ones? Squaring
            # removes the sign, so it doesn't care about direction — only about how
            # big the swings have been.

            # Together they produce a signal like: "move in the direction the first
            # moment points, but scale the step size down if the second moment says
            # this parameter has been volatile.
            #
            # It comes from statistics (and is used in a somewhat analagous way here)
            # - **1st moment** = the mean (average value) — where is the centre of mass?
            # - **2nd moment** = the variance (average squared deviation) — how spread out is it?
            first_moment[param_idx]  = first_moment_decay  * first_moment[param_idx]  + (1 - first_moment_decay)  * param.grad
            second_moment[param_idx] = second_moment_decay * second_moment[param_idx] + (1 - second_moment_decay) * param.grad ** 2
            # Why bias correction?
            # ===
            # Both moment buffers start at 0.0. At step 1, the running average is heavily dragged
            # toward zero just because of that initialisation — not because the gradient is small.
            # This is an initialisation artifact that would cause the model to take a tiny timid
            # step when it should be taking a full one.
            #
            # Dividing by (1 - decay ** (step + 1)) undoes that drag:
            #
            #   step 1:  divide by (1 - 0.85^1)  = 0.15   → rescales strongly  (buffer is cold)
            #   step 10: divide by (1 - 0.85^10) = 0.803  → rescales barely    (buffer is warming)
            #   step 50: divide by (1 - 0.85^50) ≈ 1.0    → no effect at all   (buffer is fully warm)
            #
            # The correction is strong early and quietly fades away as the buffers fill with real data.
            first_moment_bias_corrected  = first_moment[param_idx]  / (1 - first_moment_decay  ** (step + 1))
            second_moment_bias_corrected = second_moment[param_idx] / (1 - second_moment_decay ** (step + 1))
            # This is the acutal weight update.
            #
            # Why -= and not +=?
            # ===
            # The gradient points in the direction of steepest ascent. But we want to move against the gradient to reduce loss.
            #
            # Why multiply by first_moment_bias_corrected?
            # ===
            # The first moment tracks the **average direction** — is this parameter's gradient consistently pointing left, or right, or all over the place? It's a smoothed version of the gradient itself, so it retains sign (+/-) which travels via the multiplication
            #
            # Why divide by second_moment_bias_corrected ** 0.5 + epsilon?
            # ===
            # dividing by the square root of the second moment. This is each
            # parameter's personal speed limiter. If a parameter has been getting
            # wild noisy gradients, its second moment is large, so this division
            # produces a small number — the update is reined in. If gradients have
            # been small and stable, the division produces a larger number — the
            # update gets a boost.
            #
            # Why * learning_rate_current_step?
            # ===
            # scales the whole thing down to a sensible magnitude. Without this,
            # even a well-shaped update might be enormous and overshoot.
            #
            # Why + epilson?
            # ===
            # Just prevents division by zero if the second moment ever hits exactly
            # 0.0.
            param.data -= learning_rate_current_step * first_moment_bias_corrected / (second_moment_bias_corrected ** 0.5 + epsilon)
            # Doesn't this lose info?
            # ===
            # Yes it does discard the gradient but deliberately. The information
            # isn't lost because it has already been consumed by the data and also
            # to update the first and second moments.
            #
            # If you didn't zero it out here, the next step's gradient would pile on
            # top of the previous step's gradient and the update would be
            # double-counted due to back propogation doing += on `grad`
            param.grad = 0

        print(f"step {step+1:4d} / {num_steps:4d} | loss {loss.data:.4f}")

    save_model(state_dict, saved_filename)

# in (0, 1], control the "creativity" of generated text, low to high
temperature = 0.5
number_of_items_to_generate = 20
print("\n--- inference (new, hallucinated names) ---")
# This generates that number of samples.
for sample_number in range(number_of_items_to_generate):
    keys: KVCache = [[] for _ in range(n_layer)]
    values: KVCache = [[] for _ in range(n_layer)]
    # This is seeding the generation with the BOS token — the same "start of name" signal the model was trained on.

    # During training, every name was wrapped with BOS on both sides: `[BOS, e,
    # m, m, a, BOS]`. The model learned that BOS at the start means "a name is
    # about to begin" and should be followed by a plausible first character. BOS
    # at the end means "stop". So by setting `token_idx = BOS` here, we're
    # telling the model "you're at the start of a new name — what character
    # should come first?"
    token_idx: int = BOS
    # We build up the outputs in each name this way and accumulate them into a single string.
    generated_chars: list[str] = []
    # Why block_size here?
    # ====
    # It's a safety cap — the maximum number of tokens the model can generate
    # before we force-stop it. In theory the model should stop itself by
    # predicting BOS (the end-of-name signal), and for short names it always
    # will. But `block_size` is there as a hard limit in case the model gets
    # stuck in a loop and never predicts BOS — without it, generation could run
    # forever.
    for position_idx in range(block_size):
        logits: Vector = gpt(token_idx, position_idx, keys, values)
        # What does temperature do here?
        # ====
        # Changes the *shape* of the probability distribution that softmax produces.
        #
        # Imagine the model's logits for three characters are `[2.0, 1.5, 1.0]`:
        #
        # # temperature = 1.0 (no change)
        # softmax([2.0, 1.5, 1.0]) → [0.51, 0.31, 0.18]  # spread out, uncertain

        # # temperature = 0.5 (divide by 0.5 = multiply by 2)
        # softmax([4.0, 3.0, 2.0]) → [0.71, 0.26, 0.10]  # more confident becaue the first one is more dominant

        # # temperature = 0.1 (very small)
        # softmax([20.0, 15.0, 10.0]) → [0.99, 0.007, 0.00003]  # almost deterministic because the first one is super dominant

        probs: Vector = softmax([l / temperature for l in logits])
        # This samples the next token from the probabily distribution
        #
        # Why range(vocab_size)?
        # ===
        # Because that is the set of encoded IDs within our vocabulary that we can choose from.
        #
        # Why `weights=[p.data for p in probs]`?
        # ===
        # Because `random.choices` expects a list of weights corresponding to each element in `range(vocab_size)`.
        #
        # Essentially this causes random.choices to pick an element in accordance with the weight and steer it in that direction.
        token_idx = random.choices(range(vocab_size), weights=[p.data for p in probs])[0]
        if token_idx == BOS:
            break
        generated_chars.append(uchars[token_idx])
    print(f"sample {sample_number+1:2d}: {''.join(generated_chars)}")
