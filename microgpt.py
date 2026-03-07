# See https://karpathy.github.io/2026/02/12/microgpt/
import os
import math
import random

random.seed(42)

# Corpus Grabbing and Preparation
# ======

def get_corpus_and_persist_locally():
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
    def backward(self):
        # out and visited are initialized here so we can avoid resetting them each recursive call
        topo = []
        visited = set()
        def build_topological_graph(node: 'Value'):
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

# Every token (character) gets converted into a vector of this size. So each token is represented as 16 numbers. Bigger = the model can encode more nuanced meaning per token, but more parameters to train.
n_embed = 16
# In self-attention, instead of one big attention computation, you split `n_embd` into `n_head` parallel "heads" — each one attends to different aspects of the sequence (e.g. one head might learn grammar, another learns position). Each head works on `n_embd / n_head = 4` dimensions. They all run in parallel then get concatenated back to n_embd:
#
#   input:         [n_embd = 16]
#   split:         [4] [4] [4] [4]   ← 4 heads, each size 4
#   each head:     [4] [4] [4] [4]   ← attend independently
#   concatenate:   [n_embed = 16]     ← back to original size
n_head = 4
# Number of transformer blocks (layers) stacked on top of each other.
# Each layer (technically a "TransformerBlock" containing a "MultiHeadAttention" + "FeedForward" sublayer)
# takes the output of the previous one as input — same shape [n_embd] in and out, so they stack cleanly.
# Each layer learns increasingly abstract patterns:
#
#   Layer 1 (shallow)  — raw token patterns:        "'i' often follows 'l'"
#   Layer 2            — higher order patterns:      "this looks like a name ending"
#   Layer 3 (deep)     — abstract representations:  "this is a feminine name pattern"
#
# Analogy: editing a document in passes — first pass fix spelling, second fix grammar, third fix flow.
# Each pass builds on the previous one.
n_layer = 1
head_dim = n_embd // n_head
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
def matrix(nout: int, nin: int):
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
    # wpe[3]` is the same vector whether position 3 contains `'a'` or `'z'` or BOS. It encodes *"what does it mean to be in slot 3"*, nothing more.
    'wpe': matrix(block_size, n_embed),
    #  Language Model Head
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
