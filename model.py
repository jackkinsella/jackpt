from helpers import matrix, linear, softmax, rmsnorm, dot, Vector, Matrix
from value import Value

# Type aliases — used throughout for readability
# Rank3Tensor: a 3D array of Values — "tensor" is the general term for arrays of any number
#              of dimensions: rank-1 = vector, rank-2 = matrix, rank-3+ = tensor
# KVCache:     a rank-3 tensor used to store per-layer key (or value) vectors, one vector
#              per token seen so far. shape: [n_layer][n_tokens_so_far][n_embed]
#              keys and values each have their own separate KVCache instance.
Rank3Tensor = list[list[Vector]]
KVCache     = Rank3Tensor


def build_model(vocab_size: int, n_embed: int, n_head: int, n_layer: int, block_size: int) -> tuple[dict, list[Value], int, int, int, int]:
    # head_dim: the number of dimensions each attention head operates on.
    # n_embed is split evenly across n_head heads, so each head works on a head_dim-sized slice.
    head_dim = n_embed // n_head

    #  state_dict = the model's learnable parameters  - every matrix of weights
    state_dict = {
        #  Weight matrix, Token Embeddings
        # It is a lookup table. When model sees a token, e.g. 'a' with index 0, it looks up what that character is as a learned vector (16 dims here due to n_embed 16)
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
        # When this is reached, the transformer will have `x`, the transformer's internal summary of "given everything I've seen so far, what should come next?" But it's 16 opaque numbers — not a probability distribution over characters. You can't sample from it yet.
        #  You THEN do a matrix multiply: `lm_head @ x`, giving a scalar per row.
        # lm_head[0]  · x  =  0.71   ← score for 'a'
        # lm_head[1]  · x  = -0.32   ← score for 'b'
        # lm_head[2]  · x  =  1.44   ← score for 'c'
        # lm_head[3]  · x  =  0.22   ← score for 'd'
        #
        # A dot product is high when two vectors **point in the same direction**. So what `lm_head` has learned during training is: row `i` points in the direction that `x` points when character `i` is the right answer. If the transformer's internal state `x` is saying "this looks like a name ending in a vowel", then the rows for `'a'`, `'e'`, `'i'` etc will have high dot products with it.
        # This allows the system to give output in terms of the tokens that the user expects
        #
        # lm_head` is essentially asking 27 questions simultaneously — one per token — each of the form: **"does the hidden state `x` look like the situation where this token should come next?"**
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
        # - V: A **value** — a vector optimised for *transferring content*
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
        # searching for something else. K and V are separate for the same reason —
        # the matching signal (K) and the content actually passed along (V) can differ.

        # Head concept introduced in the math at this stage BTW
        # In the `wq`/`wk`/`wv` multiply. That's where the 16D vector gets projected and then **sliced** into 4×4D chunks. Before that point — in `wte`, `wpe`, and the residual stream — everything is just plain 16D vectors with no head concept at all.
        # This system works with multiple heads... however, for mathematical efficiency, the matrix is `[16×16]` which implicitly contains all 4 heads' query projections packed together. After the multiply you slice the 16D result into 4 chunks of 4D
        # x[1] (16D) → wq [16×16] → q (16D) → split → [4D] [4D] [4D] [4D]
        #                                                 head0 head1 head2 head3
        #
        # wo** — goes the other direction. It takes the concatenated 4×4D back to 16D, and here heads are *dissolving* rather than being created. The whole point is that after `wo` there are no more heads — just one unified 16D vector. So you could say heads are a consideration in `wo` too, but in the opposite sense: it's where they stop existing.

        # attn_wq = Weight matrix, Query
        # Query projection: input embedding → query vector
        # "Given my current representation, what pattern am I searching for in the context?"
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
    # 2. Parameters are knobs that backprop tunes (before they are just random noise) - but the forward pass runs, computes loss (how wrong was it), then backprop computes grad for every parameter ("nudge up and down") and gradient descent adjusts them slightly. Repeat millions of times until loss is low.
    params: list[Value] = [p for mat in state_dict.values() for row in mat for p in row]

    return state_dict, params, n_embed, n_head, n_layer, head_dim


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

# Full pipeline
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

# gpt function
# ====
# gpt` is an **autoregressive** function — each call produces one token, and the
# output of one call becomes the input to the next. The "auto" means
# self-referential. Specifically, it is a single autoregressive step.
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
def gpt(token_idx: int, pos_idx: int, keys: KVCache, values: KVCache,
        state_dict: dict, n_head: int, n_layer: int, head_dim: int) -> Vector:

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
    # Why addition
    # ====
    # Combine info about token meaning and position via simple addition!
    # Mathematically: information is lost — the sum is not invertible (i.e. many possible inputs lead to that output)
    # Functionally: no information that the model needs is lost — because the model learns to work with the sum directly, and `wte`/`wpe` co-evolve during training to make the sum as informative as possible
    x: Vector = [t + p for t, p in zip(tok_emb, pos_emb)]  # Vector length n_embed (16)
    x         = rmsnorm(x)                                  # Vector length n_embed (16)

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
        # This is used as `output = f(x) + x` where x is the residual
        #
        # We save this at the top of the attention block because x will get
        # modified later on but we want the original. The idea is that attention
        # only needs to compute a *correction* — what new information should be
        # added — rather than recomputing the entire representation from scratch.
        # The original `x_in` carries through unchanged and the attention output
        # is folded in on top of it — we don't want to throw it out.
        #
        # Math explanation: The chain rule would otherwise cause the gradients to
        # shrink towards zero as things go through each layer so we need some
        # other way to preserve the healthy gradients.
        # For most operations in a neural network, that number is less than 1:
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
        # Without it the entire 16D query has to encode *all* the things the current token is simultaneously searching for, collapsed into a single number per past token.
        # Mathematically, this is because a dot product is a symmetric, linear
        # operation. It can only measure one kind of similarity at a time. If
        # your query is pulling in two different directions at once (e.g. "I
        # want vowels AND I want name-starters"), those two signals interfere
        # with each other in the same dot product. The score for any given past
        # token is one blended number that can't cleanly separate the two
        # questions.
        #
        # By separating out into 4 heads in 4D each, each head has its own independent query, key, and value projection. Head 1's query asks a completely different clean question. They never interfere because they operate in separate 4D subspaces and produce separate weighted sums.
        for head_idx in range(n_head):
            # This is necessary due to how data is represented (four heads concatenated onto a single 16D vector)
            head_start: int       = head_idx * head_dim                          # int — start index of this head's slice
            head_slice: slice     = slice(head_start, head_start + head_dim)     # slice [head_start:head_start+head_dim] — reused for query, keys, values
            query_head: Vector    = query[head_slice]                            # Vector length head_dim (4)
            # Why are key_heads and value_heads matrices whereas query_head is just a vector?
            # =====
            # Because `query_head` is just this one token's question, but `key_heads` is the answer catalogue for every token the model has ever seen in this sequence.
            #
            # `layer_keys` is the KV cache — it has grown by one entry every
            # time `gpt()` was called. So on call 10, `layer_keys` has 10 key
            # vectors in it, one per past token. `key_heads` slices out this
            # head's 4D chunk from each of those 10 vectors, giving a matrix of
            # 10 × 4D vectors — one row per past token.
            #
            # Basically, the keys/values are always plural because attention is the act of comparing that one token against the entire history.
            key_heads: Matrix     = [key_vec[head_slice] for key_vec in layer_keys]    # Matrix shape [n_tokens][head_dim]
            value_heads: Matrix   = [val_vec[head_slice] for val_vec in layer_values]  # Matrix shape [n_tokens][head_dim]
            # The attn_logits bit computes one attention score per past token — "how relevant is each past token to what I'm currently searching for?"
            #
            # Why do we divide by `head_dim ** 0.5`? (i.e. sqrt of head_size)
            # ====
            # This is needed to prevent the scale of the numbers getting too large and causing problems for softmax below (specifically, one
            # score becomes close to 1.0 and everything else collapses to 0, and
            # gradient is nearly 0 so backprop stalls). Dot products
            # naturally grow larger as the number of dimensions increases — with
            # 4 dimensions you're summing 4 products. So a larger `head_dim`
            # produces larger raw scores, purely as a side effect of having more
            # terms in the sum, not because the vectors are actually more
            # similar — therefore it needs some correction.
            #
            # Why divide by sqrt of head_size instead of head_size?
            # =====
            # When you sum `head_dim` independent random terms, each with
            # variance 1, the total variance is `head_dim` — but the **standard
            # deviation** (which is what actually determines the spread of
            # values on the number line) is √`head_dim`. Standard deviation is
            # the square root of variance.
            attn_logits: Vector   = [dot(query_head, key_heads[token_idx]) / head_dim**0.5 for token_idx in range(len(key_heads))]  # Vector length n_tokens — raw attention scores
            # attn_logits` at this point is a vector of raw scores — one per
            # past token — saying "how relevant is each past token to what I'm
            # searching for?" But they're unbounded numbers, could be anything
            # like `[0.3, -1.2, 2.1, 0.8]`. You can't use them directly as
            # mixing weights because they don't sum to 1 and could be negative.
            #
            # softmax` converts them into a proper probability distribution — all positive, all summing to 1:
            attn_weights: Vector  = softmax(attn_logits)                                                                            # Vector length n_tokens — probabilities summing to 1
            head_out: Vector      = [dot(attn_weights, [value_heads[token_idx][dim_idx] for token_idx in range(len(value_heads))]) for dim_idx in range(head_dim)]  # Vector length head_dim (4) — weighted sum of value vectors
            # This is the result of all this attention stuff (calculated per head and accumulated in this structure)
            attn_output.extend(head_out)  # concatenate this head's 4D output into attn_output

        # The job of `attn_wo` is to mix the 4 heads' outputs together. Each
        # head independently produced a 4D chunk representing what it found, and
        # those chunks are just concatenated end-to-end in `attn_output`.
        # Without `attn_wo`, the heads would never interact — head 0's findings
        # would stay in dimensions 0-3 and never influence how head 1's findings
        # in dimensions 4-7 are interpreted. `attn_wo` is what learned, during
        # training, how to blend those four independent signals into one
        # coherent updated representation.
        x_in = linear(attn_output, state_dict[f'layer{layer_idx}.attn_wo'])
        # This step ensures that the residual freeway is in place in the mathematics.
        x_in = [attn_out + residual for attn_out, residual in zip(x_in, x_residual)]

        # 2) MLP block
        # =====
        #
        # Another residual?
        # ======
        # This is a separate one to the attention block. By the time we get to this line, x_in already has the attention block's output folded into it. It just saves that new one as a baseline.
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
        # sheet of paper rather than keeping everything in your head.
        x_in = linear(x_in, state_dict[f'layer{layer_idx}.mlp_fc1'])                  # Vector length 4*n_embed (64) — expand
        # without it, stacking linear transformations (matrix multiplies) on top
        # of each other is mathematically equivalent to just one matrix
        # multiply. No matter how many `linear()` calls you chain together, you
        # could always collapse them into a single matrix. The network would
        # have no more expressive power than a single layer. This is what lets
        # the MLP learn genuinely non-linear functions.
        #
        # After relu, the 64D vector contains a sparse pattern of activations —
        # some neurons fired, most are zero.
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
    # You can't sample from 16 numbers. You need one score per possible next token — 27 scores (one per character in the vocabulary). That's what `lm_head` does.
    #
    # Each of its 27 rows is a 16D vector that has learned, during training, to point in the direction that `x` points when a particular token is the right answer.
    # The dot product is high when two vectors point in similar directions. So if `x` is saying "this looks like a name about to end in a vowel", the rows for `'a'`, `'e'`, `'i'` will have high dot products with it, producing high logits for those tokens.
    logits: Vector = linear(x, state_dict['lm_head'])           # Vector length vocab_size (27)
    return logits
