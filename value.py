import math

class Value():
    """
     Briefly, a Value wraps a single scalar number (.data) and tracks how it was
     computed. Think of each operation as a little lego block: it takes some
     inputs, produces an output (the forward pass), and it knows how its output
     would change with respect to each of its inputs (the local gradient).
     That's all the information autograd needs from each block. Everything else
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
