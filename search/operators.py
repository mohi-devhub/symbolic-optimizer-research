import torch

# Safe versions of functions to avoid NaNs or explosions
def add(x, y): return x + y
def sub(x, y): return x - y
def mul(x, y): return x * y
def div(x, y): return x / (y + 1e-8)
def sign(x): return torch.sign(x)
def interp(x, y, alpha): return (1 - alpha) * x + alpha * y

# Dictionary of available symbolic operations
OPERATORS = {
    "add": add,
    "sub": sub,
    "mul": mul,
    "div": div,
    "sign": sign,
    "interp": interp,
}
