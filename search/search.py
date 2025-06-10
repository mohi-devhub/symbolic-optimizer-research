import torch
import random
from search.operators import OPERATORS

# Toy gradient descent task: minimize (x - 5)^2
def evaluate_program(program, epochs=20):
    x = torch.tensor([0.0], requires_grad=True)
    optimizer = torch.optim.SGD([x], lr=0.1)
    
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = (x - 5) ** 2
        loss.backward()

        # Fake optimizer step using symbolic program
        grad = x.grad.detach()
        momentum = torch.tensor([0.0])
        lr = 0.1
        update = execute_program(program, grad, momentum, lr)
        x.data -= update
    return (x.item() - 5) ** 2  # Final distance from 5

def execute_program(program, grad, momentum, lr):
    x, y = grad, momentum
    for op in program:
        func = OPERATORS[op]

        if op == "interp":
            x = func(x, y, 0.9)  # 3-arg version
        elif op in ["sign"]:  # unary ops
            x = func(x)
        else:  # binary ops
            x = func(x, y)
    return lr * x


def random_program(length=3):
    return random.choices(list(OPERATORS.keys()), k=length)

def mutate_program(program):
    new_prog = program[:]
    i = random.randint(0, len(program)-1)
    new_prog[i] = random.choice(list(OPERATORS.keys()))
    return new_prog

# Search loop
def run_symbolic_search(rounds=20):
    best_prog = random_program()
    best_score = evaluate_program(best_prog)

    for _ in range(rounds):
        candidate = mutate_program(best_prog)
        score = evaluate_program(candidate)
        if score < best_score:
            best_prog, best_score = candidate, score
            print("New best:", candidate, "Score:", score)

    print("Final best program:", best_prog, "Score:", best_score)
