# Symbolic Optimizer Reimplementation (Lion)

This project reimplements the Lion optimizer from the NeurIPS 2023 paper:
**"Symbolic Discovery of Optimization Algorithms"**.

## Goals

- Implement the Lion optimizer.
- Train a model on a simple task using Lion and compare with AdamW.
- (Later) Build a minimal symbolic program search simulator.
- Document the process and publish it on GitHub.

## Project Timeline

- **Day 1**: Environment setup, paper reading, GitHub repo.
- **Day 2**: Implement Lion and baseline AdamW.
- **Day 3**: Simulate symbolic program search (optional).
- **Day 4**: Evaluate Lion, visualize, finalize repo.

## Installation

Run this to install dependencies:

```bash
pip install torch numpy matplotlib sympy
```

## Usage

To run training and compare Lion with AdamW:

```bash
python train.py

```

## Symbolic Optimizer Search

Run a simplified evolutionary search:

```bash
python run_search.py
```
