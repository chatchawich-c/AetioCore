# Contributing to semflow

We welcome bug reports, feature requests, documentation improvements, and pull requests from researchers and developers across Python, R, and Rust.

## Development Setup

1. Clone the repository and install Rust (`rustup`) and Python ($\ge 3.10$).
2. Build and test the pure Rust core:
   ```bash
   cargo test --workspace --exclude sem-py
   ```
3. Build the Python extension in a virtual environment:
   ```bash
   python3 -m venv .venv && source .venv/bin/activate
   pip install maturin numpy polars pytest
   maturin develop --release
   ```
4. Run numerical parity tests against `lavaan` and `cSEM` golden fixtures:
   ```bash
   pytest -s benchmarks/test_parity.py
   ```

## Pull Request Guidelines

* **Parity Gate:** Any new statistical estimator must include a golden parity test verifying agreement with established reference implementations (`lavaan` or `cSEM`) within documented tolerances.
* **Zero-Copy Preservation:** Avoid unnecessary $N \times p$ allocations inside inner optimization loops.
* **Code Style:** Run `cargo fmt --check` and `cargo clippy --workspace --exclude sem-py` before submitting.

## Reporting Issues

Please open a GitHub Issue with a minimal reproducible example (formula string, synthetic data snippet, OS, and Python/R version).
