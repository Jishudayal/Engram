# Contributing to Engram

## Setup

```bash
git clone https://github.com/engram-ai/engram
cd engram
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Running tests

```bash
make test
```

## Code style

```bash
make lint    # check
make format  # fix
```

## Adding a storage adapter

1. Create `engram/adapters/your_backend.py`
2. Implement all methods from `engram.core.adapter.AbstractAdapter`
3. Add optional dependency group to `pyproject.toml`
4. Add tests under `tests/unit/adapters/`

## Submitting a PR

- Keep PRs focused — one thing per PR
- Add tests for new behaviour
- Run `make lint` and `make test` before opening
- Reference the issue number in the PR description

## Good first issues

Check the [issues](https://github.com/engram-ai/engram/issues) tab for tickets labeled `good first issue`.
