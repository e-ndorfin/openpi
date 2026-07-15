# Repository guide

OpenPI contains robotics vision-language-action models, training and inference code, and a lightweight remote-policy client. Preserve the distinction between the Python 3.11 main project and the Python 3.7-compatible client package.

## Layout

- `src/openpi/models/`: JAX/Flax model definitions, tokenizers, LoRA support, and shared model interfaces.
- `src/openpi/models_pytorch/`: PyTorch implementations and pinned Transformers replacement files.
- `src/openpi/policies/`: robot/environment adapters and the common inference policy API.
- `src/openpi/training/`: training configs, data loaders, checkpointing, sharding, optimizers, and weight loading.
- `src/openpi/shared/` and `src/openpi/transforms.py`: shared utilities, array typing, normalization, image handling, and data transforms.
- `packages/openpi-client/`: minimal websocket client package; it supports Python 3.7 and must not acquire main-project dependencies.
- `scripts/`: training, normalization-statistics, and policy-serving entry points.
- `examples/`: platform-specific conversion, inference, and evaluation workflows.
- `third_party/`: git submodules. Do not edit vendored/submodule code unless the task explicitly requires it.

## Environment and commands

- Use `uv`; the checked-in `.venv` may already be populated.
- Initial setup: `git submodule update --init --recursive`, then `GIT_LFS_SKIP_SMUDGE=1 uv sync` and `GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .`.
- Run a focused test first: `uv run pytest path/to/test_file.py -q`.
- Run the normal test suite with `uv run pytest --strict-markers -m "not manual"`.
- Lint with `uv run ruff check .`; format with `uv run ruff format .`.
- The full repository check is `uv run pre-commit run --all-files`; it can update `uv.lock` and auto-fix Ruff findings, so inspect the resulting diff.
- Training and checkpoint-backed inference are GPU- and resource-intensive. Do not launch them as routine validation.
- Tests marked `manual` may download large checkpoints or require special hardware/services; exclude them unless explicitly requested.

## Coding conventions

- Follow Ruff settings in `pyproject.toml`: 120-character lines, Python 3.11 for the main package, single-line sorted imports, and the configured lint rule set.
- In `packages/openpi-client`, retain Python 3.7 syntax and its minimal dependency boundary.
- Add or update colocated `*_test.py` tests for behavior changes. Prefer small synthetic fixtures and fake observations over checkpoint downloads.
- Preserve array shape, dtype, and batching contracts. Model inputs flow through dataset-specific repacking, data transforms, normalization, model transforms, and finally `model.Observation`/actions.
- Keep JAX and PyTorch paths behaviorally aligned when changing shared model or policy interfaces, and test both where the code path applies.
- Add training variants through the typed factories and registered configs in `src/openpi/training/config.py`; keep policy transforms usable by both training and inference.
- Treat files under `src/openpi/models_pytorch/transformers_replace/` as pinned upstream replacements, not ordinary application modules.

## Operational cautions

- `assets/`, `checkpoints/`, `data/`, and `wandb/` are local/generated data and are gitignored. Do not commit model weights, datasets, run output, or secrets.
- Check `git status` before editing and preserve unrelated user changes in this experimental codebase.
- Policy servers default to port 8000. This workspace is on a remote cluster, so when handing off a server include an SSH tunnel command such as `ssh -L 8000:localhost:8000 zachtang@165.132.144.103`.
