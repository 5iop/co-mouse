# Repository Guidelines

## Project Structure & Module Organization
- `config.py` centralizes hyperparameters, device selection (CUDA-only), and cache settings; update here instead of scattering constants.
- `dataset.py` builds loaders and cached tensors from `boun-mouse-dynamics-dataset/users`; caches live in `cache/`.
- `model.py` defines the Conditional LSTM-VAE; `train.py` drives training/checkpointing to `checkpoints/` and TensorBoard logs under `runs/`.
- `generate.py` loads a checkpoint and writes CSV/PNG trajectories to `outputs/`.
- Test/diagnostics scripts: `run_tests.py`, `test_data_format.py`, `quick_test.py`, `check_gpu.py`, `diagnose_bottleneck.py`.
- Large artifacts (`checkpoints/`, `outputs/`, `runs/`, `cache/`) should stay unversioned.

## Build, Test, and Development Commands
- Install deps: `python -m venv venv && venv\Scripts\activate && pip install -r requirements.txt`.
- Data sanity: `python dataset.py` (verifies loader) and `python model.py` (prints architecture/loss sample).
- Main tests: `python run_tests.py --mode quick` (GPU + dataloader + mini-train), `--mode full` (longer subset), `--mode gpu` (capability only).
- Train: `python train.py --resume checkpoints/checkpoint_epoch_20.pt` (optional resume). Monitor with `tensorboard --logdir runs`.
- Generate: `python generate.py --checkpoint checkpoints/best_model.pt --num_samples 5 --temperature 1.0`.

## Coding Style & Naming Conventions
- Python 3.8+ with 4-space indents; follow PEP8; keep functions/variables `snake_case`, classes `PascalCase`.
- Use explicit configs via `Config` rather than literals; pass devices from config to avoid CPU/GPU drift.
- Prefer type hints and small helper functions over long blocks; keep logging consistent with existing `print_section` helpers.

## Testing Guidelines
- Tests assume CUDA; `Config` raises if missing. Run on a GPU box or patch device logic only locally.
- Add new checks as `test_*.py` at repo root or extend `run_tests.py` modes; include small data paths and deterministic seeds.
- For quick iterations, lower `MAX_SESSIONS_PER_USER` and `BATCH_SIZE`; restore defaults before benchmarks.

## Commit & Pull Request Guidelines
- No existing Git history; use concise, imperative messages (e.g., `Add cache invalidation for format change`).
- PRs should explain motivation, approach, and results; link issues; include runtime/VRAM notes for training changes and sample output/plots when altering generation.
- Avoid committing datasets, caches, checkpoints, `runs/`, or `__pycache__/`; keep secrets out of config/logs.

## Security & Configuration Tips
- Validate dataset paths before training; clear `cache/` when changing preprocessing to prevent stale tensors.
- GPU is mandatory by default; if running CPU-only for debugging, document the temporary patch in your PR and revert before merging.
