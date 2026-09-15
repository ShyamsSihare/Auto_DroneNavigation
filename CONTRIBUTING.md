# Contributing

Contributions, issue reports, and replication attempts are welcome. A few project-specific notes before you start:

## Ground rules

1. **No fabricated numbers.** This project's core value proposition is that every figure, table, and claim is a real, reproducible output of the code in `src/`. If you add a result, add the code that produces it, and make it reproducible via the existing checkpoint mechanism (see `docs/ARCHITECTURE.md` → "Checkpointing").
2. **Gradient changes require a self-test.** If you modify `VQC.forward_and_param_shift_grad_batch`, `MLP.backward`, or any other hand-derived gradient, add or update the corresponding finite-difference check in `tests/test_gradients.py` in the same PR.
3. **State reductions in budget honestly.** If you reduce an episode/seed count for tractability, say so in a code comment next to the change (following the existing `# [manuscript: ...]` convention in `Config`), not just in a commit message.

## Development setup

```bash
git clone <this-repo> && cd qacldn
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest tests/ -v          # should complete in <5s
```

## Before opening a PR

- [ ] `pytest tests/ -v` passes
- [ ] `python -m py_compile src/analysis.py` succeeds (no syntax errors)
- [ ] If you changed `src/analysis.py`, regenerate the notebook:
      `jupytext --to notebook src/analysis.py -o notebooks/QACLDN_experimental_analysis.ipynb`
- [ ] If you changed any figure-generating cell, re-execute the notebook and confirm all outputs render with zero error cells
- [ ] If you changed reward weights, physical parameters, or agent hyperparameters, update `docs/METHODOLOGY.md` to match (do not let the docs drift from the code)

## Reporting issues

Please include:
- The exact command you ran and the full traceback
- Whether you're using the vendored `checkpoint.pkl` behaviour or a clean run
- Your `numpy`/`scipy`/`pandas`/`matplotlib` versions (`pip freeze | grep -E "numpy|scipy|pandas|matplotlib"`)
