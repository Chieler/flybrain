# Local Review — uncommitted changes

**Reviewed**: 2026-09-12
**Branch**: master (uncommitted working tree)
**Decision**: APPROVE with comments

## Summary

Phase 2 neural benchmark (`simulation.py`) + Phase 5 pygame viewer (`viewer.py`) +
pygame pin + HANDOFF docs. No CRITICAL/HIGH. No security issues (no credentials,
no injection, no network; CLI path args are local dev only). All `viewer.py` calls
into `evaluate`/`brain`/`simulation` statically verified against sources — every
symbol and checkpoint key (`gain`, `bias`, `data_dir`, `params`) matches.

## Findings

### CRITICAL
None.

### HIGH
None.

### MEDIUM
None.

### LOW

- **viewer.py:118 — dead guard.** `if args.controller == "neural" and not (args.data or args.checkpoint)`.
  `--checkpoint` defaults to `"checkpoint.json"`, so `args.data or args.checkpoint` is
  always truthy — `p.error` never fires. Harmless (missing file raises `FileNotFoundError`
  at `load_checkpoint`) but gives a false sense of validation. Fix: default `--checkpoint`
  to `None`, or drop the guard.

- **simulation.py:232 — `b.params.neural_dt` mutated, not restored.** `_neural_benchmark`
  leaves `neural_dt=0.005` on the Brain after the dt loop. Throwaway CLI path, process exits
  immediately → no real impact. Acceptable; a `ponytail:` note is the most it warrants.

- **viewer.py — no test couples the suite to pygame.** Deliberate: headless smoke-check done,
  viewer is a thin shell over the already-tested `run_episode`. Noted, not a defect.

## Validation

| Check | Result |
|---|---|
| Type check | Skipped (no mypy config) |
| Lint | Skipped (none configured) |
| Tests | Not re-run — changes don't touch the 26-test suite paths; viewer API calls statically verified |
| Build | N/A (Python) |

## Files reviewed

- `simulation.py` — Modified (added `_neural_benchmark`, CLI dispatch)
- `viewer.py` — Added (pygame replay)
- `requirements.txt` — Modified (pygame==2.6.1 pin)
- `HANDOFF.md` — Modified (docs; stated numbers match benchmark output)

Both LOW items optional. Dead-guard one-line fix: default `--checkpoint` to `None`.
