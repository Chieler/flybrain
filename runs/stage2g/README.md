# Stage 2g — fully-trainable recurrent baseline

Trains an n=8 echo-state net end-to-end (W_in + W + W_out = 190 params) by
block-scaled CEM against the frozen Stage 2f aligned reward, with a matched
recurrence-off ablation (126 params, W=0 / leak=1). Fresh train(70)/dev(71)/
gate(72) splits are frozen and sha256-hashed before training; the gate is scored
once. Escalation (dev arrivals not improved, or go/no-go failed) is a HARD HALT:
n=16 + CMA-ES is not implemented and requires a written spec amendment first.

Spec: `docs/superpowers/specs/2026-09-21-stage2g-trainable-recurrent-design.md`.
Run: `python evaluate_stage2g.py`. Results (filled in after the one-shot run):
recurrent __, ablation __. Interpretation is bounded (pass-only asymmetry; both-pass
scoped to "recurrence not necessary within this gate and policy family"; no matched
n=8 readout-only control, so exact attribution is not claimed).
