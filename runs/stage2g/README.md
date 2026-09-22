# Stage 2g — fully-trainable recurrent baseline

Trains an n=8 echo-state net end-to-end (W_in + W + W_out = 190 params) by
block-scaled CEM against the frozen Stage 2f aligned reward, with a matched
recurrence-off ablation (126 params, W=0 / leak=1). Fresh train(70)/dev(71)/
gate(72) splits are frozen and sha256-hashed before training; the gate is scored
once. Escalation (dev arrivals not improved, or go/no-go failed) is a HARD HALT:
n=16 + CMA-ES is not implemented and requires a written spec amendment first.

Spec: `docs/superpowers/specs/2026-09-21-stage2g-trainable-recurrent-design.md`.
Run: `python evaluate_stage2g.py`. Interpretation is bounded (pass-only asymmetry;
both-pass scoped to "recurrence not necessary within this gate and policy family";
no matched n=8 readout-only control, so exact attribution is not claimed).

## One-shot result (2026-09-22, gate spent)

Both arms **FAILED** the 0.90/0.80 gate. Recurrent overall **0.310** (cross 0.583,
regular 0.250, asymmetric 0.295); recurrence-off ablation overall **0.200** (cross
0.417, regular 0.159, asymmetric 0.182). Training precondition held (not escalated):
recurrent go/no-go True, warm dev arrivals 15/46 -> trained 16/46 (a single-arrival
gain over 30 CEM iterations). See `gate_results.json` / `results.json`.

Bounded reading: neither arm cleared the fresh gate, so under the pass-only asymmetry
the outcome is confounded between optimization budget, capacity (n=8) and coverage and
does NOT implicate the observation interface. The recurrent arm beats its ablation by
0.11 overall — suggestive that recurrence helped under this budget, but no positive
claim is licensed since neither passed. Note the end-to-end 190-param black-box CEM
search here (0.310) underperforms the readout-only Stage 2f recurrent gate (0.660):
consistent with the higher-dimensional search being harder under the same budget, not
evidence about the interface. Advancing requires the n=16 + CMA-ES spec amendment.
