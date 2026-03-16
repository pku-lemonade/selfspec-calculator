## Context

`Roadmap.md` defines the intended comparison more narrowly than the current estimator:

- conventional full-precision inference is "stop-and-go" and pays setup each token,
- draft generation is serialized because it is autoregressive,
- verify uses a pipelined burst where setup is paid once,
- the verifier always computes `K+1` burst steps and discards the unusable suffix after mismatch.

The current `layer-pipelined` implementation diverges in two ways:

1. It treats `K=0` as a pipelined token period, which makes the baseline artificially optimistic.
2. It uses mismatch-stop semantics for verify latency/energy, which contradicts the Roadmap's "always computes `K+1` tokens" contract.

## Goals / Non-Goals

**Goals:**
- Make `K=0` in `layer-pipelined` mode behave like the Roadmap's stop-and-go full-precision baseline.
- Make verify-burst latency/energy in `layer-pipelined` mode independent of acceptance outcome, except for committed-token normalization and commit-only writes.
- Keep draft serialized in all cases.
- Preserve report compatibility while updating semantics/metadata.

**Non-Goals:**
- Do not redesign the overlap fixes added in earlier changes.
- Do not add a new schedule mode; this change corrects semantics inside existing `layer-pipelined`.
- Do not build a token-level global simulator.

## Decisions

### 1. `K=0` bypasses the pipelined verify-burst timing path

When `stats.k == 0`, `layer-pipelined` will not reinterpret the token as a verify burst throughput calculation. It will use the existing serialized full-precision token latency already produced by the phase accounting.

Rationale:
- with no drafted burst, there is no verify burst to amortize;
- this directly matches the Roadmap's stop-and-go baseline.

### 2. Verify burst executes fixed `K+1` work in `layer-pipelined`

For `stats.k > 0`, the pipelined verify burst will always execute:
- `K` drafted-token verify steps, and
- `1` bonus verify step,
regardless of accepted-prefix outcome.

Acceptance still affects:
- committed tokens per burst `E[a+1]`,
- commit-only HBM writes,
- energy/token and latency/token normalization.

Rationale:
- this follows `Roadmap.md` §5.2.2 and §5.3.3 exactly.

### 3. Pipelined verify burst latency is fixed per `(K, L_prompt)` point

The pipelined burst latency in `layer-pipelined` mode becomes:

`T_burst = T_draft_serial + sum(T_verify_step_i for i=0..K-1) + T_verify_bonus`

where each verify step period remains the bottleneck of:
- per-layer compute throughput, and
- shared memory service time.

The per-burst setup term remains amortized once through the bonus/setup path already modeled in the burst tail.

Rationale:
- fixed burst execution is the Roadmap contract;
- acceptance no longer changes executed verify work.

### 4. Report metadata must stop claiming mismatch-stop behavior

Pipeline metadata and notes will be updated to reflect:
- fixed `K+1` verify-burst execution,
- stop-and-go baseline for `K=0`,
- acceptance affecting committed outputs, not verify execution length.

## Risks / Trade-offs

- [Breaks prior fine-grained mismatch-stop tests] -> Mitigation: update tests to the Roadmap contract and keep change scoped to `layer-pipelined`.
- [Roadmap vs prior OpenSpec artifacts conflict] -> Mitigation: this change is explicitly justified as a semantics correction to match the Roadmap source of truth.
- [Break-even numbers shift significantly] -> Mitigation: this is the intended outcome; previous comparisons were using an invalid baseline.

## Migration Plan

1. Update `layer-pipelined` timing branch in `estimate_point`.
2. Skip pipelined reinterpretation when `K=0`.
3. Remove acceptance-conditioned verify-step execution from pipelined latency/energy.
4. Update report notes / pipeline metadata.
5. Replace mismatch-stop tests with fixed-verify-burst tests and baseline-semantics tests.

## Open Questions

- Should a follow-up change introduce a separate explicit schedule label for "Roadmap verify-burst pipelined" vs any future more aggressive overlap mode?
