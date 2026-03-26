## Context

The Roadmap describes verify burst timing as a pipelined sequence of tokens after an initial fill cost. The current estimator uses only the steady-state verify period and therefore omits the first-token serialized latency through the layer pipeline.

## Goals / Non-Goals

**Goals:**
- Add a serialized first-token verify fill term for `layer-pipelined` verify bursts.
- Keep existing draft-serialized semantics unchanged.

**Non-Goals:**
- Do not redesign the overall schedule modes.

## Decisions

- Verify burst latency for `K>0` will be:
  - first drafted verify token: fully serialized verify latency,
  - remaining drafted verify tokens: bottleneck step periods,
  - bonus token: one bonus bottleneck period.

## Risks / Trade-offs

- Results will shift upward for larger `K`; this is expected and correct.
