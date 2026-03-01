## Context

The estimator currently computes dynamic energy from operation and movement counts. There is no static/leakage term, so long-latency operating points are undercounted in total energy. This change must remain small and backward compatible while adding a physically meaningful leakage term.

## Goals / Non-Goals

**Goals:**
- Add per-component leakage power knobs with default-zero behavior.
- Compute leakage energy from total leakage power and effective burst latency.
- Keep schedule semantics correct: serialized uses serialized burst time; layer-pipelined uses pipelined burst time.
- Include leakage contribution in final energy/token metrics and report output.

**Non-Goals:**
- Modeling fine-grained power-gating or component duty-cycle transitions.
- Adding cycle-accurate idle/active state simulation.
- Changing existing dynamic-energy counting formulas.

## Decisions

### 1) Leakage power configuration mirrors modeled components
- **Decision:** Introduce a leakage-power structure with one field per modeled component (arrays, DAC/ADC paths, analog periphery blocks, attention/softmax/elementwise engines, buffers/control, SRAM/HBM/fabric), in mW.
- **Rationale:** Matches user requirement to specify leakage for all components and keeps coefficient ownership explicit.
- **Alternative considered:** Single scalar chip leakage power. Rejected because it loses component attribution and calibration flexibility.

### 2) v1 leakage is always-on over burst wall-clock time
- **Decision:** For each burst, leakage energy is `E_leak_burst_pJ = (sum component leakage mW) * T_burst_effective_ns`.
- **Rationale:** mW·ns naturally maps to pJ and provides a simple, stable first-order model.
- **Alternative considered:** Per-stage or per-component active-time weighting. Rejected for this small change due to complexity and missing utilization signals.

### 3) Effective burst time follows selected schedule
- **Decision:** Use serialized burst latency in serialized mode and pipelined burst latency in layer-pipelined mode.
- **Rationale:** Leakage tracks real elapsed time, so schedule-dependent timing must drive leakage energy.
- **Alternative considered:** Always use serialized phase-sum latency. Rejected because it overcharges leakage in pipelined mode.

### 4) Leakage is additive to total energy, preserving dynamic accounting
- **Decision:** Keep dynamic accounting unchanged, then add leakage at burst-total level before per-token normalization.
- **Rationale:** Low-risk integration and backward compatibility with existing breakdown logic.
- **Alternative considered:** Redistribute leakage across stage/channel breakdowns in v1. Rejected as unnecessary for initial rollout.

### 5) Report includes explicit leakage summary
- **Decision:** Add leakage summary metadata to report (total leakage power and leakage energy per point) so users can audit contribution.
- **Rationale:** Avoids hidden energy shifts and supports validation/debug.
- **Alternative considered:** No dedicated leakage report fields. Rejected due to poor observability.

## Risks / Trade-offs

- **[Model simplification]** Always-on leakage can overestimate when real hardware gates idle blocks.  
  **Mitigation:** Keep defaults at zero and document assumption; future extension can add duty-cycle factors.

- **[Schema growth]** Per-component leakage fields increase config surface area.  
  **Mitigation:** Default all fields to zero and keep names aligned with existing component taxonomy.

- **[Metric interpretation]** Users may confuse power vs energy units.  
  **Mitigation:** Document units clearly (`mW`, `pJ`, `ns`) and formulas in README/report notes.

- **[Backward-compatibility risk]** Existing tests may assume purely dynamic totals.  
  **Mitigation:** Keep zero defaults and add regression tests showing no-change behavior at zero leakage.
