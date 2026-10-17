# Proposed Batch2 — NOT authorized, NOT submitted

Exactly52 new30k runs on Liver/Arm representatives, MP+Fixed+Full+FullCGA,
same P2 freeze/seed/split.54 unique configs include2 reused full defaults.
No PPP reruns. No gamma/Tmin or additional renderer panels.

- LGR beta[.4,.8,1],lambda[.2,.55,1],Nq[1,2,3],half-window[1,2,4]mm:
  16new runs after default reuse.
- CGA qd[.94,.98,.99] x B[.001,.005,.05]:16new runs.
- CGA qp[.05,.10,.20] x qA[.6,.8,.9]:16new runs.
- tau_alpha[1e-5,1e-4,1e-3]:4new runs.
- Quality–population panel: reuse final five CGA variants,0new runs. It is a
  cross-variant trade-off, not a matched-capacity causal isolation. If a true split
  threshold panel is preferred instead, [.4,.6,1]mm adds4new runs (total56),
  not included in this proposed52-run authorization request.

Budget rationale: final Fixed default selected responsibility fractions are
approximately.00174–.00340(Liver) and.00091–.00156(Arm). The old B=.02/.05/.12
grid would mostly leave the budget inactive. B=.001/.005/.05 spans a potentially
binding cap and the unchanged loose default. This is a hypothesis to test, not
an asserted plateau. Final default local-contribution thresholds were0; if qp
points also yield zero thresholds, report quantile saturation rather than
claiming unrestricted robustness. Inspect event thresholds along with quality/#G.

Ranks/test optima do not retune defaults. These52 runs are only a proposed plan;
all execution_authorized flags are false. Await explicit user approval.
