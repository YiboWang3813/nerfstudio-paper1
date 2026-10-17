# Pre-existing rotated local-covariance discrepancy

This is a P0 discovery, not a P0 algorithm change. Production has intentionally
not been corrected without a decision about the default-path preservation rule.

`ops/ours_renderer/include/shader/shared.cuh::ours_quat_to_rot` creates row-major R.
`plane_cond2d_device.cuh::mat3_sigma_world` first constructs M = R diag(s²), then uses:

```cpp
Sigma_w[i*3+j] = M[i*3+0]*Rg[j]
              + M[i*3+1]*Rg[3+j]
              + M[i*3+2]*Rg[6+j];
```

These are the columns of R, so the result is M R, not M R-transpose. The
conventional Gaussian covariance would index `Rg[j*3+0/1/2]` instead. The same
helper is called by the pre-P0 and post-P0 local renderer. The global ray
integral uses its own rotated/whitened Gaussian computation.

Concrete algebraic counterexample: let R be a 45-degree axial rotation and
diag(s²)=diag(1,4,1). The leading 2x2 block of R diag(s²) R is
`[[-1.5,-2.5],[2.5,1.5]]`, whereas R diag(s²) R-transpose is
`[[2.5,-1.5],[-1.5,2.5]]`. The former is not symmetric and is not a Gaussian
covariance. Consequently, changing the Methods covariance formula to the
current code's formula would not be a scientifically adequate wording fix.

The existing PyTorch mathematical reference uses R diag(s²) R-transpose. P0
found substantial CUDA/reference output differences for nonidentity
quaternions in all three local modes. Identity-quaternion comparisons and
pre/post-default comparisons are separate tests; passing those does not resolve
this discrepancy. Rotated CUDA forward/backward finite-difference checks test
the actual implementation, not whether it realizes the intended Gaussian math.

Decision required before claiming full Methods/code semantic freeze: authorize
a separately scoped production covariance correction and reassess affected
results, or explicitly keep this P0 as an infrastructure-only milestone while
the discrepancy remains unresolved. P0 does not silently fix this helper or
invalidate Stage-2 merely because ablation interfaces were added.

The source archive, pre/post probes, and logs under
`outputs/preproduction/regression/p0_semantic_freeze_20260908/` preserve the
evidence. The current plan counts assume the existing full path is retained;
they are not final budgets for a future covariance-corrected method.
