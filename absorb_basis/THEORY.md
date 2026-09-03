# THEORY.md — Formal analysis (paper §4 draft)

目的：把 T1–T5 寫成可直接進論文的定理與證明；對照 SVDQuant 論文
（arXiv:2411.05007）§4.1–4.2 的 Prop 4.1/4.2，逐步顯示「他們優化的
每個量都是我們對應目標的鬆弛」。實驗閉環：`svdbasis_qdiff128.json`
（λ→∞ = plain-SVD basis 放進我們 pipeline 的直接對照）。

Notation. A linear layer computes `Y = X Wᵀ` with weight `W ∈ R^{m×n}`
(m outputs, n inputs) and calibration activations stacked as
`X ∈ R^{N×n}`. Let `H = XᵀX ⪰ 0` be the (unnormalized) second-moment
matrix, `m̄ = mean(diag(H))`, and for `λ > 0` define the damped metric
matrix `H_λ = H + λ m̄ I ≻ 0` with Cholesky factor `H_λ = C Cᵀ`.
For any `A ∈ R^{m×n}` define the (semi-)norm

    ‖A‖_H := ( tr(A H Aᵀ) )^{1/2},     ‖A‖_{H_λ} analogously.

`Q(·)` denotes a quantizer; `σ_i(·)` singular values in decreasing
order. SVDQuant's decomposition (their §4.2) is
`Ŵ = L₁L₂ + R` with `L₁L₂` the rank-r **unweighted** truncated SVD
(Eckart–Young in ‖·‖_F). Ours is `W = L* + R*` with `L*` the rank-r
minimizer of `‖W − L‖_{H_λ}` (H-SVD), computed as
`L* = (WC)_r C⁻¹` where `(·)_r` is SVD truncation.

---

## T1 — The deployed objective is an identity, not a bound

**Proposition 1.** For any weight perturbation `Δ ∈ R^{m×n}`
(e.g. `Δ = R − Q(R)`),

    ‖X Δᵀ‖_F² = tr(Δ H Δᵀ) = ‖Δ‖_H².

*Proof.* `‖XΔᵀ‖_F² = tr(ΔXᵀXΔᵀ)`. ∎

Remark. The first term of SVDQuant's Prop 4.1 bounds this same
quantity by `‖X‖_F ‖Δ‖_F` (see T4). We take the left-hand side itself
as the design objective throughout: basis selection (T2), robustness
(T3), residual quantization (T4), and smoothing (T5) all optimize
`‖·‖_H`-type functionals of the *same* quantity.

---

## T2 — Weighted Eckart–Young: the basis is optimal for the true
## objective, and dominates plain SVD

**Theorem 2 (optimality).** Over all matrices `L` of rank ≤ r,

    L* = argmin_L ‖W − L‖_{H_λ}   is given by   L* = (W C)_r C⁻¹,

with optimal value `‖W − L*‖_{H_λ}² = Σ_{i>r} σ_i²(WC)`. The minimizer
is unique iff `σ_r(WC) > σ_{r+1}(WC)`.

*Proof.* The map `φ(A) = AC` satisfies `‖A‖_{H_λ} = ‖φ(A)‖_F`
(since `tr(AH_λAᵀ)=tr(ACCᵀAᵀ)`), is a linear bijection of `R^{m×n}`
(C invertible), and preserves rank. Hence
`min_{rank L ≤ r} ‖(W−L)C‖_F = min_{rank L' ≤ r} ‖WC − L'‖_F`
with `L' = LC`, which by Eckart–Young–Mirsky is solved by the
truncated SVD `(WC)_r`; pull back by `C⁻¹`. ∎

**Corollary 2.1 (dominance over SVDQuant's basis).** Let
`L_svd = W_r` be the unweighted rank-r truncated SVD. Then

    ‖W − L*‖_{H_λ}²  ≤  ‖W − L_svd‖_{H_λ}² ,

with equality iff the top-r right singular subspace of `W` is already
optimal for `WC` (in particular, whenever `H ∝ I`). Conversely
`‖W − L_svd‖_F ≤ ‖W − L*‖_F`: each basis is optimal only in its own
metric — but by Proposition 1 the metric that equals the deployed
layer error is `‖·‖_H`, not `‖·‖_F`.

Remark (anisotropy). The gap is governed by the anisotropy of `H`;
empirically diag(H) of diffusion-transformer hooks spans several
orders of magnitude (calibration covariances in this repo; e.g. the
FLUX temb channel energies span ~10 orders), so the equality case is
far from practice. The end-to-end counterpart of this corollary is the
λ→∞ ablation row (`svdbasis_qdiff128.json`).

---

## T3 — The λ family: SVDQuant as an endpoint, and λ as exact
## spectral-robustness

**Proposition 3.1 (family inclusion / endpoint).** Assume
`σ_r(W) > σ_{r+1}(W)`. Then `L*(λ) → W_r = L_svd` as `λ → ∞`
(convergence of the residual subspace in the gap metric).

*Proof sketch.* `C(λ) = (λm̄)^{1/2} · chol(I + H/(λm̄))
= (λm̄)^{1/2}(I + H/(2λm̄) + O(λ⁻²))`, so
`WC(λ)/(λm̄)^{1/2} → W`; by Wedin's theorem the singular value gap of
`W` makes the top-r singular subspaces of `WC(λ)` converge to those of
`W`, and `C(λ)⁻¹` converges to the corresponding rescaled identity on
that subspace, giving `L*(λ) → W_r`. ∎

Consequently **SVDQuant's decomposition is the λ→∞ degenerate point of
our one-parameter family**, and Algorithm 1's λ grid selects a member
of the family on calibration data. On the selection criterion itself
the selected member is, by construction, no worse than any fixed grid
member (a no-regret statement over the finite menu; generalization to
the official test set is validated empirically).

**Proposition 3.2 (λ = exact robust reweighting).** For every `Δ`,

    max_{‖E‖₂ ≤ ε, H+E ⪰ 0} tr(Δ (H+E) Δᵀ) = ‖Δ‖_H² + ε‖Δ‖_F²
                                            = ‖Δ‖²_{H+εI},

attained at `E = εI`. Hence with `ε = λ m̄`,

    L*(λ) = argmin_{rank ≤ r} max_{‖E‖₂ ≤ λm̄} ‖W − L‖²_{H+E} :

the damped H-SVD is the *exact* min–max solution under a spectral
uncertainty ball around the estimated `Ĥ` — the natural model for
finite-sample estimation error of the second moment. λ therefore
interpolates between the empirical-optimal basis (λ→0) and the
distribution-free basis (λ→∞ = SVDQuant), and is selected on data.

*Proof.* `tr(ΔEΔᵀ) ≤ ‖E‖₂ tr(ΔΔᵀ) ≤ ε‖Δ‖_F²` with equality at
`E = εI` (feasible: `H + εI ⪰ 0`); the bound holds simultaneously for
all Δ, so min and max interchange trivially. Apply Theorem 2 with
`H_λ = H + εI`. ∎

---

## T4 — SVDQuant's Prop 4.1 is a Cauchy–Schwarz relaxation of our
## objective; the residual quantizer is objective-aligned vs. unaligned

**Proposition 4 (relaxation chain).** For any `Δ`,

    ‖Δ‖_H = ‖XΔᵀ‖_F ≤ ‖X‖₂ ‖Δ‖_F ≤ ‖X‖_F ‖Δ‖_F ,

with equality in the second inequality only if every row of `Δ` lies
in the top singular direction of `X`, and in the third only if `X` is
rank one. SVDQuant's Prop 4.1 controls the weight-error term by the
right-most expression and their Eckart–Young step minimizes `‖R‖_F`
(then Prop 4.2 converts `‖R‖_F` to an expected naive-rounding error
under a Gaussian-type regularity). **They minimize the right-hand
side; we minimize the left-hand side directly** — at the basis stage
exactly (T2), and at the quantization stage as follows.

**Proposition 4′ (smooth-then-SVD = diagonal-metric approximation).**
SVDQuant's actual rank-r stage operates AFTER smoothing: with factors
`λ ∈ R^n_{>0}`, `x̂ = x·diag(λ)⁻¹`, `Ŵ = W·diag(λ)`, they take the
truncated SVD `L̂ = (Ŵ)_r` and deploy the lora on `x̂`. In raw
coordinates the deployed low-rank branch is `L_λ = L̂·diag(λ)⁻¹`, and

    L_λ = argmin_{rank ≤ r} ‖(W−L)·diag(λ)‖_F
        = argmin_{rank ≤ r} ‖W−L‖_{diag(λ²)} ,

i.e. smooth-then-SVD is exactly the weighted low-rank problem with the
**diagonal** metric `diag(λ²)` (proof: right-multiplication by the
invertible `diag(λ)` is a rank-preserving bijection; Eckart–Young).
The basis hierarchy is therefore: metric `I` (plain SVD of raw W) →
`diag(λ²)` (SVDQuant) → `H_λ` (ours, full second moment). By Theorem 2,

    ‖W − L*‖_{H_λ} ≤ ‖W − L_λ‖_{H_λ}   for every λ,

so the dominance claim holds against SVDQuant's *actual*
smooth-then-SVD stage, not merely against plain SVD; the remaining gap
they cannot close with any diagonal has two sources: (i) the
off-diagonal correlations of `H` (a diagonal metric cannot rotate the
subspace toward correlated directions), and (ii) their `λ` is an
amax-mix with a weight-side exponent (SmoothQuant heritage), so one
diagonal simultaneously serves basis weighting *and* residual-quantizer
conditioning — objectives our pipeline decouples (full-`H` basis;
separate post-split `s` for the quantizer, T5). Note the λ→∞ ablation
row instantiates metric `I`; the main tables compare against their
complete pipeline (smoothing included); the intermediate
`diag(diag(H_λ))`-metric row is a proposed ablation to separate the
value of the diagonal from the value of the correlations.

**Quantizer alignment.** Our residual quantizer is GPTQ on the exact
NVFP4 two-level kernel grid (per-tensor/-channel top scale ×
per-group-16 e4m3 micro-scale, act-order): per coordinate it performs
the OBS closed-form update that exactly minimizes the *same* metric
`tr(Δ H_g Δᵀ)` given all previously quantized coordinates (`H_g` =
the GPTQ Hessian; in the smoothed domain the analytic transform
`H/(s⊗s)` is used, never re-collected). SVDQuant quantizes `R` by
per-group absmax rounding, the minimizer of the elementwise
`max|Δ|` — a quantity unaligned with `‖Δ‖_H`. (We claim per-step OBS
optimality, not global optimality; the grid equals the deployed kernel
decode exactly, so there is no train/deploy grid mismatch on either
side.) Their Prop 4.2 remains valid for our pipeline as a coarse
magnitude bound; it is simply not the quantity our stages optimize.

---

## T5 — Closed-form smoothing: exact minimizer of the separable bound
## on the residual-aware objective

Model the 4-bit branch error for a smoothed layer,
`e = Q_a(X diag(s)⁻¹) (diag(s) W_resᵀ) − X W_resᵀ`, to first order in
the activation-quantization noise `E_a = Q_a(X/s) − X/s`:

    ‖e‖_F ≈ ‖E_a (W_res diag(s))ᵀ‖_F ≤ ‖E_a‖_F · ‖W_res diag(s)‖_F .

Under the proportional-noise model `E[E_a²]_{ik} ∝ X_{ik}²/s_k²`
(rounding noise scales with channel magnitude), with
`a_k = Σ_i X_{ik}² = diag(H)_k` and `b_k = Σ_j (W_res)_{jk}²`:

    E‖e‖_F² ≲ ( Σ_k a_k / s_k² ) · ( Σ_k b_k s_k² ).

**Proposition 5.** The bound's exact global minimizer is

    s_k* = (a_k / b_k)^{1/4} = ( rms_{x,k} / rms_{w,k} )^{1/2},

i.e. the α = 0.5 member of our family `s_k = (rms_x/rms_w)^α` (stored
full-strength, exponentiated at build time), with the weight statistics
taken on `W_res(λ*)` — the tensor actually being quantized.

*Proof.* By Cauchy–Schwarz,
`(Σ a_k/s_k²)(Σ b_k s_k²) ≥ (Σ √(a_k b_k))²`, a constant independent
of `s`, with equality iff `(a_k/s_k²) ∝ (b_k s_k²)` for all k, i.e.
`s_k⁴ ∝ a_k/b_k`; the global scale of `s` is quantization-neutral
(absorbed by the two-level scales) and fixed by geometric-mean
normalization. ∎

Contrast: SVDQuant's smoothing factors are per-layer grid searches
inherited from SmoothQuant with no optimality statement, and — under
the strict self-containment ruling — are calibration artifacts of
their pipeline. Our `s*` is closed-form from our own statistics
(`diag(H)` is free), residual-aware, and guarded by the ≥3/4
end-to-end gate (accept only if it beats the no-S incumbent on the
four calibration criteria), which yields the no-regret fallback: the
deployed configuration is never worse than the pure-λ* base on the
selection criterion.

---

## Scope and honest limitations

1. No end-to-end dominance theorem (either direction) is claimed once
   activation quantization and cross terms are included: `Q_a` errors
   depend on distributional properties beyond second moments, and the
   H-optimal basis may increase the *unweighted* `‖R‖_F` that enters
   magnitude-type bounds (their Prop 4.2). The theory establishes
   stage-wise exact optimality for the deployed first-order objective
   and strict relaxation of that objective in SVDQuant's analysis;
   pipeline-level comparison is settled empirically (six models,
   29/30 official metrics, zero-dependency; `PLAN.md`).
2. GPTQ claims are per-step (OBS) optimality, not global.
3. Prop 3.1 needs the singular gap `σ_r > σ_{r+1}` (generic).
4. Prop 5's proportional-noise model linearizes the two-level grid;
   α ≠ 0.5 winners in practice (0.25 on SANA/FLUX-dev) reflect the
   model's approximation error — which is precisely why the α grid and
   the gate exist.

## Experimental closure（2026-09-02 完成，`results/svdbasis_qdiff128.json`）

λ→∞（plain-SVD basis，即 SVDQuant 的分解）放進我們其餘完全相同的
pipeline（同 NVFP4 GPTQ on H、同 kernel、無 S、flux 同 adanorm），
qdiff-128 四判準對決各模型 λ\* 基底：

| 模型（λ\*） | λ\* vs λ→∞ | ΔPSNR(λ\*−∞) |
|---|---|---|
| PixArt（0.1） | **4:0** | +1.13 dB |
| SANA（0.3） | **4:0** | +0.35 dB |
| SDXL-base（0.001） | **4:0** | +0.29 dB |
| FLUX-schnell（0.01） | **3:1** | +0.11 dB |
| SDXL-Turbo（0.3，網格上緣） | 1:3 | −0.26 dB |
| FLUX.1-dev（0.3，網格上緣） | 0:4 | −0.58 dB |

**判讀**：(i) 4/6 模型 H 加權基底明確勝出（PixArt 差距達 1.1dB），
Corollary 2.1 的端到端體現；(ii) 兩個 λ\* 已頂到網格上緣（0.3）的
模型，最優點在網格之外、甚至趨向 SVD 端——**支持 T3 的家族觀點而非
「H-SVD 恆勝」的教條**：沒有固定基底全贏，正確主張是「阻尼族
⊇ {SVDQuant 端點}，由校準資料逐模型選擇」。論文寫法：λ 網格應延伸
大 λ 端（含 ∞ 端點），Algorithm 1 自動落點；per-model λ\* 分布
（0.001→∞）本身就是 anisotropy 重要性隨模型變化的量化證據。
- Anisotropy evidence: per-hook `H` diag/eigen spectra span orders of
  magnitude（cov 檔可直接出圖），justifying strictness in Cor 2.1
  where it wins.
- Grand results: `PLAN.md`（六模型 29/30）、`PLAN_SELFSMOOTH.md`
  （零依賴消融與成本）。
