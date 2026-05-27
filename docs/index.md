# hmodFFT Documentation

`hmodFFT` provides sparse matrices and matrix-free linear operators for temporal
piecewise-polynomial discretizations on a uniform interval partition.

The central design idea is simple:

- Legendre bases are discontinuous on the time mesh and are the internal basis
  used by most operations.
- Lagrange bases are continuous on the time mesh and are represented through
  Legendre coefficients whenever an operation is easier in Legendre form.
- Standard bilinear forms are assembled as SciPy sparse matrices.
- Bilinear forms containing the modified Hilbert transform are exposed as
  SciPy `LinearOperator`s and applied with FFT-based transforms.
- Weighted residual operators project vectorized pointwise residual functions
  and return standard or Hilbert-tested residual vectors.

## Recommended Reading Order

1. [Mathematical background](mathematical-background.md) introduces the spaces,
   basis conventions, and the hybrid ODE formulation.
2. [Matrix and operator assembly guide](matrix-assembly.md) documents the main
   routines, weighted residuals, and naming conventions.
3. [Hybrid ODE worked example](hybrid-ode-example.md) shows how the pieces are
   combined into a complete solve.
4. [Development notes](development.md) explain source installation and tests.

The notebooks [../notebooks/ode_hybrid.ipynb](../notebooks/ode_hybrid.ipynb)
and [../notebooks/ode_nonliner_hybrid.ipynb](../notebooks/ode_nonliner_hybrid.ipynb)
contain the linear and nonlinear hybrid ODE examples in notebook form. GitHub
renders notebooks statically; to execute the cells, run them locally or open
them through the Binder links in the project README.

## Main Modules

| Module | Purpose |
| --- | --- |
| `hmod.standard_matrices` | Sparse matrices for standard derivative and mass bilinear forms. |
| `hmod.hilbert_matrices` | FFT-backed `LinearOperator`s for bilinear forms containing `\mathcal{H}_T`. |
| `hmod.non_linear_operators` | Weighted residual helpers for nonlinear or coefficient-dependent terms. |
| `hmod.polynomial_bases` | Basis transforms, evaluators, prolongation matrices, and derivative matrices. |
| `hmod.dof_handling` | Helpers for imposing fixed degrees of freedom in matrix-free systems. |
| `hmod.preconditioning` | BPX preconditioner and GMRES iteration counter. |
| `hmod.norms` | Numerical `L2` norm and `H^{1/2}` seminorm helpers. |

## What The Package Builds

For trial basis functions $\phi_m$ and test basis functions $\psi_r$, the main
operators represent entries of the form

```math
A_{r,m} =
\langle \partial_t^i \phi_m, \partial_t^j \psi_r \rangle_I
```

or

```math
A^{\mathcal{H}}_{r,m} =
\langle \partial_t^i \phi_m,
        \mathcal{H}_T \partial_t^j \psi_r \rangle_I .
```

The first basis name in an assembly function is always the trial basis and the
second basis name is always the test basis, for example
`get_legendre_lagrange_matrix_for_derivatives` has Legendre trial functions and
Lagrange test functions.
