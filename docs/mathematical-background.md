# Mathematical Background

The package is written for temporal discretizations on

```math
I = (0,T)
```

with a uniform partition into $n_t$ intervals. The local mesh width is
$h = T / n_t$.

## Model Hybrid Initial-Value ODE

The notebook example solves

```math
\partial_t u + \mu u = f \quad \text{on } I,
\qquad
u(0) = 0.
```

The hybrid variational formulation is: find $u$ in the trial space $X$ such
that

```math
b(u,v)
= \langle \partial_t u, (\mathcal{H}_T + I)v \rangle_I
+ \mu \langle u, (\mathcal{H}_T + I)v \rangle_I
= \langle f, (\mathcal{H}_T + I)v \rangle_I .
```

Here $\mathcal{H}_T$ is the modified Hilbert transform. A typical norm for the
trial space in the notebook is

```math
\|u\|_X^2 =
\|u\|_{H^{1/2}_{0,}(I)}^2 + \mu \|u\|_{L^2(I)}^2 .
```

In the ODE system, the mass contribution is weighted by $\mu>0$. On the discrete
level, this leads to four building blocks:

```math
\langle \partial_t u_h, v_h \rangle_I,
\qquad
\langle u_h, v_h \rangle_I,
\qquad
\langle \partial_t u_h, \mathcal{H}_T v_h \rangle_I,
\qquad
\langle u_h, \mathcal{H}_T v_h \rangle_I.
```

`hmodFFT` provides these blocks for arbitrary derivative orders and for all
combinations of Legendre and Lagrange trial and test bases.

## Periodic Transform And Hybrid ODE

For periodic functions, write

```math
u(t)=\sum_{k\in\mathbb Z}u^k e^{i\omega_k t},
\qquad
u^k=\frac1T\int_0^T u(t)e^{-i\omega_k t}\,dt,
\qquad
\omega_k=\frac{2\pi k}{T}.
```

The sign convention used by the package is

```math
\mathcal H_{\mathrm{per}}e^{i\omega_k t}
=
\begin{cases}
i\,\operatorname{sgn}(k)e^{i\omega_k t},&k\ne0,\\
0,&k=0.
\end{cases}
```

Thus $\mathcal H_{\mathrm{per}}\sin(2\pi t/T)=\cos(2\pi t/T)$; this is the
negative of the frequently used $-i\,\operatorname{sgn}(k)$ convention. The
periodic $H^{1/2}$ seminorm is normalized as

```math
|u|_{H^{1/2}_{\mathrm{per}}(I)}^2
=T\sum_{k\ne0}|\omega_k|\,|u^k|^2
=\langle \partial_tu,\mathcal H_{\mathrm{per}}u\rangle_I.
```

The [periodic notebook](../notebooks/ode_hybrid_periodic.ipynb) solves the same
ODE with $u(0)=u(T)$ and replaces $\mathcal H_T$ by
$\mathcal H_{\mathrm{per}}$ in the hybrid bilinear form.
For periodic $u$, the standard derivative term satisfies
$\langle\partial_tu,u\rangle_I=0$, while the Hilbert mass term vanishes by
skew-adjointness. Hence, for $\mu>0$,

```math
b_{\mathrm{per}}(u,u)
=|u|_{H^{1/2}_{\mathrm{per}}(I)}^2
+\mu\|u\|_{L^2(I)}^2.
```

If $\mu=0$, constants remain in the nullspace and an additional mean constraint
is needed.

The implementation uses the unnormalized forward DFT

```math
(\mathsf F_Nx)_a=\sum_{l=0}^{N-1}x_l e^{-2\pi i al/N}
```

and an inverse FFT normalized by $1/N$. Half-cell phases needed for individual
Fourier coefficients cancel in the Galerkin factorization. Only the continuous
frequency $k=0$ is annihilated: FFT bin $a=0$ also contains
$k=\pm N,\pm2N,\ldots$, so the complete zero bin is not set to zero.

## Bases

Two polynomial bases are used throughout the package.

| Basis | Continuity | Typical use | Number of DOFs |
| --- | --- | --- | --- |
| Legendre | Discontinuous across intervals | Internal basis for most operations and right-hand side projections | `(p + 1) * nt` |
| Lagrange, nonperiodic | Continuous across interval interfaces | Initial-value trial and test spaces | `p * nt + 1` |
| Lagrange, periodic | Continuous with the endpoints identified | Periodic trial and test spaces | `p * nt` |

The Legendre coefficient ordering is degree-major:

```text
[degree 0 on all intervals,
 degree 1 on all intervals,
 ...,
 degree p on all intervals]
```

This ordering is important when applying matrices or evaluating Legendre
coefficient vectors manually.

## Lagrange Through Legendre

Most operations are easiest in Legendre basis. If a Lagrange basis is requested,
the package applies the transformation matrix returned by
`hmod.polynomial_bases.get_lagrange_to_legendre_matrix`.

For example, a Lagrange-Lagrange matrix is assembled conceptually as

```math
A_{\text{Lag},\text{Lag}}
= T_{\text{test}}^T A_{\text{Leg},\text{Leg}} T_{\text{trial}},
```

where $T_{\text{trial}}$ and $T_{\text{test}}$ map Lagrange coefficients into Legendre
coefficients on the same uniform mesh.

With `periodic=True`, the last local endpoint of the last interval maps to
global Lagrange degree of freedom zero. Consequently,
`get_lagrange_to_legendre_matrix(p, nt, periodic=True)` has shape
`(nt * (p + 1), nt * p)`. The default `periodic=False` transformation has
shape `(nt * (p + 1), nt * p + 1)`.
Likewise, periodic Lagrange points exclude the duplicate point $T$, and a
periodic prolongation from `nt_coarse` to `nt_fine` has shape
`(nt_fine * p, nt_coarse * p)`.

## Rows, Columns, And Derivatives

All matrices and linear operators follow the finite element convention:

- columns correspond to trial coefficients,
- rows correspond to test functionals,
- `derivatives_trial=i` means `\partial_t^i` is applied to the trial function,
- `derivatives_test=j` means `\partial_t^j` is applied to the test function.

Thus an operator with `derivatives_trial=1` and `derivatives_test=0` represents
the temporal derivative in the trial argument.

## Boundary Topology

Assembly routines do not impose homogeneous initial conditions. In a continuous
nonperiodic Lagrange space, the first degree of freedom corresponds to the
value at $t = 0$. For the condition $u(0)=0$, remove or restrict that first
degree of freedom after assembly.

For matrix-free systems, use `hmod.dof_handling.DofRestrictorSymmetric`:

```python
import numpy as np
from hmod.dof_handling import DofRestrictorSymmetric

B0 = DofRestrictorSymmetric(
    B,
    restricted_dofs=np.array([0]),
    restricted_dof_values=np.array([0.0]),
)
```

For an explicitly assembled matrix, the same homogeneous condition is often
implemented by slicing:

```python
A0 = A[1:, 1:]
rhs0 = rhs[1:]
```

For periodic assembly, pass `periodic=True`. The values at $0$ and $T$ are then
represented by one shared degree of freedom. Do not apply an initial-value
restriction or slice the assembled matrix and right-hand side.
