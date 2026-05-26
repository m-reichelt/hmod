# Mathematical Background

The package is written for temporal discretizations on

$$
I = (0,T)
$$

with a uniform partition into $n_t$ intervals. The local mesh width is
$h = T / n_t$.

## Model Hybrid ODE

The notebook example solves

$$
\partial_t u + \mu u = f \quad \text{on } I,
\qquad
u(0) = 0.
$$

The hybrid variational formulation is: find $u$ in the trial space $X$ such
that

$$
b(u,v)
= \langle \partial_t u, (\mathcal{H}_T + I)v \rangle_I
+ \mu \langle u, (\mathcal{H}_T + I)v \rangle_I
= \langle f, (\mathcal{H}_T + I)v \rangle_I .
$$

Here $\mathcal{H}_T$ is the modified Hilbert transform. A typical norm for the
trial space in the notebook is

$$
\|u\|_X^2 =
\|u\|_{H^{1/2}(I)}^2 + \|u\|_{L^2(I)}^2 .
$$

In the ODE system, the mass contribution is weighted by $\mu>0$. On the discrete
level, this leads to four building blocks:

$$
\langle \partial_t u_h, v_h \rangle_I,
\qquad
\langle u_h, v_h \rangle_I,
\qquad
\langle \partial_t u_h, \mathcal{H}_T v_h \rangle_I,
\qquad
\langle u_h, \mathcal{H}_T v_h \rangle_I.
$$

`hmodFFT` provides these blocks for arbitrary derivative orders and for all
combinations of Legendre and Lagrange trial and test bases.

## Bases

Two polynomial bases are used throughout the package.

| Basis | Continuity | Typical use | Number of DOFs |
| --- | --- | --- | --- |
| Legendre | Discontinuous across intervals | Internal basis for most operations and right-hand side projections | `(p + 1) * nt` |
| Lagrange | Continuous across intervals | Trial and test space for continuous time discretizations | `p * nt + 1` |

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

$$
A_{\text{Lag},\text{Lag}}
= T_{\text{test}}^T A_{\text{Leg},\text{Leg}} T_{\text{trial}},
$$

where $T_{\text{trial}}$ and $T_{\text{test}}$ map Lagrange coefficients into Legendre
coefficients on the same uniform mesh.

## Rows, Columns, And Derivatives

All matrices and linear operators follow the finite element convention:

- columns correspond to trial coefficients,
- rows correspond to test functionals,
- `derivatives_trial=i` means `\partial_t^i` is applied to the trial function,
- `derivatives_test=j` means `\partial_t^j` is applied to the test function.

Thus an operator with `derivatives_trial=1` and `derivatives_test=0` represents
the temporal derivative in the trial argument.

## Initial Condition

Assembly routines do not impose homogeneous initial conditions. In a continuous
Lagrange space, the first degree of freedom corresponds to the value at
$t = 0$. For the condition $u(0)=0$, remove or restrict that first degree of
freedom after assembly.

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
