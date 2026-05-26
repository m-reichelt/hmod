# Hybrid ODE Worked Example

This page turns the notebook `notebooks/ode_hybrid.ipynb` into a compact
script-style reference. The problem is

```math
\partial_t u + \mu u = f,
\qquad
u(0)=0,
\qquad
I=(0,T).
```

The manufactured solution is

```math
u(t) = \sin(\pi t),
\qquad
f(t) = \pi \cos(\pi t) + \mu \sin(\pi t).
```

## Imports And Data

```python
import numpy as np
import scipy.sparse.linalg as spla

from hmod.standard_matrices import (
    get_lagrange_lagrange_matrix_for_derivatives,
    get_legendre_lagrange_matrix_for_derivatives,
    project_rhs_onto_legendre_basis,
)
from hmod.hilbert_matrices import (
    get_hilbert_matrix_for_derivatives_lagrange_lagrange,
    get_hilbert_matrix_for_derivatives_legendre_lagrange,
)
from hmod.dof_handling import DofRestrictorSymmetric
from hmod.polynomial_bases import (
    LegendreBasisEvaluator,
    get_lagrange_to_legendre_matrix,
)
from hmod.preconditioning import BPXPreconditioner, GMRESCounter
from hmod.norms import compute_l2_norm, compute_h12_seminorm

T = 5.0
mu = 3.0
p = 3

nt_coarsest = 2
refinements = 4
nt = nt_coarsest * 2**refinements

u = lambda t: np.sin(np.pi * t)
dt_u = lambda t: np.pi * np.cos(np.pi * t)
f = lambda t: dt_u(t) + mu * u(t)
```

## Project The Right-Hand Side

The load function is represented in the discontinuous Legendre basis.

```python
f_h = project_rhs_onto_legendre_basis(
    f,
    nt=nt,
    polynomial_degree_test=p,
    T=T,
)
```

## Assemble The Left-Hand Side

The standard terms are sparse matrices:

```python
A_t = get_lagrange_lagrange_matrix_for_derivatives(
    polynomial_degree_trial=p,
    polynomial_degree_test=p,
    derivatives_trial=1,
    derivatives_test=0,
    nt=nt,
    T=T,
)

M_t = get_lagrange_lagrange_matrix_for_derivatives(
    polynomial_degree_trial=p,
    polynomial_degree_test=p,
    derivatives_trial=0,
    derivatives_test=0,
    nt=nt,
    T=T,
)
```

The terms containing the modified Hilbert transform are matrix-free
`LinearOperator`s:

```python
AH_t = get_hilbert_matrix_for_derivatives_lagrange_lagrange(
    polynomial_degree_trial=p,
    polynomial_degree_test=p,
    derivatives_trial=1,
    derivatives_test=0,
    nt=nt,
    T=T,
)

MH_t = get_hilbert_matrix_for_derivatives_lagrange_lagrange(
    polynomial_degree_trial=p,
    polynomial_degree_test=p,
    derivatives_trial=0,
    derivatives_test=0,
    nt=nt,
    T=T,
)
```

Convert the sparse pieces to linear operators and form the complete system:

```python
B = (
    AH_t
    + spla.aslinearoperator(A_t)
    + mu * (MH_t + spla.aslinearoperator(M_t))
)
```

## Impose The Initial Condition

The first Lagrange degree of freedom is `u(0)`. For `u(0)=0`, restrict it:

```python
B0 = DofRestrictorSymmetric(
    B,
    restricted_dofs=np.array([0]),
    restricted_dof_values=np.array([0.0]),
)
```

## Assemble The Load Vector

The projected right-hand side `f_h` is in Legendre basis, while the test space is
Lagrange. Therefore the right-hand side uses Legendre-Lagrange operators:

```python
M_rhs = spla.aslinearoperator(
    get_legendre_lagrange_matrix_for_derivatives(
        polynomial_degree_trial=p,
        polynomial_degree_test=p,
        derivatives_trial=0,
        derivatives_test=0,
        nt=nt,
        T=T,
    )
) + get_hilbert_matrix_for_derivatives_legendre_lagrange(
    polynomial_degree_trial=p,
    polynomial_degree_test=p,
    derivatives_trial=0,
    derivatives_test=0,
    nt=nt,
    T=T,
)

rhs = M_rhs @ f_h
rhs0 = rhs[1:]
```

## Solve With GMRES

The system is elliptic in the hybrid norm but non-symmetric, so the example uses
GMRES with a BPX preconditioner for the `H^{1/2}` plus mass norm.

```python
BPX = BPXPreconditioner(
    mu=mu,
    n_refinements=refinements,
    sobolev_exponent=0.5,
    polynomial_degree=p,
    nt_coarse=nt_coarsest,
    T=T,
)

counter = GMRESCounter(print_residual=True)
sol, info = spla.gmres(B0, rhs0, M=BPX, callback=counter, rtol=1e-10)
```

Add the homogeneous initial degree of freedom back:

```python
u_h_lagrange = np.zeros(sol.shape[0] + 1)
u_h_lagrange[1:] = sol
```

## Evaluate And Compute Errors

The evaluator expects Legendre coefficients, so transform from Lagrange to
Legendre first:

```python
T_lag_to_leg = get_lagrange_to_legendre_matrix(polynomial_degree=p, nt=nt)
u_h_legendre = T_lag_to_leg @ u_h_lagrange

evaluator = LegendreBasisEvaluator(
    dofs=u_h_legendre,
    polynomial_degree=p,
    nt=nt,
    T=T,
)
```

Compute errors:

```python
error = lambda t: evaluator.evaluate(t) - u(t)

L2_error = compute_l2_norm(error, nt=nt, T=T)
H12_error = compute_h12_seminorm(error, T=T, quad_tol=1e-8)
X_error = np.sqrt(L2_error**2 + H12_error**2)
```

At this point, `evaluator.evaluate(t_points)` can be used for plotting or
post-processing.
