# Matrix And Operator Assembly

This page documents the main public routines for assembling standard matrices
and Hilbert-transform operators.

## Naming Convention

Function names encode the trial and test basis:

```text
get_<trial>_<test>_matrix_for_derivatives
get_hilbert_matrix_for_derivatives_<trial>_<test>
```

The available basis combinations are:

| Trial basis | Test basis | Standard matrix routine | Hilbert operator routine |
| --- | --- | --- | --- |
| Legendre | Legendre | `get_legendre_legendre_matrix_for_derivatives` | `get_hilbert_matrix_for_derivatives_legendre_legendre` |
| Legendre | Lagrange | `get_legendre_lagrange_matrix_for_derivatives` | `get_hilbert_matrix_for_derivatives_legendre_lagrange` |
| Lagrange | Legendre | `get_lagrange_legendre_matrix_for_derivatives` | `get_hilbert_matrix_for_derivatives_lagrange_legendre` |
| Lagrange | Lagrange | `get_lagrange_lagrange_matrix_for_derivatives` | `get_hilbert_matrix_for_derivatives_lagrange_lagrange` |

The first basis name is the trial basis. The second basis name is the test
basis. Rows correspond to tests and columns correspond to trials.

## Standard Matrices

Standard matrices live in `hmod.standard_matrices`. They return sparse CSR
matrices and represent

```math
A_{r,m}
= \langle \partial_t^i \phi_m, \partial_t^j \psi_r \rangle_I.
```

Example: assemble a derivative matrix and a mass matrix in continuous Lagrange
basis.

```python
from hmod.standard_matrices import get_lagrange_lagrange_matrix_for_derivatives

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

Useful choices of derivative orders:

| `derivatives_trial` | `derivatives_test` | Bilinear form |
| --- | --- | --- |
| `0` | `0` | `\langle u, v \rangle_I` |
| `1` | `0` | `\langle \partial_t u, v \rangle_I` |
| `0` | `1` | `\langle u, \partial_t v \rangle_I` |
| `1` | `1` | `\langle \partial_t u, \partial_t v \rangle_I` |

Different polynomial degrees for trial and test spaces are supported through
`polynomial_degree_trial` and `polynomial_degree_test`.

## Hilbert Operators

Hilbert-transform routines live in `hmod.hilbert_matrices`. They return SciPy
`LinearOperator` objects and represent

```math
A^{\mathcal{H}}_{r,m}
= \langle \partial_t^i \phi_m,
        \mathcal{H}_T \partial_t^j \psi_r \rangle_I.
```

They are matrix-free because the corresponding operators are dense in the
physical basis. Internally, the package uses sine/cosine transforms and a
zeta-series kernel to apply the modified Hilbert transform efficiently.

Example: assemble the Hilbert derivative and Hilbert mass parts in continuous
Lagrange basis.

```python
from hmod.hilbert_matrices import get_hilbert_matrix_for_derivatives_lagrange_lagrange

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

The Hilbert routines use the same argument names and order as the corresponding
standard matrix routines:
`polynomial_degree_trial`, `polynomial_degree_test`, `derivatives_trial`,
`derivatives_test`, `nt`, and `T`.

## Combining Sparse Matrices And Linear Operators

Standard sparse matrices can be converted to `LinearOperator`s and added to the
Hilbert operators:

```python
import scipy.sparse.linalg as spla

B = (
    AH_t
    + spla.aslinearoperator(A_t)
    + mu * (MH_t + spla.aslinearoperator(M_t))
)
```

This is the common pattern for the hybrid ODE formulation.

## Right-Hand Sides

For a function $f$, project it into the discontinuous Legendre basis:

```python
from hmod.standard_matrices import project_rhs_onto_legendre_basis

f_h = project_rhs_onto_legendre_basis(f, nt=nt, polynomial_degree_test=p, T=T)
```

To assemble the functional

```math
\langle f_h, (\mathcal{H}_T + I)v_h \rangle_I
```

with a Legendre representation of $f_h$ and a Lagrange test basis, combine the
Legendre-Lagrange standard and Hilbert operators:

```python
import scipy.sparse.linalg as spla

from hmod.standard_matrices import get_legendre_lagrange_matrix_for_derivatives
from hmod.hilbert_matrices import get_hilbert_matrix_for_derivatives_legendre_lagrange

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
```

For a homogeneous Lagrange initial condition, remove the first test functional:

```python
rhs0 = rhs[1:]
```

## Weighted Residuals

Nonlinear and coefficient-dependent terms often start from a pointwise
residual function $g(t)$ that is only available through evaluation. The helper
`hmod.non_linear_operators.WeightedResidual` projects such a residual into a
Legendre space by quadrature and then returns the tested residual vector.

There are two residual types:

| Residual type | Returned entries |
| --- | --- |
| `ResidualType.Standard` | `\langle g, v_h \rangle_I` |
| `ResidualType.Hilbert` | `\langle g, \mathcal{H}_T v_h \rangle_I` |

The routine is useful when assembling a nonlinear matrix-free action. The
residual function should accept NumPy arrays of points and return values with
the same shape. The below code corresponds to the bilinear form $\langle c(t) \partial_t u, (\mathcal{H}_T + I)v \rangle_I$:

```python
import hmod.non_linear_operators as nops

weighted_standard = nops.WeightedResidual(
    residual_type=nops.ResidualType.Standard,
    polynomial_degree_test=p,
    polynomial_degree_projection=p + 2,
    nt=nt,
    T=T,
)

weighted_hilbert = nops.WeightedResidual(
    residual_type=nops.ResidualType.Hilbert,
    polynomial_degree_test=p,
    polynomial_degree_projection=p + 2,
    nt=nt,
    T=T,
)

residual_fun = lambda t: coefficient(t) * evaluator.evaluate_derivative(t)

standard_residual = weighted_standard.apply(residual_fun)
hilbert_residual = weighted_hilbert.apply(residual_fun)
hybrid_residual = standard_residual + hilbert_residual
```

Here `polynomial_degree_projection` controls the intermediate discontinuous
Legendre projection used for the pointwise residual. In polynomial coefficient
examples, choose it large enough to represent the product exactly; for general
nonlinear functions, increase the projection degree according to the desired
accuracy.

## Basis Transforms And Evaluation

Use `hmod.polynomial_bases.get_lagrange_to_legendre_matrix` to transform a
continuous Lagrange coefficient vector into the internal Legendre ordering:

```python
from hmod.polynomial_bases import (
    LegendreBasisEvaluator,
    get_lagrange_to_legendre_matrix,
)

T_lag_to_leg = get_lagrange_to_legendre_matrix(polynomial_degree=p, nt=nt)
u_legendre = T_lag_to_leg @ u_lagrange

evaluator = LegendreBasisEvaluator(
    dofs=u_legendre,
    polynomial_degree=p,
    nt=nt,
    T=T,
)

values = evaluator.evaluate(t_points)
```

The evaluator expects Legendre coefficients in degree-major ordering.

## Preconditioning

The hybrid ODE system is elliptic but not symmetric. A typical solve uses
SciPy GMRES with the BPX preconditioner:

```python
import scipy.sparse.linalg as spla

from hmod.preconditioning import BPXPreconditioner, GMRESCounter

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

`BPXPreconditioner` acts on the reduced vector after the homogeneous initial
condition has been removed.

## Converting Operators To Dense Matrices

For testing and debugging it can be useful to materialize a sparse matrix or a
matrix-free `LinearOperator` as a dense NumPy array. This should generally not
be done in production code. The Hilbert-transform operators are deliberately
implemented matrix-free because the dense matrices can be large and expensive to
store/apply.

Sparse matrices can be converted directly:

```python
A_dense = A_t.toarray()
```

For a `LinearOperator`, use the helper in `hmod.matrix_tools`:

```python
from hmod.matrix_tools import linear_operator_to_matrix

AH_dense = linear_operator_to_matrix(AH_t)
```

The helper applies the operator to the identity matrix, so the result has the
same shape as the operator. This is convenient for comparisons against reference
matrices in tests, but it costs dense matrix memory and work.
