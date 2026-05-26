# hmodFFT

[![Open the hybrid ODE notebook in Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/m-reichelt/hmod/main?labpath=notebooks%2Fode_hybrid.ipynb)

`hmodFFT` is a Python/Rust package for matrix-free applications of temporal
operators that involve the modified Hilbert transform. The package focuses on
uniform partitions of an interval `(0, T)` and on trial and test spaces given by
piecewise polynomial bases.

The main routines assemble or apply bilinear forms of the type

$$
\langle \partial_t^i u, \partial_t^j v \rangle_I
$$

and

$$
\langle \partial_t^i u, \mathcal{H}_T \partial_t^j v \rangle_I,
$$

where `\mathcal{H}_T` denotes the modified Hilbert transform on
`I = (0, T)`. Standard matrices are assembled as sparse SciPy matrices.
Hilbert-transform matrices are represented as SciPy `LinearOperator`s and use
FFT-based transforms for efficient application.

## Installation

Install the published package with

```bash
pip install hmodFFT
```

and import it in Python as `hmod`:

```python
import hmod as hm
```

For development from a checkout:

```bash
pip install -e ".[tests]"
```

The package is built with `maturin`, because part of the basis evaluation code
is implemented in Rust.

## Executable Binder Example

You can run the hybrid ODE notebook directly in the browser with Binder:

[![Open the hybrid ODE notebook in Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/m-reichelt/hmod/main?labpath=notebooks%2Fode_hybrid.ipynb)

Binder starts a temporary JupyterLab environment and installs the released
`hmodFFT` package from PyPI together with the notebook dependencies. The first
launch may take a few minutes while Binder builds the environment.


## Quick Example

The following builds the standard and Hilbert parts of the temporal bilinear
form

$$
\langle \partial_t u, (\mathcal{H}_T + I)v \rangle_I
+ \mu \langle u, (\mathcal{H}_T + I)v \rangle_I
$$

for continuous Lagrange trial and test functions.

```python
import scipy.sparse.linalg as spla

from hmod.standard_matrices import get_lagrange_lagrange_matrix_for_derivatives
from hmod.hilbert_matrices import get_hilbert_matrix_for_derivatives_lagrange_lagrange

T = 5.0
nt = 32
p = 3
mu = 3.0

A = get_lagrange_lagrange_matrix_for_derivatives(
    polynomial_degree_trial=p,
    polynomial_degree_test=p,
    derivatives_trial=1,
    derivatives_test=0,
    nt=nt,
    T=T,
)
M = get_lagrange_lagrange_matrix_for_derivatives(
    polynomial_degree_trial=p,
    polynomial_degree_test=p,
    derivatives_trial=0,
    derivatives_test=0,
    nt=nt,
    T=T,
)

AH = get_hilbert_matrix_for_derivatives_lagrange_lagrange(
    polynomial_degree_trial=p,
    polynomial_degree_test=p,
    derivatives_trial=1,
    derivatives_test=0,
    nt=nt,
    T=T,
)
MH = get_hilbert_matrix_for_derivatives_lagrange_lagrange(
    polynomial_degree_trial=p,
    polynomial_degree_test=p,
    derivatives_trial=0,
    derivatives_test=0,
    nt=nt,
    T=T,
)

B = AH + spla.aslinearoperator(A) + mu * (MH + spla.aslinearoperator(M))
```

The matrices are assembled before imposing homogeneous initial conditions. For
a Lagrange space, the first degree of freedom is the value at `t = 0`.

## Documentation

The GitHub documentation is organized as plain Markdown:

- [Documentation overview](docs/index.md)
- [Mathematical background](docs/mathematical-background.md)
- [Matrix and operator assembly guide](docs/matrix-assembly.md)
- [Hybrid ODE worked example](docs/hybrid-ode-example.md)
- [Development notes](docs/development.md)

The notebook [notebooks/ode_hybrid.ipynb](notebooks/ode_hybrid.ipynb) contains
the same hybrid ODE example. GitHub renders the notebook statically; use the
Binder badge above or run it locally to execute the cells.


## References

For the analytical background and the modified Hilbert transform, see
[Elliptic space-time methods for parabolic PDEs with applications](https://repository.tugraz.at/publications/p02r1-7rs49)
and the author's
[dissertation](https://online.tugraz.at/tug_online/wbAbs.showThesis?pThesisNr=93032&pOrgNr=&pAutorNr=#).

## License

This project is released under the MIT license. See [LICENSE](LICENSE).
