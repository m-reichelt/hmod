# hmodFFT

`hmodFFT` is a Python/Rust package for matrix-free applications of temporal
operators that involve the modified or periodic Hilbert transform. The package
focuses on uniform partitions of an interval $I = (0, T)$ and on trial and test
spaces given by piecewise polynomial bases.

The main routines assemble or apply bilinear forms of the type

```math
\langle \partial_t^i u, \partial_t^j v \rangle_I
```

and

```math
\langle \partial_t^i u, \mathcal{H}_T \partial_t^j v \rangle_I,
```

where $\mathcal{H}_T$ denotes the modified Hilbert transform on
$I = (0, T)$. Passing `periodic=True` selects the periodic transform
$\mathcal H_{\mathrm{per}}$ and identifies the Lagrange degrees of freedom at
$0$ and $T$. Standard matrices are assembled as sparse SciPy matrices.
Hilbert-transform matrices are represented as SciPy `LinearOperator`s and use
FFT-based transforms for efficient application. The default is
`periodic=False`.

## Executable Notebooks

| Notebook | Topic | Run in Binder |
| --- | --- | --- |
| [notebooks/ode_hybrid.ipynb](notebooks/ode_hybrid.ipynb) | Linear hybrid ODE solve with BPX-preconditioned GMRES. | [![Open in Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/m-reichelt/hmod/main?labpath=notebooks%2Fode_hybrid.ipynb) |
| [notebooks/ode_hybrid_periodic.ipynb](notebooks/ode_hybrid_periodic.ipynb) | Periodic linear hybrid ODE solve with periodic FFT operators. | [![Open in Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/m-reichelt/hmod/main?labpath=notebooks%2Fode_hybrid_periodic.ipynb) |
| [notebooks/ode_nonliner_hybrid.ipynb](notebooks/ode_nonliner_hybrid.ipynb) | Nonlinear hybrid ODE solve using weighted residual operators. | [![Open in Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/m-reichelt/hmod/main?labpath=notebooks%2Fode_nonliner_hybrid.ipynb) |

Binder starts a temporary JupyterLab environment and installs the released
`hmodFFT` package from PyPI together with the notebook dependencies. The first
launch may take a few minutes while Binder builds the environment.

## Installation

Install the published package with

```bash
pip install hmodFFT
```

and import it in Python as `hmod`:

```python
import hmod as hm
```

For development from a checkout in the root folder of the repository, run

```bash
maturin develop -r
```

This will install the package inside the current Python environment.
The package is built with [maturin](https://www.maturin.rs/), because part of the basis evaluation code
is implemented in Rust. Note, that the Rust toolchain must be installed to build the package from source.

## Quick Example

The following builds the standard and Hilbert parts of the temporal bilinear
form

```math
\langle \partial_t u, (\mathcal{H}_T + I)v \rangle_I
+ \mu \langle u, (\mathcal{H}_T + I)v \rangle_I
```

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
a Lagrange space, the first degree of freedom is the value at $t = 0$.

For a periodic problem, pass `periodic=True` to the standard and Hilbert matrix
routines, basis transforms, prolongation matrices, and `BPXPreconditioner`.
A degree-$p$ periodic Lagrange space on $n_t$ intervals has $n_t p$ degrees of
freedom rather than $n_t p+1$: the right endpoint of the last interval wraps to
the first degree of freedom. Periodicity is therefore built into the basis, and
no initial degree of freedom is removed.

## Documentation

The GitHub documentation is organized as plain Markdown:

- [Documentation overview](docs/index.md)
- [Mathematical background](docs/mathematical-background.md)
- [Matrix and operator assembly guide](docs/matrix-assembly.md)
- [Hybrid ODE worked example](docs/hybrid-ode-example.md)
- [Periodic hybrid ODE notebook](notebooks/ode_hybrid_periodic.ipynb)
- [Development notes](docs/development.md)

The notebook [notebooks/ode_hybrid.ipynb](notebooks/ode_hybrid.ipynb) contains
the same hybrid ODE example, while
[notebooks/ode_hybrid_periodic.ipynb](notebooks/ode_hybrid_periodic.ipynb)
uses the periodic transform and periodic Lagrange topology. The notebook
[notebooks/ode_nonliner_hybrid.ipynb](notebooks/ode_nonliner_hybrid.ipynb)
contains a nonlinear variant using the weighted residual operators. GitHub
renders notebooks statically; use the Binder links above or run them locally to
execute the cells.


## How to Cite

A software paper for `hmodFFT` is currently under construction. For now, please
cite the dissertation:

```bibtex
@phdthesis{Reichelt2026EllipticSpaceTime,
  author  = {Reichelt, Michael},
  title   = {Elliptic Space-Time Methods for Parabolic PDEs with Applications},
  school  = {Graz University of Technology},
  year    = {2026},
  month   = jan,
  doi     = {10.3217/p02r1-7rs49},
  url     = {https://repository.tugraz.at/theses/93032}
}
```

## References

For the analytical background and the modified Hilbert transform, see
[Elliptic space-time methods for parabolic PDEs with applications](https://repository.tugraz.at/publications/p02r1-7rs49)
and the author's
[dissertation](https://online.tugraz.at/tug_online/wbAbs.showThesis?pThesisNr=93032&pOrgNr=&pAutorNr=#).

## License

This project is released under the MIT license. See [LICENSE](LICENSE).
