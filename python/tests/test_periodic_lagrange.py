import numpy as np
import pytest
import scipy.sparse as sp

import hmod.polynomial_bases as pb
import hmod.standard_matrices as sm
import hmod.non_linear_operators as nops


def _endpoint_identification_matrix(nt: int, polynomial_degree: int):
    """Embed periodic nodal DOFs into the legacy vector including both endpoints."""
    n_periodic_dofs = nt * polynomial_degree
    rows = np.concatenate(
        [np.arange(n_periodic_dofs), np.array([n_periodic_dofs])]
    )
    cols = np.concatenate(
        [np.arange(n_periodic_dofs), np.array([0])]
    )
    values = np.ones(n_periodic_dofs + 1)
    return sp.csr_matrix(
        (values, (rows, cols)),
        shape=(n_periodic_dofs + 1, n_periodic_dofs),
    )


@pytest.mark.parametrize("polynomial_degree", [1, 2, 3])
@pytest.mark.parametrize("nt", [1, 4])
def test_periodic_lagrange_to_legendre_is_endpoint_identification(
    polynomial_degree, nt
):
    transform_nonperiodic = pb.get_lagrange_to_legendre_matrix(
        polynomial_degree, nt
    )
    transform_periodic = pb.get_lagrange_to_legendre_matrix(
        polynomial_degree, nt, periodic=True
    )
    identify_endpoint = _endpoint_identification_matrix(
        nt, polynomial_degree
    )

    assert transform_nonperiodic.shape == (
        nt * (polynomial_degree + 1),
        nt * polynomial_degree + 1,
    )
    assert transform_periodic.shape == (
        nt * (polynomial_degree + 1),
        nt * polynomial_degree,
    )
    assert np.allclose(
        transform_periodic.toarray(),
        (transform_nonperiodic @ identify_endpoint).toarray(),
        rtol=0.0,
        atol=1e-14,
    )


def test_periodic_lagrange_points_omit_duplicate_endpoint():
    polynomial_degree = 2
    nt = 3
    T = 3.0

    points_nonperiodic = pb.get_lagrange_points(
        polynomial_degree, nt, T
    )
    points_periodic = pb.get_lagrange_points(
        polynomial_degree, nt, T, periodic=True
    )

    assert np.array_equal(
        points_nonperiodic,
        np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]),
    )
    assert np.array_equal(points_periodic, points_nonperiodic[:-1])
    assert np.array_equal(
        pb.get_langrange_points(
            polynomial_degree, nt, T, periodic=True
        ),
        points_periodic,
    )


def test_periodic_p1_prolongation_wraps_last_coarse_element():
    prolongation = pb.get_lagrange_prolongation_matrix(
        nt_coarse=2,
        nt_fine=4,
        polynomial_degree=1,
        periodic=True,
    )
    expected = np.array(
        [
            [1.0, 0.0],
            [0.5, 0.5],
            [0.0, 1.0],
            [0.5, 0.5],
        ]
    )

    assert prolongation.shape == (4, 2)
    assert np.allclose(
        prolongation.toarray(), expected, rtol=0.0, atol=1e-14
    )


@pytest.mark.parametrize("polynomial_degree", [1, 2, 3])
def test_periodic_prolongation_matches_endpoint_identification(
    polynomial_degree,
):
    nt_coarse = 3
    nt_fine = 6
    prolongation_nonperiodic = pb.get_lagrange_prolongation_matrix(
        nt_coarse, nt_fine, polynomial_degree
    )
    prolongation_periodic = pb.get_lagrange_prolongation_matrix(
        nt_coarse,
        nt_fine,
        polynomial_degree,
        periodic=True,
    )
    identify_coarse_endpoint = _endpoint_identification_matrix(
        nt_coarse, polynomial_degree
    )
    identify_fine_endpoint = _endpoint_identification_matrix(
        nt_fine, polynomial_degree
    )

    expected = prolongation_nonperiodic @ identify_coarse_endpoint
    actual = identify_fine_endpoint @ prolongation_periodic
    assert prolongation_periodic.shape == (
        nt_fine * polynomial_degree,
        nt_coarse * polynomial_degree,
    )
    assert np.allclose(
        actual.toarray(), expected.toarray(), rtol=0.0, atol=1e-13
    )


def test_periodic_p1_standard_mass_and_derivative_are_cyclic():
    nt = 5
    T = 2.5
    h = T / nt
    mass = sm.get_lagrange_lagrange_matrix_for_derivatives(
        polynomial_degree_trial=1,
        polynomial_degree_test=1,
        derivatives_trial=0,
        derivatives_test=0,
        nt=nt,
        T=T,
        periodic=True,
    ).toarray()
    derivative = sm.get_lagrange_lagrange_matrix_for_derivatives(
        polynomial_degree_trial=1,
        polynomial_degree_test=1,
        derivatives_trial=1,
        derivatives_test=0,
        nt=nt,
        T=T,
        periodic=True,
    ).toarray()

    expected_mass = np.zeros((nt, nt))
    expected_derivative = np.zeros((nt, nt))
    for row in range(nt):
        expected_mass[row, row] = 2.0 * h / 3.0
        expected_mass[row, (row - 1) % nt] = h / 6.0
        expected_mass[row, (row + 1) % nt] = h / 6.0
        expected_derivative[row, (row - 1) % nt] = -0.5
        expected_derivative[row, (row + 1) % nt] = 0.5

    assert mass.shape == (nt, nt)
    assert derivative.shape == (nt, nt)
    assert np.allclose(mass, expected_mass, rtol=0.0, atol=1e-14)
    assert np.allclose(
        derivative, expected_derivative, rtol=0.0, atol=1e-14
    )
    assert np.allclose(derivative + derivative.T, 0.0, atol=1e-14)
    assert np.allclose(derivative @ np.ones(nt), 0.0, atol=1e-14)


@pytest.mark.parametrize(
    "builder, expected_shape",
    [
        (sm.get_legendre_legendre_matrix_for_derivatives, (9, 6)),
        (sm.get_legendre_lagrange_matrix_for_derivatives, (6, 6)),
        (sm.get_lagrange_legendre_matrix_for_derivatives, (9, 3)),
        (sm.get_lagrange_lagrange_matrix_for_derivatives, (6, 3)),
    ],
)
def test_periodic_flag_propagates_through_standard_basis_pairs(
    builder, expected_shape
):
    matrix = builder(
        polynomial_degree_trial=1,
        polynomial_degree_test=2,
        derivatives_trial=0,
        derivatives_test=0,
        nt=3,
        T=1.0,
        periodic=True,
    )

    assert matrix.shape == expected_shape


@pytest.mark.parametrize(
    "builder,args",
    [
        (pb.get_lagrange_to_legendre_matrix, (0, 3)),
        (pb.get_lagrange_points, (0, 3, 1.0)),
        (pb.get_lagrange_prolongation_matrix, (3, 6, 0)),
    ],
)
def test_periodic_degree_zero_has_clear_error(builder, args):
    with pytest.raises(
        ValueError,
        match=r"(?i)periodic Lagrange",
    ):
        builder(*args, periodic=True)


def test_periodic_false_preserves_legacy_defaults():
    polynomial_degree = 2
    nt = 3
    T = 1.5

    transform_default = pb.get_lagrange_to_legendre_matrix(
        polynomial_degree, nt
    )
    transform_false = pb.get_lagrange_to_legendre_matrix(
        polynomial_degree, nt, periodic=False
    )
    assert np.array_equal(
        transform_default.toarray(), transform_false.toarray()
    )

    points_default = pb.get_lagrange_points(polynomial_degree, nt, T)
    points_false = pb.get_lagrange_points(
        polynomial_degree, nt, T, periodic=False
    )
    assert np.array_equal(points_default, points_false)

    prolongation_default = pb.get_lagrange_prolongation_matrix(
        nt, 2 * nt, polynomial_degree
    )
    prolongation_false = pb.get_lagrange_prolongation_matrix(
        nt, 2 * nt, polynomial_degree, periodic=False
    )
    assert np.array_equal(
        prolongation_default.toarray(), prolongation_false.toarray()
    )

    matrix_default = sm.get_lagrange_lagrange_matrix_for_derivatives(
        polynomial_degree,
        polynomial_degree,
        1,
        0,
        nt,
        T,
    )
    matrix_false = sm.get_lagrange_lagrange_matrix_for_derivatives(
        polynomial_degree,
        polynomial_degree,
        1,
        0,
        nt,
        T,
        periodic=False,
    )
    assert np.array_equal(matrix_default.toarray(), matrix_false.toarray())


@pytest.mark.parametrize(
    "residual_type",
    [nops.ResidualType.Standard, nops.ResidualType.Hilbert],
)
def test_periodic_weighted_residual_uses_periodic_test_space(residual_type):
    nt = 4
    polynomial_degree = 1
    residual = nops.WeightedResidual(
        residual_type=residual_type,
        polynomial_degree_test=polynomial_degree,
        polynomial_degree_projection=polynomial_degree,
        nt=nt,
        T=1.0,
        periodic=True,
    )

    result = residual.apply(lambda t: np.ones_like(t))

    assert result.shape == (nt * polynomial_degree,)
    if residual_type == nops.ResidualType.Hilbert:
        assert np.linalg.norm(result) < 1e-13
