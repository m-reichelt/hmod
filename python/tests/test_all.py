import pytest
import hmod


def test_sum_as_string():
    assert hmod.sum_as_string(1, 1) == "2"
