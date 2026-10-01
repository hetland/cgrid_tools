import numpy as np
import pytest

from cgrid_tools.geo import coriolis, great_circle_dx


def test_coriolis_at_equator_is_zero() -> None:
    assert np.isclose(coriolis(np.array([0.0]))[0], 0.0)


def test_coriolis_at_30_degrees() -> None:
    expected = 2.0 * 7.2921159e-5 * 0.5
    assert np.isclose(coriolis(np.array([30.0]))[0], expected)


def test_great_circle_dx_unit_degree_at_equator() -> None:
    pytest.importorskip("pyproj")
    lon = np.array([[0.0, 1.0]])
    lat = np.array([[0.0, 0.0]])

    dx = great_circle_dx(lon, lat)

    assert dx.shape == (1, 1)
    assert np.isclose(dx[0, 0], 111_319.49079327357, rtol=1.0e-6)
