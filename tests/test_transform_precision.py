"""Model preprocessing must not impose an fp32 error floor on fp64 kernels."""
import numpy as np
import pytest
import sympy as sp
from URDFParser.Joint import Joint, _simplify_transform


@pytest.mark.parametrize("angle", [0.27931, -0.27931, 0.123456789, 0.5235987756])
def test_simplification_preserves_rotation_coefficients_and_orthogonality(angle):
    c, s = np.cos(angle), np.sin(angle)
    matrix = sp.Matrix([[1, 0, 0], [0, c, -s], [0, s, c]])
    actual = np.asarray(_simplify_transform(matrix), float)
    np.testing.assert_allclose(actual, np.asarray(matrix, float), atol=2e-15, rtol=0)
    np.testing.assert_allclose(actual @ actual.T, np.eye(3), atol=2e-15, rtol=0)


def test_revolute_transform_keeps_noncardinal_origin_rotation():
    joint = Joint('shoulder', 0, 'parent', 'child')
    joint.set_origin_xyz([0.0, 0.100000003, 0.237780001])
    # G1 left shoulder: near-zero pitch/yaw previously made cosines round
    # independently to one, so the resulting matrix was no longer a rotation.
    joint.set_origin_rpy([0.27931, 5.4949e-5, -0.00019159])
    joint.set_type('revolute', [0.0, 1.0, 0.0])
    for angle in [0.0, 0.7, -1.2]:
        actual = np.asarray(joint.get_transformation_matrix_function()(angle), float)
        rot = actual[:3, :3]
        np.testing.assert_allclose(rot @ rot.T, np.eye(3), atol=3e-15, rtol=0)


def test_quarter_turn_cleanup_remains_exact():
    joint = Joint('quarter_turn', 0, 'parent', 'child')
    joint.set_origin_xyz([0.0, 0.0, 0.0])
    joint.set_origin_rpy([1.5708, 0.0, 0.0])
    joint.set_type('revolute', [0.0, 1.0, 0.0])
    actual = np.asarray(joint.get_transformation_matrix_function()(0.0), float)
    assert np.all(np.isin(actual[:3, :3], [-1.0, 0.0, 1.0]))
