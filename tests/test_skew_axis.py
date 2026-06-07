"""Phase-6 STAGE 1: arbitrary / skew <axis> support (Tier-B motion subspace).

A non-cardinal (skew) <axis> used to make the parser leave S=None and crash
downstream. The parser now accepts a general unit axis and emits a DENSE
6-vector motion subspace S (Tier B), while CARDINAL axes still fall out to the
exact historical literal S / rz-ry-rx transform (Tier A, byte-identical).

This module covers the PARSER + Robot S-classification contract. The CUDA
codegen Tier-B emit for inverse_dynamics + crba is validated separately against
the RBDReference numpy oracle and Pinocchio (RevoluteUnaligned /
PrismaticUnaligned).
"""
import contextlib
import io
import os

import numpy as np

from URDFParser import URDFParser

FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "skew_axis_arm.urdf")


def _parse(path, **kw):
    with contextlib.redirect_stdout(io.StringIO()):
        return URDFParser().parse(path, **kw)


def test_skew_axis_parses_dense_unit_S():
    robot = _parse(FIXTURE)
    assert robot is not None
    assert robot.get_num_joints() == 3

    inv_sqrt3 = 1.0 / np.sqrt(3.0)
    inv_sqrt2 = 1.0 / np.sqrt(2.0)

    # joint_1: revolute about (1,1,1) -> S = [axis_unit; 0]
    np.testing.assert_allclose(
        robot.get_S_by_id(0), [inv_sqrt3, inv_sqrt3, inv_sqrt3, 0, 0, 0], atol=1e-12)
    # joint_2: revolute about (1,1,0)
    np.testing.assert_allclose(
        robot.get_S_by_id(1), [inv_sqrt2, inv_sqrt2, 0, 0, 0, 0], atol=1e-12)
    # joint_3: prismatic along (0,1,1) -> S = [0; axis_unit]
    np.testing.assert_allclose(
        robot.get_S_by_id(2), [0, 0, 0, 0, inv_sqrt2, inv_sqrt2], atol=1e-12)


def test_skew_joints_classified_non_cardinal():
    robot = _parse(FIXTURE)
    for jid in range(robot.get_num_joints()):
        assert not robot.S_is_cardinal_by_id(jid), jid
    assert robot.robot_has_skew_axis() is True


def test_signed_index_helpers_fail_loudly_on_skew():
    """Un-ported (signed-index) algorithms must raise, not silently emit wrong
    code, when they hit a skew joint. The Tier-B dense path is the supported
    one for STAGE 1 (inverse_dynamics + crba)."""
    robot = _parse(FIXTURE)
    import pytest

    for fn in (robot.get_S_index_by_id, robot.get_S_sign_by_id):
        with pytest.raises(ValueError, match="SKEW/general"):
            fn(0)


def test_cardinal_robot_unchanged():
    """A cardinal-axis robot still classifies as all-cardinal and exposes the
    signed-index fast path (the byte-identical Tier-A guarantee in code form)."""
    iiwa = os.path.join(
        os.path.dirname(__file__), "..", "..", "robot_assets", "iiwa14.urdf")
    if not os.path.exists(iiwa):
        import pytest
        pytest.skip("iiwa14.urdf asset not present")
    robot = _parse(iiwa)
    assert robot.robot_has_skew_axis() is False
    for jid in range(robot.get_num_joints()):
        assert robot.S_is_cardinal_by_id(jid)
        # cardinal joints keep their signed unit index (no raise)
        robot.get_signed_S_index_by_id(jid)
