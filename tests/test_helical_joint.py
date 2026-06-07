"""Phase-6: HELICAL / SCREW joint support (parser + Robot S-classification).

A helical (screw) joint is 1-DOF (NQ=NV=1) coupling rotation about an axis with
translation along it by a `pitch` (meters / radian). URDF has no native helical
type, so it is an extension: `<joint type="helical">` (or "screw") with the
pitch carried on the <axis> as `pitch="..."`. Its motion subspace is a SINGLE
column with a COUPLED linear part S = [axis_unit; pitch*axis_unit], so it is
intrinsically Tier B (non-cardinal) and routes through the dense-6-vector path.

End-to-end dynamics equivalence vs pinocchio JointModelHelicalUnaligned lives in
RBDReference/tests/test_helical_joint_equivalence.py.
"""
import contextlib
import io
import os

import numpy as np

from URDFParser import URDFParser

FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "helical_arm.urdf")


def _parse(path, **kw):
    with contextlib.redirect_stdout(io.StringIO()):
        return URDFParser().parse(path, **kw)


def test_helical_parses_coupled_single_column_S():
    robot = _parse(FIXTURE)
    assert robot is not None
    assert robot.get_num_joints() == 3
    # 1-DOF per joint: NQ == NV == 3 (vector add, no manifold).
    assert robot.get_num_pos() == robot.get_num_vel() == 3

    # joint_1: helical about Z, pitch 0.05 -> S = [0,0,1, 0,0,0.05]
    np.testing.assert_allclose(
        robot.get_S_by_id(0), [0, 0, 1, 0, 0, 0.05], atol=1e-12)
    # joint_2: plain revolute X -> S = [1,0,0, 0,0,0]
    np.testing.assert_allclose(
        robot.get_S_by_id(1), [1, 0, 0, 0, 0, 0], atol=1e-12)
    # joint_3: screw about (1,1,1)/sqrt(3), pitch -0.03
    u = 1.0 / np.sqrt(3.0)
    np.testing.assert_allclose(
        robot.get_S_by_id(2), [u, u, u, -0.03 * u, -0.03 * u, -0.03 * u], atol=1e-12)


def test_helical_classified_non_cardinal_triggers_tierB():
    """A helical joint's coupled S has >=2 nonzero entries -> NON-cardinal, so
    `robot_has_skew_axis()` trips and the algos take the dense Tier-B path. The
    predicate must catch a coupled linear part, not only a skew ANGULAR axis."""
    robot = _parse(FIXTURE)
    assert not robot.S_is_cardinal_by_id(0)   # cardinal-Z screw: coupled -> Tier B
    assert robot.S_is_cardinal_by_id(1)       # plain revolute stays Tier A
    assert not robot.S_is_cardinal_by_id(2)   # skew screw -> Tier B
    assert robot.robot_has_skew_axis() is True


def test_helical_signed_index_helpers_fail_loudly():
    """The signed-index fast path has no single unit index for a coupled S, so it
    must raise (Tier-B consumers use the dense get_S_by_id instead)."""
    robot = _parse(FIXTURE)
    import pytest

    for jid in (0, 2):
        for fn in (robot.get_S_index_by_id, robot.get_S_sign_by_id):
            with pytest.raises(ValueError, match="SKEW/general"):
                fn(jid)


def test_screw_alias_matches_helical():
    """`type="screw"` is an accepted alias for `type="helical"`; joint_3 uses it."""
    robot = _parse(FIXTURE)
    assert robot.get_joint_by_id(2).jtype == "screw"
    assert robot.get_joint_by_id(2).get_num_dof() == 1
