"""Parser-level verification that continuous joints are modeled correctly.

A URDF ``continuous`` joint is an unbounded revolute: 1-DOF, NQ=NV=1, joint
limits ``[-inf, +inf]``, and a single cardinal motion subspace column (so it
reuses the revolute ``S`` machinery and downstream codegen unchanged). GRiD
stores its generalized coordinate as a raw scalar angle (NQ=1), which differs
from Pinocchio's SO(2) ``(cos, sin)`` (NQ=2) representation *only* in the raw-q
comparison; the dynamics outputs are identical (verified against Pinocchio in
the RBDReference equivalence suite for the gen3 robot).
"""
import contextlib
import io
import math

import numpy as np

from URDFParser import URDFParser
from URDFParser.Joint import Joint


def _continuous_urdf(tmp_path, axis="0 0 1"):
    urdf = (
        '<robot name="cont">'
        '<link name="base"/>'
        '<link name="l1"><inertial><mass value="1.0"/>'
        '<inertia ixx="1" ixy="0" ixz="0" iyy="1" iyz="0" izz="1"/>'
        '</inertial></link>'
        '<joint name="jc" type="continuous"><parent link="base"/>'
        f'<child link="l1"/><axis xyz="{axis}"/></joint>'
        '</robot>'
    )
    path = tmp_path / "cont.urdf"
    path.write_text(urdf)
    return str(path)


def test_continuous_joint_parses_as_unbounded_revolute(tmp_path):
    urdf = _continuous_urdf(tmp_path)
    with contextlib.redirect_stdout(io.StringIO()):
        robot = URDFParser().parse(urdf)
    assert robot is not None
    joint = robot.get_joints_ordered_by_id()[0]
    assert joint.jtype == "continuous"
    # 1-DOF, NQ == NV == 1 (no quaternion / SO(2) coordinate expansion).
    assert joint.get_num_dof() == 1
    assert robot.get_num_vel() == 1
    assert robot.get_num_pos() == 1
    # Unbounded limits.
    lower, upper = joint.get_joint_limits()
    assert lower == -math.inf and upper == math.inf
    # Single cardinal motion subspace column (Z rotation) -> reuses the
    # single-signed-index S machinery (no general-axis emit needed).
    S = np.asarray(joint.get_joint_subspace()).ravel()
    assert S.shape == (6,)
    assert np.count_nonzero(S) == 1
    assert np.isclose(abs(S[2]), 1.0)


def test_continuous_joint_transform_matches_revolute(tmp_path):
    """A continuous joint about Z produces the same X(theta) as a revolute."""
    urdf = _continuous_urdf(tmp_path)
    with contextlib.redirect_stdout(io.StringIO()):
        robot = URDFParser().parse(urdf)
    cont = robot.get_joints_ordered_by_id()[0]
    xfunc = cont.get_transformation_matrix_function()

    # build an equivalent revolute joint and compare the transform numerically
    rev = Joint("jr", 0, "base", "l1")
    rev.set_origin_xyz([0.0, 0.0, 0.0])
    rev.set_origin_rpy([0.0, 0.0, 0.0])
    rev.set_type("revolute", [0.0, 0.0, 1.0])
    rfunc = rev.get_transformation_matrix_function()

    for theta in (0.0, 0.3, 1.5707, 2.5, -1.1):
        np.testing.assert_allclose(
            np.asarray(xfunc(theta), dtype=float),
            np.asarray(rfunc(theta), dtype=float),
            atol=1e-12,
        )
