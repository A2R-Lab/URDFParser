"""Parser-side GROUNDWORK tests for planar and spherical joints.

These joints now PARSE into a correct native representation instead of
raising UnsupportedJointTypeError:

  * planar    -- 3 DOF, NQ==NV==3 (vector group), 6x3 motion subspace
                 (2 in-plane translations + 1 normal-axis rotation).
  * spherical -- 3 DOF, NV=3 / NQ=4 (unit quaternion), 6x3 angular-only
                 motion subspace.

What this slice DELIVERS (parser + Robot bookkeeping):
  - dof / local_q_dim / position_symbols / 6x3 S,
  - multi-slot q/v index ranges for the non-root multi-DOF joint,
  - NQ accounting that counts the spherical quaternion offset.

What is DEFERRED (documented, not landed here):
  - numpy-reference dynamics recursions consuming the multi-column S,
  - the SO(3) per-joint retract in RBDReference.integrate/dIntegrate for
    spherical (today hardcoded to a single free-flyer prefix),
  - the CUDA codegen emit of a multi-column non-root S
    (Robot.get_S_index_by_id assumes a single signed unit axis).
See docs/open-tasks/joint_types_plan.md.
"""
import contextlib
import io

import numpy as np

from URDFParser import URDFParser
from URDFParser.Joint import Joint


def _single_joint_urdf(tmp_path, jtype, axis="0 0 1"):
    urdf = (
        f'<robot name="{jtype}_bot">'
        '<link name="base"/>'
        '<link name="l1"><inertial><mass value="1.0"/>'
        '<inertia ixx="1" ixy="0" ixz="0" iyy="1" iyz="0" izz="1"/>'
        '</inertial></link>'
        f'<joint name="j1" type="{jtype}"><parent link="base"/>'
        f'<child link="l1"/><axis xyz="{axis}"/></joint>'
        '</robot>'
    )
    path = tmp_path / f"{jtype}.urdf"
    path.write_text(urdf)
    return str(path)


def test_planar_joint_set_type_native_representation():
    j = Joint("planar_j", 0, "base", "l1")
    j.set_origin_xyz([0.0, 0.0, 0.0])
    j.set_origin_rpy([0.0, 0.0, 0.0])
    j.set_type("planar", [0.0, 0.0, 1.0])
    assert j.jtype == "planar"
    assert j.get_num_dof() == 3
    assert j.local_q_dim == 3
    assert len(j.position_symbols) == 3
    S = np.asarray(j.get_joint_subspace())
    assert S.shape == (6, 3)
    # columns match position_symbols order [px, py, theta], in internal
    # [wx,wy,wz,vx,vy,vz] spatial order:
    np.testing.assert_allclose(S[:, 0], [0, 0, 0, 1, 0, 0])  # translation +X
    np.testing.assert_allclose(S[:, 1], [0, 0, 0, 0, 1, 0])  # translation +Y
    np.testing.assert_allclose(S[:, 2], [0, 0, 1, 0, 0, 0])  # rotation +Z


def test_spherical_joint_set_type_native_representation():
    j = Joint("ball_j", 0, "base", "l1")
    j.set_origin_xyz([0.0, 0.0, 0.0])
    j.set_origin_rpy([0.0, 0.0, 0.0])
    j.set_type("spherical")
    assert j.jtype == "spherical"
    assert j.get_num_dof() == 3          # NV = 3
    assert j.local_q_dim == 4            # NQ = 4 (unit quaternion)
    assert len(j.position_symbols) == 4
    S = np.asarray(j.get_joint_subspace())
    assert S.shape == (6, 3)
    # angular-only identity block on the rotation rows.
    np.testing.assert_allclose(S[:3, :], np.eye(3))
    np.testing.assert_allclose(S[3:, :], np.zeros((3, 3)))


def test_planar_transform_function_consumes_three_coordinates():
    j = Joint("planar_j", 0, "base", "l1")
    j.set_origin_xyz([0.0, 0.0, 0.0])
    j.set_origin_rpy([0.0, 0.0, 0.0])
    j.set_type("planar", [0.0, 0.0, 1.0])
    xfunc = j.get_transformation_matrix_hom_function()
    # px, py, theta=0 -> pure translation
    X = np.asarray(xfunc([0.5, -0.3, 0.0]), dtype=float)
    np.testing.assert_allclose(X[:3, 3], [0.5, -0.3, 0.0], atol=1e-12)
    np.testing.assert_allclose(X[:3, :3], np.eye(3), atol=1e-12)
    # rotation about +Z by pi/2 maps +X -> +Y in the homogeneous frame
    X2 = np.asarray(xfunc([0.0, 0.0, np.pi / 2]), dtype=float)
    np.testing.assert_allclose(X2[:3, 3], [0.0, 0.0, 0.0], atol=1e-9)


def test_planar_robot_parses_with_three_dof_block(tmp_path):
    urdf = _single_joint_urdf(tmp_path, "planar")
    with contextlib.redirect_stdout(io.StringIO()):
        robot = URDFParser().parse(urdf)
    assert robot is not None
    assert robot.get_num_vel() == 3
    assert robot.get_num_pos() == 3   # planar is a vector group: NQ == NV
    jid = robot.get_joints_ordered_by_id()[0].get_id()
    assert robot.get_joint_index_q(jid) == [0, 1, 2]
    assert robot.get_joint_index_v(jid) == [0, 1, 2]


def test_spherical_robot_parses_with_quaternion_nq_offset(tmp_path):
    urdf = _single_joint_urdf(tmp_path, "spherical")
    with contextlib.redirect_stdout(io.StringIO()):
        robot = URDFParser().parse(urdf)
    assert robot is not None
    assert robot.get_num_vel() == 3
    assert robot.get_num_pos() == 4   # spherical: NQ = NV + 1 (quaternion)
    jid = robot.get_joints_ordered_by_id()[0].get_id()
    assert robot.get_joint_index_q(jid) == [0, 1, 2, 3]
    assert robot.get_joint_index_v(jid) == [0, 1, 2]
