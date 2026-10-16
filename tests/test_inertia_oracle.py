"""Pinocchio as the oracle for what the parser builds from a URDF.

Two checks over every fixture pinocchio can also build: the spatial inertia of
every body (mass, first moment and the tensor about the joint origin, fixed-joint
children folded into their parent exactly as pinocchio folds them) and the
world placement of every joint frame at random configurations. The tensor check
is what the 2026-10-16 UR5e finding needed: an <inertial><origin rpy> that is
not zero rotates the tensor, and the parser used to take it unrotated.
"""
import contextlib
import glob
import io
import os

import numpy as np
import pytest

pin = pytest.importorskip("pinocchio")
from URDFParser import URDFParser  # noqa: E402

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")
# Fixtures pinocchio's URDF reader cannot build: helical, spherical and
# translation joints are parser extensions with no URDF joint type (their
# equivalence lives in the sibling tests against hand-built pinocchio models).
NOT_IN_PINOCCHIO_URDF = {"helical_arm.urdf", "spherical_arm.urdf", "mixed_spherical_arm.urdf",
                         "translation_arm.urdf"}
# Joint types whose single scalar q the kinematics check can feed to both sides
# directly; the manifold-valued joints keep their own tests.
SCALAR_JOINTS = {"revolute", "continuous", "prismatic"}

# pinocchio's URDF loader ignores a planar joint's <axis>, so a skew planar
# normal has no URDF oracle; test_planar_translation_decomposition checks that
# fixture's kinematics against a programmatic pinocchio model. Its inertias are
# still checked here.
NO_URDF_KINEMATICS_ORACLE = {"skew_planar_arm.urdf"}

ALL = sorted(os.path.basename(p) for p in glob.glob(os.path.join(FIXTURES, "*.urdf")))
ORACLE_FIXTURES = [f for f in ALL if f not in NOT_IN_PINOCCHIO_URDF]
KINEMATICS_FIXTURES = [f for f in ORACLE_FIXTURES if f not in NO_URDF_KINEMATICS_ORACLE]


def _parse(name):
    with contextlib.redirect_stdout(io.StringIO()):
        return URDFParser().parse(os.path.join(FIXTURES, name), floating_base=False)


def _pin_model(name):
    return _snapped_pin_model(os.path.join(FIXTURES, name))


def _snapped_pin_model(path):
    """pinocchio built from the URDF with every <origin rpy> snapped to the
    quarter-turn grid the parser snaps joint origins to (Joint._snap_to_pi_grid,
    tolerance 1e-5): a URDF that writes 3.14159 describes a robot 2.7e-6 rad
    away from the one the parser models, and the oracle must model the parser's
    robot. RBDReference's pinocchio backend snaps the same way."""
    from bs4 import BeautifulSoup
    from URDFParser.Joint import _snap_to_pi_grid
    soup = BeautifulSoup(open(path).read(), "xml")
    for origin in soup.find_all("origin"):
        if origin.has_attr("rpy") and len(origin["rpy"].split()) == 3:
            origin["rpy"] = " ".join(repr(float(_snap_to_pi_grid(float(v)))) for v in origin["rpy"].split())
    return pin.buildModelFromXML(str(soup))


def _pin_joint_of_body(model, link_name):
    """The pinocchio joint carrying a URDF link (a fixed-joint child maps to the
    joint its body was folded into; a parser-decomposed planar joint's real link
    maps to pinocchio's single planar joint)."""
    return model.frames[model.getFrameId(link_name)].parentJoint


@pytest.mark.parametrize("name", ORACLE_FIXTURES)
def test_every_body_inertia_matches_pinocchio(name):
    robot = _parse(name)
    model = _pin_model(name)
    checked = 0
    for joint in robot.get_joints_ordered_by_id():
        link = robot.get_link_by_name(joint.child)
        if link.is_dummy_link():
            continue        # massless intermediate of a decomposed multi-DOF joint
        S = np.asarray(link.get_spatial_inertia(), dtype=float)        # angular first
        P = model.inertias[_pin_joint_of_body(model, link.name)].matrix()      # linear first
        scale = max(np.abs(P).max(), 1.0)
        np.testing.assert_allclose(S[3, 3], P[0, 0], rtol=0, atol=1e-12 * scale, err_msg=f"{name} {joint.name} mass")
        np.testing.assert_allclose(S[:3, 3:], P[3:, :3], rtol=0, atol=1e-12 * scale, err_msg=f"{name} {joint.name} first moment")
        np.testing.assert_allclose(S[:3, :3], P[3:, 3:], rtol=0, atol=1e-12 * scale, err_msg=f"{name} {joint.name} inertia about the joint origin")
        np.testing.assert_allclose(S[3:, 3:], P[:3, :3], rtol=0, atol=1e-12 * scale, err_msg=f"{name} {joint.name} mass block")
        checked += 1
    assert checked > 0


def test_rotated_inertial_origin_is_rotated_into_the_link_frame():
    """The fixture's upper-arm tensor is given in a frame rotated by a quarter
    turn about x plus two more angles; unrotated it is 3 % off the link-frame
    tensor pinocchio reports. The folded tool plate behind the fixed joint has
    its own rotated inertial and must land in the wrist exactly."""
    robot = _parse("rotated_inertial_arm.urdf")
    model = _pin_model("rotated_inertial_arm.urdf")
    link = robot.get_link_by_name("upper-arm")
    S = np.asarray(link.get_spatial_inertia(), dtype=float)
    P = model.inertias[model.getJointId("shoulder-pan")].matrix()
    np.testing.assert_allclose(S[:3, :3], P[3:, 3:], rtol=0, atol=1e-14)
    raw = np.asarray(link.inertia.to_matrix(), dtype=float)
    c = np.array([float(link.origin.translation.x), float(link.origin.translation.y), float(link.origin.translation.z)])
    cx = np.array([[0, -c[2], c[1]], [c[2], 0, -c[0]], [-c[1], c[0], 0]])
    unrotated = raw + link.mass * cx @ cx.T
    assert np.abs(unrotated - P[3:, 3:]).max() > 1e-3
    wrist = np.asarray(robot.get_link_by_name("wrist").get_spatial_inertia(), dtype=float)
    assert robot.get_fixed_joint_by_name("tool0-flange") is not None
    np.testing.assert_allclose(wrist[:3, :3], model.inertias[model.getJointId("wrist-slide")].matrix()[3:, 3:], rtol=0, atol=1e-14)


@pytest.mark.parametrize("name", KINEMATICS_FIXTURES)
def test_joint_placements_match_pinocchio_at_random_configurations(name):
    """World placement of every real body's joint frame. A planar joint the
    parser decomposes into x, y and yaw sub-joints is fed to pinocchio as its
    native q = [x, y, cos(yaw), sin(yaw)]; the sub-joints' massless
    intermediate links have no pinocchio counterpart and are composed through."""
    robot = _parse(name)
    joints = robot.get_joints_ordered_by_id()
    model = _pin_model(name)
    data = model.createData()
    assert all(j.jtype in SCALAR_JOINTS for j in joints), f"{name}: non-scalar parser joint"
    rng = np.random.default_rng(20261016)
    for _ in range(3):
        q = rng.uniform(-1.0, 1.0, len(joints))
        # pinocchio configuration: scalar joints copy their value; a decomposed
        # planar joint takes the parser's [x, y, yaw] as [x, y, cos, sin].
        q_pin = np.zeros(model.nq)
        pending = []
        for k, joint in enumerate(joints):
            if model.existJointName(joint.name):
                jm = model.joints[model.getJointId(joint.name)]
                if jm.nq == 1:
                    q_pin[jm.idx_q] = q[k]
                else:   # a continuous joint: pinocchio's unbounded revolute is (cos, sin)
                    assert jm.nq == 2 and joint.jtype == "continuous", f"{name} {joint.name}: pinocchio nq {jm.nq}"
                    q_pin[jm.idx_q:jm.idx_q + 2] = [np.cos(q[k]), np.sin(q[k])]
                continue
            pending.append(q[k])
            link = robot.get_link_by_name(joint.child)
            if link.is_dummy_link():
                continue
            jm = model.joints[_pin_joint_of_body(model, link.name)]
            assert jm.nq == 4 and len(pending) == 3, f"{name} {joint.name}: unexpected decomposition"
            x, y, yaw = pending
            q_pin[jm.idx_q:jm.idx_q + 4] = [x, y, np.cos(yaw), np.sin(yaw)]
            pending = []
        pin.forwardKinematics(model, data, q_pin)
        world = {-1: np.eye(4)}
        checked = 0
        for k, joint in enumerate(joints):
            X = np.asarray(joint.get_transformation_matrix_hom_function()(q[k]), dtype=float).reshape(4, 4)
            world[joint.get_id()] = world[robot.get_parent_id(joint.get_id())] @ X
            link = robot.get_link_by_name(joint.child)
            if link.is_dummy_link():
                continue
            expected = data.oMi[_pin_joint_of_body(model, link.name)].homogeneous
            np.testing.assert_allclose(world[joint.get_id()], expected, rtol=0, atol=1e-12, err_msg=f"{name} {joint.name}")
            checked += 1
        assert checked > 0
