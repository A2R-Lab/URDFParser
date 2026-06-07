"""Phase-6 STAGE 3: planar + translation/cartesian joints via parse-time
DECOMPOSITION into cardinal 1-DOF sub-joints + zero-mass dummy links.

Both joint types expand at parse time into a chain of cardinal prismatic/
revolute sub-joints joined by massless dummy links. Every emitted sub-joint is
a 1-DOF cardinal joint -> Tier-A byte-identical machinery, so NO algorithm /
kernel code changes are needed. The decomposition is EXACT, validated here
against pinocchio's NATIVE JointModelPlanar / JointModelTranslation.

  planar  -> prismatic(X) -> prismatic(Y) -> revolute(Z)   [2 dummy links]
  translation -> prismatic(X) -> prismatic(Y) -> prismatic(Z)   [2 dummy links]
"""
import contextlib
import io
import os

import numpy as np
import pytest

from URDFParser import URDFParser
from RBDReference import RBDReference

pin = pytest.importorskip("pinocchio")

FIXDIR = os.path.join(os.path.dirname(__file__), "fixtures")
PLANAR = os.path.join(FIXDIR, "planar_arm.urdf")
TRANSLATION = os.path.join(FIXDIR, "translation_arm.urdf")


def _parse(path, **kw):
    with contextlib.redirect_stdout(io.StringIO()):
        return URDFParser().parse(path, **kw)


# ---- shared inertia constants (match the URDF fixtures verbatim) ----
_LINK1 = dict(m=1.3, c=[0.05, 0.02, 0.10],
              I=dict(ixx=0.05, ixy=0.01, ixz=0.005, iyy=0.06, iyz=0.002, izz=0.04))
_LINK2 = dict(m=0.9, c=[0.04, -0.03, 0.08],
              I=dict(ixx=0.04, ixy=0.003, ixz=0.001, iyy=0.05, iyz=0.004, izz=0.03))
_ORIGIN = [0.1, 0.0, 0.15]      # planar/cart joint origin xyz
_ELBOW = [0.0, 0.0, 0.20]       # elbow joint origin xyz


def _inertia(spec):
    I = spec["I"]
    Imat = np.array([[I["ixx"], I["ixy"], I["ixz"]],
                     [I["ixy"], I["iyy"], I["iyz"]],
                     [I["ixz"], I["iyz"], I["izz"]]])
    return pin.Inertia(spec["m"], np.array(spec["c"]), Imat)


def _build_pin(model_joint):
    """Native pinocchio model: base -[model_joint]-> link1 -[revolute Y]-> link2.
    `model_joint` is a JointModelPlanar or JointModelTranslation."""
    model = pin.Model()
    j1 = model.addJoint(0, model_joint, pin.SE3(np.eye(3), np.array(_ORIGIN)), "j1")
    model.appendBodyToJoint(j1, _inertia(_LINK1), pin.SE3.Identity())
    j2 = model.addJoint(j1, pin.JointModelRY(), pin.SE3(np.eye(3), np.array(_ELBOW)), "j2")
    model.appendBodyToJoint(j2, _inertia(_LINK2), pin.SE3.Identity())
    return model


def _pin_planar_q(px, py, theta, theta_last):
    # JointModelPlanar uses q = [x, y, cos, sin]; elbow revolute appends theta_last.
    return np.array([px, py, np.cos(theta), np.sin(theta), theta_last])


def _pin_translation_q(x, y, z, theta_last):
    return np.array([x, y, z, theta_last])


def _planar_G(theta):
    """Velocity-coordinate change v_pin = G(theta) @ qd_chain for the planar
    block (chain order [px, py, theta]).

    The cardinal prismatic->prismatic->revolute decomposition is an EXACT planar
    mechanism, but parameterizes velocity in STACKED-JOINT coordinates (px, py in
    the PARENT/world frame, omega about Z) -- whereas pinocchio's JointModelPlanar
    uses the SE(2) BODY-TWIST [vx, vy, omega] measured in the (co-rotating) joint
    frame. The two planar parameterizations of the SAME configuration manifold
    differ by the orthogonal q-dependent rotation G(theta) = Rz(theta)^T acting
    on the in-plane translation pair (omega is shared). A fixed-axis cardinal
    chain CANNOT reproduce pin's co-rotating axes, so this convention difference
    is intrinsic, not a bug. Dynamics are exact in EITHER basis and related by:
        M_chain   = G^T M_pin G,   tau_chain = G^T tau_pin
    (verified to machine precision below). For TRANSLATION, theta is absent ->
    G == I -> the decomposition matches pin BIT-FOR-BIT (separate test).
    """
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_planar_decomposition_matches_native_pinocchio_under_G(seed):
    """The decomposed planar block reproduces pinocchio's JointModelPlanar
    dynamics EXACTLY, up to the orthogonal velocity-basis change G(theta) that
    relates the stacked-joint coordinates to pin's SE(2) body twist. The full
    mass-matrix block obeys M_chain = G^T M_pin G to machine precision, and the
    gravity (configuration) torque obeys tau = G^T tau_pin (G acts only on the
    3-DOF planar block; the elbow column passes through unchanged)."""
    rng = np.random.default_rng(seed)
    robot = _parse(PLANAR)
    assert robot is not None
    assert robot.get_num_vel() == 4
    assert robot.get_num_pos() == 4   # decomposed: NQ == NV (vector group)
    ref = RBDReference(robot)

    model = _build_pin(pin.JointModelPlanar())
    data = model.createData()

    q = rng.uniform(-0.5, 0.5, 4)     # [px, py, theta, elbow]
    q_pin = _pin_planar_q(q[0], q[1], q[2], q[3])

    # 4x4 block-diagonal velocity-basis change: G(theta) on the planar block,
    # identity on the elbow DOF.
    G = np.eye(4)
    G[:3, :3] = _planar_G(q[2])

    # mass matrix: M_chain == G^T M_pin G
    M_ref = ref.crba(q)
    M_pin = pin.crba(model, data, q_pin)
    M_pin = np.triu(M_pin) + np.triu(M_pin, 1).T   # pin fills upper triangle only
    np.testing.assert_allclose(M_ref, G.T @ M_pin @ G, atol=1e-9, rtol=1e-9)

    # gravity torque (qd = qdd = 0): tau_chain == G^T tau_pin
    tau_g_ref = ref.inverse_dynamics(q, np.zeros(4), np.zeros(4))[0]
    tau_g_pin = pin.computeGeneralizedGravity(model, data, q_pin)
    np.testing.assert_allclose(tau_g_ref, G.T @ tau_g_pin, atol=1e-9, rtol=1e-9)


def test_planar_decomposition_self_consistent():
    """Beyond the pin G-transform check, the decomposed planar mechanism must be
    a self-consistent rigid body: forward_dynamics inverts inverse_dynamics on
    its OWN (stacked) coordinates. This validates the full RNEA/ABA value path
    through the dummy-link chain independent of pinocchio's parameterization."""
    rng = np.random.default_rng(11)
    robot = _parse(PLANAR)
    ref = RBDReference(robot)
    for _ in range(8):
        q = rng.uniform(-0.5, 0.5, 4)
        qd = rng.uniform(-1.0, 1.0, 4)
        qdd = rng.uniform(-1.0, 1.0, 4)
        tau = ref.inverse_dynamics(q, qd, qdd)[0]
        qdd_back = ref.forward_dynamics(q, qd, tau)
        np.testing.assert_allclose(qdd_back, qdd, atol=1e-9, rtol=1e-9)


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_translation_decomposition_matches_native_pinocchio(seed):
    rng = np.random.default_rng(seed)
    robot = _parse(TRANSLATION)
    assert robot is not None
    assert robot.get_num_vel() == 4
    assert robot.get_num_pos() == 4
    ref = RBDReference(robot)

    model = _build_pin(pin.JointModelTranslation())
    data = model.createData()

    q = rng.uniform(-0.5, 0.5, 4)
    qd = rng.uniform(-1.0, 1.0, 4)
    qdd = rng.uniform(-1.0, 1.0, 4)
    q_pin = _pin_translation_q(q[0], q[1], q[2], q[3])

    tau_ref = ref.inverse_dynamics(q, qd, qdd)[0]
    tau_pin = pin.rnea(model, data, q_pin, qd, qdd)
    np.testing.assert_allclose(tau_ref, tau_pin, atol=1e-10, rtol=1e-10)

    M_ref = ref.crba(q)
    M_pin = pin.crba(model, data, q_pin)
    M_pin = np.triu(M_pin) + np.triu(M_pin, 1).T
    np.testing.assert_allclose(M_ref, M_pin, atol=1e-10, rtol=1e-10)

    qdd_ref = ref.forward_dynamics(q, qd, tau_pin)
    qdd_pin = pin.aba(model, data, q_pin, qd, tau_pin)
    np.testing.assert_allclose(qdd_ref, qdd_pin, atol=1e-9, rtol=1e-9)


def test_planar_q_round_trip():
    """The user-facing planar config [px, py, theta] maps to the 3 sub-joint
    coordinates (in chain order) and back consistently."""
    robot = _parse(PLANAR)
    sub_names = ["planar_joint__sub0", "planar_joint__sub1", "planar_joint__sub2"]
    q = np.array([0.3, -0.2, 0.7, 1.1])   # [px, py, theta, elbow]
    # each sub-joint reads exactly one user coordinate, in order.
    for k, sname in enumerate(sub_names):
        jid = robot.get_joint_by_name(sname).get_id()
        idx = robot.get_joint_index_q(jid)
        assert idx == k, (sname, idx)
        assert robot.q_for_joint(jid, q) == pytest.approx(q[k])


def test_decomposition_dummy_links_flagged():
    robot = _parse(PLANAR)
    dummies = [l for l in robot.links if l.is_dummy_link() and l.get_name() != "base"]
    # two intermediate dummy links for a planar joint
    names = sorted(l.get_name() for l in dummies)
    assert names == ["planar_joint__dummy0", "planar_joint__dummy1"]
    for l in dummies:
        assert l.get_spatial_inertia() is not None
        assert np.allclose(l.get_spatial_inertia(), 0.0)
        # a massless dummy is NOT misclassified as the world base
        assert robot.get_parent_id(l.get_id()) is not None


def test_all_subjoints_cardinal():
    """Every decomposed sub-joint is a cardinal 1-DOF joint (Tier A) -> the
    signed-index fast path applies, no Tier-B/C code is reached."""
    for path in (PLANAR, TRANSLATION):
        robot = _parse(path)
        assert robot.robot_has_skew_axis() is False
        for jid in range(robot.get_num_joints()):
            assert robot.S_is_cardinal_by_id(jid), (path, jid)
