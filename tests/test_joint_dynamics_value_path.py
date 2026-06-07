"""Phase-6 STAGE 2 tests: <dynamics damping>/<dynamics friction> VALUE path
and degenerate-<inertial> strict-mode detection.

Covers:
  * parser reads damping AND friction (friction was NEVER parsed before),
  * the RBDReference value-path bias tau += damping*qd + friction*sign(qd)
    (opt-in via RBDReference(use_joint_dynamics=True); DEFAULT off to stay
    consistent with bare pin.rnea/pin.aba which ignore model.damping/friction),
  * inverse_dynamics <-> forward_dynamics consistency with the bias active,
  * strict-mode rejection of a degenerate (zero-mass / non-PD) <inertial> on a
    real moving body, with lenient mode preserving legacy silent-zeroing.
"""
import contextlib
import copy
import io
import os

import numpy as np
import pytest

from URDFParser import URDFParser
from URDFParser.errors import URDFParseError

try:
    from RBDReference import RBDReference
except Exception:  # pragma: no cover - RBDReference optional when run standalone
    RBDReference = None

_FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")
DAMP_FRIC = os.path.join(_FIXTURES, "damping_friction_arm.urdf")
DEGENERATE = os.path.join(_FIXTURES, "degenerate_inertial_arm.urdf")


def _parse(path, **kw):
    with contextlib.redirect_stdout(io.StringIO()):
        return URDFParser().parse(path, **kw)


# --------------------------------------------------------------------------- #
# Parser: damping + friction                                                  #
# --------------------------------------------------------------------------- #
def test_parser_reads_damping_and_friction():
    robot = _parse(DAMP_FRIC)
    assert robot is not None
    assert robot.robot_has_joint_damping()
    assert robot.robot_has_joint_friction()
    # joint_1: damping 0.7 + friction 0.3, joint_2: damping 0.4, friction 0.
    damps = sorted(robot.get_damping_by_id(j) for j in range(robot.get_num_vel()))
    frics = sorted(robot.get_friction_by_id(j) for j in range(robot.get_num_vel()))
    assert damps == pytest.approx([0.4, 0.7])
    assert frics == pytest.approx([0.0, 0.3])


def test_friction_defaults_zero_when_absent():
    # iiwa14 declares damping but no friction -> friction parses to 0 (no crash).
    iiwa = os.path.join(os.path.dirname(__file__), "..", "..", "robot_assets", "iiwa14.urdf")
    robot = _parse(iiwa)
    assert robot.robot_has_joint_damping()
    assert not robot.robot_has_joint_friction()


# --------------------------------------------------------------------------- #
# Value-path bias (RBDReference oracle)                                        #
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(RBDReference is None, reason="RBDReference unavailable")
def test_bias_default_off_is_no_op():
    """DEFAULT RBDReference (use_joint_dynamics=False) must NOT apply the bias,
    so it stays identical to a robot with damping/friction stripped."""
    robot = _parse(DAMP_FRIC)
    n = robot.get_num_vel()
    np.random.seed(0)
    q, qd, qdd = np.random.randn(n), np.random.randn(n), np.random.randn(n)

    c_default = RBDReference(robot).inverse_dynamics(q, qd, qdd)[0]
    stripped = copy.deepcopy(robot)
    for j in range(stripped.get_num_bodies()):
        stripped.get_joint_by_id(j).set_damping(0)
        stripped.get_joint_by_id(j).set_friction(0)
    c_stripped = RBDReference(stripped).inverse_dynamics(q, qd, qdd)[0]
    np.testing.assert_allclose(c_default, c_stripped, atol=1e-12)


@pytest.mark.skipif(RBDReference is None, reason="RBDReference unavailable")
def test_bias_matches_analytic_damping_plus_friction():
    robot = _parse(DAMP_FRIC)
    n = robot.get_num_vel()
    np.random.seed(1)
    q, qd, qdd = np.random.randn(n), np.random.randn(n), np.random.randn(n)

    ref_on = RBDReference(robot, use_joint_dynamics=True)
    ref_off = RBDReference(robot, use_joint_dynamics=False)
    c_on = ref_on.inverse_dynamics(q, qd, qdd)[0]
    c_off = ref_off.inverse_dynamics(q, qd, qdd)[0]

    b = np.array([robot.get_damping_by_id(j) for j in range(n)])
    f = np.array([robot.get_friction_by_id(j) for j in range(n)])
    expected = b * qd + f * np.sign(qd)
    np.testing.assert_allclose(c_on - c_off, expected, atol=1e-12)
    # the bias must actually be nonzero for this fixture / qd
    assert np.linalg.norm(expected) > 0


@pytest.mark.skipif(RBDReference is None, reason="RBDReference unavailable")
def test_inverse_forward_dynamics_consistency_with_bias():
    """ABA(q, qd, ID(q, qd, qdd)) must recover qdd with the bias active in BOTH
    surfaces (same sign convention) -- the round-trip self-check."""
    robot = _parse(DAMP_FRIC)
    n = robot.get_num_vel()
    np.random.seed(3)
    q, qd, qdd = np.random.randn(n), np.random.randn(n), np.random.randn(n)
    ref = RBDReference(robot, use_joint_dynamics=True)
    c = ref.inverse_dynamics(q, qd, qdd)[0]
    qdd_aba = ref.aba(q, qd, c)
    qdd_fd = ref.forward_dynamics(q, qd, c)
    np.testing.assert_allclose(qdd_aba, qdd, atol=1e-7)
    np.testing.assert_allclose(qdd_fd, qdd_aba, atol=1e-9)


@pytest.mark.skipif(RBDReference is None, reason="RBDReference unavailable")
def test_friction_term_is_odd_in_velocity():
    robot = _parse(DAMP_FRIC)
    ref = RBDReference(robot, use_joint_dynamics=True)
    n = robot.get_num_vel()
    qd = np.array([0.5] * n)
    bp = ref._joint_dynamics_bias(qd)
    bn = ref._joint_dynamics_bias(-qd)
    # viscous damping (b*qd) and Coulomb friction (f*sign) are both odd in qd.
    np.testing.assert_allclose(bp, -bn, atol=1e-12)
    assert np.linalg.norm(bp) > 0


# --------------------------------------------------------------------------- #
# Degenerate <inertial> strict-mode detection                                 #
# --------------------------------------------------------------------------- #
def test_degenerate_inertial_lenient_default_preserved():
    # Lenient (default) silently keeps the legacy zeroing -> parse succeeds.
    robot = _parse(DEGENERATE)
    assert robot is not None


def test_degenerate_inertial_strict_raises():
    with pytest.raises(URDFParseError) as excinfo:
        _parse(DEGENERATE, strict_inertial=True)
    # error must name the offending link
    assert "link2" in str(excinfo.value)


def test_strict_mode_accepts_valid_robot():
    # A fully-specified robot (iiwa14) passes strict mode (root exempt, all
    # moving bodies have valid inertials).
    iiwa = os.path.join(os.path.dirname(__file__), "..", "..", "robot_assets", "iiwa14.urdf")
    robot = _parse(iiwa, strict_inertial=True)
    assert robot is not None
