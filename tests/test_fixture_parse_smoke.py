"""Standalone parse smoke for the fixtures consumed ONLY by downstream repos.

branching_skew_arm.urdf, spherical_arm.urdf and mixed_spherical_arm.urdf are
referenced by the parent GRiD repo's CUDA-equivalence suites, not by any test
in this repo — so without this smoke a parser regression on them would be
invisible to URDFParser's own CI (and they would look prunable). Keep this in
sync if fixtures gain/lose external consumers.
"""
import contextlib
import io
import os

import numpy as np

from URDFParser import URDFParser

FIXDIR = os.path.join(os.path.dirname(__file__), "fixtures")


def _parse(name, **kw):
    with contextlib.redirect_stdout(io.StringIO()):
        return URDFParser().parse(os.path.join(FIXDIR, name), **kw)


def test_branching_skew_arm_parses_with_dense_S():
    robot = _parse("branching_skew_arm.urdf")
    assert robot is not None
    assert robot.robot_has_skew_axis()
    # at least one joint must carry a non-cardinal (dense) motion subspace
    dense = [jid for jid in range(robot.get_num_joints())
             if not robot.S_is_cardinal_by_id(jid)]
    assert dense, "branching_skew_arm lost its skew (Tier-B) joints"
    for jid in dense:
        S = np.asarray(robot._get_flat_S_by_id(jid), dtype=float)
        assert S.shape == (6,) and np.count_nonzero(S) >= 2


def test_spherical_fixtures_parse_with_nq_ne_nv():
    for name in ("spherical_arm.urdf", "mixed_spherical_arm.urdf"):
        robot = _parse(name)
        assert robot is not None
        assert robot.robot_has_spherical(), name
        sph = [jid for jid in range(robot.get_num_bodies())
               if robot.joint_is_spherical(jid)]
        assert sph, f"{name} lost its spherical joints"
        # quaternion positions vs 3-wide tangent: NQ > NV
        assert robot.get_num_pos() > robot.get_num_vel(), name
        for jid in sph:
            vblk = robot.get_joint_index_v(jid)
            assert len(list(vblk)) == 3, f"{name} joint {jid} v-block"
