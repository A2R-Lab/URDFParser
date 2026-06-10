"""URDF <limit> velocity/effort parsing.

The position limits (lower/upper) were already parsed; this covers the newly
added velocity + effort attributes (metadata, surfaced on the binding handle —
not consumed by any kernel). Unspecified attributes parse as None.
"""
import contextlib
import io
import math

from URDFParser import URDFParser


def _limit_urdf(tmp_path, limit_attrs):
    urdf = (
        '<robot name="lim">'
        '<link name="base"/>'
        '<link name="l1"><inertial><mass value="1.0"/>'
        '<inertia ixx="1" ixy="0" ixz="0" iyy="1" iyz="0" izz="1"/>'
        '</inertial></link>'
        '<joint name="j1" type="revolute"><parent link="base"/>'
        f'<child link="l1"/><axis xyz="0 0 1"/><limit {limit_attrs}/></joint>'
        '</robot>'
    )
    path = tmp_path / "lim.urdf"
    path.write_text(urdf)
    return str(path)


def test_velocity_effort_limits_parsed(tmp_path):
    urdf = _limit_urdf(tmp_path, 'lower="-1.5" upper="1.5" velocity="2.5" effort="120.0"')
    with contextlib.redirect_stdout(io.StringIO()):
        robot = URDFParser().parse(urdf)
    assert robot.get_joint_limits_by_id(0) == [-1.5, 1.5]
    assert robot.get_velocity_limit_by_id(0) == 2.5
    assert robot.get_effort_limit_by_id(0) == 120.0


def test_velocity_effort_limits_optional(tmp_path):
    # only position limits present -> velocity/effort stay None
    urdf = _limit_urdf(tmp_path, 'lower="-1.0" upper="1.0"')
    with contextlib.redirect_stdout(io.StringIO()):
        robot = URDFParser().parse(urdf)
    assert robot.get_velocity_limit_by_id(0) is None
    assert robot.get_effort_limit_by_id(0) is None
    # and an unbounded position side is still inf (unchanged behavior)
    lo, hi = robot.get_joint_limits_by_id(0)
    assert lo == -1.0 and hi == 1.0 and not math.isinf(lo)
