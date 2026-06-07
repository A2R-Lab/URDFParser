"""Tests for the typed URDF parser exception path.

Replaces the old ``print()+exit()`` failure mode (an uncatchable SystemExit
that killed the host process) with a catchable, typed exception.
"""
import contextlib
import io

import pytest

from URDFParser import URDFParseError, UnsupportedJointTypeError, URDFParser
from URDFParser.Joint import Joint


def _write_urdf(tmp_path, joint_xml, name="bot"):
    urdf = (
        f'<robot name="{name}">'
        '<link name="base"/>'
        '<link name="l1"><inertial><mass value="1.0"/>'
        '<inertia ixx="1" ixy="0" ixz="0" iyy="1" iyz="0" izz="1"/>'
        '</inertial></link>'
        f'{joint_xml}'
        '</robot>'
    )
    path = tmp_path / f"{name}.urdf"
    path.write_text(urdf)
    return str(path)


@pytest.mark.parametrize("bad_type", ["bogus", "floating2", "wheel", "gimbal"])
def test_unsupported_joint_type_raises_typed_error(tmp_path, bad_type):
    """An unsupported joint type raises a catchable typed error, not exit()."""
    urdf = _write_urdf(
        tmp_path,
        f'<joint name="j1" type="{bad_type}"><parent link="base"/>'
        '<child link="l1"/><axis xyz="0 0 1"/></joint>',
    )
    parser = URDFParser()
    with pytest.raises(UnsupportedJointTypeError) as excinfo:
        with contextlib.redirect_stdout(io.StringIO()):
            parser.parse(urdf)
    # carries the offending type + is part of the URDFParseError family
    assert excinfo.value.jtype == bad_type
    assert isinstance(excinfo.value, URDFParseError)
    assert "j1" in str(excinfo.value)


def test_unsupported_joint_type_caught_as_base_class(tmp_path):
    """Callers can catch the whole family with the base URDFParseError."""
    urdf = _write_urdf(
        tmp_path,
        '<joint name="weird" type="bogus"><parent link="base"/>'
        '<child link="l1"/><axis xyz="0 0 1"/></joint>',
    )
    with pytest.raises(URDFParseError):
        with contextlib.redirect_stdout(io.StringIO()):
            URDFParser().parse(urdf)


def test_set_type_raises_directly():
    """Joint.set_type raises the typed error directly (unit level)."""
    j = Joint("j", 0, "a", "b")
    j.set_origin_xyz([0.0, 0.0, 0.0])
    j.set_origin_rpy([0.0, 0.0, 0.0])
    with pytest.raises(UnsupportedJointTypeError):
        j.set_type("bogus", [0.0, 0.0, 1.0])


def test_valid_revolute_still_parses(tmp_path):
    """A supported joint type parses normally (no regression)."""
    urdf = _write_urdf(
        tmp_path,
        '<joint name="j1" type="revolute"><parent link="base"/>'
        '<child link="l1"/><axis xyz="0 0 1"/>'
        '<limit lower="-1.0" upper="1.0" effort="1" velocity="1"/></joint>',
    )
    with contextlib.redirect_stdout(io.StringIO()):
        robot = URDFParser().parse(urdf)
    assert robot is not None
    assert robot.get_num_vel() == 1
    assert robot.get_num_pos() == 1
