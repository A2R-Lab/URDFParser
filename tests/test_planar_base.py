"""The planar_base parse option.

planar_base=True attaches the URDF root to a world link through a planar joint
named planar_base_joint (plane normal +Z, identity origin). The option must
give the same Robot as a URDF that declares that joint, and it excludes
floating_base.
"""
import contextlib
import io
import os

import numpy as np
import pytest

from URDFParser import URDFParser
from URDFParser.errors import URDFParseError

FIXDIR = os.path.join(os.path.dirname(__file__), "fixtures")
PLANAR_ARM = os.path.join(FIXDIR, "planar_arm.urdf")

_BODY = """
  <link name="base">
    <inertial><origin xyz="0.01 0.02 0.03"/><mass value="4.0"/>
      <inertia ixx="0.2" ixy="0.01" ixz="0.0" iyy="0.3" iyz="0.02" izz="0.25"/></inertial>
  </link>
  <link name="arm">
    <inertial><origin xyz="0.0 0.05 0.1"/><mass value="1.5"/>
      <inertia ixx="0.03" ixy="0.0" ixz="0.001" iyy="0.04" iyz="0.0" izz="0.02"/></inertial>
  </link>
  <joint name="shoulder" type="revolute">
    <parent link="base"/><child link="arm"/>
    <origin xyz="0.1 0.0 0.3" rpy="0 0.2 0"/><axis xyz="0 1 0"/>
    <limit lower="-1.5" upper="2.0" effort="10" velocity="3"/>
  </joint>
"""

_WORLD_PLANAR = """
  <link name="world"/>
  <joint name="planar_base_joint" type="planar">
    <parent link="world"/><child link="base"/><axis xyz="0 0 1"/>
  </joint>
"""

_WORLD_FIXED = """
  <link name="world"/>
  <joint name="world_to_base" type="fixed">
    <parent link="world"/><child link="base"/><origin xyz="{xyz}" rpy="{rpy}"/>
  </joint>
"""

_WORLD_PLANAR_ORIGIN = """
  <link name="world"/>
  <joint name="planar_base_joint" type="planar">
    <parent link="world"/><child link="base"/><origin xyz="{xyz}" rpy="{rpy}"/>
    <axis xyz="0 0 1"/>
  </joint>
"""


def _write(tmp_path, name, *parts):
    path = tmp_path / name
    path.write_text('<?xml version="1.0"?>\n<robot name="planar_base_case">'
                    + "".join(parts) + "</robot>\n")
    return str(path)


def _parse(path, **kw):
    with contextlib.redirect_stdout(io.StringIO()):
        return URDFParser().parse(path, **kw)


def _signature(robot):
    """Every field of the parsed tree that the generated model depends on."""
    joints = []
    for joint in robot.get_joints_ordered_by_id():
        joints.append((
            joint.get_id(), joint.get_name(), joint.jtype, joint.get_parent(),
            joint.get_child(), tuple(joint.joint_limits),
            np.array(joint.get_transformation_matrix_function()(0.37), dtype=float).tolist(),
            np.array(joint.get_joint_subspace(), dtype=float).tolist(),
            float(joint.get_damping()),
        ))
    links = []
    for link in robot.get_links_ordered_by_id():
        links.append((link.get_id(), link.get_name(), bool(link.is_dummy_link()),
                      np.array(link.get_spatial_inertia(), dtype=float).tolist()))
    return joints, links, robot.get_num_pos(), robot.get_num_vel()


def test_option_equals_declared_planar_joint(tmp_path):
    declared = _parse(_write(tmp_path, "declared.urdf", _WORLD_PLANAR, _BODY))
    option = _parse(_write(tmp_path, "option.urdf", _BODY), planar_base=True)
    assert _signature(option) == _signature(declared)
    names = [j.get_name() for j in option.get_joints_ordered_by_id()]
    assert names == ["planar_base_joint__sub0", "planar_base_joint__sub1",
                     "planar_base_joint__sub2", "shoulder"]
    assert [j.jtype for j in option.get_joints_ordered_by_id()[:3]] == [
        "prismatic", "prismatic", "continuous"]
    assert option.get_num_pos() == option.get_num_vel() == 4
    assert option.planar_base is True
    assert option.floating_base is False
    assert declared.planar_base is False


def test_default_is_unchanged(tmp_path):
    path = _write(tmp_path, "fixed.urdf", _BODY)
    robot = _parse(path)
    assert robot.planar_base is False
    assert _signature(robot) == _signature(_parse(path, planar_base=False))
    assert robot.get_num_vel() == 1


def test_base_limits_are_unbounded(tmp_path):
    robot = _parse(_write(tmp_path, "option.urdf", _BODY), planar_base=True)
    joints = robot.get_joints_ordered_by_id()
    for joint in joints[:3]:
        assert joint.joint_limits == [float("-inf"), float("inf")]
    assert joints[3].joint_limits == [-1.5, 2.0]


def test_mutually_exclusive_with_floating_base(tmp_path):
    path = _write(tmp_path, "fixed.urdf", _BODY)
    with pytest.raises(URDFParseError, match="mutually exclusive"):
        _parse(path, floating_base=True, planar_base=True)


@pytest.mark.parametrize("xyz,rpy", [("0 0 0", "0 0 0"), ("0.3 -0.2 0.5", "0 0 0.4")])
def test_explicit_world_root_with_fixed_joint_is_replaced(tmp_path, xyz, rpy):
    """The fixed joint from world is replaced and its origin stays on the first
    sub-joint, the same model as a planar joint declared with that origin."""
    option = _parse(_write(tmp_path, "world_fixed.urdf",
                           _WORLD_FIXED.format(xyz=xyz, rpy=rpy), _BODY), planar_base=True)
    declared = _parse(_write(tmp_path, "world_planar.urdf",
                             _WORLD_PLANAR_ORIGIN.format(xyz=xyz, rpy=rpy), _BODY))
    assert _signature(option) == _signature(declared)
    assert option.get_fixed_joint_by_name("world_to_base") is None


def test_explicit_world_root_with_moving_joint_raises(tmp_path):
    moving = """
  <link name="world"/>
  <joint name="slide" type="prismatic">
    <parent link="world"/><child link="base"/><axis xyz="1 0 0"/>
    <limit lower="-1" upper="1" effort="1" velocity="1"/>
  </joint>
"""
    with pytest.raises(URDFParseError, match="to be fixed"):
        _parse(_write(tmp_path, "moving.urdf", moving, _BODY), planar_base=True)


def test_explicit_world_root_with_two_children_raises(tmp_path):
    second = """
  <link name="other">
    <inertial><mass value="1.0"/>
      <inertia ixx="0.1" ixy="0" ixz="0" iyy="0.1" iyz="0" izz="0.1"/></inertial>
  </link>
  <joint name="world_to_other" type="fixed">
    <parent link="world"/><child link="other"/>
  </joint>
"""
    path = _write(tmp_path, "two.urdf",
                  _WORLD_FIXED.format(xyz="0 0 0", rpy="0 0 0"), _BODY, second)
    with pytest.raises(URDFParseError, match="exactly one child joint"):
        _parse(path, planar_base=True)


def test_existing_planar_base_joint_name_raises(tmp_path):
    path = _write(tmp_path, "declared.urdf", _WORLD_PLANAR, _BODY)
    with pytest.raises(URDFParseError, match="already uses"):
        _parse(path, planar_base=True)


def test_declared_planar_fixture_gains_a_second_planar_block():
    """A URDF with its own non-root planar joint still accepts the option."""
    robot = _parse(PLANAR_ARM, planar_base=True)
    names = [j.get_name() for j in robot.get_joints_ordered_by_id()]
    assert names[:3] == ["planar_base_joint__sub0", "planar_base_joint__sub1",
                         "planar_base_joint__sub2"]
    assert robot.get_num_vel() == 7
