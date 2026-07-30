"""Chained-mimic flattening in ``resolve_mimic_targets``.

A mimic whose target is itself a mimic is legal URDF (SDF/robot vendors emit
them for coupled grippers). The parser flattens the chain at resolve time so
every downstream consumer (dense index maps, codegen, oracle) sees a one-hop
table: q_c = m_c*(m_b*q_a + o_b) + o_c  ==>  target=a, multiplier=m_c*m_b,
offset=m_c*o_b + o_c. Cycles raise; a chain ending at a mimic-of-fixed
degenerates to a constant exactly like the one-hop case.
"""
import contextlib
import io

import numpy as np
import pytest

from URDFParser import URDFParser


_INERTIAL = (
    '<inertial><mass value="1.0"/>'
    '<inertia ixx="1" ixy="0" ixz="0" iyy="1" iyz="0" izz="1"/></inertial>'
)


def _revolute(name, parent, child, mimic=None):
    mimic_xml = ""
    if mimic is not None:
        tgt, mult, off = mimic
        mimic_xml = f'<mimic joint="{tgt}" multiplier="{mult}" offset="{off}"/>'
    return (
        f'<joint name="{name}" type="revolute"><parent link="{parent}"/>'
        f'<child link="{child}"/><axis xyz="0 0 1"/>'
        '<origin xyz="0 0 0.1" rpy="0 0 0"/>'
        '<limit lower="-3" upper="3" effort="10" velocity="10"/>'
        f"{mimic_xml}</joint>"
    )


def _parse(tmp_path, joints_xml, n_links, name="bot"):
    links = '<link name="base"/>' + "".join(
        f'<link name="l{k}">{_INERTIAL}</link>' for k in range(1, n_links + 1)
    )
    urdf = f'<robot name="{name}">{links}{joints_xml}</robot>'
    path = tmp_path / f"{name}.urdf"
    path.write_text(urdf)
    parser = URDFParser()
    with contextlib.redirect_stdout(io.StringIO()):
        return parser.parse(str(path))


def test_two_hop_chain_flattens_to_root_target(tmp_path):
    robot = _parse(
        tmp_path,
        _revolute("j1", "base", "l1")
        + _revolute("j2", "l1", "l2", mimic=("j1", 2.0, 0.1))
        + _revolute("j3", "l2", "l3", mimic=("j2", -1.5, 0.2)),
        3,
    )
    j1 = robot.get_joint_by_name("j1")
    j3 = robot.get_joint_by_name("j3")
    assert j3.mimic_target_id == j1.get_id()
    assert j3.get_mimic_multiplier() == pytest.approx(-3.0)          # -1.5 * 2.0
    assert j3.get_mimic_offset() == pytest.approx(0.05)              # -1.5*0.1 + 0.2
    # semantic check: the flattened relation reproduces the composed value
    q = np.zeros(robot.get_num_pos())
    q[robot.get_joint_index_q(j1.get_id())] = 0.7
    q_j2 = 2.0 * 0.7 + 0.1
    expected = -1.5 * q_j2 + 0.2
    assert robot.q_for_joint(j3.get_id(), q) == pytest.approx(expected)


def test_three_hop_chain_flattens_transitively(tmp_path):
    robot = _parse(
        tmp_path,
        _revolute("j1", "base", "l1")
        + _revolute("j2", "l1", "l2", mimic=("j1", 2.0, 0.5))
        + _revolute("j3", "l2", "l3", mimic=("j2", 3.0, -1.0))
        + _revolute("j4", "l3", "l4", mimic=("j3", 0.5, 0.25)),
        4,
    )
    j1 = robot.get_joint_by_name("j1")
    j4 = robot.get_joint_by_name("j4")
    assert j4.mimic_target_id == j1.get_id()
    # m = 0.5*3*2 ; o = 0.5*(3*0.5 - 1) + 0.25
    assert j4.get_mimic_multiplier() == pytest.approx(3.0)
    assert j4.get_mimic_offset() == pytest.approx(0.5)


def test_one_hop_mimic_unchanged(tmp_path):
    robot = _parse(
        tmp_path,
        _revolute("j1", "base", "l1")
        + _revolute("j2", "l1", "l2", mimic=("j1", 2.0, 0.1)),
        2,
    )
    j1 = robot.get_joint_by_name("j1")
    j2 = robot.get_joint_by_name("j2")
    assert j2.mimic_target_id == j1.get_id()
    assert j2.get_mimic_multiplier() == pytest.approx(2.0)
    assert j2.get_mimic_offset() == pytest.approx(0.1)


def test_mimic_cycle_raises(tmp_path):
    with pytest.raises(ValueError, match="mimic cycle"):
        _parse(
            tmp_path,
            _revolute("j1", "base", "l1")
            + _revolute("j2", "l1", "l2", mimic=("j3", 1.0, 0.0))
            + _revolute("j3", "l2", "l3", mimic=("j2", 1.0, 0.0)),
            3,
        )


def test_self_mimic_raises(tmp_path):
    with pytest.raises(ValueError, match="mimic cycle"):
        _parse(
            tmp_path,
            _revolute("j1", "base", "l1")
            + _revolute("j2", "l1", "l2", mimic=("j2", 1.0, 0.0)),
            2,
        )


def test_chain_onto_mimic_of_fixed_degenerates_to_constant(tmp_path):
    fixed = (
        '<joint name="jf" type="fixed"><parent link="base"/>'
        '<child link="l1"/><origin xyz="0 0 0.1" rpy="0 0 0"/></joint>'
    )
    robot = _parse(
        tmp_path,
        fixed
        + _revolute("j2", "l1", "l2", mimic=("jf", 2.0, 0.3))
        + _revolute("j3", "l2", "l3", mimic=("j2", 4.0, 0.1)),
        3,
    )
    j3 = robot.get_joint_by_name("j3")
    assert j3.mimic_target_id == -1
    # constant = 4.0*0.3 + 0.1 folded into the offset, like the one-hop case
    assert j3.get_mimic_offset() == pytest.approx(1.3)
