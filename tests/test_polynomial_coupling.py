"""Polynomial couplings (2026-11-03, closed loops by constraint embedding).

``<mimic joint="l" polycoef="c0 c1 c2 c3 c4" q0="..."/>`` is GRiD's reading of
the MuJoCo joint equality: q = sum_i c_i (q_l - q0)^i. The follower owns no
coordinate (like a mimic) but its fold multiplier dq/dq_l varies with q_l, so
the constant accessors raise and consumers read ``coupling_alpha`` and its
derivatives. A degree-one polynomial is stored as the linear kind. Chains
through a polynomial raise. ``reduce_couplings=False`` keeps every follower
independent and records the couplings for the full-tree projection.
"""
import contextlib
import io

import numpy as np
import pytest

from URDFParser import URDFParser
from URDFParser.URDFParser import MimicResolutionError


_INERTIAL = (
    '<inertial><mass value="1.0"/>'
    '<inertia ixx="1" ixy="0" ixz="0" iyy="1" iyz="0" izz="1"/></inertial>'
)


def _revolute(name, parent, child, mimic_xml=""):
    return (
        f'<joint name="{name}" type="revolute"><parent link="{parent}"/>'
        f'<child link="{child}"/><axis xyz="0 0 1"/>'
        '<origin xyz="0 0 0.1" rpy="0 0 0"/>'
        '<limit lower="-3" upper="3" effort="10" velocity="10"/>'
        f"{mimic_xml}</joint>"
    )


def _parse(tmp_path, joints_xml, n_links, name="poly", **kw):
    links = '<link name="base"/>' + "".join(
        f'<link name="l{k}">{_INERTIAL}</link>' for k in range(1, n_links + 1)
    )
    path = tmp_path / f"{name}.urdf"
    path.write_text(f'<robot name="{name}">{links}{joints_xml}</robot>')
    with contextlib.redirect_stdout(io.StringIO()):
        return URDFParser().parse(str(path), **kw)


_POLY = '<mimic joint="j1" polycoef="0.1 0.5 0.2 -0.1 0.05" q0="0.3"/>'




def test_polynomial_follower_parses(tmp_path):
    robot = _parse(tmp_path, _revolute("j1", "base", "l1") + _revolute("j2", "l1", "l2", _POLY), 2)
    assert robot.get_num_vel() == 1 and robot.get_num_pos() == 1
    j2 = robot.get_joint_by_name("j2")
    assert j2.is_mimic_joint() and not j2.is_linear_coupling()
    assert j2.get_coupling_kind() == "polynomial"
    assert j2.get_mimic_target_id() == robot.get_joint_by_name("j1").get_id()
    assert robot.get_joint_index_v(j2.get_id()) == robot.get_joint_index_v(j2.get_mimic_target_id())
    with pytest.raises(TypeError):
        j2.get_mimic_multiplier()
    with pytest.raises(TypeError):
        j2.get_mimic_offset()
    c = [0.1, 0.5, 0.2, -0.1, 0.05]
    for ql in (-1.0, 0.3, 0.7, 2.0):
        d = ql - 0.3
        value = sum(ci * d ** i for i, ci in enumerate(c))
        alpha = sum(i * ci * d ** (i - 1) for i, ci in enumerate(c) if i >= 1)
        alpha_d = sum(i * (i - 1) * ci * d ** (i - 2) for i, ci in enumerate(c) if i >= 2)
        alpha_dd = sum(i * (i - 1) * (i - 2) * ci * d ** (i - 3) for i, ci in enumerate(c) if i >= 3)
        assert abs(j2.coupling_value(ql) - value) < 1e-14
        assert abs(j2.coupling_alpha(ql) - alpha) < 1e-14
        assert abs(j2.coupling_alpha_d(ql) - alpha_d) < 1e-14
        assert abs(j2.coupling_alpha_dd(ql) - alpha_dd) < 1e-14
        # q_for_joint feeds the follower's transform the polynomial value
        assert abs(robot.q_for_joint(j2.get_id(), np.array([ql])) - value) < 1e-14


def test_degree_one_polycoef_is_the_linear_kind(tmp_path):
    xml = '<mimic joint="j1" polycoef="0.2 -1.5" q0="0.4"/>'
    robot = _parse(tmp_path, _revolute("j1", "base", "l1") + _revolute("j2", "l1", "l2", xml), 2)
    j2 = robot.get_joint_by_name("j2")
    assert j2.is_linear_coupling() and j2.get_coupling_kind() == "linear"
    assert abs(j2.get_mimic_multiplier() + 1.5) < 1e-14
    assert abs(j2.get_mimic_offset() - (0.2 + 1.5 * 0.4)) < 1e-14


def test_polycoef_excludes_multiplier(tmp_path):
    xml = '<mimic joint="j1" multiplier="2" polycoef="0 1 1"/>'
    with pytest.raises(Exception):
        _parse(tmp_path, _revolute("j1", "base", "l1") + _revolute("j2", "l1", "l2", xml), 2)


def test_chain_through_polynomial_raises(tmp_path):
    chain = (_revolute("j1", "base", "l1") + _revolute("j2", "l1", "l2", _POLY)
             + _revolute("j3", "l2", "l3", '<mimic joint="j2" multiplier="2"/>'))
    with pytest.raises(MimicResolutionError):
        _parse(tmp_path, chain, 3)
    chain = (_revolute("j1", "base", "l1") + _revolute("j2", "l1", "l2", '<mimic joint="j1" multiplier="2"/>')
             + _revolute("j3", "l2", "l3", '<mimic joint="j2" polycoef="0 1 1"/>'))
    with pytest.raises(MimicResolutionError):
        _parse(tmp_path, chain, 3)


def test_full_tree_parse_releases_couplings(tmp_path):
    joints = (_revolute("j1", "base", "l1") + _revolute("j2", "l1", "l2", _POLY)
              + _revolute("j3", "l2", "l3", '<mimic joint="j1" multiplier="2" offset="0.1"/>'))
    reduced = _parse(tmp_path, joints, 3)
    full = _parse(tmp_path, joints, 3, reduce_couplings=False)
    assert reduced.get_num_vel() == 1
    assert full.get_num_vel() == 3 and full.get_num_pos() == 3
    assert not any(j.is_mimic_joint() for j in full.get_joints_ordered_by_id())
    assert reduced.couplings == []
    kinds = {r["follower"]: r for r in full.couplings}
    assert set(kinds) == {"j2", "j3"}
    assert kinds["j2"]["kind"] == "polynomial" and kinds["j2"]["polycoef"] == [0.1, 0.5, 0.2, -0.1, 0.05]
    assert kinds["j2"]["q0"] == 0.3 and kinds["j2"]["leader"] == "j1"
    assert kinds["j3"]["kind"] == "linear" and kinds["j3"]["multiplier"] == 2.0 and kinds["j3"]["offset"] == 0.1
    assert kinds["j3"]["leader_id"] == full.get_joint_by_name("j1").get_id()
    # the full tree's follower joints own their own coordinates
    assert full.get_joint_index_v(full.get_joint_by_name("j2").get_id()) != full.get_joint_index_v(full.get_joint_by_name("j1").get_id())
