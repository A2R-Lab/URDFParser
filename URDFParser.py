from bs4 import BeautifulSoup
import numpy as np
import sympy as sp
import copy
import warnings
from .Robot import Robot
from .Link import Link
from .Joint import Joint, Fixed_Joint
from .errors import URDFParseError

class URDFParser:
    def __init__(self):
        pass
    
    def parse(
        self,
        filename,
        floating_base = False,
        using_quaternion = True,
        alpha_tie_breaker = None,
        joint_ordering = "pinocchio_order",
        floating_base_convention = "pinocchio",
        strict_inertial = False,
    ):
        # strict_inertial defaults to False (lenient) to preserve every existing
        # flow: a missing/degenerate <inertial> is silently zeroed as today. When
        # True, a degenerate inertial on a real (non-root, non-dummy) link raises
        # URDFParseError naming the link instead of producing broken dynamics.
        self.strict_inertial = strict_inertial
        Joint.floating_base = floating_base
        try:
            # parse the file
            urdf_file = open(filename, "r")
            self.soup = BeautifulSoup(urdf_file.read(),"xml").find("robot")
            # set up the robot object
            self.robot = Robot(
                self.soup["name"],
                floating_base,
                using_quaternion,
                floating_base_convention=floating_base_convention,
            )
            # collect links
            self.parse_links()
            # collect joints
            self.parse_joints()
            # remove all fixed joints, renumber links and joints, and build parent and subtree lists
            resolved_joint_ordering = self.resolve_joint_ordering(alpha_tie_breaker, joint_ordering)
            self.renumber_linksJoints(using_quaternion, resolved_joint_ordering)
            # report joint ordering to user
            self.print_joint_order()
            # return the robot object
            return copy.deepcopy(self.robot)
        except URDFParseError:
            # Typed, structured parser errors (e.g. an unsupported joint type)
            # are propagated so callers can catch and report them instead of
            # silently receiving None. This replaces the old print()+exit()
            # uncatchable SystemExit failure mode.
            raise
        except Exception:
            # Backwards-compatible catch-all: any other malformed-URDF failure
            # still degrades to None (historical behavior). Valid URDFs are
            # unaffected.
            return None

    def resolve_joint_ordering(self, alpha_tie_breaker, joint_ordering):
        if alpha_tie_breaker is not None:
            return "alphabetical_order" if alpha_tie_breaker else "urdf_order"
        if joint_ordering not in ("urdf_order", "alphabetical_order", "pinocchio_order"):
            raise ValueError(
                "joint_ordering must be one of 'urdf_order', 'alphabetical_order', or 'pinocchio_order'"
            )
        return joint_ordering

    def to_float(self, string_arr):
        if isinstance(string_arr, str):
            string_arr = string_arr.split()
        try:
            return [float(value) for value in string_arr]
        except Exception as exc:
            raise ValueError(f"Could not parse numeric values from {string_arr!r}") from exc

    def parse_links(self):
        lid = 0
        for raw_link in self.soup.find_all('link', recursive=False):
            # construct link object
            curr_link = Link(raw_link["name"],lid)
            lid = lid + 1
            # parse inertial properties
            raw_inertial = raw_link.find("inertial")
            if raw_inertial == None:
                # see docs/open-tasks/notes.md (URDFParser.py:74) re: degenerate inertial detection
                print("Link [" + curr_link.name + "] does not have inertial properties. Assuming this is the fixed world base frame. Else there is an error with your URDF file.")
                curr_link.set_origin_xyz([0, 0, 0])
                curr_link.set_origin_rpy([0, 0, 0])
                curr_link.set_inertia(0, 0, 0, 0, 0, 0, 0)
                # Record the absence so strict validation can flag it if this
                # turns out NOT to be the root (a real moving body with no
                # <inertial> declared). Lenient mode keeps the legacy zeroing.
                curr_link.missing_inertial = True
            else:
                raw_origin = raw_inertial.find("origin")
                if raw_origin is None:
                    curr_link.set_origin_xyz([0.0, 0.0, 0.0])
                    curr_link.set_origin_rpy([0.0, 0.0, 0.0])
                else:
                    origin_xyz = self.to_float(raw_origin["xyz"]) if raw_origin.has_attr("xyz") else [0.0, 0.0, 0.0]
                    origin_rpy = self.to_float(raw_origin["rpy"]) if raw_origin.has_attr("rpy") else [0.0, 0.0, 0.0]
                    curr_link.set_origin_xyz(origin_xyz)
                    curr_link.set_origin_rpy(origin_rpy)
                # get mass and inertia values
                raw_inertia = raw_inertial.find("inertia")
                curr_link.set_inertia(float(raw_inertial.find("mass")["value"]), \
                                      float(raw_inertia["ixx"]), \
                                      float(raw_inertia["ixy"]), \
                                      float(raw_inertia["ixz"]), \
                                      float(raw_inertia["iyy"]), \
                                      float(raw_inertia["iyz"]), \
                                      float(raw_inertia["izz"]))
            # store
            self.robot.add_link(copy.deepcopy(curr_link))

    # Multi-DOF joint types that are DECOMPOSED at parse time into a chain of
    # cardinal 1-DOF sub-joints + zero-mass dummy links (Phase-6 STAGE 3). The
    # decomposition is EXACT (a Featherstone reduction through massless links)
    # and routes every emitted joint through the byte-identical Tier-A cardinal
    # machinery, so NO downstream algorithm / kernel code changes. The native
    # 6x3 `Joint.set_type('planar')` representation is retained only as the
    # numpy-reference oracle; the PARSER emits the decomposed chain. SPHERICAL is
    # NOT here (a manifold, cannot decompose -- stage 4).
    _DECOMPOSED_JOINT_TYPES = ("planar", "translation", "cartesian")

    def parse_joints(self):
        jid = 0
        for raw_joint in self.soup.find_all('joint', recursive=False):
            jtype = raw_joint["type"]
            if jtype in self._DECOMPOSED_JOINT_TYPES:
                # A multi-DOF translation/planar joint expands into a chain of
                # cardinal 1-DOF sub-joints + intermediate dummy links. The
                # chain's user-facing (q, v) order matches the native joint's
                # coordinate order (planar: [px, py, theta]; translation:
                # [x, y, z]) so a user config round-trips through the sub-joints.
                jid = self._decompose_multidof_joint(raw_joint, jtype, jid)
                continue
            # construct joint object
            curr_joint = Joint(raw_joint["name"], jid, \
                               raw_joint.find("parent")["link"], \
                               raw_joint.find("child")["link"])
            jid += 1
            # get origin position and rotation
            raw_origin = raw_joint.find("origin")
            if raw_origin is None:
                curr_joint.set_origin_xyz([0.0, 0.0, 0.0])
                curr_joint.set_origin_rpy([0.0, 0.0, 0.0])
            else:
                joint_xyz = self.to_float(raw_origin["xyz"]) if raw_origin.has_attr("xyz") else [0.0, 0.0, 0.0]
                joint_rpy = self.to_float(raw_origin["rpy"]) if raw_origin.has_attr("rpy") else [0.0, 0.0, 0.0]
                curr_joint.set_origin_xyz(joint_xyz)
                curr_joint.set_origin_rpy(joint_rpy)
            # set joint type and axis of motion for joints if applicable
            raw_axis = raw_joint.find("axis")
            if raw_axis is None:
                curr_joint.set_type(raw_joint["type"])
            else:
                # HELICAL/SCREW (extension): URDF has no native helical type, so
                # the screw pitch is carried on the <axis> as a custom `pitch`
                # attribute (meters / radian, matching pinocchio's convention
                # translation = pitch * angle). It is ignored for non-helical
                # types. e.g. <axis xyz="0 0 1" pitch="0.05"/>.
                pitch = float(raw_axis["pitch"]) if raw_axis.has_attr("pitch") else 0.0
                curr_joint.set_type(raw_joint["type"], self.to_float(raw_axis["xyz"]), pitch=pitch)
            raw_dynamics = raw_joint.find("dynamics")
            if raw_dynamics is None:
                curr_joint.set_damping(0)
                curr_joint.set_friction(0)
            else:
                # Both <dynamics> attributes are optional per the URDF spec; a
                # missing attribute defaults to 0 (a no-op bias term).
                curr_joint.set_damping(
                    float(raw_dynamics["damping"]) if raw_dynamics.has_attr("damping") else 0
                )
                curr_joint.set_friction(
                    float(raw_dynamics["friction"]) if raw_dynamics.has_attr("friction") else 0
                )

            # parse limits (upper/lower)
            raw_limit = raw_joint.find("limit")
            jtype = raw_joint["type"]

            lower = upper = None

            if jtype in ("revolute", "prismatic", "continuous"):
                if raw_limit is not None:
                    if raw_limit.has_attr("lower"): lower = float(raw_limit["lower"])
                    if raw_limit.has_attr("upper"): upper = float(raw_limit["upper"])

                if jtype == "continuous":
                    lower = float("-inf")
                    upper = float("inf")

                if lower is None: lower = float("-inf")
                if upper is None: upper = float("inf")

                curr_joint.joint_limits = [lower, upper]

                # velocity / effort limits (both optional per URDF spec; metadata
                # only — surfaced on the handle, not consumed by any kernel).
                if raw_limit is not None:
                    if raw_limit.has_attr("velocity"):
                        curr_joint.set_velocity_limit(float(raw_limit["velocity"]))
                    if raw_limit.has_attr("effort"):
                        curr_joint.set_effort_limit(float(raw_limit["effort"]))

            # parse <mimic> tag (record by name; resolve to jid post-renumber).
            raw_mimic = raw_joint.find("mimic")
            if raw_mimic is not None:
                if not raw_mimic.has_attr("joint"):
                    raise ValueError(
                        f"Joint '{curr_joint.get_name()}' has a <mimic> tag without "
                        "a `joint` attribute."
                    )
                mimic_target_name = raw_mimic["joint"]
                mimic_multiplier = (
                    float(raw_mimic["multiplier"])
                    if raw_mimic.has_attr("multiplier") else 1.0
                )
                mimic_offset = (
                    float(raw_mimic["offset"])
                    if raw_mimic.has_attr("offset") else 0.0
                )
                curr_joint.set_mimic(mimic_target_name, mimic_multiplier, mimic_offset)

            # store
            self.robot.add_joint(copy.deepcopy(curr_joint))

    def _cardinal_normal_index(self, axis):
        """Return the index (0/1/2) of a cardinal plane-normal <axis> (default
        +Z). Skew planar normals are not decomposed (would need non-cardinal
        in-plane axes); the parser rejects them rather than emit wrong dynamics."""
        a = np.asarray(axis, dtype=np.float64)
        nrm = np.linalg.norm(a)
        if nrm == 0.0:
            raise URDFParseError("Planar joint has a zero-length plane-normal <axis>.")
        a = a / nrm
        for index in range(3):
            if np.isclose(abs(a[index]), 1.0) and all(
                np.isclose(a[k], 0.0) for k in range(3) if k != index
            ):
                return index
        raise URDFParseError(
            "Planar joint with a non-cardinal plane-normal axis "
            f"{list(axis)} is not supported by parse-time decomposition "
            "(only cardinal normals X/Y/Z)."
        )

    def _make_cardinal_unit(self, index):
        v = [0.0, 0.0, 0.0]
        v[index] = 1.0
        return v

    def _decompose_multidof_joint(self, raw_joint, jtype, jid):
        """Decompose a planar / translation joint into a chain of cardinal 1-DOF
        sub-joints joined by zero-mass dummy links (Phase-6 STAGE 3).

        The decomposition is EXACT: stacking the cardinal translations/rotation
        through massless intermediate links reproduces the multi-DOF joint's
        transform and motion subspace, while every emitted sub-joint is a
        cardinal 1-DOF revolute/prismatic -> Tier-A byte-identical machinery
        (no new kernel code). The chain's (q, v) order matches the native
        coordinate order so a user config round-trips.

          planar  -> prismatic(in-plane axis a) -> prismatic(in-plane axis b)
                     -> revolute(plane normal)        [2 dummy links]
          translation/cartesian -> prismatic X -> prismatic Y -> prismatic Z
                                                              [2 dummy links]

        The ORIGINAL joint's <origin> (fixed parent transform) is placed on the
        FIRST sub-joint; the rest are identity (the variable transforms compose
        in the joint frame, so the fixed offset belongs at the head of the
        chain). <dynamics damping/friction> apply to every sub-joint (per-DOF,
        matching pinocchio); both default to 0 -> byte-neutral.
        """
        name = raw_joint["name"]
        parent = raw_joint.find("parent")["link"]
        child = raw_joint.find("child")["link"]
        if raw_joint.find("mimic") is not None:
            raise URDFParseError(
                f"Joint '{name}' ({jtype}) is decomposed at parse time and "
                "cannot also be a <mimic> joint.")
        raw_origin = raw_joint.find("origin")
        if raw_origin is None:
            origin_xyz, origin_rpy = [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]
        else:
            origin_xyz = self.to_float(raw_origin["xyz"]) if raw_origin.has_attr("xyz") else [0.0, 0.0, 0.0]
            origin_rpy = self.to_float(raw_origin["rpy"]) if raw_origin.has_attr("rpy") else [0.0, 0.0, 0.0]
        raw_dynamics = raw_joint.find("dynamics")
        damping = friction = 0.0
        if raw_dynamics is not None:
            damping = float(raw_dynamics["damping"]) if raw_dynamics.has_attr("damping") else 0.0
            friction = float(raw_dynamics["friction"]) if raw_dynamics.has_attr("friction") else 0.0

        # Build the ordered list of (sub-joint type, cardinal axis). The order
        # IS the user-facing (q, v) order.
        if jtype == "planar":
            raw_axis = raw_joint.find("axis")
            normal = self.to_float(raw_axis["xyz"]) if raw_axis is not None else [0.0, 0.0, 1.0]
            n_index = self._cardinal_normal_index(normal)
            in_plane = [k for k in range(3) if k != n_index]
            steps = [
                ("prismatic", self._make_cardinal_unit(in_plane[0])),
                ("prismatic", self._make_cardinal_unit(in_plane[1])),
                # CONTINUOUS (not revolute): a URDF planar joint imposes NO limit
                # on the in-plane rotation -- continuous is handled identically to
                # revolute in the dynamics, but is correctly treated as unbounded
                # (no joint-limit table entry). Using `revolute` here would emit an
                # infinite limit literal.
                ("continuous", self._make_cardinal_unit(n_index)),
            ]
        else:  # translation / cartesian: pure 3-DOF translation
            steps = [
                ("prismatic", self._make_cardinal_unit(0)),
                ("prismatic", self._make_cardinal_unit(1)),
                ("prismatic", self._make_cardinal_unit(2)),
            ]

        nsteps = len(steps)
        # Intermediate dummy links (nsteps - 1): zero-mass, zero-inertia, flagged
        # is_dummy so the world-base-frame heuristic + strict-inertial check skip
        # them (a massless link is NOT the base just because it has no inertia).
        dummy_names = [f"{name}__dummy{k}" for k in range(nsteps - 1)]
        for dname in dummy_names:
            dummy = Link(dname, len(self.robot.links))
            dummy.set_origin_xyz([0.0, 0.0, 0.0])
            dummy.set_origin_rpy([0.0, 0.0, 0.0])
            dummy.set_inertia(0, 0, 0, 0, 0, 0, 0)
            dummy.set_dummy(True)
            self.robot.add_link(copy.deepcopy(dummy))

        chain_links = [parent] + dummy_names + [child]
        for k, (sub_type, axis) in enumerate(steps):
            sub_name = f"{name}__sub{k}" if nsteps > 1 else name
            sub = Joint(sub_name, jid, chain_links[k], chain_links[k + 1])
            jid += 1
            if k == 0:
                sub.set_origin_xyz(origin_xyz)
                sub.set_origin_rpy(origin_rpy)
            else:
                sub.set_origin_xyz([0.0, 0.0, 0.0])
                sub.set_origin_rpy([0.0, 0.0, 0.0])
            sub.set_type(sub_type, axis)
            sub.set_damping(damping)
            sub.set_friction(friction)
            sub.joint_limits = [float("-inf"), float("inf")]
            self.robot.add_joint(copy.deepcopy(sub))
        return jid

    def remove_fixed_joints(self):
        # start at the leaves and work upwards
        for curr_joint in reversed(self.robot.get_joints_ordered_by_id()):
            if curr_joint.jtype == "fixed":
                # updated fixed transforms and parents of grandchild_joints
                # to account for the additional fixed transform
                # X_grandchild = X_granchild * X_child
                for gcjoint in self.robot.get_joints_by_parent_name(curr_joint.child):
                    gcjoint.set_parent(curr_joint.get_parent())
                    gcjoint.set_transformation_matrix(gcjoint.get_transformation_matrix() * curr_joint.get_transformation_matrix())
                    gcjoint.set_transformation_matrix_hom(
                        curr_joint.get_transformation_matrix_hom() * gcjoint.get_transformation_matrix_hom()
                    )
                # combine inertia tensors of child and parent at parent
                # note:  if X is the transform from A to B the I_B = X^T I_A X
                # note2: inertias in the same from add so I_parent_final = I_parent + X^T I_child X
                child_link = self.robot.get_link_by_name(curr_joint.child)
                parent_link = self.robot.get_link_by_name(curr_joint.parent)
                child_I = child_link.get_spatial_inertia()
                curr_Xmat = np.reshape(np.array(curr_joint.get_transformation_matrix()).astype(float),(6,6))
                transformed_Imat = np.matmul(np.matmul(np.transpose(curr_Xmat),child_I),curr_Xmat)
                parent_link.set_spatial_inertia(parent_link.get_spatial_inertia() + transformed_Imat)
                
                # save the fixed joint for later
                joint_hom = sp.matrix2numpy(curr_joint.get_transformation_matrix_hom()).astype(float)
                parent_joints = self.robot.get_joints_by_child_name(parent_link.get_name())
                parent_joint = parent_joints[0] if parent_joints else None
                parent_joint_name = parent_joint.get_name() if parent_joint is not None else -1
                fj = Fixed_Joint(curr_joint.get_id(), curr_joint.get_name(), parent_joint_name, joint_hom)
                self.robot.add_fixed_joint(fj)
                # update any fixed joints that had the current joint as the parent
                for fixed_joint in self.robot.fixed_joints:
                    if fixed_joint.parent_name == curr_joint.get_name():
                        fixed_joint.set_parent(parent_joint_name)
                        new_hom = joint_hom @ fixed_joint.get_transformation_matrix_hom()
                        fixed_joint.set_transformation_matrix_hom(new_hom)

                # delete the bypassed fixed joint and link
                self.robot.remove_joint(curr_joint)
                self.robot.remove_link(child_link)
        
        # renumber fixed joints (arbitarily) starting at the highest joint id to avoid conflicts with existing joint ids
        total_joints = self.robot.get_num_joints()
        for fj_id in range(len(self.robot.fixed_joints)):
            self.robot.fixed_joints[fj_id].set_id(total_joints + fj_id)

    def build_subtree_lists(self):
        subtree_lid_lists = {}
        # initialize all subtrees to include itself
        for lid in self.robot.get_links_dict_by_id().keys():
            subtree_lid_lists[lid] = [lid]
        # start at the leaves and build up!
        for curr_joint in self.robot.get_joints_ordered_by_id(reverse=True):
            parent_lid = self.robot.get_link_by_name(curr_joint.parent).get_id()
            child_lid = self.robot.get_link_by_name(curr_joint.child).get_id()
            # add the child's subtree list to the parent (includes the child)
            if child_lid in subtree_lid_lists.keys():
                subtree_lid_lists[parent_lid] = list(set(subtree_lid_lists[parent_lid]).union(set(subtree_lid_lists[child_lid])))
        # save to the links
        for link in self.robot.links:
            curr_subtree = subtree_lid_lists[link.get_id()]
            link.set_subtree(copy.deepcopy(curr_subtree))

    def sort_child_joints(self, child_joints, joint_ordering):
        if joint_ordering == "urdf_order":
            return child_joints
        if joint_ordering == "alphabetical_order":
            return sorted(child_joints, key=lambda joint: joint.name)
        if joint_ordering == "pinocchio_order":
            return sorted(child_joints, key=lambda joint: (joint.child, joint.name))
        raise ValueError(
            "joint_ordering must be one of 'urdf_order', 'alphabetical_order', or 'pinocchio_order'"
        )

    def dfs_order_update(self, parent_name, joint_ordering = "pinocchio_order", next_lid = 0, next_jid = 0):
        while True:
            child_joints = self.robot.get_joints_by_parent_name(parent_name)
            parent_id = self.robot.get_link_by_name(parent_name).lid
            child_joints = self.sort_child_joints(child_joints, joint_ordering)
            for curr_joint in child_joints:
                # save the new id
                curr_joint.set_id(next_jid)
                # save the next_lid to the child
                child = self.robot.get_link_by_name(curr_joint.child)
                child.set_id(next_lid)
                child.set_parent_id(parent_id)
                # recurse
                next_lid, next_jid = self.dfs_order_update(child.name, joint_ordering, next_lid + 1, next_jid + 1)
            # return to parent
            return next_lid, next_jid

    def bfs_order(self, root_name):
        # initialize
        next_lid = 0
        next_jid = 0
        next_parent_names = [(root_name,-1)]
        self.robot.get_link_by_name(root_name).set_bfs_id(-1)
        self.robot.get_link_by_name(root_name).set_bfs_level(-1)
        # until there are no parent to parse
        while len(next_parent_names) != 0:
            # get the next parent and save its level
            (parent_name, parent_level) = next_parent_names.pop(0)
            next_level = parent_level + 1
            # then until there are no children to parse (of that parent)
            child_joints = self.robot.get_joints_by_parent_name(parent_name)
            while len(child_joints) != 0:
                # update the current link
                curr_joint = child_joints.pop(0)
                curr_joint.set_bfs_id(next_jid)
                curr_joint.set_bfs_level(next_level)
                # append the child to the list of future possible parents
                curr_child_name = curr_joint.get_child()
                next_parent_names.append((curr_child_name,next_level))
                # update the child
                curr_link = self.robot.get_link_by_name(curr_child_name)
                curr_link.set_bfs_id(next_lid)
                curr_link.set_bfs_level(next_level)
                # update the global lid, jid
                next_lid += 1
                next_jid += 1

    def floating_base_adjust(self, root_link_name, using_quaternion = True):
        if not self.robot.floating_base:
            return root_link_name
        if root_link_name == "world":
            root_children = self.robot.get_joints_by_parent_name("world")
            if len(root_children) != 1:
                raise ValueError(
                    "Floating-base conversion for an explicit URDF world root currently expects "
                    f"exactly one child joint from 'world', found {len(root_children)}."
                )
            floating_joint = root_children[0]
            if floating_joint.get_child() == "world":
                raise ValueError(
                    "Floating-base conversion encountered an invalid self-loop from 'world' to 'world'."
                )
            floating_joint.name = "floating_base_joint"
            floating_joint.using_quaternion = using_quaternion
            floating_joint.set_type("floating")
            floating_joint.set_damping(0)
            return "world"
        # add world link
        world = Link("world",-2) # -2 is temporary and unique
        world.set_origin_xyz([0, 0, 0])
        world.set_origin_rpy([0, 0, 0])
        world.set_inertia(0, 0, 0, 0, 0, 0, 0)
        self.robot.add_link(copy.deepcopy(world))
        # add floating joint
        floating_joint = Joint("floating_base_joint", -2, "world", root_link_name, using_quaternion)
        floating_joint.set_origin_xyz([0,0,0])
        floating_joint.set_origin_rpy([0,0,0])
        floating_joint.set_type("floating")
        floating_joint.set_damping(0)
        self.robot.add_joint(copy.deepcopy(floating_joint))
        return "world" # world link is now the root

    def renumber_linksJoints(self, using_quaternion = True, joint_ordering = "pinocchio_order"):
        # find the root link
        link_names = set([link.name for link in self.robot.get_links_ordered_by_id()])
        links_that_are_children = set([joint.get_child() for joint in self.robot.get_joints_ordered_by_id()])
        root_link_name = list(link_names.difference(links_that_are_children))[0]
        # adjust for floating base if applicable
        root_link_name = self.floating_base_adjust(root_link_name, using_quaternion)
        # start renumbering at -1
        self.robot.get_link_by_name(root_link_name).set_id(-1)
        # generate the standard dfs ordering of joints/links
        self.dfs_order_update(root_link_name, joint_ordering)
        # remove all fixed joints where applicable (merge links)
        self.remove_fixed_joints()
        # recompute the dfs ordering of joints/links to account for removed fixed joints
        self.dfs_order_update(root_link_name, joint_ordering)
        # also save a bfs parse ordering and levels of joints/links and build subtree lists
        self.bfs_order(root_link_name)
        self.build_subtree_lists()
        # resolve <mimic> targets now that final jids are stable
        self.resolve_mimic_targets()
        self.robot.refresh_joint_metadata()
        # the renumbered root link is the (intentionally massless) base frame;
        # flag it so strict inertial validation never rejects it.
        root_link = self.robot.get_link_by_name(root_link_name)
        if root_link is not None:
            root_link.set_dummy(True)
        self.validate_inertials(root_link_name)

    def validate_inertials(self, root_link_name):
        """Guard against a degenerate/missing <inertial> on a real moving body
        (zero or non-positive-definite mass/inertia → singular/broken dynamics).
        The root/base frame and dummy links are exempt (intentionally massless).
        strict_inertial=True RAISES; lenient mode (the default) now WARNS instead
        of silently zeroing (the silent path was a footgun — e.g. rizon4's
        zero-inertia links produced broken dynamics with no signal)."""
        strict = getattr(self, "strict_inertial", False)
        bad = [link.get_name() for link in self.robot.get_links_ordered_by_id()
               if link.get_name() != root_link_name and not link.is_dummy_link()
               and (getattr(link, "missing_inertial", False) or link.has_degenerate_inertial())]
        if not bad:
            return
        msg = (f"Link(s) {bad} have a degenerate/missing <inertial> (zero or "
               "non-positive-definite mass/inertia), which yields singular/broken "
               "dynamics. Provide a valid <inertial>, or parse with "
               "strict_inertial=True to reject this as an error.")
        if strict:
            raise URDFParseError(msg)
        warnings.warn("URDFParser: " + msg, stacklevel=2)

    def resolve_mimic_targets(self):
        """Resolve each mimic joint's `mimic_joint_name` to its current jid.

        Must run after the final renumbering pass so `mimic_target_id` is
        stable. Fails loudly (`ValueError`) if a mimic joint references a
        joint that isn't part of the parsed model — silently dropping such
        a relation produces wrong dynamics derivatives downstream (the bug
        this support closes).
        """
        for joint in self.robot.get_joints_ordered_by_id():
            if not getattr(joint, "is_mimic", False):
                continue
            target_name = joint.get_mimic_joint_name()
            target = self.robot.get_joint_by_name(target_name)
            if target is None:
                # Allow mimic of a fixed joint: that's effectively a constant
                # coordinate, which means this mimic joint also degenerates
                # to a constant offset relative to its parent. Resolve by
                # leaving mimic_target_id as -1 and dof=0 (already the case).
                if self.robot.get_fixed_joint_by_name(target_name) is not None:
                    joint.mimic_target_id = -1
                    continue
                raise ValueError(
                    f"Joint '{joint.get_name()}' mimics unknown joint "
                    f"'{target_name}'. Available joints: "
                    f"{[j.get_name() for j in self.robot.get_joints_ordered_by_id()]}"
                )
            if getattr(target, "is_mimic", False):
                raise ValueError(
                    f"Joint '{joint.get_name()}' mimics '{target_name}', which is "
                    "itself a mimic joint. Chained mimics are not supported."
                )
            joint.mimic_target_id = target.get_id()

    def print_joint_order(self):
        print("------------------------------------------")
        print("Assumed Input Joint Configuration Ordering")
        print("------------------------------------------")
        for curr_joint in self.robot.get_joints_ordered_by_id():
            print(curr_joint.get_name())
        print("------------------------------------------")
        print("Total of n = " + str(self.robot.get_num_vel()) + " dof")
        print("Total of n = " + str(self.robot.get_num_joints()) + " joints")
        print("Total of n = " + str(self.robot.get_num_links()) + (" links (including world frame for floating base)" \
                                                                   if self.robot.floating_base else " links"))
        print("------------------------------------------")
        print("Fixed Joints Found (if any):")
        print("------------------------------------------")
        for fj in self.robot.fixed_joints:
            print(fj.get_name() + " (id: " + str(fj.get_id()) + ", parent: " + str(fj.parent_name) + ")")
        print("------------------------------------------")
