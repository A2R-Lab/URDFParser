from .Link import Link
from .Joint import Joint
from .SpatialAlgebra import Quaternion_Tools

class Robot:
    # initialization
    def __init__(self, name, floating_base = False, using_quaternion = True):
        self.name = name
        self.floating_base = floating_base
        self.links = []
        self.joints = []
        self.using_quaternion = using_quaternion

    def next_none(self, iterable):
        try:
            return next(iterable)
        except:
            return None
        
    def get_joint_index_q(self, joint_id):
        if self.floating_base:
            if joint_id == 0:
                return [0,1,2,3,4,5,6] if self.using_quaternion else [0,1,2,3,4,5]
            else:
                return joint_id + (6 if self.using_quaternion else 5)
        else:
            return joint_id

    def get_joint_index_v(self, joint_id):
        if self.floating_base:
            if joint_id == 0:
                return [0,1,2,3,4,5]
            else:
                return joint_id + 5
        else:
            return joint_id

    def get_joint_index_f(self, joint_id):
        if self.floating_base:
            if joint_id == 0:
                return [0, 1, 2, 3, 4, 5]
            else:
                return joint_id + 5
        else:
            return joint_id

    #################
    #    Setters    #
    #################

    def add_joint(self, joint):
        self.joints.append(joint)

    def add_link(self, link):
        self.links.append(link)

    def remove_joint(self, joint):
        self.joints.remove(joint)

    def remove_link(self, link):
        self.links.remove(link)

    #########################
    #    Generic Getters    #
    #########################

    def get_num_pos(self):
        """
        Returns the robot's total number of position degrees of freedom. 
        This corresponds to the size of the position(q) array.

        Output:
        - (int) - total position degrees of freedom
        """
        return self.get_num_vel() + (1 if (self.floating_base and self.using_quaternion) else 0)

    def get_num_vel(self):
        """
        Returns the robot's total number of velocity degrees of freedom.
        This corresponds to the size of the velocity(qd) array.

        Output:
        - (int) - total velocity degrees of freedom
        """
        return sum([joint.get_num_dof() for joint in self.joints])
    
    def get_num_bodies(self):
        return self.get_num_links_effective()

    def get_num_cntrl(self):
        return self.get_num_joints()

    def get_name(self):
        return self.name

    def is_serial_chain(self):
        return all([jid - self.get_parent_id(jid) == 1 for jid in range(self.get_num_joints())])

    def get_parent_id(self, lid):
        return self.get_link_by_id(lid).get_parent_id()

    def get_parent_ids(self, lids):
        return [self.get_parent_id(lid) for lid in lids]

    def get_unique_parent_ids(self, lids):
        return list(set(self.get_parent_ids(lids)))

    def get_parent_id_array(self):
        return [tpl[1] for tpl in sorted([(link.get_id(),link.get_parent_id()) for link in self.links], key=lambda tpl: tpl[0])[1:]]

    def has_repeated_parents(self, jids):
        return len(self.get_parent_ids(jids)) != len(self.get_unique_parent_ids(jids))

    def get_subtree_by_id(self, lid):
        return sorted(self.get_link_by_id(lid).get_subtree())

    def get_total_subtree_count(self):
        return sum([len(self.get_subtree_by_id(lid)) for lid in range(self.get_num_joints())])

    def get_ancestors_by_id(self, jid):
        ancestors = []
        curr_id = jid
        while True:
            curr_id = self.get_parent_id(curr_id)
            if curr_id == -1:
                break
            else:
                ancestors.append(curr_id)
        return ancestors

    def get_total_ancestor_count(self):
        return sum([len(self.get_ancestors_by_id(jid)) for jid in range(self.get_num_joints())])

    def get_is_ancestor_of(self, jid, jid_of):
        return jid in self.get_ancestors_by_id(jid_of)

    def get_is_in_subtree_of(self, jid, jid_of):
        return jid in self.get_subtree_by_id(jid_of)

    def get_max_bfs_level(self):
        return sorted(self.joints, key=lambda joint: joint.bfs_level, reverse = True)[0].bfs_level

    def get_ids_by_bfs_level(self, level):
        return [joint.jid for joint in self.get_joints_by_bfs_level(level)]

    def get_bfs_level_by_id(self, jid):
        return(self.get_joint_by_id(jid).get_bfs_level())

    def get_max_bfs_width(self):
        return max([len(self.get_ids_by_bfs_level(level)) for level in range(self.get_max_bfs_level() + 1)])

    def get_is_leaf_node(self, jid):
        return len(self.get_subtree_by_id(jid)) == 1

    def get_leaf_nodes(self):
        return list(filter(lambda jid: self.get_is_leaf_node(jid), range(self.get_num_joints())))

    def get_total_leaf_nodes(self):
        return len(self.get_leaf_nodes())

    ###############
    #    Joint    #
    ###############

    def get_num_joints(self):
        return len(self.joints)

    def get_joint_by_id(self, jid):
        return self.next_none(filter(lambda fjoint: fjoint.jid == jid, self.joints))

    def get_joint_by_name(self, name):
        return self.next_none(filter(lambda fjoint: fjoint.name == name, self.joints))

    def get_joints_by_bfs_level(self, level):
        return list(filter(lambda fjoint: fjoint.bfs_level == level, self.joints))

    def get_joints_ordered_by_id(self, reverse = False):
        return sorted(self.joints, key=lambda item: item.jid, reverse = reverse)

    def get_joints_ordered_by_name(self, reverse = False):
        return sorted(self.joints, key=lambda item: item.name, reverse = reverse)

    def get_joints_dict_by_id(self):
        return {joint.jid:joint for joint in self.joints}

    def get_joints_dict_by_name(self):
        return {joint.name:joint for joint in self.joints}

    def get_joints_by_parent_name(self, parent_name):
        return list(filter(lambda fjoint: fjoint.parent == parent_name, self.joints))

    def get_joints_by_child_name(self, child_name):
        return list(filter(lambda fjoint: fjoint.child == child_name, self.joints))

    def get_joint_by_parent_child_name(self, parent_name, child_name):
        return self.next_none(filter(lambda fjoint: fjoint.parent == parent_name and fjoint.child == child_name, self.joints))

    def get_damping_by_id(self, jid):
        return self.get_joint_by_id(jid).get_damping()

    ##############
    #    Link    #
    ##############

    def get_num_links(self):
        return len(self.links)

    def get_num_links_effective(self):
        # subtracting base link from total # of links
        return get_num_links() - 1

    def get_link_by_id(self, lid):
        return self.next_none(filter(lambda flink: flink.lid == lid, self.links))

    def get_link_by_name(self, name):
        return self.next_none(filter(lambda flink: flink.name == name, self.links))

    def get_links_by_bfs_level(self, level):
        return list(filter(lambda flink: flink.bfs_level == level, self.links))

    def get_links_ordered_by_id(self, reverse = False):
        return sorted(self.links, key=lambda item: item.lid, reverse = reverse)

    def get_links_ordered_by_name(self, reverse = False):
        return sorted(self.links, key=lambda item: item.name, reverse = reverse)

    def get_links_dict_by_id(self):
        return {link.lid:link for link in self.links}

    def get_links_dict_by_name(self):
        return {link.name:link for link in self.links}

    ##############
    #    XMAT    #
    ##############

    def get_Xmat_by_id(self, jid):
        return self.get_joint_by_id(jid).get_transformation_matrix()

    def get_Xmat_by_name(self, name):
        return self.get_joint_by_name(name).get_transformation_matrix()

    def get_Xmats_by_bfs_level(self, level):
        return [joint.get_transformation_matrix() for joint in self.get_joints_by_bfs_level(level)]

    def get_Xmats_ordered_by_id(self, reverse = False):
        return [joint.get_transformation_matrix() for joint in self.get_joints_ordered_by_id(reverse)]

    def get_Xmats_ordered_by_name(self, reverse = False):
        return [joint.get_transformation_matrix() for joint in self.get_joints_ordered_by_name(reverse)]

    def get_Xmats_dict_by_id(self):
        return {joint.jid:joint.get_transformation_matrix() for joint in self.joints}

    def get_Xmats_dict_by_name(self):
        return {joint.name:joint.get_transformation_matrix() for joint in self.joints}

    ###################
    #    XMAT_Func    #
    ###################

    def get_Xmat_Func_by_id(self, jid):
        return self.get_joint_by_id(jid).get_transformation_matrix_function()

    def get_Xmat_Func_by_name(self, name):
        return self.get_joint_by_name(name).get_transformation_matrix_function()

    def get_Xmat_Funcs_by_bfs_level(self, level):
        return [joint.get_transformation_matrix_function() for joint in self.get_joints_by_bfs_level(level)]

    def get_Xmat_Funcs_ordered_by_id(self, reverse = False):
        return [joint.get_transformation_matrix_function() for joint in self.get_joints_ordered_by_id(reverse)]

    def get_Xmat_Funcs_ordered_by_name(self, reverse = False):
        return [joint.get_transformation_matrix_function() for joint in self.get_joints_ordered_by_name(reverse)]

    def get_Xmat_Funcs_dict_by_id(self):
        return {joint.jid:joint.get_transformation_matrix_function() for joint in self.joints}

    def get_Xmat_Funcs_dict_by_name(self):
        return {joint.name:joint.get_transformation_matrix_function() for joint in self.joints}

    ##################
    #    XMAT_hom    #
    ##################

    def get_Xmat_hom_by_id(self, jid):
        return self.get_joint_by_id(jid).get_transformation_matrix_hom()

    def get_Xmat_hom_by_name(self, name):
        return self.get_joint_by_name(name).get_transformation_matrix_hom()

    def get_Xmats_hom_by_bfs_level(self, level):
        return [joint.get_transformation_matrix_hom() for joint in self.get_joints_by_bfs_level(level)]

    def get_Xmats_hom_ordered_by_id(self, reverse = False):
        return [joint.get_transformation_matrix_hom() for joint in self.get_joints_ordered_by_id(reverse)]

    def get_Xmats_hom_ordered_by_name(self, reverse = False):
        return [joint.get_transformation_matrix_hom() for joint in self.get_joints_ordered_by_name(reverse)]

    def get_Xmats_hom_dict_by_id(self):
        return {joint.jid:joint.get_transformation_matrix_hom() for joint in self.joints}

    def get_Xmats_hom_dict_by_name(self):
        return {joint.name:joint.get_transformation_matrix_hom() for joint in self.joints}

    #######################
    #    Xmat_hom_Func    #
    #######################

    def get_Xmat_hom_Func_by_id(self, jid):
        return self.get_joint_by_id(jid).get_transformation_matrix_hom_function()

    def get_Xmat_hom_Func_by_name(self, name):
        return self.get_joint_by_name(name).get_transformation_matrix_hom_function()

    def get_Xmat_hom_Funcs_by_bfs_level(self, level):
        return [joint.get_transformation_matrix_hom_function() for joint in self.get_joints_by_bfs_level(level)]

    def get_Xmat_hom_Funcs_ordered_by_id(self, reverse = False):
        return [joint.get_transformation_matrix_hom_function() for joint in self.get_joints_ordered_by_id(reverse)]

    def get_Xmat_hom_Funcs_ordered_by_name(self, reverse = False):
        return [joint.get_transformation_matrix_hom_function() for joint in self.get_joints_ordered_by_name(reverse)]

    def get_Xmat_hom_Funcs_dict_by_id(self):
        return {joint.jid:joint.get_transformation_matrix_hom_function() for joint in self.joints}

    def get_Xmat_hom_Funcs_dict_by_name(self):
        return {joint.name:joint.get_transformation_matrix_hom_function() for joint in self.joints}

    ##################
    #    dXmat_hom    #
    ##################

    def get_dXmat_hom_by_id(self, jid):
        return self.get_joint_by_id(jid).get_dtransformation_matrix_hom()

    def get_dXmat_hom_by_name(self, name):
        return self.get_joint_by_name(name).get_dtransformation_matrix_hom()

    def get_dXmats_hom_by_bfs_level(self, level):
        return [joint.get_dtransformation_matrix_hom() for joint in self.get_joints_by_bfs_level(level)]

    def get_dXmats_hom_ordered_by_id(self, reverse = False):
        return [joint.get_dtransformation_matrix_hom() for joint in self.get_joints_ordered_by_id(reverse)]

    def get_dXmats_hom_ordered_by_name(self, reverse = False):
        return [joint.get_dtransformation_matrix_hom() for joint in self.get_joints_ordered_by_name(reverse)]

    def get_dXmats_hom_dict_by_id(self):
        return {joint.jid:joint.get_dtransformation_matrix_hom() for joint in self.joints}

    def get_dXmats_hom_dict_by_name(self):
        return {joint.name:joint.get_dtransformation_matrix_hom() for joint in self.joints}

    #######################
    #    dXmat_hom_Func    #
    #######################

    def get_dXmat_hom_Func_by_id(self, jid):
        return self.get_joint_by_id(jid).get_dtransformation_matrix_hom_function()

    def get_dXmat_hom_Func_by_name(self, name):
        return self.get_joint_by_name(name).get_dtransformation_matrix_hom_function()

    def get_dXmat_hom_Funcs_by_bfs_level(self, level):
        return [joint.get_dtransformation_matrix_hom_function() for joint in self.get_joints_by_bfs_level(level)]

    def get_dXmat_hom_Funcs_ordered_by_id(self, reverse = False):
        return [joint.get_dtransformation_matrix_hom_function() for joint in self.get_joints_ordered_by_id(reverse)]

    def get_dXmat_hom_Funcs_ordered_by_name(self, reverse = False):
        return [joint.get_dtransformation_matrix_hom_function() for joint in self.get_joints_ordered_by_name(reverse)]

    def get_dXmat_hom_Funcs_dict_by_id(self):
        return {joint.jid:joint.get_dtransformation_matrix_hom_function() for joint in self.joints}

    def get_dXmat_hom_Funcs_dict_by_name(self):
        return {joint.name:joint.get_dtransformation_matrix_hom_function() for joint in self.joints}

    ##############
    #    IMAT    #
    ##############

    def get_Imat_by_id(self, lid):
        return self.get_link_by_id(lid).get_spatial_inertia()

    def get_Imat_by_name(self, name):
        return self.get_joint_by_name(name).get_spatial_inertia()

    def get_Imats_by_bfs_level(self, level):
        return [link.get_spatial_inertia() for link in self.get_links_by_bfs_level()]

    def get_Imats_ordered_by_id(self, reverse = False):
        return [link.get_spatial_inertia() for link in self.get_links_ordered_by_id(reverse)]

    def get_Imats_ordered_by_name(self, reverse = False):
        return [link.get_spatial_inertia() for link in self.get_links_ordered_by_name(reverse)]

    def get_Imats_dict_by_id(self):
        return {link.lid:link.get_spatial_inertia() for link in self.links}

    def get_Imats_dict_by_name(self):
        return {link.name:link.get_spatial_inertia() for link in self.links}

    ###############
    #      S      #
    ###############

    def get_S_by_id(self, jid):
        return self.get_joint_by_id(jid).get_joint_subspace()

    def get_S_by_name(self, name):
        return self.get_joint_by_name(name).get_joint_subspace()

    def get_S_by_bfs_level(self, level):
        return [joint.get_joint_subspace() for joint in self.get_joints_by_bfs_level(level)]

    def get_Ss_ordered_by_id(self, reverse = False):
        return [joint.get_joint_subspace() for joint in self.get_joints_ordered_by_id(reverse)]

    def get_Ss_ordered_by_name(self, reverse = False):
        return [joint.get_joint_subspace() for joint in self.get_joints_ordered_by_name(reverse)]

    def get_Ss_dict_by_id(self):
        return {joint.jid:joint.get_joint_subspace() for joint in self.joints}

    def get_Ss_dict_by_name(self):
        return {joint.name:joint.get_joint_subspace() for joint in self.joints}

    def are_Ss_identical(self,jids):
        """
        Returns whether all joints have the same subspace matrix.
        If the robot has a floating base, the method will return False.
        This method is for optimizations during code generation.

        Outputs:
        - (bool) - True/False whether all joints have the same subspace matrix
        """
        if self.floating_base: return False
        return all(all(self.get_S_by_id(jid) == self.get_S_by_id(jids[0])) for jid in jids)
    
    def get_S_inds(self, n):
        """
        Returns the index of the 1 in each joint's subspace matrix up to joint n.
        If the robot has a floating base, then the first six entries in the
        S_inds list will be the indices of the 1's in each column of the 
        floating base's subspace matrix.

        Inputs:
        -   (int) n - the total number of joints to get indices for

        Outputs:
        -   [(int)] - the index of the 1 in each of the n subspace matrices
        """
        if self.floating_base:
            fb_S = self.get_S_by_id(0).T.tolist() # break fb S into each column
            S_inds = []
            for dof in fb_S:
                S_inds.append(str(dof.index(1))) # take the indices of the 1's in each column
            for jid in range(1,n):
                S_inds.append(str(self.get_S_by_id(jid).tolist().index(1))) # take the rest
        else:
            S_inds = [str(self.get_S_by_id(jid).tolist().index(1)) for jid in range(n)]
        return S_inds