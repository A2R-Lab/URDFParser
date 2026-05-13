# import numpy as np
import numpy as np
import sympy as sp
from .SpatialAlgebra import Origin, Translation, Rotation, Quaternion_Tools

class Joint:
    floating_base = False
    def __init__(self, name, jid, parent, child, using_quaternion = False):
        self.name = name         # name
        self.jid = jid           # temporary ID (replaced by standard DFS parse ordering)
        self.urdf_jid = jid      # URDF ordered ID
        self.bfs_jid = jid       # temporary ID (replaced by BFS parse ordering)
        self.bfs_level = 0       # temporary level (replaced by BFS parse ordering)
        self.origin = Origin()   # Fixed origin location
        self.jtype = None        # type of joint
        self.parent = parent     # parent link name
        self.child = child       # child link name TODO - currently unused
        self.theta = sp.symbols("theta") # Free 1D joint variable
        self.Xmat_sp = None      # Sympy X matrix placeholder
        self.Xmat_sp_free = None # Sympy X_free matrix placeholder
        self.Xmat_sp_hom = None      # Sympy X homogenous 4x4 matrix placeholder
        self.Xmat_sp_hom_free = None # Sympy X_free homogenous 4x4  matrix placeholder
        self.Smat_sp = None      # Sympy S matrix placeholder (usually a vector)
        self.damping = 0         # damping placeholder
        self.dof = 0             # dof placeholder
        # for floating base
        self.using_quaternion = using_quaternion
        self.x_fb = sp.symbols("x_fb")
        self.y_fb = sp.symbols("y_fb")
        self.z_fb = sp.symbols("z_fb")
        self.q1_fb = sp.symbols("q1_fb")
        self.q2_fb = sp.symbols("q2_fb")
        self.q3_fb = sp.symbols("q3_fb")
        self.q4_fb = sp.symbols("q4_fb")
        self.roll_fb = sp.symbols("roll_fb")
        self.pitch_fb = sp.symbols("pitch_fb")
        self.yaw_fb = sp.symbols("yaw_fb")
        self.joint_limits = []
        self.position_symbols = []
        self.local_q_dim = 0
        self.dXmat_sp_hom_blocks = []
        self.d2Xmat_sp_hom_blocks = []

    def set_id(self, id_in):
        self.jid = id_in

    def set_parent(self, parent_name):
        self.parent = parent_name

    def set_child(self, child_name):
        self.child = child_name

    def set_bfs_id(self, id_in):
        self.bfs_id = id_in

    def set_bfs_level(self, level_in):
        self.bfs_level = level_in

    def set_origin_xyz(self, x, y = None, z = None):
        self.origin.set_translation(x,y,z)

    def set_origin_rpy(self, r, p = None, y = None):
        self.origin.set_rotation(r,p,y)

    def set_damping(self, damping):
        self.damping = damping

    def set_transformation_matrix(self, matrix_in):
        self.Xmat_sp = matrix_in

    def set_transformation_matrix_hom(self, matrix_in):
        self.Xmat_sp_hom = sp.nsimplify(matrix_in, tolerance=1e-6, rational=True).evalf()
        self.position_symbols = [self.theta]
        self.local_q_dim = 1
        self._build_homogeneous_transform_derivatives()

    def _build_homogeneous_transform_derivatives(self):
        if self.Xmat_sp_hom is None:
            self.dXmat_sp_hom = None
            self.d2Xmat_sp_hom = None
            self.dXmat_sp_hom_blocks = []
            self.d2Xmat_sp_hom_blocks = []
            return

        if not self.position_symbols:
            self.dXmat_sp_hom = sp.zeros(4, 4)
            self.d2Xmat_sp_hom = sp.zeros(4, 4)
            self.dXmat_sp_hom_blocks = []
            self.d2Xmat_sp_hom_blocks = []
            self.local_q_dim = 0
            return

        self.dXmat_sp_hom_blocks = [
            sp.diff(self.Xmat_sp_hom, symbol) for symbol in self.position_symbols
        ]
        self.d2Xmat_sp_hom_blocks = [
            [
                sp.diff(self.dXmat_sp_hom_blocks[row_ind], self.position_symbols[col_ind])
                for col_ind in range(len(self.position_symbols))
            ]
            for row_ind in range(len(self.position_symbols))
        ]
        self.dXmat_sp_hom = self.dXmat_sp_hom_blocks[0]
        self.d2Xmat_sp_hom = self.d2Xmat_sp_hom_blocks[0][0]

    def _local_q_lambdify_args(self):
        if self.jtype == "floating":
            if self.using_quaternion:
                return [[self.x_fb, self.y_fb, self.z_fb, self.q1_fb, self.q2_fb, self.q3_fb, self.q4_fb]]
            return [[self.x_fb, self.y_fb, self.z_fb, self.roll_fb, self.pitch_fb, self.yaw_fb]]
        return self.theta

    def _axis_scale(self, axis, index):
        value = float(axis[index])
        if np.isclose(abs(value), 1.0):
            return value
        return None

    def set_type(self, jtype, axis = None):
        self.jtype = jtype
        self.origin.build_fixed_transform()
        if self.jtype in ('revolute', 'continuous'):
            self.dof = 1
            self.position_symbols = [self.theta]
            self.local_q_dim = 1
            axis_scale = self._axis_scale(axis, 2)
            if axis_scale is not None:
                self.Xmat_sp_free = self.origin.rotation.rot(self.origin.rotation.rz(axis_scale * self.theta))
                self.Xmat_sp_hom_free = self.origin.rotation.rot_hom(self.origin.rotation.rz(axis_scale * self.theta))
                self.S = np.array([0,0,axis_scale,0,0,0])
            else:
                axis_scale = self._axis_scale(axis, 1)
                if axis_scale is not None:
                    self.Xmat_sp_free = self.origin.rotation.rot(self.origin.rotation.ry(axis_scale * self.theta))
                    self.Xmat_sp_hom_free = self.origin.rotation.rot_hom(self.origin.rotation.ry(axis_scale * self.theta))
                    self.S = np.array([0,axis_scale,0,0,0,0])
                else:
                    axis_scale = self._axis_scale(axis, 0)
                    if axis_scale is not None:
                        self.Xmat_sp_free = self.origin.rotation.rot(self.origin.rotation.rx(axis_scale * self.theta))
                        self.Xmat_sp_hom_free = self.origin.rotation.rot_hom(self.origin.rotation.rx(axis_scale * self.theta))
                        self.S = np.array([axis_scale,0,0,0,0,0])
        elif self.jtype == 'prismatic':
            self.dof = 1
            self.position_symbols = [self.theta]
            self.local_q_dim = 1
            axis_scale = self._axis_scale(axis, 2)
            if axis_scale is not None:
                self.Xmat_sp_free = self.origin.translation.xlt(self.origin.translation.skew(0,0,axis_scale * self.theta))
                self.Xmat_sp_hom_free = self.origin.translation.gen_tx_hom(0,0,axis_scale * self.theta)
                self.S = np.array([0,0,0,0,0,axis_scale])
            else:
                axis_scale = self._axis_scale(axis, 1)
                if axis_scale is not None:
                    self.Xmat_sp_free = self.origin.translation.xlt(self.origin.translation.skew(0,axis_scale * self.theta,0))
                    self.Xmat_sp_hom_free = self.origin.translation.gen_tx_hom(0,axis_scale * self.theta,0)
                    self.S = np.array([0,0,0,0,axis_scale,0])
                else:
                    axis_scale = self._axis_scale(axis, 0)
                    if axis_scale is not None:
                        self.Xmat_sp_free = self.origin.translation.xlt(self.origin.translation.skew(axis_scale * self.theta,0,0))
                        self.Xmat_sp_hom_free = self.origin.translation.gen_tx_hom(axis_scale * self.theta,0,0)
                        self.S = np.array([0,0,0,axis_scale,0,0])
        elif self.jtype == 'fixed':
            self.dof = 0
            self.position_symbols = []
            self.local_q_dim = 0
            self.Xmat_sp_free = sp.eye(6)
            self.Xmat_sp_hom_free = sp.eye(4)
            self.S = np.array([0,0,0,0,0,0])
        elif self.jtype == 'floating':
            self.dof = 6
            if self.using_quaternion:
                self.position_symbols = [
                    self.x_fb,
                    self.y_fb,
                    self.z_fb,
                    self.q1_fb,
                    self.q2_fb,
                    self.q3_fb,
                    self.q4_fb,
                ]
            else:
                self.position_symbols = [
                    self.x_fb,
                    self.y_fb,
                    self.z_fb,
                    self.roll_fb,
                    self.pitch_fb,
                    self.yaw_fb,
                ]
            self.local_q_dim = len(self.position_symbols)
            if self.using_quaternion:
                self.qt = Quaternion_Tools()
                quat_rot = self.qt.quat_to_rot_sp(self.q1_fb,self.q2_fb,self.q3_fb,self.q4_fb)
                rot = self.origin.rotation.rot(quat_rot)
                self.Xmat_sp_hom_free = self.origin.rotation.rot_hom(quat_rot)
            else:
                rpy_rot = self.origin.rotation.rx(self.roll_fb) * \
                          self.origin.rotation.ry(self.pitch_fb) * \
                          self.origin.rotation.rz(self.yaw_fb)
                rot = self.origin.rotation.rot(rpy_rot)
                self.Xmat_sp_hom_free = self.origin.rotation.rot_hom(rpy_rot)
            trans = self.origin.translation.xlt(self.origin.translation.skew(self.x_fb, self.y_fb, self.z_fb))
            self.Xmat_sp_hom_free[:3,3] = sp.Matrix([self.x_fb, self.y_fb, self.z_fb])
            self.Xmat_sp_free = rot*trans
            # User-facing floating-base vectors follow Pinocchio order
            # [vx, vy, vz, wx, wy, wz], while GRiD's internal spatial vectors
            # use [wx, wy, wz, vx, vy, vz].
            self.S = np.array(
                [
                    [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                ],
                dtype=np.float64,
            )
        else:
            print('Only revolute and fixed joints currently supported (outside of floating base)!')
            exit()
        self.Xmat_sp = self.Xmat_sp_free * self.origin.Xmat_sp_fixed
        # remove numerical noise (e.g., URDF's often specify angles as 3.14 or 3.14159 but that isn't exactly PI)
        self.Xmat_sp = sp.nsimplify(self.Xmat_sp, tolerance=1e-6, rational=True).evalf()
        if self.jtype != 'floating':
            # homogenous transform needs to "sum" translation and rotation
            self.Xmat_sp_hom = sp.eye(4)
            self.Xmat_sp_hom[:3,:3] = (self.Xmat_sp_hom_free[:3,:3] * self.origin.Xmat_sp_hom_fixed[:3,:3]).transpose()
            self.Xmat_sp_hom[:3,3] = self.Xmat_sp_hom_free[:3,3] + self.origin.Xmat_sp_hom_fixed[:3,3]
            self.Xmat_sp_hom = sp.nsimplify(self.Xmat_sp_hom, tolerance=1e-6, rational=True).evalf()
            # and derivative
            self._build_homogeneous_transform_derivatives()
        else:
            self.Xmat_sp_hom = self.Xmat_sp_hom_free * self.origin.Xmat_sp_hom_fixed
            self.Xmat_sp_hom = sp.nsimplify(self.Xmat_sp_hom, tolerance=1e-6, rational=True).evalf()
            self._build_homogeneous_transform_derivatives()

    def get_transformation_matrix_function(self):
        if self.jtype == "floating":
            return sp.utilities.lambdify(self._local_q_lambdify_args(), self.Xmat_sp, 'numpy')
        else:
            return sp.utilities.lambdify(self.theta, self.Xmat_sp, 'numpy')

    def get_transformation_matrix(self):
        return self.Xmat_sp

    def get_transformation_matrix_hom_function(self):
        return sp.utilities.lambdify(self._local_q_lambdify_args(), self.Xmat_sp_hom, 'numpy')

    def get_transformation_matrix_hom(self):
        return self.Xmat_sp_hom

    def get_dtransformation_matrix_hom_function(self):
        return sp.utilities.lambdify(self._local_q_lambdify_args(), self.dXmat_sp_hom, 'numpy')

    def get_d2transformation_matrix_hom_function(self):
        return sp.utilities.lambdify(self._local_q_lambdify_args(), self.d2Xmat_sp_hom, 'numpy')

    def get_dtransformation_matrix_hom(self):
        return self.dXmat_sp_hom

    def get_d2transformation_matrix_hom(self):
        return self.d2Xmat_sp_hom

    def get_local_q_dim(self):
        return self.local_q_dim

    def get_dtransformation_matrix_hom_local(self, local_index):
        return self.dXmat_sp_hom_blocks[local_index]

    def get_d2transformation_matrix_hom_local(self, local_index_i, local_index_j):
        return self.d2Xmat_sp_hom_blocks[local_index_i][local_index_j]

    def get_d2transformation_matrix_local(self, local_index_i, local_index_j):
        return sp.diff(
            sp.diff(self.Xmat_sp, self.position_symbols[local_index_i]),
            self.position_symbols[local_index_j],
        )

    def get_dtransformation_matrix_hom_local_function(self, local_index):
        return sp.utilities.lambdify(
            self._local_q_lambdify_args(),
            self.get_dtransformation_matrix_hom_local(local_index),
            'numpy',
        )

    def get_d2transformation_matrix_hom_local_function(self, local_index_i, local_index_j):
        return sp.utilities.lambdify(
            self._local_q_lambdify_args(),
            self.get_d2transformation_matrix_hom_local(local_index_i, local_index_j),
            'numpy',
        )

    def get_d2transformation_matrix_local_function(self, local_index_i, local_index_j):
        return sp.utilities.lambdify(
            self._local_q_lambdify_args(),
            self.get_d2transformation_matrix_local(local_index_i, local_index_j),
            'numpy',
        )

    def get_joint_subspace(self):
        return self.S

    def get_damping(self):
        return self.damping

    def get_name(self):
        return self.name

    def get_id(self):
        return self.jid

    def get_bfs_id(self):
        return self.bfs_id

    def get_bfs_level(self):
        return self.bfs_level

    def get_parent(self):
        return self.parent

    def get_child(self):
        return self.child
    
    def get_num_dof(self):
        return self.dof
    
    def get_joint_limits(self):
        return self.joint_limits

# Need to retain fixed joints for possible kinematic use later
class Fixed_Joint:
    def __init__(self, jid_in, name, parent_name, hom_xfrm):
        self.jid = jid_in                    # original ID
        self.name = name                # name
        self.parent_name = parent_name  # parent joint name
        self.Xmat_hom = hom_xfrm

    def set_id(self, jid_in):
        self.jid = jid_in

    def set_parent(self, parent_in):
        self.parent_name = parent_in

    def set_transformation_matrix_hom(self, hom_xfrm):
        self.Xmat_hom = hom_xfrm

    def get_id(self):
        return self.jid

    def get_name(self):
        return self.name

    def get_parent(self):
        return self.parent_name

    def get_transformation_matrix_hom(self):
        return self.Xmat_hom
