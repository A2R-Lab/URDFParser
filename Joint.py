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
        if not Joint.floating_base:
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

    def set_type(self, jtype, axis = None):
        self.jtype = jtype
        self.origin.build_fixed_transform()
        if self.jtype == 'revolute':
            self.dof = 1
            if axis[2] == 1:
                self.Xmat_sp_free = self.origin.rotation.rot(self.origin.rotation.rz(self.theta))
                self.Xmat_sp_hom_free = self.origin.rotation.rot_hom(self.origin.rotation.rz(self.theta))
                self.S = np.array([0,0,1,0,0,0])
            elif axis[1] == 1:
                self.Xmat_sp_free = self.origin.rotation.rot(self.origin.rotation.ry(self.theta))
                self.Xmat_sp_hom_free = self.origin.rotation.rot_hom(self.origin.rotation.ry(self.theta))
                self.S = np.array([0,1,0,0,0,0])
            elif axis[0] == 1:
                self.Xmat_sp_free = self.origin.rotation.rot(self.origin.rotation.rx(self.theta))
                self.Xmat_sp_hom_free = self.origin.rotation.rot_hom(self.origin.rotation.rx(self.theta))
                self.S = np.array([1,0,0,0,0,0])
        elif self.jtype == 'prismatic':
            self.dof = 1
            if axis[2] == 1:
                self.Xmat_sp_free = self.origin.translation.xlt(self.origin.translation.skew(0,0,self.theta))
                self.Xmat_sp_hom_free = self.origin.translation.gen_tx_hom(0,0,self.theta)
                self.S = np.array([0,0,0,0,0,1])
            elif axis[1] == 1:
                self.Xmat_sp_free = self.origin.translation.xlt(self.origin.translation.skew(0,self.theta,0))
                self.Xmat_sp_hom_free = self.origin.translation.gen_tx_hom(0,self.theta,0)
                self.S = np.array([0,0,0,0,1,0])
            elif axis[0] == 1:
                self.Xmat_sp_free = self.origin.translation.xlt(self.origin.translation.skew(self.theta,0,0))
                self.Xmat_sp_hom_free = self.origin.translation.gen_tx_hom(self.theta,0,0)
                self.S = np.array([0,0,0,1,0,0])
        elif self.jtype == 'fixed':
            self.dof = 0
            self.Xmat_sp_free = sp.eye(6)
            self.Xmat_sp_hom_free = sp.eye(4)
            self.S = np.array([0,0,0,0,0,0])
        elif self.jtype == 'floating':
            self.dof = 6
            if self.using_quaternion:
                self.qt = Quaternion_Tools()
                rot = self.origin.rotation.rot(self.qt.quat_to_rot_sp(self.q1_fb,self.q2_fb,self.q3_fb,self.q4_fb))
            else:
                rot = self.origin.rotation.rot(self.origin.rotation.rx(self.roll_fb) * \
                                               self.origin.rotation.ry(self.pitch_fb) * \
                                               self.origin.rotation.rz(self.yaw_fb))
            trans = self.origin.translation.xlt(self.origin.translation.skew(self.x_fb, self.y_fb, self.z_fb))
            self.Xmat_sp_free = rot*trans
            self.S = np.eye(6)
        else:
            print('Only revolute and fixed joints currently supported (outside of floating base)!')
            exit()
        self.Xmat_sp = self.Xmat_sp_free * self.origin.Xmat_sp_fixed
        # remove numerical noise (e.g., URDF's often specify angles as 3.14 or 3.14159 but that isn't exactly PI)
        self.Xmat_sp = sp.nsimplify(self.Xmat_sp, tolerance=1e-6, rational=True).evalf()
        if not Joint.floating_base:
            # homogenous transform needs to "sum" translation and rotation
            self.Xmat_sp_hom = sp.eye(4)
            self.Xmat_sp_hom[:3,:3] = (self.Xmat_sp_hom_free[:3,:3] * self.origin.Xmat_sp_hom_fixed[:3,:3]).transpose()
            self.Xmat_sp_hom[:3,3] = self.Xmat_sp_hom_free[:3,3] + self.origin.Xmat_sp_hom_fixed[:3,3]
            self.Xmat_sp_hom = sp.nsimplify(self.Xmat_sp_hom, tolerance=1e-6, rational=True).evalf()
            # and derivative
            self.dXmat_sp_hom = sp.diff(self.Xmat_sp_hom,self.theta)
            # and second derivative
            self.d2Xmat_sp_hom = sp.diff(self.dXmat_sp_hom,self.theta)

    def get_transformation_matrix_function(self):
        if self.jtype == "floating":
            if self.using_quaternion:
                return sp.utilities.lambdify([[self.x_fb, self.y_fb, self.z_fb, self.q1_fb, self.q2_fb, self.q3_fb, self.q4_fb]], self.Xmat_sp, 'numpy')
            else:
                return sp.utilities.lambdify([[self.x_fb, self.y_fb, self.z_fb, self.roll_fb, self.pitch_fb, self.yaw_fb]], self.Xmat_sp, 'numpy')
        else:
            return sp.utilities.lambdify(self.theta, self.Xmat_sp, 'numpy')

    def get_transformation_matrix(self):
        return self.Xmat_sp

    def get_transformation_matrix_hom_function(self):
        return sp.utilities.lambdify(self.theta, self.Xmat_sp_hom, 'numpy')

    def get_transformation_matrix_hom(self):
        return self.Xmat_sp_hom

    def get_dtransformation_matrix_hom_function(self):
        return sp.utilities.lambdify(self.theta, self.dXmat_sp_hom, 'numpy')

    def get_d2transformation_matrix_hom_function(self):
        return sp.utilities.lambdify(self.theta, self.d2Xmat_sp_hom, 'numpy')

    def get_dtransformation_matrix_hom(self):
        return self.dXmat_sp_hom

    def get_d2transformation_matrix_hom(self):
        return self.d2Xmat_sp_hom

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
