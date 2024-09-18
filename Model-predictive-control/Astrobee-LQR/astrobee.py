from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import casadi as ca
import numpy as np
from util import *
from filterpy.kalman import KalmanFilter
import control


class Astrobee(object):
    def __init__(self,
                 mass=9.6,
                 inertia=np.diag([0.1534, 0.1427, 0.1623]),
                 h=0.01,
                 **kwargs):
        """
        Astrobee Robot free-flying dynamics.

        :param mass: mass of the Astrobee
        :type mass: float
        :param inertia: inertia tensor of the Astrobee
        :type inertia: np.diag
        :param h: sampling time of the discrete system, defaults to 0.01
        :type h: float, optional
        """

        # Model
        self.nonlinear_model = self.astrobee_dynamics
        self.n = 12
        self.m = 6
        self.dt = h

        # Model prperties
        self.mass = mass
        self.inertia = inertia

        # Linearized model for continuous and discrete time
        self.Ac = None
        self.Bc = None
        self.Ad = None
        self.Bd = None

        # KF activation variable
        self.kf_activated = False

        # Set single agent properties
        self.set_casadi_options()

    def set_casadi_options(self):
        """
        Helper function to set casadi options.
        """
        self.fun_options = {
            "jit": False,
            "jit_options": {"flags": ["-O2"]}
        }

    def astrobee_dynamics(self, x, u):
        """
        Astrobee nonlinear dynamics.

        :param x: state
        :type x: ca.MX
        :param u: control input
        :type u: ca.MX
        :return: state time derivative
        :rtype: ca.MX
        """

        # State extraction
        p = x[0:3]
        v = x[3:6]
        e = x[6:9]
        w = x[9:]

        # 3D Force
        f = u[0:3]

        # 3D Torque
        tau = u[3:]

        # Model
        pdot = v
        vdot = ca.mtimes(r_mat(e), f) / self.mass
        edot = ca.mtimes(rot_jac_mat(e), w)
        wdot = ca.mtimes(ca.inv(self.inertia), tau + ca.mtimes(skew(w),
                         ca.mtimes(self.inertia, w)))

        dxdt = [pdot, vdot, edot, wdot]

        return ca.vertcat(*dxdt)

    def create_linearized_dynamics(self, x_bar=None, u_bar=None):
        """
        Helper function to populate Ac and Bc with continuous-time
        dynamics of the system.
        """

        # Set CasADi variables
        x = ca.MX.sym('x', self.n)
        u = ca.MX.sym('u', self.m)

        # Jacobian of exact discretization
        Ac = ca.Function('Ac', [x, u], [ca.jacobian(
                         self.astrobee_dynamics(x, u), x)])
        Bc = ca.Function('Bc', [x, u], [ca.jacobian(
                         self.astrobee_dynamics(x, u), u)])

        # Linearization points
        if x_bar is None:
            x_bar = np.zeros((12, 1))

        if u_bar is None:
            u_bar = np.zeros((6, 1))

        self.Ac = np.asarray(Ac(x_bar, u_bar))
        self.Bc = np.asarray(Bc(x_bar, u_bar))

        return self.Ac, self.Bc

    def linearized_dynamics(self, x, u):
        """
        Linear dynamics for the Astrobee, continuous time.

        :param x: state
        :type x: np.ndarray, ca.DM, ca.MX
        :param u: control input
        :type u: np.ndarray, ca.DM, ca.MX
        :return: state derivative
        :rtype: np.ndarray, ca.DM, ca.MX
        """

        xdot = self.Ac @ x + self.Bc @ u

        return xdot

    def c2d(self, A, B, C, D):
        """
        Continuous to Discrete-time dynamics
        """
        # create a continuous time system in state space form
        continuous_system = control.ss(A, B, C, D)
        # create a discrete time system in state space form
        discrete_system   = control.c2d(continuous_system, self.dt, method='zoh')
        # extract the discrete time matrices
        ( Ad_list , Bd_list , Cd_list , Dd_list ) = control.ssdata( discrete_system  )
        
        # convret the list to numpy arrays
        Ad = np . array ( Ad_list )
        Bd = np . array ( Bd_list )
        Cd = np . array ( Cd_list )
        Dd = np . array ( Dd_list )
        
        return Ad,Bd,Cd,Dd

    def set_discrete_dynamics(self, Ad, Bd):
        """
        Helper function to populate discrete-time dynamics

        :param Ad: discrete-time transition matrix
        :type Ad: np.ndarray, ca.DM
        :param Bd: discrete-time control input matrix
        :type Bd: np.ndarray, ca.DM
        """

        self.Ad = Ad
        self.Bd = Bd

    def linearized_discrete_dynamics(self, x, u):
        """
        Method to propagate discrete-time dynamics for Astrobee

        :param x: state
        :type x: np.ndarray, ca.DM
        :param u: control input
        :type u: np.ndarray, ca.DM
        :return: state after dt seconds
        :rtype: np.ndarray, ca.DM
        """

        if self.Ad is None or self.Bd is None:
            print("Set discrete-time dynamics with set_discrete_dynamcs(Ad, Bd) method.")
            return np.zeros(x.shape[0])

        if self.kf_activated:
            wp = np.random.uniform(-0.5, 0.5, (3, 1))
            w = np.vstack((wp, np.zeros((9, 1))))
        else:
            w = np.zeros((12, 1))

        x_next = self.Ad @ x + self.Bd @ u + w

        return x_next

    # =============================================== #
    #             Kalman Filter modules               #
    # =============================================== #

    def set_kf_params(self, C, Q, R):
        """
        Set the Kalman Filter variables.

        :param C: observation matrix
        :type C: numpy.array
        :param Q: process noise
        :type Q: numpy.array
        :param R: measurement noise
        :type R: numpy.array
        """
        self.C_KF = C
        self.Q_KF = Q
        self.R_KF = R

    def init_kf(self, x):
        """
        Initialize the Kalman Filter estimator.
        """

        # Initialize filter object
        self.kf_estimator = KalmanFilter(dim_x=6, dim_z=3, dim_u=3)

        # Set filter parameters
        self.kf_estimator.F = self.Ad[0:6, 0:6].reshape(6, 6)
        self.kf_estimator.B = self.Bd[0:6, 0:3].reshape(6, 3)

        self.kf_estimator.H = self.C_KF
        self.kf_estimator.Q = self.Q_KF
        self.kf_estimator.R = self.R_KF

        # Set initial estimation
        self.kf_estimator.x = x

        # TODO: Uncomment the line below for process noise
        self.kf_activated = True
