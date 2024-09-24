from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import casadi as ca
import numpy as np
import numpy.matlib as nmp
from util import *
from filterpy.kalman import KalmanFilter


class Astrobee(object):
    def __init__(self,
                 iface='casadi',
                 mass=9.6,
                 inertia=np.diag([0.1534, 0.1427, 0.1623]),
                 h=0.1,
                 **kwargs):
        """
        Astrobee Robot, NMPC tester class.

        :param mass: mass of the Astrobee
        :type mass: float
        :param inertia: inertia tensor of the Astrobee
        :type inertia: np.diag
        :param h: sampling time of the discrete system, defaults to 0.01
        :type h: float, optional
        """

        # Model
        self.solver = iface
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
        Pendulum nonlinear dynamics.

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

    def create_linearized_dynamics(self):
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
        x_bar = np.zeros((self.n, 1))
        u_bar = np.zeros((self.m, 1))

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

    def casadi_c2d(self, A, B, C, D):
        """
        Continuous to Discrete-time dynamics
        """
        # Set CasADi variables
        x = ca.MX.sym('x', A.shape[1])
        u = ca.MX.sym('u', B.shape[1])

        # Create an ordinary differential equation dictionary. Notice that:
        # - the 'x' argument is the state
        # - the 'ode' contains the equation/function we wish to discretize
        # - the 'p' argument contains the parameters that our function/equation
        #   receives. For now, we will only need the control input u
        ode = {'x': x, 'ode': ca.DM(A) @ x + ca.DM(B) @ u, 'p': ca.vertcat(u)}

        # Here we define the options for our CasADi integrator - it will take care of the
        # numerical integration for us: fear integrals no more!
        options = {"abstol": 1e-5, "reltol": 1e-9, "max_num_steps": 100, "tf": self.dt}

        # Create the integrator
        self.Integrator = ca.integrator('integrator', 'cvodes', ode, options)

        # Now we have an integrator CasADi function. We wish now to take the partial
        # derivaties w.r.t. 'x', and 'u', to obtain Ad and Bd, respectively. That's wher
        # we use ca.jacobian passing the integrator we created before - and extracting its
        # value after the integration interval 'xf' (our dt) - and our variable of interest
        Ad = ca.Function('jac_x_Ad', [x, u], [ca.jacobian(
                         self.Integrator(x0=x, p=u)['xf'], x)])
        Bd = ca.Function('jac_u_Bd', [x, u], [ca.jacobian(
                         self.Integrator(x0=x, p=u)['xf'], u)])

        # If you print Ad and Bd, they will be functions that can be evaluated at any point.
        # Now we must extract their value at the linearization point of our chosing!
        x_bar = np.zeros((12, 1))
        u_bar = np.zeros((6, 1))

        return np.asarray(Ad(x_bar, u_bar)), np.asarray(Bd(x_bar, u_bar)), C, D

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

        x_next = self.Ad @ x + self.Bd @ u

        return x_next

    def create_discrete_time_dynamics(self):
        """
        Helper method to create the discrete-time dynamics.

        Abstracts the operations we did on the last assignment.
        """

        A, B = self.create_linearized_dynamics()
        C = np.diag(np.ones(12))
        D = np.zeros((12, 6))
        self.Ad, self.Bd, self.Cd, self.Dd = self.casadi_c2d(A, B, C, D)

        return self.Ad, self.Bd, self.Cd, self.Dd

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
