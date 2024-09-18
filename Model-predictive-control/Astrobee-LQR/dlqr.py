import casadi as ca
import numpy as np
import scipy.linalg
from control.matlab import lqr


class DLQR(object):

    def __init__(self, A, B, C, Q=ca.DM.eye(4), R=ca.DM.ones(1, 1)):
        """
        Discrete-time LQR class.
        """

        # System matrices
        self.A = A
        self.B = B
        self.C = C

        self.Q = Q
        self.R = R

        self.K = None
        self.P = None

        self.r = None

        print(self)                             # You can comment this line

    def __str__(self):
        return """
            Linear-Quadratic Regulator class for discrete-time systems.
            Implements the following controller:
            u_t = - K @ (x - r)                  - method: feedback(x)
            where:
              x   - system state
              r   - system reference, set with set_reference(r) method
              K   - LQR feedback gain matrix
        """

    def set_system(self, A, B, C):
        """
        Set system matrices.

        :param A: state space A matrix
        :type A: casadi.DM
        :param B: state space B matrix
        :type B: casadi.DM
        :param C: state space C matrix
        :type C: casadi.DM
        """
        self.A = A
        self.B = B
        self.C = C

    def get_lqr_gain(self, Q=None, R=None):
        """
        Get LQR feedback gain.

        :return: LQR feedback gain K
        :rtype: casadi.DM
        :return: LQR infinite-horizon weight P
        :rtype: casadi.DM
        """
        A_np = np.asarray(self.A)
        B_np = np.asarray(self.B)

        if Q is None:
            Q_np = np.asarray(self.Q)
        else:
            Q_np = Q

        if R is None:
            R_np = np.asarray(self.R)
        else:
            R_np = R

        P_np = np.matrix(scipy.linalg.solve_discrete_are(A_np,
                         B_np, Q_np, R_np))
        K_np = np.matrix(scipy.linalg.inv(B_np.T @ P_np @ B_np
                         + R_np) @ (B_np.T @ P_np @ A_np))

        self.P = P_np

        self.K = K_np

        return self.K, self.P

    def set_reference(self, r):
        """
        Helper method to populate the desired reference.

        :param r: desired reference (12, 1)
        :type r: np.ndarray, ca.DM
        """

        self.r = r

    def feedback(self, x):
        """
        State feedback LQR.

        :param x: state
        :type x: casadi.DM
        :param K: LQR feedback gain
        :type K: casadi.DM, 1x4
        """

        if self.r is None:
            print("Reference not set!\nSet with set_reference(r) method.")
            return np.zeros((6, 1))

        u = self.K @ (self.r - x)

        return u
