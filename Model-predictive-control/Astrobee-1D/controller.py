import numpy as np
from control.matlab import place


class Controller(object):

    def __init__(self):
        """
        Nonlinear Controller class. Implements the controller:
        u_t = - L @ x_t + lr @ r
        """

        # Hint:
        self.p1 = 0.0
        self.p2 = 0.0
        self.p3 = 0.0
        self.p4 = 0.0

        self.poles = [self.p1, self.p2]
        self.poles2 = [self.p1, self.p2, self.p3, self.p4]
        self.L = np.zeros((1, 2))
        self.L2 = np.zeros((2, 4))
        self.i_term = 0.0
        self.Ki = 0.0

        self.dt = None
        self.use_integral = False

        #print(self)                             # You can comment this line

    def __str__(self):
        return """
            Controller class. Implements the controller:
            u_t = - L @ x_t - Ki * i
        """

    def set_system(self, A, B, C, D):
        """
        Set system matrices.

        :param A: state space A matrix
        :type A: np.ndarray
        :param B: state space B matrix
        :type B: np.ndarray
        :param C: state space C matrix
        :type C: np.ndarray
        :param D: state space D matrix
        :type D: np.ndarray
        """
        self.A = A
        self.B = B
        self.C = C
        self.D = D

    def get_closed_loop_gain(self, p=None):
        """
        Get the closed loop gain for the specified poles.

        :param p: pole list, defaults to self.p
        :type p: [type], optional
        :return: [description]
        :rtype: [type]
        """
        if p is None:
            p = self.poles

        A = self.A.tolist()
        B = self.B.tolist()

        L = place(A, B, p)
        self.L[0, 0] = L[0, 0]
        self.L[0, 1] = L[0, 1]

    

        return self.L

    def get_closed_loop_gain_2axis(self, p=None):
        """
        Get the closed loop gain for the specified poles.

        :param p: pole list, defaults to self.p
        :type p: [type], optional
        :return: [description]
        :rtype: [type]
        """
        if p is None:
            p = self.poles2

        A = self.A.tolist()
        B = self.B.tolist()
        print("A:",A,"B:",B)

        L2 = place(A, B, p)
        print("L2:",self.L)
        self.L2[0, 0] = L2[0, 0]
        self.L2[0, 1] = L2[0, 1]
        self.L2[0, 2] = L2[0, 2]
        self.L2[0, 3] = L2[0, 3]
        self.L2[1, 0] = L2[1, 0]
        self.L2[1, 1] = L2[1, 1]
        self.L2[1, 2] = L2[1, 2]
        self.L2[1, 3] = L2[1, 3]

        return self.L2

    def set_poles(self, p, p2=None, p3=None, p4=None):
        """
        Set closed loop poles. If 'p' is a list of poles, then the remaining
        inputs are ignored. Otherwise, [p,p2,p3,p4] are set as poles.

        :param p: pole 1 or pole list
        :type p: list or scalar
        :param p2: pole 2, defaults to None
        :type p2: scalar, optional
        """

        if isinstance(p, list):
            self.poles = p
        else:
            self.p1 = p
            self.p2 = p2
            self.poles = [self.p1, self.p2]

    def set_sampling_time(self, dt):
        """
        Set sampling time.

        :param dt: system sampling time
        :type dt: float
        """
        self.dt = dt

    def set_reference(self, ref):
        """
        Set reference for controller.

        :param ref: 2x1 vector
        :type ref: np.ndarray
        """
        self.ref = ref

    def update_integral(self, x):
        """
        Update the integral term for integral action.

        :param x: state
        :type x: np.ndarray 2x1
        """
        if self.dt is None:
            print("[controller] System sampling time not set.\n \
                  Set dt with 'set_sampling_time' method.")

        # TODO: Complete the integral action update law

        #self.i_term = 0.0

        self.i_term = self.i_term + self.dt * (x[0] - self.ref [0,0])
        return self.i_term
        #print("self.i_term",self.i_term)

    def reset_integral(self):
        """
        Reset the integral action of the controller
        """
        self.i_term = 0.0

    def set_integral_gain(self, ki):
        """
        Set integral action gain.

        :param ki: integral gain
        :type ki: float
        """
        self.Ki = ki

    def activate_integral_action(self, dt, ki):
        """
        Helper method to activate integral control law

        :param dt: system sampling time
        :type dt: float
        :param ki: integral gain
        :type ki: float
        """
        self.use_integral = True
        self.set_sampling_time(dt)
        self.set_integral_gain(ki)
        self.reset_integral()

    def control_law(self, x):
        """
        Nonlinear control law.

        :param x: state
        :type x: np.ndarray
        :param ref: cart reference position, defaults to 10
        :type ref: float, optional
        """

        if self.use_integral is True:
            self.update_integral(x)
            u = 0.0
            u = -(self.L @ x)- self.Ki*self.i_term

            return u

        # TODO: Complete the control law

        u = 0.0
        u = - (self.L @ (x - self.ref))

        return u

    
    def control_law_2axis(self, x):
        """
        Nonlinear control law.

        :param x: state
        :type x: np.ndarray
        :param ref: cart reference position, defaults to 10
        :type ref: float, optional
        """

        if self.use_integral is True:
            self.update_integral(x)
            u = [0.0,0.0]
            u = -(self.L2 @ x)- self.Ki*self.i_term

            return u

        # TODO: Complete the control law

        u = [0.0,0.0]
        u = - (self.L2 @ (x - self.ref))
        print("u_real",u)

        return u


