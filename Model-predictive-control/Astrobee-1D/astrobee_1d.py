from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
import control

class Astrobee(object):
    def __init__(self,
                 mass=9.6,
                 mass_ac=11.3,
                 h=0.1,
                 **kwargs):
        """
        Astrobee Robot, NMPC tester class.

        :param mass: mass of the Astrobee
        :type mass: float
        :param h: sampling time of the discrete system, defaults to 0.01
        :type h: float, optional
        """

        # Model
        self.n = 2
        self.m = 1
        self.dt = h

        # Model prperties
        self.mass = mass + mass_ac

        # Linearized model for continuous and discrete time
        self.Ac = None
        self.Bc = None
        self.Ad = None
        self.Bd = None

        self.w = 0.0

    def one_axis_ground_dynamics(self):
        """
        Helper function to populate Ac and Bc with continuous-time
        dynamics of the system.
        """

        # Jacobian of exact discretization
        Ac = np.zeros((2, 2))
        Bc = np.zeros((2, 1))

        # TODO: Complete the entries of the matrices
        #       Ac and Bc. Note that the system mass
        #       is available with self.mass
        
        Ac = np.array([[0,1],[0,0]])
        Bc = np.array([[0],[1/self.mass]])
        
        self.Ac = Ac
        self.Bc = Bc

        return self.Ac, self.Bc

    def two_axis_ground_dynamics(self):
        """
        Helper function to populate Ac and Bc with continuous-time
        dynamics of the system.
        """

        # Jacobian of exact discretization
        Ac = np.zeros((4, 4))
        Bc = np.zeros((4, 1))

        # TODO: Complete the entries of the matrices
        #       Ac and Bc. Note that the system mass
        #       is available with self.mass
        
        Ac = np.array([[0,1,0,0],[0,0,0,0],[0,0,0,1],[0,0,0,0]])
        Bc = np.array([[0,0],[1/self.mass,0],[0,0],[0,1/self.mass]])
        
        self.Ac = Ac
        self.Bc = Bc

        return self.Ac, self.Bc


    def linearized_dynamics(self, x, u):
        """
        Linear dynamics for the Astrobee, continuous time.

        :param x: state
        :type x: np.ndarray
        :param u: control input
        :type u: np.ndarray
        :return: state derivative
        :rtype: np.ndarray
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
        ( Ad_list , Bd_list , Cd_list , Dd_list ) = control.ssdata ( discrete_system  )
        
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
        :type Ad: np.ndarray
        :param Bd: discrete-time control input matrix
        :type Bd: np.ndarray
        """

        self.Ad = Ad
        self.Bd = Bd

    def set_disturbance(self):
        """
        Activate disturbance acting on the system
        """
        self.w = -0.002
    
    def set_disturbance_2axis(self):
        """
        Activate disturbance acting on the system
        """
        self.w = -0.002

    def disable_disturbance(self):
        """
        Disable the disturbance effect.
        """
        self.w = 0.0

    def get_disturbance(self):
        """
        Return the disturbance value

        :return: disturbance value
        :rtype: float
        """
        return self.w

    def linearized_discrete_dynamics(self, x:np.ndarray, u:np.ndarray):
        """
        Method to propagate discrete-time dynamics for Astrobee

        :param x: state
        :type x: np.ndarray
        :param u: control input
        :type u: np.ndarray
        :return: state after dt seconds
        :rtype: np.ndarray
        """

        if self.Ad is None or self.Bd is None:
            print("Set discrete-time dynamics with set_discrete_dynamcs(Ad, Bd) method.")
            return np.zeros(x.shape[0])
        
     
        x_next = self.Ad @ x + self.Bd @ u
        
        # constant disturbance
        if self.w != 0.0:
            Bw = np.zeros((2, 1))
            Bw[1, 0] = 1
            x_next = x_next - Bw * self.w

        return x_next

    
    def linearized_discrete_dynamics_2axis(self, x:np.ndarray, u:np.ndarray):
        """
        Method to propagate discrete-time dynamics for Astrobee

        :param x: state
        :type x: np.ndarray
        :param u: control input
        :type u: np.ndarray
        :return: state after dt seconds
        :rtype: np.ndarray
        """

        if self.Ad is None or self.Bd is None:
            print("Set discrete-time dynamics with set_discrete_dynamcs(Ad, Bd) method.")
            return np.zeros(x.shape[0])
        
     
        x_next = self.Ad @ x + self.Bd @ u
        
        # constant disturbance
        if self.w != 0.0:
            Bw = np.zeros((4, 1))
            Bw[1, 0] = 1
            x_next = x_next - Bw * self.w

        return x_next

    def poles_zeros(self, Ad, Bd, Cd, Dd):
        """
        Plots the system poles and zeros.

        :param Ad: state transition matrix
        :type Ad: np.ndarray
        :param Bd: control matrix
        :type Bd: np.ndarray
        :param Cd: state-observation matrix
        :type Cd: np.ndarray
        :param Dd: control-observation matrix
        :type Dd: np.ndarray
        """
        # dt == 0 -> Continuous time system
        # dt != 0 -> Discrete time system
        sys = control.StateSpace(Ad, Bd, Cd, Dd, dt=self.dt)
        control.pzmap(sys)
        plt.show()
        return
