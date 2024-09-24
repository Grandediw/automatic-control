"""
Model Predictive Control - CasADi interface
Adapted from Helge-André Langåker work on GP-MPC
Customized by Pedro Roque for EL2700 Model Predictive Countrol Course at KTH
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import numpy as np
import cvxpy as cp
import polytope as pc

class MPC(object):

    def __init__(self, model,
                 dynamics,
                 N    : int,
                 Q    : np.ndarray = None,
                 P    : np.ndarray = None,
                 R    : np.ndarray = None,
                 ulb  : np.ndarray = None,
                 uub  : np.ndarray = None,
                 xlb  : np.ndarray = None,
                 xub  : np.ndarray = None,
                 terminal_constraint : pc.Polytope = None,
                 solver_opts         : dict        = None):
        """
        Constructor for the MPC class.

        :param model: System model
        :type model: Astrobee
        :param dynamics: Astrobee dynamics model
        :type dynamics: ca.Function
        :param N: horizion length
        :type N: int
        :param Q: state weight matrix, defaults to None
        :type Q: np.ndarray, optional
        :param P: terminal state weight matrix, defaults to None
        :type P: np.ndarray, optional
        :param R: control weight matrix, defaults to None
        :type R: np.ndarray, optional
        :param ulb: control lower bound vector, defaults to None
        :type ulb: np.ndarray, optional
        :param uub: control upper bound vector, defaults to None
        :type uub: np.ndarray, optional
        :param xlb: state lower bound vector, defaults to None
        :type xlb: np.ndarray, optional
        :param xub: state upper bound vector, defaults to None
        :type xub: [type], optional
        :param terminal_constraint: terminal constriant polytope, defaults to None
        :type terminal_constraint: Polytope, optional
        :param solver_opts: additional solver options, defaults to None.
                            solver_opts['print_time'] = False
                            solver_opts['ipopt.tol'] = 1e-8
        :type solver_opts: dictionary, optional
        """

        build_solver_time = -time.time()
        self.dt = model.dt
        self.Nx, self.Nu = model.n, model.m
        self.Nt = N
        print("Horizon steps: ", N * self.dt)
        self.dynamics = dynamics

        # Initialize variables
        self.set_cost_functions()
        self.x_sp = None

        # Cost function weights
        if P is None:
            P = np.eye(self.Nx) * 10
        if Q is None:
            Q = np.eye(self.Nx)
        if R is None:
            R = np.eye(self.Nu) * 0.01

        self.Q = Q
        self.P = P
        self.R = R

        if xub is None:
            xub = np.full((self.Nx), np.inf)
        if xlb is None:
            xlb = np.full((self.Nx), -np.inf)
        if uub is None:
            uub = np.full((self.Nu), np.inf)
        if ulb is None:
            ulb = np.full((self.Nu), -np.inf)

        # Starting state parameters - add slack here
        x0      = cp.Parameter(self.Nx)
        u0      = cp.Parameter(self.Nu)
        x0_ref  = cp.Parameter(self.Nx)

        # Create optimization variables
        
        
        x_vars = cp.Variable((self.Nx, self.Nt + 1))
        u_vars = cp.Variable((self.Nu, self.Nt))
              
        
        self.num_var = x_vars.size + u_vars.size


        # Set initial values
        obj      =0
        con_eq   = []
        con_ineq = []
        
        #1) initial state constraint 
        con_eq += [ x_vars[:,0] - x0 == 0]

        # Generate MPC Problem
        for t in range(self.Nt):

            # Get variables
            x_t =  x_vars[:,t]
            u_t =  u_vars[:,t]

            # Dynamics constraint
            x_t_next = self.dynamics(x_t, u_t)
            con_eq += [x_t_next - x_vars[:,t + 1] == 0]

            # Input constraints
            if uub is not None:
                non_inf_indx = (uub!=np.inf).flatten() # it is better for cvxpy to no include unbounded constraints at all
                if np.any(non_inf_indx):
                    con_ineq += [u_t[non_inf_indx] <= uub[non_inf_indx].flatten()]
               
            if ulb is not None:
                non_inf_indx = (ulb != - np.inf).flatten() # it is better for cvxpy to no include unbounded constraints at all
                if np.any(non_inf_indx):
                    con_ineq += [u_t[non_inf_indx] >= ulb[non_inf_indx].flatten()]

            # State constraints
            if xub is not None:
                non_inf_indx = (xub != np.inf).flatten() # it is better for cvxpy to no include unbounded constraints at all
                if np.any(non_inf_indx):
                    con_ineq     += [x_t[non_inf_indx] <= xub[non_inf_indx].flatten()]
            if xlb is not None:
                non_inf_indx = (xlb != - np.inf).flatten() # it is better for cvxpy to no include unbounded constraints at all
                if np.any(non_inf_indx):
                    con_ineq     += [x_t[non_inf_indx] >= xlb[non_inf_indx].flatten()]

            # Objective Function / Cost Function
            obj += self.running_cost((x_t - x0_ref), self.Q, u_t, self.R)

        # Terminal Cost
        obj += self.terminal_cost(x_vars[:,self.Nt] - x0_ref, self.P)

        # Terminal contraint
        if terminal_constraint is not None:
            # Should be a polytope
            H_N = terminal_constraint.A
            if H_N.shape[1] != self.Nx:
                print("Terminal constraint with invalid dimensions.")
                exit()

            H_b = terminal_constraint.b
            con_ineq += [H_N @ x_vars[:,self.Nt] <= H_b]
           

      
        self.solver = cp.Problem(cp.Minimize(obj), con_eq + con_ineq)
        self.solver_opts = {'MSK_IPAR_INTPNT_SOLVE_FORM': 'MSK_SOLVE_DUAL'}
        if solver_opts is not None:
            self.solver_opts.update(solver_opts) # other options from the user
      
        build_solver_time += time.time()
        print('\n________________________________________')
        print('# Time to build mpc solver: %f sec' % build_solver_time)
        print('# Number of variables: %d' % self.num_var)
        print('# Number of equality constraints: %d' % len(con_eq))
        print('# Number of inequality constraints: %d' % len(con_ineq))
        print('----------------------------------------')
        
        self.x_vars = x_vars
        self.u_vars = u_vars
        self.x0     = x0
        self.u0     = u0
        self.x0_ref = x0_ref
        

    def set_cost_functions(self):
        """
        Helper method to create CasADi functions for the MPC cost objective.
        """
        # Create functions and function variables for calculating the cost


        # Instantiate function
        self.running_cost = lambda x, Q, u, R: cp.quad_form(x,Q) + cp.quad_form(u,R)
        self.terminal_cost = lambda x, P: cp.quad_form(x,P)

    def solve_mpc(self, x0, u0=None):
        """
        Solve the optimal control problem

        :param x0: starting state
        :type x0: np.ndarray
        :param u0: optimal control guess, defaults to None
        :type u0: np.ndarray, optional
        :return: predicted optimal states and optimal control inputs
        :rtype: ca.DM
        """

        # Initial state
        if u0 is None:
            u0 = np.zeros(self.Nu)
        if self.x_sp is None:
            self.x_sp = np.zeros(self.Nx)

        # Initialize variables
        self.optvar_x0 = np.full((1, self.Nx), x0.T)


        print('\nSolving MPC with %d step horizon' % self.Nt)
        solve_time = -time.time()
        
        self.x0.value = x0.flatten()
        self.u0.value = u0.flatten()
        self.x0_ref.value = self.x_sp.flatten()

        # Solve NLP
        self.solver.solve(solver=cp.MOSEK, warm_start=True, mosek_params={'MSK_IPAR_INTPNT_SOLVE_FORM': 'MSK_SOLVE_DUAL', 'MSK_DPAR_INTPNT_CO_TOL_REL_GAP': 1e-6})
        status = self.solver.status


        solve_time += time.time()
        print('\nMPC took %f seconds to solve.' % (solve_time))
        print('MPC cost: ', self.solver.value)
        if status != cp.OPTIMAL:
            print("The MPC problem was not solved to optimality. Recorded status was:", status)
            exit()

        return self.x_vars.value, self.u_vars.value
    
    
    def mpc_controller(self, x0):
        """
        MPC controller wrapper.
        Gets first control input to apply to the system.

        :param x0: initial state
        :type x0: np.ndarray
        :return: control input
        :rtype: ca.DM
        """

        _, u_pred = self.solve_mpc(x0)

        return u_pred[:,0].reshape(self.Nu, 1)

    def set_reference(self, x_sp):
        """
        Set the controller reference state

        :param x_sp: desired reference state
        :type x_sp: np.ndarray
        """
        self.x_sp = x_sp
