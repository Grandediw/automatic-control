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




class FiniteOptimization(object):

    def __init__(self, model, 
                       dynamics,
                       total_time=10.0, 
                       rendezvous_time=5.0,
                       R=None,
                       u_lim=None,
                       ref_type="2d", 
                       solver_opts=None):
        """
        Finize optimization solver for minimum energy transfer.

        :param model: system class
        :type model: Python class
        :param dynamics: system dynamics function
        :type dynamics: np.ndarray, ca.DM, ca.MX
        :param total_time: total optimization time, defaults to 10
        :type total_time: float, optional
        :param rendezvous_time: time to rendezvous, defaults to 5
        :type rendezvous_time: float, optional
        :param R: weight matrix for the cost function, defaults to None
        :type R: np.ndarray, optional
        :param ref_type: reference type (2d or 3d - with Z), defaults to '2d'
        :type ref_type: string, optional
        :param solver_opts: optional solver parameters, defaults to None
        :type solver_opts: dictionary, optional
        :param u_lim: input constraints, defaults to None
        :type u_lim: np.ndarray, optional
        """

        build_solver_time = -time.time()
        self.dt           = model.dt
        self.model        = model
        self.Nx, self.Nu  = model.n, model.m
        self.Nt           = int(total_time / self.dt)
        self.Ntr          = int(rendezvous_time / self.dt)
        self.dynamics     = dynamics
        

        if ref_type == "2d":
            self.Nr = 3

            # Tolerances
            self.pos_tol = 0.001 * np.ones((2, 1))
            self.att_tol = 0.001
        else:
            self.Nr = 4

            # Tolerances
            self.pos_tol = 0.001 * np.ones((3, 1))
            self.att_tol = 0.001

        if u_lim is not None:
            u_lb = -u_lim
            u_ub = u_lim

   
        # Cost function weights
        if R is None:
            R = np.eye(self.Nu) * 0.01
        self.R = R
        
       
        
        self.parameters = {}
        
        # Parameters of the optimization problem
        x0 : cp.Parameter           = cp.Parameter((self.Nx,1))
        u0 : cp.Parameter           = cp.Parameter((self.Nu,1))
        xr : dict[int:cp.Parameter] = {step:cp.Parameter((self.Nr,1)) for step in range( self.Ntr,  self.Ntr+(self.Nt - self.Ntr) )}
    
        self.parameters  = {"reference_trajectory": xr, "initial_state": x0, "initial_input": u0}
        
        
        # create a dictionary to store the variables over time steps
        self.x_vars_dict = {step:cp.Variable((self.Nx,1)) for step in range(self.Nt + 1)}
        self.u_vars_dict = {step:cp.Variable((self.Nu,1)) for step in range(self.Nt)}
        
         
        self.num_var = len(self.x_vars_dict) + len(self.u_vars_dict)


        # Set initial values
        obj = 0
        con_eq = []
        con_ineq = []
        
        
        # This section sets the constraints for the optimization problem
        # The following constraints are set:
        
        # 1) Initial state constraint
        # 2) Dynamics constraint
        # 3) State constraints for rendezvous
        # 4) Input constraints
        
        
        # 1) Initial state constraint
        con_eq.append( (self.x_vars_dict[0] - x0) == 0)


        for t in range(self.Nt):

            # Get variables
            x_t = self.x_vars_dict[t]
            u_t = self.u_vars_dict[t]

            # Dynamics constraint
            x_t_next = self.dynamics(x_t, u_t)
            con_eq.append((x_t_next - self.x_vars_dict[t + 1]) == 0) # 2) Dynamics constraint

            # 3) State constraints for rendezvous
            if t > self.Ntr:
                # Make sure that we target the reference then
                if ref_type == "2d":
                    x_ref = xr[t]

                    # Rendezvous tolerance
                    # 
                    # TODO: use 'x_ref', 'x_t', 'self.pos_tol' and 'self.att_tol'
                    #       to define the maximum error tolerance for the position
                    #       in X, Y and angle theta, by adjusting 'con_ineq',
                    #       - take inspiration from
                    #       the example below for 3D
                    con_ineq.append((x_ref[0:2] - x_t[0:2]) <= self.pos_tol)
                    con_ineq.append((x_ref[0:2] - x_t[0:2]) >= -self.pos_tol)
                
                    con_ineq.append((x_ref[2] - x_t[4]) <=  self.att_tol)
                    con_ineq.append((x_ref[2] - x_t[4]) >= -self.att_tol)
        
                else:
                    x_ref = xr[t]

                    con_ineq.append((x_ref[0:3] - x_t[0:3]) <= self.pos_tol)
                    con_ineq.append((x_ref[0:3] - x_t[0:3]) >= -self.pos_tol)
                
                    con_ineq.append((x_ref[3] - x_t[6]) <=  self.att_tol)
                    con_ineq.append((x_ref[3] - x_t[6]) >= -self.att_tol)
                    

            # 4) Input constraints
            if u_lim is not None:
                con_ineq.append(u_t <= u_ub)
                con_ineq.append(u_t >= u_lb)


            # Objective Function / Cost Function
            obj += cp.quad_form(u_t, R) # u.T R u # quadratic form
            

        num_eq_con   = len(con_eq )
        num_ineq_con = len(con_ineq)
        obj          = cp.Minimize(obj)
        self.solver  = cp.Problem(obj, con_eq + con_ineq)
        

        build_solver_time += time.time()
        print('----------------------------------------')
        print('# Time to build solver: %f sec' % build_solver_time)
        print('# Number of variables: %d' % self.num_var)
        print('# Number of equality constraints: %d' % num_eq_con)
        print('# Number of inequality constraints: %d' % num_ineq_con)
        print('----------------------------------------')
        pass


    def solve_problem(self, x0, xr):
        """
        Solve the optimization problem.

        :param x0: starting state
        :type x0: np.ndarray
        :param xr: target set of states
        :type xr: dict[int,np.ndarray]
        :return: optimal states and control inputs
        :rtype: np.ndarray
        """

        # Initial state
        u0 = np.zeros((self.Nu,1))
        
        # Set initial values for the constraints
        self.parameters["initial_state"].value  = x0
        self.parameters["initial_input"].value  = u0
        for step in self.parameters["reference_trajectory"].keys():
            self.parameters["reference_trajectory"][step].value = xr[step]
       
        

        print('\nSolving a total of %d time-steps' % self.Nt)
        solve_time = -time.time()

        
        # Solve NLP
        sol    = self.solver.solve(verbose = False,solver=cp.MOSEK)
        solve_time += time.time()
        optimal_input_trajectory = [u.value for u in self.u_vars_dict.values()]
        optimal_state_trajectory = [x.value for x in self.x_vars_dict.values()]
        print('Solver took %f seconds to obtain a solution.' % (solve_time))

        return optimal_state_trajectory, optimal_input_trajectory
