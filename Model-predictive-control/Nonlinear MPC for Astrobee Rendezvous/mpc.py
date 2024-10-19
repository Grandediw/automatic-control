"""
Model Predictive Control - CasADi interface
Adapted from Helge-Andre Langaker work on GP-MPC
Customized by Pedro Roque for EL2700 Model Predictive Countrol Course at KTH
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import numpy as np
import casadi as ca
import casadi.tools as ctools
import yaml
from util import *


class MPC(object):

    def __init__(self, model, dynamics,
                 param='P1', N=10, trajectory_tracking=False,
                 ulb=None, uub=None, xlb=None, xub=None,
                 terminal_constraint=None, tuning_file=None,
                 solver_opts=None):
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

        # State flags
        self.fw_propagating = False
        self.trajectory_tracking = trajectory_tracking

        build_solver_time = -time.time()
        self.model = model
        self.dt = model.dt
        self.Nx, self.Nu = model.n, model.m
        self.Nt = N
        print("Horizon steps: ", N * self.dt)
        self.dynamics = dynamics

        # Initialize variables
        self.set_cost_functions_quat()
        self.x_sp = None


        self.cpu_times = []
        self.costs = []
        self.horizon_lengths = []

        # Cost function weights
        Q, R, P = self.load_params(param, tuning_file)

        self.Q = ca.MX(Q)
        self.P = ca.MX(P)
        self.R = ca.MX(R)

        if xub is None:
            xub = np.full((self.Nx), np.inf)
        if xlb is None:
            xlb = np.full((self.Nx), -np.inf)
        if uub is None:
            uub = np.full((self.Nu), np.inf)
        if ulb is None:
            ulb = np.full((self.Nu), -np.inf)

        # Starting state parameters - add slack here
        x0 = ca.MX.sym('x0', self.Nx)
        if self.trajectory_tracking:
            # TODO: remove 'raise NotImplementedError' and create the desired symbolic param
            #raise NotImplementedError
            x_ref = ca.MX.sym('x_ref', self.Nx*(self.Nt+1),)
        else:
            x_ref = ca.MX.sym('x_ref', self.Nx,)
        u0 = ca.MX.sym('u0', self.Nu)
        param_s = ca.vertcat(x0, x_ref, u0)

        # Create optimization variables
        opt_var = ctools.struct_symMX([(ctools.entry('u', shape=(self.Nu,), repeat=self.Nt),
                                        ctools.entry('x', shape=(self.Nx,), repeat=self.Nt + 1),
                                        )])
        self.opt_var = opt_var
        self.num_var = opt_var.size

        # Decision variable boundries
        self.optvar_lb = opt_var(-np.inf)
        self.optvar_ub = opt_var(np.inf)

        # Set initial values
        obj = ca.MX(0)
        con_eq = []
        con_ineq = []
        con_ineq_lb = []
        con_ineq_ub = []
        con_eq.append(opt_var['x', 0] - x0)

        # Generate MPC Problem
        for t in range(self.Nt):

            # Get variables
            x_t = opt_var['x', t]
            if self.trajectory_tracking:
                # TODO: remove 'raise NotImplementedError' and obtain the desired step in the reference trajectory -> ecessary logic to obtain the
#desired reference at each time-step.
                #raise NotImplementedError
                x_r = x_ref[t*self.Nx:(t+1)*self.Nx]
            else:
                x_r = x_ref
            u_t = opt_var['u', t]

            # Dynamics constraint
            x_t_next = self.dynamics(x_t, u_t)
            con_eq.append(x_t_next - opt_var['x', t + 1])

            # Input constraints
            if uub is not None:
                con_ineq.append(u_t)
                con_ineq_ub.append(uub)
                con_ineq_lb.append(np.full((self.Nu,), -ca.inf))
            if ulb is not None:
                con_ineq.append(u_t)
                con_ineq_ub.append(np.full((self.Nu,), ca.inf))
                con_ineq_lb.append(ulb)

            # State constraints
            if xub is not None:
                con_ineq.append(x_t)
                con_ineq_ub.append(xub)
                con_ineq_lb.append(np.full((self.Nx,), -ca.inf))
            if xlb is not None:
                con_ineq.append(x_t)
                con_ineq_ub.append(np.full((self.Nx,), ca.inf))
                con_ineq_lb.append(xlb)

            # Objective Function / Cost Function
            obj += self.running_cost(x_t, x_r, self.Q, u_t, self.R)

        # Terminal Cost
        obj += self.terminal_cost(opt_var['x', self.Nt], x_r, self.P)

        # Terminal contraint
        if terminal_constraint is not None:
            # Should be a polytope
            H_N = terminal_constraint.A
            if H_N.shape[1] != self.Nx:
                print("Terminal constraint with invalid dimensions.")
                exit()

            H_b = terminal_constraint.b
            con_ineq.append(ca.mtimes(H_N, opt_var['x', self.Nt]))
            con_ineq_lb.append(-ca.inf * ca.DM.ones(H_N.shape[0], 1))
            con_ineq_ub.append(H_b)

        # Equality constraints bounds are 0 (they are equality constraints),
        # -> Refer to CasADi documentation
        num_eq_con = ca.vertcat(*con_eq).size1()
        num_ineq_con = ca.vertcat(*con_ineq).size1()
        con_eq_lb = np.zeros((num_eq_con,))
        con_eq_ub = np.zeros((num_eq_con,))

        # Set constraints
        con = ca.vertcat(*(con_eq + con_ineq))
        self.con_lb = ca.vertcat(con_eq_lb, *con_ineq_lb)
        self.con_ub = ca.vertcat(con_eq_ub, *con_ineq_ub)

        # Build NLP Solver (can also solve QP)
        nlp = dict(x=opt_var, f=obj, g=con, p=param_s)
        options = {
            'ipopt.print_level': 0,
            'print_time': False,
            'verbose': False,
            'expand': True
        }
        
        # options = {
        # 'qpsol': 'osqp',  # Use qpoases as the QP solver
        # 'qpsol_options': {
        # 'printLevel': 'none',  # Adjust verbosity of QP solver
        # },
        # 'hessian_approximation': 'limited-memory',  # Hessian approximation for SQP
        # 'max_iter': 10,  # Set maximum solver iterations
        # 'print_time': False,  # Disable solver time printouts
        # 'verbose': False  # Turn off verbose output
        # }
        # options = {
        # 'qpsol': 'qpoases',  # Use qpoases as the QP solver
        # 'qpsol_options': {'printLevel': 'none'},
        # 'hessian_approximation': 'limited-memory',  # Hessian approximation for SQP
        # 'max_iter': 50,  # Limit iterations
        # 'tol_du': 1e-4,  # Dual infeasibility tolerance
        # 'tol_pr': 1e-4,  # Primal infeasibility tolerance
        # 'print_time': False,
        # 'verbose': False
        # }
        if solver_opts is not None:
            options.update(solver_opts)
        self.solver = ca.nlpsol('mpc_solver', 'ipopt' #'sqpmethod'
                                , nlp, options)
     

        build_solver_time += time.time()
        print('\n________________________________________')
        print('# Time to build mpc solver: %f sec' % build_solver_time)
        print('# Number of variables: %d' % self.num_var)
        print('# Number of equality constraints: %d' % num_eq_con)
        print('# Number of inequality constraints: %d' % num_ineq_con)
        print('----------------------------------------')
        pass

    def load_params(self, param, tuning_file=None):
        """
        Parameters loader function.
        Loads yaml parameters to generate Q, R and P.

        :param param: parameter setting ('P1', 'P2', 'P3')
        :type param: string
        """

        if param not in ['P1', 'P2', 'P3']:
            print("Wrong param option. Select param='P1' or 'P2' or 'P3'.")
            exit()

        if tuning_file is None:
            f_path = "tuning.yaml"
        else:
            f_path = tuning_file

        with open(f_path, 'r') as stream:
            parameters = yaml.safe_load(stream)

            # Create numpy diags
            Q_diag = np.asarray(parameters[param]['Q'])
            R_diag = np.asarray(parameters[param]['R'])
            P_diag = parameters[param]['P_mult'] * Q_diag

            # Get matrices
            Q = np.diag(Q_diag)
            R = np.diag(R_diag)
            P = np.diag(P_diag)

            return Q, R, P

        raise NotADirectoryError("Wrong directory for yaml file.")

    def set_cost_functions_quat(self):
        """
        Helper method to create CasADi functions for the MPC cost objective.
        """
        # Create functions and function variables for calculating the cost
        Q = ca.MX.sym('Q', self.Nx - 1, self.Nx - 1)
        P = ca.MX.sym('P', self.Nx - 1, self.Nx - 1)
        R = ca.MX.sym('R', self.Nu, self.Nu)

        x = ca.MX.sym('x', self.Nx)
        xr = ca.MX.sym('xr', self.Nx)
        u = ca.MX.sym('u', self.Nu)

        # Prepare variables
        p = x[0:3]
        v = x[3:6]
        q = x[6:10]
        w = x[10:]

        pr = xr[0:3]
        vr = xr[3:6]
        qr = xr[6:10]
        wr = xr[10:]

        # Calculate errors
        ep = p - pr
        ev = v - vr
        ew = w - wr
        eq = ca.MX.ones(3, 1) * (1 - ca.mtimes(qr.T, q)**2)

        e_vec = ca.vertcat(*[ep, ev, eq, ew])

        # Calculate running cost
        ln = ca.mtimes(ca.mtimes(e_vec.T, Q), e_vec) \
            + ca.mtimes(ca.mtimes(u.T, R), u)

        self.running_cost = ca.Function('ln', [x, xr, Q, u, R], [ln])

        # Calculate terminal cost
        V = ca.mtimes(ca.mtimes(e_vec.T, P), e_vec)
        self.terminal_cost = ca.Function('V', [x, xr, P], [V])

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
            self.x_sp = np.zeros(self.Nx * (self.Nt + 1))

        # Initialize variables
        self.optvar_x0 = np.full((1, self.Nx), x0.T)

        # Initial guess of the warm start variables
        self.optvar_init = self.opt_var(0)
        self.optvar_init['x', 0] = self.optvar_x0[0]

        solve_time = -time.time()

        param = ca.vertcat(x0, self.x_sp, u0)
        args = dict(x0=self.optvar_init,
                    lbx=self.optvar_lb,
                    ubx=self.optvar_ub,
                    lbg=self.con_lb,
                    ubg=self.con_ub,
                    p=param)

        # Solve NLP
        sol = self.solver(**args)
        status = self.solver.stats()['return_status']
        optvar = self.opt_var(sol['x'])

        cost = sol['f']

        solve_time += time.time()

        self.cpu_times.append(solve_time)
        self.costs.append(float(cost))
        self.horizon_lengths.append(self.Nt)


        metrics = self.get_performance_metrics()
        print("Max CPU Time: %f" % metrics["max_cpu_time"])
        print("Avg CPU Time: %f" % metrics["avg_cpu_time"])
        print("Max Cost: %f" % metrics["max_cost"])
        print("Avg Cost: %f" % metrics["avg_cost"])

        print('MPC - CPU time: %f seconds  |  Cost: %f  |  Horizon length: %d ' % (solve_time, sol['f'], self.Nt))

       
        return optvar['x'], optvar['u']

    def mpc_controller(self, x0, t):
        """
        MPC controller wrapper.
        Gets first control input to apply to the system.

        :param x0: initial state
        :type x0: np.ndarray
        :return: control input
        :rtype: ca.DM
        """
        if self.trajectory_tracking:
            x_traj = self.model.get_trajectory(t, self.Nt + 1, self.fw_propagating)
            x_sp = x_traj.reshape(self.Nx * (self.Nt + 1), order='F')
            self.set_reference(x_sp)
        _, u_pred = self.solve_mpc(x0)

        # Calculate error to first state
        error = self.calculate_error(x0, self.x_sp[0:13])

        return u_pred[0], error

    def astrobee_sim_controller(self, x0, xh):
        """
        Controller to be used in the Astrobee simulator

        :param x0: [description]
        :type x0: [type]
        :param xh: [description]
        :type xh: [type]
        :return: [description]
        :rtype: [type]
        """
        x_traj = self.model.forward_propagate(xh, self.Nt + 1, radius=0.5)
        x_sp = x_traj.reshape(self.Nx * (self.Nt + 1), order='F')
        self.set_reference(x_sp)
        _, u_pred = self.solve_mpc(x0)

        # Calculate error to first state
        error = self.calculate_error(x0, x_traj[:, 0])

        return u_pred[0], error

    def calculate_error(self, x, xr):
        """
        Calculate error, used for logging

        :param x: [description]
        :type x: [type]
        :param xr: [description]
        :type xr: [type]
        :return: [description]
        :rtype: [type]
        """

        # Prepare xr
        xr = xr.reshape((xr.shape[0], 1))

        # Prepare variables
        p = x[0:3]
        v = x[3:6]

        pr = xr[0:3]
        vr = xr[3:6]

        # Calculate errors
        ep = p - pr
        ev = v - vr
        q = x[6:10]
        qr = xr[6:10]

        w = x[10:]
        wr = xr[10:]

        eq = ca.DM.ones(3, 1) * (1 - ca.mtimes(qr.T, q)**2)
        ew = w - wr

        return np.concatenate((ep, ev, eq, ew)).reshape((12, 1))

    def set_reference(self, x_sp):
        """
        Set the controller reference state

        :param x_sp: desired reference state
        :type x_sp: np.ndarray
        """
        self.x_sp = x_sp

    def set_forward_propagation(self):
        """
        Helper function to set reference forward propagation
        based of current state and velocity measurements.
        """
        self.fw_propagating = True

    def get_performance_metrics(self):
   
        if not self.cpu_times or not self.costs:
            return {"error": "No data available. Ensure MPC has been solved at least once."}
    
        max_cpu_time = max(self.cpu_times)
        avg_cpu_time = sum(self.cpu_times) / len(self.cpu_times)
    
        max_cost = max(self.costs)
        avg_cost = sum(self.costs) / len(self.costs)
    
        return {
            "max_cpu_time": max_cpu_time,
            "avg_cpu_time": avg_cpu_time,
            "max_cost": max_cost,
            "avg_cost": avg_cost
        }
    
    def get_mpc_performance(self):
        metrics = self.get_performance_metrics()
        max_ct = metrics["max_cpu_time"]
        avg_ct = metrics["avg_cpu_time"]
        return max_ct, avg_ct



    