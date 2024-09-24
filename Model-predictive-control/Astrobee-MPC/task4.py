import numpy as np
import polytope as pc
import scipy

from astrobee import Astrobee
from mpc import MPC
from simulation import EmbeddedSimEnvironment
from set_operations import SetOperations

SET_TYPE = "zero"  # Terminal invariant set type: select 'zero' or 'LQR'
CASE_SELECTION = "simulate"  # Select either "attitude", "translation", "simulate"

# Create pendulum and controller objects
abee = Astrobee()

# Get the system discrete-time dynamics
A, B, _, _ = abee.create_discrete_time_dynamics()

# Solve the ARE for our system to extract the terminal weight matrix P
Q = np.eye(12)
#Q = np.diag([1, 1, 1, 101, 101, 101, 1, 1, 1, 101, 101, 101])
R = np.eye(6) * 0.01
P_LQR = np.matrix(scipy.linalg.solve_discrete_are(A, B, Q, R))

# Instantiate controller
u_lim = np.array([[0.85, 0.41, 0.41, 0.085, 0.041, 0.041]]).T
x_lim = np.array([[1.2, 0.1, 0.1,
                  0.5, 0.5, 0.5,
                  0.2, 0.2, 0.2,
                  0.1, 0.1, 0.1]]).T

# Translation Dynamics
At = A[0:6, 0:6].reshape((6, 6))
Bt = B[0:6, 0:3].reshape((6, 3))
Qt = Q[0:6, 0:6].reshape((6, 6))
Rt = R[0:3, 0:3].reshape((3, 3)) 
x_lim_t = x_lim[0:6, :].reshape((6, 1))
u_lim_t = u_lim[0:3, :].reshape((3, 1))
set_ops_t = SetOperations(At, Bt, Qt, Rt, xlb=-x_lim_t, xub=x_lim_t)

# Attitude Dynamics
Aa = A[6:, 6:].reshape((6, 6))
Ba = B[6:, 3:].reshape((6, 3))
Qa = Q[6:, 6:].reshape((6, 6))
Ra = R[3:, 3:].reshape((3, 3))
x_lim_a = x_lim[6:, :].reshape((6, 1))
u_lim_a = u_lim[3:, :].reshape((3, 1))
set_ops_a = SetOperations(Aa, Ba, Qa, Ra, xlb=-x_lim_a, xub=x_lim_a)

# Part I
# TODO: For Q1, change N=10 to the different values of N and inv_set_type to "LQR" or "zero"
if CASE_SELECTION == "translation":
    # Q1
    KN_XN, all_sets, _ = set_ops_t.getNstepControllableSet(uub=u_lim_t, ulb=-u_lim_t, N=5, inv_set_type=SET_TYPE)
    set_ops_t.plotNsets(all_sets, plot_type=CASE_SELECTION)

    # Q2
    kns_u, _, _ = set_ops_t.getNstepControllableSet(uub=u_lim_t, ulb=-u_lim_t, N=5, inv_set_type=SET_TYPE)
    kns_3u, _, _ = set_ops_t.getNstepControllableSet(uub=3 * u_lim_t, ulb=-3 * u_lim_t, N=5, inv_set_type=SET_TYPE)
    sets = {'|u|': kns_u, '3 |u|': kns_3u}
    set_ops_t.plotNsets(sets, plotU=True, plot_type=CASE_SELECTION)
    exit()

elif CASE_SELECTION == "attitude":
    # Q1
    KN_XN, all_sets, _ = set_ops_a.getNstepControllableSet(uub=u_lim_a, ulb=-u_lim_a, N=10, inv_set_type=SET_TYPE)
    set_ops_a.plotNsets(all_sets, plot_type=CASE_SELECTION)

    # Q2
    kns_u, _, _ = set_ops_a.getNstepControllableSet(uub=u_lim_a, ulb=-u_lim_a, N=5, inv_set_type=SET_TYPE)
    kns_3u, _, _ = set_ops_a.getNstepControllableSet(uub=3 * u_lim_a, ulb=-3 * u_lim_a, N=5, inv_set_type=SET_TYPE)
    sets = {'|u|': kns_u, '3 |u|': kns_3u}
    set_ops_a.plotNsets(sets, plotU=True, plot_type=CASE_SELECTION)
    exit()

elif CASE_SELECTION == "simulate":
    if SET_TYPE == "zero":
        Xf_t = set_ops_t.zeroSet()
        Xf_a = set_ops_a.zeroSet()
    elif SET_TYPE == "LQR":
        # Create constraint polytope for translation and attitude
        Cub = np.eye(3)
        Clb = -1 * np.eye(3)

        Cb_t = np.concatenate((u_lim_t, u_lim_t), axis=0)
        C_t = np.concatenate((Cub, Clb), axis=0)

        Cb_a = np.concatenate((u_lim_a, u_lim_a), axis=0)
        C_a = np.concatenate((Cub, Clb), axis=0)

        Ct = pc.Polytope(C_t, Cb_t)
        Ca = pc.Polytope(C_a, Cb_a)

        # Get the LQR set for each of these
        Xf_t = set_ops_t.LQRSet(Ct)
        Xf_a = set_ops_a.LQRSet(Ca)
    else:
        print("Wrong choice of SET_TYPE, select 'zero' or 'LQR'.")

    Xf = pc.Polytope(scipy.linalg.block_diag(Xf_t.A, Xf_a.A), np.concatenate((Xf_t.b, Xf_a.b), axis=0))
 

# Part II - look into the MPC class and answer Q3
MPC_HORIZON = 50
# TODO: inspect the MPC class
ctl = MPC(model=abee,
          dynamics=abee.linearized_discrete_dynamics,
          Q=Q, R=R, P=P_LQR, N=MPC_HORIZON,
          ulb=-u_lim, uub=u_lim,
          xlb=-x_lim, xub=x_lim,
          terminal_constraint=Xf)

# Part III
# TODO: Answer Q4-Q7
# Set controller reference
x_d = np.zeros((12, 1))
ctl.set_reference(x_d)

# Set initial state
x0 = np.zeros((12, 1))
x0[0] = 0.2
x0[6] = 0.08
sim_env = EmbeddedSimEnvironment(model=abee,
                                 dynamics=abee.linearized_discrete_dynamics,
                                 controller=ctl.mpc_controller,
                                 time=20)
t, y, u = sim_env.run(x0)
sim_env.visualize()
