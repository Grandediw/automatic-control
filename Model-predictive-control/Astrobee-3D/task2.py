import numpy as np

from astrobee_3d  import Astrobee
from optimization import FiniteOptimization
from simulation   import EmbeddedSimEnvironment

# Create pendulum and controller objects
bumble = Astrobee()

# Get the system discrete-time dynamics
A, B = bumble.cartesian_ground_dynamics()
Ad, Bd, Cd, Dd = bumble.c2d(A, B, np.eye(6), np.zeros((6, 3)))
bumble.set_discrete_dynamics(Ad, Bd)

# Get controller

R = np.eye(3) * 10


# Without u_lim we can do 20s - with u_lim, we need 30! Important to write to them
u_lim = np.array([[0.85, 0.42, 0.04]]).T
ctl = FiniteOptimization(model          = bumble, 
                         dynamics       = bumble.linearized_discrete_dynamics,
                         total_time     = 30.0,
                         rendezvous_time= 25.0,
                         R              = R,
                         u_lim          = u_lim)



# We want Bumble to go after Honey! So now, we will just create a simple
# trajectory for Honey to do while waiting rendezvous
honey = Astrobee()
honey.set_trajectory(time=30.0)
x_r        = honey.get_trajectory(t_start=25.0)
x_full_ref = honey.get_trajectory(t_start=0.0)

x0 = np.array([[0.1, 0, 0, 0, 0.01, 0]]).T
x_star, u_star = ctl.solve_problem(x0, x_r)

# # Initialize simulation environment
sim_env = EmbeddedSimEnvironment(model=bumble,
                                 dynamics=bumble.linearized_discrete_dynamics,
                                 controller=u_star,
                                 time=30.0)
sim_env.set_window(1)

t, y, u = sim_env.run(x0, x_ref=x_full_ref, online_plot=False)
sim_env.visualize_error()
sim_env.visualize_prediction_vs_reference(x_pred=x_star, x_ref=x_full_ref, control=u_star)
sim_env.visualize_state_vs_reference(state=y, ref=x_full_ref, control=u_star)

# Activate broken thruster and re-run
sim_env.broken_thruster()
t, y, u = sim_env.run(x0, x_ref=x_full_ref)
sim_env.visualize_error()

# Get the system discrete-time dynamics with Z
queen = Astrobee(axis="3d")
A, B = queen.cartesian_3d_dynamics()
Ad, Bd, Cd, Dd = queen.c2d(A, B, np.eye(8), np.zeros((8, 4)))
queen.set_discrete_dynamics(Ad, Bd)

# Get controller for 3D
R = np.eye(4) * 10
ctl_wz = FiniteOptimization(queen, queen.linearized_discrete_dynamics,
                            total_time=40.0, rendezvous_time=35.0, ref_type="3d",
                            R=R)

honey.set_trajectory(time=40.0, type="3d")
x_r = honey.get_trajectory(t_start=35.0)
x_full_ref = honey.get_trajectory(t_start=0.0)

x0 = np.array([[1.0, 0, 0, 0, 0, 0, 0, 0]]).T
x_star, u = ctl_wz.solve_problem(x0, x_r)

# Simulate 3D
sim_env = EmbeddedSimEnvironment(model=queen,
                                 dynamics=queen.linearized_discrete_dynamics,
                                 controller=u,
                                 time=40.0)
t, y, u = sim_env.run(x0, x_ref=x_full_ref)
sim_env.visualize_error()
