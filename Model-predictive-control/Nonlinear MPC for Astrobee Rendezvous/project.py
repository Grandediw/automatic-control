import numpy as np
from astrobee import Astrobee
from mpc import MPC
from simulation import EmbeddedSimEnvironment

# TODO: Set the path to the trajectory file:
#       eg.: trajectory_quat = '/home/roque/Project Assignment/Dataset/trajectory_quat.txt'
trajectory_quat = 'Dataset/trajectory_quat.txt'

# TODO: complete the 'tuning_file_path' variable to the path of your tuning.yaml
#       eg.: tuning_file_path = '/home/roque/Project Assignment/tuning.yaml'
tuning_file_path = 'tuning.yaml'

# Q1
# TODO: Set the Astrobee dynamics in Astrobee->astrobee_dynamics_quat
abee = Astrobee(trajectory_file=trajectory_quat)

# If successful, test-dynamics should not complain
print("Performing dynamics test on model: ")
abee.test_dynamics()
print("Model successfully passed dynamics test :)\n")

# Instantiate controller
u_lim, x_lim = abee.get_limits()

# Create MPC Solver
# TODO: Select the parameter type with the argument param='P1'  - or 'P2', 'P3'
MPC_HORIZON = 10
ctl = MPC(model=abee,
          dynamics=abee.model,
          param='P3',
          N=MPC_HORIZON,
          ulb=-u_lim, uub=u_lim,
          xlb=-x_lim, xub=x_lim,
          tuning_file=tuning_file_path)

# Q2: Reference tracking
# TODO: adjust the tuning.yaml parameters for better performance
x_d = abee.get_static_setpoint()
ctl.set_reference(x_d)
# Set initial state
x0 = abee.get_initial_pose()

sim_env = EmbeddedSimEnvironment(model=abee,
                                 dynamics=abee.model,
                                 controller=ctl.mpc_controller,
                                 time=80)
t, y, u = sim_env.run(x0)
sim_env.visualize()  # Visualize state propagation

# Q3: Activate Tracking
# TODO: complete the MPC class for reference tracking
tracking_ctl = MPC(model=abee,
                   dynamics=abee.model,
                   param='P3',
                   N=MPC_HORIZON,
                   trajectory_tracking=True,
                   ulb=-u_lim, uub=u_lim,
                   xlb=-x_lim, xub=x_lim)
sim_env_tracking = EmbeddedSimEnvironment(model=abee,
                                        dynamics=abee.model,
                                        controller=tracking_ctl.mpc_controller,
                                        time=80)
t, y, u = sim_env_tracking.run(x0)
sim_env_tracking.visualize()  # Visualize state propagation
sim_env_tracking.visualize_error()

# Test 3: Activate forward propagation
# TODO: complete the MPC Astrobee class to be ready for forward propagation
abee.test_forward_propagation()
tracking_ctl.set_forward_propagation()
t, y, u = sim_env_tracking.run(x0)
sim_env_tracking.visualize()  # Visualize state propagation
sim_env_tracking.visualize_error()

# PERFORMANCE EVALUATION


# max_ct, avg_ct = tracking_ctl.get_mpc_performance()

# def compute_score(max_ct, avg_ct, cvg_t, ss_p, ss_a, convergence_pose_error, convergence_attitude_error):
#     score = 0.0
    
#     # Penalize max computational time above 0.1s
#     score -= max(round((max_ct - 0.1) * 100, 3), 0.0) * 0.1

#     # Penalize average computational time
#     if avg_ct > 0.1:
#         score += (0.1 - avg_ct) * 30
#     else:
#         score += max((0.1 - avg_ct), 0.0) * 5

#     # Factor in convergence time
#     score += max((35.0 - cvg_t), 0.0) * 0.1

#     # Factor in steady-state errors

#     score += (convergence_pose_error - ss_p) * 100
#     score += np.rad2deg(convergence_attitude_error - ss_a) * 1

#     return score

# # Call compute_score using the parameters
# score = compute_score(max_ct, avg_ct, cvg_t, ss_p, ss_a,0.1,0.1)

# print(f"Computed Score: {score}")