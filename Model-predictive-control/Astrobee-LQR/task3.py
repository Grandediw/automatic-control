import numpy as np

from astrobee import Astrobee
from dlqr import DLQR
from simulation import EmbeddedSimEnvironment

# ------------------------------
# Part I - LQR Design
# ------------------------------
# Instantiate an Astrobee
abee = Astrobee(h=0.1)

# Linearization around reference point
x_star = np.zeros((12, 1))
x_star[0] = 1
x_star[1] = 0.5
x_star[2] = 0.1
x_star[6] = 0.087
x_star[7] = 0.077
x_star[8] = 0.067

A, B = abee.create_linearized_dynamics(x_bar=x_star)

C = np.diag(np.ones(12))
D = np.zeros((12, 6))

Ad, Bd, Cd, Dd = abee.c2d(A, B, C, D)

ctl = DLQR(Ad, Bd, C)
abee.set_discrete_dynamics(Ad, Bd)

# TODO: Check eigenvalues, and verify that for each left eigenvector v of Ad
#       corresponding to an eigenvalue not inside the unit circle, v @ Bd != 0  PBH
E, V = np.linalg.eig(Ad.T)

print(E)
print(np.round(V,3))

PBH_test= np.round(V.T @ Bd,3)

print(PBH_test)
np.round(V.T @ Bd,3)




R_coefficients = np.ones(6)
Q_coefficients = np.ones(12)

print(Q_coefficients)

#R_coefficients = np.eye(6)
#Q_coefficients = np.eye(12)

# TODO: uncomment the code below to adjust the coefficients of Q and R
#Q_coefficients[0:3] = 100
#Q_coefficients[3:6] = 100  # <- point 2
#Q_coefficients[6:9] = 100
#Q_coefficients[9:12] = 100  
# R_coefficients[0:3] = 0
# R_coefficients[3:6] = 0

Q = np.diag(Q_coefficients)
R = (np.diag(R_coefficients))

# R = (np.diag(R_coefficients))*10 <- point 1

print(Q)
print(R)

K, P = ctl.get_lqr_gain(Q, R)


R_coefficients_tune = np.ones(6)
Q_coefficients_tune = np.ones(12)

# TODO: uncomment the code below to adjust the coefficients of Q and R
Q_coefficients_tune [0:3] = (0.54/0.06)**2
Q_coefficients_tune [3:6] = (0.8/0.03)**2  # <- point 2
Q_coefficients_tune [6:9] = (10**-6/10**(-7))**2
Q_coefficients_tune [9:12] = 0   
R_coefficients_tune[0:3] = (9/0.85)**2
R_coefficients_tune[3:6] = (1/0.04)**2

rho = 1

Q_tune = np.diag(Q_coefficients_tune)
R_tune = (np.diag(R_coefficients_tune))*rho

# R = (np.diag(R_coefficients))*10 <- point 1

print(Q_tune)
print(R_tune)




K, P = ctl.get_lqr_gain(Q_tune, R_tune)

# Set reference for controller
ctl.set_reference(x_star)


sim_env = EmbeddedSimEnvironment(model=abee,
                                 dynamics=abee.linearized_discrete_dynamics,
                                 controller=ctl.feedback,
                                 time=20)

# Starting pose
x0 = np.zeros((12, 1))

t, y, u = sim_env.run(x0)
sim_env.evaluate_performance(t, y, u)

# ------------------------------
# Part II - LQG Design
# ------------------------------
# Output feedback - measure position, attitude and angular velocity
#             Goal - estimate linear velocity
C = np.eye(3)
C = np.hstack((C, np.zeros((3, 3))))

# Create the matrices for Qn and Rn
# TODO: adjust the values of Qn and Rn to answer Q4 and Q5 - they start at 0
Q_diag = np.vstack((np.ones((3, 1)) * 1, np.zeros((3, 1))))
R_diag = np.vstack((np.ones((3, 1)) * 1))
Qn = np.diag(Q_diag.reshape(6, ))
Rn = np.diag(R_diag.reshape(3, ))

print("Qn",Qn,"Rn",Rn)

abee.set_kf_params(C, Qn, Rn)
abee.init_kf(x0[0:6].reshape(6, 1))

sim_env_lqg = EmbeddedSimEnvironment(model=abee,
                                     dynamics=abee.linearized_discrete_dynamics,
                                     controller=ctl.feedback,
                                     time=20)
sim_env_lqg.set_estimator(True)
t, y, u = sim_env_lqg.run(x0)
