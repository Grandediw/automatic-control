import numpy as np
import casadi as ca
import matplotlib.pyplot as plt
import time


class EmbeddedSimEnvironment(object):

    def __init__(self, model, dynamics, controller, time=100.0):
        """
        Embedded simulation environment. Simulates the system given dynamics
        and a control law, plots in matplotlib.

        :param model: model object
        :type model: object
        :param dynamics: system dynamics function (x, u)
        :type dynamics: casadi.DM
        :param controller: controller function (x, r)
        :type controller: casadi.DM
        :param time: total simulation time, defaults to 100 seconds
        :type time: float, optional
        """
        self.model = model
        self.dynamics = dynamics
        self.controller = controller
        self.total_sim_time = time  # seconds
        self.dt = self.model.dt
        self.estimation_in_the_loop = False

        # Plotting definitions
        self.plt_window = float("inf")    # running plot window, in seconds, or float("inf")

    def run(self, x0):
        """
        Run simulator with specified system dynamics and control function.
        """

        print("Running simulation....")
        sim_loop_length = int(self.total_sim_time / self.dt) + 1  # account for 0th
        t = np.array([0])
        x_vec = np.array([x0]).reshape(12, 1)
        u_vec = np.empty((6, 0))

        # Start figure
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4)
        for i in range(sim_loop_length):

            if self.estimation_in_the_loop is False:
                # Get control input and obtain next state
                x = x_vec[:, -1].reshape(12, 1)
                u = self.controller(x)
                x_next = self.dynamics(x, u)
            else:
                if self.model.C_KF is None:
                    print("Please initialize the KF module.")
                    exit()
                # Get last state from dynamics
                x = x_vec[:, -1].reshape(12, 1)

                # Get measurement
                measurement_noise = np.random.uniform(-0.5, 0.5, (3, 1))
                y = self.model.C_KF @ x[0:6, :].reshape(6, 1) + measurement_noise

                # Estimate the velocity from noisy position measurements
                if i == 0:
                    self.model.kf_estimator.predict(np.zeros((3, 1)))
                else:
                    self.model.kf_estimator.predict(u_vec[0:3, -1].reshape((3, 1)))
                self.model.kf_estimator.update(y)
                x_kf = self.model.kf_estimator.x
                x_kf = np.vstack((x_kf, x[6:].reshape(6, 1))).reshape(12, 1)

                # Get control input
                u = self.controller(x_kf)

                # Propagate dynamics
                x_next = self.dynamics(x, u)

            # Store data
            t = np.append(t, t[-1] + self.dt)
            x_vec = np.append(x_vec, np.array(x_next).reshape(12, 1), axis=1)
            u_vec = np.append(u_vec, np.array(u).reshape(6, 1), axis=1)

            # Get plot window values:
            if self.plt_window != float("inf"):
                l_wnd = 0 if int(i + 1 - self.plt_window / self.dt) < 1 else int(i + 1 - self.plt_window / self.dt)
            else:
                l_wnd = 0

        ax1.clear()
        ax1.set_title("Astrobee")
        ax1.plot(t[l_wnd:], x_vec[0, l_wnd:], 'r--',
                 t[l_wnd:], x_vec[1, l_wnd:], 'b--',
                 t[l_wnd:], x_vec[2, l_wnd:], 'g--',)
        ax1.legend(["x1", "x2", "x3"])
        ax1.set_ylabel("Position [m]")

        ax2.clear()
        ax2.plot(t[l_wnd:], x_vec[3, l_wnd:], 'r--',
                 t[l_wnd:], x_vec[4, l_wnd:], 'g--',
                 t[l_wnd:], x_vec[5, l_wnd:], 'b--')
        ax2.legend(["x4", "x4", "x5"])
        ax2.set_ylabel("Velocity [m/s]")

        ax3.clear()
        ax3.plot(t[l_wnd:], x_vec[6, l_wnd:], 'r--',
                 t[l_wnd:], x_vec[7, l_wnd:], 'g--',
                 t[l_wnd:], x_vec[8, l_wnd:], 'b--')
        ax3.legend(["x6", "x7", "x8"])
        ax3.set_ylabel("Attitude [rad]")

        # ax4.clear()
        # ax4.plot(t[l_wnd:], x_vec[9, l_wnd:], 'r--',
        #          t[l_wnd:], x_vec[10, l_wnd:], 'g--',
        #          t[l_wnd:], x_vec[11, l_wnd:], 'b--')
        # ax4.legend(["x9", "x10", "x11"])
        # ax4.set_ylabel("Angular Velocity")

        ax4.clear()
        ax4.set_title("Astrobee")
        ax4.plot(t[:-1], u_vec[0, :], 'r--',
                 t[:-1], u_vec[1, :], 'b--',
                 t[:-1], u_vec[2, :], 'g--',)
        ax4.legend(["u1", "u2", "u3"])
        ax4.set_ylabel("Input Forces [N]")

        plt.show()
        return t, x_vec, u_vec

    def set_window(self, window):
        """
        Set the plot window length, in seconds.

        :param window: window length [s]
        :type window: float
        """
        self.plt_window = window

    def set_estimator(self, value):
        """Enable or disable the KF estimator in the loop.

        :param value: desired state
        :type value: boolean
        """
        if isinstance(value, bool) is not True:
            print("set_estimator needs to recieve a boolean variable")
            exit()

        self.estimation_in_the_loop = value

    def evaluate_performance(self, t, y, u):
        """
        Evaluate the system performance.

        :param t: timesteps
        :type t: np.ndarray
        :param y: system output
        :type y: np.ndarray
        :param u: control input
        :type u: np.ndarray
        """

        max_dist = 0
        max_spd = 0

        for timestep in t:
            if timestep >= 12:
                i = np.where(t == timestep)
                current_dist = np.linalg.norm(y[:3, i] - np.array([[[1, 0.5, 0.1]]]).T)
                current_spd = np.linalg.norm(y[3:6, i])
                if current_dist > max_dist:
                    max_dist = current_dist
                if current_spd > max_spd:
                    max_spd = current_spd

        print('Max distance to reference:')
        print('   ', max_dist)
        print('Max speed:')
        print('   ', max_spd)

        print('Max forces:')
        print('   x: ', max(abs(u[0, :])))
        print('   y: ', max(abs(u[1, :])))
        print('   z: ', max(abs(u[2, :])))

        print('Max torques:')
        print('   x: ', max(abs(u[3, :])))
        print('   y: ', max(abs(u[4, :])))
        print('   z: ', max(abs(u[5, :])))

        print('Max Euler angle deviations:')
        print('   roll: ', max(abs(y[6, 121:] - 0.087)))
        print('   pitch: ', max(abs(y[7, 121:] - 0.077)))
        print('   yaw: ', max(abs(y[8, 121:] - 0.067)))
