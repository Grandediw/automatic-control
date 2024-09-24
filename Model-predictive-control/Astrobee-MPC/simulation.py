import numpy as np
import matplotlib.pyplot as plt


class EmbeddedSimEnvironment(object):

    def __init__(self, model, dynamics, controller, time=100.0):
        """
        Embedded simulation environment. Simulates the syste given dynamics
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

    def run(self, x0):
        """
        Run simulator with specified system dynamics and control function.
        """

        print("Running simulation....")
        sim_loop_length = int(self.total_sim_time / self.dt) + 1  # account for 0th
        t = np.array([0])
        x_vec = np.array([x0]).reshape(12, 1)
        u_vec = np.empty((6, 0))

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
                process_noise = np.random.uniform(-0.005, 0.005, (3, 1))
                y = self.model.C_KF @ x[0:6, :].reshape(6, 1) + process_noise

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

        self.t = t
        self.x_vec = x_vec
        self.u_vec = u_vec
        self.sim_loop_length = sim_loop_length
        return t, x_vec, u_vec

    def visualize(self):
        """
        Offline plotting of simulation data
        """
        variables = list([self.t, self.x_vec, self.u_vec, self.sim_loop_length])
        if any(elem is None for elem in variables):
            print("Please run the simulation first with the method 'run'.")

        t = self.t
        x_vec = self.x_vec
        u_vec = self.u_vec

        print("------- SIMULATION STATUS -------")
        print("Energy used: ", np.sum(np.abs(u_vec * 10)))
        print("Position integral error: ", np.sum(np.abs(x_vec[0:3, :])))
        print("Attitude integral error: ", np.sum(np.abs(x_vec[6:9, :])))

        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4)
        fig2, (ax5, ax6) = plt.subplots(2)
        ax1.clear()
        ax1.set_title("Astrobee States")
        ax1.plot(t, x_vec[0, :], 'r--',
                 t, x_vec[1, :], 'g--',
                 t, x_vec[2, :], 'b--')
        ax1.legend(["x1", "x2", "x3"])
        ax1.set_ylabel("Position [m]")
        ax1.grid()

        ax2.clear()
        ax2.plot(t, x_vec[3, :], 'r--',
                 t, x_vec[4, :], 'g--',
                 t, x_vec[5, :], 'b--')
        ax2.legend(["x3", "x4", "x5"])
        ax2.set_ylabel("Velocity [m/s]")
        ax2.grid()

        ax3.clear()
        ax3.plot(t, x_vec[6, :], 'r--',
                 t, x_vec[7, :], 'g--',
                 t, x_vec[8, :], 'b--')
        ax3.legend(["x6", "x7", "x8"])
        ax3.set_ylabel("Attitude [rad]")
        ax3.grid()

        ax4.clear()
        ax4.plot(t, x_vec[9, :], 'r--',
                 t, x_vec[10, :], 'g--',
                 t, x_vec[11, :], 'b--')
        ax4.legend(["x9", "x10", "x11"])
        ax4.set_ylabel("Ang. velocity [rad/s]")
        ax4.grid()

        # Plot control input
        ax5.clear()
        ax5.set_title("Astrobee Control inputs")
        ax5.plot(t[:-1], u_vec[0, :], 'r--',
                 t[:-1], u_vec[1, :], 'g--',
                 t[:-1], u_vec[2, :], 'b--')
        ax5.legend(["u0", "u1", "u2"])
        ax5.set_ylabel("Force input [N]")
        ax5.grid()

        ax6.clear()
        ax6.plot(t[:-1], u_vec[3, :], 'r--',
                 t[:-1], u_vec[4, :], 'g--',
                 t[:-1], u_vec[5, :], 'b--')
        ax6.legend(["u3", "u4", "u5"])
        ax6.set_ylabel("Torque input [Nm]")
        ax6.grid()

        plt.show()

    def set_estimator(self, value):
        """Enable or disable the KF estimator in the loop.

        :param value: desired state
        :type value: boolean
        """
        if isinstance(value, bool) is not True:
            print("set_estimator needs to recieve a boolean variable")
            exit()

        self.estimation_in_the_loop = value
