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

        # Sim data
        self.t = self.x_vec = self.u_vec = self.sim_loop_length = None

        # Plotting definitions
        # running plot window, in seconds, or float("inf")
        self.plt_window = float("inf")

    def run(self, x0, online_plot=False):
        """
        Run simulator with specified system dynamics and control function.

        :param x0: initial state
        :type x0: list, np.ndarray
        :param online_plot: boolean to plot online or not
        :type online_plot: boolean
        """

        print("Running simulation....")
        sim_loop_length = int(self.total_sim_time /
                              self.dt) + 1  # account for 0th
        t = np.array([0])
        x_vec = np.array([x0]).reshape(2, 1)      #reshape(2,1) if one.axis
        u_vec = np.empty((1, 0))

        # Start figure
        if online_plot:
            fig, (ax1, ax2, ax3) = plt.subplots(3)
        for i in range(sim_loop_length):
            # Iteration
            print(i, "/", (sim_loop_length - 1))
            # Get control input and obtain next state
            x = x_vec[:, -1].reshape(2, 1)      #reshape(2,1) if one.axis
            u = self.controller(x)
            x_next = self.dynamics(x, u)

            # Store data
            t = np.append(t, t[-1] + self.dt)
            x_vec = np.append(x_vec, np.array(x_next).reshape(2, 1), axis=1)
            u_vec = np.append(u_vec, u.reshape(1, 1))

            if online_plot:
                # Get plot window values:
                if self.plt_window != float("inf"):
                    l_wnd = 0 if int(
                        i + 1 - self.plt_window / self.dt) < 1 else int(i + 1 - self.plt_window / self.dt)
                else:
                    l_wnd = 0

                ax1.clear()
                ax1.set_title("Astrobee")
                ax1.plot(t[l_wnd:], x_vec[0, l_wnd:], 'r--')
                ax1.legend(["x1"])
                ax1.set_ylabel("Position [m]")

                ax2.clear()
                ax2.plot(t[l_wnd:], x_vec[1, l_wnd:], 'r--')
                ax2.legend(["x2"])
                ax2.set_ylabel("Velocity [m/s]")

                ax3.clear()
                ax3.plot(t[l_wnd:-1], u_vec[l_wnd:], 'r--')
                ax3.legend(["u"])
                ax3.set_ylabel("Force input [n]")

                plt.pause(0.001)

        if online_plot:
            plt.show()

        # Store data internally for offline plotting
        self.t = t
        self.x_vec = x_vec
        self.u_vec = u_vec
        self.sim_loop_length = sim_loop_length

        return t, x_vec, u_vec

    def visualize(self):
        """
        Offline plotting of simulation data
        """
        variables = list(
            [self.t, self.x_vec, self.u_vec, self.sim_loop_length])
        if any(elem is None for elem in variables):
            print("Please run the simulation first with the method 'run'.")

        t = self.t
        x_vec = self.x_vec
        u_vec = self.u_vec

        fig, (ax1, ax2, ax3) = plt.subplots(3)
        ax1.clear()
        ax1.set_title("Astrobee")
        ax1.plot(t, x_vec[0, :], 'r--')
        ax1.legend(["x1"])
        ax1.set_ylabel("Position [m]")

        ax2.clear()
        ax2.plot(t, x_vec[1, :], 'r--')
        ax2.legend(["x2"])
        ax2.set_ylabel("Velocity [m/s]")

        ax3.clear()
        ax3.plot(t[:-1], u_vec, 'r--')
        ax3.legend(["u"])
        ax3.set_ylabel("Force input [n]")


        plt.show()

    def set_window(self, window):
        """
        Set the plot window length, in seconds.

        :param window: window length [s]
        :type window: float
        """
        self.plt_window = window




class EmbeddedSimEnvironment_2axis(object):

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

        # Sim data
        self.t = self.x_vec = self.u_vec = self.sim_loop_length = None

        # Plotting definitions
        # running plot window, in seconds, or float("inf")
        self.plt_window = float("inf")

    def run(self, x0, online_plot=False):
        """
        Run simulator with specified system dynamics and control function.

        :param x0: initial state
        :type x0: list, np.ndarray
        :param online_plot: boolean to plot online or not
        :type online_plot: boolean
        """

        print("Running simulation....")
        sim_loop_length = int(self.total_sim_time /
                              self.dt) + 1  # account for 0th
        t = np.array([0])
        x_vec = np.array([x0]).reshape(4, 1)      #reshape(2,1) if one.axis
        u_vec = np.empty((2, 1))
  

        # Start figure
        if online_plot:
            fig, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(6)
        for i in range(sim_loop_length):
            # Iteration
            print(i, "/", (sim_loop_length - 1))
            # Get control input and obtain next state
            x = x_vec[:, -1].reshape(4, 1)      #reshape(2,1) if one.axis
            u = self.controller(x)
            x_next = self.dynamics(x, u)

            # Store data
            t = np.append(t, t[-1] + self.dt)
            x_vec = np.append(x_vec, np.array(x_next).reshape(4, 1), axis=1)
            u_vec = np.append(u_vec, u.reshape(2, 1),axis=1)

            #print("u in simulation",u_vec)
            #print("x in simulation",x_vec)

            if online_plot:
                # Get plot window values:
                if self.plt_window != float("inf"):
                    l_wnd = 0 if int(
                        i + 1 - self.plt_window / self.dt) < 1 else int(i + 1 - self.plt_window / self.dt)
                else:
                    l_wnd = 0

                ax1.clear()
                ax1.set_title("Astrobee")
                ax1.plot(t[l_wnd:], x_vec[0, l_wnd:], 'r--')
                ax1.legend(["x1"])
                ax1.set_ylabel("P(x) [m]")

                ax2.clear()
                ax2.plot(t[l_wnd:], x_vec[1, l_wnd:], 'r--')
                ax2.legend(["x2"])
                ax2.set_ylabel("V(x) [m/s]")

                ax3.clear()
                ax3.plot(t[l_wnd:-1], u_vec[0,l_wnd:], 'r--')
                ax3.legend(["u"])
                ax3.set_ylabel("F(x) [n]")

                ax4.clear()
                ax4.plot(t[l_wnd:], x_vec[2, l_wnd:], 'r--')
                ax4.legend(["y1"])
                ax4.set_ylabel("P(y) [m]")

                ax5.clear()
                ax5.plot(t[l_wnd:], x_vec[3, l_wnd:], 'r--')
                ax5.legend(["y2"])
                ax5.set_ylabel("V(y) [m/s]")

                ax6.clear()
                ax6.plot(t[l_wnd:-1], u_vec[1,l_wnd:], 'r--')
                ax6.legend(["u2"])
                ax6.set_ylabel("F(y) [n]")

                plt.pause(0.001)

        if online_plot:
            plt.show()

        # Store data internally for offline plotting
        self.t = t
        self.x_vec = x_vec
        self.u_vec = u_vec
        self.sim_loop_length = sim_loop_length

        return t, x_vec, u_vec

    def visualize(self):
        """
        Offline plotting of simulation data
        """
        variables = list(
            [self.t, self.x_vec, self.u_vec, self.sim_loop_length])
        if any(elem is None for elem in variables):
            print("Please run the simulation first with the method 'run'.")

        t = self.t
        x_vec = self.x_vec
        u_vec = self.u_vec


        fig, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(6)
        ax1.clear()
        ax1.set_title("Astrobee")
        ax1.plot(t, x_vec[0, :], 'r--')
        ax1.legend(["x1"])
        ax1.set_ylabel("P(x) [m]")

        ax2.clear()
        ax2.plot(t, x_vec[1, :], 'r--')
        ax2.legend(["x2"])
        ax2.set_ylabel("V(x) [m/s]")

        ax3.clear()
        ax3.plot(t[:-1], u_vec[0,0:401], 'r--')
        ax3.legend(["u"])
        ax3.set_ylabel("F(x) [n]")

        ax4.clear()
        ax4.plot(t, x_vec[2, :], 'r--')
        ax4.legend(["y1"])
        ax4.set_ylabel("P(y) [m]")

        ax5.clear()
        ax5.plot(t, x_vec[3, :], 'r--')
        ax5.legend(["y2"])
        ax5.set_ylabel("V(y) [m/s]")

        ax6.clear()
        ax6.plot(t[:-1], u_vec[1,0:401], 'r--')
        ax6.legend(["u2"])
        ax6.set_ylabel("F(y) [n]")


        plt.show()

    def set_window(self, window):
        """
        Set the plot window length, in seconds.

        :param window: window length [s]
        :type window: float
        """
        self.plt_window = window

