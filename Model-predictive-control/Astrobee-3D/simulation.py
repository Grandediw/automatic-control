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
        
        self.model          = model
        self.dynamics       = dynamics
        self.controller     = controller
        self.total_sim_time = time  # seconds
        self.dt             = self.model.dt
        self.broke_thruster = False

        # Plotting definitions
        self.plt_window = float("inf")    # running plot window, in seconds, or float("inf")

    def run(self, x0 : np.ndarray, x_ref , online_plot: bool =False):  #: x_ref:dict[int,np.ndarray]|None
        """
        Run simulator with specified system dynamics and control function.
        """

        print("Running simulation....")
        sim_loop_length = int(self.total_sim_time / self.dt)  # account for 0th
        time_span       = np.arange(0., self.total_sim_time + self.dt, self.dt, )
        self.sim_3d     = False
        
        if self.model.n == 8:
            self.sim_3d = True
            
        x_list = []
        u_list = []
        e_list = []

        # create figures for plotting
        if online_plot:
            fig, (ax1, ax2, ax3, ax4) = plt.subplots(4)
        
        
        x_t = x0
        x_list.append(x_t)
        
        # save initial error
        if not self.sim_3d:
                error_p   = x_ref[0][:2].reshape(2, 1)  - x0[0:2]
                error_att = x_ref[0][2]  - x0[4]
        else:
                error_p   = x_ref[0][:3].reshape(3, 1) - x0[0:3]
                error_att = x_ref[0][3] - x0[6]
            
        error = np.concatenate((error_p, error_att.reshape(1, 1)), axis=0)
        e_list.append(error)
        
        
        # save error and state history
        for i in range(sim_loop_length):

            # Get control input and obtain next state
            x = x_t
            u = self.controller[i]
            
            if self.broke_thruster:
                u[1] = u[1] + np.random.uniform(-0.07, 0.07, (1, 1))
            x_next = self.dynamics(x, u)

            # Store data
            x_list.append(x_next) # this list is one step ahead
            u_list.append(u)
            
            x_t = x_next

            if not self.sim_3d:
                error_p   = x_ref[i][:2].reshape(2, 1)  - x[0:2]
                error_att = x_ref[i][2]  - x[4]
            else:
                error_p   = x_ref[i][:3].reshape(3, 1) - x[0:3]
                error_att = x_ref[i][3] - x[6]
            
            error = np.concatenate((error_p, error_att.reshape(1, 1)), axis=0)
            e_list.append(error)


        # Store data internally for offline plotting
        self.t     = time_span
        self.x_vec = np.hstack(x_list)
        self.u_vec = np.hstack(u_list)
        self.e_vec = np.hstack(e_list)
        self.sim_loop_length = sim_loop_length
        
        if online_plot:
            for i in range(sim_loop_length):
                # Get plot window values:
                if self.plt_window != float("inf"):
                    l_wnd = 0 if int(i + 1 - self.plt_window / self.dt) < 1 else int(i + 1 - self.plt_window / self.dt)
                else:
                    l_wnd = 0

                if not self.sim_3d:
                    ax1.clear()
                    ax1.set_title("Astrobee")
                    ax1.plot(time_span[l_wnd:], self.x_vec[0, l_wnd:], 'r--')
                    ax1.plot(time_span[l_wnd:], self.x_vec[1, l_wnd:], 'g--')
                    ax1.legend(["x1", "x2"])
                    ax1.set_ylabel("Position [m]")

                    ax2.clear()
                    ax2.plot(time_span[l_wnd:], self.x_vec[2, l_wnd:], 'r--')
                    ax2.plot(time_span[l_wnd:], self.x_vec[3, l_wnd:], 'g--')
                    ax2.legend(["x3", "x4"])
                    ax2.set_ylabel("Velocity [m/s]")

                    ax3.clear()
                    ax3.plot(time_span[l_wnd:], self.x_vec[4, l_wnd:], 'r--')
                    ax3.plot(time_span[l_wnd:], self.x_vec[5, l_wnd:], 'g--')
                    ax3.legend(["x5", "x6"])
                    ax3.set_ylabel("Attitude [rad] / Ang. velocity [rad/s]")

                    ax4.clear()
                    ax4.plot(time_span[l_wnd:-1], self.u_vec[0, l_wnd:], 'r--')
                    ax4.plot(time_span[l_wnd:-1], self.u_vec[1, l_wnd:], 'g--')
                    ax4.plot(time_span[l_wnd:-1], self.u_vec[2, l_wnd:], 'b--')
                    ax4.legend(["u1", "u2", "u3"], loc='upper left')
                    ax4.set_ylabel("Force input [n] / Torque [nm]")
                else:
                    ax1.clear()
                    ax1.set_title("Astrobee")
                    ax1.plot(time_span[l_wnd:], self.x_vec[0, l_wnd:], 'r--')
                    ax1.plot(time_span[l_wnd:], self.x_vec[1, l_wnd:], 'g--')
                    ax1.plot(time_span[l_wnd:], self.x_vec[2, l_wnd:], 'b--')
                    ax1.legend(["x1", "x2", "x3"])
                    ax1.set_ylabel("Position [m]")

                    ax2.clear()
                    ax2.plot(time_span[l_wnd:], self.x_vec[3, l_wnd:], 'r--')
                    ax2.plot(time_span[l_wnd:], self.x_vec[4, l_wnd:], 'g--')
                    ax2.plot(time_span[l_wnd:], self.x_vec[5, l_wnd:], 'b--')
                    ax2.legend(["x4", "x5", "x6"])
                    ax2.set_ylabel("Velocity [m/s]")

                    ax3.clear()
                    ax3.plot(time_span[l_wnd:], self.x_vec[6, l_wnd:], 'r--')
                    ax3.plot(time_span[l_wnd:], self.x_vec[7, l_wnd:], 'g--')
                    ax3.legend(["x5", "x6"])
                    ax3.set_ylabel("Attitude [rad] / Ang. velocity [rad/s]")

                    ax4.clear()
                    ax4.plot(time_span[l_wnd:-1], self.u_vec[0, l_wnd:], 'r--')
                    ax4.plot(time_span[l_wnd:-1], self.u_vec[1, l_wnd:], 'g--')
                    ax4.plot(time_span[l_wnd:-1], self.u_vec[2, l_wnd:], 'b--')
                    ax4.plot(time_span[l_wnd:-1], self.u_vec[3, l_wnd:], 'k--')
                    ax4.legend(["u1", "u2", "u3", "u4"], loc='upper left')
                    ax4.set_ylabel("Force input [n] / Torque [nm]")

                plt.pause(0.001)

        if online_plot:
            plt.show()
        

        return time_span, self.x_vec, self.u_vec

    def visualize(self):
        """
        Offline plotting of simulation data
        """
        variables = list([self.t, self.x_vec, self.u_vec, self.sim_loop_length])
        if any(elem is None for elem in variables):
            print("Please run the simulation first with the method 'run'.")

        t = self.t
        self.x_vec = self.x_vec
        self.u_vec = self.u_vec

        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4)
        if not self.sim_3d:
            ax1.clear()
            ax1.set_title("Astrobee")
            ax1.plot(t, self.x_vec[0, :], 'r--')
            ax1.plot(t, self.x_vec[1, :], 'g--')
            ax1.legend(["x1", "x2"])
            ax1.set_ylabel("Position [m]")

            ax2.clear()
            ax2.plot(t, self.x_vec[2, :], 'r--')
            ax2.plot(t, self.x_vec[3, :], 'g--')
            ax2.legend(["x3", "x4"])
            ax2.set_ylabel("Velocity [m/s]")

            ax3.clear()
            ax3.plot(t, self.x_vec[4, :], 'r--')
            ax3.plot(t, self.x_vec[5, :], 'g--')
            ax3.legend(["x5", "x6"])
            ax3.set_ylabel("Attitude [rad] / Ang. velocity [rad/s]")

            ax4.clear()
            ax4.plot(t[:-1], self.u_vec[0, :], 'r--')
            ax4.plot(t[:-1], self.u_vec[1, :], 'g--')
            ax4.plot(t[:-1], self.u_vec[2, :], 'b--')
            ax4.legend(["u1", "u2", "u3"], loc='upper left')
            ax4.set_ylabel("Force input [n] / Torque [nm]")
        else:
            ax1.clear()
            ax1.set_title("Astrobee")
            ax1.plot(t, self.x_vec[0, :], 'r--')
            ax1.plot(t, self.x_vec[1, :], 'g--')
            ax1.plot(t, self.x_vec[2, :], 'b--')
            ax1.legend(["x1", "x2", "x3"])
            ax1.set_ylabel("Position [m]")

            ax2.clear()
            ax2.plot(t, self.x_vec[3, :], 'r--')
            ax2.plot(t, self.x_vec[4, :], 'g--')
            ax2.plot(t, self.x_vec[5, :], 'b--')
            ax2.legend(["x4", "x5", "x6"])
            ax2.set_ylabel("Velocity [m/s]")

            ax3.clear()
            ax3.plot(t, self.x_vec[6, :], 'r--')
            ax3.plot(t, self.x_vec[7, :], 'g--')
            ax3.legend(["x5", "x6"])
            ax3.set_ylabel("Attitude [rad] / Ang. velocity [rad/s]")

            ax4.clear()
            ax4.plot(t[:-1], self.u_vec[0, :], 'r--')
            ax4.plot(t[:-1], self.u_vec[1, :], 'g--')
            ax4.plot(t[:-1], self.u_vec[2, :], 'b--')
            ax4.plot(t[:-1], self.u_vec[3, :], 'k--')
            ax4.legend(["u1", "u2", "u3", "u4"], loc='upper left')
            ax4.set_ylabel("Force input [n] / Torque [nm]")

        plt.show()

    def visualize_error(self, x_pred=None):
        """
        Offline plotting of simulation data
        """
        variables = list([self.t, self.e_vec, self.u_vec, self.sim_loop_length])
        if any(elem is None for elem in variables):
            print("Please run the simulation first with the method 'run'.")

        t = self.t
        e_vec = self.e_vec
        self.u_vec = self.u_vec

        fig, (ax1, ax2, ax3) = plt.subplots(3)
        if not self.sim_3d:
            ax1.clear()
            ax1.set_title("Astrobee tracking error")
            ax1.plot(t, e_vec[0, :], 'r--')
            ax1.plot(t, e_vec[1, :], 'g--')
            if x_pred is not None:
                x_np = np.asarray(x_pred).reshape(self.x_vec.shape)
                ax1.plot(t, x_np[0, :], 'r')
                ax1.plot(t, x_np[1, :], 'g')
                ax1.legend(["x1", "x2", "x*1", "x*2"])
            else:
                ax1.legend(["x1", "x2"])
            ax1.set_ylabel("Position error [m]")

            ax2.clear()
            ax2.plot(t, e_vec[2, :], 'r--')
            ax2.legend(["x5", "x6"])
            ax2.set_ylabel("Attitude error [rad]")

            ax3.clear()
            ax3.plot(t[:-1], self.u_vec[0, :], 'r--')
            ax3.plot(t[:-1], self.u_vec[1, :], 'g--')
            ax3.plot(t[:-1], self.u_vec[2, :], 'b--')
            ax3.legend(["u1", "u2", "u3"], loc='upper left')
            ax3.set_ylabel("Force input [n] / Torque [nm]")
        else:
            ax1.clear()
            ax1.set_title("Astrobee")
            ax1.plot(t, e_vec[0, :], 'r--')
            ax1.plot(t, e_vec[1, :], 'g--')
            ax1.plot(t, e_vec[2, :], 'b--')
            ax1.legend(["x1", "x2", "x3"])
            ax1.set_ylabel("Position error [m]")

            ax2.clear()
            ax2.plot(t, e_vec[3, :], 'r--')
            ax2.legend(["x5", "x6"])
            ax2.set_ylabel("Attitude error [rad] ")

            ax3.clear()
            ax3.plot(t[:-1], self.u_vec[0, :], 'r--')
            ax3.plot(t[:-1], self.u_vec[1, :], 'g--')
            ax3.plot(t[:-1], self.u_vec[2, :], 'b--')
            ax3.plot(t[:-1], self.u_vec[3, :], 'k--')
            ax3.legend(["u1", "u2", "u3", "u4"], loc='upper left')
            ax3.set_ylabel("Force input [n] / Torque [nm]")

        plt.show()

    def visualize_prediction_vs_reference(self, x_pred, x_ref, control):
        fig, (ax1, ax2, ax3) = plt.subplots(3)
        variables = list([self.t])
        
        
        x_ref = np.hstack([state for state in x_ref.values()])
        
        if any(elem is None for elem in variables):
            print("Please run the simulation first with the method 'run'.")

        t = self.t
        x_pred = np.asarray(x_pred).T.reshape(6, 301)
        u = np.asarray(control).T.reshape(3, 300)
        ax1.clear()
        ax1.set_title("Prediction vs Reference")
        ax1.plot(t, x_pred[0, :], 'r')
        ax1.plot(t, x_pred[1, :], 'g')
        ax1.plot(t[:-1], x_ref[0, :], 'r--')
        ax1.plot(t[:-1], x_ref[1, :], 'g--')
        ax1.set_ylabel("Position [m]")

        ax2.clear()
        ax2.plot(t, x_pred[4, :], 'r')
        ax2.plot(t[:-1], x_ref[2, :], 'r--')
        ax2.legend(["x5", "x6"])
        ax2.set_ylabel("Attitude [rad] ")

        ax3.clear()
        ax3.plot(t[:-1], u[0, :], 'r')
        ax3.plot(t[:-1], u[1, :], 'g')
        ax3.plot(t[:-1], u[2, :], 'b')
        ax3.legend(["Fx", "Fy", "Fz"], loc='upper left')
        ax3.set_ylabel("Control")
        plt.show()
        return

    def visualize_state_vs_reference(self, state, ref, control):
        fig, (ax1, ax2, ax3) = plt.subplots(3)
        variables = list([self.t])
        if any(elem is None for elem in variables):
            print("Please run the simulation first with the method 'run'.")
        
        ref = np.hstack([state for state in ref.values()])
        t = self.t
        u = np.asarray(control).T.reshape(3, 300)
        ax1.clear()
        ax1.set_title("State vs Reference")
        ax1.plot(t, state[0, :], 'r')
        ax1.plot(t, state[1, :], 'g')
        ax1.plot(t[:-1], ref[0, :], 'r--')
        ax1.plot(t[:-1], ref[1, :], 'g--')

        ax1.set_ylabel("Position [m]")

        ax2.clear()
        ax2.plot(t, state[4, :], 'r')
        ax2.plot(t[:-1], ref[2, :], 'r--')
        ax2.legend(["x5", "x6"])
        ax2.set_ylabel("Attitude [rad] ")

        ax3.clear()
        ax3.plot(t[:-1], u[0, :], 'r')
        ax3.plot(t[:-1], u[1, :], 'g')
        ax3.plot(t[:-1], u[2, :], 'b')
        ax3.legend(["Fx", "Fy", "Fz"], loc='upper left')
        ax3.set_ylabel("Control")
        plt.show()
        return

    def set_window(self, window):
        """
        Set the plot window length, in seconds.

        :param window: window length [s]
        :type window: float
        """
        self.plt_window = window

    def broken_thruster(self, act=True):
        """
        Simulate Astrobee broken thruster.

        :param act: activate or deactivate broken thruster
        :type act: boolean
        """

        self.broke_thruster = True
