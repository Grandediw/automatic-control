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
        self.cvg_t = None  # Initialize convergence time

    def run(self, x0):
        """
        Run simulator with specified system dynamics and control function.
        """

        print("Running simulation....")
        sim_loop_length = int(self.total_sim_time / self.dt) + 1  # account for 0th
        t = np.array([0])
        x_vec = np.array([x0]).reshape(self.model.n, 1)
        u_vec = np.empty((6, 0))
        e_vec = np.empty((12, 0))

        for i in range(sim_loop_length):

            # Get control input and obtain next state
            x = x_vec[:, -1].reshape(self.model.n, 1)
            u, error = self.controller(x, i * self.dt)
            x_next = self.dynamics(x, u)
            x_next[6:10] = x_next[6:10] / ca.norm_2(x_next[6:10])

            # Store data
            t = np.append(t, t[-1] + self.dt)
            x_vec = np.append(x_vec, np.array(x_next).reshape(self.model.n, 1), axis=1)
            u_vec = np.append(u_vec, np.array(u).reshape(self.model.m, 1), axis=1)
            e_vec = np.append(e_vec, error.reshape(12, 1), axis=1)

            # Controllo di convergenza solo se cvg_t non è ancora stato impostato
            if self.cvg_t is None:
                # Estrai gli errori di posizione e atteggiamento
                position_error = e_vec[0:3, -1]  # ex, ey, ez in metri
                attitude_error = e_vec[6:9, -1]  # e_roll, e_pitch, e_yaw in radianti

                # Definisci le soglie
                pos_threshold = 0.05  # 5 cm
                att_threshold = np.deg2rad(10)  # 10 gradi in radianti

                # Verifica se tutti gli errori di posizione sono entro 5 cm
                pos_converged = np.all(np.abs(position_error) <= pos_threshold)

                # Verifica se tutti gli errori di atteggiamento sono entro 10 gradi
                att_converged = np.all(np.abs(attitude_error) <= att_threshold)

                if pos_converged and att_converged:
                    self.cvg_t = t[-1]
                  #  print(f"Convergenza raggiunta a t = {self.cvg_t:.2f} secondi.")
                    # Opzionalmente, puoi terminare la simulazione qui
                    # break

        # Dopo il ciclo, verifica se la convergenza è stata raggiunta
        if self.cvg_t is not None:
            print(f"Convergenza raggiunta a t = {self.cvg_t:.2f} secondi.")
        else:
            print("I criteri di convergenza non sono stati soddisfatti entro il tempo di simulazione.")


        _, error = self.controller(x_next, i * self.dt)
        e_vec = np.append(e_vec, error.reshape(12, 1), axis=1)
        self.t = t
        self.x_vec = x_vec
        self.u_vec = u_vec
        self.e_vec = e_vec
        self.sim_loop_length = sim_loop_length
        ss_p = self.e_vec[0:3, -1]
        self.ss_p = np.linalg.norm(ss_p)
        print(f"Steady-State Position Error (ss_p): {ss_p}")    
        ss_a = self.e_vec[6:9, -1]
        self.ss_a = np.linalg.norm(ss_a)
        print(f"Steady-State Position Error (ss_a): {ss_a}")   
        return t, x_vec, u_vec#, self.cvg_t, self.ss_p, self.ss_a
    

    def visualize(self):
        """
        Offline plotting of simulation data
        """
        variables = [self.t, self.x_vec, self.u_vec, self.sim_loop_length]
        if any(elem is None for elem in variables):
            print("Please run the simulation first with the method 'run'.")

        t = self.t
        x_vec = self.x_vec
        u_vec = self.u_vec

        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 15))
        fig2, (ax5, ax6) = plt.subplots(2, 1, figsize=(10, 10))

        # Plot States
        ax1.clear()
        ax1.set_title("Astrobee States - Position")
        ax1.plot(t, x_vec[0, :], 'r--',
                 t, x_vec[1, :], 'g--',
                 t, x_vec[2, :], 'b--')
        # if self.cvg_t is not None:
        #     ax1.axvline(self.cvg_t, color='k', linestyle=':', label='Convergenza')
        #     ax1.legend(["x1", "x2", "x3", "Convergenza"])
        # else:
        ax1.legend(["x1", "x2", "x3"])
        ax1.set_ylabel("Position [m]")
        ax1.grid()

        ax2.clear()
        ax2.set_title("Astrobee States - Velocity")
        ax2.plot(t, x_vec[3, :], 'r--',
                 t, x_vec[4, :], 'g--',
                 t, x_vec[5, :], 'b--')
        ax2.legend(["x4", "x5", "x6"])
        ax2.set_ylabel("Velocity [m/s]")
        ax2.grid()

        ax3.clear()
        ax3.set_title("Astrobee States - Attitude")
        ax3.plot(t, x_vec[6, :], 'r--',
                 t, x_vec[7, :], 'g--',
                 t, x_vec[8, :], 'b--')
        ax3.legend(["x7", "x8", "x9"])
        ax3.set_ylabel("Attitude [rad]")
        ax3.grid()

        ax4.clear()
        ax4.set_title("Astrobee States - Angular Velocity")
        ax4.plot(t, x_vec[10, :], 'r--',
                 t, x_vec[11, :], 'g--',
                 t, x_vec[12, :], 'b--')
        ax4.legend(["x11", "x12", "x13"])
        ax4.set_ylabel("Ang. velocity [rad/s]")
        ax4.grid()

        # Plot Control Inputs
        ax5.clear()
        ax5.set_title("Astrobee Control Inputs - Forces")
        ax5.plot(t[:-1], u_vec[0, :], 'r--',
                 t[:-1], u_vec[1, :], 'g--',
                 t[:-1], u_vec[2, :], 'b--')
        ax5.legend(["u0", "u1", "u2"])
        ax5.set_ylabel("Force input [N]")
        ax5.grid()

        ax6.clear()
        ax6.set_title("Astrobee Control Inputs - Torques")
        ax6.plot(t[:-1], u_vec[3, :], 'r--',
                 t[:-1], u_vec[4, :], 'g--',
                 t[:-1], u_vec[5, :], 'b--')
        ax6.legend(["u3", "u4", "u5"])
        ax6.set_ylabel("Torque input [Nm]")
        ax6.grid()

        plt.tight_layout()
        plt.show()

    def visualize_error(self):
        """
        Offline plotting of simulation error data
        """
        variables = [self.t, self.e_vec, self.u_vec, self.sim_loop_length]
        if any(elem is None for elem in variables):
            print("Please run the simulation first with the method 'run'.")

        t = self.t
        e_vec = self.e_vec
        u_vec = self.u_vec

        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 15))
        fig2, (ax5, ax6) = plt.subplots(2, 1, figsize=(10, 10))

        # Plot Errors
        ax1.clear()
        ax1.set_title("Trajectory Error - Position")
        ax1.plot(t, e_vec[0, :], 'r--',
                 t, e_vec[1, :], 'g--',
                 t, e_vec[2, :], 'b--')
        # if self.cvg_t is not None:
        #     ax1.axvline(self.cvg_t, color='k', linestyle=':', label='Convergenza')
        #     ax1.legend(["ex", "ey", "ez", "Convergenza"])
        # else:
        ax1.legend(["ex", "ey", "ez"])
        ax1.set_ylabel("Position Error [m]")
        ax1.grid()

        ax2.clear()
        ax2.set_title("Trajectory Error - Velocity")
        ax2.plot(t, e_vec[3, :], 'r--',
                 t, e_vec[4, :], 'g--',
                 t, e_vec[5, :], 'b--')
        ax2.legend(["evx", "evy", "evz"])
        ax2.set_ylabel("Velocity Error [m/s]")
        ax2.grid()

        ax3.clear()
        ax3.set_title("Trajectory Error - Attitude")
        ax3.plot(t, e_vec[6, :], 'r--',
                 t, e_vec[7, :], 'g--',
                 t, e_vec[8, :], 'b--')
        ax3.legend(["eroll", "epitch", "eyaw"])
        ax3.set_ylabel("Attitude Error [rad]")
        ax3.grid()

        ax4.clear()
        ax4.set_title("Trajectory Error - Angular Velocity")
        ax4.plot(t, e_vec[9, :], 'r--',
                 t, e_vec[10, :], 'g--',
                 t, e_vec[11, :], 'b--')
        ax4.legend(["eωx", "eωy", "eωz"])
        ax4.set_ylabel("Angular Velocity Error [rad/s]")
        ax4.grid()

        # Plot Control Inputs
        ax5.clear()
        ax5.set_title("Astrobee Control Inputs - Forces")
        ax5.plot(t[:-1], u_vec[0, :], 'r--',
                 t[:-1], u_vec[1, :], 'g--',
                 t[:-1], u_vec[2, :], 'b--')
        ax5.legend(["u0", "u1", "u2"])
        ax5.set_ylabel("Force input [N]")
        ax5.grid()

        ax6.clear()
        ax6.set_title("Astrobee Control Inputs - Torques")
        ax6.plot(t[:-1], u_vec[3, :], 'r--',
                 t[:-1], u_vec[4, :], 'g--',
                 t[:-1], u_vec[5, :], 'b--')
        ax6.legend(["u3", "u4", "u5"])
        ax6.set_ylabel("Torque input [Nm]")
        ax6.grid()

        plt.tight_layout()
        plt.show()


