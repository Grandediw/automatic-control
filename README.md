# Automatic Control Project: Rotary Inverted Pendulum

## Overview

This repository contains the project files, simulations, and documentation for controlling a rotary inverted pendulum using linear control techniques. The project involves system linearization, stability analysis, and controller design. The primary goal is to stabilize the pendulum at zero position and velocity using two different controllers and evaluating their performance.

## Repository Structure

- **`StefanoTonini248413_Report.pdf`**: A detailed report explaining the system description, linearization process, controller design, stability analysis, and simulation results.
  
- **`StefanoTonini248413_Project.m`**: MATLAB script containing the control system design and simulation for the rotary inverted pendulum.
  
- **`f_NLDyna.m`**: MATLAB function file implementing the nonlinear dynamics of the rotary inverted pendulum system.

## Project Details

### 1. System Description

The rotary inverted pendulum system consists of a pendulum attached to a rotary arm driven by a motor. The nonlinear dynamics of the system are described using the following equation:

\[ M(q)\ddot{q} + C(q, \dot{q})\dot{q} + f_v(\dot{q}) + G(q) = \tau \]

The system is linearized around the origin to design the controllers.

### 2. Linearization and Stability Analysis

The system is linearized around the equilibrium point, and its stability is analyzed using eigenvalue analysis. The system is found to be unstable, but controllable, as indicated by the full-rank controllability matrix.

### 3. Controller Design

Two controllers, **K1** and **K2**, are designed to achieve stability:

- **K1**: Achieves a convergence rate of 2 with a focus on performance.
- **K2**: Minimizes the control effort required while maintaining the same convergence rate as **K1**.

Both controllers are evaluated through simulations, and the results are compared to determine which controller provides better performance with lower energy consumption.

### 4. Simulation and Results

MATLAB simulations are performed to evaluate the closed-loop performance of both controllers. Results are plotted for the pendulum's angular displacement and velocity, with both controllers successfully stabilizing the system.

## Getting Started

### Prerequisites

To run the MATLAB scripts in this repository, you need:

- **MATLAB**: For executing the simulation scripts and analyzing the system.
- **Control Systems Toolbox**: For designing and analyzing control systems.

### Running the Simulations

1. Open **MATLAB** and navigate to the directory where the files are located.
2. Run `StefanoTonini248413_Project.m` to simulate the control of the rotary inverted pendulum using the designed controllers.

### Simulation Files

- **`f_NLDyna.m`**: Defines the nonlinear dynamics of the rotary inverted pendulum.
- **`StefanoTonini_Project.m`**: Simulates the system's behavior under the two controllers **K1** and **K2**.

## Conclusion

The project successfully demonstrates the application of linear control techniques for stabilizing a rotary inverted pendulum. The controller **K2** was found to be more efficient due to its lower control effort, making it more suitable for real-world applications.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
