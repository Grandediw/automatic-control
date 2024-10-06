# Automatic Control Repository

## Overview

This repository is dedicated to projects and solutions related to automatic control systems, featuring a variety of control strategies, simulations, and theoretical analyses. The repository contains MATLAB scripts, simulations, and reports focusing on system dynamics, stability, and control designs for different types of systems, including but not limited to PID controllers, state-space models, and advanced nonlinear control methods.

## Repository Structure

- **`src/`**: Contains the source code and control system implementations.
  - MATLAB files and functions used to simulate different control systems.
  
- **`docs/`**: Documentation and reports explaining the various control methods used in the projects.
  - Reports and notes on control theory, system dynamics, and the design process.

- **`simulations/`**: Simulations demonstrating the behavior of systems under different control strategies.
  - MATLAB/Simulink models for automatic control systems and their simulations.

- **`projects/`**: Contains specific control projects, each focusing on a particular control problem (e.g., rotary inverted pendulum, motor control).
  
- **`README.md`**: Overview of the repository and general instructions.

## Projects

### Rotary Inverted Pendulum

This project demonstrates the control of a rotary inverted pendulum using linear control techniques. The system is linearized, and two controllers are designed and evaluated through MATLAB simulations. The project files for the rotary pendulum are available in the **rotary-pendulum** branch.

### Future Projects

More projects involving control systems such as robotic arm control, temperature regulation, and motor control will be added to this repository in the future.

## Key Features

- **Control Theory Implementation**: Includes linear control techniques, state-space modeling, and nonlinear dynamics control.
- **Simulation Models**: MATLAB/Simulink models for visualizing system responses and control performance.
- **Documentation**: Detailed reports and notes covering system dynamics, control design, and analysis.
  
## Getting Started

### Prerequisites

To run the simulations and code in this repository, ensure you have the following installed:

- **MATLAB**: Required to execute MATLAB scripts and run simulations.
- **Control Systems Toolbox**: Needed for control design and analysis.
- **Simulink** (Optional): For running Simulink models.

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Grandediw/automatic-control.git
   cd automatic-control
   ```

2. Open MATLAB and navigate to the cloned directory.

## Contributing

Contributions to this repository are welcome. If you have improvements, optimizations, or additional control projects, feel free to open a pull request or raise an issue.

### How to Contribute

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes.
4. Commit your changes (`git commit -m 'Add new feature'`).
5. Push to the branch (`git push origin feature-branch`).
6. Open a pull request.

## License

This repository is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
