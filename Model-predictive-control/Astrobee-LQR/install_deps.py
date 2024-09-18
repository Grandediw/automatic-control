import subprocess
import sys


def install(package):
    print(subprocess.check_call([sys.executable, "-m", "pip", "install", package]))


if __name__ == "__main__":
    install("matplotlib")
    install("casadi")
    install("numpy")
    install("control")
    install("filterpy")
    install("cvxpy")

    try:
        import matplotlib
    except Exception:
        print("ERROR: Matplotlib package not installed. Please install it manually for your environment")
        exit()

    try:
        import casadi
    except Exception:
        print("ERROR: CasADi package not installed. Please install it manually for your environment")
        exit()

    try:
        import numpy
    except Exception:
        print("ERROR: Numpy package not installed. Please install it manually for your environment")
        exit()

    try:
        import control
    except Exception:
        print("ERROR: Control package not installed. Please install it manually for your environment")
        exit()

    try:
        import filterpy
    except Exception:
        print("ERROR: Filterpy package not installed. Please install it manually for your environment")
        exit()
    
    try:
        import cvxpy
    except Exception:
        print("ERROR: Control package not installed. Please install it manually for your environment")
        exit()
