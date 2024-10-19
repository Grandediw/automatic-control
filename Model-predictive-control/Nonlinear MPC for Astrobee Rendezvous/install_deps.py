import subprocess
import sys


def install(package):
    print(subprocess.check_call([sys.executable, "-m", "pip", "install", package]))


if __name__ == "__main__":
    install("cvxopt")
    install("scipy")
    install("matplotlib")
    install("casadi")
    install("numpy")
    install("control")
    install("filterpy")
    install("polytope")
    install("pyyaml")

    try:
        import matplotlib
    except Exception:
        print("ERROR: matplotlib package not installed. Please install it manually for your environment")
        exit()

    try:
        import casadi
    except Exception:
        print("ERROR: CasADi package not installed. Please install it manually for your environment")
        exit()

    try:
        import numpy
    except Exception:
        print("ERROR: numpy package not installed. Please install it manually for your environment")
        exit()

    try:
        import control
    except Exception:
        print("ERROR: control package not installed. Please install it manually for your environment")
        exit()

    try:
        import filterpy
    except Exception:
        print("ERROR: filterpy package not installed. Please install it manually for your environment")
        exit()

    try:
        import polytope
    except Exception:
        print("ERROR: polytope package not installed. Please install it manually for your environment")
        exit()

    try:
        import scipy
    except Exception:
        print("ERROR: scipy package not installed. Please install it manually for your environment")
        exit()

    try:
        import cvxopt
    except Exception:
        print("ERROR: cvxopt package not installed. Please install it manually for your environment")
        exit()

    try:
        import yaml
    except Exception:
        print("ERROR: cvxopt package not installed. Please install it manually for your environment")
        exit()
