import subprocess
import sys


def install(package):
    print(subprocess.check_call([sys.executable, "-m", "pip", "install", package]))


if __name__ == "__main__":
    install("cvxopt==1.2.7")
    install("scipy==1.7.1")
    install("matplotlib==3.4.3")
    install("casadi==3.5.5")
    install("numpy==1.21.2")
    install("control==0.9.0")
    install("filterpy==1.4.5")
    install("polytope==0.2.3")
    install("yaml")
    install("cvxpy")

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
    
    try :
        import cvxpy
    except Exception:
        print("ERROR: cvxpy package not installed. Please install it manually for your environment")
        exit()