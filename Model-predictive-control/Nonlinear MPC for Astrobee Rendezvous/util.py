"""
Set of utility functions used across the package.
"""

import casadi as ca
import numpy as np


def skew(v):
    """
    Returns the skew matrix of a vector v

    :param v: vector
    :type v: ca.MX
    :return: skew matrix of v
    :rtype: ca.MX
    """

    sk = ca.MX.zeros(3, 3)

    # Extract vector components
    x = v[0]
    y = v[1]
    z = v[2]

    sk[0, 1] = -z
    sk[1, 0] = z
    sk[0, 2] = y
    sk[2, 0] = -y
    sk[1, 2] = -x
    sk[2, 1] = x

    return sk


def xi_mat(q):
    """
    Generate the matrix for quaternion dynamics Xi,
    from Trawney's Quaternion tutorial.
    :param q: unit quaternion
    :type q: ca.MX
    :return: Xi matrix
    :rtype: ca.MX
    """
    Xi = ca.MX(4, 3)

    # Extract states
    qx = q[0]
    qy = q[1]
    qz = q[2]
    qw = q[3]

    # Generate Xi matrix
    Xi[0, 0] = qw
    Xi[0, 1] = -qz
    Xi[0, 2] = qy

    Xi[1, 0] = qz
    Xi[1, 1] = qw
    Xi[1, 2] = -qx

    Xi[2, 0] = -qy
    Xi[2, 1] = qx
    Xi[2, 2] = qw

    Xi[3, 0] = -qx
    Xi[3, 1] = -qy
    Xi[3, 2] = -qz

    return Xi


def r_mat_q(q):
    """
    Generate rotation matrix from unit quaternion
    :param q: unit quaternion
    :type q: ca.MX
    :return: rotation matrix, SO(3)
    :rtype: ca.MX
    """

    Rmat = ca.MX(3, 3)

    # Extract states
    qx = q[0]
    qy = q[1]
    qz = q[2]
    qw = q[3]

    Rmat[0, 0] = 1 - 2 * qy**2 - 2 * qz**2
    Rmat[0, 1] = 2 * qx * qy - 2 * qz * qw
    Rmat[0, 2] = 2 * qx * qz + 2 * qy * qw

    Rmat[1, 0] = 2 * qx * qy + 2 * qz * qw
    Rmat[1, 1] = 1 - 2 * qx**2 - 2 * qz**2
    Rmat[1, 2] = 2 * qy * qz - 2 * qx * qw

    Rmat[2, 0] = 2 * qx * qz - 2 * qy * qw
    Rmat[2, 1] = 2 * qy * qz + 2 * qx * qw
    Rmat[2, 2] = 1 - 2 * qx**2 - 2 * qy**2

    return Rmat


def r_mat_q_np(q):
    """
    Generate rotation matrix from unit quaternion
    :param q: unit quaternion
    :type q: ca.MX
    :return: rotation matrix, SO(3)
    :rtype: ca.MX
    """

    Rmat = np.zeros((3, 3))

    # Extract states
    qx = q[0]
    qy = q[1]
    qz = q[2]
    qw = q[3]

    Rmat[0, 0] = 1 - 2 * qy**2 - 2 * qz**2
    Rmat[0, 1] = 2 * qx * qy - 2 * qz * qw
    Rmat[0, 2] = 2 * qx * qz + 2 * qy * qw

    Rmat[1, 0] = 2 * qx * qy + 2 * qz * qw
    Rmat[1, 1] = 1 - 2 * qx**2 - 2 * qz**2
    Rmat[1, 2] = 2 * qy * qz - 2 * qx * qw

    Rmat[2, 0] = 2 * qx * qz - 2 * qy * qw
    Rmat[2, 1] = 2 * qy * qz + 2 * qx * qw
    Rmat[2, 2] = 1 - 2 * qx**2 - 2 * qy**2

    return Rmat


def inv_skew(sk):
    """
    Retrieve the vector from the skew-symmetric matrix.
    :param sk: skew symmetric matrix
    :type sk: ca.MX
    :return: vector corresponding to SK matrix
    :rtype: ca.MX
    """

    v = ca.MX.zeros(3, 1)

    v[0] = sk[2, 1]
    v[1] = sk[0, 2]
    v[2] = sk[1, 0]

    return v


def inv_skew_np(sk):
    """
    Retrieve the vector from the skew-symmetric matrix.
    :param sk: skew symmetric matrix
    :type sk: ca.MX
    :return: vector corresponding to SK matrix
    :rtype: ca.MX
    """

    v = np.zeros((3, 1))

    v[0] = sk[2, 1]
    v[1] = sk[0, 2]
    v[2] = sk[1, 0]

    return v


def xi_mat_np(q):
    """
    Generate the matrix for quaternion dynamics Xi,
    from Trawney's Quaternion tutorial.
    :param q: unit quaternion
    :type q: ca.MX
    :return: Xi matrix
    :rtype: ca.MX
    """
    Xi = np.zeros((4, 3))

    # Extract states
    qx = q[0]
    qy = q[1]
    qz = q[2]
    qw = q[3]

    # Generate Xi matrix
    Xi[0, 0] = qw
    Xi[0, 1] = -qz
    Xi[0, 2] = qy

    Xi[1, 0] = qz
    Xi[1, 1] = qw
    Xi[1, 2] = -qx

    Xi[2, 0] = -qy
    Xi[2, 1] = qx
    Xi[2, 2] = qw

    Xi[3, 0] = -qx
    Xi[3, 1] = -qy
    Xi[3, 2] = -qz

    return Xi
