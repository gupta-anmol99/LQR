import numpy as np
from scipy import linalg


def get_k(m_cart, m_bob, g, l):
    m = m_bob + m_cart
    a = 1 / (l * (4.0 / 3 - m_bob / m))

    A = np.array([[0, 1, 0, 0],
                  [0, 0, g * a, 0],
                  [0, 0, 0, 1],
                  [0, 0, g * a, 0]])

    B = np.array([[0], [1 / m], [0], [-a]])

    R = np.eye(1, dtype=int)  # choose R (weight for input)
    Q = 5 * np.eye(4, dtype=int)  # choose Q (weight for state)

    P = linalg.solve_continuous_are(A, B, Q, R)

    K = np.dot(np.linalg.inv(R),
               np.dot(B.T, P))

    return K
