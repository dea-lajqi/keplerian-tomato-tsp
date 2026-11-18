
import numpy as np
from scipy.optimize import newton

# Konstanta gravitacionale e Henes
# Njesia: m^3/s^2
MU_MOON = 4.9048695e12

# Nje dite ka 86400 sekonda
DAY2SEC = 86400.0


def solve_kepler(M, e):
    """Zgjidh ekuacionin M = E - e*sin(E)."""

    def equation(E):
        return E - e * np.sin(E) - M

    def derivative(E):
        return 1 - e * np.cos(E)

    return newton(equation, M, fprime=derivative)


def orbital_state(a, e, inc, omega_node, omega_peri, M):
    """
    Kthen pozicionin (m) dhe shpejtesine (m/s).

    a: metra
    e: ekscentriciteti
    kendet: radiane
    """

    E = solve_kepler(M, e)

    x = a * (np.cos(E) - e)
    y = a * np.sqrt(1 - e**2) * np.sin(E)

    n = np.sqrt(MU_MOON / a**3)
    factor = a * n / (1 - e * np.cos(E))

    vx = -factor * np.sin(E)
    vy = factor * np.sqrt(1 - e**2) * np.cos(E)

    cO, sO = np.cos(omega_node), np.sin(omega_node)
    ci, si = np.cos(inc), np.sin(inc)
    cw, sw = np.cos(omega_peri), np.sin(omega_peri)

    rotation = np.array([
        [cO*cw - sO*sw*ci, -cO*sw - sO*cw*ci, sO*si],
        [sO*cw + cO*sw*ci, -sO*sw + cO*cw*ci, -cO*si],
        [sw*si, cw*si, ci]
    ])

    position = rotation @ np.array([x, y, 0.0])
    velocity = rotation @ np.array([vx, vy, 0.0])

    return position, velocity


def orbital_state_at_time(elements, time_days, t0_days=0.0):
    """
    Llogarit pozicionin dhe shpejtesine
    ne nje kohe te caktuar, ne dite.
    """

    a, e, inc, omega_node, omega_peri, M0 = elements

    n = np.sqrt(MU_MOON / a**3)

    elapsed_seconds = (time_days - t0_days) * DAY2SEC

    M = (M0 + n * elapsed_seconds) % (2 * np.pi)

    return orbital_state(
        a, e, inc, omega_node, omega_peri, M
    )


if __name__ == "__main__":
    E = solve_kepler(0.5, 0.1)
    print("Anomalia ekscentrike:", E)
