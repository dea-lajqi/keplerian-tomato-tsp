
import numpy as np
from scipy.optimize import newton

# Konstanta gravitacionale e Henes (m^3/s^2)
MU_MOON = 4.9048695e12

# Nje dite = 86400 sekonda
DAY2SEC = 86400.0


# 1. ZGJIDHJA E EKUACIONIT TE KEPLERIT

def solve_kepler(M, e):
    """
    Zgjidh ekuacionin:
    M = E - e * sin(E)
    """

    def equation(E):
        return E - e * np.sin(E) - M

    def derivative(E):
        return 1 - e * np.cos(E)

    return newton(equation, M, fprime=derivative)


# 2. POZICIONI DHE SHPEJTESIA ORBITALE

def orbital_state(a, e, inc, omega_node, omega_peri, M):
    """
    Kthen pozicionin (m) dhe shpejtesine (m/s).

    a: gjysmeboshti i madh ne metra
    e: ekscentriciteti
    inc: inklinimi ne radiane
    omega_node: RAAN ne radiane
    omega_peri: argumenti i periapsisit ne radiane
    M: anomalia mesatare ne radiane
    """

    # Gjejme anomaline ekscentrike
    E = solve_kepler(M, e)

    # Pozicioni ne planin orbital
    x = a * (np.cos(E) - e)
    y = a * np.sqrt(1 - e**2) * np.sin(E)

    # Shpejtesia ne planin orbital
    n = np.sqrt(MU_MOON / a**3)
    factor = a * n / (1 - e * np.cos(E))

    vx = -factor * np.sin(E)
    vy = factor * np.sqrt(1 - e**2) * np.cos(E)

    # Matrica e rrotullimit
    cO, sO = np.cos(omega_node), np.sin(omega_node)
    ci, si = np.cos(inc), np.sin(inc)
    cw, sw = np.cos(omega_peri), np.sin(omega_peri)

    rotation = np.array([
        [
            cO*cw - sO*sw*ci,
            -cO*sw - sO*cw*ci,
            sO*si
        ],
        [
            sO*cw + cO*sw*ci,
            -sO*sw + cO*cw*ci,
            -cO*si
        ],
        [
            sw*si,
            cw*si,
            ci
        ]
    ])

    position = rotation @ np.array([x, y, 0.0])
    velocity = rotation @ np.array([vx, vy, 0.0])

    return position, velocity


# 3. GJENDJA ORBITALE NE NJE KOHE TE CAKTUAR

def orbital_state_at_time(elements, time_days, t0_days=0.0):
    """
    Llogarit pozicionin dhe shpejtesine
    e nje domateje ne nje kohe te caktuar.

    elements: 6 elementet orbitale
    time_days: koha ne dite
    """

    a, e, inc, omega_node, omega_peri, M0 = elements

    # Levizja mesatare orbitale
    n = np.sqrt(MU_MOON / a**3)

    # Koha e kaluar ne sekonda
    elapsed_seconds = (time_days - t0_days) * DAY2SEC

    # Anomalia mesatare ne kohen e kerkuar
    M = (M0 + n * elapsed_seconds) % (2 * np.pi)

    return orbital_state(
        a, e, inc, omega_node, omega_peri, M
    )


# 4. TEST I THJESHTE

if __name__ == "__main__":
    E = solve_kepler(0.5, 0.1)
    print("Anomalia ekscentrike:", E)



def orbital_state_corrected(elements, time_days, t0_days=0.0):
    """
    Elementi i gjashte eshte true anomaly (nu).
    E konverton ne mean anomaly (M0).
    """
    a, e, inc, Omega, omega, nu = elements

    E_anomaly = 2 * np.arctan2(
        np.sqrt(1 - e) * np.sin(nu / 2),
        np.sqrt(1 + e) * np.cos(nu / 2)
    )

    M0 = E_anomaly - e * np.sin(E_anomaly)

    corrected_elements = [
        a, e, inc, Omega, omega, M0
    ]

    return orbital_state_at_time(
        corrected_elements, time_days, t0_days
    )
