
import numpy as np
from scipy.optimize import brentq

from orbital import MU_MOON, orbital_state_at_time, DAY2SEC


def stumpff(z):
    if z > 1e-8:
        s = np.sqrt(z)
        C = (1 - np.cos(s)) / z
        S = (s - np.sin(s)) / s**3
    elif z < -1e-8:
        s = np.sqrt(-z)
        C = (np.cosh(s) - 1) / (-z)
        S = (np.sinh(s) - s) / s**3
    else:
        C = 0.5
        S = 1 / 6

    return C, S


def lambert_zero_rev(r1, r2, tof_seconds, long_way=False):
    """
    Lambert me zero revolucione.

    r1, r2: pozicionet ne metra
    tof_seconds: koha e fluturimit ne sekonda

    Kthen shpejtesite e transferimit ne nisje
    dhe ne mberritje (m/s).
    """

    r1 = np.asarray(r1, dtype=float)
    r2 = np.asarray(r2, dtype=float)

    R1 = np.linalg.norm(r1)
    R2 = np.linalg.norm(r2)

    cos_theta = np.clip(
        np.dot(r1, r2) / (R1 * R2), -1.0, 1.0
    )

    theta = np.arccos(cos_theta)

    if long_way:
        theta = 2 * np.pi - theta

    if abs(1 - cos_theta) < 1e-12:
        raise ValueError("Gjeometri e papershtatshme")

    A = np.sin(theta) * np.sqrt(
        R1 * R2 / (1 - cos_theta)
    )

    if abs(A) < 1e-12:
        raise ValueError("Transferim degjenerues")

    def calculate(z):
        C, S = stumpff(z)

        if C <= 0:
            return None

        y = R1 + R2 + A * (z * S - 1) / np.sqrt(C)

        if y <= 0:
            return None

        x = np.sqrt(y / C)

        time = (
            x**3 * S + A * np.sqrt(y)
        ) / np.sqrt(MU_MOON)

        return time, y

    def residual(z):
        result = calculate(z)

        if result is None:
            return np.nan

        return result[0] - tof_seconds

    # Kerkojme interval me ndryshim shenje
    z_values = np.linspace(-100, 4*np.pi**2 - 1e-5, 600)

    root = None
    previous_z = None
    previous_f = None

    for z in z_values:
        f = residual(z)

        if not np.isfinite(f):
            continue

        if previous_f is not None and f * previous_f < 0:
            root = brentq(residual, previous_z, z)
            break

        previous_z = z
        previous_f = f

    if root is None:
        raise ValueError("Nuk u gjet zgjidhje Lambert")

    _, y = calculate(root)

    f = 1 - y / R1
    g = A * np.sqrt(y / MU_MOON)
    gdot = 1 - y / R2

    if abs(g) < 1e-12:
        raise ValueError("Transferim i papershtatshem")

    v_depart = (r2 - f*r1) / g
    v_arrive = (gdot*r2 - r1) / g

    return v_depart, v_arrive


def compute_transfer(elements1, elements2, start_day, tof_days):
    """
    Llogarit Delta-V per dy domate.
    """

    if tof_days <= 0:
        raise ValueError("Koha e fluturimit duhet pozitive")

    r1, v1 = orbital_state_at_time(elements1, start_day)
    r2, v2 = orbital_state_at_time(
        elements2, start_day + tof_days
    )

    best_dv = float("inf")

    for long_way in [False, True]:
        try:
            vt1, vt2 = lambert_zero_rev(
                r1, r2, tof_days * DAY2SEC, long_way
            )

            dv_depart = np.linalg.norm(vt1 - v1)
            dv_arrive = np.linalg.norm(vt2 - v2)

            total_dv = dv_depart + dv_arrive

            best_dv = min(best_dv, total_dv)

        except (ValueError, OverflowError, FloatingPointError):
            continue

    return best_dv



def find_best_transfer(
    elements1,
    elements2,
    start_day,
    min_tof,
    max_tof,
    dv_limit,
    steps=50
):
    """
    Provon disa kohe fluturimi dhe gjen
    transferimin me Delta-V me te vogel.
    """

    best_tof = None
    best_dv = float("inf")

    times = np.linspace(min_tof, max_tof, steps)

    for tof in times:
        dv = compute_transfer(
            elements1,
            elements2,
            start_day,
            tof
        )

        if dv < best_dv:
            best_dv = dv
            best_tof = tof

    return best_tof, best_dv, best_dv <= dv_limit
