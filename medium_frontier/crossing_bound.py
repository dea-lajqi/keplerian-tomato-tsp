"""Epoch-independent angular-momentum bounds for the original Medium instance."""
from pathlib import Path
import json

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
INSTANCE = ROOT / "data/spoc4/kttsp/problems/medium.kttsp"
MU_MOON = 4.9048695e12


def crossing_bounds(instance=INSTANCE):
    lines = instance.read_text().splitlines()
    parameters = next(line.split() for line in lines if line.startswith("p "))
    normal, exceptional = map(float, parameters[-3:-1])
    elements = np.array([
        [float(value) for value in line.split()]
        for line in lines if line.strip() and not line.startswith(("c", "p"))
    ])
    if elements.shape != (181, 6):
        raise ValueError("Expected 181 Medium orbit elements")
    a, e, inclination, raan = elements[:, :4].T
    normals = np.column_stack((
        np.sin(inclination) * np.sin(raan),
        -np.sin(inclination) * np.cos(raan),
        np.cos(inclination),
    ))
    momentum = np.sqrt(MU_MOON * a * (1 - e**2))[:, None] * normals
    apocentre = a * (1 + e)
    bounds = np.linalg.norm(momentum[:, None] - momentum[None, :], axis=2)
    bounds /= np.maximum(apocentre[:, None], apocentre[None, :])

    low = np.flatnonzero(a < 8e6)
    high = np.flatnonzero(a >= 8e6)
    ordered_low = low[np.argsort(inclination[low])]
    groups = np.split(ordered_low, np.flatnonzero(np.diff(inclination[ordered_low]) > 0.5) + 1)
    low_high = float(bounds[np.ix_(low, high)].min())
    between = [
        float(bounds[np.ix_(left, right)].min())
        for i, left in enumerate(groups) for right in groups[i + 1:]
    ]
    if low_high <= normal or any(bound <= exceptional for bound in between):
        raise ValueError("The necessary crossing bound does not follow for this instance")
    return {
        "low_group_sizes": [len(group) for group in groups],
        "high_count": len(high),
        "low_to_high_minimum_m_per_s": low_high,
        "between_low_groups_minima_m_per_s": between,
        "necessary_exceptional_crossings": 2 * (len(groups) - 1),
    }


if __name__ == "__main__":
    print(json.dumps(crossing_bounds(), indent=2))
