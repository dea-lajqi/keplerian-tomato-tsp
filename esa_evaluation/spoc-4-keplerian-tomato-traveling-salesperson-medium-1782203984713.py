from typing import Union
from pathlib import Path

import numpy as np
import pykep as pk

### Constants
MU_MOON = 4.904869500000000e12  # Gravitational parameter of the moon in m^3/s^2 (G*M_moon)
R_MOON = 1737.4e3  # Radius of the moon in meters


class KTTSP:
    """
    User-defined problem (UDP) class for the Keplerian Tomato Traveling Salesperson Problem (KTTSP).

    The formal specification of the KTTSP can be found on its page on the Optimise platform:

    https://optimize.esa.int/

    This class conforms to the pygmo UDP format.
    """

    def __init__(self, file_path: str, max_revs: int = 20):
        """
        This class is initialzied with a file path to a KTTSP instance, 
        containing orbital paramaters of the tomatoes.
        
        
        Args:
            file_path (Union[Path, str]): The path to the KTTSP instance file.
        """
        self.file_path = self._get_file_path(file_path)
        self.max_revs = max_revs
        self.t0 = None
        self.tomato_orbital_parameters = np.empty((0, 6), dtype=float)
        self._load_problem()

    def _load_problem(self):
        """Load a KTTSP instance from disk and initialize all dependent attributes."""
        if not self.file_path.exists():
            raise FileNotFoundError(f"KTTSP instance file not found: {self.file_path}")

        header = None
        orbital_rows = []
        with self.file_path.open("r", encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("c"):
                    continue
                if line.startswith("p "):
                    parts = line.split()
                    if len(parts) != 8 or parts[1].lower() != "kttsp":
                        raise ValueError(f"Malformed KTTSP header line: {line}")
                    header = parts
                    continue

                parts = line.split()
                if len(parts) != 6:
                    raise ValueError(f"Expected 6 orbital parameters per tomato, got: {line}")
                orbital_rows.append([float(value) for value in parts])

        if header is None:
            raise ValueError(f"Missing 'p kttsp' header in file: {self.file_path}")
        if not orbital_rows:
            raise ValueError(f"No orbital parameter rows found in file: {self.file_path}")

        # Header format: p kttsp t0[d] min_tof[d] max_total_time[d] dv_threshold[m/s] dv_threshold_exception[m/s]
        self.t0 = pk.epoch(float(header[2])) 
        self.min_tof = float(header[3])
        self.max_time = float(header[4])
        self.dv_threshold = float(header[5])
        self.dv_threshold_exception = float(header[6])
        self.n_dv_exception = int(header[7])

        self.tomato_orbital_parameters = np.asarray(orbital_rows, dtype=float)
        self.n_tomatoes = self.tomato_orbital_parameters.shape[0]
        self.tomatoes = []
        for row in self.tomato_orbital_parameters:
            a, e, incl, raan, arg_periapsis, true_anomaly = row
            elements = [a, e, incl, raan, arg_periapsis, true_anomaly]
            tomato = pk.planet.keplerian(self.t0, elements, MU_MOON, 0.0, 0.0, 0.0)
            self.tomatoes.append(tomato)

    def _get_file_path(
        self,
        path: Path,
    ) -> Path:
        """
        A convenience function that searches for a graph file
        in a hierarchy of directories.

        Args:
            path (Path):
                The (relative) file path.

        Raises:
            FileNotFoundError:
                Raises an error if the file does not exist.

        Returns:
            Path:
                A filesystem path.
        """

        fpath = str(path)
        root = Path(__file__)
        for _ in range(len(root.parents)):
            root = root.parent.resolve().absolute()
            path = root / fpath

            if path.exists():
                break

        if not path.exists():
            raise FileNotFoundError(f"File '{path}' does not exist.")

        return Path(path)
    
    def _read_solution(self, file_path: Union[Path, str]):
        """Read a .sol solution file and return a chromosome compatible with ``fitness``."""
        solution_path = self._get_file_path(file_path)
        if not solution_path.exists():
            raise FileNotFoundError(f"KTTSP solution file not found: {solution_path}")

        raw_values = []
        with solution_path.open("r", encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("c"):
                    continue
                for token in line.split():
                    raw_values.append(float(token))

        expected_len = 3 * self.n_tomatoes - 2
        if len(raw_values) != expected_len:
            raise ValueError(
                f"Malformed solution file: expected {expected_len} values for "
                f"n_tomatoes={self.n_tomatoes}, got {len(raw_values)}"
            )

        split_a = self.n_tomatoes - 1
        split_b = 2 * self.n_tomatoes - 2

        times = raw_values[:split_a]
        tofs = raw_values[split_a:split_b]
        order_values = raw_values[split_b:]

        order = []
        for value in order_values:
            idx = int(round(value))
            if abs(value - idx) > 1e-9:
                raise ValueError(f"Order value is not an integer: {value}")
            if idx < 0 or idx >= self.n_tomatoes:
                raise ValueError(
                    f"Order value out of range [0, {self.n_tomatoes - 1}]: {idx}"
                )
            order.append(idx)

        return times + tofs + order

    def get_bounds(self):
        """
        Bounds for all chromosome elements.

        Returns:
            Tuple[list]: Bound constraint violations for each element in the chromosome.
        """
        # The decision variables are the departure times, the times of flight and the order of tomatoes to visit
        lb = [0] * (self.n_tomatoes - 1) + [self.min_tof] * (self.n_tomatoes - 1) + [0] * self.n_tomatoes
        ub = [self.max_time] * (self.n_tomatoes - 1) + [self.max_time] * (self.n_tomatoes - 1) + [self.n_tomatoes - 1] * self.n_tomatoes
        return (lb, ub)

    def fitness(self, x):
        """
        Computes the fitness of the provided chromosome x, which represents a candidate solution to the KTTSP.
        Corresponds to the total time taken to visit all tomatoes in the order specified by x.

        Args:
            x: A chromosome representing a candidate solution to the KTTSP of shape (3 * n_tomatoes - 2,). 
                The first n_tomatoes-1 entries represent the departure times, the next n_tomatoes-1 entries represent 
                the times of flight, and the final n_tomatoes entries represent the order of tomatoes to visit.

        Returns:
            list: An list containing the total time taken to visit all tomatoes, the permutation 
                constraint violation, the delta v constraint violations, and the time constraint violations.
        """
        # Unpack chromosome x into the times of departure, the times of flight and the order of tomatoes to visit
        times = x[:self.n_tomatoes - 1]
        tofs = x[self.n_tomatoes - 1:2 * self.n_tomatoes - 2]
        order = x[2 * self.n_tomatoes - 2:]
        # Permutation constraint should be 0 if all tomatoes are visited exactly once
        permutation_constraint = len(set(map(int, order))) - self.n_tomatoes
        # dv/time constraints should be smaller than 0
        time_counter = 0
        dv_counter = 0
        dv_exception_counter = 0
        dvs = []
        for i in range(self.n_tomatoes - 1):
            dv = self.compute_transfer(int(order[i]), int(order[i + 1]), times[i], tofs[i])
            dvs.append(dv)
            if dv <= self.dv_threshold_exception + 1e-6:
                dv_counter += 1
            if self.dv_threshold < dv <= self.dv_threshold_exception + 1e-6:
                dv_exception_counter += 1
            if i < self.n_tomatoes - 2 and times[i] + tofs[i] <= (times[i + 1] + 1e-6):
                time_counter += 1
        """print(self.dv_threshold_exception, self.dv_threshold)
        print(np.array(dvs)[:20] )
        print(np.array(dvs)[20:40])
        print(np.array(dvs)[40:60])
        print((np.array(dvs)[:20] > self.dv_threshold_exception).sum())
        print((np.array(dvs)[20:40] > self.dv_threshold_exception).sum())
        print((np.array(dvs)[40:60] > self.dv_threshold_exception).sum())"""
        total_time = times[-1] + tofs[-1]
        dv_constraint = dv_counter - (self.n_tomatoes - 1)  # Should be 0 if all legs are below the dv threshold
        # Should be smaller than 0 if no leg is above the exception threshold
        dv_exception_constraint = dv_exception_counter - self.n_dv_exception
        time_constraint = time_counter - (self.n_tomatoes - 2)  # Should be 0 if all departure times are before the next departure time
        # return [total_time, permutation_constraint] + dv_constraints + time_constraints
        return [total_time, permutation_constraint, dv_constraint, time_constraint, dv_exception_constraint]

    def compute_transfer(self, i_from: int, i_to: int, t_start: float, tof: float):
        """
        Compute total DeltaV for transfer between two tomatoes.

        Parameters:
            i_from (int): Index of departure tomato.
            i_to (int): Index of arrival tomato.
            t_start (float): Start time in days.
            tof (float): Time of flight in days.
        Returns:
            float: Total DeltaV required for the transfer.
        """
        tof_sec = tof * pk.DAY2SEC

        r_i, v_i = self.tomatoes[i_from].eph(t_start)
        r_j, v_j = self.tomatoes[i_to].eph(t_start + tof)

        best_dv = float('inf')
        for cw in [False, True]:
            try:
                lambert_solution = pk.lambert_problem(r_i, r_j, tof_sec, MU_MOON, cw, self.max_revs)
            except Exception:
                continue
            for v1, v2 in zip(lambert_solution.get_v1(), lambert_solution.get_v2()):
                delta_v_i = np.linalg.norm(np.array(v1) - np.array(v_i))
                delta_v_j = np.linalg.norm(np.array(v2) - np.array(v_j))
                total_dv = delta_v_i + delta_v_j
                if total_dv < best_dv:
                    best_dv = total_dv
        return best_dv

    def get_nobj(self):
        # Our objective is to minimize total time of flight
        return 1
        
    def get_nix(self):
        # Has to be dependent on instance size
        return self.n_tomatoes

    def get_nec(self):
        # 1 permutation constraint: each tomato must be visited exactly once
        return 3

    def get_nic(self):
        # n-1 dv constraints: the change in velocity between consecutive legs must be below a certain threshold
        # n-2 time constraints: the departure time added to the time of flight must be below the next departure time
        return 1

    def example(self):
        """
        Load a solution from a .sol file and return the corresponding chromosome."""
        return self._read_solution("data/spoc4/kttsp/examples/medium.sol")


udp = KTTSP("data/spoc4/kttsp/problems/medium.kttsp")
