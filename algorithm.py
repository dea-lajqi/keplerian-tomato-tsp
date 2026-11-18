
from read_data import read_instance
from orbital import orbital_state
from lambert import compute_transfer, find_best_transfer
import numpy as np


# 1. FUNKSIONET BAZE TE GREEDY
#Domatet qe ende si kemi vizitu
def get_candidates(tomatoes, visited):
    return [
        i for i in range(len(tomatoes))
        if i not in visited
    ]

#merr te dhenat orbitale te nje domateje
def get_orbit(tomatoes, tomato_id):
    return tomatoes[tomato_id]

#zgjedh domaten me koston me te vogel
def choose_next_tomato(candidates, costs):
    best_tomato = None
    best_cost = float("inf")

    for tomato in candidates:
        cost = costs[tomato]

        if cost < best_cost:
            best_cost = cost
            best_tomato = tomato

    return best_tomato

#perserit zgjedhjen derisa te i rendise te gjitha domatet
def build_greedy_order(tomatoes, costs):
    visited = []
    order = []

    while len(visited) < len(tomatoes):
        candidates = get_candidates(tomatoes, visited)
        next_tomato = choose_next_tomato(candidates, costs)

        order.append(next_tomato)
        visited.append(next_tomato)

    return order


# 2. GREEDY ME MATRICE

#zgjedh domaten tjeter sipas kostos nga domatja aktuale
def greedy_with_matrix(cost_matrix, start):
    n = len(cost_matrix)

    order = [start]
    visited = {start}
    current = start

    while len(visited) < n:
        best_cost = float("inf")
        best_tomato = None

        for tomato in range(n):
            if tomato in visited:
                continue

            cost = cost_matrix[current][tomato]

            if cost < best_cost:
                best_cost = cost
                best_tomato = tomato

        order.append(best_tomato)
        visited.add(best_tomato)
        current = best_tomato

    return order

#mbledhja e kostove te te gjitha kalimeve
def total_cost(order, cost_matrix):
    total = 0

    for i in range(len(order) - 1):
        current = order[i]
        next_tomato = order[i + 1]

        total += cost_matrix[current][next_tomato]

    return total


# 3. MULTI-START GREEDY

#provon nisje nga domate te ndryshme 
def multi_start_greedy(cost_matrix):
    n = len(cost_matrix)

    best_order = None
    best_cost = float("inf")

    for start in range(n):
        order = greedy_with_matrix(cost_matrix, start)
        cost = total_cost(order, cost_matrix)

        print(
            "Nisja:", start,
            "Renditja:", order,
            "Kostoja:", cost
        )

        if cost < best_cost:
            best_cost = cost
            best_order = order.copy()

    return best_order, best_cost


# 4. LOCAL SEARCH

#i nderron vendet dy domateve ne renditje
def swap_tomatoes(order, i, j):
    new_order = order.copy()

    new_order[i], new_order[j] = (
        new_order[j], new_order[i]
    )

    return new_order

#provon ndrerrime me permiresu rezultatin
def local_search(order, cost_matrix):
    best_order = order.copy()
    best_cost = total_cost(best_order, cost_matrix)

    improved = True

    while improved:
        improved = False

        for i in range(len(best_order)):
            for j in range(i + 1, len(best_order)):

                new_order = swap_tomatoes(
                    best_order, i, j
                )

                new_cost = total_cost(
                    new_order, cost_matrix
                )

                if new_cost < best_cost:
                    best_order = new_order
                    best_cost = new_cost
                    improved = True

    return best_order, best_cost


# 5. TESTIMI I GREEDY DHE LOCAL SEARCH

def test_algorithm():
    cost_matrix = [
        [0, 8, 3, 10],
        [7, 0, 5, 2],
        [6, 4, 0, 9],
        [3, 8, 5, 0]
    ]

    best_order, best_cost = multi_start_greedy(
        cost_matrix
    )

    print("\nRenditja me e mire:", best_order)
    print("Kostoja me e mire:", best_cost)

    new_order, new_cost = local_search(
        best_order, cost_matrix
    )

    print("Pas Local Search:", new_order)
    print("Kostoja pas Local Search:", new_cost)


# 6. TESTIMI ME TE DHENAT REALE TE ESA-S

def test_real_orbit():

    # Lexojme Easy Mode
    data = read_instance("data/easy.kttsp")

    tomatoes, t0, dt_min, t_max, dv_max, dv_exception, E = data

    print("\n================================")
    print("TESTIMI ME DOMATET REALE")
    print("================================")

    print("Numri i domateve:", len(tomatoes))
    print("t0:", t0)
    print("Koha minimale e fluturimit:", dt_min)
    print("Koha maksimale e misionit:", t_max)
    print("Kufiri Delta-V:", dv_max)

    # POZICIONI DHE SHPEJTESIA

    a, e, inc, omega_node, omega_peri, M = tomatoes[0]

    position, velocity = orbital_state(
        a, e, inc, omega_node, omega_peri, M
    )

    print("\nDOMATJA E PARE")
    print("Pozicioni:", position)
    print("Shpejtesia:", velocity)

    # TESTI I PARE LAMBERT

    dv = compute_transfer(
        tomatoes[0],
        tomatoes[1],
        start_day=t0,
        tof_days=1.0
    )

    print("\nTRANSFERIMI 0 -> 1")
    print("Delta-V:", dv, "m/s")
    print("Kufiri:", dv_max, "m/s")
    print("Brenda kufirit:", dv <= dv_max)

    # KERKIMI I KOHES ME TE MIRE

    best_tof, best_dv, feasible = find_best_transfer(
        tomatoes[0],
        tomatoes[1],
        start_day=t0,
        min_tof=dt_min,
        max_tof=5.0,
        dv_limit=dv_max,
        steps=50
    )

    print("\nKERKIMI I TRANSFERIMIT 0 -> 1")
    print("Koha me e mire:", best_tof, "dite")
    print("Delta-V me e vogel:", best_dv, "m/s")
    print("Brenda kufirit:", feasible)



def search_next_tomato(tomatoes, current, visited,
                       current_time, dt_min, dv_max,
                       max_tof=5.0, steps=30):

    best_tomato = None
    best_arrival = float("inf")
    best_tof = None
    best_dv = None

    for target in range(len(tomatoes)):

        if target in visited:
            continue

        for tof in np.linspace(dt_min, max_tof, steps):

            dv = compute_transfer(
                tomatoes[current],
                tomatoes[target],
                start_day=current_time,
                tof_days=float(tof)
            )

            if dv <= dv_max:
                arrival = current_time + tof

                if arrival < best_arrival:
                    best_tomato = target
                    best_arrival = arrival
                    best_tof = tof
                    best_dv = dv

                break

    return best_tomato, best_arrival, best_tof, best_dv


def greedy_real(tomatoes, t0, dt_min, t_max, dv_max,
                start=0, max_tof=5.0, steps=30):

    order = [start]
    visited = {start}
    current = start
    current_time = t0

    while len(visited) < len(tomatoes):

        target, arrival, tof, dv = search_next_tomato(
            tomatoes, current, visited,
            current_time, dt_min, dv_max,
            max_tof=max_tof, steps=steps
        )

        if target is None:
            print("Nuk u gjet transferim i vlefshem.")
            break

        if arrival - t0 > t_max:
            print("U tejkalua koha maksimale e misionit.")
            break

        order.append(target)
        visited.add(target)

        print(
            "Domatja:", target,
            "| Koha:", round(arrival, 4),
            "| Delta-V:", round(dv, 2)
        )

        current = target
        current_time = arrival

    return order, current_time


# 7. EKZEKUTIMI

if __name__ == "__main__":
    test_algorithm()
    test_real_orbit()
    print("\nGREEDY ME DOMATE REALE")

    data = read_instance("data/easy.kttsp")
    tomatoes, t0, dt_min, t_max, dv_max, dv_exception, E = data

    order, finish_time = greedy_real(
        tomatoes[:5],
        t0,
        dt_min,
        t_max,
        dv_max,
        start=0,
        max_tof=5.0,
        steps=30
    )

    print("Renditja:", order)
    print("Domate te vizituara:", len(order), "/ 5")
    print("Koha e fundit:", finish_time)
