
from validate_dv import valid_dv


# Funksion per kontrollimin e renditjes, a i kontrollojme te gjitha domatet
def valid_order(order, n):
    if len(order) != n:
        return False

    if sorted(order) != list(range(n)):
        return False

    return True


# Shembull me 3 domate
order1 = [2, 0, 1]
order2 = [2, 0, 0]

print("Renditja 1:", valid_order(order1, 3))
print("Renditja 2:", valid_order(order2, 3))

#departures — kohët e nisjes.
#flights — kohëzgjatjet.
#t0 — koha kur mund të fillojë misioni.
#dt_min — kohëzgjatja minimale e fluturimit.
#t_max — koha maksimale e misionit.

#a jane kohet e udhetimeve te sakta
def valid_times(departures, flights, t0, dt_min, t_max):

    # Kontrollojme numrin e udhetimeve
    if len(departures) != len(flights):
        return False

    previous_arrival = t0

    for i in range(len(departures)):
        departure = departures[i]
        flight = flights[i]

        # Nuk mund te nisemi para se te kemi mberritur
        if departure < previous_arrival:
            return False

        # Fluturimi nuk mund te jete shume i shkurter
        if flight < dt_min:
            return False

        # Llogarisim kohen e mberritjes
        arrival = departure + flight

        # Misioni nuk mund ta tejkaloje kohen maksimale
        if arrival > t_max:
            return False

        previous_arrival = arrival

    return True


# Testimi
departures = [0, 2]
flights = [1, 1.5]

print("Kohet jane valide:",
      valid_times(departures, flights, 0, 0.001, 200))


def valid_solution(order, departures, flights, delta_vs, t0, dt_min, t_max, dv_max, dv_exception, E):

    n = len(order)

    if len(departures) != n-1 or len(flights) != n-1 or len(delta_vs) != n-1:
        return False
    if valid_order(order, n) is not True:
        return False
    if not valid_times(departures, flights, t0, dt_min, t_max):
        return False
    if not valid_dv(delta_vs, dv_max, dv_exception, E):
        return False
    return True

order = [2, 0, 1]
departures = [0, 2]
flights = [1, 1.5]
delta_vs = [80, 90]

print(valid_solution(
    order, departures, flights, delta_vs,
    0, 0.001, 200, 100, 600, 5
))