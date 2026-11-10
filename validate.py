
# Funksion per kontrollimin e renditjes
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
