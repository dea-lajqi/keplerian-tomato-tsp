import random

def generate_order(n):
    tomatoes = list(range(n))
    random.shuffle(tomatoes)
    return tomatoes

for i in range(3):
    order = generate_order(5)
    print(order)

def mission_time(departures, flights):
    return departures[-1]+flights[-1]

departures = [0, 2]
flights = [1, 1.5]
print(mission_time(departures, flights))

current_time = 9.5
best_time = 12.5


times = [15, 12, 18, 9, 11]

best_time = float("inf")

for current_time in times:
    if current_time < best_time:
        best_time = current_time

print(best_time)


current_order = [2, 0, 1]
current_time = 9.5

best_order = [0, 1, 2]
best_time = 12.5

if current_time < best_time:
    best_time = current_time
    best_order = current_order.copy()

print("Koha me e mire:", best_time)
print("Renditja me e mire:", best_order)

orders = [
    [2, 0, 1],
    [1, 2, 0],
    [0, 1, 2]
]

times = [9.5, 7.0, 11.0]

best_time = float("inf")
best_order = None