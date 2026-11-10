
# Hapim skedarin me te dhenat e ESA-s
with open("data/easy.kttsp", "r") as file:
    lines = file.readlines()

# Krijojme nje liste bosh per domatet
tomatoes = []


for line in lines:
    line = line.strip()

    # Anashkalojme komentet dhe rreshtat bosh
    if line.startswith("c") or not line:
        continue

    # Lexojme parametrat e problemit
    if line.startswith("p"):
        parts = line.split()

        t0 = float(parts[2])
        dt_min = float(parts[3])
        t_max = float(parts[4])
        dv_max = float(parts[5])
        dv_exception = float(parts[6])
        E = int(parts[7])

        continue

    # Lexojme te dhenat e domateve
    values = [float(value) for value in line.split()]
    tomatoes.append(values)


print("Numri i domateve:", len(tomatoes))
print("Domatja e pare:", tomatoes[0])

print("Domatja e dyte:", tomatoes[1])
print("Parametri a i domates se pare:", tomatoes[0][0])
print("Parametri e i domates se pare:", tomatoes[0][1])


print("Koha fillestare:", t0)
print("Koha minimale e fluturimit:", dt_min)
print("Koha maksimale:", t_max)
print("Delta-V maksimal:", dv_max)
print("Delta-V special:", dv_exception)
print("Numri i perjashtimeve:", E)
