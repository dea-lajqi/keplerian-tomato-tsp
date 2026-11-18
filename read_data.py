
def read_instance(filename):
    with open(filename, "r") as file:
        lines = file.readlines()

    tomatoes = []

    for line in lines:
        line = line.strip()

        if line.startswith("c") or not line:
            continue

        if line.startswith("p"):
            parts = line.split()

            t0 = float(parts[2])
            dt_min = float(parts[3])
            t_max = float(parts[4])
            dv_max = float(parts[5])
            dv_exception = float(parts[6])
            E = int(parts[7])

            continue

        values = [float(value) for value in line.split()]
        tomatoes.append(values)

    return tomatoes, t0, dt_min, t_max, dv_max, dv_exception, E


if __name__ == "__main__":
    data = read_instance("data/easy.kttsp")

    tomatoes = data[0]

    print("Numri i domateve:", len(tomatoes))
    print("Domatja e pare:", tomatoes[0])
