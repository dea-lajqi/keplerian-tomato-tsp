
import json
import importlib.util
from pathlib import Path

# 1. Lexojme rezultatin tone Easy
result_file = Path(
    "results/esa/easy_20261120T080457Z_b0dd1006_8_verified.json"
)

data = json.loads(result_file.read_text(encoding="utf-8"))

# 2. Ndertojme decisionVector per ESA
transfers = data["transfers"]

decision_vector = (
    [row[2] for row in transfers]
    + [row[3] for row in transfers]
    + data["path"]
)

print("Domatet:", len(data["path"]))
print("Decision vector:", len(decision_vector))

# 3. Ngarkojme kodin origjinal te ESA-s
esa_file = next(Path("esa_evaluation").glob("*small*.py"))

spec = importlib.util.spec_from_file_location(
    "esa_small", esa_file
)

module = importlib.util.module_from_spec(spec)

# Kodi i ESA-s krijon nje objekt gjate importimit.
# Ai kerkon nje dataset ne C:\data.
# Per kete arsye, e ngarkojme vetem pjesen para atij objekti.
source = esa_file.read_text(encoding="utf-8")

marker = '\nudp = KTTSP('

if marker not in source:
    raise RuntimeError(
        "Nuk u gjet pika e inicializimit te ESA-s."
    )

source = source.split(marker, 1)[0]

exec(
    compile(source, str(esa_file), "exec"),
    module.__dict__
)

# 4. Krijojme vleresuesin me dataset-in lokal
dataset = Path("data/easy.kttsp").resolve()

print("Dataset exists:", dataset.exists())

if not dataset.exists():
    raise FileNotFoundError(dataset)

evaluator = module.KTTSP(str(dataset))

# 5. Ekzekutojme fitness() zyrtar
print("\nDuke ekzekutuar ESA fitness...")

fitness = evaluator.fitness(decision_vector)

print("\n===== ESA OFFICIAL EVALUATION =====")
print("Fitness:", fitness)
print("Mission duration:", fitness[0], "days")
print("Constraints:", fitness[1:])

# 6. Kontrollojme vlefshmerine
valid = all(value <= 1e-8 for value in fitness[1:])

print("VALID:", valid)

if valid:
    print("SUCCESS! Easy solution accepted.")
else:
    print("INVALID! Constraints violated.")
