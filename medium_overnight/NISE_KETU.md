# Nise këtu — Medium overnight

**Këta hapa nuk prekin Easy ose punën e Codex-it.** Mos fshij asnjë fajll origjinal.

## Ku ta vendosësh

1. Shkarko ZIP-in dhe zgjidhe **Extract All**.
2. Kopjo **folderin e plotë** `medium_overnight` në folderin kryesor të `keplerian-tomato-tsp`, ku e ke `optimize_medium.py`.
3. Në VS Code hape WSL terminalin nga folderi kryesor i projektit (rreshti që fillon me `(kttsp-medium)` është në rregull).

## Testo para nisjes

Kopjo komandat **një nga një**:

```bash
python medium_overnight/test_offline.py
```

```bash
python medium_overnight/inspect_baseline.py
```

```bash
"$HOME/miniforge3/envs/kttsp-medium/bin/python" -B medium_overnight/campaign.py verify-best
```

Rezultati i tretë duhet të tregojë `202.8125` si ESA-valid. Nëse del gabim, mos e nis kërkimin e natës.

**Prova e shkurtër me PyKEP (36 sekonda)**

```bash
"$HOME/miniforge3/envs/kttsp-medium/bin/python" -u -B medium_overnight/campaign.py run --directory results/esa/medium_overnight/smoke --hours 0.01 --workers 1 --seed 73
```

Pas provës duhet të shfaqet `FINAL` dhe një rezultat. Është normale nëse rezultati ende është `202.8125`.

## Nise për natën (maksimum 8 orë aktive, dy workers)

```bash
bash medium_overnight/start.sh
```

Duhet të shkruajë `Medium process STARTED: PID ...`.

**Për me pa progresin:**

```bash
tail -f results/esa/medium_overnight/run_20261122/campaign.log
```

Me `Ctrl+C` del vetëm nga shfaqja e log-ut; optimizimi i ndarë vazhdon.

**Për me pa rezultatin në çdo moment:**

```bash
bash medium_overnight/status.sh
```

**Për ta ndalë me kujdes:**

```bash
bash medium_overnight/stop.sh
```

**Për ta vazhduar me kohën e mbetur:**

```bash
bash medium_overnight/start.sh
```

**Për ta validu më të mirin me ESA:**

```bash
bash medium_overnight/verify.sh
```

**Për ta eksportuar si `.sol` për ESA:**

```bash
"$HOME/miniforge3/envs/kttsp-medium/bin/python" -B medium_overnight/export_sol.py
```

### Kujdes për laptopin

Mbaje në rrymë, mos e lër Windows të hyjë në Sleep, dhe mos e mbyll WSL-in. Medium përdor **dy workers**, me prioritet të ulët dhe me affinity në dy procesorë logjikë kur mbështetet. Kjo i lë burime Easy-t, por nuk është garanci që dy proceset nuk do të ndikojnë fare njëri-tjetrin. Kontrollo `free -h` dhe `uptime` pas nisjes.

Ky program **nuk garanton 146 ditë**. Ai kërkon, ruan vetëm përmirësime të validuara me evaluator-in origjinal dhe nuk ndryshon bazën prej 202.8125 ditësh.
