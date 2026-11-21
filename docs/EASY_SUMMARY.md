# KTTSP - Easy | Shenime te projektit

## Rezultati i ruajtur
- `results/KTTSP_49_PYKEP_VALIDATED.json`: zgjidhja 49/49 e raportuar
  si e verifikuar lokalisht me PyKEP (kontrolli zyrtar nuk eshte konfirmuar).
- `results/KTTSP_46_DAY183_VALIDATED.json`: zgjidhja e meparshme 46/49.
- `data/easy.kttsp`: te dhenat origjinale te Easy.

## Si eshte punuar
1. Leximi i te dhenave dhe gjendjeve orbitale (`read_data.py`, `orbital.py`).
2. Beam Search dhe kerkime heuristike per nje rruge fillestare.
3. Permiresim i renditjes nga 46/49 drejt 49/49 me ndryshime te udhezuara.
4. Kerkim / optimizim i koheve te transferimeve dhe verifikim me PyKEP.

**Kujdes:** `kttsp_reach_49.py` rinderton rrugen e njohur. Skriptet
`kttsp_complete_solver.py` dhe `kttsp_search_until_49.py` kerkojne
rruge, por ende nuk eshte vertetuar qe ato arrijne 49/49 nga zero.
Mos pretendo se 49/49 gjendet automatikisht nga zero pa nje prove te tille.

## Skedaret kryesore
- `beam_search.py`: baza e kerkimit me beam.
- `kttsp_complete_solver.py`: kerkuesi i pergjithshem eksperimental.
- `kttsp_search_until_49.py`: kerkuesi pa limit kohe.
- `kttsp_49_dynamic_programming.py`: optimizim i koheve nga nje rruge baze.
- `kttsp_reach_49.py`: rindertimi dhe kontrolli me PyKEP.
- `experiments/easy/`: provat e tjera, te ruajtura per dokumentim.

Per te ekzekutuar nje skript nga eksperimente/easy, perdor:
`python run_easy_experiment.py EMRI.py`

## Medium
- `data/medium.kttsp` ruhet me vete. Mos nderro parametrat e Easy
  ne kodet ekzistuese; Medium ka limite te ndryshme.

## Siguria
- Rezultatet ekzistuese ne `results/` nuk levizen ose mbishkruhen.
- Nje backup ZIP krijohet ne `backups/` para cdo organizimi.
- Per te rikthyer skriptet ne rrenje: `python organizo_kttsp.py --undo`.
