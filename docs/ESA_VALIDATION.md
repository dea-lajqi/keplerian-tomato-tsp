# KTTSP me vlerësuesin zyrtar të ESA-s

## Pse ndryshon verifikimi

Përshkrimi i sfidës e quan kolonën e gjashtë anomali të vërtetë,
pra këndin gjeometrik të trupit në orbitë. Vlerësuesi zyrtar ia jep
kolonën drejtpërdrejt konstruktorit `planet.keplerian` të PyKEP 2.6.4.
Ai konstruktor e lexon si anomali mesatare, këndin që rritet
uniformisht me kohën. Pozicioni dhe shpejtësia e llogaritur në një
ditë të caktuar ndryshojnë, duke ndryshuar edhe ΔV-në e transferimit.

`kttsp_esa_solver.py` përdor `compute_transfer` për çdo kosto.
Për çdo rrugë të plotë ndërton vektorin zyrtar dhe thërret
`fitness` nga skedari origjinal i ESA-s. Kontrollon edhe kufijtë
e vektorit dhe ditën e fundit të misionit. Rezultatet e vjetra
janë arkiva të modelit lokal. Provat dhe rekordet e reja ruhen
veçmas në `results/esa/`.

## Si funksionon kërkimi

Programi merr rendin e vizitave nga një rrugë e ruajtur dhe provon
si atë rend, ashtu edhe rendin e kundërt. Rezervon përjashtimet
për transferimet që kanë pasur kostot më të larta në orarin e vjetër.
Pastaj provon nisje dhe fluturime të reja me hapa prej 0.25, 0.125
dhe 0.0625 ditësh. Kostot e vjetra përdoren vetëm për renditjen e
transferimeve të vështira. Çdo pranim bazohet te kostoja e re e ESA-s.
Një rrugë e pjesshme shfaqet vetëm si progres në historikun e provës.

## Ekzekutimi

Përdor një mjedis të veçantë me `pykep=2.6.4`. Shkarko
[vlerësuesin Easy](https://api.optimize.esa.int/media/problems/spoc-4-keplerian-tomato-traveling-salesperson-small-1782203910047.py)
si `small.py` dhe
[vlerësuesin Medium](https://api.optimize.esa.int/media/problems/spoc-4-keplerian-tomato-traveling-salesperson-medium-1782203984713.py)
si `medium.py` në dosjen e vlerësuesve. Brenda asaj dosjeje vendos
dataset-et zyrtare `easy.kttsp` dhe `medium.kttsp` te
`data/spoc4/kttsp/problems/`.

```powershell
python -B kttsp_esa_solver.py easy --official-dir PATH_TO_EVALUATORS --seconds 20
python -B kttsp_esa_solver.py medium --official-dir PATH_TO_EVALUATORS --seconds 20
```

Në instalime ku importi i plotë i PyKEP ngec te një modul opsional,
`--native-core` ngarkon modulet origjinale orbitale `core` dhe
`planet` të PyKEP 2.6.4. Ky ishte konfigurimi i provës lokale.

| Instanca | Vizita | Ditë | Përjashtime |
| --- | ---: | ---: | ---: |
| Easy | 49/49 | 122.75 | 5 |
| Medium | 181/181 | 202.8125 | 5 |

Rezultatet janë objektivat e vlerësuesit, të llogaritur lokalisht.
