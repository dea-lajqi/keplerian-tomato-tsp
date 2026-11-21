# KTTSP Medium

## Ekzekutimi

### Nisja nga zero

Nga rrënja e projektit, përdor `--scratch` edhe kur repoja përmban
rekordin 181/181:

```powershell
conda activate kttsp
python kttsp_medium_solver.py --scratch --minutes 5 --seed 42
```

Programi nis nga `data/medium.kttsp` dhe krijon një nën­dosje të re
te `.kttsp-scratch/` për çdo ekzekutim. Ai shfaq adresën e dosjes
dhe ruan aty JSON-et dhe historikun CSV. Skedarët në `results/`
mbeten të disponueshëm për verifikim dhe rinisje. Për të provuar
ndarjen në grupe që në nisje, shto `--coverage`. Rezultati i
kërkimit varet nga buxheti dhe parametrat e provës.

### Vazhdimi nga rekordi i ruajtur

Pa `--scratch`, programi ngarkon rekordin nga `results/` kur
skedari ekziston. Nga terminali me mjedisin `kttsp`:

```powershell
conda activate kttsp
python kttsp_medium_solver.py --minutes 5 --seed 42
python kttsp_medium_solver.py --verify-only
```

Për të vazhduar nga rekordi i ruajtur me kërkim më të gjerë:

```powershell
python kttsp_medium_solver.py --minutes 20 --seed 42 --optimize
```

Për t'i dhënë përparësi mbulimit të grupeve orbitale:

```powershell
python kttsp_medium_solver.py --minutes 30 --seed 43 --optimize --coverage
```

Për të futur domate të mbetura duke rindërtuar pjesë të rekordit:

```powershell
python kttsp_medium_solver.py --minutes 30 --seed 44 --focused
```

Programi lexon `data/medium.kttsp` me `read_instance` nga projekti.
Skedari përcakton 181 domate, epokën 0, maksimumin 500 ditë dhe
minimumin 0.01 ditë për çdo fluturim. Çdo transferim mund të kushtojë
deri në 100 m/s. Deri në pesë transferime lejohen mbi 100 dhe deri në
600 m/s. Indekset e domateve në JSON fillojnë nga 0.

Pas një ekzekutimi, programi ruan rekordin e verifikuar te
`results/KTTSP_MEDIUM_BEST_VERIFIED.json`. Ekzekutimi pasues e
rillogarit rekordin nga skedari dhe vazhdon kërkimin prej tij.
Çdo transferim në JSON ka pesë fusha, sipas radhës
`[burimi, destinacioni, dita_e_nisjes, ditët_e_fluturimit, delta_v]`.

Te `results/`, çdo ekzekutim ka një identifikues të vetin, edhe kur
përdoret e njëjta farë. Programi krijon skedarë të veçantë për
rekordin fillestar të ngarkuar, përmirësimet e verifikuara dhe
rezultatin përfundimtar. Këta JSON përmbajnë rrugën, ditët e
vizitave, parametrat dhe përmbledhjen e verifikimit. JSON-i
përfundimtar përfshin edhe kohën dhe metodën e çdo prove.
`KTTSP_MEDIUM_HISTORY.csv` shton nga një rresht për çdo JSON.
`KTTSP_MEDIUM_BEST_VERIFIED.json` shërben për rinisje dhe
përditësohet pasi përmirësimi është verifikuar dhe arkivuar.

## Si ndërtohet rruga

1. **Greedy Search** fillon nga një domate e zgjedhur nga orbitat e
   dataset-it. Në çdo hap vazhdon me një degë. Dega merr pikë sipas
   ditës së mbërritjes plus një penalitet për çdo përjashtim. Synimi
   është të ndërtojë shpejt një rrugë dhe të kursejë përjashtimet.
2. **Beam Search** përdor të njëjtin zgjerim dhe mban deri në tri degë
   në çdo hap. Kjo lejon disa zgjedhje për domaten e fundit dhe
   numrin e përjashtimeve, sepse hapi më i shpejtë mund ta bllokojë
   rrugën më vonë.
3. **Local Search** heq disa vizita nga fundi i rekordit dhe provon
   një vazhdim tjetër nga pjesa e ruajtur. Herë pas here kërkimi
   riniset nga një domate tjetër. Me `--optimize`, disa prova
   rindërtojnë një pjesë më të gjatë me deri në pesë degë.
4. **Kohët e transferimit** zgjidhen duke provuar disa kohë pritjeje
   dhe fluturimi. Fillimisht kontrollohen deri në dhjetë domate me
   orbita të ngjashme, ose deri në 18 në provat më të gjera. Kur aty
   mungon çdo vazhdim, kontrollohen të gjitha domatet e pavizituara
   me një dritare më të madhe.
5. **Lambert** llogarit shpejtësitë e nevojshme për të lidhur
   pozicionin e nisjes me pozicionin e mbërritjes brenda kohës së
   fluturimit. ΔV është shuma e dy ndryshimeve të shpejtësisë,
   për nisjen dhe për takimin me domaten. PyKEP provon të dy
   drejtimet dhe deri në 20 rrotullime të plota për çdo drejtim.
6. **Futja lokale** me `--optimize` zgjedh deri në 14 transferime
   pranë orbitës së një domateje të munguar. Për secilin transferim
   provon ta zëvendësojë me dy fluturime që futen në të njëjtën
   hapësirë kohore. Orari i pjesës tjetër të rrugës ruhet.
7. **Beam Search me grupe orbitale** me `--coverage` ndan domatet
   në pesë grupe nga gjysmëboshti dhe pjerrësia. Një nisje kalon
   nga orbitat e ulëta prograde në orbitat e larta me pjerrësi
   më të vogël, zbret te orbitat e ulëta polare, kthehet te
   orbitat e larta me pjerrësi më të madhe dhe përfundon te
   orbitat e ulëta retrograde. Provat mund të ndjekin edhe
   drejtimin e kundërt. Ndryshimi i pikës së ndarjes së orbitave
   të larta ndryshon edhe domatet që mund të shërbejnë për hyrjen
   dhe daljen nga grupi polar. Kërkimi rezervon nga një përjashtim
   për secilin kalim të ardhshëm mes grupeve.
8. **Rindërtimi i segmentit** me `--focused` zgjedh dy pika të
   rrugës pranë orbitës së domates që mungon. Ruan vizitat e
   segmentit dhe shton domaten e re duke provuar renditje e
   orare të tjera. Segmenti duhet të arrijë pikën e fundit para
   nisjes së transferimit vijues. Transferimet para dhe pas
   segmentit ruhen. Gjerësia e beam-it dhe hapi i kohës ndryshojnë
   mes provave, ndërsa numri i përjashtimeve llogaritet për
   të gjithë misionin.
9. **Rillogaritja e vazhdimit** fut domaten midis dy vizitave dhe
   lejon që transferimet pas saj të nisen në ditë të tjera.
   Për çdo transferim ruan kufirin e kategorisë origjinale.
   Dy fluturimet që zëvendësojnë një transferim ndajnë buxhetin
   e tij. Nëse rruga arrin përsëri para një nisjeje të vjetër,
   mund të ripërdorë orarin nga ajo pikë.

Rekordi krahasohet së pari sipas numrit të domateve. Kur numri
është i njëjtë, përparësi ka mbërritja më herët dhe pastaj numri
më i vogël i përjashtimeve. Kohët e provuara janë diskrete dhe
gjerësia e beam-it është e kufizuar. Rezultati është rekordi i
këtij kërkimi heuristik. Për provë të optimalitetit do të duhej
një kufi i sipërm i justifikuar ose një kërkim shterues.

## Funksionet

| Funksioni | Çfarë bën dhe pse duhet |
| --- | --- |
| `Instance.end` | Shton kohëzgjatjen te epoka për të marrë ditën e fundit të lejuar. |
| `lambert_options` | Gjeneron shpejtësitë Lambert dhe ΔV për secilën zgjidhje të PyKEP. |
| `verify` | Kontrollon indeksat, vizitat e përsëritura, rendin, kohët dhe kufijtë. Avancon orbitat nga epoka me PyKEP, rillogarit Lambert dhe avancon trajektoren e transferimit për të kontrolluar mbërritjen brenda 1 m dhe shpejtësinë Lambert brenda 0.001 m/s. |
| `route_value` | Vendos cili rekord është më i mirë kur krahasohen dy rrugë. |
| `Experiment.__init__` | Krijon një identifikues unik dhe mban parametrat e provës. |
| `Experiment.export` | Shkruan një JSON të ri me rrugën dhe verifikimin, pastaj shton rreshtin përkatës në CSV. |
| `Experiment.resume` | Vazhdon një eksperiment të papërfunduar me të njëjtin identifikues dhe llogarit kohën e kaluar ndaj afatit fillestar. |
| `save_best` | Përditëson skedarin për rinisje nga rekordi pasi rezultati është arkivuar. |
| `Search.__init__` | Përgatit buxhetin, farën e rastësisë dhe rendit destinacionet sipas ngjashmërisë së madhësisë dhe planit orbital. |
| `Search.state` | Llogarit pozicionin dhe shpejtësinë me `orbital_state_corrected` dhe i mban për ripërdorim. |
| `Search.cost` | Merr ΔV më të vogël për një çift domatesh dhe një orar të caktuar. |
| `Search.successors` | Provon orare dhe kthen vazhdimet e mundshme. Për çdo destinacion mban mbërritjen më të hershme në secilën kategori ΔV. |
| `Search.record` | Dërgon kandidatin për verifikim dhe e ruan vetëm kur përmirëson rekordin. |
| `Search.report` | Shfaq vizitat, ditën, përjashtimet dhe sekondat e kërkimit. |
| `Search.build` | Zgjeron rrugët hap pas hapi dhe përzgjedh degët që vazhdojnë. |
| `Search.insert_visit` | Provon të fusë një domate të munguar mes dy vizitave ekzistuese. |
| `Search.orbital_stages` | Llogarit pesë grupet e domateve nga madhësia dhe pjerrësia e orbitës. |
| `Search.segment_options` | Provon nisje dhe mbërritje në një rrjetë ditësh dhe kthen mbërritjen më të hershme për çdo kategori ΔV të lejuar. |
| `Search.rebuild_segment` | Mban disa renditje për vizitat e segmentit. Kur dy degë arrijnë të njëjtën domate me të njëjtat vizita dhe përjashtime, mban degën që mbërrin më herët sepse ajo mund të presë. |
| `Search.focused_pass` | Zgjedh dritare me madhësi të ndryshme pranë domateve të pavizituara dhe ia jep rindërtimit. |
| `Search.retime_insertion` | Fut domaten dhe rillogarit transferimet pas saj, duke kontrolluar buxhetin e përjashtimeve. |
| `Search.retiming_pass` | Provon pozicionet e futjes ku orbitat fqinje janë më të ngjashme me orbitën e domates që mungon. |
| `Search.guided_successors` | Provon transferimet drejt grupit aktual, me orare më të imta për kalimet mes lartësive dhe me buxhet të rezervuar për grupet e ardhshme. |
| `Search.build_guided` | Zgjeron dhe mban disa rrugë sipas grupeve. Për orbitat e larta i jep përparësi përparimit sipas pjerrësisë, që të mos mbeten domate të izoluara. |
| `Search.run` | Drejton nisjet, rindërtimet e fundit të rrugës dhe verifikimin përfundimtar. |
| `Search.finish_attempt` | Regjistron kohën, metodën dhe rezultatin e çdo prove, përfshirë provat pa përmirësim. |
| `main` | Lexon argumentet dhe dataset-in, pastaj nis kërkimin ose verifikimin. |

## Rezultati i provës 5-minutëshe

Nisja nga dataset-i ruajti 124 domate para një gabimi shkrimi në
Windows. Pas shtimit të riprovimit për ruajtjen, prova e plotë
rillogariti atë rekord dhe vazhdoi prej tij me farën 42.

| Matja | Vlera |
| --- | ---: |
| Domate të vizituara | 159 / 181 |
| Dita e mbërritjes | 196.9150 |
| Transferime të verifikuara | 158 |
| Përjashtime | 5 / 5 |
| ΔV më e madhe e një transferimi normal | 99.6225 m/s |
| ΔV më e madhe e një përjashtimi | 574.0792 m/s |
| Fluturimi më i shkurtër | 0.01 ditë |
| Prova të kërkimit | 40 |
| Koha e kërkimit | 300.0 s |

## Rezultati i optimizimit 20-minutësh

Prova me `--minutes 20 --seed 42 --optimize` vizitoi 159 nga 181
domate në **191.1450 ditë**, ose **5.7700 ditë më herët** se rekordi
fillestar. Përdori 5 nga 5 përjashtimet dhe kreu 68 prova në
1 200 sekonda. Një ekzekutim i veçantë riverifikoi të 158
transferimet. Fluturimi më i shkurtër zgjati 0.01 ditë. ΔV
maksimale ishte 99.6225 m/s për transferimet normale dhe
585.4740 m/s për përjashtimet.

| Metoda | Prova | Rekorde të reja |
| --- | ---: | ---: |
| Rindërtimi i fundit të rrugës | 48 | 6 |
| Rinisja me Beam Search | 15 | 0 |
| Futja mes dy vizitave | 5 | 0 |

Të gjashta rekordet shkurtuan kohëzgjatjen me të njëjtin numër
vizitash. Prova 39 ruajti 112 vizitat e para, mbajti deri në pesë
degë dhe provoi kohë fluturimi me hapa prej 0.125 ditësh. Ajo
gjeti rekordin përfundimtar pas 673.797 sekondash kërkim.

- [Kopja e rekordit 5-minutësh](../results/KTTSP_MEDIUM_BASELINE_159_5MIN.json)
- [Arkivi përfundimtar i eksperimentit](../results/KTTSP_MEDIUM_20261119T204451Z_b582e1bd_008_final.json)
- [Rekordi për rinisje](../results/KTTSP_MEDIUM_BEST_VERIFIED.json)
- [Historiku CSV](../results/KTTSP_MEDIUM_HISTORY.csv)

## Pse kërkimi ngeci në 159 vizita

Rekordi me mbërritje në ditën 191.1450 vizitonte 119 nga 121
domate në orbitat e larta dhe nga 20 në orbitat e ulëta prograde
dhe retrograde. Të 20 domatet në orbitat e ulëta polare mungonin.
Tre nga pesë përjashtimet u përdorën për kalime mes lartësive dhe
dy për kalime mes orbitave të larta. Një nga kalimet e kushtueshme
ishte kthimi nga grupi i ulët retrograd në orbitat e larta.

Kur kërkimi rindërton vetëm fundin, pjesa e ruajtur mund t'i ketë
përcaktuar tashmë kalimet e kushtueshme. Ndarja e re e kërkimit
e vendos grupin e ulët polar midis dy grupeve të larta dhe
e vendos njërin nga grupet e tjera të ulëta në secilin skaj.
Kjo lë katër kalime mes lartësive dhe një përjashtim shtesë
për një transferim brenda grupeve.

## Rezultati i eksperimentit 30-minutësh

Me farën 43, eksperimenti përfundoi pas 1 800 sekondash me
**179/181 domate**, mbërritje në **205.6875 ditë** dhe **5/5
përjashtime**. Mungojnë domatet 29 dhe 62, me indeksim nga 0.
Një ekzekutim i veçantë riverifikoi të 178 transferimet. ΔV
maksimale ishte 99.9980 m/s për transferimet normale dhe
596.9213 m/s për përjashtimet. Fluturimi më i shkurtër zgjati
0.0625 ditë.

| Metoda | Prova | Rekorde të reja | Vizita të shtuara |
| --- | ---: | ---: | ---: |
| Beam Search me grupe orbitale | 80 | 23 | 20 |
| Rindërtimi i fundit të rrugës | 15 | 2 | 0 |
| Rinisja me Beam Search | 5 | 0 | 0 |
| Futja mes dy vizitave | 6 | 0 | 0 |

Prova 9 gjeti 179 vizita në 229.8850 ditë. Prova 90 ruajti
137 vizitat e para dhe rindërtoi vazhdimin me deri në pesë
degë. Kjo dha rekordin në 205.6875 ditë pas 1 532.468
sekondash kërkim. Futja me orare të fiksuara nuk gjeti një
vend të vlefshëm për dy domatet e mbetura.

- [Kopja e rekordit fillestar 159/181](../results/KTTSP_MEDIUM_BASELINE_159_20MIN.json)
- [Arkivi përfundimtar 30-minutësh](../results/KTTSP_MEDIUM_20261119T211112Z_2db10abe_027_final.json)
- [Historiku CSV](../results/KTTSP_MEDIUM_HISTORY.csv)

## Rezultati i eksperimentit të fokusuar

| Faza | Domate | Dita e mbërritjes | Përjashtime |
| --- | ---: | ---: | ---: |
| Rekordi fillestar | 179 | 205.6875 | 5 |
| Rindërtimi i segmentit rreth domates 29 | 180 | 205.6875 | 5 |
| Futja e domates 62 dhe rillogaritja e vazhdimit | 181 | 214.2500 | 5 |

Domatja 29 u fut midis 166 dhe 86 duke ndryshuar rendin dhe
oraret brenda një segmenti me dy pika fikse. Prova të tilla për
domaten 62 mbetën të kufizuara nga nisja e transferimit vijues.
Transferimi i mëparshëm 41 → 34 mbërrinte në ditën 160.9375.
Rruga e re 41 → 62 → 34 mbërrin te 34 në ditën 167.5625.
Dy transferimet kushtojnë 98.6796 dhe 99.1628 m/s, prandaj
futja nuk harxhoi një përjashtim të ri.

Faza e parë u ndal pas 13 provave. Faza e dytë u nis nga rekordi
i verifikuar 180/181, me të njëjtin identifikues dhe afat, duke
përdorur `--focused --resume-experiment ID --initial-attempts 13`.
Pasi u vizituan të 181 domatet, eksperimenti u mbyll rreth 277
sekonda nga nisja, brenda kufirit 30 minuta. PyKEP riverifikoi
të 180 transferimet. ΔV maksimale ishte 99.9980 m/s për
transferimet normale dhe 596.9213 m/s për përjashtimet.
Fluturimi më i shkurtër ishte 0.0625 ditë. Numri i vizitave
arriti maksimumin e dataset-it. Koha minimale e misionit
mbetet e paprovuar.

- [Kopja e rekordit fillestar 179/181](../results/KTTSP_MEDIUM_BASELINE_179_30MIN.json)
- [Përmirësimi me 180 vizita](../results/KTTSP_MEDIUM_20261119T215354Z_3c26c12c_002_improvement.json)
- [Arkivi përfundimtar me 181 vizita](../results/KTTSP_MEDIUM_20261119T215354Z_3c26c12c_005_final.json)
- [Historiku CSV](../results/KTTSP_MEDIUM_HISTORY.csv)
