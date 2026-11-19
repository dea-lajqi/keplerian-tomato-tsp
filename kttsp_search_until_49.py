"""KTTSP Easy - KERKIM I RRUGES (jo lexues i nje zgjidhjeje 49/49).

Nisja pa limit: python kttsp_search_until_49.py --scratch --until-49
Nisja nga 46 checkpoint: python kttsp_search_until_49.py --until-49

Input: data/easy.kttsp, read_data.py, orbital.py, PyKEP, NumPy.
Checkpoint-i 46/49 (nese ekziston) PERDORET vetem si rezultat i nje faze
paraprake te kerkimit. Asnje rruge 49/49 apo kohe e saj nuk eshte ne kod.

Fazat: (1) Beam search (kur duhet), (2) large-neighborhood search mbi
renditjen, (3) dynamic programming per kohe/fluturime,
(4) validim i pavarur i cdo transferimi me PyKEP.

Nuk premtohet rezultat 49/49 ne cdo ekzekutim (heuristike, jo optimizim ekzakt).
"""
import argparse
import itertools
import json
import math
import random
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
import pykep as pk
from read_data import read_instance
from orbital import orbital_state_corrected, MU_MOON

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data' / 'easy.kttsp'
BACKUP = ROOT / 'results' / 'KTTSP_46_DAY183_VALIDATED.json'
OUT_DIR = ROOT / 'results'
# Kjo eshte rrjeta e kerkimit, jo renditja e domateve.
TOFS = (.25,.5,.75,1.,1.25,1.5,1.75,2.,2.5,3.,3.5,4.,4.5,5.,6.,7.,8.,10.,12.)
DP_STEP = .25
NORMAL_TOL = 1.e-7


def run(args):
    if not DATA.is_file():
        raise SystemExit(f'Mungon {DATA}')
    tomatoes, t0, dt_min, t_max, dv_normal, dv_exception, max_exc = read_instance(str(DATA))
    n = len(tomatoes)
    if n < 2:
        raise SystemExit('Dataset i pavlefshem')
    OUT_DIR.mkdir(exist_ok=True)
    start_clock = time.monotonic()
    end_clock = float('inf') if args.until_49 else start_clock + 60 * args.minutes
    if args.until_49:
        print('KERKIM PA KUFIZIM KOHOR: ndalet vetem nese gjenden 49/49 ose me Ctrl+C.', flush=True)
        print('Suksesi NUK garantohet: heuristika mund te ngece; rekordet ruhen ne results/.', flush=True)
    rng = random.Random(args.seed)
    best = None
    evaluated = 0
    attempts = 0

    @lru_cache(maxsize=120000)
    def state(i, t):
        return orbital_state_corrected(tomatoes[i], t)

    @lru_cache(maxsize=500000)
    def dv(i, j, departure, tof):
        if i == j or tof < dt_min or departure < t0 - 1e-9 or departure + tof > t_max + 1e-9:
            return math.inf
        try:
            r1, v1 = state(i, departure)
            r2, v2 = state(j, round(departure + tof, 6))
            best_dv = math.inf
            for cw in (False, True):
                try:
                    lp = pk.lambert_problem(
                        np.asarray(r1, dtype=float).tolist(),
                        np.asarray(r2, dtype=float).tolist(),
                        float(tof) * 86400., MU_MOON, cw, 20)
                    for vd, va in zip(lp.v0, lp.v1):
                        val = float(np.linalg.norm(np.asarray(vd) - v1) +
                                    np.linalg.norm(np.asarray(va) - v2))
                        if val < best_dv:
                            best_dv = val
                except (ValueError, RuntimeError, OverflowError):
                    pass
            return best_dv
        except (ValueError, RuntimeError, OverflowError, FloatingPointError):
            return math.inf

    def validate(sol, fresh=True):
        """Verifikon rrugen/oret/DV, pa u bazuar vetem te vlerat ne JSON."""
        path, legs = sol['path'], sol['transfers']
        if len(path) != len(set(path)) or not all(0 <= i < n for i in path):
            return None
        if len(legs) != len(path) - 1:
            return None
        last_day = float(t0)
        ex = 0
        fixed = []
        for k, (i, j, dep, tof, _) in enumerate(legs):
            if (i, j) != (path[k], path[k+1]) or dep < last_day - 1e-7:
                return None
            if tof < dt_min or dep + tof > t_max + 1e-7:
                return None
            if fresh:
                dv.cache_clear()  # Rillogaritje e pavarur, por me e ngadalte.
            actual = dv(int(i), int(j), round(float(dep), 6), round(float(tof), 6))
            if not math.isfinite(actual) or actual > dv_exception + NORMAL_TOL:
                return None
            ex += actual > dv_normal + NORMAL_TOL
            if ex > max_exc:
                return None
            fixed.append([int(i), int(j), float(dep), float(tof), float(actual)])
            last_day = dep + tof
        return {'path': path[:], 'transfers': fixed,
                'arrival': last_day, 'exceptions': ex,
                'verified_pykep': True, 'verified_official': False}

    def save(sol, label='SEARCH'):
        output = OUT_DIR / f'KTTSP_{len(sol["path"])}_{label}_CANDIDATE.json'
        output.write_text(json.dumps(sol, indent=2), encoding='utf-8')
        print(f'RUAJTA: {output}', flush=True)

    def announce(sol):
        nonlocal best
        if sol is None:
            return False
        candidate_arrival = float(sol.get('arrival',sol['transfers'][-1][2]+sol['transfers'][-1][3] if sol['transfers'] else t0))
        if best is None or (len(sol['path']), -candidate_arrival) > (len(best['path']), -best['arrival']):
            checked = validate(sol, fresh=False)
            if checked is None:
                return False
            best = checked
            save(checked)
            print(f'REKORD I RI: {len(best["path"])}/{n} | '
                  f'dita {best["arrival"]:.2f} | perjashtime {best["exceptions"]}/{max_exc}', flush=True)
            return True
        return False

    def orbit_distance(a, b):
        """Shenues i lire i ngjashmerise; jo llogaritje e ΔV."""
        x, y = tomatoes[a], tomatoes[b]
        da = abs(math.log(max(x[0],1.)/max(y[0],1.)))
        di = abs(x[2]-y[2]); di = min(di, 2*math.pi-di)
        de = abs(x[1]-y[1]);
        return 1.8*da + .65*di + .7*de

    @lru_cache(maxsize=120000)
    def successor_options(i, j, arrival, wait_limit):
        """Alternativa sipas ndarjes normal / exception dhe koha e mberritjes.

        Prerja e koheve eshte heuristike. Nje rikthim i vonuar mund te jete
        me i mire ne orbita te varura nga koha, prandaj ruajme disa bucket-a.
        """
        t_begin = math.ceil((arrival-1e-8)/DP_STEP)*DP_STEP
        found = []
        waits = int(wait_limit / DP_STEP)
        for h in range(waits + 1):
            dep = round(t_begin + h * DP_STEP, 6)
            if dep >= t_max:
                break
            for tof in TOFS:
                fin = round(dep + tof, 6)
                if fin > t_max:
                    continue
                val = dv(i, j, dep, tof)
                if not math.isfinite(val) or val > dv_exception + NORMAL_TOL:
                    continue
                extra = int(val > dv_normal + NORMAL_TOL)
                found.append((fin, extra, [i,j,dep,tof,val]))
                if len(found) >= 200:
                    break
            if len(found) >= 200:
                break
        # Diversitet ne kohe dhe perjashtime (0.5 dite per bucket).
        found.sort(key=lambda z: (z[1], z[0], z[2][4]))
        result = []
        seen = set()
        count_per_class = {0: 0, 1: 0}
        for item in found:
            extra = item[1]
            key = (extra, int(item[0] * 2 + 1e-9))
            if key in seen or count_per_class[extra] >= 5:
                continue
            seen.add(key)
            count_per_class[extra] += 1
            result.append(item)
        return tuple((fin,ex,tuple(leg)) for fin,ex,leg in result)

    def retime(order, prefix=None, cutoff=None, do_print=False):
        """DP mbi rrugen e plote: kerkon kohe dhe Delta-V per cdo kalim."""
        if len(order) != len(set(order)):
            return None
        # Nese fillimi i nje checkpoint-i mbetet identik, e ruajme ate pjese.
        prefix_legs = []
        if prefix is not None:
            oldpath, oldlegs = prefix['path'], prefix['transfers']
            keep = 0
            for a,b in zip(order, oldpath):
                if a != b: break
                keep += 1
            # Mos ngrij kohen e segmenteve qe ndryshohen indirekt:
            # ruaj maksimumi 3 transferimet e para.
            prefix_legs = [list(x) for x in oldlegs[:max(0,min(3,keep-1))]]
        start_day = (prefix_legs[-1][2] + prefix_legs[-1][3]) if prefix_legs else float(t0)
        used = sum(x[4] > dv_normal + NORMAL_TOL for x in prefix_legs)
        if used > max_exc:
            return None
        # (exceptions, arrival_bucket): (arrival, transfers)
        states = {(used, int(start_day * 2)):(start_day, prefix_legs)}
        start_at = len(prefix_legs)
        for pos in range(start_at, len(order)-1):
            if time.monotonic() >= (cutoff or end_clock):
                return None
            i,j = order[pos],order[pos+1]
            expanded = {}
            for (e,_), (arr, legs) in states.items():
                # Para ndryshimeve te medha lejojme me shume kohe pritjeje.
                waits = 7.0 if len(order) == n else 3.0
                for fin,extra,leg in successor_options(i,j,round(arr,6),waits):
                    ex = e + extra
                    if ex > max_exc:
                        continue
                    key = (ex, int(fin*2 + 1e-9))
                    if key not in expanded or fin < expanded[key][0]:
                        expanded[key] = (fin, legs + [list(leg)])
            if not expanded:
                return None
            # Heuristic beam: ruajme diversitet ne numer perjashtimesh/kohe.
            ranked = sorted(expanded.items(), key=lambda kv: (kv[0][0],kv[1][0]))
            # Gjithmone ruaj te pakten nje mundesi per secilin numer
            # perjashtimesh; perndryshe priten rruget qe duan 5 exceptions.
            selected = []
            kept_per_ex = {}
            for key, value in ranked:
                ex = key[0]
                if kept_per_ex.get(ex,0) >= 3:
                    continue
                kept_per_ex[ex] = kept_per_ex.get(ex,0) + 1
                selected.append((key,value))
            states = dict(selected)
            if do_print:
                fastest = min(s[0] for s in states.values())
                print(f'  HAPI {pos+1:02d}/{len(order)-1} | {i}->{j} | '
                      f'dita min {fastest:.2f} | alternativa {len(states)}', flush=True)
        if not states:
            return None
        day, legs = min(states.values(), key=lambda x:(x[0],len(x[1])))
        ex = sum(x[4] > dv_normal + NORMAL_TOL for x in legs)
        return {'path':list(order), 'transfers':legs, 'arrival':day, 'exceptions':ex}

    # FAZA A. Nga checkpoint-i qe kemi fituar me Beam Search (per te kursyer kohe).
    if BACKUP.is_file() and not args.scratch:
        prior = json.loads(BACKUP.read_text(encoding='utf-8'))
        prior_path = [int(x) for x in prior['path']]
        if prior.get('verified') is False or len(set(prior_path)) != len(prior_path):
            print('Kujdes: verifikim i checkpoint-it...')
        cp = {'path': prior_path, 'transfers': [list(x) for x in prior['transfers']]}
        print(f'NISJE NGA CHECKPOINT BEAM SEARCH: {len(prior_path)}/{n}', flush=True)
        if not announce(cp):
            raise SystemExit('Checkpoint-i 46 nuk e kaloi kontrollin me PyKEP.')
    else:
        print('PA CHECKPOINT: po ndertojme rrugen me Beam Search nga dataset-i.',flush=True)
        # Beam Search nga zero, me disa pika nisjeje. Nuk ka rruge te paracaktuar.
        seeds = list(range(n))
        rng.shuffle(seeds)
        beam = [(float(t0), 0, [i], []) for i in seeds]
        width = args.beam
        for depth in range(1,n):
            if time.monotonic() >= end_clock:
                break
            next_beam = []
            for arrival, used, path, legs in beam:
                remaining = [i for i in range(n) if i not in path]
                remaining.sort(key=lambda j: orbit_distance(path[-1],j))
                # Ngjashmeria orbitale e orienton kerkimin; edhe disa rastesisht.
                near = remaining[:min(14,len(remaining))]
                extra = rng.sample(remaining, min(4,len(remaining)))
                for j in list(dict.fromkeys(near+extra)):
                    for fin, e, leg in successor_options(path[-1],j,round(arrival,6),2.0)[:3]:
                        if used+e<=max_exc:
                            next_beam.append((fin,used+e,path+[j],legs+[list(leg)]))
            if not next_beam:
                break
            # Diversitet sipas domates se fundit, numrit dhe bucket kohor.
            next_beam.sort(key=lambda x:(x[1],x[0]))
            selected = []; seen = set()
            for x in next_beam:
                key = (x[2][-1],x[1],int(x[0]//2))
                if key in seen: continue
                seen.add(key); selected.append(x)
                if len(selected)>=width:break
            beam = selected
            candidate = max(beam,key=lambda q:(len(q[2]),-q[0]))
            announce({'path':candidate[2],'transfers':candidate[3]})
            print(f'Beam depth {depth+1}/{n} | rruget {len(beam)} | '
                  f'best {len(best["path"]) if best else 0}/{n}',flush=True)
            if best is not None and len(best['path'])==n:
                break

    if best is None:
        raise SystemExit('S’u gjet as rruge fillestare. Rrit --minutes.')
    if len(best['path']) == n:
        print('Te gjitha domatet jane perfshire ne fazen e pare.')
    else:
        # FAZA B. LNS: kerkojme nje rruge qe perfshin te gjitha domatet.
        # Nuk shkruajme askund listën finale 49/49 ose vendin e 4,11,17.
        base = dict(best)
        missing = sorted(set(range(n))-set(base['path']))
        print('MUNGON:',missing, '| kerkohen permutime dhe ndryshime te renditjes.',flush=True)

        def reversal_candidates(sol):
            p = sol['path']; leg = sol['transfers']
            candidates = [(0, 0, tuple(p), 'pa-ndryshim')]
            for a in range(1,len(p)-1):
                for b in range(a+2, min(len(p),a+9)):
                    q = p[:]; q[a:b] = reversed(q[a:b])
                    # Prioritet: blloqet qe mund te heqin dy kalime me exception.
                    old_ex = 0
                    for k in (a-1,b-1):
                        if 0 <= k < len(leg) and leg[k][4] > dv_normal:
                            old_ex += 1
                    candidates.append((-old_ex,b-a,tuple(q),f'reverse[{a}:{b}]'))
            candidates.sort(key=lambda x:(x[0],x[1]))
            return candidates

        def insertion_sites(path, targets):
            scores=[]
            for pos in range(1,len(path)):
                left,right=path[pos-1],path[pos]
                # Kosto e lire per orientim - jo Delta-V Lambert.
                d = min(orbit_distance(left,m) for m in targets)
                d += min(orbit_distance(m,right) for m in targets)
                scores.append((d,pos))
            scores.sort()
            # Te gjitha pozicionet jane te mundshme, por me te afertat se pari.
            return [pos for _,pos in scores]

        if len(missing)<=3:
            # Një klaster me te gjitha domatet qe mungojne: provon CDO renditje.
            clusters = list(itertools.permutations(missing))
        else:
            clusters = []
            for k in (1,2,3):
                for block in itertools.combinations(missing,min(k,len(missing))):
                    clusters.extend(itertools.permutations(block))
                    if len(clusters)>250:break
                if len(clusters)>250:break

        reversals = reversal_candidates(base)
        print(f'KANDIDATE: {len(reversals)} ndryshime renditjeje; '
              f'{len(clusters)} renditje te domateve qe mungojne.',flush=True)
        attempts=0
        # Faza e strukturuar, duke provuar fillimisht kthimet e blloqeve
        # qe heqin exception dhe pozicionet me profil orbital te perafert.
        for _, _, revpath, label in reversals:
            if time.monotonic()>=end_clock or (best and len(best['path'])==n):break
            sites = insertion_sites(revpath, missing)
            # Fillimisht 15 pozicionet me premtuese dhe permutimet e mundshme.
            for pos in sites[:min(15,len(sites))]:
                if time.monotonic()>=end_clock or (best and len(best['path'])==n):break
                for cluster in clusters:
                    if time.monotonic()>=end_clock:break
                    order = list(revpath[:pos])+list(cluster)+list(revpath[pos:])
                    if len(order)!=len(set(order)):
                        continue
                    attempts+=1
                    candidate = retime(order,prefix=base)
                    if candidate is not None:
                        announce(candidate)
                        if len(candidate['path'])==n:
                            break
                    if attempts%10 == 0:
                        print(f'Prova {attempts} | {label} | futja ne {pos} | '
                              f'best {len(best["path"])}/{n} | '
                              f'{time.monotonic()-start_clock:.0f}s',flush=True)

        # Faza e dyte: ndryshime te rastit nga rruga me e mire,
        # me futje te 1-3 domateve (nese varianti i strukturuar s'mjafton).
        while time.monotonic()<end_clock and len(best['path'])<n:
            base = best
            path = base['path'][:]
            unvisited = list(set(range(n))-set(path))
            if not unvisited:
                break
            m = rng.randint(1,min(3,len(unvisited)))
            targets = rng.sample(unvisited,m)
            for target in targets:
                pos = rng.randint(1,len(path))
                path.insert(pos,target)
            if len(path)>4:
                i=rng.randint(1,len(path)-3)
                j=min(len(path),i+rng.randint(2,7))
                if rng.random()<.7: path[i:j]=reversed(path[i:j])
                else:
                    chunk = path[i:j]
                    rng.shuffle(chunk)
                    path[i:j] = chunk
            attempts+=1
            candidate=retime(path,prefix=base)
            if candidate is not None:
                announce(candidate)
            if attempts%25 == 0:
                print(f'Prova {attempts} | best {len(best["path"])}/{n} | '
                      f'{time.monotonic()-start_clock:.0f}s',flush=True)

    print('\n=== REZULTATI I KERKIMIT ===')
    print(f'{len(best["path"])}/{n} domate | dita {best["arrival"]:.2f} | '
          f'perjashtime {best["exceptions"]}/{max_exc}')
    if len(best['path']) == n:
        # Rillogaritje e vertete me PyKEP per çdo kalim.
        checked = validate(best, fresh=True)
        if checked is None:
            raise RuntimeError('Validimi perfundimtar me PyKEP deshtoi.')
        save(checked, 'FINAL_SEARCH')
        print(f'START | tomato {checked["path"][0]} | dita {t0:.2f}')
        for k, (i,j,dep,tof,dval) in enumerate(checked['transfers'],1):
            tag=' | EXCEPTION' if dval > dv_normal + NORMAL_TOL else ''
            print(f'HAPI {k:02d}/{n-1} | {i}->{j} | nisja {dep:.2f} | '
                  f'fluturimi {tof:.2f} | mberritja {dep+tof:.2f} | '
                  f'DV {dval:.2f}{tag}')
        print(f'SUCCESS {n}/{n} | PYKEP VALIDATED | '
              f'dita {checked["arrival"]:.2f} | exc {checked["exceptions"]}/{max_exc}')
    else:
        print('NUK u arrit numri maksimal ne kete ekzekutim. '
              'Ruhet kandidati me i mire; prova mund te vazhdohet me kohe me te gjate.')
    print('VLERESUESI ZYRTAR NUK ESHTE EKZEKUTUAR.')

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--minutes',type=float,default=20,help='kufiri kohor ne minuta (injorohet me --until-49)')
    parser.add_argument('--until-49',action='store_true',help='kerko pa afat kohor deri ne 49/49 ose Ctrl+C')
    parser.add_argument('--beam',type=int,default=80,help='gjeresia e Beam Search nga zero')
    parser.add_argument('--seed',type=int,default=20261119,help='fara e rastit per riprodhim')
    parser.add_argument('--scratch',action='store_true',help='mos lexo checkpoint-in; nis nga dataset-i')
    a=parser.parse_args()
    if a.minutes<=0 and not a.until_49: parser.error('--minutes duhet te jete > 0')
    try:
        run(a)
    except KeyboardInterrupt:
        print('\nKERKIMI U NDAL ME CTRL+C. Rezultatet e ruajtura ne results/ nuk fshihen.', flush=True)
