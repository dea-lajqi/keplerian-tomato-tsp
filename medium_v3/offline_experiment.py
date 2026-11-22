#!/usr/bin/env python3
"""Reproducible 5-operator geometry proposal benchmark. NOT ESA validity."""
import argparse,json,random,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
sys.path.insert(0,str(HERE))
from topology import geometry_cost,shell_crossings
from analysis import load_sol,problem_elements

def run_experiment(elements,route,tries=4000,seed=2026):
    inc=[e[2] for e in elements];axes=[e[0] for e in elements]
    orig_shells=shell_crossings(route,axes)
    rng=random.Random(seed)
    modes=['adjacent','twoopt','insert','cluster-swap','bottleneck']
    n=len(route)
    def proxy(r):
        return sum(geometry_cost(r[i],r[i+1],elements)
                   for i in range(n-1) if (axes[r[i]]>8e6)==(axes[r[i+1]]>8e6))
    base=proxy(route);res={}
    for mode in modes:
        vals=[];seen=set()
        for _ in range(tries):
            q=route[:]
            if mode=='adjacent':
                i=rng.randrange(1,n-2);q[i],q[i+1]=q[i+1],q[i]
            elif mode=='twoopt':
                i=rng.randrange(1,n-9);j=i+rng.randint(2,7);q[i:j]=q[i:j][::-1]
            elif mode=='insert':
                i=rng.randrange(1,n-1);j=min(n-2,max(1,i+rng.randint(-7,7)))
                z=q.pop(i);q.insert(j,z)
            elif mode=='cluster-swap':
                i=rng.randrange(1,n-1)
                targets=[j for j in range(1,n-1) if j!=i and
                         (axes[q[i]]>8e6)==(axes[q[j]]>8e6) and abs(inc[q[i]]-inc[q[j]])<.07]
                if not targets:continue
                j=rng.choice(targets);q[i],q[j]=q[j],q[i]
            else:
                i=rng.choice([41,53,55,59,66,112,117,125,131,137]);k=rng.randint(1,3)
                a=max(1,i-k);b=min(n-1,i+k+1)
                segment=q[a:b];rng.shuffle(segment);q[a:b]=segment
            key=tuple(q)
            if q==route or key in seen or shell_crossings(q,axes)!=orig_shells:continue
            seen.add(key);vals.append(proxy(q))
        res[mode]={'unique_orders':len(vals),'lower_geometric_proxy':sum(x<base-1e-8 for x in vals),
                   'best_proxy_delta':min(vals)-base if vals else None}
    return {'baseline_geometry_proxy':base,'shell_crossings':orig_shells,
            'trials_per_mode':tries,'random_seed':seed,'caution':'NOT ESA-valid results',
            'operators':res}

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--tries',type=int,default=4000)
    p.add_argument('--seed',type=int,default=2026)
    p.add_argument('--instance',type=Path,default=ROOT/'data/spoc4/kttsp/problems/medium.kttsp')
    p.add_argument('--source',type=Path,default=ROOT/'results/esa/medium_20261120T080509Z_bc1f8d8f_4_verified.json')
    args=p.parse_args()
    els,_=problem_elements(args.instance);sol=load_sol(args.source)
    print(json.dumps(run_experiment(els,sol['path'],args.tries,args.seed),indent=2))
if __name__=='__main__':main()
