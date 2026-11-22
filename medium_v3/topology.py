"""Physics-informed candidate generation for the ESA Medium instance.

The graph score is ONLY a proposal heuristic; it is not a physical transfer
and never certifies a mission. True candidates are evaluated by ESA PyKEP.
"""
from __future__ import annotations
import math
import random
from dataclasses import dataclass

MU = 4.9048695e12

@dataclass(frozen=True)
class Move:
    left: int
    right: int  # inclusive last tomato, NOT edge index
    route: list[int]
    mode: str
    proxy_delta: float


def shell_of(axes, a):
    return int(axes[a] > 8e6)


def shell_crossings(route, axes):
    return sum(shell_of(axes, a) != shell_of(axes, b) for a,b in zip(route,route[1:]))


def geometry_cost(a, b, elements):
    if int(elements[a][0] > 8e6) != int(elements[b][0] > 8e6):
        return 450.0
    # Plane change approximation, not a lower bound on the actual Lambert ΔV.
    ai, _, inc_i, node_i, _, _ = elements[a]
    aj, _, inc_j, node_j, _, _ = elements[b]
    c = math.cos(inc_i)*math.cos(inc_j) + math.sin(inc_i)*math.sin(inc_j)*math.cos(node_i-node_j)
    theta=math.acos(max(-1.0,min(1.0,c)))
    return 2*math.sqrt(MU/(0.5*(ai+aj)))*math.sin(theta/2)


def proxy(route, elements):
    return sum(geometry_cost(a,b,elements) for a,b in zip(route,route[1:]))


def local_proxy_delta(original, new, elements, left, right):
    lo=max(0,left-1);hi=min(len(original)-1,right+1)
    old=sum(geometry_cost(original[i],original[i+1],elements) for i in range(lo,hi))
    now=sum(geometry_cost(new[i],new[i+1],elements) for i in range(lo,hi))
    return now-old


def candidate_moves(sol, elements, rng:random.Random, count=32, mode='mixed'):
    """Moves inside a fixed orbital shell block. Keeps complete permutations.

    Different from V1: guided by baseline long legs, cluster/plane similarity,
    and a geometric proposal score, with a small exploration fraction.
    """
    path=sol['path'];tofs=sol['tof']; n=len(path)
    axes=[p[0] for p in elements]
    crossing_orig=shell_crossings(path,axes)
    lengths=list(range(1,n-1))
    weights=[0.8 + min(8.,float(tofs[i-1]+tofs[i])) for i in lengths]
    mode_set=['bottleneck_insert','adjacent','small_reverse','near_plane_swap','block_shuffle','endgame']
    if mode!='mixed':mode_set=[mode]
    found={}
    for _ in range(max(30,count*25)):
        m=rng.choice(mode_set)
        if m=='endgame':
            center=rng.randrange(max(1,n-19),n-1)
        else:
            center=rng.choices(lengths,weights=weights,k=1)[0]
        radius=rng.choice([2,3,4,5,6])
        left=max(0,center-radius);right=min(n-1,center+radius)
        # Never include a shell-crossing as an interior node: preserving the five
        # expensive bridges is a deliberate strategy for this search arm.
        while left<center and shell_of(axes,path[left])!=shell_of(axes,path[center]):left+=1
        while right>center and shell_of(axes,path[right])!=shell_of(axes,path[center]):right-=1
        if right-left<3:continue
        q=path[:]
        a,b=left+1,right
        interior=q[a:b]
        if len(interior)<2:continue
        if m=='bottleneck_insert':
            i=rng.randrange(len(interior));j=rng.randrange(len(interior));z=interior.pop(i);interior.insert(j,z)
        elif m=='adjacent':
            i=rng.randrange(len(interior)-1);interior[i],interior[i+1]=interior[i+1],interior[i]
        elif m=='small_reverse':
            i,j=sorted(rng.sample(range(len(interior)),2));interior[i:j+1]=reversed(interior[i:j+1])
        elif m=='near_plane_swap':
            pairs=[(i,j) for i in range(len(interior)) for j in range(i+1,len(interior))
                   if geometry_cost(interior[i],interior[j],elements)<90.]
            if not pairs:continue
            i,j=rng.choice(pairs);interior[i],interior[j]=interior[j],interior[i]
        elif m=='block_shuffle':
            rng.shuffle(interior)
        else:
            rng.shuffle(interior)
        q[a:b]=interior
        if q==path or shell_crossings(q,axes)!=crossing_orig:continue
        val=local_proxy_delta(path,q,elements,left,right)
        key=tuple(q)
        if key not in found:found[key]=Move(left,right,q,m,val)
        if len(found)>=count*3:break
    allmoves=list(found.values());allmoves.sort(key=lambda x:x.proxy_delta)
    top=allmoves[:count//2]
    pool=allmoves[count//2:];rng.shuffle(pool)
    return top+pool[:max(0,count-len(top))]


def long_transfer_analysis(sol,elements):
    path=sol['path'];tofs=sol['tof'];axes=[p[0] for p in elements]
    expensive=[(round(tofs[i],6),i+1,path[i],path[i+1],
                round(geometry_cost(path[i],path[i+1],elements),2))
               for i in range(len(tofs)) if tofs[i]>3]
    return {'tomatoes':len(path),'shell_counts':{
       'inner':sum(a<=8e6 for a in axes),'outer':sum(a>8e6 for a in axes)},
       'shell_crossings':shell_crossings(path,axes),
       'flights_over_3_days':len(expensive),
       'longest_transfers':sorted(expensive,reverse=True)[:20]}
