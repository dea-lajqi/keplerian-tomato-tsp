"""Exchange coverage between the two high-shell passages and repair both boundaries."""
import argparse
from collections import Counter
from functools import lru_cache
from itertools import combinations, pairwise, permutations
from pathlib import Path
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from global_event_easy import geometry
from medium_coupled.endgame import decode
from medium_coupled.high_shell import Campaign
from medium_coupled.run import Paused
from pareto_easy import BudgetExpired, arrival


class Exchange(Campaign):
    def __init__(self,args):
        super().__init__(args)
        self.normal_matrix,self.allowed=geometry(self.udp,4.9048695e12)[0]

    @lru_cache(maxsize=25000)
    def floor(self,start,remaining,end):
        """Minimum mandatory exception count over all allowed orders."""
        if not remaining:
            return int(not self.normal_matrix[start,end]) if self.allowed[start,end] else 100
        return min((int(not self.normal_matrix[start,b])+
                     self.floor(b,remaining[:i]+remaining[i+1:],end)
                    for i,b in enumerate(remaining) if self.allowed[start,b]),default=100)

    def windows(self,span):
        first,second=self.shells[1:3]
        return (first-span+1,first+1),(second,second+span)

    def plans(self):
        for span in (4,5,6):
            (l,r),(p,q)=self.windows(span)
            A=tuple(self.path[l+1:r])
            B=tuple(self.path[p+1:q])
            outside=sum(self.exception[:l])+sum(self.exception[r:p])+sum(self.exception[q:])
            room=5-outside
            proposals=[]
            for u,v in ((1,0),(0,1),(1,1),(2,0),(0,2),(2,1),(1,2),(2,2)):
                if max(u,v)>span-2:
                    continue
                for X in combinations(A,u):
                    for Y in combinations(B,v):
                        one=tuple(sorted((set(A)-set(X))|set(Y)))
                        two=tuple(sorted((set(B)-set(Y))|set(X)))
                        first=self.floor(self.path[l],one,self.path[r])
                        second=self.floor(self.path[p],two,self.path[q])
                        self.count["geometry_plans"]+=1
                        if first+second<=room:
                            proposals.append((first+second,abs(u-v),X,Y,one,two,first,second))
            # Alternate unbalanced moves and exchanges instead of consuming
            # the entire budget on one combinatorial family.
            proposals.sort(key=lambda x:(x[0],-len(x[2])-len(x[3]),x[1],x[2],x[3]))
            bins={}
            for row in proposals:
                key=len(row[2]),len(row[3])
                bins.setdefault(key,[]).append(row)
            for c in range(self.args.per_family):
                for key,group in bins.items():
                    if c<len(group):
                        yield span,(l,r,p,q),group[c],room

    def test(self,window,one,two,cap1,cap2,stop,plan):
        l,r,p,q=window
        stats=Counter()
        saved1=[];saved2=[]

        for left,right,members,cap,batch in ((l,r,one,cap1,saved1),(p,q,two,cap2,saved2)):
            def submit(part,right=right,left=left,batch=batch,cap=cap):
                path,d,f=(part[k] for k in ("path","dep","tof"))
                patch=tuple((a,b,dep,tof,float(self.udp.compute_transfer(a,b,dep,tof)))
                            for (a,b),dep,tof in zip(pairwise(path[left:]),d[left:],f[left:]))
                actual=sum(leg[-1]>self.normal for leg in patch)
                if actual<=cap:
                    batch.append(patch)
            reserve=5-sum(self.exception[:left])-cap
            decode(self.udp,self.seed,left,self.dep[right],.125,.027,stop,stats,submit,
                   reserve=reserve,endpoint=self.path[right],targets_override=(*members,self.path[right]))
            if not batch:
                self.count["anchored_window_failure"]+=1
                break
        self.count.update({k:v for k,v in stats.items() if k!="max_depth"})
        result={"completed_windows":[len(saved1),len(saved2)]}
        if not saved1 or not saved2:
            return result
        complete=[]
        for A in sorted(saved1,key=lambda legs:arrival(legs[-1]))[:3]:
            for B in sorted(saved2,key=lambda legs:arrival(legs[-1]))[:3]:
                all_legs=(*self.legs[:l],*A,*self.legs[r:p],*B,*self.legs[q:])
                candidate=self.build(all_legs)
                if self.submit(candidate,"passage_exchange",plan):
                    gain=arrival(self.legs[q-1])-arrival(B[-1])
                    complete.append((gain,candidate,len(all_legs)-len(self.legs[q:])))
        result["complete"]=len(complete)
        if complete:
            gain,candidate,right=max(complete,key=lambda row:row[0])
            result["arrival_gain_at_second_join"]=gain
            if time.monotonic()<stop and gain>.04:
                legs=[(*edge,d,t,float(self.udp.compute_transfer(*edge,d,t)))
                      for edge,d,t in zip(pairwise(candidate["path"]),candidate["dep"],candidate["tof"])][:right]
                result["propagation"]=self.propagate(candidate,legs,plan,stop,.125,self.state["best"]-.005)
        return result

    def run(self):
        tested={tuple(row["key"]) for row in self.state["trials"]}
        for span,window,row,room in self.plans():
            if time.monotonic()>=self.stop:
                break
            _,_,X,Y,one,two,m1,m2=row
            for cap1 in range(m1,room-m2+1):
                plan=(span,",".join(map(str,X)),",".join(map(str,Y)),cap1)
                if plan in tested:
                    continue
                start=time.monotonic()
                if start>=self.stop:
                    break
                data={}
                try:
                    stop=min(self.stop,start+self.args.trial_seconds)
                    if self.args.floating:
                        data=self.floating(window,one,two,cap1,room-cap1,stop,plan)
                    else:
                        data=self.test(window,one,two,cap1,room-cap1,stop,plan)
                except (BudgetExpired,Paused):
                    data["time_cap"]=True
                self.state["trials"].append({"key":plan,"seconds":time.monotonic()-start,**data})
                self.save()
                print(plan,data,flush=True)
        self.state["wall_seconds"]=self.state.get("wall_seconds",0.)+time.monotonic()-self.started
        self.save()

    def floating(self,window,one,two,first_cap,second_cap,stop,plan):
        """Permit the low-shell excursion and the full downstream clock to move."""
        l,r,p,q=window
        orbit=self.udp.tomato_orbital_parameters

        def best_order(a,targets,b,cap):
            best=None
            for order in permutations(targets):
                edges=tuple(pairwise((a,*order,b)))
                if all(self.allowed[x,y] for x,y in edges):
                    spend=sum(not self.normal_matrix[x,y] for x,y in edges)
                    if spend<=cap:
                        merit=sum(abs(float(orbit[x,2]-orbit[y,2])) for x,y in edges)
                        option=(spend,merit,order)
                        if best is None or option<best:
                            best=option
            return None if best is None else best[2]

        A=best_order(self.path[l],one,self.path[r],first_cap)
        B=best_order(self.path[p],two,self.path[q],second_cap)
        if A is None or B is None:
            return {"no_order":True}
        path=(*self.path[:l+1],*A,*self.path[r:p+1],*B,*self.path[q:])
        if len(path)!=181 or len(set(path))!=181:
            raise ValueError("Exchange does not cover exactly 181 tomatoes")
        oracle_stop=min(stop,time.monotonic()+5.)
        forward,depth,stats=self.sweep(path,tuple(self.legs[:l]),None,oracle_stop,
                                      .078125,goal=self.state["best"]+.30)
        out={"forward":{"depth":depth,"stats":stats},"changed_legs":sum(a!=b for a,b in zip(path,self.path))}
        if depth==180-l:
            for _,legs in forward.values():
                row=self.build(legs)
                if self.submit(row,"floating_passage_exchange",plan) and time.monotonic()<stop:
                    out["polish"]=self.refine(row,max(150,l),0.,stop,"floating_exchange_polish",plan)
        else:
            _,legs=min(forward.values(),key=lambda x:x[0])
            out["last_arrival"]=arrival(legs[-1])
            if time.monotonic()<stop:
                back,reached,rs=self.sweep(path,legs,None,stop,.046875,True,
                                          self.state["best"]-.001)
                out["backward"]={"depth":reached,"stats":rs}
                if reached==180-len(legs):
                    for clock,tail in back.values():
                        if arrival(legs[-1])<=self.state["best"]-.001-clock+1e-9:
                            self.submit(self.build((*legs,*tail[::-1])),"floating_exchange_join",plan)
        return out


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--directory",required=True,type=Path)
    p.add_argument("--seed-record",required=True,type=Path)
    p.add_argument("--baseline-record",type=Path)
    p.add_argument("--per-family",type=int,default=5)
    p.add_argument("--floating",action="store_true")
    p.add_argument("--seconds",type=float,default=120.)
    p.add_argument("--trial-seconds",type=float,default=9.)
    Exchange(p.parse_args()).run()
