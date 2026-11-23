"""Medium route surgery with time-dependent beam reconstruction and exception lookahead."""
import argparse
from collections import Counter, defaultdict
import itertools
import json
from pathlib import Path
import sys
import time


sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from global_event_easy import geometry
from medium_coupled.checkpoint import atomic_json
from medium_coupled.endgame import decode
from medium_coupled.high_shell import Campaign
from medium_coupled.meet import SideGrid
from medium_coupled.run import solve, Paused
from medium_overnight.validation import verify
from medium_v4.driver import unpack
from pareto_easy import BudgetExpired, arrival


class Surgery(Campaign):
    def __init__(self,args):
        super().__init__(args)
        (self.normal_edges,self.allowed),self.geometry_floor=geometry(self.udp,4.9048695e12)
        self.inc=self.udp.tomato_orbital_parameters[:,2]
        self.long_edges=sorted(self.long)
        self.state.setdefault("configuration",{
            "mode":args.mode,"seed":str(args.seed_record.resolve()),"beam":args.beam,
            "deadline":args.goal,"phase":args.phase})
        for path in args.anchors:
            sol=unpack(json.loads(path.read_text()))
            verify(self.udp,sol)
            for (a,b),d,f in zip(itertools.pairwise(sol["path"]),sol["dep"],sol["tof"]):
                dv=float(self.udp.compute_transfer(a,b,d,f))
                row=(a,b,d,f,dv)
                if row not in self.anchors[a,b]:
                    self.anchors[a,b].append(row)

    def route_floor(self,path):
        return sum(not self.normal_edges[a,b] for a,b in itertools.pairwise(path))

    def all_allowed(self,path):
        return all(self.allowed[a,b] for a,b in itertools.pairwise(path))

    def sample(self,grid,a,b,clock,allow,stats,fine=None,wait=3.5):
        options=list(self.anchors[a,b])
        for lattice in (grid,fine) if fine is not None else (grid,):
            for leg in lattice.edges(a,b,clock-lattice.origin,wait,7.4,allow):
                if leg is None:
                    continue
                d,f=lattice.origin+leg[2],leg[3]
                stats["replays"]+=1
                v=float(self.udp.compute_transfer(a,b,d,f))
                options.append((a,b,d,f,v))
        return options

    def block(self,prefix,members,sink,due,reserve,stop,stats):
        """Joint subset/order/timing beam, preserving each exact exception count."""
        origin=arrival(prefix[-1])
        if origin+(len(members)+1)*self.udp.min_tof>=due:
            return []
        grid=SideGrid(self.udp,origin,due,.078125,self.args.phase,stop,False)
        fine=SideGrid(self.udp,origin,due,.0546875,.01,stop,False)
        spend=sum(l[-1]>self.normal for l in prefix)
        maximum=self.udp.n_dv_exception-reserve
        states={(0,prefix[-1][1],spend):(origin,())}
        full=(1<<len(members))-1
        try:
            for depth in range(len(members)+1):
                update={}
                for (mask,a,used),(clock,chain) in states.items():
                    grid.check()
                    pending=[(1<<i,b) for i,b in enumerate(members) if not mask&(1<<i)]
                    choices=pending if depth<len(members) else [(0,sink)]
                    for bit,b in choices:
                        if not self.allowed[a,b]:
                            continue
                        other=[c for k,c in pending if k!=bit]
                        normal_exit=(any(self.normal_edges[b,c] for c in other)
                                     if other else (b==sink or self.normal_edges[b,sink]))
                        mandatory=int(not normal_exit)
                        need=int(not self.normal_edges[a,b])
                        if used+need+mandatory>maximum:
                            continue
                        options=self.sample(grid,a,b,clock,used+mandatory<maximum,
                                stats,fine if (depth<2 or not other) else None,
                                min(5.,max(1.,due-clock)))
                        for leg in options:
                            e=int(leg[-1]>self.normal)
                            end=arrival(leg)
                            if (leg[2]<clock-1e-9 or end>due+1e-9 or
                                    used+e+mandatory>maximum or leg[-1]>self.cap):
                                continue
                            key=(mask|bit,b,used+e)
                            if key not in update or end<update[key][0]:
                                update[key]=(end,(*chain,leg))
                stats["block_labels"]+=len(update)
                if not update:
                    stats["block_blocked_depth"]=depth
                    return []
                if depth==len(members):
                    return [(*prefix,*chain) for (mask,_,_),(_,chain) in
                            sorted(update.items(),key=lambda x:x[1][0]) if mask==full]
                quotas=Counter()
                states={}
                for key,val in sorted(update.items(),key=lambda row:row[1][0]+.30*row[0][2]):
                    if quotas[key[0],key[2]]>=5:
                        continue
                    quotas[key[0],key[2]]+=1
                    states[key]=val
                    if len(states)>=self.args.beam:
                        break
        finally:
            for lattice in (grid,fine):
                stats.update(lattice.stats)
        return []

    def complete(self,path,prefix,goal,step,stop,key,stats,*,permitted_extra=()):
        if time.monotonic()>=stop or arrival(prefix[-1])>=goal:
            return
        saved={}
        flags=None
        if getattr(self.args,"reserve_gate",False):
            flags=[int((a,b) in permitted_extra or
                       min(self.axis(a),self.axis(b))*2<max(self.axis(a),self.axis(b)))
                   for a,b in itertools.pairwise(path)]
        try:
            f,depth,details=self.sweep(path,prefix,flags,stop,step,goal=goal,saved=saved)
        except BudgetExpired:
            self.save_gate_prefixes(path,saved,key)
            raise
        self.save_gate_prefixes(path,saved,key)
        stats["furthest"] = len(prefix)+depth
        if "blocked_edge" in details:
            stats["blocking_edge"]=details["blocked_edge"]
        for cut in (126,161,170,175,178):
            if cut in saved:
                stats[f"arrival_at_{cut}"]=min(arrival(legs[-1]) for _,legs in saved[cut].values())
        if depth==180-len(prefix):
            for _,legs in f.values():
                candidate=self.build(legs)
                valid=self.submit(candidate,"nonlocal_forward",key)
                if (valid and getattr(self.args,"polish",False) and time.monotonic()<stop
                        and arrival(legs[-1])<self.state["best"]+.15):
                    try:
                        stats["polished"]=solve(self.udp,candidate,max(len(prefix),139),2.4,0.,
                            min(stop,time.monotonic()+4.),
                            lambda q:self.submit(q,"nonlocal_forward_polish",key),self.count)
                    except Paused:
                        stats["polish_time_cap"]=True
        for cut in (175,172,169):
            if cut not in saved or time.monotonic()>=stop:
                continue
            for _,legs in saved[cut].values():
                virtual={"path":list(path),"dep":[x[2] for x in legs]+self.dep[cut:],
                         "tof":[x[3] for x in legs]+self.tof[cut:]}
                s=Counter()
                try:
                    decode(self.udp,virtual,cut,goal,.026,.007,min(stop,time.monotonic()+2.5),
                           s,lambda q:self.submit(q,"nonlocal_terminal_repair",key),
                           extra_anchors=tuple(x for (a,b),rows in self.anchors.items()
                                               if a in path[cut:] and b in path[cut:] for x in rows))
                except BudgetExpired:
                    s["terminal_time_cap"]+=1
                self.count.update({k:v for k,v in s.items() if k!="max_depth"})
                stats["terminal_max_depth"]=max(stats.get("terminal_max_depth",0),s["max_depth"])
                if time.monotonic()>=stop:
                    break

    def save_gate_prefixes(self,path,saved,key):
        if not getattr(self.args,"save_prefixes",False):
            return
        outfile=self.args.directory/"gate_pool.json"
        profile=json.loads(outfile.read_text()) if outfile.exists() else {}
        for cut in (112,116,120,121,126,156,158,159,160,161,165,169,172,175):
            for spend,(_,legs) in saved.get(cut,{}).items():
                if spend>5:
                    continue
                mask=0
                for node in (legs[0][0],*(l[1] for l in legs)):
                    mask|=1<<node
                mark=f"{cut}-{spend}-{legs[-1][1]}-{mask}"
                if mark not in profile or arrival(legs[-1])<arrival(profile[mark]["legs"][-1]):
                    profile[mark]={"legs":legs,"remaining":path[cut+1:],
                                   "arrival":arrival(legs[-1]),"spend":spend,"plan":key}
        if profile:
            atomic_json(outfile,profile)

    def neighborhoods(self):
        if self.args.mode=="windows":
            for width in self.args.spans:
                for i in self.args.positions or self.long_edges:
                    l=i-width//2
                    r=l+width
                    yield ("block",i,l,r),l,r,None
        elif self.args.mode=="exchange":
            # A four-crossing topology has disjoint high passages. Assign up
            # to three near-plane targets across the middle low-shell visit.
            _,exit_gate,entry_gate,_=self.shells
            left=tuple(range(exit_gate-8,exit_gate+1))
            right=tuple(range(entry_gate+1,entry_gate+10))
            proposals=[]
            for count in (1,2,3):
                families=[]
                for side in (0,1):
                    for take in itertools.combinations(left if side==0 else right,count):
                        nodes=tuple(self.path[j] for j in take)
                        if max(self.inc[x] for x in nodes)-min(self.inc[x] for x in nodes)>.21:
                            continue
                        for gap in (right if side==0 else left):
                            if max(abs(self.inc[self.path[gap]]-self.inc[n]) for n in nodes)>.23:
                                continue
                            chosen=set(take)
                            # Retain unrelated coverage while changing the low-shell
                            # entry or exit order and separating the two edits in time.
                            path=[]
                            for i,b in enumerate(self.path):
                                if i not in chosen:
                                    path.append(b)
                                if i==gap:
                                    path.extend(nodes)
                            local=self.route_floor(path)
                            if local>5 or not self.all_allowed(path):
                                continue
                            score=sum(abs(self.inc[a]-self.inc[b]) for a,b in itertools.pairwise(path))
                            families.append((local,score,side,take,gap,path))
                families.sort(key=lambda row:row[:5])
                proposals.extend(families[:self.args.per_size])
            for _,_,side,take,gap,path in proposals:
                l=min((*take,gap))-1
                yield ("move",side,*take,gap),l,180,tuple(path)
        elif self.args.mode=="final_gate":
            gate=self.shells[-1]
            l=gate-17
            high=self.path[l+1:gate+1]
            low=self.path[gate+1:]
            families={}
            for size in (1,2,3,4):
                for start in range(len(high)-size+1):
                    piece=high[start:start+size]
                    base=high[:start]+high[start+size:]
                    for place in range(len(base)+1):
                        pattern=base[:place]+piece+base[place:]
                        if pattern==high:
                            continue
                        candidate=(*self.path[:l+1],*pattern)
                        if not all(self.normal_edges[a,b] for a,b in itertools.pairwise(candidate[l:])):
                            continue
                        for k in (0,1,2,3,5,8):
                            lower=low[:]
                            lower[0],lower[k]=lower[k],lower[0]
                            if not self.allowed[pattern[-1],lower[0]]:
                                continue
                            full=(*candidate,*lower)
                            if not self.all_allowed(full) or self.route_floor(full)>5:
                                continue
                            edits=set(itertools.pairwise(full[l:]))-set(self.edges[l:])
                            priority=(sum(not self.anchors[edge] for edge in edits),
                                      abs(self.inc[pattern[-1]]-self.inc[lower[0]]),-size)
                            families.setdefault((pattern[-1],lower[0]),[]).append(
                                (priority,start,size,place,k,full))
            for repetition in range(2):
                for _,proposals in sorted(families.items(),key=lambda row:min(p[0] for p in row[1])):
                    ranked=sorted(proposals,key=lambda x:x[:5])
                    if repetition<len(ranked):
                        _,start,size,place,k,full=ranked[repetition]
                        yield ("final_gate",start,size,place,k),l,180,full
        else:
            # Moving the polar low passage changes which high-shell orbits
            # connect to each shell crossing, while preserving complete coverage.
            start,out,back,end=self.shells
            low=self.path[out+1:back+1]
            shell=self.path[start+1:out+1]+self.path[back+1:end+1]
            rest=self.path[end+1:]
            for j in range(out-start-12,out-start+12):
                if not 0<j<len(shell):
                    continue
                for direction,group in ((0,low),(1,low[::-1])):
                    path=(*self.path[:start+1],*shell[:j],*group,*shell[j:],*rest)
                    if tuple(self.path)!=path and self.all_allowed(path) and self.route_floor(path)<=5:
                        yield ("crossing",j,direction),start,180,path

    def run(self):
        done={tuple(row["key"]) for row in self.state["trials"]}
        for key,l,r,order in self.neighborhoods():
            if time.monotonic()>=self.stop:
                break
            if key in done:
                continue
            started=time.monotonic()
            stop=min(self.stop,started+self.args.trial_seconds)
            stats=Counter()
            observations={}
            try:
                if order is None:
                    members=self.path[l+1:r]
                    reserve=self.route_floor(self.path[r:])
                    due=arrival(self.legs[r-1])+.85
                    repairs=self.block(tuple(self.legs[:l]),members,self.path[r],due,
                                       reserve,min(stop,time.monotonic()+6.),stats)
                    observations["window_arrivals"]=[arrival(legs[-1]) for legs in repairs]
                    for legs in repairs[:2]:
                        path=tuple(self.build(legs)["path"]+self.path[r+1:])
                        if time.monotonic()<stop:
                            self.complete(path,legs,self.args.goal,.078125,stop,key,observations)
                else:
                    self.complete(order,tuple(self.legs[:l]),self.args.goal,.078125,stop,key,observations)
            except BudgetExpired:
                observations["time_cap"]=True
            self.count.update({k:v for k,v in stats.items() if k!="block_blocked_depth"})
            self.state["trials"].append({"key":key,"seconds":time.monotonic()-started,
                                         "search":dict(stats),**observations})
            self.save()
            print(key,observations,flush=True)
        self.state["wall_seconds"]=self.state.get("wall_seconds",0.)+time.monotonic()-self.started
        self.save()


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--directory",required=True,type=Path)
    p.add_argument("--seed-record",required=True,type=Path)
    p.add_argument("--baseline-record",type=Path)
    p.add_argument("--anchors",type=Path,nargs="*",default=())
    p.add_argument("--seconds",type=float,default=150.)
    p.add_argument("--trial-seconds",type=float,default=10.)
    p.add_argument("--beam",type=int,default=48)
    p.add_argument("--per-size",type=int,default=12)
    p.add_argument("--phase",type=float,default=.017)
    p.add_argument("--goal",type=float,default=194.)
    p.add_argument("--mode",choices=("windows","exchange","crossing","final_gate"),required=True)
    p.add_argument("--positions",type=int,nargs="+")
    p.add_argument("--spans",type=int,nargs="+",default=(7,10,12))
    p.add_argument("--save-prefixes",action="store_true")
    p.add_argument("--reserve-gate",action="store_true")
    p.add_argument("--polish",action="store_true")
    Surgery(p.parse_args()).run()
