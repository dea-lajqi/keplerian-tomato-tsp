"""Reorder the entire last high-shell block on a shared exact event graph."""
import argparse
from collections import Counter
import json
from pathlib import Path
import random
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from global_event_easy import fit_transfer_window
from medium_coupled.checkpoint import atomic_json
from medium_coupled.exact_terminal import ExactTerminal, pareto
from medium_coupled.gate_prefix_polish import PrefixPolish
from medium_coupled.meet import SideGrid
from medium_coupled.run import Paused, solve
from pareto_easy import BudgetExpired, arrival


class HighExact(ExactTerminal):
    def run(self):
        gate=self.shells[-1]
        boundary=gate if self.args.right_index is None else self.args.right_index
        left=boundary-self.args.width
        group=tuple(self.path[left+1:boundary+1])
        if any(self.axis(node) < 8e6 for node in self.path[left:boundary+1]):
            raise ValueError("Normal high-shell block crosses into a low-shell group")
        prefix=tuple(self.legs[:left])
        if self.args.spend_at_entry:
            i=left-1
            a,b=self.edges[i]
            release=arrival(self.legs[i-1])
            due=arrival(self.legs[i])
            options=[]
            for phase in range(4):
                rows,_=fit_transfer_window(self.udp,a,b,release,due,self.cap,
                  min(self.stop,time.monotonic()+4),random.Random(9801+i+phase))
                options.extend(leg for leg in rows if leg[-1]>self.normal)
            if not options:
                self.state["entry_exception_missing"]=True
                self.save()
                return
            leg=min(options,key=arrival)
            prefix=(*prefix[:-1],leg)
            self.state["entry_exception"]={"leg":leg,"immediate_gain":due-arrival(leg)}
            self.save()
        prefix_exceptions=sum(leg[-1]>self.normal for leg in prefix)
        if prefix_exceptions>4:
            raise ValueError("No exception remains for the final shell crossing")
        release=arrival(prefix[-1])
        stop=min(self.stop-30.,time.monotonic()+self.args.atlas_seconds)
        if self.args.duration_cap is not None:
            self.args.max_flight=min(self.args.max_flight,self.args.duration_cap)
        if self.args.atlas_input:
            saved=json.loads(self.args.atlas_input.read_text())
            events={tuple(map(int,k.split(","))):pareto(v) for k,v in saved["rows"].items()}
            num=len(saved["done"])
        else:
            events,num=self.build_atlas(group,release,stop)
        if self.args.duration_cap is not None:
            events={key:tuple(leg for leg in rows if leg[3]<=self.args.duration_cap)
                    for key,rows in events.items()}
        self.state["atlas"]={"pair_meshes":num,"events":sum(map(len,events.values())),
                             "nodes":len(group),"release":release}
        self.save()
        starts={}
        a=self.path[left]
        stop=min(self.stop,time.monotonic()+16.)
        grid=SideGrid(self.udp,release,self.args.goal,.059375,.013,stop,False)
        for b in group:
            if not self.normal_edges[a,b] or time.monotonic()>=stop:
                continue
            bounds=max(release,min(self.args.goal,release+7.1))
            try:
                rows,_=fit_transfer_window(self.udp,a,b,release,bounds,self.normal,stop,
                                random.Random(42391+b))
                self.count["initial_queries"]+=1
                options=[*rows,*self.anchors[a,b]]
                for row in grid.edges(a,b,0.,4.,7.,False):
                    if row is not None:
                        d,f=release+row[2],row[3]
                        dv=float(self.udp.compute_transfer(a,b,d,f))
                        options.append((a,b,d,f,dv))
                accepted=[l for l in options if l[-1]<=self.normal and l[2]>=release-1e-9]
                if accepted:
                    starts[b]=(*prefix,min(accepted,key=arrival))
            except BudgetExpired:
                break
        self.count.update(grid.stats)
        edges={k:(tuple(l[2] for l in v),v) for k,v in events.items()}
        visited={tuple(t["key"]) for t in self.state["trials"]}
        poolpath=self.args.directory/("gate_pool.json" if boundary==gate else "mid_pool.json")
        pool=json.loads(poolpath.read_text()) if poolpath.exists() else {}
        carry=[]
        for due in self.args.gates:
            key=("normal_high",due)
            if key in visited or time.monotonic()>=self.stop:
                continue
            begin=time.monotonic()
            stats=Counter()
            output={}
            try:
                metrics,paths=self.execute(group,{b:arrival(p[-1]) for b,p in starts.items()},
                                           events,due,False,carry_boundary=self.args.carry_boundary)
                carry.extend(metrics.pop("carried_paths",[]))
                if self.args.repair_frontier and not paths:
                    candidates=metrics.pop("one_missing",[])
                    allowed=[];added=[]
                    for release,a,b in candidates:
                        if self.normal_edges[a,b] and due-release>self.udp.min_tof:
                            allowed.append((release,a,b))
                    for release,a,b in allowed[:24]:
                        if time.monotonic()>=self.stop:
                            break
                        try:
                            fits,_=fit_transfer_window(self.udp,a,b,release,due,self.normal,
                                        min(self.stop,time.monotonic()+3),random.Random(72191+a*181+b))
                        except BudgetExpired:
                            break
                        stats["frontier_continuous_queries"]+=1
                        if fits:
                            old=events.get((a,b),())
                            if self.args.duration_cap is not None:
                                fits=[leg for leg in fits if leg[3]<=self.args.duration_cap]
                            newer=pareto((*old,*fits))
                            events[a,b]=newer
                            edges[a,b]=tuple(l[2] for l in newer),newer
                            added.extend(fits)
                    output["one_missing_count"]=len(candidates)
                    output["one_missing_time_and_geometry"]=len(allowed)
                    output["measured_repair_events"]=added
                    if added:
                        metrics,paths=self.execute(group,{b:arrival(p[-1]) for b,p in starts.items()},
                                                   events,due,False)
                output.update(metrics)
                output["earliest_gates"]={str(order[-1]):day for day,order in paths}
                for day,order in sorted(paths)[:10]:
                    partial=self.reconstruct(order,starts,edges)
                    b=partial["path"][-1]
                    label=f"{b}-{left}"
                    def accept(sol):
                        actual,spent=PrefixPolish.measure(self,sol,stats)
                        if (not actual or spent>prefix_exceptions or
                                (self.args.duration_cap is not None and any(
                                    leg[3]>self.args.duration_cap+1e-9 for leg in actual[left:]))):
                            stats["rejected_prefix_budget"]+=int(bool(actual))
                            return False
                        finish=arrival(actual[-1])
                        if label not in pool or finish<pool[label]["arrival"]-1e-9:
                            pool[label]={"legs":actual,"remaining":self.path[boundary+1:],
                                         "arrival":finish,"spend":spent,"plan":key}
                            atomic_json(poolpath,pool)
                        return True
                    accept(partial)
                    try:
                        solve(self.udp,partial,left,3.7,0.,min(self.stop,time.monotonic()+4.),
                              accept,stats,cap_flags=[l[-1]>self.normal for l in prefix]+[False]*len(group))
                    except Paused:
                        if time.monotonic()>=self.stop:
                            break
            except TimeoutError:
                output["time_cap"]=True
            self.count.update(stats)
            self.state["trials"].append({"key":key,"seconds":time.monotonic()-begin,**output})
            self.save()
            print(key,output,flush=True)
        if self.args.carry_boundary and boundary<gate:
            attempted=set()
            for t,missing,order in sorted(carry):
                if len(attempted)>=30 or time.monotonic()>=self.stop:
                    break
                partial=self.reconstruct(order,starts,edges)
                actual,spent=PrefixPolish.measure(self,partial,self.count)
                if not actual or spent>prefix_exceptions:
                    continue
                pending=self.path[boundary+1:]
                for j in range(min(9,gate-boundary)+1):
                    if time.monotonic()>=self.stop or len(attempted)>=30:
                        break
                    rest=(*pending[:j],missing,*pending[j:])
                    new=tuple(partial["path"])+rest
                    key=("carry_missing",missing,order[-1],j)
                    if key in attempted or key in visited:
                        continue
                    if not all(self.normal_edges[a,b] for a,b in
                               zip(new[len(actual):gate],new[len(actual)+1:gate+1])):
                        continue
                    attempted.add(key)
                    begin=time.monotonic()
                    details={"first_block_arrival":t}
                    try:
                        self.complete(new,tuple(actual),self.args.mission_goal,.0615234375,
                                      min(self.stop,begin+9.),key,details)
                    except BudgetExpired:
                        details["time_cap"]=True
                    self.state["trials"].append({"key":key,"seconds":time.monotonic()-begin,**details})
                    self.save()
                    print(key,details,flush=True)
        if boundary!=gate:
            already={tuple(t["key"]) for t in self.state["trials"]}
            discretionary={l[:2] for l in self.legs if not self.args.spend_at_entry and l[-1]>self.normal and
                           min(self.axis(l[0]),self.axis(l[1]))*2>=max(self.axis(l[0]),self.axis(l[1]))}
            for label,item in sorted(pool.items(),key=lambda kv:kv[1]["arrival"])[:8]:
                key="propagate_high_block",label
                if key in already or time.monotonic()>=self.stop:
                    continue
                begin=time.monotonic()
                prefix=tuple(map(tuple,item["legs"]))
                path=tuple(self.build(prefix)["path"]+item["remaining"])
                output={}
                try:
                    self.complete(path,prefix,self.args.mission_goal,.0712890625,
                                 min(self.stop,begin+12.),key,output,
                                 permitted_extra=discretionary)
                except BudgetExpired:
                    output["time_cap"]=True
                self.state["trials"].append({"key":key,"seconds":time.monotonic()-begin,**output})
                self.save()
        self.state["wall_seconds"]=self.state.get("wall_seconds",0.)+time.monotonic()-self.started
        self.save()


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--directory",type=Path,required=True)
    p.add_argument("--seed-record",type=Path,required=True)
    p.add_argument("--baseline-record",type=Path)
    p.add_argument("--anchors",type=Path,nargs="*",default=())
    p.add_argument("--seconds",type=float,default=245.)
    p.add_argument("--atlas-seconds",type=float,default=161.)
    p.add_argument("--width",type=int,choices=range(6,21),default=20)
    p.add_argument("--goal",type=float,default=189.0)
    p.add_argument("--gates",type=float,nargs="+",default=(187.4,188.0,188.7,189.0))
    p.add_argument("--right-index",type=int,choices=range(40,160))
    p.add_argument("--mission-goal",type=float,default=191.65)
    p.add_argument("--spend-at-entry",action="store_true")
    p.add_argument("--atlas-input",type=Path)
    p.add_argument("--seed-atlas",type=Path,help="Import events and add the requested mesh.")
    p.add_argument("--duration-cap",type=float,help="Bound flight duration inside this exact block.")
    p.add_argument("--repair-frontier",action="store_true")
    p.add_argument("--carry-boundary",action="store_true")
    p.add_argument("--mesh",type=float,nargs=2,action="append",
                   metavar=("STEP","PHASE"),help="Absolute lattice step and phase relative to the prefix.")
    p.set_defaults(mode="high_atlas",phase=None,beam=None,max_flight=8.,
                   seed_atlas=None,meshes=((.0833333333333333,.007),(.064453125,.023)),
                   reserve_gate=True,save_prefixes=True,polish=True)
    args=p.parse_args()
    if args.mesh:
        args.meshes=tuple(tuple(pair) for pair in args.mesh)
    HighExact(args).run()
