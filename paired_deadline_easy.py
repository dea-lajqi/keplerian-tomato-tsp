"""Join fresh forward and target-deadline suffix trajectories around shell crossings."""
from collections import defaultdict
import argparse
import time
from pathlib import Path

import numpy as np

from focus_easy import retime
from global_dp_easy import Grid
from paired_crossings_easy import Search, SEED
from pareto_easy import arrival


class DeadlineSearch(Search):
    def reverse(self, path, goal, wait, phase, budget):
        grid = Grid(self.module, .0625, phase, self.deadline, finish_day=goal,directional=True)
        self.stats["reverse_passes"] += 1
        anchors = defaultdict(list)
        for a,b in zip(path,path[1:]):
            for leg in self.anchors.get((a,b),()):
                dep = goal-arrival(leg)
                if dep>=0.:
                    anchors[b,a].append((b,a,dep,leg[3],leg[4]))
        schedules,depth = retime(grid,path[::-1],anchors,wait=wait,flight=9.5,max_exceptions=budget)
        restored = []
        for schedule in schedules:
            legs = []
            for b,a,d,f,_ in reversed(schedule):
                dep = goal-d-f
                if dep<0.:
                    break
                dv = self.cost(a,b,dep,f)
                if dv>self.udp.dv_threshold_exception+1e-6:
                    break
                legs.append((a,b,dep,f,dv))
            else:
                restored.append(tuple(legs))
        return restored,depth

    def deadline_trial(self):
        pool = sorted(self.population.items(),key=lambda row:row[1][0])
        i = self.iteration
        goal = (101., 105., 108., 110., 111.4, 113., 117.)[i%7]
        if i%3==2 and not self.args.five_crossings:
            parent_path,parent_entry = pool[i//3 % min(len(pool),12)]
            path = parent_path
        else:
            n = self.udp.n_tomatoes
            groups = sorted(tuple(sorted(g)) for g in self.topology.cuts if len(g)==3)
            self.rng.shuffle(groups)
            start_node = self.rng.choice(groups[0])
            end_node = self.rng.choice(groups[1])
            if self.args.five_crossings:
                end_node = self.rng.choice([j for j in range(n) if self.shell[j]])
                if i%2:
                    start_node,end_node = end_node,start_node
            forced = ((n,start_node),(end_node,n))
            proposal = self.topology.propose(self.rng,min(self.deadline,time.monotonic()+8.),
                                              set(zip(self.reference,self.reference[1:])),
                                              (9,14,20)[i%3],2.1,forced=forced)
            if proposal:
                path,_,_ = proposal
                parent_entry = None
            else:
                self.stats["topology_misses"] += 1
                if self.args.five_crossings:
                    return
                path,parent_entry = pool[i % min(len(pool),12)]
        crossings = [j for j,(a,b) in enumerate(zip(path,path[1:]))
                     if self.shell[a]!=self.shell[b]]
        middle = [j for j in crossings if 7<j<len(path)-8] or crossings
        pivot = middle[(i//2)%len(middle)]
        radius = (3,4,5,6)[i%4]
        left = max(1,pivot-radius)
        right = min(len(path)-2,pivot+radius+1)
        trial = {"iteration":i,"kind":"deadline_join","path":path,"goal":goal,
                 "block":[left,right],"crossing":path[pivot:pivot+2],
                 "changed_arcs":len(set(zip(path,path[1:]))-set(zip(self.reference,self.reference[1:])))}
        trial["crossings"] = len(crossings)
        self.history.append(trial)
        grid = self.grid(.0625,(i%4)*.013)
        if self.args.five_crossings:
            complete,depth = retime(grid,path,self.anchors,wait=7.,flight=9.5)
            trial["forward_complete_depth"] = depth
            for candidate in complete:
                self.submit(candidate,trial,"five-shell-crossing route")
        pairs = list(zip(path[left:right],path[left+1:right+1]))
        mandatory = sum(not self.allowed[0][a,b] for a,b in pairs)
        lack = lambda seq: sum(not self.allowed[0][a,b] for a,b in zip(seq,seq[1:]))
        prefix_min, suffix_min = lack(path[:left+1]),lack(path[right:])
        budget = self.udp.n_dv_exception
        allocation_slack = budget-mandatory-prefix_min-suffix_min
        if allocation_slack<0:
            self.stats["allocation_rejections"] += 1
            return
        prefixes,d = retime(grid,path[:left+1],self.anchors,wait=5.,flight=9.,
                           max_exceptions=prefix_min+allocation_slack)
        suffixes,e = self.reverse(path[right:],goal,5.,(i%4)*.009,suffix_min+allocation_slack)
        trial["forward_depth"],trial["backward_depth"] = d,e
        if not prefixes or not suffixes:
            return
        self.stats["two_sided_profiles"] += 1
        # Keep every nondominated exception count, including the normal-only
        # boundary when the faster boundary spends the last spare exception.
        for prefix in prefixes:
            release = arrival(prefix[-1])
            for suffix in suffixes:
                due = suffix[0][2]
                outside = sum(l[4]>self.udp.dv_threshold for l in (*prefix,*suffix))
                budget = self.udp.n_dv_exception-outside
                trial["best_join_span"] = max(due-release,trial.get("best_join_span",-1000.))
                if budget < mandatory:
                    self.stats["join_budget_shortages"] += 1
                    continue
                if due-release < len(pairs)*self.udp.min_tof:
                    self.stats["join_chronology_shortages"] += 1
                    continue
                seeds,_ = retime(grid,path[:right+1],self.anchors,prefix=prefix,
                                 wait=4.,flight=8.5,
                                 max_exceptions=self.udp.n_dv_exception-
                                 sum(l[4]>self.udp.dv_threshold for l in suffix))
                if seeds:
                    block = seeds[0][left:]
                    if arrival(block[-1])<=due+1e-6:
                        self.submit((*prefix,*block,*suffix),trial,"lattice shell join")
                    seeds = [block]
                # Unanchored block times from the archive are projected onto
                # the two boundaries as a near-feasible multiple-shooting seed.
                old = parent_entry[1][left:right] if parent_entry else None
                if old is None:
                    old = tuple(seeds[0]) if seeds else tuple(
                        min(self.anchors.get(pair,((a,b,0.,2.2,0.),)),
                            key=lambda l:l[3]) for pair in pairs for a,b in (pair,))
                durations = np.array([l[3] for l in old])
                available = max(len(pairs)*self.udp.min_tof,due-release)
                for frac in (1.,.83):
                    times = np.maximum(self.udp.min_tof,durations*min(frac,available/durations.sum()))
                    slack = max(0.,available-times.sum())/len(times)
                    dep = release+np.r_[0.,np.cumsum(times[:-1]+slack)]
                    init = np.column_stack((dep,times))
                    for variant,limits in enumerate(self.limits(pairs,old,budget)):
                        if variant>=2:
                            break
                        block = self.fit(pairs,init,release,due,limits,2.8,trial)
                        if block is None:
                            continue
                        end = arrival(block[-1])
                        trial["earliest_block_end"] = min(end,trial.get("earliest_block_end",1000.))
                        if end<=due+1e-6:
                            self.submit((*prefix,*block,*suffix),trial,"deadline-constrained shell join")
                        else:
                            self.finish(path,(*prefix,*block,*suffix),(*prefix,*block),trial)

    def topology_trial(self):
        self.deadline_trial()

    def crossing_trial(self):
        self.deadline_trial()


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds",type=float,default=540.)
    parser.add_argument("--iterations",type=int,default=1000)
    parser.add_argument("--seed",type=int,default=9210)
    parser.add_argument("--incumbent",type=Path,default=SEED)
    parser.add_argument("--resume",type=Path)
    parser.add_argument("--five-crossings",action="store_true")
    DeadlineSearch(parser.parse_args()).run()
