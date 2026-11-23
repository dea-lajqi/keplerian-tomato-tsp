"""Enumerate geometry-admissible paths at the exact incumbent cut."""
from collections import Counter
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from continuous_windows_easy import FILES, SEED
from global_dp_easy import evaluator
from global_event_easy import geometry
from small_v2 import atomic_json


def main():
    mod=evaluator()
    normal,allowed=geometry(mod.udp,mod.MU_MOON)[0]
    route=json.loads(SEED.read_text())["path"]
    group={4,11,17}
    start=route.index(37)
    end=route.index(19)
    subset=set(route[start+1:end])
    report={}
    for budget in (2,3):
        count=Counter()
        sequences=[]

        def enumerate_segment(a,rest,spent,path):
            if not rest:
                b=19
                total=spent+int(not normal[a,b])
                if allowed[a,b] and total<=budget:
                    completed=(*path,b)
                    crosses=[(x,y) for x,y in zip(completed,completed[1:]) if (x in group)!=(y in group)]
                    count[tuple(crosses)]+=1
                    sequences.append(completed)
                return
            for b in sorted(rest):
                if a in group and rest&group and b not in group:
                    continue
                if not allowed[a,b]:
                    continue
                left=rest-{b}
                used=spent+int(not normal[a,b])
                needed=1 if b in group else 2 if left&group else 0
                if used+needed<=budget:
                    enumerate_segment(b,left,used,(*path,b))

        enumerate_segment(37,subset,0,(37,))
        report[str(budget)]={"orders":len(sequences),"crossing_pairs":[(p,v) for p,v in count.most_common()],
                             "first_edges":dict(Counter(p[1] for p in sequences)),
                             "paths":sequences}
        print(budget,len(sequences),count)
    atomic_json(FILES/"small_v9/geometry_audit.json",report)


if __name__=="__main__":
    main()
