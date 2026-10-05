"""Process-shared output budget; expressions plus conditions, separate from RAM.

Each lease belongs to a compilation attempt, not a runtime intermediate.
Claims precede expansion into file bytes. A failed attempt releases its claim;
completed files keep their exact size charged until the vector is emitted.
"""
import fcntl,json,os
from pathlib import Path
from direct_sympy_savepoints import atomic,canonical

class ArtifactBudget:
    def __init__(self,path,limit,*,create=False):
        if type(limit)is not int or limit<1:raise ValueError('Positive accumulated artifact budget required')
        self.path=Path(path);self.limit=limit;self.path.parent.mkdir(parents=True,exist_ok=True)
        if create:
            with self.path.with_suffix('.lock').open('a')as lock:
                fcntl.flock(lock,fcntl.LOCK_EX)
                if self.path.exists():raise ValueError('Artifact budget already exists; inspect before restarting')
                atomic(self.path,canonical({'limit':limit,'claims':{},'peakClaimedCharacters':0}))
    def change(self,name,amount,*,exact=False):
        if type(amount)is not int or amount<0:raise ValueError('Nonnegative claim required')
        with self.path.with_suffix('.lock').open('a')as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            state=json.loads(self.path.read_text())
            if state['limit']!=self.limit:raise ValueError('Incompatible accumulated artifact budget')
            claims=state['claims'];requested=amount if exact else max(amount,claims.get(name,0))
            total=sum(claims.values())-claims.get(name,0)+requested
            if total>self.limit:raise ValueError(f'Accumulated expression and condition budget exceeded: requested={total} limit={self.limit}')
            if requested:claims[name]=requested
            else:claims.pop(name,None)
            state['peakClaimedCharacters']=max(total,state['peakClaimedCharacters'])
            atomic(self.path,canonical(state))
    def lease(self,name):return ArtifactLease(self,name)

class ArtifactLease:
    def __init__(self,budget,name):
        if not isinstance(name,str)or not name:raise ValueError('Unique lease name required')
        self.budget=budget;self.name=name;self.claimed=0;self.committed=False
    def claim(self,amount):
        if amount>self.claimed:self.budget.change(self.name,amount);self.claimed=amount
    def commit(self,amount):
        self.budget.change(self.name,amount,exact=True);self.claimed=amount;self.committed=True
    def __enter__(self):return self
    def __exit__(self,*error):
        if not self.committed:self.budget.change(self.name,0,exact=True)
