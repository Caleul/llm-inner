import sys,json,pathlib,time,resource
sys.path.insert(0,'helpers')
from direct_sympy_scan_backend import EquivalentScans
state=json.loads(pathlib.Path('artifacts/direct-sympy-square/state/frontier.json').read_text())['payload'];r=state['records'][-1]
region=(pathlib.Path('artifacts/direct-sympy-square/state/objects')/(r['digest']+'.expr')).read_text();backend=EquivalentScans();rows=[]
for label,text in [('near-match','('+region[:-1]+'!'),('present','('+region+')')]:
 start=time.monotonic();expected=text.find(region);old=time.monotonic()-start
 start=time.monotonic();actual=backend.literal_find(text,region);new=time.monotonic()-start
 assert actual==expected
 rows.append({'case':label,'regionCharacters':len(region),'expressionCharacters':len(text),'stdlibSeconds':old,'anchoredSeconds':new,'position':actual})
print(json.dumps({'runs':rows,'peakRSSBytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'scope':'Exact literal search on the actual gate producer; not compilation or coordinate parity.'}))
