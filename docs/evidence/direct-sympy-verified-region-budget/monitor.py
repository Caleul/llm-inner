"""Observe one owned compilation process; RAM is separate from string budget."""
import json
from pathlib import Path
import sys
import time
import psutil

process=psutil.Process(int(sys.argv[1]))
identity=process.create_time()
started=time.monotonic()
available=psutil.virtual_memory().available
limit=int(available*.8)
peak=0;samples=0;reason='process ended'
while process.is_running() and process.status()!=psutil.STATUS_ZOMBIE:
    if process.create_time()!=identity:raise RuntimeError('Process identity changed')
    children=process.children(recursive=True)
    rss=0
    for item in [process,*children]:
        try:rss+=item.memory_info().rss
        except psutil.NoSuchProcess:pass
    peak=max(peak,rss);samples+=1
    if rss>limit or time.monotonic()-started>600:
        reason='RAM admission exceeded' if rss>limit else '600-second observation limit'
        for item in reversed(children):
            try:item.kill()
            except psutil.NoSuchProcess:pass
        process.kill();break
    time.sleep(.2)
Path(sys.argv[2]).write_text(json.dumps({'pid':process.pid,'createTime':identity,
    'initialAvailableRamBytes':available,'ramLimitBytes':limit,'peakAggregateRssBytes':peak,
    'samples':samples,'seconds':time.monotonic()-started,'stop':reason},indent=2)+'\n')
