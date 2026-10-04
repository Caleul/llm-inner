"""Compiler entry with equivalent scans and separate execution provenance."""
import hashlib
import json
import faulthandler
import signal
from pathlib import Path
import time
from direct_sympy_scan_backend import BACKEND_VERSION, install
from direct_sympy_checkpoint import main


if __name__=='__main__':
    faulthandler.register(signal.SIGUSR1)
    started=time.monotonic()
    with install() as backend:
        try:result=main()
        finally:
            print(json.dumps({'scanBackend':BACKEND_VERSION,
                'sources':{name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in ('direct_sympy_scan_backend.py','direct_sympy_scan_cli.py')},
                'elapsedSeconds':time.monotonic()-started,**backend.summary()}),flush=True)
    raise SystemExit(result)
