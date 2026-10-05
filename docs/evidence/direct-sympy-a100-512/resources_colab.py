import json,os,subprocess
print(json.dumps({'cpuCount':os.cpu_count(),'meminfo':open('/proc/meminfo').read(),'nvidiaSMI':subprocess.run(['nvidia-smi','--query-gpu=name,memory.total,driver_version','--format=csv,noheader'],capture_output=True,text=True).stdout}))
