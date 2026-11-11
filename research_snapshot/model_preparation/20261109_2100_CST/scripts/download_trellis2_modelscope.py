"""Download normal public ModelScope files and verify pinned cross-host hashes."""
import concurrent.futures
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import threading
import time
import urllib.error
import urllib.request

P=Path(__file__).resolve().parents[1]
PLAN=json.loads((P/'state/trellis2_modelscope_plan.json').read_text())
SP=P/'state/trellis2_modelscope_download.json'
lock=(P/'state/trellis2_modelscope_download.lock').open('a')
fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
mutex=threading.RLock()
state={'status':'running','pid':os.getpid(),'started_at':time.time(),'disk_reserve_gib':10,'files':{}}

def save():
    with mutex:
        state['updated_at']=time.time();tmp=SP.with_suffix('.tmp')
        tmp.write_text(json.dumps(state,indent=2)+'\n');tmp.replace(SP)

def digest(path):
    h=hashlib.sha256()
    with path.open('rb')as stream:
        for chunk in iter(lambda:stream.read(8*1024**2),b''):h.update(chunk)
    return h.hexdigest()

def download(rec):
    key=rec['repo']+'/'+rec['filename'];out=Path(rec['destination']);out.parent.mkdir(parents=True,exist_ok=True)
    partial=out.with_suffix(out.suffix+'.modelscope.part')
    with mutex:state['files'][key]={**rec,'status':'pending'};save()
    for attempt in range(1,4):
        try:
            if out.exists():
                assert out.stat().st_size==rec['size'] and digest(out)==rec['sha256']
            else:
                start=partial.stat().st_size if partial.exists() else 0
                assert shutil.disk_usage(P).free > (rec['size']-start)+10*1024**3,'DiskReserve10GiB'
                if start<rec['size']:
                    headers={'Range':f'bytes={start}-'} if start else {}
                    request=urllib.request.Request(rec['url'],headers=headers)
                    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
                    with opener.open(request,timeout=60)as response:
                        if response.status==206:
                            assert response.headers.get('Content-Range','').startswith(f'bytes {start}-')
                        elif start:start=0
                        with partial.open('ab'if start else'wb')as target:
                            last=0
                            while True:
                                chunk=response.read(4*1024**2)
                                if not chunk:break
                                assert shutil.disk_usage(P).free>10*1024**3,'DiskReserve10GiB'
                                target.write(chunk);start+=len(chunk)
                                if time.monotonic()-last>5:
                                    with mutex:state['files'][key].update(status='downloading',bytes=start,attempt=attempt);save()
                                    last=time.monotonic()
                assert partial.stat().st_size==rec['size'],f'Length {partial.stat().st_size()} != {rec["size"]}'
                assert digest(partial)==rec['sha256'],'SHA256 mismatch'
                partial.replace(out)
            if rec.get('expected_git_blob'):
                body=out.read_bytes();blob=hashlib.sha1(b'blob '+str(len(body)).encode()+b'\0'+body).hexdigest()
                assert blob==rec['expected_git_blob'],'Original HF git blob mismatch'
            with mutex:state['files'][key].update(status='verified',bytes=rec['size'],verified_sha256=rec['sha256']);save()
            print('verified',key,rec['size'],flush=True);return
        except Exception as error:
            with mutex:state['files'][key].update(status='failed',reason=repr(error),attempt=attempt);save()
            if isinstance(error,urllib.error.HTTPError)and error.code in (401,403):break
            if 'mismatch' in str(error) or 'DiskReserve' in str(error):break
            if attempt<3:time.sleep(5)

save()
with concurrent.futures.ThreadPoolExecutor(max_workers=2)as pool:
    list(pool.map(download,PLAN['files']))
state['status']='verified_all' if all(v['status']=='verified'for v in state['files'].values())else'finished_with_failures'
state['finished_at']=time.time();save()
if state['status']=='verified_all':
    oldpath=P/'state/trellis2.json';original=json.loads(oldpath.read_text())
    archive=P/'state/trellis2.before_modelscope_20261110.json'
    if not archive.exists():shutil.copyfile(oldpath,archive)
    for rec in original['files'].values():
        key=rec['repo']+'/'+rec['filename']
        if key in state['files']:
            entry=state['files'][key]
            rec.update(status='verified_downloaded',sha256=entry['verified_sha256'],
                       download_source='ModelScope public API',modelscope_revision=entry['revision'],
                       verified_against_original_hf_revision=True)
            rec.pop('reason',None)
    original.update(status='weights_verified_pending_runtime',failure_count=0,runtime_validated=False,
                    updated_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    supplemental_download_state=str(SP))
    temp=oldpath.with_suffix('.tmp');temp.write_text(json.dumps(original,indent=2)+'\n');temp.replace(oldpath)
