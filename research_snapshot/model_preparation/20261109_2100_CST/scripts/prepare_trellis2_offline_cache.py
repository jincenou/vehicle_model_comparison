"""Create an isolated HF-layout cache from already verified exact files."""
import hashlib
import json
from pathlib import Path
P=Path(__file__).resolve().parents[1]
state=json.loads((P/'state/trellis2.json').read_text())
assert state['status']=='weights_verified_pending_runtime' and state['failure_count']==0
root=P/'cache/trellis2_offline/hub'
records=[]
for f in state['files'].values():
    assert f['status'].startswith('verified')
    source=Path(f['destination']);assert source.is_file() and source.stat().st_size==f['size']
    h=hashlib.sha256()
    with source.open('rb')as stream:
        for b in iter(lambda:stream.read(8*1024**2),b''):h.update(b)
    assert h.hexdigest()==f['sha256']
    repo=root/('models--'+f['repo'].replace('/','--'))
    target=repo/'snapshots'/f['revision']/f['filename'];target.parent.mkdir(parents=True,exist_ok=True)
    if target.is_symlink():assert target.resolve()==source.resolve()
    else:target.symlink_to(source)
    (repo/'refs').mkdir(exist_ok=True);(repo/'refs/main').write_text(f['revision'])
    records.append({'source':str(source),'target':str(target),'sha256':h.hexdigest()})
(P/'state/trellis2_offline_cache.json').write_text(json.dumps({'status':'verified','files':records},indent=2)+'\n')
print('verified cached files',len(records))
