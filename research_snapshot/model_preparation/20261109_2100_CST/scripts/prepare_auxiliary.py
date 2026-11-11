"""Fetch official non-HF dependencies and inventory read-only DreamSim cache."""
from pathlib import Path
import hashlib
import json
import os
import fcntl
import shutil
import tarfile
import urllib.request
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
lock = (ROOT / 'state/auxiliary.lock').open('w')
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
state_path = ROOT / 'state/auxiliary.json'
state = json.loads(state_path.read_text()) if state_path.exists() else {'artifacts': {}}
state.update(status='running', pid=os.getpid())
def save():
    state['updated_at'] = datetime.now(timezone.utc).isoformat()
    tmp = state_path.with_suffix('.tmp')
    tmp.write_text(json.dumps(state, indent=2))
    tmp.replace(state_path)

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        while b := f.read(8 * 1024 * 1024):
            h.update(b)
    return h.hexdigest()

def fetch(key, url, relative):
    p = ROOT / relative
    row = state['artifacts'].get(key, {})
    if p.is_file() and row.get('sha256') == sha(p):
        return p
    row = {'url': url, 'path': str(p), 'status': 'downloading', 'hash_kind': 'local content hash; upstream digest not provided'}
    state['artifacts'][key] = row
    save()
    try:
        if shutil.disk_usage(ROOT).free < 10 * 1024**3:
            raise RuntimeError('InsufficientDiskReserve10GiB')
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(p.suffix + '.partial')
        request = urllib.request.Request(url, headers={'User-Agent': 'nuScenes-baseline-preparation'})
        with urllib.request.urlopen(request, timeout=60) as response, tmp.open('wb') as out:
            expected = int(response.headers.get('Content-Length', 0))
            while block := response.read(1024 * 1024):
                out.write(block)
        if expected and tmp.stat().st_size != expected:
            raise RuntimeError('IncompleteDownload')
        row.update(sha256=sha(tmp), bytes=tmp.stat().st_size, status='downloaded_length_checked')
        tmp.replace(p)
        save()
        print(key, row['status'], row['bytes'], flush=True)
        return p
    except Exception as e:
        row.update(status='failed', error=type(e).__name__, http_status=getattr(e, 'code', None))
        save()
        print(key, row['status'], row['error'], flush=True)
        return None

save()
fetch('realesrgan_x4plus', 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth', 'weights/auxiliary/RealESRGAN_x4plus.pth')
fetch('dinov2_vitl14_reg', 'https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_reg4_pretrain.pth', 'weights/auxiliary/dinov2_vitl14_reg4_pretrain.pth')
try:
    if not state.get('dinov2_code_commit'):
        request = urllib.request.Request('https://api.github.com/repos/facebookresearch/dinov2/commits/main', headers={'User-Agent': 'nuScenes-baseline-preparation'})
        with urllib.request.urlopen(request, timeout=30) as response:
            state['dinov2_code_commit'] = json.load(response)['sha']
        save()
    archive = fetch('dinov2_code', 'https://codeload.github.com/facebookresearch/dinov2/tar.gz/' + state['dinov2_code_commit'], 'sources/dinov2.tar.gz')
    if archive and not (ROOT / 'sources' / ('dinov2-' + state['dinov2_code_commit'])).exists():
        with tarfile.open(archive) as t:
            t.extractall(ROOT / 'sources', filter='data')
except Exception as e:
    state['dinov2_code_error'] = type(e).__name__

cache = Path(('/opt/rigid-benchmark/project/workplace1/reconviagen_demo/cache/refinement/dreamsim'.replace('/opt/rigid-benchmark/project', __import__("os").environ.get('RIGID_PROJECT_ROOT', '/opt/rigid-benchmark/project'))))
for name in ['dino_vitb16_pretrain.pth', 'dino_vitb16_single_lora/adapter_config.json', 'dino_vitb16_single_lora/adapter_model.safetensors']:
    p = cache / name
    state['artifacts']['dreamsim/' + name] = {'path': str(p), 'status': 'read_only_local_cache' if p.is_file() else 'missing',
        'sha256': sha(p) if p.is_file() else None, 'hash_kind': 'local content hash; upstream digest not provided'}
state['status'] = 'finished_with_blockers' if any(x['status'] in ['failed', 'missing'] for x in state['artifacts'].values()) or state.get('dinov2_code_error') else 'listed_auxiliary_files_ready'
save()
