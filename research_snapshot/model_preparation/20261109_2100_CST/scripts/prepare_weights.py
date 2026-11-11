"""Pinned, resumable weight preparation. Existing sources are read-only."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
for key in list(os.environ):
    if 'proxy' in key.lower():
        os.environ.pop(key)
os.environ.setdefault('HF_HUB_DISABLE_XET', '1')
os.environ.setdefault('HF_HUB_DOWNLOAD_TIMEOUT', '60')
os.environ.setdefault('HF_HUB_ETAG_TIMEOUT', '30')
os.environ['HF_HUB_CACHE'] = str(ROOT / 'cache/hub')
from huggingface_hub import HfApi, hf_hub_download

def now():
    return datetime.now(timezone.utc).isoformat()

def digest(path, algorithm='sha256'):
    h = hashlib.new(algorithm)
    with path.open('rb') as f:
        if algorithm == 'sha1':
            h.update(f'blob {path.stat().st_size}\0'.encode())
        while chunk := f.read(8 * 1024 * 1024):
            h.update(chunk)
    return h.hexdigest()

def atomic(path, data):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    temp.replace(path)

def error(e):
    # Do not persist signed URLs, credentials, or request headers.
    return {'type': type(e).__name__, 'http_status': getattr(getattr(e, 'response', None), 'status_code', None)}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model', choices=['trellis2', 'hunyuan21', 'recon02'])
    args = parser.parse_args()
    lock = (ROOT / f'state/{args.model}.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state_path = ROOT / f'state/{args.model}.json'
    state = {'model': args.model, 'pid': os.getpid(), 'started_at': now(), 'status': 'inventory', 'files': {}, 'repositories': {}, 'runtime_validated': False}
    def save():
        state['updated_at'] = now()
        atomic(state_path, state)
    save()
    tasks = []
    ab = Path(('/opt/rigid-benchmark/assetbench'.replace('/opt/rigid-benchmark/assetbench', __import__("os").environ.get('RIGID_ASSETBENCH', '/opt/rigid-benchmark/assetbench')).replace('/opt/rigid-benchmark/project', __import__("os").environ.get('RIGID_PROJECT_ROOT', '/opt/rigid-benchmark/project'))))
    rc = Path(('/opt/rigid-benchmark/project/workplace1/reconviagen_demo/cache'.replace('/opt/rigid-benchmark/project', __import__("os").environ.get('RIGID_PROJECT_ROOT', '/opt/rigid-benchmark/project'))))
    if args.model in ['trellis2', 'hunyuan21']:
        old = json.loads((ab / 'state' / ('baseline_weight_downloads.json' if args.model == 'trellis2' else 'hunyuan_weight_downloads.json')).read_text())
        for key, entry in old['files'].items():
            if args.model == 'trellis2' and 'TRELLIS' not in key:
                continue
            repo = entry.get('repo', old.get('repo'))
            filename = entry.get('filename', key)
            candidates = [entry['path']] if entry.get('path') else []
            if repo == 'microsoft/TRELLIS-image-large':
                candidates.append(str(rc / 'Stable-X--trellis-vggt-v0-2' / filename))
            tasks.append(dict(repo=repo, revision=entry.get('revision', old.get('revision')), filename=filename,
                              size=entry['bytes_expected'], sha256=entry.get('sha256_expected') or entry.get('sha256'),
                              blob=None if entry.get('sha256_expected') else entry.get('git_blob_expected'), candidates=candidates))
    deps = {
        'trellis2': [
            ('facebook/dinov3-vitl16-pretrain-lvd1689m', 'ea8dc2863c51be0a264bab82070e3e8836b02d51', None),
            ('briaai/RMBG-2.0', '2fd7b4b0f2393ddb39a267fb62df6c010928a6c6', None)],
        'hunyuan21': [('facebook/dinov2-giant', '611a9d42f2335e0f921f1e313ad3c1b7178d206d', None)],
        'recon02': [
            ('Stable-X/trellis-vggt-v0-2', None, rc / 'Stable-X--trellis-vggt-v0-2'),
            ('Stable-X/vggt-object-v0-1', None, rc / 'Stable-X--vggt-object-v0-1'),
            ('ZhengPeng7/BiRefNet', 'e2bf8e4460fc8fa32bba5ea4d94b3233d367b0e4', rc / 'ZhengPeng7--BiRefNet')]
    }
    pins_path = ROOT / f'state/{args.model}_revisions.json'
    pins = json.loads(pins_path.read_text()) if pins_path.exists() else {}
    api = HfApi()
    for repo, revision, cached in deps[args.model]:
        try:
            info = api.model_info(repo, revision=revision or pins.get(repo), files_metadata=True)
            pins[repo] = info.sha
            atomic(pins_path, pins)
            state['repositories'][repo] = {'revision': info.sha, 'gated': info.gated}
            for f in info.siblings:
                name = f.rfilename
                if not (name.endswith(('.json', '.yaml', '.safetensors', '.py')) or name in ['LICENSE', 'README.md']):
                    continue
                if '/' in name and not name.startswith('ckpts/'):
                    continue
                lfs = f.lfs
                sha = getattr(lfs, 'sha256', None) if lfs else None
                tasks.append(dict(repo=repo, revision=info.sha, filename=name, size=f.size, sha256=sha,
                                  blob=f.blob_id if not lfs else None,
                                  candidates=[str(cached / name)] if cached else []))
        except Exception as e:
            state['repositories'][repo] = {'status': 'inventory_failed', **error(e)}
            save()
    atomic(ROOT / f'state/{args.model}_manifest.json', tasks)
    state['status'] = 'verifying_and_downloading'
    save()
    for task in tasks:
        key = task['repo'] + '/' + task['filename']
        dest = ROOT / 'weights' / task['repo'].replace('/', '--') / task['filename']
        row = {**task, 'destination': str(dest), 'status': 'checking', 'started_at': now()}
        state['files'][key] = row
        save()
        print(now(), key, 'checking', flush=True)
        try:
            def valid(p):
                if not p.is_file() or (task['size'] is not None and p.stat().st_size != task['size']):
                    return False
                sha = digest(p)
                if task['sha256'] and sha != task['sha256']:
                    return False
                if task['blob'] and digest(p, 'sha1') != task['blob']:
                    return False
                if not task['sha256'] and not task['blob']:
                    raise RuntimeError('NoTrustedChecksum')
                row['sha256'] = sha
                return True
            found = False
            for source in [dest] + [Path(x) for x in task['candidates']]:
                source_valid = valid(source)
                row.setdefault('candidate_checks', []).append({'path': str(source), 'exists': source.is_file(), 'verified': source_valid})
                if source_valid:
                    if source != dest:
                        dest.parent.mkdir(parents=True, exist_ok=True)
                        if dest.exists() or dest.is_symlink():
                            raise RuntimeError('DestinationConflict')
                        dest.symlink_to(source)
                    row.update(status='verified_reused' if dest.is_symlink() else 'verified_downloaded', source=str(source.resolve()))
                    found = True
                    break
            if not found:
                if dest.exists() or dest.is_symlink():
                    raise RuntimeError('ChecksumMismatchExistingDestination')
                # Leave sufficient room for the existing training queue.
                if shutil.disk_usage(ROOT).free < 10 * 1024**3 + (task['size'] or 0):
                    raise RuntimeError('InsufficientDiskReserve10GiB')
                row['status'] = 'downloading'
                save()
                for attempt in range(1, 4):
                    try:
                        p = Path(hf_hub_download(task['repo'], task['filename'], revision=task['revision'],
                                                local_dir=str(ROOT / 'weights' / task['repo'].replace('/', '--'))))
                        if not valid(p):
                            raise RuntimeError('DownloadedChecksumMismatch')
                        row.update(status='verified_downloaded', source=str(p))
                        break
                    except Exception as e:
                        row['attempt'] = attempt
                        row['error'] = error(e)
                        save()
                        if row['error']['http_status'] in [401, 403, 404] or attempt == 3:
                            raise
                        time.sleep(10 * attempt)
            row['finished_at'] = now()
        except Exception as e:
            row.update(status='blocked' if error(e)['http_status'] in [401, 403] else 'failed', error=error(e))
            if isinstance(e, RuntimeError):
                row['reason'] = str(e)
        print(now(), key, row['status'], flush=True)
        save()
    failures = sum(x['status'] not in ['verified_reused', 'verified_downloaded'] for x in state['files'].values())
    failures += sum(x.get('status') == 'inventory_failed' for x in state['repositories'].values())
    state.update(status='finished_with_blockers' if failures else 'listed_weights_verified', finished_at=now(), failure_count=failures)
    save()

if __name__ == '__main__':
    main()
