"""Replace source-image symlinks with byte-verified copies in StreetGS train roots."""

import concurrent.futures
import hashlib
import json
from pathlib import Path
import shutil
import threading
import time


ROOT = Path(__file__).resolve().parents[1]
SCENES = ['waymo_val_031', 'waymo_val_032', 'waymo_val_033',
          'waymo_val_034', 'waymo_val_038']
STATE = ROOT / 'state/streetgs_materialize_queue.json'
REPORTS = ROOT / 'reports/streetgs_materialized'


def hash_file(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def save(state):
    temp = STATE.with_suffix('.tmp.json')
    temp.write_text(json.dumps(state, indent=2) + '\n')
    temp.replace(STATE)


def run(scene):
    number = scene[-3:]
    train = ROOT / f'inputs/streetgs_train_only/{number}'
    full = ROOT / f'inputs/streetgs_full_conversion/{number}'
    split = json.loads((ROOT / f'configs/streetgs_source_splits/{scene}.json').read_text())
    audit = json.loads((ROOT / f'reports/streetgs_train_only/{scene}.json').read_text())
    assert audit['status'] == 'physically_isolated_train_only'
    assert train.resolve() == Path(audit['train_root']).resolve()
    assert len(split['train_frame_ids']) == audit['train_frames']
    REPORTS.mkdir(parents=True, exist_ok=True)
    lines = []
    total = 0
    copied = 0
    for folder in ('images', 'dynamic_mask', 'ego_pose'):
        for target in sorted((train / folder).iterdir()):
            assert target.is_file(), target
            if target.is_symlink():
                source = target.resolve()
            else:
                new_frame = int(target.name.split('_')[0].split('.')[0])
                old_frame = split['train_frame_ids'][new_frame]
                suffix = target.name[len(f'{new_frame:06d}'):]
                source = full / folder / f'{old_frame:06d}{suffix}'
            assert source.is_file() and source.parent == full / folder
            source_hash = hash_file(source)
            if target.is_symlink():
                temp = target.with_name(target.name + '.materializing')
                assert not temp.exists()
                shutil.copyfile(source, temp)
                target_hash = hash_file(temp)
                assert source_hash == target_hash
                temp.replace(target)
                copied += 1
            else:
                target_hash = hash_file(target)
                assert source_hash == target_hash
            assert not target.is_symlink()
            total += target.stat().st_size
            lines.append(json.dumps({'train_file': str(target), 'source_file': str(source),
                                     'sha256': source_hash, 'bytes': target.stat().st_size}))
    expected_images = audit['train_frames'] * 5
    assert len(list((train / 'images').iterdir())) == expected_images
    assert len(list((train / 'dynamic_mask').iterdir())) == expected_images
    assert len(list((train / 'ego_pose').iterdir())) == expected_images + audit['train_frames']
    file_manifest = REPORTS / f'{scene}_files.jsonl'
    file_manifest.write_text('\n'.join(lines) + '\n')
    result = {'status': 'verified_materialized_train_only', 'scene': scene,
              'source_split_sha256': audit['source_split_sha256'],
              'train_frames': audit['train_frames'], 'heldout_frames': audit['heldout_frames'],
              'verified_files': len(lines), 'copied_symlinks': copied, 'materialized_bytes': total,
              'file_manifest': str(file_manifest), 'file_manifest_sha256': hash_file(file_manifest),
              'heldout_rgb_lidar_present': False, 'completed_at': time.time()}
    (REPORTS / f'{scene}.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


def main():
    state = {'status': 'running', 'started_at': time.time(), 'eligible_scenes': SCENES,
             'jobs': []}
    save(state)
    lock = threading.Lock()
    def worker(scene):
        try:
            result = run(scene)
        except Exception as exc:
            result = {'scene': scene, 'status': 'failed', 'reason': repr(exc)}
        with lock:
            state['jobs'].append(result)
            save(state)
            print(json.dumps({'scene': scene, 'status': result['status']}), flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(worker, scene) for scene in SCENES]
        for future in futures:
            future.result()
    state['status'] = 'completed' if all(j['status'] == 'verified_materialized_train_only' for j in state['jobs']) else 'finished_with_failures'
    state['finished_at'] = time.time()
    save(state)


if __name__ == '__main__':
    main()
