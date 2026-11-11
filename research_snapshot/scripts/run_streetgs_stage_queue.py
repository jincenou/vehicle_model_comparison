"""Stage converted StreetGS scenes without exposing heldout frames to training."""

import concurrent.futures
import json
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
SCENES = ['waymo_val_030', 'waymo_val_031', 'waymo_val_032', 'waymo_val_033',
          'waymo_val_034', 'waymo_val_035', 'waymo_val_037', 'waymo_val_038']
STATE = ROOT / 'state/streetgs_stage_queue.json'
LOGS = ROOT / 'logs'


def save(state):
    STATE.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATE.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(state, indent=2) + '\n')
    tmp.replace(STATE)


def stage(scene):
    log_path = LOGS / f'streetgs_stage_{scene}.log'
    with log_path.open('w') as log:
        command = [sys.executable, '-B', str(ROOT / 'scripts/stage_streetgs_train_only.py'),
                   '--scene', scene]
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        return {'scene': scene, 'status': 'failed', 'exit_code': result.returncode,
                'log': str(log_path)}
    audit_path = ROOT / f'reports/streetgs_train_only/{scene}.json'
    audit = json.loads(audit_path.read_text())
    assert audit['status'] == 'physically_isolated_train_only'
    assert audit['heldout_rgb_lidar_present_in_train_root'] is False
    return {'scene': scene, 'status': 'completed', 'audit': str(audit_path),
            'train_frames': audit['train_frames'], 'heldout_frames': audit['heldout_frames']}


def main():
    conversion = json.loads((ROOT / 'state/streetgs_conversion_queue.json').read_text())
    assert conversion['status'] == 'completed'
    assert all(job['status'] == 'completed' for job in conversion['jobs'])
    for scene in SCENES:
        assert (ROOT / f'inputs/streetgs_full_conversion/{scene[-3:]}/pointcloud.npz').is_file()
    LOGS.mkdir(parents=True, exist_ok=True)
    state = {'status': 'running', 'started_at': time.time(), 'scenes': SCENES,
             'jobs': []}
    save(state)
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        pending = {pool.submit(stage, scene): scene for scene in SCENES}
        for done in concurrent.futures.as_completed(pending):
            try:
                result = done.result()
            except Exception as exc:
                result = {'scene': pending[done], 'status': 'failed', 'error': repr(exc)}
            state['jobs'].append(result)
            save(state)
            print(json.dumps(result), flush=True)
    state['status'] = 'completed' if all(j['status'] == 'completed' for j in state['jobs']) else 'finished_with_failures'
    state['finished_at'] = time.time()
    save(state)


if __name__ == '__main__':
    main()
