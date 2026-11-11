"""Train two fully source-covered nuScenes scenes for extraction only."""

import asyncio
import json
import shutil
import subprocess
import time
from pathlib import Path

from run_splatad_human_train_only_queue import ROOT, PYTHON, environment


STATE = ROOT / 'state/splatad_reconstruction_only_queue.json'
PRIOR = ROOT / 'state/splatad_strict_actor_queue.json'
SOCKET = ROOT / 'state/vehicle_cmp_tmux.sock'
SCENES = ('scene-0066', 'scene-0067')
GPUS = (3, 4)


def save(state):
    state['updated_at'] = time.time()
    temp = STATE.with_suffix('.tmp.json')
    temp.write_text(json.dumps(state, indent=2) + '\n')
    temp.replace(STATE)


def gpu_free(index):
    applications = set(subprocess.check_output(
        ['nvidia-smi', '--query-compute-apps=gpu_uuid', '--format=csv,noheader'],
        text=True).splitlines())
    cards = subprocess.check_output(
        ['nvidia-smi', '--query-gpu=index,uuid', '--format=csv,noheader'],
        text=True).splitlines()
    return any(int(parts[0]) == index and parts[1] not in applications
               for line in cards if (parts := [part.strip() for part in line.split(',')]))


async def run(state, scene, stage, gpu):
    command = [str(PYTHON), '-B', '-u',
               str(ROOT / 'scripts/train_splatad_human_train_only_scene.py'),
               '--scope', 'reconstruction_only', '--scene', scene, '--stage', stage]
    log = ROOT / 'logs' / f'splatad_reconstruction_only_{stage}_{scene}.log'
    row = state['scenes'][scene]
    row.update(status='running', stage=stage, gpu=gpu, command=command,
               log=str(log), started_at=time.time())
    save(state)
    with log.open('w') as stream:
        process = await asyncio.create_subprocess_exec(*command, cwd=ROOT,
                                                        env=environment(gpu),
                                                        stdout=stream, stderr=stream)
        row['pid'] = process.pid
        save(state)
        code = await process.wait()
    report = ROOT / 'reports' / f'splatad_reconstruction_only_{stage}_{scene}_validation.json'
    passed = code == 0 and report.is_file() and json.loads(report.read_text()).get('status') == 'passed'
    row.update(status=stage + '_validated' if passed else 'failed_' + stage,
               exit_code=code, finished_at=time.time(), validation=str(report))
    save(state)
    return passed


async def main():
    preflight = json.loads((ROOT / 'reports/nuscenes_reconstruction_only_preflight.json').read_text())
    assert preflight['status'] == 'reconstruction_only_frozen'
    assert {row['scene'] for row in preflight['scenes']} == set(SCENES)
    for row in preflight['scenes']:
        assert row['status'] == 'ready_reconstruction_only'
        assert row['independent_real_heldout_blocks'] == 0
        isolated = json.loads((ROOT / 'reports/splatad_train_only/reconstruction_only'
                               / f'{row["scene"]}.json').read_text())
        assert isolated['status'] == 'verified_reconstruction_only'
        assert isolated['source_split_sha256'] == row['manifest_sha256']
    if STATE.exists():
        state = json.loads(STATE.read_text())
        assert state['status'] == 'waiting_for_gpu_owners', state['status']
        assert set(state['scenes']) == set(SCENES)
        assert all(row['status'] == 'queued' for row in state['scenes'].values())
        state['gpu_pool'] = GPUS
        state['reserved_gpus'] = [0, 1, 2]
        state['rescheduled_at'] = time.time()
    else:
        state = {'method': 'SplatAD', 'variant': 'reconstruction_only_no_real_heldout',
                 'status': 'waiting_for_gpu_owners', 'started_at': time.time(),
                 'real_heldout_eligible': False, 'heldout_reason': 'no_independent_real_heldout',
                 'auto_retry': False, 'gpu_pool': GPUS,
                 'reserved_gpus': [0, 1, 2],
                 'scenes': {scene: {'status': 'queued'} for scene in SCENES}}
    save(state)
    while not all(gpu_free(index) for index in GPUS):
        state['status'] = 'waiting_for_gpu_release'
        save(state)
        await asyncio.sleep(180)
    state['status'] = 'pilot_running'
    save(state)
    if not await run(state, 'scene-0067', 'pilot', GPUS[0]):
        state['status'] = 'blocked_pilot_failure'
        save(state)
        return
    state['status'] = 'formal_running'
    save(state)

    async def worker(scene, gpu):
        if shutil.disk_usage(ROOT).free < 20 * 1024**3:
            state['scenes'][scene]['status'] = 'blocked_low_disk'
            save(state)
            return
        await run(state, scene, 'formal', gpu)

    await asyncio.gather(*(worker(scene, gpu) for scene, gpu in zip(SCENES, GPUS)))
    state.update(status='finished' if all(row['status'] == 'formal_validated'
                                           for row in state['scenes'].values())
                 else 'finished_with_failures', finished_at=time.time())
    save(state)


if __name__ == '__main__':
    asyncio.run(main())
