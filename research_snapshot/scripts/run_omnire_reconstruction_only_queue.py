"""Train two nuScenes source-covered scenes as extraction-only OmniRe runs."""

import json
import shutil
import subprocess
import time
from pathlib import Path

from run_omnire_human_legacy_no_smpl_queue import (
    ROOT, REPO, CONFIG, SOURCE_CONFIG, PYTHON, SKY_PYTHON,
    SKY_SCRIPT, SKY_MODEL, digest, environment,
)


STATE = ROOT / 'state/omnire_reconstruction_only_queue.json'
PRIOR = ROOT / 'state/splatad_strict_actor_queue.json'
SOCKET = ROOT / 'state/vehicle_cmp_tmux.sock'
SCENES = ('scene-0066', 'scene-0067')
GPU = 6


def save(state):
    state['updated_at'] = time.time()
    temporary = STATE.with_suffix('.tmp.json')
    temporary.write_text(json.dumps(state, indent=2) + '\n')
    temporary.replace(STATE)


def gpu_free():
    apps = set(subprocess.check_output(
        ['nvidia-smi', '--query-compute-apps=gpu_uuid', '--format=csv,noheader'],
        text=True).splitlines())
    cards = subprocess.check_output(
        ['nvidia-smi', '--query-gpu=index,uuid', '--format=csv,noheader'],
        text=True).splitlines()
    return any(int(parts[0]) == GPU and parts[1] not in apps
               for line in cards if (parts := [part.strip() for part in line.split(',')]))


def invoke(state, scene, stage, args, cwd=REPO):
    log = ROOT / 'logs' / f'omnire_reconstruction_only_{stage}_{scene}.log'
    row = state['scenes'][scene]
    row.update(status=stage, command=args, log=str(log), gpu=GPU,
               started_stage_at=time.time())
    save(state)
    with log.open('w') as stream:
        process = subprocess.Popen(args, cwd=cwd, env=environment(GPU),
                                   stdout=stream, stderr=subprocess.STDOUT)
        row['pid'] = process.pid
        save(state)
        code = process.wait()
    row.update(exit_code=code, finished_stage_at=time.time())
    save(state)
    if code:
        raise RuntimeError(f'{scene} {stage} exited {code}: {log}')


def prepare(state, scene):
    base = ROOT / 'inputs/omnire_reconstruction_only' / scene
    processed = base / 'processed/trainval/000'
    split = ROOT / 'configs/nuscenes_reconstruction_only' / f'{scene}.json'
    report = ROOT / 'reports' / f'omnire_reconstruction_only_{scene}_input_audit.json'
    invoke(state, scene, 'prepare', [str(PYTHON), '-B', '-u',
           str(ROOT / 'scripts/prepare_omnire_human_scene.py'),
           '--scope', 'reconstruction_only', '--scene', scene, '--process'], cwd=ROOT)
    audit = json.loads(report.read_text())
    assert audit['status'] == 'native_preprocess_completed_reconstruction_only'
    assert audit['source_split_sha256'] == digest(split)
    assert audit['real_heldout_eligible'] is False
    assert len(list((processed / 'images').glob('*.jpg'))) == audit['processed_images']
    invoke(state, scene, 'sky_masks', [str(SKY_PYTHON), '-B', '-u', str(SKY_SCRIPT),
           '--scene_dir', str(processed), '--model', str(SKY_MODEL), '--device', 'cuda:0'])
    assert len(list((processed / 'sky_masks').glob('*.png'))) == audit['processed_images']
    return base / 'processed/trainval'


def train(state, scene, stage, data):
    root = ROOT / 'runs/omnire_reconstruction_only' / stage
    name = scene + '_legacy_no_smpl_reconstruction_only'
    run = root / 'nuScenes' / name
    assert not run.exists(), f'Refuse overwriting {run}'
    args = [str(PYTHON), '-B', '-u', 'tools/train.py',
            '--config_file', str(CONFIG), '--output_root', str(root),
            '--project', 'nuScenes', '--run_name', name,
            'data.data_root=' + str(data), 'data.scene_idx=0',
            'data.pixel_source.test_image_stride=0']
    if stage == 'pilot':
        args.append('trainer.optim.num_iters=100')
    invoke(state, scene, stage + '_train', args)
    checkpoint = run / 'checkpoint_final.pth'
    assert checkpoint.is_file() and checkpoint.stat().st_size > 0
    invoke(state, scene, stage + '_reload', [str(PYTHON), '-B', '-u',
           'tools/eval.py', '--resume_from', str(checkpoint),
           '--render_video_postfix', stage + '_reload'])
    videos = list((run / 'videos_eval').glob(f'full_set_*_{stage}_reload_rgbs.mp4'))
    assert videos and all(path.stat().st_size > 0 for path in videos)
    state['scenes'][scene].update(status=stage + '_validated',
                                  checkpoint=str(checkpoint),
                                  native_video=str(videos[-1]),
                                  real_heldout_eligible=False,
                                  heldout_reason='no_independent_real_heldout')
    save(state)


def main():
    assert digest(CONFIG) == digest(SOURCE_CONFIG)
    assert 'load_smpl: false' in CONFIG.read_text()
    assert 'SMPLNodes:' not in CONFIG.read_text()
    assert SKY_MODEL.is_dir()
    preflight = json.loads((ROOT / 'reports/nuscenes_reconstruction_only_preflight.json').read_text())
    assert preflight['status'] == 'reconstruction_only_frozen'
    assert {row['scene'] for row in preflight['scenes']} == set(SCENES)
    assert all(row['independent_real_heldout_blocks'] == 0 for row in preflight['scenes'])
    if STATE.exists():
        state = json.loads(STATE.read_text())
        assert state['status'] == 'waiting_for_gpu_owners', state['status']
        assert set(state['scenes']) == set(SCENES)
        assert all(row['status'] == 'queued' for row in state['scenes'].values())
        state['reserved_gpus'] = [0, 1, 2]
        state['rescheduled_at'] = time.time()
    else:
        state = {'method': 'OmniRe', 'variant': 'retained_scene002_no_smpl_reconstruction_only',
                 'status': 'waiting_for_gpu_owners', 'started_at': time.time(),
                 'config': str(CONFIG), 'config_sha256': digest(CONFIG),
                 'gpu': GPU, 'auto_retry': False,
                 'real_heldout_eligible': False,
                 'heldout_reason': 'no_independent_real_heldout',
                 'reserved_gpus': [0, 1, 2],
                 'scenes': {scene: {'status': 'queued'} for scene in SCENES}}
    save(state)
    while not gpu_free():
        state['status'] = 'waiting_for_gpu6'
        save(state)
        time.sleep(180)
    try:
        state['status'] = 'pilot_running'
        save(state)
        pilot_data = prepare(state, 'scene-0067')
        train(state, 'scene-0067', 'pilot', pilot_data)
        state['status'] = 'formal_running'
        save(state)
        for scene in SCENES:
            if shutil.disk_usage(ROOT).free < 20 * 1024**3:
                state['scenes'][scene]['status'] = 'blocked_low_disk'
                save(state)
                continue
            try:
                data = pilot_data if scene == 'scene-0067' else prepare(state, scene)
                train(state, scene, 'formal', data)
            except Exception as exc:
                state['scenes'][scene].update(status='failed_formal', reason=repr(exc))
                save(state)
        state.update(status='finished' if all(row['status'] == 'formal_validated'
                                               for row in state['scenes'].values())
                     else 'finished_with_failures', finished_at=time.time())
    except Exception as exc:
        state.update(status='blocked_pilot_failure', reason=repr(exc),
                     finished_at=time.time())
    save(state)


if __name__ == '__main__':
    main()
