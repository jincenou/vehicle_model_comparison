"""Queue selected human nuScenes scenes under the retained legacy OmniRe config.

The source config is byte-identical to the earlier scene-002 experiment. It
uses DeformableNodes for people and does not load SMPL. Every native training
root contains train RGB/LiDAR only; heldout observations remain outside it.
"""

import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import threading
import time


ROOT = Path(__file__).resolve().parents[1]
REPO = Path(('/opt/rigid-benchmark/drivestudio'.replace('/opt/rigid-benchmark/drivestudio', __import__("os").environ.get('RIGID_DRIVESTUDIO', '/opt/rigid-benchmark/drivestudio')).replace('/opt/rigid-benchmark/external', __import__("os").environ.get('RIGID_EXTERNAL_ROOT', '/opt/rigid-benchmark/external'))))
CONFIG = ROOT / 'configs/omnire_reviewed288/omnire_legacy_nuscenes_no_smpl.yaml'
SOURCE_CONFIG = REPO / 'logs/omnire_nuscenes/recon/scene002_full2/config.yaml'
PYTHON = Path(('/opt/conda/envs/ab_cmp_omnire/bin/python'.replace('/opt/conda', __import__("os").environ.get('RIGID_CONDA_ROOT', '/opt/conda')).replace('/opt/rigid-benchmark/project', __import__("os").environ.get('RIGID_PROJECT_ROOT', '/opt/rigid-benchmark/project'))))
SKY_PYTHON = Path(('/opt/conda/envs/garden_hunyuan3d/bin/python'.replace('/opt/conda', __import__("os").environ.get('RIGID_CONDA_ROOT', '/opt/conda')).replace('/opt/rigid-benchmark/project', __import__("os").environ.get('RIGID_PROJECT_ROOT', '/opt/rigid-benchmark/project'))))
SKY_SCRIPT = REPO / 'datasets/tools/extract_sky_masks_hf.py'
SKY_MODEL = ROOT / 'weights/hf/hub/models--nvidia--segformer-b5-finetuned-cityscapes-1024-1024/snapshots/2c6f153e4c23c229e2fa2b188eb250607e030cd8'
STATE = ROOT / 'state/omnire_human_legacy_no_smpl_queue.json'
LOCK = threading.Lock()
STATUS = {}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save():
    with LOCK:
        value = {'method': 'OmniRe', 'variant': 'retained_scene002_no_smpl',
                 'config': str(CONFIG), 'config_sha256': digest(CONFIG),
                 'policy': 'Train-only RGB/LiDAR; same retained non-SMPL OmniRe config; no automatic model or config fallback.',
                 'scenes': STATUS, 'updated_at': time.time()}
        temp = STATE.with_suffix('.tmp.json')
        temp.write_text(json.dumps(value, indent=2) + '\n')
        temp.replace(STATE)


def update(scene, **fields):
    with LOCK:
        STATUS[scene].update(fields)
    save()


def environment(gpu):
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=str(gpu), PYTHONPATH=str(REPO),
               CUDA_HOME=('/opt/conda/envs/evol'.replace('/opt/conda', __import__("os").environ.get('RIGID_CONDA_ROOT', '/opt/conda')).replace('/opt/rigid-benchmark/project', __import__("os").environ.get('RIGID_PROJECT_ROOT', '/opt/rigid-benchmark/project'))),
               TORCH_EXTENSIONS_DIR=str(ROOT / 'cache/torch_ext/omnire'),
               HF_HOME=str(ROOT / 'cache/hf'),
               HF_HUB_CACHE=str(ROOT / 'weights/hf/hub'),
               TORCH_HOME=str(ROOT / 'cache/torch'),
               XDG_CACHE_HOME=str(ROOT / 'cache/xdg'),
               MPLCONFIGDIR=str(ROOT / 'cache/matplotlib'),
               PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='4',
               TORCH_CUDA_ARCH_LIST='8.9', MAX_JOBS='4')
    env['PATH'] = str(PYTHON.parent) + os.pathsep + env.get('PATH', '')
    return env


def command(scene, stage, gpu, args, cwd=REPO):
    log = ROOT / f'logs/omnire_human_legacy_{stage}_{scene}.log'
    update(scene, stage=stage, gpu=gpu, command=args, log=str(log), started_at=time.time())
    with log.open('w') as output:
        process = subprocess.Popen(args, cwd=cwd, env=environment(gpu),
                                   stdout=output, stderr=subprocess.STDOUT)
        update(scene, pid=process.pid)
        code = process.wait()
    update(scene, exit_code=code, finished_at=time.time())
    if code:
        raise RuntimeError(f'{scene} {stage} failed: {log}')
    return log


def prepare(scene, gpu):
    base = ROOT / 'inputs/omnire_human_reviewed95' / scene
    processed = base / 'processed/trainval/000'
    audit_path = ROOT / f'reports/omnire_human_reviewed95_{scene}_input_audit.json'
    if not audit_path.is_file() or json.loads(audit_path.read_text()).get('status') != 'native_preprocess_completed':
        command(scene, 'prepare', gpu,
                [str(PYTHON), '-B', '-u', str(ROOT / 'scripts/prepare_omnire_human_scene.py'),
                 '--scene', scene, '--process'], cwd=ROOT)
    audit = json.loads(audit_path.read_text())
    split = ROOT / f'configs/nuscenes_human_splits_reviewed95/{scene}.json'
    assert audit['status'] == 'native_preprocess_completed'
    assert audit['source_split_sha256'] == digest(split)
    assert audit['heldout_rgb_lidar_accessible'] is False
    assert len(list((processed / 'images').glob('*.jpg'))) == audit['processed_images']
    sky_count = len(list((processed / 'sky_masks').glob('*.png')))
    if sky_count != audit['processed_images']:
        assert sky_count < audit['processed_images']
        command(scene, 'sky_masks', gpu,
                [str(SKY_PYTHON), '-B', '-u', str(SKY_SCRIPT),
                 '--scene_dir', str(processed), '--model', str(SKY_MODEL),
                 '--device', 'cuda:0'])
    assert len(list((processed / 'sky_masks').glob('*.png'))) == audit['processed_images']
    update(scene, data_ready=True, train_images=audit['processed_images'],
           split_sha256=audit['source_split_sha256'])
    return base / 'processed/trainval'


def train(scene, gpu, stage):
    data = prepare(scene, gpu)
    run_name = scene + '_legacy_no_smpl'
    root = ROOT / f'runs/omnire_human_reviewed95/{stage}'
    run = root / 'nuScenes' / run_name
    assert not run.exists(), f'Refuse to overwrite existing run: {run}'
    args = [str(PYTHON), '-B', '-u', 'tools/train.py', '--config_file', str(CONFIG),
            '--output_root', str(root), '--project', 'nuScenes',
            '--run_name', run_name, 'data.data_root=' + str(data),
            'data.scene_idx=0', 'data.pixel_source.test_image_stride=0']
    if stage == 'pilot':
        args.append('trainer.optim.num_iters=100')
    command(scene, stage + '_train', gpu, args)
    checkpoint = run / 'checkpoint_final.pth'
    assert checkpoint.is_file() and checkpoint.stat().st_size > 0
    update(scene, checkpoint=str(checkpoint), checkpoint_bytes=checkpoint.stat().st_size)
    command(scene, stage + '_reload', gpu,
            [str(PYTHON), '-B', '-u', 'tools/eval.py', '--resume_from', str(checkpoint),
             '--render_video_postfix', stage + '_reload'])
    outputs = list((run / 'videos_eval').glob(f'full_set_*_{stage}_reload_RigidNodes_rgbs.mp4'))
    assert outputs and all(path.stat().st_size > 0 for path in outputs)
    update(scene, stage=stage + '_validated', native_rigid_video=str(outputs[-1]))


def formal_worker(gpu, scenes):
    for scene in scenes:
        try:
            if shutil.disk_usage(ROOT).free < 20 * 1024**3:
                raise RuntimeError('Less than 20 GiB free before scene launch')
            train(scene, gpu, 'formal')
        except Exception as exc:
            update(scene, stage='failed', reason=repr(exc), finished_at=time.time())


def main():
    assert digest(CONFIG) == digest(SOURCE_CONFIG)
    assert 'load_smpl: false' in CONFIG.read_text()
    assert 'SMPLNodes:' not in CONFIG.read_text()
    assert SKY_MODEL.is_dir()
    preflight_path = ROOT / 'reports/nuscenes_human_reviewed95_preflight.json'
    while True:
        if preflight_path.is_file():
            preflight = json.loads(preflight_path.read_text())
            if preflight.get('status') in ('completed', 'completed_with_blocked_scenes'):
                break
        time.sleep(60)
    assert len(preflight['scenes']) == 62
    shared_scene = 'scene-0067'
    scenes = [row['scene'] for row in preflight['scenes']
              if row['status'] == 'ready' and row['scene'] != shared_scene]
    for row in preflight['scenes']:
        if row['scene'] == shared_scene:
            STATUS[shared_scene] = {'stage': 'blocked_shared_source_split',
                                    'reason': 'Current reviewed 288 vehicle and human source observations share this scene.'}
        elif row['status'] != 'ready':
            STATUS[row['scene']] = {'stage': 'blocked_preflight', 'reason': row['status']}
    for scene in scenes:
        split = ROOT / f'configs/nuscenes_human_splits_reviewed95/{scene}.json'
        assert split.is_file()
        STATUS[scene] = {'stage': 'queued', 'split': str(split),
                         'split_sha256': digest(split)}
    save()
    # The vehicle queue owns GPUs 0, 5 and 6 first. It may have per-scene
    # failures; those do not prevent the independent human scene queue.
    vehicle_state = ROOT / 'state/omnire_legacy_no_smpl_queue.json'
    while True:
        previous = json.loads(vehicle_state.read_text())
        stages = [v['stage'] for v in previous['scenes'].values()]
        if 'pilot_failed' in stages:
            for scene in scenes:
                update(scene, stage='blocked_previous_pilot_failed')
            return
        if all(s in ('formal_validated', 'failed') for s in stages):
            break
        time.sleep(180)
    assert 'scene-0007' in scenes, 'Human pilot scene missing from reviewed split'
    try:
        train('scene-0007', 6, 'pilot')
    except Exception as exc:
        update('scene-0007', stage='pilot_failed', reason=repr(exc))
        return
    assignments = {gpu: scenes[i::3] for i, gpu in enumerate((0, 5, 6))}
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(formal_worker, gpu, scenes)
                   for gpu, scenes in assignments.items()]
        for future in futures:
            future.result()
    save()


if __name__ == '__main__':
    main()
