"""Authorized DriveStudio StreetGS variant: train-only nuScenes, async GPU queue.

No changes to the read-only repository/environment. No model/config fallback.
GPU slots wait for foreign jobs to finish. Each scene has its own native run.
"""
import asyncio
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

from run_omnire_legacy_no_smpl_queue import environment, PYTHON, REPO, ROOT

BASE = ROOT / 'configs/streetgs_drivestudio_nusc_v1'
CONFIG = BASE / 'streetgs.yaml'
MANIFEST = BASE / 'manifest.json'
STATE = ROOT / 'state/streetgs_drivestudio_nusc_v1.json'
RUNS = ROOT / 'runs/streetgs_drivestudio_nusc_v1'
STATUS = {}
WORKERS = {}
META = {}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save():
    value = dict(META, scenes=STATUS, workers=WORKERS, updated_at=time.time())
    tmp = STATE.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(STATE)


def update(scene, **values):
    STATUS[scene].update(values)
    save()


async def idle(gpu):
    p = await asyncio.create_subprocess_exec(
        'nvidia-smi', '-i', str(gpu), '--query-gpu=memory.used',
        '--format=csv,noheader,nounits', stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE)
    out, _ = await p.communicate()
    return p.returncode == 0 and int(out.decode().strip()) < 1000


async def wait_slot(gpu, pending=None):
    while True:
        if pending is not None and pending.empty():
            return False
        if shutil.disk_usage(ROOT).free < 15 * 1024**3:
            WORKERS[str(gpu)] = {'stage': 'waiting_disk_space'}
        elif await idle(gpu):
            return True
        else:
            WORKERS[str(gpu)] = {'stage': 'waiting_foreign_gpu_job'}
        save()
        await asyncio.sleep(60)


async def command(scene, stage, gpu, args):
    log = ROOT / f'logs/streetgs_drivestudio_{stage}_{scene}.log'
    update(scene, stage=stage, gpu=gpu, command=args, log=str(log), started_at=time.time())
    env = environment(gpu)
    env['TMPDIR'] = str(ROOT / 'cache/streetgs_drivestudio_tmp')
    Path(env['TMPDIR']).mkdir(parents=True, exist_ok=True)
    with log.open('w') as output:
        p = await asyncio.create_subprocess_exec(
            *args, cwd=REPO, env=env, stdout=output, stderr=asyncio.subprocess.STDOUT)
        update(scene, pid=p.pid)
        code = await p.wait()
    update(scene, exit_code=code, finished_at=time.time())
    if code:
        raise RuntimeError(f'{stage} exit={code}; log={log}')


async def train(row, gpu, stage):
    scene = row['scene']
    assert sha(CONFIG) == META['config_sha256']
    assert sha(REPO / 'configs/streetgs.yaml') == META['config_sha256']
    assert sha(row['split']) == row['split_sha256']
    data = Path(row['data_root'])
    assert len(list((data / 'images').glob('*.jpg'))) == row['images']
    assert len(list((data / 'sky_masks').glob('*.png'))) == row['images']
    out = RUNS / stage
    run = out / 'nuScenes' / scene
    assert not run.exists(), f'Refuse overwrite: {run}'
    args = [str(PYTHON), '-B', '-u', 'tools/train.py', '--config_file', str(CONFIG),
            '--output_root', str(out), '--project', 'nuScenes', '--run_name', scene,
            'dataset=nuscenes/6cams', 'data.data_root=' + str(data.parent),
            'data.scene_idx=0', 'data.pixel_source.load_smpl=false',
            'data.pixel_source.test_image_stride=0']
    if stage == 'pilot':
        args += ['trainer.optim.num_iters=100']
    await command(scene, stage + '_train', gpu, args)
    checkpoint = run / 'checkpoint_final.pth'
    assert checkpoint.is_file() and checkpoint.stat().st_size > 0
    update(scene, checkpoint=str(checkpoint), checkpoint_bytes=checkpoint.stat().st_size)
    # A native reload/render after the pilot is the batch gate. Formal native
    # training already renders its full set; keep its checkpoint status distinct.
    if stage == 'pilot':
        await command(scene, 'pilot_reload', gpu, [str(PYTHON), '-B', '-u',
                      'tools/eval.py', '--resume_from', str(checkpoint),
                      '--render_video_postfix', 'pilot_reload'])
        videos = list((run / 'videos_eval').glob('full_set_*pilot_reload*rgbs.mp4'))
        assert videos and all(p.stat().st_size > 0 for p in videos)
        update(scene, stage='pilot_validated', pilot_checkpoint=str(checkpoint))
    else:
        update(scene, stage='training_completed', checkpoint_reload_validated=False)


async def worker(gpu, queue):
    while not queue.empty():
        if not await wait_slot(gpu, queue):
            break
        try:
            row = queue.get_nowait()
        except asyncio.QueueEmpty:
            break
        scene = row['scene']
        WORKERS[str(gpu)] = {'stage': 'training', 'scene': scene}
        save()
        try:
            await train(row, gpu, 'formal')
        except Exception as exc:
            update(scene, stage='failed', reason=repr(exc), finished_at=time.time())
        finally:
            queue.task_done()
    WORKERS[str(gpu)] = {'stage': 'finished'}
    save()


async def main():
    manifest = json.loads(MANIFEST.read_text())
    META.update({k: v for k, v in manifest.items() if k != 'scenes'})
    META['manifest_sha256'] = sha(MANIFEST)
    if STATE.exists():
        raise RuntimeError('State already exists; inspect before explicit restart.')
    for row in manifest['scenes']:
        STATUS[row['scene']] = {'stage': 'queued', 'real_heldout_eligible': row['real_heldout_eligible']}
    save()
    pilot = next(x for x in manifest['scenes'] if x['scene'] == 'scene-0065')
    await wait_slot(2)
    try:
        await train(pilot, 2, 'pilot')
    except Exception as exc:
        update(pilot['scene'], stage='pilot_failed', reason=repr(exc))
        META['queue_stage'] = 'blocked_at_pilot_no_fallback'
        save()
        return
    META['queue_stage'] = 'formal_batch_running'
    q = asyncio.Queue()
    for row in manifest['scenes']:
        q.put_nowait(row)
    await asyncio.gather(*(worker(gpu, q) for gpu in range(8)))
    META['queue_stage'] = 'finished'
    save()


if __name__ == '__main__':
    with (ROOT / 'state/streetgs_drivestudio_nusc_v1.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        asyncio.run(main())
