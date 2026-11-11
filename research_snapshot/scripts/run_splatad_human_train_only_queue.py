"""Run the frozen nuScenes human scenes through native SplatAD after one pilot."""

import asyncio
import json
import os
import shutil
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path(('/opt/conda/envs/splatad/bin/python'.replace('/opt/conda', __import__("os").environ.get('RIGID_CONDA_ROOT', '/opt/conda')).replace('/opt/rigid-benchmark/project', __import__("os").environ.get('RIGID_PROJECT_ROOT', '/opt/rigid-benchmark/project'))))
STATE = ROOT / 'state/splatad_human_train_only_queue.json'
PREFLIGHT = ROOT / 'reports/nuscenes_human_reviewed95_preflight.json'
GPUS = (3, 4, 7)
EXCLUDED = {'scene-0067': 'Current reviewed 288 vehicle and human source observations share this scene; require a joint split.'}


def save(state):
    state['updated_at'] = time.time()
    tmp = STATE.with_suffix('.tmp.json')
    tmp.write_text(json.dumps(state, indent=2) + '\n')
    tmp.replace(STATE)


def environment(gpu):
    env = dict(os.environ)
    env.update(CUDA_VISIBLE_DEVICES=str(gpu), PYTHONDONTWRITEBYTECODE='1',
               CUDA_HOME=('/opt/conda/envs/splatad'.replace('/opt/conda', __import__("os").environ.get('RIGID_CONDA_ROOT', '/opt/conda')).replace('/opt/rigid-benchmark/project', __import__("os").environ.get('RIGID_PROJECT_ROOT', '/opt/rigid-benchmark/project'))),
               TORCH_CUDA_ARCH_LIST='8.9', MAX_JOBS='4',
               OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4',
               HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
               WANDB_MODE='disabled', PYTHONUNBUFFERED='1')
    for key, sub in {'HF_HOME': 'hf', 'TORCH_HOME': 'torch',
                     'TORCH_EXTENSIONS_DIR': 'torch_ext/splatad',
                     'XDG_CACHE_HOME': 'xdg', 'PIP_CACHE_DIR': 'pip',
                     'TMPDIR': 'tmp', 'MPLCONFIGDIR': 'mpl',
                     'NUMBA_CACHE_DIR': 'numba', 'CUDA_CACHE_PATH': 'cuda'}.items():
        path = ROOT / 'cache' / sub
        path.mkdir(parents=True, exist_ok=True)
        env[key] = str(path)
    env['PATH'] = str(PYTHON.parent) + ':' + env.get('PATH', '')
    env['LD_LIBRARY_PATH'] = ('/opt/conda/envs/splatad/lib:/opt/conda/envs/splatad/lib64:'.replace('/opt/conda', __import__("os").environ.get('RIGID_CONDA_ROOT', '/opt/conda')).replace('/opt/rigid-benchmark/project', __import__("os").environ.get('RIGID_PROJECT_ROOT', '/opt/rigid-benchmark/project'))) + env.get('LD_LIBRARY_PATH', '')
    env['PYTHONPATH'] = str(ROOT / 'adapters') + ':' + str(ROOT / 'scripts')
    return env


async def invoke(state, scene, stage, gpu, command):
    log = ROOT / 'logs' / f'splatad_human_train_only_{stage}_{scene}.log'
    row = state['scenes'][scene]
    row.update(stage=stage, status='running', gpu=gpu, command=command,
               log=str(log), started_at=time.time())
    save(state)
    with log.open('a') as stream:
        process = await asyncio.create_subprocess_exec(*command, cwd=ROOT,
                                                        env=environment(gpu),
                                                        stdout=stream, stderr=stream)
        row['pid'] = process.pid
        save(state)
        code = await process.wait()
    row.update(exit_code=code, finished_at=time.time())
    save(state)
    return code


async def prepare(state, scene, gpu):
    report = ROOT / 'reports/splatad_train_only/human_reviewed95' / f'{scene}.json'
    if not report.exists():
        command = [str(PYTHON), '-B', str(ROOT / 'scripts/materialize_splatad_train_only_scene.py'),
                   '--scope', 'human', '--scene', scene]
        code = await invoke(state, scene, 'materialize', gpu, command)
        if code:
            state['scenes'][scene].update(status='failed_materialize')
            save(state)
            return False
    result = json.loads(report.read_text())
    source = json.loads((ROOT / 'configs/nuscenes_human_splits_reviewed95' / f'{scene}.json').read_text())
    assert result['status'] == 'verified_train_only'
    assert result['scene'] == scene and result['heldout_rgb_lidar_present'] is False
    assert result['heldout_blocks'] == len(source['heldout_sample_tokens'])
    return True


async def train(state, scene, stage, gpu, resume=False, attempt=None):
    command = [str(PYTHON), '-B', '-u', str(ROOT / 'scripts/train_splatad_human_train_only_scene.py'),
               '--scene', scene, '--stage', stage]
    if resume:
        command.append('--resume')
    if attempt:
        command.extend(['--attempt', attempt])
    code = await invoke(state, scene, stage, gpu, command)
    validation = ROOT / 'reports' / f'splatad_human_reviewed95_{stage}_{scene}_validation.json'
    passed = code == 0 and validation.exists() and json.loads(validation.read_text()).get('status') == 'passed'
    state['scenes'][scene].update(status=f'{stage}_validated' if passed else f'failed_{stage}')
    save(state)
    return passed


async def main():
    preflight = json.loads(PREFLIGHT.read_text())
    ready = sorted(x['scene'] for x in preflight['scenes'] if x['status'] == 'ready')
    assert len(ready) == 62 and 'scene-0007' in ready and 'scene-0067' in ready
    if STATE.exists():
        state = json.loads(STATE.read_text())
        assert state['status'] == 'formal_running', state['status']
        assert set(state['scenes']) == set(ready)
        state['pid'] = os.getpid()
        state['gpu_pool'] = GPUS
        state['migration_resumed_at'] = time.time()
        state['reserved_gpus'] = [0, 1, 2]
        for scene in ready:
            row = state['scenes'][scene]
            if row['status'] in ('formal_validated', 'blocked_shared_source_split', 'queued'):
                continue
            validation = ROOT / 'reports' / f'splatad_human_reviewed95_formal_{scene}_validation.json'
            if validation.exists() and json.loads(validation.read_text()).get('status') == 'passed':
                row['status'] = 'formal_validated'
                continue
            base = ROOT / 'runs/splatad_human_reviewed95/formal' / scene / 'splatad/human_reviewed95_train_only_v1/nerfstudio_models'
            checkpoints = sorted(base.glob('step-*.ckpt'))
            if checkpoints:
                row.update(status='queued_resume', resume_checkpoint=str(checkpoints[-1]))
            else:
                attempt = 'human_reviewed95_train_only_v1_gpu34567_restart1'
                new_base = ROOT / 'runs/splatad_human_reviewed95/formal' / scene / 'splatad' / attempt
                assert not new_base.exists(), f'Refuse overwriting restart attempt: {new_base}'
                row.update(status='queued_restart', restart_attempt=attempt,
                           restart_reason='Interrupted during GPU migration before first checkpoint')
        save(state)
    else:
        state = {'method': 'SplatAD', 'variant': 'nuScenes train-only human reviewed95',
                 'status': 'pilot_preparing', 'started_at': time.time(), 'pid': os.getpid(),
                 'gpu_pool': GPUS, 'auto_retry': False,
                 'policy': 'Real heldout RGB/LiDAR physically absent. Internal eval uses a train shadow. No automatic config fallback.',
                 'scenes': {scene: {'status': 'queued'} for scene in ready}}
        for scene, reason in EXCLUDED.items():
            state['scenes'][scene] = {'status': 'blocked_shared_source_split', 'reason': reason}
        save(state)
        pilot = 'scene-0007'
        if not await prepare(state, pilot, GPUS[0]):
            state['status'] = 'blocked_pilot_preparation'
            save(state)
            return
        state['status'] = 'pilot_training'
        save(state)
        if not await train(state, pilot, 'pilot', GPUS[0]):
            state['status'] = 'blocked_pilot_failure'
            save(state)
            return
    pending = asyncio.Queue()
    for scene in sorted(ready, key=lambda s: 0 if state['scenes'][s]['status'] in ('queued_resume', 'queued_restart') else 1):
        if state['scenes'][scene]['status'] in ('queued', 'queued_resume', 'queued_restart'):
            pending.put_nowait(scene)
    state['status'] = 'formal_running'
    save(state)

    async def worker(gpu):
        while not pending.empty():
            scene = pending.get_nowait()
            if shutil.disk_usage(ROOT).free < 20 * 1024**3:
                state['scenes'][scene].update(status='blocked_low_disk')
                save(state)
                pending.task_done()
                continue
            try:
                resume = state['scenes'][scene]['status'] == 'queued_resume'
                attempt = state['scenes'][scene].get('restart_attempt')
                if await prepare(state, scene, gpu):
                    await train(state, scene, 'formal', gpu, resume=resume, attempt=attempt)
            except Exception as exc:
                state['scenes'][scene].update(status='failed_exception', reason=repr(exc))
                save(state)
            pending.task_done()

    await asyncio.gather(*(worker(gpu) for gpu in GPUS))
    state['status'] = 'finished' if all(row['status'] in ('formal_validated', 'blocked_shared_source_split')
                                         for row in state['scenes'].values()) else 'finished_with_failures'
    state['finished_at'] = time.time()
    save(state)


if __name__ == '__main__':
    asyncio.run(main())
