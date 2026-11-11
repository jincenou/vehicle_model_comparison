"""Low-priority native-alpha pilot: wait for a genuinely unreserved GPU.

One analytic fixture gates four real Gaussian pilots. No formal metrics, model
generation, retries, preemption, asset copies, or changes to existing queues.
The child inherits the shared GPU lock so a controller exit cannot free a busy
device. Existing reservations include CPU phases and rigid-generation priority.
"""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

R = Path(__file__).resolve().parents[1]
P = R / 'model_preparation/20261109_2100_CST'
sys.path.insert(0, str(P / 'scripts'))
from dispatch_rigid111_direct import available

ADAPTER = R / 'scripts/render_calibration_alpha_pilot.py'
STATE = R / 'state/calibration_alpha_gpu_pilot_v1.json'
PREFIX = Path(os.environ.get('RIGID_CONDA_ROOT', '/opt/conda')) / 'envs/ab_eval'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(state):
    state['updated_at'] = time.time()
    temporary = STATE.with_suffix('.tmp')
    temporary.write_text(json.dumps(state, indent=2, allow_nan=False)+'\n')
    temporary.replace(STATE)


def main():
    with STATE.with_suffix('.lock').open('a') as owner:
        fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if STATE.exists():
            raise RuntimeError('Existing attempt must be audited; no implicit restart')
        sources = [Path(__file__), ADAPTER,
                   R / 'evaluation_nuscenes_v1/scripts/alpha_extent_measurement.py',
                   R / 'evaluation_nuscenes_v1/scripts/spatial_calibration.py',
                   P / 'scripts/dispatch_rigid111_direct.py']
        signatures = {str(p): sha(p) for p in sources}
        jobs = [dict(method='fixture_gaussian', sample_id='calibration_fixture_v3', status='queued')]
        jobs += [dict(method=m, sample_id='scene-0064_t000637', status='queued')
                 for m in ['ours', 'ah', 'trellis_original', 'recon02']]
        state = dict(pid=os.getpid(), status='waiting_unreserved_gpu', jobs=jobs,
                     formal_metrics_enabled=False, native_assets_modified=False,
                     disk_reserve_gib=10, code_sha256=signatures, started_at=time.time())
        save(state)
        for job in jobs:
            lease = None
            gpu = None
            while lease is None:
                if any(sha(p) != h for p, h in signatures.items()):
                    state.update(status='stopped_code_changed'); save(state); return
                if shutil.disk_usage(R).free < 10 * 1024**3:
                    state.update(status='waiting_disk_space'); save(state)
                    time.sleep(60); continue
                for index in range(8):
                    handle = (P / 'runs/rigid111_gpu_locks' / f'gpu_{index}.lock').open('a')
                    try:
                        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        ready, detail = available(str(index))
                        if ready:
                            lease, gpu = handle, str(index)
                            break
                    except BlockingIOError:
                        pass
                    except Exception as error:
                        state['last_probe_error'] = repr(error)
                    handle.close()
                if lease is None:
                    state.update(status='waiting_unreserved_gpu'); save(state)
                    time.sleep(60)
            try:
                job.update(status='running', gpu=gpu, started_at=time.time())
                state.update(status='running_native_alpha_pilot'); save(state)
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, OMP_NUM_THREADS='2',
                           OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='2', MAX_JOBS='2',
                           LD_LIBRARY_PATH=str(PREFIX / 'lib'),
                           TORCH_EXTENSIONS_DIR=str(R / 'cache/torch_ext/ab_eval'))
                log_path = R / 'logs' / f'calibration_alpha_gpu_{job["method"]}_v1.log'
                command = [str(PREFIX / 'bin/python'), '-B', '-u', str(ADAPTER),
                           '--sample-id', job['sample_id'], '--method', job['method'],
                           '--resolutions', '512', '1024']
                with log_path.open('x') as log:
                    child = subprocess.Popen(command, env=env, stdout=log, stderr=log,
                                             pass_fds=(owner.fileno(), lease.fileno()))
                    job.update(child_pid=child.pid, command=command, log=str(log_path)); save(state)
                    code = child.wait()
                result_path = R / 'exports/calibration_alpha_pilot_v1' / job['sample_id'] / job['method'] / 'state.json'
                result = json.loads(result_path.read_text())
                if code or result['status'] != 'measured_pending_cross_backend_review':
                    raise RuntimeError(f'Native measurement failed: exit={code}, status={result["status"]}')
                if job['method'] == 'fixture_gaussian' and not result.get('fixture_check', {}).get('passed'):
                    raise RuntimeError('Analytic Gaussian extent did not pass')
                job.update(status='measured_pending_review', result=str(result_path), result_sha256=sha(result_path))
            except Exception as error:
                job.update(status='failed', reason=repr(error))
                if job['method'] == 'fixture_gaussian':
                    for pending in jobs[1:]:
                        pending.update(status='blocked_fixture_failure')
                    state['status'] = 'stopped_fixture_failure'
                    save(state); return
            finally:
                job.pop('child_pid', None)
                job['finished_at'] = time.time()
                lease.close(); save(state)
        state.update(status='pilot_finished_pending_review', finished_at=time.time()); save(state)


if __name__ == '__main__':
    main()
