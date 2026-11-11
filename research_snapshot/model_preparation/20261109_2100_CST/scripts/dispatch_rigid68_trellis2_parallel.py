"""Audited parent-only handoff; preserve native child, add idle GPU workers.

Uses the existing frozen68 state/output paths and immutable native adapters.
It never retries failures, regenerates existing folders, or signals a GPU child.
"""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import copy
import fcntl
import json
import os
from pathlib import Path
import queue
import signal
import subprocess
import threading
import time

import dispatch_rigid111_direct as old
import dispatch_rigid68_continuation as prior

ROOT = old.P/'runs/trellis2_rigid68_v1'
OUTPUT = old.P/'runs/trellis2_rigid111_v1'
STATE = ROOT/'batch_state.json'
GPUS = ['2', '3', '4', '5']

def identity(pid):
    p = Path('/proc')/str(pid)
    try:
        stat = (p/'stat').read_text().split(') ', 1)[1].split()
        cmd = [v.decode() for v in (p/'cmdline').read_bytes().split(b'\0') if v]
        return dict(pid=pid, state=stat[0], session_id=int(stat[3]),
                    start_ticks=int(stat[19]), command=cmd)
    except (FileNotFoundError, ProcessLookupError):
        return None

def assert_identity(actual, expected):
    assert actual and actual['start_ticks'] == expected['start_ticks'], 'PID identity changed'
    if actual['state'] != 'Z':
        assert actual['command'] == expected['command'], 'Process command changed'

def job_mode(job, exists):
    if job['status'] in ('completed', 'failed'):
        return 'retain'
    if job['status'] == 'queued':
        assert not exists, 'Never regenerate an existing directory'
        return 'new'
    if job['status'] == 'running':
        assert exists and job['stage'] in ('generate', 'native_reload', 'glb_reload'), 'Only known native stages may be adopted'
        assert job['gpu'] in GPUS
        return 'adopt'
    raise AssertionError('Unsupported state: '+job['status'])

def inspect():
    rows, signature, _ = prior.method_preflight('trellis2')
    state = old.read(STATE)
    assert all(state[k] == v for k, v in signature.items()), 'Prior immutable signature mismatch'
    assert state['method'] == 'trellis2'
    assert {x['uid'] for x in state['jobs']} == {x['uid'] for x in rows}
    assert old.sha(state['recovery_plan']) == state['recovery_plan_sha256']
    assert old.sha(ROOT/'pilot_gate.json') == state['pilot_gate_sha256']
    parent = identity(state['pid'])
    assert parent and parent['state'] != 'Z'
    assert str(Path(prior.__file__).resolve()) in parent['command']
    for method in ['ah', 'trellis_original', 'hunyuan21', 'recon02']:
        peer = old.read(old.P/'runs'/f'{method}_rigid68_v1/batch_state.json')
        assert peer['pid'] == parent['pid']
        assert peer['status'] in ('completed', 'completed_with_failures_or_missing_views')
        assert all(j['status'] in ('completed', 'failed', 'blocked_missing_eligible_real_view') for j in peer['jobs'])
    modes, children = Counter(), {}
    for job in state['jobs']:
        mode = job_mode(job, (OUTPUT/job['sample_id']).exists())
        modes[mode] += 1
        if job['status'] == 'completed':
            old.verify_completed(job)
        if mode == 'adopt':
            child = identity(job['child_pid'])
            assert child and child['state'] != 'Z'
            assert child['command'] == job['command']
            assert child['session_id'] == child['pid'] != parent['session_id'], 'Child must have a separate session'
            assert len([j for j in state['jobs'] if j.get('status') == 'running' and j.get('gpu') == job['gpu']]) == 1
            children[job['sample_id']] = child
    assert modes['adopt'] <= 1
    return state, rows, dict(parent=parent, children=children, modes=dict(modes),
        native_signature=signature, new_controller_sha256=old.sha(__file__), gpu_pool=GPUS,
        reserve_gib=10, native_parameters_changed=False, failures_retried=False)

def handoff():
    # Validate all saved outputs BEFORE terminating the lightweight parent.
    # SIGSTOP is unsuitable here: the host resumed the stopped tmux process.
    before, rows, audit = inspect()
    parent = audit['parent']
    folder = ROOT/('parallel_handoff_'+str(time.time_ns()))
    folder.mkdir()
    old.save(folder/'pre_termination_batch_state.json', before)
    old.save(folder/'pre_termination_audit.json', audit)
    assert_identity(identity(parent['pid']), parent)
    # Only the controller PID is signaled; all known children are session leaders.
    os.kill(parent['pid'], signal.SIGTERM)
    for _ in range(50):
        if not old.live(parent['pid']):
            break
        time.sleep(.1)
    assert not old.live(parent['pid']), 'Parent has not exited; no concurrent state writer allowed'
    # Read after parent exit: it may have completed a stage during the preflight.
    state = old.read(STATE)
    assert all(state[k] == v for k, v in audit['native_signature'].items())
    assert {j['uid'] for j in state['jobs']} == {r['uid'] for r in rows}
    children = {}
    for job in state['jobs']:
        if job['status'] != 'running':
            continue
        job_mode(job, (OUTPUT/job['sample_id']).exists())
        # Also close the Popen -> child_pid snapshot window. Match the exact
        # native argv, never an unrelated process or another object's folder.
        matches = []
        for proc in Path('/proc').iterdir():
            if not proc.name.isdigit():
                continue
            try:
                item = identity(int(proc.name))
            except PermissionError:
                continue
            if item and item['state'] != 'Z' and item['command'] == job['command']:
                matches.append(item)
        assert len(matches) <= 1, 'Duplicate native stages detected'
        if matches:
            child = matches[0]
            assert child['session_id'] == child['pid'] != parent['session_id']
            job['child_pid'] = child['pid']
            children[job['sample_id']] = child
        else:
            # A terminated/reaped stage is not assumed successful. execute()
            # checks the completed native state and hashes before reload only.
            children[job['sample_id']] = None
    audit.update(children=children, signal_sent_to_parent='SIGTERM',
                 no_child_signaled=True, final_snapshot_after_parent_exit=True)
    old.save(folder/'prior_batch_state.json', state)
    old.save(folder/'audit.json', audit)
    return state, rows, audit, folder

def run():
    # Prevent two simultaneous handoff attempts before touching the prior parent.
    with (ROOT/'parallel_handoff.lock').open('a') as handoff_lock:
        fcntl.flock(handoff_lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        state, rows, audit, archive = handoff()
        with (ROOT/'controller.lock').open('a') as lock1, (OUTPUT/'controller.lock').open('a') as lock2:
            fcntl.flock(lock1, fcntl.LOCK_EX|fcntl.LOCK_NB)
            fcntl.flock(lock2, fcntl.LOCK_EX|fcntl.LOCK_NB)
            run_locked(state, rows, audit, archive)

def run_locked(state, rows, audit, archive):
    byid = {r['sample_id']: r for r in rows}
    state.update(pid=os.getpid(), schema='rigid68_parallel_handoff_v2',
                 prior_controller_sha256=state['controller_sha256'], controller_sha256=old.sha(__file__),
                 parent_only_handoff=str(archive/'audit.json'), handoff_at=time.time(),
                 gpu_pool=GPUS, workers=len(GPUS), status='running', disk_reserve_gib=10)
    guard = threading.RLock()
    def snapshot():
        with guard:
            state.update(updated_at=time.time(), counts=dict(Counter(j['status'] for j in state['jobs'])))
            old.save(STATE, state)
    snapshot()
    work = queue.Queue()
    adopted = {}
    for job in state['jobs']:
        if job['status'] == 'queued':
            work.put(job)
        elif job['status'] == 'running':
            adopted[job['gpu']] = job
            job['adopted_child_identity'] = audit['children'][job['sample_id']]
    snapshot()

    def execute(job, gpu, adoption=False, lease_fd=None):
        folder = OUTPUT/job['sample_id']
        try:
            for key, path in [('cohort_manifest_sha256', prior.COHORT),
                              ('source_image_manifest_sha256', old.MANIFEST),
                              ('adapter_sha256', old.S/old.METHODS['trellis2']['adapter']),
                              ('glb_reload_sha256', old.GLB_RELOAD),
                              ('backend_controller_sha256', old.__file__),
                              ('controller_sha256', __file__)]:
                assert old.sha(path) == state[key], key
            env = old.environment('trellis2')
            env['CUDA_VISIBLE_DEVICES'] = gpu
            if adoption:
                saved = job['adopted_child_identity']
                while saved is not None:
                    current = identity(saved['pid'])
                    if current is None or current['state'] == 'Z':
                        if current:
                            assert_identity(current, saved)
                        break
                    assert_identity(current, saved)
                    with guard:
                        job.update(stage='adopting_prior_native_stage', last_child_observed_at=time.time())
                        snapshot()
                    time.sleep(10)
                generated = old.read(folder/'state.json')
                assert generated['status'] == 'generated_pending_reload', 'Adopted native stage did not complete: '+str(generated.get('reason', generated['status']))
                assert generated['sample_id'] == job['sample_id']
                for name, expected in generated['output_hashes'].items():
                    assert Path(name).name == name and old.sha(folder/name) == expected
                with guard:
                    job.pop('child_pid', None)
                    job.update(adopted_generation_process_exit_code=None,
                               adopted_generation_exit_observed=False,
                               adoption_evidence='Native serialized hashes verified; both independent reload stages required')
                    snapshot()
            else:
                assert not folder.exists(), 'No existing output is ever regenerated'
                with guard:
                    job.update(status='running', gpu=gpu, stage='generate', started_at=time.time(),
                               run_dir=str(folder), generation_scheduled=True)
                    snapshot()
            commands = old.commands('trellis2', byid[job['sample_id']], folder, env)
            if adoption:
                commands = [x for x in commands if x[0] != 'generate']
            logpath = ROOT/(job['sample_id']+'.log')
            with logpath.open('a') as log:
                for stage, command, stage_env in commands:
                    # Low space delays the next stage; it never kills a native child.
                    while old.shutil.disk_usage(old.P).free <= 10*1024**3:
                        with guard:
                            job.update(waiting_reason='disk_reserve_10GiB'); snapshot()
                        time.sleep(15)
                    with guard:
                        job.pop('waiting_reason', None)
                        job.update(stage=stage, command=command, log=str(logpath)); snapshot()
                    child = subprocess.Popen(command, env=stage_env, stdout=log, stderr=log,
                                             start_new_session=True,
                                             pass_fds=() if lease_fd is None else (lease_fd,))
                    with guard:
                        job['child_pid'] = child.pid; snapshot()
                    rc = child.wait()
                    with guard:
                        job.pop('child_pid', None)
                        job.setdefault('stage_exit_codes', {})[stage] = rc
                        snapshot()
                    assert rc == 0, f'{stage} exited {rc}; see {logpath}'
            result = old.validate_output('trellis2', folder)
            with guard:
                job.update(status='completed', **result)
        except Exception as error:
            with guard:
                job.update(status='failed', reason=repr(error))
        finally:
            with guard:
                job.pop('child_pid', None)
                job['finished_at'] = time.time()
                snapshot()

    def worker(gpu):
        leases = old.P/'runs/rigid111_gpu_locks'
        with (leases/('gpu_'+gpu+'.lock')).open('a') as lease:
            # Match the shared lock used by the human dispatcher.
            fcntl.flock(lease, fcntl.LOCK_EX)
            if gpu in adopted:
                execute(adopted[gpu], gpu, adoption=True, lease_fd=lease.fileno())
            while not work.empty():
                ready, detail = old.available(gpu)
                if not ready or detail['free_mib'] <= 45000 or old.shutil.disk_usage(old.P).free <= 10*1024**3:
                    with guard:
                        state.setdefault('slots', {})[gpu] = dict(status='waiting_resources', observation=detail)
                        snapshot()
                    time.sleep(10)
                    continue
                try:
                    job = work.get_nowait()
                except queue.Empty:
                    return
                with guard:
                    state.setdefault('slots', {})[gpu] = dict(status='running', sample_id=job['sample_id'])
                try:
                    execute(job, gpu, lease_fd=lease.fileno())
                finally:
                    work.task_done()
            with guard:
                state.setdefault('slots', {})[gpu] = dict(status='finished'); snapshot()

    with ThreadPoolExecutor(max_workers=len(GPUS)) as pool:
        for future in [pool.submit(worker, gpu) for gpu in GPUS]:
            future.result()
    state.update(status='completed' if all(j['status'] == 'completed' for j in state['jobs']) else 'completed_with_failures_or_missing_views', finished_at=time.time())
    snapshot()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--handoff', action='store_true')
    args = parser.parse_args()
    if args.handoff:
        run()
    else:
        _, _, audit = inspect()
        print(json.dumps(audit, indent=2))
