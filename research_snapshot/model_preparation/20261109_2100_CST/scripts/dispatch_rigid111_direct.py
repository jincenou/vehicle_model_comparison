"""Isolated 111-object continuation. Fixed independent GPUs; native pilot per method.

No model imports occur in this controller. --check-only never launches processes.
Original 432 manifests, adapters, outputs and states remain immutable.
"""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import traceback

from PIL import Image

P = Path(__file__).resolve().parents[1]
R = P.parents[1]
S = P / 'scripts'
MANIFEST = R / 'inputs/rigid111_image_inputs_v1/manifest.json'
RECON_MANIFEST = R / 'inputs/recon02_rigid111_real_multiview_v1/manifest.json'
PILOT = 'scene-0063_t000314'
PROVENANCE = S / 'rigid111_adapter_provenance.json'
GLB_RELOAD = S / 'render_hunyuan21_demo_reload.py'
ENVS = Path(('/opt/conda/envs'.replace('/opt/conda', __import__("os").environ.get('RIGID_CONDA_ROOT', '/opt/conda')).replace('/opt/rigid-benchmark/project', __import__("os").environ.get('RIGID_PROJECT_ROOT', '/opt/rigid-benchmark/project'))))
METHODS = {
    'ah': dict(gpu='0', env='ab_ah', adapter='run_ah_rigid111_object.py', reload='reload_ah_rigid111_object.py'),
    'trellis_original': dict(gpu='1', env='ab_cmp_recon02', adapter='run_trellis_original_object_rigid111.py'),
    'hunyuan21': dict(gpu='2', env='ab_cmp_hunyuan21', adapter='run_hunyuan21_frozen_object_rigid111.py'),
    'recon02': dict(gpu='3', env='ab_cmp_recon02', adapter='run_recon02_frozen_object_rigid111.py', reload='reload_recon02_frozen_object_rigid111.py'),
    'trellis2': dict(gpu='5', env='ab_cmp_trellis2', adapter='run_trellis2_object_rigid111.py'),
}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, data):
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    tmp.replace(path)


def contained(path, root):
    assert Path(path).resolve().is_relative_to(Path(root).resolve()), (path, root)


def live(pid):
    if not isinstance(pid, int) or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return not Path(f'/proc/{pid}/stat').read_text().split(') ', 1)[1].startswith('Z ')
    except (ProcessLookupError, FileNotFoundError):
        return False
    except PermissionError:
        return True  # Unobservable other-owner processes are not presumed absent.


def old_reservations():
    """Respect CPU phases of existing dispatchers as well as CUDA process occupancy."""
    candidates = list((R / 'state').glob('*.json'))
    candidates += list((P / 'runs').glob('*vehicle432*/batch_state.json'))
    candidates += list((R / 'runs/ah_batches').glob('*/state.json'))
    reservations = []
    for path in candidates:
        try:
            data = read(path)
        except FileNotFoundError:
            continue
        except (ValueError, OSError) as error:
            raise RuntimeError(f'Cannot safely inspect reservation {path}: {error}')
        if not isinstance(data, dict):
            continue
        controller_live = live(data.get('pid'))
        if controller_live and data.get('gpu') is not None:
            reservations.append(dict(gpu=str(data['gpu']), source=str(path), pid=data['pid']))
        jobs = data.get('jobs', [])
        jobs = list(jobs.values()) if isinstance(jobs, dict) else list(jobs) if isinstance(jobs, list) else []
        if isinstance(data.get('official_example'), dict):
            jobs.append(data['official_example'])
        for job in jobs:
            if not isinstance(job, dict) or job.get('status') not in ('running', 'preparing'):
                continue
            gpu = job.get('gpu', job.get('gpu_index'))
            child_live = any(live(job.get(k)) for k in ('child_pid', 'pid', 'worker_pid'))
            if gpu is not None and (controller_live or child_live):
                reservations.append(dict(gpu=str(gpu), source=str(path), controller_live=controller_live, child_live=child_live))
    return reservations


def available(gpu):
    inventory = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.free', '--format=csv,noheader,nounits'], text=True)
    occupied = set(subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid', '--format=csv,noheader'], text=True).splitlines())
    reservations = old_reservations()
    entry = next((line.split(',') for line in inventory.splitlines() if line.split(',')[0].strip() == gpu), None)
    assert entry is not None, f'GPU {gpu} unavailable'
    uuid = entry[1].strip()
    ready = uuid not in {x.strip() for x in occupied} and not any(x['gpu'] == gpu for x in reservations)
    # Prior pipelines used >40 GiB (TRELLIS.2 >45 GiB); an idle healthy 48-GiB card meets these.
    minimum = 45000 if gpu == METHODS['trellis2']['gpu'] else 40000
    return ready and int(entry[2]) > minimum, dict(uuid=uuid, free_mib=int(entry[2]), reservations=[x for x in reservations if x['gpu'] == gpu])


def validate_population(rows):
    assert len(rows) == 111, f'Expected frozen111; got {len(rows)}'
    assert len({x['uid'] for x in rows}) == len(rows), 'Duplicate UID'
    assert len({x['sample_id'] for x in rows}) == len(rows), 'Duplicate sample_id requires an explicit output-key adapter'
    assert PILOT in {x['sample_id'] for x in rows}, 'Approved real pilot absent'
    for row in rows:
        sid = row['sample_id']
        assert '/' not in sid and '..' not in sid and sid.startswith('scene-'), sid
        assert row['category'] and row['scene'] and row['uid']


def preflight(method):
    cfg = METHODS[method]
    records = read(MANIFEST)['objects']
    validate_population(records)
    for row in records:
        assert row['real_input_count'] == 1 and row['generated_input_count'] == 0, 'Frozen source must be real'
        for key in ['image_rgba'] + (['source_rgb', 'source_mask'] if method == 'ah' else []):
            assert sha(row[key]) == row[key+'_sha256'], (row['sample_id'], key)
        with Image.open(row['image_rgba']) as im:
            assert im.mode == 'RGBA' and im.size == (512, 512)
            assert im.getchannel('A').getbbox() is not None, 'Empty input mask'
    signature = dict(image_manifest_sha256=sha(MANIFEST), controller_sha256=sha(__file__),
                     adapter_sha256=sha(S / cfg['adapter']), provenance_sha256=sha(PROVENANCE))
    assert (ENVS / cfg['env'] / 'bin/python').is_file()
    provenance = read(PROVENANCE)
    used = {cfg['adapter']} | ({cfg['reload']} if 'reload' in cfg else set())
    for entry in provenance['adapters']:
        if Path(entry['destination']).name in used:
            assert sha(entry['destination']) == entry['destination_sha256'], 'Isolated adapter changed'
            assert sha(entry['source']) == entry['source_sha256'], 'Legacy adapter changed since isolation'
    if 'reload' in cfg:
        signature['reload_sha256'] = sha(S / cfg['reload'])
    if method in ('hunyuan21', 'recon02', 'trellis2'):
        signature['glb_reload_sha256'] = sha(GLB_RELOAD)
        assert (ENVS / 'ab_cmp_hunyuan21/bin/python').is_file()
    if method == 'recon02':
        rows = read(RECON_MANIFEST)['objects']; validate_population(rows)
        assert {x['uid'] for x in rows} == {x['uid'] for x in records}, 'Recon cohort mismatch'
        base = {x['uid']: x for x in records}
        for row in rows:
            assert row['sample_id'] == base[row['uid']]['sample_id']
            assert row['generated_input_count'] == 0 and row['real_input_count'] == len(row['views']) >= 0
            raw_ids = []
            for view in row['views']:
                assert view['split'] == 'train' and sha(view['image']) == view['sha256']
                assert view.get('raw_image') and view.get('raw_sha256'), 'Real view provenance required'
                assert sha(view['raw_image']) == view['raw_sha256']
                raw_ids.append(view.get('sample_data_token') or view['raw_image'])
            assert len(raw_ids) == len(set(raw_ids)), 'Do not duplicate real views'
        signature['manifest_sha256'] = sha(RECON_MANIFEST)
        records = rows
    else:
        signature['manifest_sha256'] = signature['image_manifest_sha256']
    if method == 'hunyuan21':
        assert read(P / 'runs/hunyuan21_official_demo_v1/state.json')['official_example_validated']
    return records, signature


def environment(method):
    cfg = METHODS[method]; prefix = ENVS / cfg['env']
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=cfg['gpu'], CUDA_HOME=str(prefix),
               LD_LIBRARY_PATH=str(prefix / 'lib'), PATH=str(prefix / 'bin')+':'+os.environ['PATH'],
               HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', PYTHONDONTWRITEBYTECODE='1',
               OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', MAX_JOBS='2',
               TMPDIR=str(P / 'cache/tmp'), XDG_CACHE_HOME=str(P / 'cache/xdg'),
               MPLCONFIGDIR=str(P / 'cache/matplotlib'))
    if method in ('recon02', 'trellis_original'):
        env.update(TORCH_HOME=str(P / 'cache/recon02_torch'), HF_HOME=str(P / 'cache/recon02_offline'))
    elif method == 'hunyuan21':
        env.update(TORCH_HOME=str(P / 'cache/torch'), HF_HOME=str(P / 'cache/hunyuan_offline'), HF_HUB_CACHE=str(P / 'cache/hunyuan_offline/hub'))
    elif method == 'trellis2':
        env.update(TORCH_HOME=str(P / 'cache/trellis2_torch'), HF_HOME=str(P / 'cache/trellis2_offline'),
                   HF_HUB_CACHE=str(P / 'cache/trellis2_offline/hub'),
                   TORCH_EXTENSIONS_DIR=str(P / 'cache/torch_ext/trellis2'), TRITON_CACHE_DIR=str(P / 'cache/triton/trellis2'))
    else:
        base = R.parent / 'assetbench'
        env.update(AB_AH_CRADIO_PATH=str(base / 'weights/ah/C-RADIO'), AB_AH_VAE_PATH=str(base / 'weights/ah/dc-ae'),
                   HF_HOME=str(R / 'cache/hf'), HF_HUB_CACHE=str(R / 'cache/hf/hub'), HF_MODULES_CACHE=str(R / 'cache/hf/modules'),
                   TORCH_HOME=str(R / 'cache/torch'), TORCH_EXTENSIONS_DIR=str(R / 'cache/torch_ext/ab_ah'),
                   XDG_CACHE_HOME=str(R / 'cache/xdg'), TRITON_CACHE_DIR=str(R / 'cache/triton'), TMPDIR=str(R / 'cache/tmp'))
    for key in ['TMPDIR', 'XDG_CACHE_HOME', 'MPLCONFIGDIR', 'TORCH_EXTENSIONS_DIR', 'TRITON_CACHE_DIR']:
        if key in env:
            Path(env[key]).mkdir(parents=True, exist_ok=True)
    return env


def prepare_ah(row, phase, env):
    root = R / 'inputs/ah/rigid111_v1' / row['sample_id']
    root.mkdir(parents=True, exist_ok=False)
    folder = root / row['sample_id']; folder.mkdir()
    for key, name in [('source_rgb', 'frame.jpeg'), ('source_mask', 'mask.png')]:
        assert sha(row[key]) == row[key+'_sha256']
        shutil.copyfile(row[key], folder / name)
    save(root / 'source_manifest.json', dict(sample_id=row['sample_id'], uid=row['uid'], image=str(folder / 'frame.jpeg'),
         mask=str(folder / 'mask.png'), source_image_sha256=row['source_rgb_sha256'], source_mask_sha256=row['source_mask_sha256'],
         real_input_count=1, generated_input_count=0, data_nature='real_photo', frozen_manifest_sha256=sha(MANIFEST)))
    env.update(AB_JOB_ID=row['sample_id'], AB_INPUT_ROOT=str(root), AB_PHASE=phase)


def commands(method, row, folder, env):
    cfg = METHODS[method]; py = str(ENVS / cfg['env'] / 'bin/python'); sid = row['sample_id']
    base = [py, '-B', '-u', str(S / cfg['adapter'])]
    if method == 'ah':
        return [('generate', base, env), ('native_reload', [py, '-B', '-u', str(S / cfg['reload']), sid], env)]
    base += ['--sample-id', sid, '--run-dir', str(folder)]
    result = [('generate', base, env)]
    if method in ('trellis_original', 'trellis2'):
        result.append(('native_reload', base+['--reload-only'], env))
    elif method == 'recon02':
        result.append(('native_reload', [py, '-B', '-u', str(S / cfg['reload']), '--run-dir', str(folder)], env))
    if method in ('hunyuan21', 'recon02', 'trellis2'):
        blender = ENVS / 'ab_cmp_hunyuan21'
        result.append(('glb_reload', [str(blender / 'bin/python'), '-B', '-u', str(GLB_RELOAD), '--run-dir', str(folder)],
                       dict(env, LD_LIBRARY_PATH=str(blender / 'lib'), CUDA_VISIBLE_DEVICES='')))
    return result


def validate_output(method, folder):
    """Serialize/load equality and finite native tensors are runner gates; pixel count is diagnostic."""
    reports = []
    if method == 'ah':
        state = read(folder / 'job.json')
        assert state['status'] == 'official_demo_passed_pending_common_pilot'
        output = Path(state['output_ply']); validation = Path(state['reload_validation'])
        contained(output, folder); contained(validation, folder)
        report = read(validation)
        assert report['status'] == 'passed' and report['gaussian_count'] > 0 and sha(output) == report['ply_sha256']
        reports.append(validation)
    else:
        state = read(folder / 'state.json')
        output = folder / 'native_gaussians.pt' if method == 'trellis_original' else Path(state['output'])
        contained(output, folder)
        expected = state['output_hashes']['native_gaussians.pt'] if method == 'trellis_original' else state['output_sha256']
        assert sha(output) == expected
        if method in ('trellis_original', 'recon02', 'trellis2'):
            validation = folder / 'reload_validation.json'; report = read(validation)
            assert report['status'] in ('passed_pending_process_exit', 'render_passed_pending_process_exit_and_visual', 'passed')
            report.update(status='passed', process_exit_code=0, scope='Native finite serialization/reload integrity only; low visibility is retained as a diagnostic')
            save(validation, report); reports.append(validation)
        if method in ('hunyuan21', 'recon02', 'trellis2'):
            validation = folder / 'reload_render_validation.json'; report = read(validation)
            assert report['output_sha256'] == sha(output)
            assert report['mesh_objects'] > 0 and report['texture_images'] > 0 and len(report['views']) == 2
            diagnostic = []
            for path in report['views']:
                contained(path, folder)
                with Image.open(path) as im:
                    assert im.mode == 'RGBA' and im.size == (512,512)
                    alpha = im.getchannel('A'); hist = alpha.histogram()
                    diagnostic.append(dict(path=path, sha256=sha(path), alpha_nonzero_pixels=sum(hist[1:]),
                                           alpha_gt_half_pixels=sum(hist[128:]), sparse_render_diagnostic=sum(hist[1:]) <= 100))
            report.update(status='passed', process_exit_code=0, views_verified=diagnostic,
                          scope='Native mesh/texture reload integrity only; empty/sparse view is a quality diagnostic, not numerical corruption')
            save(validation, report); reports.append(validation)
    return dict(output=str(output), output_sha256=sha(output),
                validations=[dict(path=str(path), sha256=sha(path)) for path in reports], common_camera_qualified=False)


def verify_completed(job):
    assert sha(job['output']) == job['output_sha256']
    for item in job['validations']:
        assert sha(item['path']) == item['sha256'] and read(item['path'])['status'] == 'passed'


def run_method(method, resume=False):
    cfg = METHODS[method]
    root = P / 'runs' / (method+'_rigid111_v1'); root.mkdir(parents=True, exist_ok=True)
    with (root / 'controller.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        path = root / 'batch_state.json'; state = None
        try:
            rows, signature = preflight(method)
            if path.exists():
                assert resume, 'Existing state requires explicit --resume and liveness audit'
                previous = read(path)
                assert all(previous[k] == v for k,v in signature.items()), 'Immutable manifest or adapter signature changed'
                assert not live(previous.get('pid')), 'Prior controller remains live'
                assert all(not live(j.get('child_pid')) for j in previous['jobs']), 'Prior child remains live'
                assert all(j['status'] in ('queued','completed','failed','blocked_by_pilot','blocked_missing_eligible_real_view') for j in previous['jobs']), 'Interrupted in-flight jobs require explicit audit; no automatic restart'
                for job in previous['jobs']:
                    if job['status'] == 'completed': verify_completed(job)
                state = previous
            else:
                assert not resume, 'No previous state to resume'
                state = dict(schema='rigid111_direct_batch_v1', method=method, fixed_gpu=cfg['gpu'], workers=1,
                    pilot_sample_id=PILOT, manifest=str(RECON_MANIFEST if method == 'recon02' else MANIFEST),
                    status='preflight_passed', started_at=time.time(), **signature,
                    disk_reserve_gib=10, common_camera_qualified=False, formal_metrics_qualified=False,
                    jobs=[dict(sample_id=r['sample_id'],uid=r['uid'],category=r['category'],
                               status='blocked_missing_eligible_real_view' if method=='recon02' and not r['views'] else 'queued',
                               **({'reason':'No eligible real training view; original split unchanged; frozen source anchor is not independent holdout'} if method=='recon02' and not r['views'] else {})) for r in rows])
            state.update(pid=os.getpid(), resumed_at=time.time(), status='waiting_pilot')
            def snapshot():
                state.update(updated_at=time.time(), counts=dict(Counter(j['status'] for j in state['jobs'])))
                save(path, state)
            snapshot()
            byid = {x['sample_id']:x for x in rows}; jobs = {x['sample_id']:x for x in state['jobs']}
            def execute(job, phase):
                assert sha(MANIFEST) == signature['image_manifest_sha256'], 'Frozen image manifest changed'
                if method == 'recon02': assert sha(RECON_MANIFEST) == signature['manifest_sha256'], 'Frozen real views changed'
                assert sha(S / cfg['adapter']) == signature['adapter_sha256'], 'Adapter changed during batch'
                if 'reload' in cfg: assert sha(S / cfg['reload']) == signature['reload_sha256'], 'Reload adapter changed'
                if 'glb_reload_sha256' in signature: assert sha(GLB_RELOAD) == signature['glb_reload_sha256'], 'GLB reload changed'
                folder = root / job['sample_id']; assert not folder.exists(), 'Never overwrite prior object output'
                lockroot = P / 'runs/rigid111_gpu_locks'; lockroot.mkdir(exist_ok=True)
                with (lockroot / ('gpu_'+cfg['gpu']+'.lock')).open('a') as lease:
                    fcntl.flock(lease, fcntl.LOCK_EX)
                    while True:
                        ready, detail = available(cfg['gpu'])
                        space = shutil.disk_usage(P).free
                        state.update(status='waiting_gpu_or_disk', phase=phase, waiting_sample_id=job['sample_id'],
                                     gpu_observation=detail, free_bytes=space); snapshot()
                        if ready and space > 10*1024**3: break
                        time.sleep(10)
                    state.update(status='running', phase=phase); state.pop('waiting_sample_id', None)
                    job.update(status='running', gpu=cfg['gpu'], run_dir=str(folder), started_at=time.time()); snapshot()
                    try:
                        env = environment(method); row = byid[job['sample_id']]
                        if method == 'ah': prepare_ah(row, phase, env)
                        logpath = root / (job['sample_id']+'.log'); job['log'] = str(logpath)
                        with logpath.open('x') as log:
                            for stage, command, stage_env in commands(method, row, folder, env):
                                job.update(stage=stage, command=command); snapshot()
                                child = subprocess.Popen(command, env=stage_env, stdout=log, stderr=log)
                                job['child_pid'] = child.pid; snapshot()
                                rc = child.wait(); job.pop('child_pid', None); job['exit_code'] = rc; snapshot()
                                assert rc == 0, f'{stage} exited {rc}; see {logpath}'
                        job.update(status='completed', **validate_output(method, folder))
                    except Exception as error:
                        job.update(status='failed', reason=repr(error), traceback=traceback.format_exc())
                    finally:
                        job.pop('child_pid', None); job['finished_at'] = time.time(); snapshot()
            pilot = jobs[PILOT]
            if pilot['status'] == 'queued': execute(pilot, 'pilot')
            if pilot['status'] != 'completed':
                for job in state['jobs']:
                    if job['status'] == 'queued': job.update(status='blocked_by_pilot', reason='Real-input pilot failed native generation or reload; no batch launched')
                state.update(status='stopped_on_pilot_failure', finished_at=time.time()); snapshot(); return state['status']
            verify_completed(pilot)
            gate = root / 'pilot_gate.json'
            gate_data = dict(status='passed', sample_id=PILOT, uid=pilot['uid'], native_generation_and_reload=True,
                             output=pilot['output'], output_sha256=pilot['output_sha256'], validations=pilot['validations'],
                             **signature, scope='Real-input native integrity only; no output-quality filtering or common-camera qualification')
            if gate.exists():
                old = read(gate); assert old == gate_data, 'Existing pilot gate changed'
            else: save(gate, gate_data)
            state['pilot_gate_sha256'] = sha(gate); snapshot()
            for job in state['jobs']:
                if job['status'] == 'queued': execute(job, 'batch')
            state.update(status='completed' if all(j['status']=='completed' for j in state['jobs']) else 'completed_with_failures', finished_at=time.time()); snapshot()
            return state['status']
        except Exception as error:
            if state is not None:
                state.update(status='controller_error', reason=repr(error), traceback=traceback.format_exc(), updated_at=time.time())
                save(path, state)
            else:
                save(root / 'preflight_failure.json', dict(status='preflight_failed', method=method, reason=repr(error), traceback=traceback.format_exc(), at=time.time()))
            raise


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--method', choices=['all']+list(METHODS), default='all')
    ap.add_argument('--check-only', action='store_true')
    ap.add_argument('--resume', action='store_true')
    args = ap.parse_args(); selected = list(METHODS) if args.method == 'all' else [args.method]
    if args.check_only:
        result = {}
        for method in selected:
            try:
                rows, signature = preflight(method)
                result[method] = dict(status='preflight_passed', objects=len(rows), fixed_gpu=METHODS[method]['gpu'], **signature)
            except Exception as error:
                result[method] = dict(status='preflight_failed', reason=repr(error))
        print(json.dumps(result, indent=2)); raise SystemExit(0 if all(x['status']=='preflight_passed' for x in result.values()) else 1)
    with ThreadPoolExecutor(max_workers=len(selected)) as pool:
        futures = {method: pool.submit(run_method, method, args.resume) for method in selected}
        result = {}
        for method, future in futures.items():
            try: result[method] = future.result()
            except Exception as error: result[method] = 'controller_error:'+repr(error)
    print(json.dumps(result)); raise SystemExit(0 if all(v=='completed' for v in result.values()) else 1)


if __name__ == '__main__':
    main()
