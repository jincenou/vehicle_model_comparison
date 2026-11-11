"""Frozen500 continuation: 68 selected additions, immutable original111 backends.

References completed outputs in place. Adopts completed generation by running only
missing reload stages. Never regenerates a partial or previously failed output.
No signal is sent to the superseded controller or its children by this program.
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
import subprocess
import threading
import time
import traceback

from PIL import Image
import dispatch_rigid111_direct as old

P, R, S = old.P, old.R, old.S
COHORT = R / 'inputs/rigid500_nusc100_v1/manifest.json'
GPUS = {'ah':['0'], 'trellis_original':['1','4'], 'hunyuan21':['2'], 'recon02':['3'], 'trellis2':['5']}
ACTIVE = {'running','preparing'}
READY = {'queued','reload_pending','audited_recovery_pending'}


def recovery_records(path, manifest, method, rows, verify_evidence=True):
    if path is None: return {}, {}
    plan = old.read(path)
    assert plan['cohort_manifest_sha256'] == old.sha(manifest)
    assert len({(x['method'],x['sample_id']) for x in plan['records']}) == len(plan['records'])
    selected = {x['sample_id']:x for x in rows}
    result = {}
    for item in plan['records']:
        if item['method'] != method: continue
        sid = item['sample_id']
        assert sid in selected and item['uid'] == selected[sid]['uid']
        assert item['reason'] and item['evidence']
        assert item['action'] in ('resume_serialized','archive_load_only')
        if verify_evidence:
            for evidence in item['evidence']: assert old.sha(evidence['path']) == evidence['sha256']
        if item['action'] == 'resume_serialized':
            script = Path(item['script_path']).resolve()
            assert script.is_relative_to(S.resolve()) and old.sha(script)==item['script_sha256']
            assert str(script) in item['command']
            assert Path(item['command'][0]).resolve() == (old.ENVS/old.METHODS[method]['env']/'bin/python').resolve()
            assert item['sample_id'] in item['command']
        else:
            assert method == 'trellis_original', 'Only explicitly audited load-only original TRELLIS is eligible'
        result[sid] = item
    return result, dict(recovery_plan=str(Path(path).resolve()),recovery_plan_sha256=old.sha(path))


def check_recovery(item, folder):
    for evidence in item['evidence']: assert old.sha(evidence['path']) == evidence['sha256']
    if item['action'] == 'archive_load_only':
        assert {x.name for x in folder.iterdir()} == {'state.json'}, 'Any generated output forbids fresh inference'
        state = old.read(folder/'state.json')
        assert state['status']=='running' and state['stage']=='load' and state['sample_id']==item['sample_id']
        assert not old.live(state.get('pid')), 'Interrupted loader still live'
    else:
        assert old.sha(item['script_path']) == item['script_sha256']


def migration_classify(method, row, legacy_job, output_root, recoveries):
    status, details = classify(method,row,legacy_job,output_root)
    item = recoveries.get(row['sample_id'])
    if item and status == 'blocked_partial_generation':
        check_recovery(item,output_root/row['sample_id'])
        return 'audited_recovery_pending', dict(recovery=item, prior_classification=status,
            reason=item['reason'], generation_already_serialized=item['action']=='resume_serialized')
    return status, details


def cohort_rows(manifest=COHORT):
    data = old.read(manifest)
    objects = data['objects']
    assert len(objects) == 500 and len({x['uid'] for x in objects}) == 500
    previous = [x for x in objects if x['cohort_origin'] == 'previous_frozen432']
    selected = [x for x in objects if x['cohort_origin'] == 'user_approved_nonvehicle111']
    assert len(previous) == 432 and len(selected) == 68, 'Expected unchanged432 plus selected68'
    original = old.read(R / 'inputs/vehicle432_image_inputs_v1/manifest.json')['objects']
    assert {x['uid'] for x in previous} == {x['uid'] for x in original}, 'Original432 membership changed'
    images = old.read(old.MANIFEST)['objects']; byuid = {x['uid']:x for x in images}
    assert len(images) == 111 and len(byuid) == 111
    for item in selected:
        assert item['uid'] in byuid, 'Selected addition outside frozen111'
        if 'sample_id' in item: assert item['sample_id'] == byuid[item['uid']]['sample_id']
    wanted = {x['uid'] for x in selected}
    rows = [x for x in images if x['uid'] in wanted]
    assert len(rows) == 68 and len({x['sample_id'] for x in rows}) == 68
    assert old.PILOT in {x['sample_id'] for x in rows}, 'Retain the verified real-input pilot'
    return rows


def method_preflight(method, manifest=COHORT):
    images = cohort_rows(manifest)
    for row in images:
        assert row['real_input_count'] == 1 and row['generated_input_count'] == 0
        for key in ['image_rgba']+(['source_rgb','source_mask'] if method=='ah' else []):
            assert old.sha(row[key]) == row[key+'_sha256']
        with Image.open(row['image_rgba']) as im:
            assert im.mode == 'RGBA' and im.size == (512,512) and im.getchannel('A').getbbox()
    cfg = old.METHODS[method]
    assert (old.ENVS / cfg['env'] / 'bin/python').is_file()
    provenance = old.read(old.PROVENANCE)
    used = {cfg['adapter']} | ({cfg['reload']} if 'reload' in cfg else set())
    for item in provenance['adapters']:
        if Path(item['destination']).name in used:
            assert old.sha(item['destination']) == item['destination_sha256']
            assert old.sha(item['source']) == item['source_sha256']
    signature = dict(cohort_manifest=str(Path(manifest).resolve()), cohort_manifest_sha256=old.sha(manifest),
                     source_image_manifest_sha256=old.sha(old.MANIFEST), backend_controller_sha256=old.sha(old.__file__),
                     controller_sha256=old.sha(__file__), adapter_sha256=old.sha(S/cfg['adapter']),
                     adapter_provenance_sha256=old.sha(old.PROVENANCE))
    if 'reload' in cfg: signature['reload_sha256'] = old.sha(S/cfg['reload'])
    if method in ('hunyuan21','recon02','trellis2'): signature['glb_reload_sha256'] = old.sha(old.GLB_RELOAD)
    if method == 'hunyuan21': assert old.read(P/'runs/hunyuan21_official_demo_v1/state.json')['official_example_validated']
    rows = images
    if method == 'recon02':
        recon = old.read(old.RECON_MANIFEST)['objects']; byuid = {x['uid']:x for x in recon}
        rows = [byuid[x['uid']] for x in images]
        for row, image in zip(rows, images):
            assert row['sample_id'] == image['sample_id']
            assert row['generated_input_count'] == 0 and row['real_input_count'] == len(row['views']) >= 0
            seen = set()
            for v in row['views']:
                assert v['split'] == 'train' and old.sha(v['image']) == v['sha256']
                assert old.sha(v['raw_image']) == v['raw_sha256']
                identity = v.get('sample_data_token') or v['raw_image']
                assert identity not in seen, 'No duplicate real views'; seen.add(identity)
        signature['source_recon_manifest_sha256'] = old.sha(old.RECON_MANIFEST)
    legacy_path = P/'runs'/(method+'_rigid111_v1')/'batch_state.json'
    legacy = old.read(legacy_path)
    assert legacy['image_manifest_sha256'] == signature['source_image_manifest_sha256']
    assert legacy['adapter_sha256'] == signature['adapter_sha256']
    assert legacy['controller_sha256'] == signature['backend_controller_sha256']
    assert legacy['provenance_sha256'] == signature['adapter_provenance_sha256']
    if 'reload_sha256' in signature: assert legacy['reload_sha256'] == signature['reload_sha256']
    if 'glb_reload_sha256' in signature: assert legacy['glb_reload_sha256'] == signature['glb_reload_sha256']
    if method == 'recon02': assert legacy['manifest_sha256'] == signature['source_recon_manifest_sha256']
    assert len(legacy['jobs']) == 111
    byuid = {x['uid']:x for x in legacy['jobs']}
    assert all(row['uid'] in byuid and byuid[row['uid']]['sample_id']==row['sample_id'] for row in rows)
    return rows, signature, legacy


def child_live(job):
    pid = job.get('child_pid')
    if not old.live(pid): return False
    expected = job.get('command', [])
    if expected:
        try:
            actual = Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')
            actual = [x.decode(errors='replace') for x in actual if x]
        except FileNotFoundError:
            return False
        except PermissionError:
            return True
        # argv[0] may resolve symlinks, but the runner path and arguments must match.
        assert actual[1:] == expected[1:], f'Prior child PID {pid} identity differs; audit before adoption'
    return True


def classify(method, row, legacy_job, output_root):
    """Read-only migration classification. No existing folder ever schedules generation."""
    if legacy_job.get('status') == 'completed':
        old.verify_completed(legacy_job)
        return 'completed', dict(reused_prior_output=True, output=legacy_job['output'], output_sha256=legacy_job['output_sha256'],
                                 validations=legacy_job['validations'], run_dir=legacy_job.get('run_dir'))
    if method == 'recon02' and not row['views']:
        return 'blocked_missing_eligible_real_view', dict(reason='No eligible real training view; retained in frozen68 denominator')
    if child_live(legacy_job):
        return 'waiting_prior_child', dict(prior_child_pid=legacy_job['child_pid'])
    if legacy_job.get('status') == 'failed':
        return 'failed', dict(reason=legacy_job.get('reason','Prior failed attempt; not regenerated'), inherited_failure=True)
    folder = output_root / row['sample_id']
    if not folder.exists():
        if legacy_job.get('status') not in ('queued','blocked_by_pilot'):
            return 'blocked_partial_generation', dict(reason='Prior job was started but no complete output exists; generation will not be repeated automatically')
        ah_inputs = R/'inputs/ah/rigid111_v1'/row['sample_id']
        if method == 'ah' and ah_inputs.exists():
            return 'blocked_partial_generation', dict(reason='Prior AH staging directory exists without a completed output; manual audit required')
        return 'queued', {}
    state_path = folder / ('job.json' if method=='ah' else 'state.json')
    if not state_path.exists():
        return 'blocked_partial_generation', dict(reason='Existing output directory has no generation status; never overwrite or regenerate')
    state = old.read(state_path)
    allowed = {'ah':{'inference_completed_pending_reload','official_demo_passed_pending_common_pilot'},
               'trellis_original':{'generated_pending_reload'}, 'trellis2':{'generated_pending_reload'},
               'recon02':{'generated_pending_independent_reload'}, 'hunyuan21':{'generated_and_mesh_reload_passed_pending_render'}}
    if state.get('status') not in allowed[method]:
        return 'blocked_partial_generation', dict(reason='Existing generation state is '+str(state.get('status'))+'; no automatic regeneration', prior_generation_reason=state.get('reason'))
    return 'reload_pending', dict(reused_prior_generation=True, run_dir=str(folder),
                                 prior_generation_exit_code='not inherited; completed serialization must pass independent reload',
                                 prior_generation_state_sha256=old.sha(state_path))


def reload_report_present(method, stage, folder):
    path = folder/('reload_validation/validation.json' if method=='ah' else 'reload_render_validation.json' if stage=='glb_reload' else 'reload_validation.json')
    if not path.exists(): return False
    data = old.read(path)
    allowed = {'passed','passed_pending_process_exit','render_passed_pending_process_exit_and_visual','rendered_pending_visual_validation'}
    # GLB reload records structure and images without a mandatory status in older versions.
    if stage=='glb_reload':
        assert data['mesh_objects'] > 0 and data['texture_images'] > 0 and len(data['views']) == 2
        assert data['output_sha256'] == old.sha(old.read(folder/'state.json')['output'])
        for image in data['views']: assert Path(image).is_file()
    else:
        assert data.get('status') in allowed, 'Existing incomplete reload report needs audit; do not overwrite'
        if method=='ah':
            assert data['ply_sha256'] == old.sha(old.read(folder/'job.json')['output_ply'])
        else:
            for view in data['views']:
                assert old.sha(view['path']) == view['sha256']
    return True


def commands_for(method, row, folder, env, generation_allowed):
    commands = old.commands(method,row,folder,env)
    if generation_allowed: return commands
    result = []
    for stage, command, stage_env in commands:
        if stage == 'generate': continue
        if not reload_report_present(method,stage,folder): result.append((stage,command,stage_env))
    return result


def run_method(method, manifest=COHORT, resume=False, recovery_plan=None):
    rows, signature, legacy = method_preflight(method,manifest)
    recoveries, recovery_signature = recovery_records(recovery_plan,manifest,method,rows,verify_evidence=not resume)
    signature.update(recovery_signature)
    cfg = old.METHODS[method]
    output_root = P/'runs'/(method+'_rigid111_v1')
    root = P/'runs'/(method+'_rigid68_v1'); root.mkdir(parents=True,exist_ok=True)
    # Both locks are kept for the complete continuation, preventing a stale111 restart.
    with (root/'controller.lock').open('a') as new_lock, (output_root/'controller.lock').open('a') as legacy_lock:
        fcntl.flock(new_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        fcntl.flock(legacy_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert not old.live(legacy.get('pid')), 'Superseded111 controller still live/stopped; root must terminate parent only after snapshot'
        path = root/'batch_state.json'
        byuid = {x['uid']:x for x in rows}; legacy_byuid = {x['uid']:x for x in legacy['jobs']}
        lock = threading.RLock()
        if path.exists():
            assert resume, 'Use --resume only after auditing prior continuation'
            state = old.read(path)
            assert all(state[k]==v for k,v in signature.items())
            assert not old.live(state.get('pid'))
            assert all(not old.live(x.get('child_pid')) for x in state['jobs'])
            assert all(x['status'] not in ACTIVE for x in state['jobs']), 'Interrupted continuation requires explicit audit'
            for job in state['jobs']:
                if job['status']=='completed': old.verify_completed(job)
        else:
            assert not resume
            legacy_raw = (output_root/'batch_state.json').read_bytes()
            (root/'source111_batch_snapshot.json').write_bytes(legacy_raw)
            state = dict(schema='rigid68_continuation_v1',method=method,status='migration_preflight_passed',
                started_at=time.time(),**signature,workers=len(GPUS[method]),gpu_pool=GPUS[method],disk_reserve_gib=10,
                source111_batch_snapshot_sha256=old.sha(root/'source111_batch_snapshot.json'),
                original432_unchanged=True,planned_new_objects=68,full_benchmark_objects=500,
                native_output_root=str(output_root),common_camera_qualified=False,formal_metrics_qualified=False,
                removed_from_queue=[dict(sample_id=x['sample_id'],uid=x['uid'],prior_status=x['status'],
                                        reason='Not in selected68: duplicate/redundancy removal; not necessarily the same physical entity')
                                    for x in legacy['jobs'] if x['uid'] not in byuid],
                jobs=[dict(sample_id=x['sample_id'],uid=x['uid'],category=x['category'],status='migration_pending') for x in rows])
            assert len(state['removed_from_queue']) == 43
        state.update(pid=os.getpid(),resumed_at=time.time())
        def snapshot():
            with lock:
                state.update(updated_at=time.time(),counts=dict(Counter(x['status'] for x in state['jobs'])))
                old.save(path,state)
        snapshot()
        try:
            # Allow every in-flight old child stage, including removed objects, to finish.
            # None is killed, restarted, or deleted; removed objects get no further stage.
            while True:
                waiting=[dict(sample_id=x['sample_id'],pid=x['child_pid'],retained=x['uid'] in byuid) for x in legacy['jobs'] if child_live(x)]
                if not waiting: break
                state.update(status='waiting_prior_child_stages',waiting_prior_children=waiting);snapshot();time.sleep(10)
            state.pop('waiting_prior_children',None)
            for job in state['jobs']:
                if job['status'] in ('migration_pending','waiting_prior_child'):
                    status, details = migration_classify(method,byuid[job['uid']],legacy_byuid[job['uid']],output_root,recoveries)
                    job.update(status=status,**details)
            snapshot()
            def execute(job,gpu,phase):
                assert old.sha(manifest)==signature['cohort_manifest_sha256']
                assert old.sha(old.MANIFEST)==signature['source_image_manifest_sha256']
                assert old.sha(S/cfg['adapter'])==signature['adapter_sha256']
                if 'reload' in cfg: assert old.sha(S/cfg['reload'])==signature['reload_sha256']
                if method=='recon02': assert old.sha(old.RECON_MANIFEST)==signature['source_recon_manifest_sha256']
                if recovery_plan is not None: assert old.sha(recovery_plan)==signature['recovery_plan_sha256']
                folder = output_root/job['sample_id']; row=byuid[job['uid']]
                generation_allowed = job['status']=='queued'
                recovery = job.get('recovery') if job['status']=='audited_recovery_pending' else None
                assert job['status'] in READY
                if generation_allowed: assert not folder.exists(), 'Never regenerate existing folder'
                leases=P/'runs/rigid111_gpu_locks';leases.mkdir(exist_ok=True)
                with (leases/('gpu_'+gpu+'.lock')).open('a') as lease:
                    fcntl.flock(lease,fcntl.LOCK_EX)
                    while True:
                        ready,detail=old.available(gpu)
                        free=old.shutil.disk_usage(P).free
                        with lock:
                            job.update(waiting_gpu=gpu,waiting_gpu_observation=detail,free_bytes=free)
                            snapshot()
                        if ready and free>10*1024**3: break
                        time.sleep(10)
                    with lock:
                        job.update(status='running',gpu=gpu,phase=phase,started_at=time.time(),run_dir=str(folder),generation_scheduled=generation_allowed)
                        job.pop('waiting_gpu',None); state['status']='running';snapshot()
                    try:
                        env=old.environment(method);env['CUDA_VISIBLE_DEVICES']=gpu
                        if recovery:
                            check_recovery(recovery,folder)
                            if recovery['action']=='archive_load_only':
                                archive=root/'interrupted_load_metadata'/job['sample_id']
                                archive.parent.mkdir(exist_ok=True)
                                assert not archive.exists()
                                folder.rename(archive)
                                old_log=output_root/(job['sample_id']+'.log')
                                if old_log.exists():
                                    assert old_log.stat().st_size < 16*1024**2
                                    (archive/'prior_controller.log').write_bytes(old_log.read_bytes())
                                generation_allowed=True
                                with lock:job.update(load_only_metadata_archive=str(archive),generation_scheduled=True);snapshot()
                        if method=='ah':
                            if generation_allowed: old.prepare_ah(row,phase,env)
                            else: env.update(AB_JOB_ID=row['sample_id'],AB_PHASE=phase,AB_INPUT_ROOT=str(R/'inputs/ah/rigid111_v1'/row['sample_id']))
                        commands=commands_for(method,row,folder,env,generation_allowed)
                        if recovery and recovery['action']=='resume_serialized':
                            commands=[('resume_serialized',recovery['command'],env)]+commands
                        prior_report_exit = {}
                        for report_path in [folder/'reload_validation.json',folder/'reload_render_validation.json',folder/'reload_validation/validation.json']:
                            if report_path.exists(): prior_report_exit[str(report_path)] = old.read(report_path).get('process_exit_code')
                        stage_exit_codes = {}
                        logpath=root/(job['sample_id']+'.log');job['log']=str(logpath)
                        with logpath.open('a') as log:
                            for stage,command,stage_env in commands:
                                with lock: job.update(stage=stage,command=command);snapshot()
                                child=subprocess.Popen(command,env=stage_env,stdout=log,stderr=log,start_new_session=True)
                                with lock: job['child_pid']=child.pid;snapshot()
                                rc=child.wait()
                                stage_exit_codes[stage] = rc
                                with lock: job.pop('child_pid',None);job.update(exit_code=rc,stage_exit_codes=dict(stage_exit_codes));snapshot()
                                assert rc==0,f'{stage} exited {rc}; see {logpath}'
                        result=old.validate_output(method,folder)
                        # A recovered report is evidence of reload checks, not evidence
                        # that this controller observed the prior process exit code.
                        for item in result['validations']:
                            report_path=Path(item['path'])
                            stage='glb_reload' if report_path.name=='reload_render_validation.json' else 'native_reload'
                            if stage not in stage_exit_codes and prior_report_exit.get(str(report_path)) != 0:
                                report=old.read(report_path)
                                report.update(process_exit_code=None,process_exit_code_observed=False,
                                              adoption_note='Prior serialized reload report revalidated; prior process exit status unavailable')
                                old.save(report_path,report);item['sha256']=old.sha(report_path)
                        with lock:job.update(status='completed',**result)
                    except Exception as error:
                        with lock:job.update(status='failed',reason=repr(error),traceback=traceback.format_exc())
                    finally:
                        with lock:job.pop('child_pid',None);job['finished_at']=time.time();snapshot()
            pilot=next(x for x in state['jobs'] if x['sample_id']==old.PILOT)
            if pilot['status'] in READY:execute(pilot,GPUS[method][0],'pilot')
            if pilot['status']!='completed':
                for job in state['jobs']:
                    if job['status'] in READY:job.update(status='blocked_by_pilot',reason='Selected real pilot has not passed native reload')
                state.update(status='stopped_on_pilot_failure',finished_at=time.time());snapshot();return state['status']
            old.verify_completed(pilot)
            gate=root/'pilot_gate.json'
            gate_data=dict(status='passed',sample_id=old.PILOT,output=pilot['output'],output_sha256=pilot['output_sha256'],
                           validations=pilot['validations'],**signature,reused_native_settings=True)
            if gate.exists():assert old.read(gate)==gate_data
            else:old.save(gate,gate_data)
            state['pilot_gate_sha256']=old.sha(gate);snapshot()
            work=queue.Queue()
            for job in state['jobs']:
                if job['status'] in READY:work.put(job)
            def worker(gpu):
                while True:
                    try:job=work.get_nowait()
                    except queue.Empty:return
                    try:execute(job,gpu,'batch')
                    except Exception as error:
                        with lock:job.update(status='controller_job_error',reason=repr(error),traceback=traceback.format_exc());snapshot()
                    finally:work.task_done()
            with ThreadPoolExecutor(max_workers=len(GPUS[method])) as pool:
                for future in [pool.submit(worker,gpu) for gpu in GPUS[method]]:future.result()
            state.update(status='completed' if all(x['status']=='completed' for x in state['jobs']) else 'completed_with_failures_or_missing_views',finished_at=time.time());snapshot()
            return state['status']
        except Exception as error:
            state.update(status='controller_error',reason=repr(error),traceback=traceback.format_exc());snapshot();raise


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--manifest',type=Path,default=COHORT)
    ap.add_argument('--method',choices=['all']+list(GPUS),default='all')
    ap.add_argument('--recovery-plan',type=Path)
    ap.add_argument('--check-only',action='store_true');ap.add_argument('--resume',action='store_true')
    args=ap.parse_args();methods=list(GPUS) if args.method=='all' else [args.method]
    if args.check_only:
        result={}
        for method in methods:
            try:
                rows,signature,legacy=method_preflight(method,args.manifest)
                recoveries,recovery_signature=recovery_records(args.recovery_plan,args.manifest,method,rows)
                signature.update(recovery_signature)
                root=P/'runs'/(method+'_rigid111_v1');byuid={x['uid']:x for x in legacy['jobs']}
                classifications=Counter(migration_classify(method,row,byuid[row['uid']],root,recoveries)[0] for row in rows)
                result[method]=dict(status='preflight_passed',objects=68,gpus=GPUS[method],classifications=dict(classifications),
                    superseded_controller_pid=legacy.get('pid'),superseded_controller_live=old.live(legacy.get('pid')),
                    startup_requires_parent_terminated=True,**signature)
            except Exception as error:result[method]=dict(status='preflight_failed',reason=repr(error))
        print(json.dumps(result,indent=2));raise SystemExit(0 if all(x['status']=='preflight_passed' for x in result.values()) else 1)
    with ThreadPoolExecutor(max_workers=len(methods)) as pool:
        futures={m:pool.submit(run_method,m,args.manifest,args.resume,args.recovery_plan) for m in methods};result={}
        for method,future in futures.items():
            try:result[method]=future.result()
            except Exception as error:result[method]='controller_error:'+repr(error)
    print(json.dumps(result));raise SystemExit(0 if all(x=='completed' for x in result.values()) else 1)


if __name__=='__main__':main()
