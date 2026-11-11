"""Additional native workers without terminating the legacy two-worker session.

Consume unstarted jobs from the opposite end of the frozen queue. The unchanged
native adapter atomically creates its output directory before model loading, so
concurrent attempts cannot generate the same asset twice or overwrite outputs.
Legacy FileExists skips are reconciled only against explicit supplemental state;
the legacy state remains untouched and the effective coverage preserves both logs.
"""
from collections import Counter
import fcntl
import os
from pathlib import Path
import signal
import subprocess
import time
from PIL import Image
import dispatch_rigid111_direct as core

P=core.P;R=core.R
LEGACY=P/'runs/trellis2_vehicle432_v1'
OLD_STATE=LEGACY/'batch_state.json'
ROOT=P/'runs/trellis2_vehicle432_parallel_v1'
STATE=ROOT/'batch_state.json'
EFFECTIVE=ROOT/'effective_batch_state.json'
MANIFEST=R/'inputs/vehicle432_image_inputs_v1/manifest.json'
ADAPTER=P/'scripts/run_trellis2_object.py'
RELOAD=P/'scripts/render_hunyuan21_demo_reload.py'
PREFIX=core.ENVS/'ab_cmp_trellis2';BLENDER=core.ENVS/'ab_cmp_hunyuan21'
STAGES=['generate','native_reload','glb_reload']
MAX_WORKERS=4

def effective_jobs(legacy_jobs,supplement_jobs):
    peers={j['sample_id']:j for j in supplement_jobs}
    result=[]
    for old in legacy_jobs:
        new=peers.get(old['sample_id']);row=dict(old,producer='legacy')
        if new and new['status'] in ('running','completed','failed'):
            row=dict(new,producer='supplement',legacy_status=old['status'],legacy_reason=old.get('reason'))
            row['legacy_existing_output_skip']=bool(old['status']=='failed' and 'FileExistsError' in str(old.get('reason')))
        result.append(row)
    return result

def acquire(gpu):
    ready,_=core.available(gpu)
    if not ready:return None
    f=(P/'runs/rigid111_gpu_locks'/f'gpu_{gpu}.lock').open('a')
    previous=signal.getsignal(signal.SIGALRM)
    def timeout(*args):raise TimeoutError()
    signal.signal(signal.SIGALRM,timeout)
    accepted=False
    try:
        signal.setitimer(signal.ITIMER_REAL,6)
        fcntl.flock(f,fcntl.LOCK_EX)
        signal.setitimer(signal.ITIMER_REAL,0)
        ready,_=core.available(gpu)
        accepted=ready and core.shutil.disk_usage(P).free>10*1024**3
    except TimeoutError:pass
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,previous)
        if not accepted:f.close()
    return f if accepted else None

def commands(job,gpu):
    env=core.environment('trellis2');env['CUDA_VISIBLE_DEVICES']=gpu
    # Preserve all original model, seed, resolution and export parameters.
    folder=LEGACY/job['sample_id']
    base=[str(PREFIX/'bin/python'),'-B','-u',str(ADAPTER),'--sample-id',job['sample_id'],'--run-dir',str(folder)]
    return [('generate',base,env),('native_reload',base+['--reload-only'],env),
        ('glb_reload',[str(BLENDER/'bin/python'),'-B','-u',str(RELOAD),'--run-dir',str(folder)],
         dict(env,LD_LIBRARY_PATH=str(BLENDER/'lib'),CUDA_VISIBLE_DEVICES=''))]

def validate(job,row):
    folder=LEGACY/job['sample_id'];generated=core.read(folder/'state.json')
    assert generated['pid']==job['generation_pid'], 'Another producer owns this output'
    assert generated['status']=='generated_pending_reload' and generated['sample_id']==job['sample_id']
    assert generated['input_sha256']==row['image_rgba_sha256']
    assert generated['seed']==42 and generated['pipeline_type']=='1024_cascade'
    assert generated['real_input_count']==1 and generated['generated_input_count']==0
    for name,h in generated['output_hashes'].items():
        assert Path(name).name==name and core.sha(folder/name)==h
    assert core.sha(generated['output'])==generated['output_sha256']
    native=core.read(folder/'reload_validation.json');mesh=core.read(folder/'reload_render_validation.json')
    assert native['status']=='passed_pending_process_exit'
    assert mesh['mesh_objects']>0 and mesh['texture_images']>0 and len(mesh['views'])==2
    assert mesh['output_sha256']==generated['output_sha256']
    for path in mesh['views']:
        assert Path(path).resolve().is_relative_to(folder.resolve())
        assert Image.open(path).getchannel('A').getbbox()
    for name,value in [('reload_validation.json',native),('reload_render_validation.json',mesh)]:
        value.update(status='passed',process_exit_code=0);core.save(folder/name,value)
    return dict(output=generated['output'],output_sha256=generated['output_sha256'],
        native_reload=str(folder/'reload_validation.json'),glb_reload=str(folder/'reload_render_validation.json'),
        validations=[dict(path=str(folder/n),sha256=core.sha(folder/n)) for n in ['reload_validation.json','reload_render_validation.json']],
        common_camera_qualified=False)

def main():
    ROOT.mkdir(exist_ok=True);(ROOT/'logs').mkdir(exist_ok=True)
    with (ROOT/'controller.lock').open('a') as control:
        fcntl.flock(control,fcntl.LOCK_EX|fcntl.LOCK_NB);assert not STATE.exists(), 'Audit existing worker state before resuming'
        old=core.read(OLD_STATE);rows=core.read(MANIFEST)['objects'];byid={r['sample_id']:r for r in rows}
        assert len(rows)==len(byid)==432
        signatures={str(MANIFEST):old['manifest_sha256'],str(ADAPTER):old['adapter_sha256'],str(RELOAD):old['glb_reload_sha256'],
                    str(Path(__file__)):core.sha(__file__),str(Path(core.__file__)):core.sha(core.__file__)}
        assert all(core.sha(p)==h for p,h in signatures.items())
        assert old['gates']=='official_and_frozen_native_generation_reload_render_passed'
        assert {j['sample_id'] for j in old['jobs']}==set(byid)
        # This exact immutable adapter provides the native atomic ownership gate.
        text=ADAPTER.read_text()
        assert text.index('O.mkdir(parents=True,exist_ok=False)')<text.index('Trellis2ImageTo3DPipeline.from_pretrained')
        jobs=[dict(sample_id=j['sample_id'],category=j['category'],status='queued') for j in reversed(old['jobs']) if j['status']=='queued']
        for j in jobs:
            row=byid[j['sample_id']]
            assert row['real_input_count']==1 and row['generated_input_count']==0
            assert core.sha(row['image_rgba'])==row['image_rgba_sha256']
        core.save(ROOT/'legacy_snapshot_at_launch.json',old)
        state=dict(pid=os.getpid(),status='running',started_at=time.time(),workers=MAX_WORKERS,gpu_pool=list(range(6)),
            signatures=signatures,manifest_sha256=old['manifest_sha256'],adapter_sha256=old['adapter_sha256'],
            glb_reload_sha256=old['glb_reload_sha256'],disk_reserve_gib=10,jobs=jobs,
            legacy_state=str(OLD_STATE),legacy_controller_untouched=True,failures_retried=False,
            allocation='Reverse unstarted queue; native atomic directory ownership; shared GPU leases',
            effective_state=str(EFFECTIVE),common_camera_qualified=False)
        running={}
        def snapshot():
            state.update(updated_at=time.time(),counts=dict(Counter(j['status'] for j in jobs)));core.save(STATE,state)
            legacy=core.read(OLD_STATE);effective=effective_jobs(legacy['jobs'],jobs)
            core.save(EFFECTIVE,dict(updated_at=time.time(),manifest_sha256=state['manifest_sha256'],membership=432,
                legacy_state=str(OLD_STATE),supplement_state=str(STATE),jobs=effective,
                counts=dict(Counter(j['status'] for j in effective)),
                collision_policy='Legacy FileExists skips are operational records; supplemental completion requires independent native and GLB reload success',
                formal_metrics_qualified=False))
        def launch(job,gpu,lease,stage_index,log):
            assert all(core.sha(p)==h for p,h in signatures.items()), 'Immutable code/input changed'
            stage,cmd,env=commands(job,gpu)[stage_index]
            child=subprocess.Popen(cmd,env=env,stdout=log,stderr=log,start_new_session=True,pass_fds=(lease.fileno(),))
            job.update(status='running',gpu=gpu,stage=stage,child_pid=child.pid,command=cmd)
            if stage_index==0:job.update(generation_pid=child.pid,started_at=time.time())
            running[job['sample_id']]=dict(job=job,gpu=gpu,lease=lease,stage_index=stage_index,log=log,child=child)
            snapshot()
        snapshot()
        while True:
            changed=False
            for sid,task in list(running.items()):
                job=task['job'];child=task['child'];rc=child.poll()
                if rc is None:continue
                job.pop('child_pid',None);job.setdefault('stage_exit_codes',{})[STAGES[task['stage_index']]]=rc
                try:
                    if rc!=0:
                        logtext=(ROOT/'logs'/f'{sid}.log').read_text()[-6000:]
                        legacy_job=next(x for x in core.read(OLD_STATE)['jobs'] if x['sample_id']==sid)
                        if task['stage_index']==0 and 'FileExistsError' in logtext and legacy_job['status']!='queued':
                            produced=core.read(LEGACY/sid/'state.json')
                            assert produced['pid']!=job['generation_pid']
                            job.update(status='delegated_to_legacy',reason='Native atomic output-directory gate: legacy producer won')
                        else:raise RuntimeError(f'{job["stage"]} exited {rc}; see {ROOT/"logs"/(sid+".log")}')
                    elif task['stage_index']<2:
                        if core.shutil.disk_usage(P).free<=10*1024**3:
                            job['waiting_reason']='disk_reserve_10GiB';continue
                        job.pop('waiting_reason',None)
                        launch(job,task['gpu'],task['lease'],task['stage_index']+1,task['log']);continue
                    else:job.update(status='completed',**validate(job,byid[sid]))
                except Exception as error:job.update(status='failed',reason=repr(error))
                job['finished_at']=time.time();task['lease'].close();task['log'].close();running.pop(sid);changed=True
            while len(running)<MAX_WORKERS and any(j['status']=='queued' for j in jobs):
                if core.shutil.disk_usage(P).free<=10*1024**3:break
                gpu=None;lease=None
                for candidate in map(str,range(6)):
                    if candidate in {t['gpu'] for t in running.values()}:continue
                    lease=acquire(candidate)
                    if lease is not None:gpu=candidate;break
                if lease is None:break
                legacy={j['sample_id']:j for j in core.read(OLD_STATE)['jobs']};job=None
                for candidate in jobs:
                    if candidate['status']!='queued':continue
                    sid=candidate['sample_id']
                    if legacy[sid]['status']!='queued':
                        candidate.update(status='delegated_to_legacy',legacy_status=legacy[sid]['status']);changed=True;continue
                    if (LEGACY/sid).exists():
                        candidate.update(status='preexisting_output_requires_audit',reason='Never regenerate an existing output directory');changed=True;continue
                    job=candidate;break
                if job is None:lease.close();break
                log=(ROOT/'logs'/f'{job["sample_id"]}.log').open('x')
                try:launch(job,gpu,lease,0,log)
                except Exception:
                    log.close();lease.close();raise
                changed=True
            snapshot()
            legacy=core.read(OLD_STATE)
            if not running and not any(j['status']=='queued' for j in jobs):
                state['status']='watching_legacy_finish'
                if not any(j['status'] in ('running','queued') for j in legacy['jobs']):break
                if not core.live(legacy['pid']):state['status']='legacy_stopped_requires_audit';break
            time.sleep(5 if running else 15)
        state.update(status='finished' if state['status']!='legacy_stopped_requires_audit' else state['status'],finished_at=time.time());snapshot()

if __name__=='__main__':main()
