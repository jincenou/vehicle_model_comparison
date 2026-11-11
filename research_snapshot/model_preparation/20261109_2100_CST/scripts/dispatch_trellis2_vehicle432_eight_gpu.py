"""Six shared-lease supplement slots plus two untouched legacy slots.

Parent-only handoff preserves independently sessioned native children. Adopted
generation requires completed hashes followed by fresh native/GLB reloads;
an unobservable exit is never reported as an observed zero exit code.
"""
from collections import Counter
import fcntl
import os
from pathlib import Path
import signal
import subprocess
import time
import dispatch_trellis2_vehicle432_supplement as prior
from dispatch_rigid68_trellis2_parallel import identity, assert_identity

core=prior.core;ROOT=prior.ROOT;STATE=prior.STATE;R=prior.R;P=prior.P
RESERVATIONS=[R/'state'/f'trellis2_vehicle_priority_gpu{g}.json' for g in range(6)]

def ready(gpu):
    _,details=core.available(gpu)
    ignored={str(STATE)}|{str(p) for p in RESERVATIONS}
    reservations=[x for x in details['reservations'] if x['source'] not in ignored]
    occupied=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader'],text=True)
    return not reservations and details['uuid'] not in {s.strip() for s in occupied.splitlines()} and details['free_mib']>45000

def acquire(gpu):
    if not ready(gpu) or core.shutil.disk_usage(P).free<=10*1024**3:return None
    handle=(P/'runs/rigid111_gpu_locks'/f'gpu_{gpu}.lock').open('a')
    old_handler=signal.getsignal(signal.SIGALRM)
    def expired(*args):raise TimeoutError()
    signal.signal(signal.SIGALRM,expired);accepted=False
    try:
        signal.setitimer(signal.ITIMER_REAL,6);fcntl.flock(handle,fcntl.LOCK_EX)
        signal.setitimer(signal.ITIMER_REAL,0)
        accepted=ready(gpu) and core.shutil.disk_usage(P).free>10*1024**3
    except TimeoutError:pass
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,old_handler)
        if not accepted:handle.close()
    return handle if accepted else None

def adoption_finished(job, saved):
    current=identity(saved['pid'])
    if current:
        assert_identity(current,saved)
        if current['state']!='Z':return False
    folder=prior.LEGACY/job['sample_id'];g=core.read(folder/'state.json')
    assert g['pid']==job['generation_pid'] and g['sample_id']==job['sample_id']
    assert g['status']=='generated_pending_reload', 'Adopted generation did not produce complete native output'
    for name,h in g['output_hashes'].items():
        assert Path(name).name==name and core.sha(folder/name)==h
    assert core.sha(g['output'])==g['output_sha256']
    return True

def handoff():
    before=core.read(STATE);parent=identity(before['pid'])
    assert parent and str(Path(prior.__file__).resolve()) in parent['command']
    assert all(core.sha(p)==h for p,h in before['signatures'].items())
    for j in before['jobs']:
        if j['status']=='running':
            assert j['stage']=='generate','Handoff must occur at generation stage; retry inspection later'
            child=identity(j['child_pid']);assert child and child['command']==j['command']
            assert child['session_id']==child['pid']!=parent['session_id']
        elif j['status']=='completed':
            assert core.sha(j['output'])==j['output_sha256']
    archive=ROOT/f'eight_gpu_handoff_{time.time_ns()}';archive.mkdir()
    core.save(archive/'before.json',before)
    assert_identity(identity(parent['pid']),parent)
    # All four child sessions remain untouched, including inherited GPU leases.
    os.kill(parent['pid'],signal.SIGTERM)
    for _ in range(100):
        if not core.live(parent['pid']):break
        time.sleep(.1)
    assert not core.live(parent['pid'])
    state=core.read(STATE);adopted={}
    for job in state['jobs']:
        if job['status']!='running':continue
        # Cover the Popen-to-snapshot interval by matching exact native argv.
        matches=[]
        for proc in Path('/proc').iterdir():
            if not proc.name.isdigit():continue
            try:item=identity(int(proc.name))
            except (PermissionError,ProcessLookupError):continue
            if item and item['state']!='Z' and item['command']==job['command']:matches.append(item)
        assert len(matches)<=1
        saved=matches[0] if matches else identity(job['child_pid'])
        if matches:
            assert saved['session_id']==saved['pid']!=parent['session_id']
            job['child_pid']=saved['pid']
        # Re-run reload stages after any handoff; never regenerate the asset.
        if saved is None:saved=dict(pid=job['child_pid'],start_ticks=-1,command=job['command'],state='Z')
        adopted[job['sample_id']]=saved
    core.save(archive/'after.json',state)
    core.save(archive/'audit.json',dict(parent=parent,adopted=adopted,no_child_signaled=True,
        native_parameters_changed=False,new_supplement_workers=6,legacy_workers=2))
    return state,adopted,archive

def run(state,adopted,archive):
    rows={r['sample_id']:r for r in core.read(prior.MANIFEST)['objects']}
    signatures=dict(state['signatures']);signatures[str(Path(__file__))]=core.sha(__file__)
    state.update(pid=os.getpid(),status='running',workers=6,total_vehicle_capacity=8,signatures=signatures,
        eight_gpu_handoff=str(archive),updated_at=time.time(),gpu_pool=list(range(6)))
    jobs=state['jobs'];running={}
    for j in jobs:
        if j['status']=='running':
            running[j['sample_id']]=dict(job=j,gpu=j['gpu'],adopted=adopted[j['sample_id']],lease=None,
                stage_index=-1,child=None,log=(ROOT/'logs'/f"{j['sample_id']}.log").open('a'))
            j['adopted_generation_exit_observed']=False
    def snapshot():
        state.update(updated_at=time.time(),counts=dict(Counter(j['status'] for j in jobs)));core.save(STATE,state)
        legacy=core.read(prior.OLD_STATE);effective=prior.effective_jobs(legacy['jobs'],jobs)
        core.save(prior.EFFECTIVE,dict(updated_at=time.time(),membership=432,manifest_sha256=state['manifest_sha256'],
            legacy_state=str(prior.OLD_STATE),supplement_state=str(STATE),jobs=effective,
            counts=dict(Counter(j['status'] for j in effective)),total_vehicle_capacity=8,formal_metrics_qualified=False))
        for gpu,path in enumerate(RESERVATIONS):
            core.save(path,dict(pid=os.getpid(),gpu=str(gpu),status='priority_reserved_for_vehicle_trellis2',
                scope='Existing human jobs finish; further claims wait until vehicle queue drains',updated_at=time.time()))
    def launch(task,index):
        assert all(core.sha(p)==h for p,h in signatures.items())
        job=task['job'];stage,cmd,env=prior.commands(job,task['gpu'])[index]
        child=subprocess.Popen(cmd,env=env,stdout=task['log'],stderr=task['log'],start_new_session=True,pass_fds=(task['lease'].fileno(),))
        task.update(child=child,stage_index=index)
        job.update(status='running',gpu=task['gpu'],stage=stage,child_pid=child.pid,command=cmd)
        if index==0:job.update(generation_pid=child.pid,started_at=time.time())
        snapshot()
    snapshot()
    while True:
        for sid,task in list(running.items()):
            job=task['job']
            try:
                if task.get('adopted'):
                    if not task.get('adoption_verified'):
                        if not adoption_finished(job,task['adopted']):continue
                        task['adoption_verified']=True;job.pop('child_pid',None)
                        job['adoption_evidence']='Verified completed generation hashes; fresh native and GLB reload required'
                        snapshot()
                    lease=acquire(task['gpu'])
                    if lease is None:continue
                    task['lease']=lease;task.pop('adopted');launch(task,1);continue
                rc=task['child'].poll()
                if rc is None:continue
                job.pop('child_pid',None);job.setdefault('stage_exit_codes',{})[prior.STAGES[task['stage_index']]]=rc
                if rc!=0:
                    text=(ROOT/'logs'/f'{sid}.log').read_text()[-6000:]
                    legacy_job=next(x for x in core.read(prior.OLD_STATE)['jobs'] if x['sample_id']==sid)
                    if task['stage_index']==0 and 'FileExistsError' in text and legacy_job['status']!='queued':
                        generated=core.read(prior.LEGACY/sid/'state.json');assert generated['pid']!=job['generation_pid']
                        job.update(status='delegated_to_legacy',reason='Native directory ownership held by legacy producer')
                    else:raise RuntimeError(f'{job["stage"]} exited {rc}')
                elif task['stage_index']<2:
                    if core.shutil.disk_usage(P).free<=10*1024**3:
                        job['waiting_reason']='disk_reserve_10GiB';continue
                    job.pop('waiting_reason',None);launch(task,task['stage_index']+1);continue
                else:job.update(status='completed',**prior.validate(job,rows[sid]))
            except Exception as error:job.update(status='failed',reason=repr(error))
            job.pop('child_pid',None);job['finished_at']=time.time()
            if task['lease']:task['lease'].close()
            task['log'].close();running.pop(sid)
        while len(running)<6 and any(j['status']=='queued' for j in jobs):
            lease=None;gpu=None
            for candidate in map(str,range(6)):
                if candidate in {t['gpu'] for t in running.values()}:continue
                lease=acquire(candidate)
                if lease is not None:gpu=candidate;break
            if lease is None:break
            legacy={j['sample_id']:j for j in core.read(prior.OLD_STATE)['jobs']};job=None
            for j in jobs:
                if j['status']!='queued':continue
                sid=j['sample_id']
                if legacy[sid]['status']!='queued':j.update(status='delegated_to_legacy');continue
                if (prior.LEGACY/sid).exists():j.update(status='preexisting_output_requires_audit');continue
                job=j;break
            if job is None:lease.close();break
            task=dict(job=job,gpu=gpu,lease=lease,log=(ROOT/'logs'/f"{job['sample_id']}.log").open('x'))
            running[job['sample_id']]=task;launch(task,0)
        snapshot()
        if not running and not any(j['status']=='queued' for j in jobs):
            for path in RESERVATIONS:
                core.save(path,dict(status='released',pid=None,gpu=None))
            state.update(status='finished_supplement',finished_at=time.time());core.save(STATE,state);break
        time.sleep(5)

def main():
    with (ROOT/'eight_gpu_handoff.lock').open('a') as transfer:
        fcntl.flock(transfer,fcntl.LOCK_EX|fcntl.LOCK_NB)
        state,adopted,archive=handoff()
        with (ROOT/'controller.lock').open('a') as control:
            fcntl.flock(control,fcntl.LOCK_EX|fcntl.LOCK_NB)
            run(state,adopted,archive)

if __name__=='__main__':main()
