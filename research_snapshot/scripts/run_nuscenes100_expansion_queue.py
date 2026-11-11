"""84 native jobs after the existing StreetGS training AND reload queues.

One scene bundle per GPU, eight independent workers. Every method has a short
native pilot and reload gate. Failures are explicit; no output/config fallback.
"""
import argparse
import asyncio
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

from run_omnire_legacy_no_smpl_queue import environment as ds_env, PYTHON as DS_PYTHON, REPO, SKY_PYTHON, SKY_SCRIPT, SKY_MODEL
from run_splatad_human_train_only_queue import environment as splat_env, PYTHON as SPLAT_PYTHON

R=Path(__file__).resolve().parents[1]
B=R/'inputs/benchmark_nuscenes100_v1'
C=R/'configs/nuscenes100_expansion'
STATE=R/'state/nuscenes100_expansion_queue.json'
METHODS=('splatad','omnire','streetgs')
S={}
def read(p): return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()
def save():
    S['updated_at']=time.time();temp=STATE.with_suffix('.tmp');temp.write_text(json.dumps(S,indent=2)+'\n');temp.replace(STATE)
def job(scene,method):return S['jobs'][method+':'+scene]
def update(scene,method,**kw):job(scene,method).update(kw);save()
def validate_manifest():
    manifest=read(C/'manifest.json')
    assert sha(B/'selection.json')==manifest['selection_sha256']==read(B/'freeze.json')['selection_sha256']
    for p,digest in manifest['source_hashes'].items():assert sha(p)==digest, f'Changed source/config: {p}'
    rows=[x for x in read(B/'selection.json')['scenes'] if x['origin']=='new28']
    assert len(rows)==28 and len(rows)*len(METHODS)==84
    for row in rows:
        assert sha(row['split'])==row['split_sha256']
        split=read(row['split']);assert not split['heldout_sample_tokens']
        assert all(f['split']=='train' for f in split['files'])
    return manifest,rows

async def slot(gpu):
    p=await asyncio.create_subprocess_exec('nvidia-smi','-i',str(gpu),'--query-gpu=memory.free','--format=csv,noheader,nounits',stdout=asyncio.subprocess.PIPE,stderr=asyncio.subprocess.PIPE)
    out,_=await p.communicate()
    if p.returncode:return False
    return int(out.strip())>=S['minimum_free_gpu_mib']
async def wait_slot(gpu):
    while True:
        if shutil.disk_usage(R).free<30*1024**3:S['workers'][str(gpu)]={'stage':'waiting_disk_30GiB_reserve'}
        elif await slot(gpu):return
        else:S['workers'][str(gpu)]={'stage':'waiting_free_gpu_memory'}
        save();await asyncio.sleep(60)

async def command(scene,method,stage,gpu,args,env,cwd):
    log=R/'logs'/f'nuscenes100_{method}_{stage}_{scene}.log'
    update(scene,method,stage=stage,status='running',gpu=gpu,command=list(map(str,args)),log=str(log),started_at=time.time())
    with log.open('x') as stream:
        process=await asyncio.create_subprocess_exec(*map(str,args),cwd=cwd,env=env,stdout=stream,stderr=asyncio.subprocess.STDOUT)
        update(scene,method,pid=process.pid)
        code=await process.wait()
    update(scene,method,exit_code=code,process_finished_at=time.time())
    if code:raise RuntimeError(f'{stage} exit={code}; {log}')

async def prepare(row,gpu):
    scene=row['scene'];base=R/'inputs/drivestudio_nuscenes100_expansion'/scene
    audit=base/'input_audit.json';processed=base/'processed/trainval/000'
    if not audit.exists():
        await command(scene,'omnire','prepare',gpu,[DS_PYTHON,'-B','-u',R/'scripts/prepare_nuscenes100_drivestudio.py','--scene',scene],ds_env(gpu),R)
    data=read(audit)
    assert data['status']=='native_preprocess_completed' and data['source_split_sha256']==row['split_sha256']
    assert len(list((processed/'images').glob('*.jpg')))==row['train_samples']*6==data['processed_images']
    count=len(list((processed/'sky_masks').glob('*.png')))
    if count!=data['processed_images']:
        assert count==0,'Incomplete prior sky masks require inspection'
        await command(scene,'omnire','sky_masks',gpu,[SKY_PYTHON,'-B','-u',SKY_SCRIPT,'--scene_dir',processed,'--model',SKY_MODEL,'--device','cuda:0'],ds_env(gpu),REPO)
    assert len(list((processed/'sky_masks').glob('*.png')))==data['processed_images']
    return processed

async def train(row,method,gpu,stage):
    scene=row['scene'];validate_manifest()
    if method=='splatad':
        await command(scene,method,stage+'_train',gpu,[SPLAT_PYTHON,'-B','-u',R/'scripts/train_splatad_nuscenes100_expansion.py','--scope','reconstruction_only','--scene',scene,'--stage',stage],splat_env(gpu),R)
        report=R/'reports'/f'splatad_nuscenes100_expansion_{stage}_{scene}_validation.json'
        result=read(report)
        assert result['status']=='passed' and result['step']==(100 if stage=='pilot' else 30000)
        assert result['split_sha256']==row['split_sha256'] and result['real_heldout_eligible'] is False
        update(scene,method,status=stage+'_validated',stage=stage+'_validated',validation=str(report),checkpoint=result['checkpoint'],checkpoint_sha256=sha(result['checkpoint']),finished_at=time.time())
        return
    processed=await prepare(row,gpu)
    config=C/(method+'.yaml');out=R/'runs'/f'{method}_nuscenes100_expansion'/stage;run=out/'nuScenes'/scene
    assert not run.exists(),f'Refuse overwrite: {run}'
    args=[DS_PYTHON,'-B','-u','tools/train.py','--config_file',config,'--output_root',out,'--project','nuScenes','--run_name',scene]
    if method=='streetgs':args+=['dataset=nuscenes/6cams']
    args+=['data.data_root='+str(processed.parent),'data.scene_idx=0','data.pixel_source.load_smpl=false','data.pixel_source.test_image_stride=0']
    if stage=='pilot':args+=['trainer.optim.num_iters=100']
    await command(scene,method,stage+'_train',gpu,args,ds_env(gpu),REPO)
    ckpt=run/'checkpoint_final.pth';assert ckpt.is_file() and ckpt.stat().st_size>0
    digest=sha(ckpt);postfix='expansion_'+stage+'_reload'
    await command(scene,method,stage+'_reload',gpu,[DS_PYTHON,'-B','-u','tools/eval.py','--resume_from',ckpt,'--render_video_postfix',postfix],ds_env(gpu),REPO)
    assert sha(ckpt)==digest
    step=100 if stage=='pilot' else 30000
    suffix='RigidNodes_rgbs' if method=='omnire' else 'rgbs'
    video=run/'videos_eval'/f'full_set_{step}_{postfix}_{suffix}.mp4'
    assert video.is_file() and video.stat().st_size>0,f'Missing native reload render: {video}'
    result={'status':'passed','scene':scene,'method':method,'stage':stage,'step':step,'checkpoint':str(ckpt),'checkpoint_sha256':digest,
            'split_sha256':row['split_sha256'],'config_sha256':sha(run/'config.yaml'),'native_render':str(video),'real_heldout_eligible':False,
            'scope':'Scene reload and native render; object extraction and fixed-camera metric validation are separate.'}
    report=R/'reports'/f'{method}_nuscenes100_expansion_{stage}_{scene}_validation.json';report.write_text(json.dumps(result,indent=2)+'\n')
    update(scene,method,status=stage+'_validated',stage=stage+'_validated',validation=str(report),checkpoint=str(ckpt),checkpoint_sha256=digest,finished_at=time.time())

async def worker(gpu,q):
    while not q.empty():
        await wait_slot(gpu)
        try:row=q.get_nowait()
        except asyncio.QueueEmpty:break
        scene=row['scene'];S['workers'][str(gpu)]={'stage':'scene_bundle','scene':scene};save()
        for method in METHODS:
            if S['pilot_gates'].get(method)!='passed':
                update(scene,method,status='blocked_pilot',stage='blocked_pilot');continue
            await wait_slot(gpu)
            try:await train(row,method,gpu,'formal')
            except Exception as exc:update(scene,method,status='failed',stage='failed',reason=repr(exc),finished_at=time.time())
        q.task_done()
    S['workers'][str(gpu)]={'stage':'finished'};save()

async def main():
    manifest,rows=validate_manifest()
    assert not STATE.exists(),'Queue already has state; inspect before restart'
    S.update(manifest,stage='waiting_existing_streetgs_training_and_reload',pid=os.getpid(),started_at=time.time(),
        workers={str(g):{'stage':'waiting_existing_streetgs_training_and_reload'} for g in range(8)},pilot_gates={},
        jobs={m+':'+x['scene']:{'method':m,'scene':x['scene'],'status':'queued','stage':'waiting_dependency','split_sha256':x['split_sha256'],'real_heldout_eligible':False} for x in rows for m in METHODS})
    save()
    while True:
        prior=read(R/'state/streetgs_drivestudio_nusc_v1.json');reload=read(R/'state/streetgs_drivestudio_reload_v1.json')
        S['dependency_observation']={'training_stage':prior.get('queue_stage'),'reload_stage':reload.get('stage'),'checked_at':time.time()};save()
        if prior.get('queue_stage')=='finished' and reload.get('stage')=='finished':break
        await asyncio.sleep(60)
    # Acquire only AFTER the reload monitor is terminal, so we cannot take its slot.
    with (R/'state/streetgs_drivestudio_nusc_v1.lock').open('a') as prior_lock:
        while True:
            try:fcntl.flock(prior_lock,fcntl.LOCK_EX|fcntl.LOCK_NB);break
            except BlockingIOError:await asyncio.sleep(60)
        S['stage']='pilot_gates';save()
        # Choose the first available GPU; never terminate a foreign process.
        while True:
            choices=[g for g in range(8) if await slot(g)]
            if choices and shutil.disk_usage(R).free>=30*1024**3:break
            await asyncio.sleep(60)
        gpu=choices[0];pilot=next(x for x in rows if x['scene']=='scene-0030')
        for method in METHODS:
            await wait_slot(gpu)
            try:
                await train(pilot,method,gpu,'pilot');S['pilot_gates'][method]='passed'
            except Exception as exc:
                S['pilot_gates'][method]='failed';update(pilot['scene'],method,status='pilot_failed',reason=repr(exc))
            save()
        S['stage']='formal_batch_running';save();q=asyncio.Queue()
        for row in rows:q.put_nowait(row)
        await asyncio.gather(*(worker(g,q) for g in range(8)))
        S['stage']='finished';S['formal_validated']=sum(x['status']=='formal_validated' for x in S['jobs'].values());save()

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--validate-only',action='store_true');args=ap.parse_args()
    if args.validate_only:
        m,rows=validate_manifest();print(json.dumps({'status':'manifest_valid','new_scenes':len(rows),'formal_jobs':len(rows)*3,'gpus':m['gpus'],'dependency':m['dependency']}))
    else:
        with (R/'state/nuscenes100_expansion_queue.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);asyncio.run(main())
