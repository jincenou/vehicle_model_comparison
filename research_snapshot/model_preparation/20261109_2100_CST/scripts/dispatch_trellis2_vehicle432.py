"""Wait for verified preparation, validate official and frozen pilots, then batch."""
import asyncio
from collections import Counter
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
from PIL import Image

P=Path(__file__).resolve().parents[1];R=P.parents[1]
O=P/'runs/trellis2_vehicle432_v1';O.mkdir(parents=True,exist_ok=True)
PREFIX=Path(('/opt/conda/envs/ab_cmp_trellis2'.replace('/opt/conda', __import__("os").environ.get('RIGID_CONDA_ROOT', '/opt/conda')).replace('/opt/rigid-benchmark/project', __import__("os").environ.get('RIGID_PROJECT_ROOT', '/opt/rigid-benchmark/project'))))
BLENDER=Path(('/opt/conda/envs/ab_cmp_hunyuan21'.replace('/opt/conda', __import__("os").environ.get('RIGID_CONDA_ROOT', '/opt/conda')).replace('/opt/rigid-benchmark/project', __import__("os").environ.get('RIGID_PROJECT_ROOT', '/opt/rigid-benchmark/project'))))
ADAPTER=P/'scripts/run_trellis2_object.py';RELOAD=P/'scripts/render_hunyuan21_demo_reload.py'
MANIFEST=R/'inputs/vehicle432_image_inputs_v1/manifest.json';SP=O/'batch_state.json'
PILOT='scene-0002_t000013'

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path,data):
    t=path.with_suffix('.tmp');t.write_text(json.dumps(data,indent=2)+'\n');t.replace(path)
def idle_gpus():
    occupied=set(subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader'],text=True).splitlines())
    inventory=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free','--format=csv,noheader,nounits'],text=True)
    reserved=set()
    for path in [R/'state/nuscenes100_expansion_queue.json',R/'runs/ah_batches/nuscenes100_new233_batch_20261110/state.json',
                 P/'runs/hunyuan21_vehicle432_v1/batch_state.json',P/'runs/trellis_original_vehicle432_v1/batch_state.json',
                 P/'runs/recon02_vehicle432_v1/batch_state.json']:
        if path.exists():
            d=json.loads(path.read_text());jobs=d.get('jobs',[]);jobs=jobs.values()if isinstance(jobs,dict)else jobs
            for job in jobs:
                if job.get('status')in ('running','preparing'):
                    g=job.get('gpu',job.get('gpu_index'))
                    if g is not None:reserved.add(str(g))
    return [a.strip()for a,b,c in (line.split(',')for line in inventory.splitlines())
            if b.strip()not in occupied and a.strip()not in reserved and int(c.strip())>45000]

async def main():
    lock=(O/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not SP.exists(),'Existing dispatcher state must be audited before resuming'
    rows=json.loads(MANIFEST.read_text())['objects'];assert len(rows)==432
    signature={'manifest_sha256':sha(MANIFEST),'adapter_sha256':sha(ADAPTER),'glb_reload_sha256':sha(RELOAD)}
    state={'status':'waiting_preparation','pid':os.getpid(),'started_at':time.time(),'phase':'preparation',**signature,
           'method':'TRELLIS.2-4B','retains_original_trellis_as_separate_method':True,'disk_reserve_gib':10,
           'common_camera_qualified':False,'workers':2,
           'official_example':{'sample_id':'official_example','status':'queued'},
           'jobs':[{'sample_id':x['sample_id'],'category':x['category'],'status':'queued'}for x in rows]}
    def snapshot():
        state.update(updated_at=time.time(),counts=dict(Counter(j['status']for j in state['jobs'])));save(SP,state)
    snapshot()
    try:
        while True:
            ready=True;checks={}
            for name,required in [('trellis2_modelscope_download.json','verified_all'),('env_trellis2.json','installed_pending_import_and_gpu_validation')]:
                path=P/'state'/name
                assert path.exists(),f'Missing preparation state {path}'
                d=json.loads(path.read_text());checks[name]={'status':d['status'],'pid':d.get('pid'),'step':d.get('current_step')}
                if d['status']!=required:
                    ready=False
                    if d['status']=='running':
                        try:
                            os.kill(d['pid'],0)
                            proc=Path('/proc')/str(d['pid'])/'stat'
                            assert not proc.read_text().split(') ',1)[1].startswith('Z ')
                        except (OSError,AssertionError):
                            checks[name]['waiting_for_repair']='Preparation process absent or exited'
                    else:
                        checks[name]['waiting_for_repair']=d.get('reason',d['status'])
            state['preparation']=checks
            state['status']='waiting_preparation_repair'if any('waiting_for_repair'in c for c in checks.values())else'waiting_preparation'
            snapshot()
            if ready:break
            await asyncio.sleep(10)
        subprocess.run([('/opt/conda/bin/python'.replace('/opt/conda', __import__("os").environ.get('RIGID_CONDA_ROOT', '/opt/conda')).replace('/opt/rigid-benchmark/project', __import__("os").environ.get('RIGID_PROJECT_ROOT', '/opt/rigid-benchmark/project'))),'-B',str(P/'scripts/prepare_trellis2_offline_cache.py')],check=True)
        for row in rows:assert sha(row['image_rgba'])==row['image_rgba_sha256']
        state.update(status='running',phase='official_example');snapshot()
        claimed=set();guard=asyncio.Lock()
        async def execute(job):
            gpu=None
            while gpu is None:
                async with guard:
                    if shutil.disk_usage(P).free>10*1024**3:
                        choices=[g for g in idle_gpus()if g not in claimed]
                        if choices:gpu=choices[-1];claimed.add(gpu)
                if gpu is None:await asyncio.sleep(5)
            folder=O/job['sample_id'];job.update(status='running',gpu=gpu,run_dir=str(folder),started_at=time.time());snapshot()
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu,CUDA_HOME=str(PREFIX),LD_LIBRARY_PATH=str(PREFIX/'lib'),
                     PATH=str(PREFIX/'bin')+':'+os.environ['PATH'],HF_HOME=str(P/'cache/trellis2_offline'),
                     HF_HUB_CACHE=str(P/'cache/trellis2_offline/hub'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
                     TORCH_HOME=str(P/'cache/trellis2_torch'),TORCH_EXTENSIONS_DIR=str(P/'cache/torch_ext/trellis2'),
                     XDG_CACHE_HOME=str(P/'cache/xdg'),TRITON_CACHE_DIR=str(P/'cache/triton/trellis2'),
                     OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',MAX_JOBS='2',PYTHONDONTWRITEBYTECODE='1',TMPDIR=str(P/'cache/tmp'))
            for key in ['TORCH_HOME','TORCH_EXTENSIONS_DIR','TRITON_CACHE_DIR']:Path(env[key]).mkdir(parents=True,exist_ok=True)
            base=[str(PREFIX/'bin/python'),'-B','-u',str(ADAPTER),'--sample-id',job['sample_id'],'--run-dir',str(folder)]
            try:
                with (P/'logs'/('trellis2_'+job['sample_id']+'.log')).open('a')as log:
                    for stage,cmd in [('generate',base),('native_reload',base+['--reload-only']),
                                      ('glb_reload',[str(BLENDER/'bin/python'),'-B','-u',str(RELOAD),'--run-dir',str(folder)])]:
                        job.update(stage=stage);snapshot();stage_env=env
                        if stage=='glb_reload':stage_env=dict(env,LD_LIBRARY_PATH=str(BLENDER/'lib'),CUDA_VISIBLE_DEVICES='')
                        child=await asyncio.create_subprocess_exec(*cmd,env=stage_env,stdout=log,stderr=log)
                        job['child_pid']=child.pid;snapshot();rc=await child.wait();assert rc==0,f'{stage} exited {rc}'
                native=json.loads((folder/'reload_validation.json').read_text());assert native['status']=='passed_pending_process_exit'
                mesh=json.loads((folder/'reload_render_validation.json').read_text())
                assert mesh['mesh_objects']>0 and mesh['texture_images']>0
                assert len(mesh['views'])==2
                for path in mesh['views']:assert Image.open(path).getchannel('A').getbbox()
                for name,v in [('reload_validation.json',native),('reload_render_validation.json',mesh)]:
                    v.update(status='passed',process_exit_code=0);save(folder/name,v)
                generated=json.loads((folder/'state.json').read_text())
                assert sha(generated['output'])==generated['output_sha256']
                job.update(status='completed',output=generated['output'],output_sha256=generated['output_sha256'],
                           native_reload=str(folder/'reload_validation.json'),glb_reload=str(folder/'reload_render_validation.json'))
            except Exception as error:job.update(status='failed',reason=repr(error))
            finally:
                claimed.remove(gpu);job.pop('child_pid',None);job['finished_at']=time.time();snapshot()
        await execute(state['official_example'])
        assert state['official_example']['status']=='completed','Official example gate failed; batch not started'
        state['phase']='frozen_pilot';snapshot();pilot=next(j for j in state['jobs']if j['sample_id']==PILOT)
        await execute(pilot);assert pilot['status']=='completed','Frozen pilot gate failed; batch not started'
        state.update(phase='batch',gates='official_and_frozen_native_generation_reload_render_passed');snapshot()
        queue=asyncio.Queue()
        for job in state['jobs']:
            if job['status']=='queued':queue.put_nowait(job)
        async def worker():
            while not queue.empty():
                try:job=queue.get_nowait()
                except asyncio.QueueEmpty:return
                await execute(job);queue.task_done()
        await asyncio.gather(worker(),worker())
        state['status']='completed'if all(j['status']=='completed'for j in state['jobs'])else'finished_with_failures'
    except Exception as error:state.update(status='stopped_on_failed_gate',reason=repr(error))
    finally:state['finished_at']=time.time();snapshot()

if __name__=='__main__':asyncio.run(main())
