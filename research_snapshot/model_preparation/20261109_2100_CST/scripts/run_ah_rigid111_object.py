# Isolated rigid111 adapter; source SHA256: 759423a3ca6c8c4ba4f8c725f8c7349dbe9e45123f0b5e2d9ba102b65c380886
"""Run the pinned AH image-directory entry with isolated inputs and explicit failures."""
import os,sys,json,time,random,runpy,hashlib,traceback
from pathlib import Path
R=Path(('/opt/rigid-benchmark/workspace'.replace('/opt/rigid-benchmark/workspace', __import__("os").environ.get('RIGID_WORKSPACE', '/opt/rigid-benchmark/workspace')).replace('/opt/rigid-benchmark/project', __import__("os").environ.get('RIGID_PROJECT_ROOT', '/opt/rigid-benchmark/project'))));B=Path(('/opt/rigid-benchmark/assetbench'.replace('/opt/rigid-benchmark/assetbench', __import__("os").environ.get('RIGID_ASSETBENCH', '/opt/rigid-benchmark/assetbench')).replace('/opt/rigid-benchmark/project', __import__("os").environ.get('RIGID_PROJECT_ROOT', '/opt/rigid-benchmark/project'))));repo=B/'repos/ah';job=R/'model_preparation/20261109_2100_CST/runs/ah_rigid111_v1'/os.environ['AB_JOB_ID'];job.mkdir(parents=True,exist_ok=False)
sys.path.insert(0,str(repo));os.chdir(repo)
import numpy as np
import torch
def reject_unlisted_download(*args, **kwargs):
 raise RuntimeError("Unlisted runtime download requested; prepare and record the exact dependency before execution")
torch.hub.download_url_to_file = reject_unlisted_download
random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
input_root=Path(os.environ['AB_INPUT_ROOT']).resolve()
assert input_root.is_relative_to(R/'inputs/ah')
source=json.loads((input_root/'source_manifest.json').read_text())
assert Path(sys.prefix).resolve()==Path(('/opt/conda/envs/ab_ah'.replace('/opt/conda', __import__("os").environ.get('RIGID_CONDA_ROOT', '/opt/conda')).replace('/opt/rigid-benchmark/project', __import__("os").environ.get('RIGID_PROJECT_ROOT', '/opt/rigid-benchmark/project')))).resolve()
assert torch.cuda.device_count()==1
assert hashlib.sha256(Path(source['image']).read_bytes()).hexdigest()==source['source_image_sha256']
assert hashlib.sha256(Path(source['mask']).read_bytes()).hexdigest()==source['source_mask_sha256']
weights=json.loads((B/'state/downloads.json').read_text());assert weights['status']=='ready'
# Fail immediately if any Python file open tries to consume real held-out observations.
def access_guard(event,args):
 if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)):
  p=os.fsdecode(args[0]);resolved=str(Path(p).resolve())
  if '/eval_refs/' in resolved or '/reserved_views/' in resolved:
   raise PermissionError('Heldout access forbidden in generation: '+resolved)
sys.addaudithook(access_guard)
args=['run_inference.py','--diffusion_checkpoint',str(B/'weights/ah/core/AH_multiview_diffusion.safetensors'),'--ahc_checkpoint',str(B/'weights/ah/core/AH_camera_estimator.safetensors'),'--lifting_checkpoint',str(B/'weights/ah/core/AH_tokengs_lifting.safetensors'),'--image_dir',str(input_root),'--output_dir',str(job/'outputs'),'--num_steps','30','--cfg_scale','2.0','--precision','bf16']
record={'status':'running','phase':os.environ['AB_PHASE'],'sample_id':source['sample_id'],'method':'AH_R1_AHC','seed':42,'real_input_ids':[source['source_image_sha256']] if source.get('real_input_count',1) else [],'synthetic_input_ids':[source['source_image_sha256']] if not source.get('real_input_count',1) else [],'data_nature':source.get('data_nature','real_photo'),'input_root':str(input_root),'env_prefix':sys.prefix,'repo_commit':'4e87f859f373cd468c24aca78ebfdacb45d99f33','patch_sha256':hashlib.sha256((B/'patches/ah_isolation_and_no_oom_switch.patch').read_bytes()).hexdigest(),'argv':args,'pid':os.getpid(),'cuda_visible_devices':os.environ['CUDA_VISIBLE_DEVICES'],'gpu_name':torch.cuda.get_device_name(0),'weights':weights['files'],'start_time':time.time(),'precision':'bf16','num_steps':30,'cfg_scale':2.0,'offload':False,'peak_vram_bytes':None,'exit_code':None}
def save():
 p=job/'job.json.tmp';p.write_text(json.dumps(record,indent=2,allow_nan=False));p.replace(job/'job.json')
save();torch.cuda.reset_peak_memory_stats()
try:
 sys.argv=args;runpy.run_path(str(repo/'run_inference.py'),run_name='__main__')
 ply=job/'outputs'/source['sample_id']/'gaussians.ply';assert ply.is_file() and ply.stat().st_size>1000,'No valid asset exported'
 record.update(status='inference_completed_pending_reload',output_ply=str(ply),exit_code=0)
except BaseException as e:
 record.update(status='failed',reason=repr(e),traceback=traceback.format_exc(),exit_code=1)
 raise
finally:
 record.update(elapsed_seconds=time.time()-record['start_time'],peak_vram_bytes=torch.cuda.max_memory_allocated());save()
