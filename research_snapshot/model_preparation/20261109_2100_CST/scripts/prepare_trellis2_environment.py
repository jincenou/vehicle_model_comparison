"""Owned TRELLIS.2 environment following the pinned upstream CUDA setup."""
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

P=Path(__file__).resolve().parents[1]
S=P/'sources/trellis2-75fbf0183001ed9876c8dbb35de6b68552ee08bd'
PREFIX=Path(('/opt/conda/envs/ab_cmp_trellis2'.replace('/opt/conda', __import__("os").environ.get('RIGID_CONDA_ROOT', '/opt/conda')).replace('/opt/rigid-benchmark/project', __import__("os").environ.get('RIGID_PROJECT_ROOT', '/opt/rigid-benchmark/project'))))
SP=P/'state/env_trellis2.json'
lock=(P/'state/env_trellis2.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert not PREFIX.exists() or SP.exists(),'Refusing to modify unowned existing environment'
state=json.loads(SP.read_text())if SP.exists()else{'steps':{},'source':str(S),'prefix':str(PREFIX)}
state.update(status='running',pid=os.getpid(),runtime_validated=False,disk_reserve_gib=10)
state.pop('reason',None)
env={k:v for k,v in os.environ.items()if 'proxy'not in k.lower() and not k.startswith('GIT_CONFIG')}
env.update(CONDA_PKGS_DIRS=str(P/'cache/conda_hunyuan21'),PIP_CACHE_DIR=str(P/'cache/pip'),
           TMPDIR=str(P/'cache/tmp'),CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',
           MAX_JOBS='2',CMAKE_BUILD_PARALLEL_LEVEL='2',TORCH_CUDA_ARCH_LIST='8.9',
           CUDA_HOME=str(PREFIX),PATH=str(PREFIX/'bin')+':'+env['PATH'],LD_LIBRARY_PATH=str(PREFIX/'lib'),
           PIP_DEFAULT_TIMEOUT='120',PIP_RETRIES='5',PIP_RESUME_RETRIES='8',
           CFLAGS='-I'+str(PREFIX/'include'),LDFLAGS='-L'+str(PREFIX/'lib'),
           GIT_CONFIG_GLOBAL='/dev/null',GIT_CONFIG_NOSYSTEM='1',
           GIT_CONFIG_COUNT='2',GIT_CONFIG_KEY_0='http.proxy',GIT_CONFIG_VALUE_0='',
           GIT_CONFIG_KEY_1='https.proxy',GIT_CONFIG_VALUE_1='')
def save():
    state['updated_at']=datetime.datetime.now(datetime.timezone.utc).isoformat()
    tmp=SP.with_suffix('.tmp');tmp.write_text(json.dumps(state,indent=2)+'\n');tmp.replace(SP)
def step(name,cmd,cwd=S):
    if state['steps'].get(name,{}).get('returncode')==0:return
    assert shutil.disk_usage(P).free>10*1024**3,'DiskReserve10GiB'
    state.update(current_step=name);state['steps'][name]={'command':cmd,'cwd':str(cwd),'status':'running'};save()
    print(name,cmd,flush=True);r=subprocess.run(cmd,cwd=cwd,env=env)
    state['steps'][name].update(returncode=r.returncode,status='passed'if r.returncode==0 else'failed');save()
    assert r.returncode==0,'StepFailed:'+name
def source(name,url,branch=None):
    path=P/'sources'/name
    # Host Git 2.x may predate GIT_CONFIG_COUNT/GIT_CONFIG_GLOBAL support.
    # Explicit -c options override the stale user proxy and propagate to submodules.
    cmd=['git','-c','http.proxy=','-c','https.proxy=','clone','--recursive']+(['--branch',branch]if branch else[])+[url,str(path)]
    step('clone_'+name,cmd)
    commit=subprocess.check_output(['git','-C',str(path),'rev-parse','HEAD'],text=True).strip()
    state.setdefault('extension_sources',{})[name]={'url':url,'commit':commit};save()
    return path

save()
try:
    conda=('/opt/conda/bin/conda'.replace('/opt/conda', __import__("os").environ.get('RIGID_CONDA_ROOT', '/opt/conda')).replace('/opt/rigid-benchmark/project', __import__("os").environ.get('RIGID_PROJECT_ROOT', '/opt/rigid-benchmark/project')))
    step('create_environment',[conda,'create','-y','-p',str(PREFIX),'--override-channels','-c','conda-forge','python=3.10','pip'])
    step('cuda_and_image_build_headers',[conda,'install','-y','-p',str(PREFIX),'--override-channels','-c','nvidia','-c','conda-forge','cuda-nvcc=12.4','cuda-cudart-dev=12.4','cuda-cccl=12.4','libjpeg-turbo','zlib'])
    python=str(PREFIX/'bin/python');constraint=P/'state/trellis2_torch_constraints.txt'
    constraint.write_text('torch==2.6.0\ntorchvision==0.21.0\nnumpy==1.26.4\ntransformers==4.57.3\nopencv-python-headless==4.11.0.86\n')
    pip=[python,'-m','pip','install','--constraint',str(constraint),'--disable-pip-version-check']
    step('pytorch',pip+['torch==2.6.0','torchvision==0.21.0','--index-url','https://download.pytorch.org/whl/cu124'])
    step('basic',pip+['numpy==1.26.4','setuptools<81','wheel','packaging','ninja==1.11.1.4','pybind11','imageio','imageio-ffmpeg','tqdm','easydict','opencv-python-headless','trimesh','transformers==4.57.3','gradio==6.0.1','tensorboard','pandas','lpips','zstandard','kornia','timm','safetensors'])
    step('pillow_simd',pip+['--no-build-isolation','pillow-simd'])
    transport=json.loads((P/'state/recon02_utils3d_transport.json').read_text())
    assert transport['commit']=='9a4eb15e4021b67b12c460c7057d642626897ec8'
    archive=Path(transport['local_archive'])
    assert hashlib.sha256(archive.read_bytes()).hexdigest()==transport['sha256']
    state['utils3d_transport']=transport;save()
    step('utils3d',pip+[str(archive)])
    step('flash_attn',pip+['--no-build-isolation','flash-attn==2.7.3'])
    nd_record=json.loads((P/'state/trellis2_nvdiffrast_archive.json').read_text())
    assert nd_record['tag']=='v0.4.0'
    assert hashlib.sha256(Path(nd_record['local_archive']).read_bytes()).hexdigest()==nd_record['sha256']
    nd=P/'sources'/('nvdiffrast-'+nd_record['commit'])
    assert (nd/'setup.py').is_file() and not (nd/'.gitmodules').exists()
    state.setdefault('extension_sources',{})['trellis2_nvdiffrast']=nd_record;save()
    step('nvdiffrast',pip+['--no-build-isolation',str(nd)])
    nr=source('trellis2_nvdiffrec','https://github.com/JeffreyXiang/nvdiffrec.git','renderutils')
    step('nvdiffrec',pip+['--no-build-isolation',str(nr)])
    cm=source('trellis2_CuMesh','https://github.com/JeffreyXiang/CuMesh.git')
    step('cumesh',pip+['--no-build-isolation',str(cm)])
    fg_record=json.loads((P/'state/trellis2_flexgemm_archive.json').read_text())
    assert hashlib.sha256(Path(fg_record['archive']).read_bytes()).hexdigest()==fg_record['sha256']
    fg=Path(fg_record['source']);assert (fg/'setup.py').is_file()and not (fg/'.gitmodules').exists()
    state.setdefault('extension_sources',{})['trellis2_FlexGEMM']=fg_record;save()
    step('flexgemm',pip+['--no-build-isolation',str(fg)])
    step('eigen_submodule',['git','-c','http.proxy=','-c','https.proxy=','submodule','update','--init','--recursive','o-voxel/third_party/eigen'])
    # Official pyproject uses Git URLs even when exact native extensions are
    # already installed. Keep these verified local builds; install other
    # requirements explicitly and retain the final pip dependency check.
    assert state['steps']['cumesh']['returncode']==state['steps']['flexgemm']['returncode']==0
    step('o_voxel_python_dependencies',pip+['plyfile==1.1.2'])
    step('o_voxel',pip+['--no-build-isolation','--no-deps',str(S/'o-voxel')])
    step('pip_check',[python,'-m','pip','check'])
    (P/'reports/trellis2_pip_freeze.txt').write_text(subprocess.check_output([python,'-m','pip','freeze','--all'],env=env,text=True))
    state.update(status='installed_pending_import_and_gpu_validation')
except Exception as error:state.update(status='failed',reason=repr(error))
finally:save()
