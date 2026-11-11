"""Install each official baseline in its own environment; stop at failed steps."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument('model', choices=['hunyuan21', 'recon02'])
args = parser.parse_args()
plan = json.loads((ROOT / 'state/environment_plans.json').read_text())[args.model]
prefix, source = Path(plan['prefix']), Path(plan['source'])
lock = (ROOT / f'state/env_{args.model}.lock').open('w')
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
state_path = ROOT / f'state/env_{args.model}.json'
state = json.loads(state_path.read_text()) if state_path.exists() else {'steps': {}, 'plan': plan}
if prefix.exists() and not state_path.exists():
    raise RuntimeError('Refusing to alter a pre-existing unowned environment')
state.update(pid=os.getpid(), status='running', runtime_validated=False)
state.pop('reason', None)
env = {k:v for k,v in os.environ.items() if 'proxy' not in k.lower()}
env.update(CONDA_PKGS_DIRS=str(ROOT / 'cache' / ('conda_' + args.model)), PIP_CACHE_DIR=str(ROOT / 'cache/pip'),
           TMPDIR=str(ROOT / 'cache/tmp'), CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1',
           MAX_JOBS='2', CMAKE_BUILD_PARALLEL_LEVEL='2', TORCH_CUDA_ARCH_LIST='8.9',
           CUDA_HOME=str(prefix), PATH=str(prefix / 'bin') + ':' + env['PATH'],
           PIP_DEFAULT_TIMEOUT='120', PIP_RETRIES='8', PIP_RESUME_RETRIES='8',
           GIT_CONFIG_COUNT='2', GIT_CONFIG_KEY_0='http.proxy', GIT_CONFIG_VALUE_0='',
           GIT_CONFIG_KEY_1='https.proxy', GIT_CONFIG_VALUE_1='')
Path(env['TMPDIR']).mkdir(parents=True, exist_ok=True)
def save():
    state['updated_at'] = datetime.now(timezone.utc).isoformat()
    tmp = state_path.with_suffix('.tmp')
    tmp.write_text(json.dumps(state, indent=2))
    tmp.replace(state_path)

def step(name, command, cwd=None):
    if state['steps'].get(name, {}).get('returncode') == 0:
        return
    if shutil.disk_usage(ROOT).free < 10 * 1024**3:
        raise RuntimeError('InsufficientDiskReserve10GiB')
    state['current_step'] = name
    state['steps'][name] = {'command': command, 'cwd': str(cwd or ROOT), 'status': 'running'}
    save()
    print(name, command, flush=True)
    result = subprocess.run(command, cwd=cwd or ROOT, env=env)
    state['steps'][name].update(returncode=result.returncode, status='passed' if result.returncode == 0 else 'failed')
    save()
    if result.returncode:
        raise RuntimeError('StepFailed:' + name)

save()
try:
    conda = ('/opt/conda/bin/conda'.replace('/opt/conda', __import__("os").environ.get('RIGID_CONDA_ROOT', '/opt/conda')).replace('/opt/rigid-benchmark/project', __import__("os").environ.get('RIGID_PROJECT_ROOT', '/opt/rigid-benchmark/project')))
    step('create_environment', [conda, 'create', '-y', '-p', str(prefix), '--override-channels',
                               '-c', 'conda-forge', 'python=3.10', 'pip'])
    step('cuda_compiler', [conda, 'install', '-y', '-p', str(prefix), '--override-channels',
                           '-c', 'nvidia', '-c', 'conda-forge', 'cuda-nvcc='+plan['cuda'],
                           'cuda-cudart-dev='+plan['cuda'], 'cuda-cccl='+plan['cuda']])
    python = str(prefix / 'bin/python')
    constraints = ROOT / f'state/{args.model}_torch_constraints.txt'
    constraints.write_text('\n'.join(plan['torch'] + plan.get('runtime_constraints', [])) + '\n')
    pip = [python, '-m', 'pip', 'install', '--disable-pip-version-check', '--constraint', str(constraints)]
    step('pytorch', pip + plan['torch'] + ['--index-url', plan['wheel_index']])
    step('build_tools', pip + ['setuptools<81', 'wheel', 'ninja==1.11.1.1', 'pybind11==2.13.4'])
    # The 1.11.1.1 wheel has a blank line before its Tag headers; modern pip
    # correctly reports missing compatibility tags. Use a fixed packaging release.
    step('ninja_wheel_metadata', pip + ['ninja==1.11.1.4'])
    build_flags = []
    if args.model == 'hunyuan21':
        # basicsr setup_requires otherwise installs a second, latest torch into
        # an isolated build env. Use the pinned runtime and explicit build tools.
        step('hunyuan_build_prerequisites', pip + ['numpy==1.24.4', 'cython', 'packaging'])
        # Blender retains the exact official cp310 wheel after its removal from
        # the normal PyPI index. Preserve bpy 4.0, Python 3.10 and native code.
        step('bpy_official_archive', pip + [
            'https://download.blender.org/pypi/bpy/bpy-4.0.0-cp310-cp310-manylinux_2_28_x86_64.whl'])
        build_flags = ['--no-build-isolation']
    step('official_requirements', pip + build_flags + ['-r', str(plan.get('requirements', source / 'requirements.txt'))], source)
    if args.model == 'hunyuan21':
        step('custom_rasterizer', pip + ['--no-build-isolation', '-e', str(source / 'hy3dpaint/custom_rasterizer')], source)
        step('mesh_painter', ['bash', 'compile_mesh_painter.sh'], source / 'hy3dpaint/DifferentiableRenderer')
    else:
        step('kaolin', pip + ['kaolin', '-f', 'https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.4.0_cu121.html'])
    step('pip_check', [python, '-m', 'pip', 'check'])
    frozen = subprocess.check_output([python, '-m', 'pip', 'freeze', '--all'], env=env, text=True)
    (ROOT / f'reports/{args.model}_pip_freeze.txt').write_text(frozen)
    state['status'] = 'installed_pending_import_and_gpu_validation'
except Exception as e:
    state.update(status='failed', reason=str(e))
finally:
    save()
