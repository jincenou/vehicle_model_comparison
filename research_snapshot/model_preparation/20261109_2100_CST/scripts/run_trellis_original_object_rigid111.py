# Isolated rigid111 adapter; source SHA256: 23dfa26629e4fb745736006c5a0c0acbabafa6683e74d91744d2987b54e0fa7e
"""Original TRELLIS from the legacy pipeline, using one frozen real RGBA."""
import os
os.environ['SPCONV_ALGO']='native'
os.environ['ATTN_BACKEND']='flash-attn'
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

P=Path(__file__).resolve().parents[1]
S=P/'sources/trellis_original_legacy_20261110'
parser=argparse.ArgumentParser()
parser.add_argument('--sample-id',required=True)
parser.add_argument('--run-dir',type=Path,required=True)
parser.add_argument('--reload-only',action='store_true')
args=parser.parse_args();O=args.run_dir.resolve()
assert O.is_relative_to((P/'runs/trellis_original_rigid111_v1').resolve())
sys.path.insert(0,str(S))
import torch
torch.empty(1,device='cuda')
from PIL import Image
import numpy as np
from trellis.utils import render_utils
from trellis.representations.gaussian import Gaussian
from recon02_serialization import save_native_gaussian


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path,value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)
def render(gs,prefix):
    ex,ins=render_utils.yaw_pitch_r_fov_to_extrinsics_intrinsics([0.,.29670597],[.2,.2],2.,40.)
    frames=render_utils.render_frames(gs,ex,ins,options={'resolution':512,'bg_color':(1,1,1)})
    paths=[]
    for i,frame in enumerate(frames['color']):
        path=O/f'{prefix}_{i}.png';Image.fromarray(frame).save(path);paths.append(path)
    return paths


if args.reload_only:
    state=json.loads((O/'state.json').read_text())
    assert state['status']=='generated_pending_reload' and state['sample_id']==args.sample_id
    assert sha(O/'native_gaussians.pt')==state['output_hashes']['native_gaussians.pt']
    packed=torch.load(O/'native_gaussians.pt',map_location='cpu',weights_only=True)
    assert packed['init_params']['sh_degree']==0
    gs=Gaussian(**packed['init_params'])
    for name,tensor in packed['params'].items():
        assert torch.isfinite(tensor).all(),name
        setattr(gs,name,tensor.cuda())
    paths=render(gs,'reload');details=[]
    for i,path in enumerate(paths):
        a=np.array(Image.open(path));b=np.array(Image.open(O/f'native_{i}.png'))
        diff=np.abs(a.astype(np.int16)-b.astype(np.int16))
        assert diff.max()<=1 and diff.mean()<=.01
        assert np.isfinite(a).all()
        details.append({'path':str(path),'sha256':sha(path),'max_pixel_difference':int(diff.max()),'foreground_pixels_rgb_lt250':int(np.count_nonzero(a.min(axis=2)<250)),'sparse_or_white_diagnostic':bool(np.count_nonzero(a.min(axis=2)<250)<=100)})
    save(O/'reload_validation.json',{'status':'passed_pending_process_exit','sample_id':args.sample_id,
                                    'output_sha256':state['output_hashes']['native_gaussians.pt'],
                                    'views':details,'scope':'native integrity; common camera not yet validated'})
else:
    O.mkdir(parents=True,exist_ok=False)
    manifest=P.parents[1]/'inputs/rigid111_image_inputs_v1/manifest.json'
    row=next(x for x in json.loads(manifest.read_text())['objects'] if x['sample_id']==args.sample_id)
    assert sha(row['image_rgba'])==row['image_rgba_sha256']
    state={'status':'running','pid':os.getpid(),'sample_id':args.sample_id,'stage':'load',
           'started_at':time.time(),'method':'TRELLIS-image-large original legacy implementation',
           'source':str(S),'weights':('/opt/rigid-benchmark/project/models/TRELLIS-image-large-gaussian'.replace('/opt/rigid-benchmark/project', __import__("os").environ.get('RIGID_PROJECT_ROOT', '/opt/rigid-benchmark/project'))),
           'input':row['image_rgba'],'input_sha256':row['image_rgba_sha256'],
           'real_input_count':1,'generated_input_count':0,'seed':42,'gpu':os.environ.get('CUDA_VISIBLE_DEVICES'),
           'manifest_sha256':sha(manifest),'common_camera_qualified':False}
    save(O/'state.json',state)
    try:
        from gen_gs import GsConfig,GsBatchGenerator
        from trellis import models
        from trellis.pipelines import samplers
        from trellis.pipelines.trellis_image_to_3d import TrellisImageTo3DPipeline
        cfg=GsConfig(model_dir=Path(state['weights']),seed=42,worker_count=1)
        pipeline=GsBatchGenerator(cfg)._load_pipeline(models,samplers,TrellisImageTo3DPipeline)
        pipeline.cuda()
        im=Image.open(row['image_rgba']);assert im.mode=='RGBA'
        state.update(stage='generate',sparse_structure_sampler=pipeline.sparse_structure_sampler_params,
                     slat_sampler=pipeline.slat_sampler_params);save(O/'state.json',state)
        gs=pipeline.run(im,seed=42,formats=['gaussian'])['gaussian'][0]
        assert gs.sh_degree==0
        save_native_gaussian(gs,O/'native_gaussians.ply')
        packed={'init_params':gs.init_params,'params':{k:getattr(gs,k).detach().cpu() for k in
                ['_xyz','_features_dc','_scaling','_rotation','_opacity']}}
        torch.save(packed,O/'native_gaussians.pt');render(gs,'native')
        state.update(status='generated_pending_reload',stage='finished',gaussians=len(gs.get_xyz),
                     output_hashes={p.name:sha(p)for p in O.glob('native_*')},
                     serialization='native finite opacity logits; all Gaussian attributes preserved',
                     peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated())
    except Exception as error:
        state.update(status='failed',reason=repr(error));raise
    finally:
        state['finished_at']=time.time();save(O/'state.json',state)
