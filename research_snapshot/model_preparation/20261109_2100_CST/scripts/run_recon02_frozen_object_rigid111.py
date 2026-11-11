# Isolated rigid111 adapter; source SHA256: bc2ccbaad340d4350773b35d53572e9278af497f4104d679832f792a9621f7c5
"""ReconViaGen v0.2 on frozen already-accepted real training views."""
import os
os.environ['SPCONV_ALGO'] = 'native'  # Official app.py setting.
import sys
import json
import hashlib
import time
from pathlib import Path

P = Path(__file__).resolve().parents[1]
S = P/'sources/recon02-f6720923260927333e22761fa722144d214733d9'
import argparse
parser=argparse.ArgumentParser()
parser.add_argument('--sample-id',required=True)
parser.add_argument('--run-dir',required=True,type=Path)
args=parser.parse_args()
O=args.run_dir.resolve()
assert O.is_relative_to((P/'runs/recon02_rigid111_v1').resolve())
M=P.parents[1]/'inputs/recon02_rigid111_real_multiview_v1/manifest.json'
manifest=json.loads(M.read_text())
row=next(x for x in manifest['objects'] if x['sample_id']==args.sample_id)
assert row['real_input_count']>0
for v in row['views']:
    assert hashlib.sha256(Path(v['image']).read_bytes()).hexdigest()==v['sha256']
    assert v['split']=='train'

O.mkdir(parents=True, exist_ok=False)
os.chdir(S);sys.path.insert(0,str(S))
import torch
torch.empty(1,device='cuda')
from PIL import Image
# The pinned official requirements use timm 0.6.7, while the pinned
# BiRefNet code imports the later timm.layers namespace. Preserve the
# required package and expose its original functions under that namespace.
import timm
import timm.models.layers
assert timm.__version__ == '0.6.7'
sys.modules['timm.layers'] = timm.models.layers
from timm.layers import DropPath, to_2tuple, trunc_normal_
assert DropPath is timm.models.layers.DropPath
assert to_2tuple is timm.models.layers.to_2tuple
assert trunc_normal_ is timm.models.layers.trunc_normal_
from trellis.pipelines import TrellisVGGTTo3DPipeline
from trellis.utils import render_utils, postprocessing_utils

paths=[Path(v['image']) for v in row['views']]
state={'status':'running','stage':'load','pid':os.getpid(),'started_at':time.time(),
       'source_commit':S.name.removeprefix('recon02-'),'gpu':os.environ['CUDA_VISIBLE_DEVICES'],
       'inputs':[{'path':str(x),'sha256':hashlib.sha256(x.read_bytes()).hexdigest()} for x in paths],
       'sample_id':args.sample_id,'manifest_sha256':hashlib.sha256(M.read_bytes()).hexdigest(),
       'real_input_count':len(paths),'generated_input_count':0,'common_camera_qualified':False,
       'settings':{'seed':42,'ss_steps':30,'ss_cfg':7.5,'ss_rescale':.7,'ss_rescale_t':5.,
                   'slat_steps':12,'slat_cfg':7.5,'slat_rescale':.5,'slat_rescale_t':3.,
                   'mode':'multidiffusion','low_vram':True,'spconv_algo':'native',
                   'glb_simplify':.95,'glb_texture_size':1024},
       'official_example_gate':str(P/'runs/recon02_official_demo_v1/state.json'),
       'compatibility':{'timm_version':timm.__version__,
                        'timm_layers_alias':'timm.models.layers (same function objects; no weight or algorithm changes)'}}


def save():
    temp=O/'state.tmp';temp.write_text(json.dumps(state,indent=2)+'\n');temp.replace(O/'state.json')


save()
try:
    pipeline=TrellisVGGTTo3DPipeline.from_pretrained(str(P/'weights/Stable-X--trellis-vggt-v0-2'))
    pipeline._device=torch.device('cuda');pipeline.low_vram=True;pipeline.birefnet_model.cuda()
    images=[pipeline.preprocess_image(Image.open(x)) for x in paths]
    for i,im in enumerate(images):im.save(O/f'preprocessed_{i}.png')
    state.update(stage='generate');save()
    outputs,_,_=pipeline.run(image=images,seed=42,formats=['gaussian','mesh'],preprocess_image=False,
        sparse_structure_sampler_params={'steps':30,'cfg_strength':7.5,'cfg_interval':[.6,1.],
                                         'guidance_rescale':.7,'rescale_t':5.},
        slat_sampler_params={'steps':12,'cfg_strength':7.5,'cfg_interval':[.6,1.],
                             'guidance_rescale':.5,'rescale_t':3.},mode='multidiffusion')
    gs=outputs['gaussian'][0];mesh=outputs['mesh'][0]
    from recon02_serialization import save_native_gaussian
    save_native_gaussian(gs, O/'native_gaussians.ply')
    import trimesh
    trimesh.Trimesh(mesh.vertices.cpu().numpy(),mesh.faces.cpu().numpy(),process=False).export(O/'native_mesh.ply')
    packed={'init_params':gs.init_params,'params':{k:getattr(gs,k).detach().cpu() for k in
            ('_xyz','_features_dc','_scaling','_rotation','_opacity')}}
    torch.save(packed,O/'native_gaussians.pt')
    state.update(stage='render',gaussians=len(gs.get_xyz),mesh_vertices=len(mesh.vertices),mesh_faces=len(mesh.faces));save()
    extr,intr=render_utils.yaw_pitch_r_fov_to_extrinsics_intrinsics([0.,.29670597],[.2,.2],2.,40.)
    frames=render_utils.render_frames(gs,extr,intr,options={'resolution':512,'bg_color':(1,1,1)})
    for i,frame in enumerate(frames['color']):Image.fromarray(frame).save(O/f'native_preview_{i}.png')
    state.update(stage='export_glb');save()
    glb=postprocessing_utils.to_glb(gs,mesh,simplify=.95,texture_size=1024,verbose=False)
    glb.export(O/'native_textured.glb')
    state.update(status='generated_pending_independent_reload',stage='finished',
                 output_hashes={x.name:hashlib.sha256(x.read_bytes()).hexdigest() for x in O.glob('native_*') if x.is_file()},
                 output=str(O/'native_textured.glb'),
                 output_sha256=hashlib.sha256((O/'native_textured.glb').read_bytes()).hexdigest(),
                 peak_allocated_bytes=torch.cuda.max_memory_allocated())
except Exception as error:
    state.update(status='failed',reason=repr(error));raise
finally:
    state['finished_at']=time.time();save()
